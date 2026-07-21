from __future__ import annotations

import asyncio
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

_ET = ZoneInfo("America/New_York")


class MomentumScreener:
    """Daily scan of the full US listed universe for momentum leaders.

    All criteria are computed on DAILY bars:
      • dollar volume  — mean(close × volume) over DOLLAR_VOL_PERIOD ≥ DOLLAR_VOL_MIN
      • APTR           — ATR(APTR_PERIOD) / close ≥ APTR_MIN      (volatility floor)
      • BMU            — % gain over BMU_PERIOD ≥ BMU_MIN         (momentum)
      • trend          — close > SMA(10)
      • not at highs   — close < highest high of the last NOT_AT_HIGHS_PERIOD days
      • extension      — (close − SMA50) / ATR ≤ MAX_50SMA_EXTENSION_APTRS
      • IPO relaxation — names with < 50 bars get eased APTR / BMU floors

    The qualified pool is ranked by dollar volume and capped at TOP_N; the bot
    then trades the FINAL_N strongest by BMU.
    """

    # ── Screen configuration ───────────────────────────────────────────────
    DOLLAR_VOL_PERIOD = 2
    DOLLAR_VOL_MIN = 100_000_000
    APTR_PERIOD = 14
    APTR_MIN = 0.04
    BMU_PERIOD = 60
    BMU_MIN = 0.30
    NOT_AT_HIGHS_PERIOD = 30
    MUST_BE_ABOVE_10SMA = True
    MAX_50SMA_EXTENSION_APTRS = 10
    IPO_MIN_BARS = 30
    IPO_APTR_MULT = 0.7
    IPO_BMU_MULT = 0.5
    TOP_N = 150          # qualified momentum-leader pool (ranked by dollar volume)
    FINAL_N = 25         # how many the bot actually trades (strongest by BMU)
    INCLUDE_ETFS = False # common stock only — leveraged ETFs (SOXS, MSTU, AAPU…) excluded

    # ── Scan mechanics ─────────────────────────────────────────────────────
    BATCH_SIZE = 150     # symbols per yfinance download call
    HISTORY_PERIOD = "6mo"
    FULL_HISTORY_BARS = 50   # below this a name counts as an "IPO" for relaxation

    _NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    _OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    def __init__(self) -> None:
        self.logger = logging.getLogger("Screener")
        self.cache_path = Path(__file__).resolve().parents[2] / "data" / "screener_cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Details of the most recent scan, for the Discord report
        self.last_picks: List[dict] = []
        self.last_qualified = 0
        self.last_scanned = 0
        # Per-criterion rejection tally — shows which filter is actually binding
        self._reject: dict[str, int] = {}

    # ── Universe ───────────────────────────────────────────────────────────

    def _fetch_universe(self) -> List[str]:
        """All US listed common-stock symbols (ETFs, test issues and
        warrants/units/preferreds removed)."""
        symbols: set[str] = set()
        headers = {"User-Agent": "Mozilla/5.0"}

        for url, sym_col in ((self._NASDAQ_LISTED, "Symbol"),
                             (self._OTHER_LISTED, "ACT Symbol")):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text), sep="|")
                if sym_col not in df.columns:
                    self.logger.warning("Unexpected symbol-file layout at %s", url)
                    continue
                # Files end with a "File Creation Time" footer row.
                df = df[~df[sym_col].astype(str).str.contains("File Creation Time", na=False)]
                if "Test Issue" in df.columns:
                    df = df[df["Test Issue"] != "Y"]
                if "ETF" in df.columns and not self.INCLUDE_ETFS:
                    df = df[df["ETF"] != "Y"]
                for raw in df[sym_col].astype(str):
                    sym = raw.strip().upper()
                    # Plain alphabetic tickers only — drops warrants/units/preferreds
                    # which carry '$', '.', '-' suffixes.
                    if sym.isalpha() and 1 <= len(sym) <= 5:
                        symbols.add(sym)
            except Exception as exc:
                self.logger.warning("Could not fetch symbol list %s: %s", url, exc)

        return sorted(symbols)

    # ── Per-symbol evaluation ──────────────────────────────────────────────

    def _drop_partial_bar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop today's bar while the session is still open — an in-progress day has
        partial volume, which would distort dollar volume and the latest close."""
        if df is None or df.empty:
            return df
        try:
            now = datetime.now(_ET)
            if now.hour >= 16:          # session finished; today's bar is complete
                return df
            last = df.index[-1]
            last_date = last.date() if hasattr(last, "date") else None
            if last_date == now.date():
                return df.iloc[:-1]
        except Exception:
            pass
        return df

    def _evaluate(self, symbol: str, df: pd.DataFrame) -> dict | None:
        """Return the metrics dict when the symbol passes every criterion, else None."""
        rej = self._reject
        if df is None or df.empty:
            rej["no_data"] += 1
            return None
        df = self._drop_partial_bar(df)
        df = df.dropna(subset=["Close"])
        bars = len(df)
        if bars < self.IPO_MIN_BARS:
            rej["short_history"] += 1
            return None

        close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
        last_close = float(close.iloc[-1])
        if last_close <= 0:
            rej["no_data"] += 1
            return None

        # 1. Dollar volume
        dollar_volume = float((close * volume).tail(self.DOLLAR_VOL_PERIOD).mean())
        if pd.isna(dollar_volume) or dollar_volume < self.DOLLAR_VOL_MIN:
            rej["dollar_vol"] += 1
            return None

        # 2. APTR — ATR as a fraction of price
        prev_close = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = float(true_range.rolling(self.APTR_PERIOD).mean().iloc[-1])
        if pd.isna(atr) or atr <= 0:
            rej["no_data"] += 1
            return None
        aptr = atr / last_close

        # Short-history (IPO) names get eased volatility/momentum floors.
        full_history = bars >= self.FULL_HISTORY_BARS
        aptr_min = self.APTR_MIN if full_history else self.APTR_MIN * self.IPO_APTR_MULT
        bmu_min = self.BMU_MIN if full_history else self.BMU_MIN * self.IPO_BMU_MULT

        if aptr < aptr_min:
            rej["aptr"] += 1
            return None

        # 3. BMU — % gain over the lookback (clipped to available history)
        lookback = min(self.BMU_PERIOD, bars - 1)
        past_close = float(close.iloc[-1 - lookback])
        if past_close <= 0:
            rej["no_data"] += 1
            return None
        bmu = (last_close / past_close) - 1.0
        if bmu < bmu_min:
            rej["bmu"] += 1
            return None

        # 4. Healthy trend — above the 10SMA
        if self.MUST_BE_ABOVE_10SMA:
            sma10 = float(close.rolling(10).mean().iloc[-1])
            if pd.isna(sma10) or last_close <= sma10:
                rej["sma10"] += 1
                return None

        # 5. Not at highs — compare against the high of the PRIOR window, excluding
        #    the latest bar (including it makes the test vacuous, since today's
        #    intraday high is almost always ≥ today's close).
        prior_high = high.iloc[-1 - self.NOT_AT_HIGHS_PERIOD:-1]
        highest = float(prior_high.max()) if len(prior_high) else float("nan")
        if pd.isna(highest) or last_close >= highest:
            rej["at_highs"] += 1
            return None

        # 6. Extension above the 50SMA, measured in ATRs (skipped for IPO names
        #    that don't have 50 bars yet).
        extension = None
        if full_history:
            sma50 = float(close.rolling(50).mean().iloc[-1])
            if not pd.isna(sma50):
                extension = (last_close - sma50) / atr
                if extension > self.MAX_50SMA_EXTENSION_APTRS:
                    rej["extension"] += 1
                    return None

        return {
            "symbol": symbol,
            "dollar_volume": dollar_volume,
            "aptr": aptr,
            "bmu": bmu,
            "extension": extension,
            "price": last_close,
            "bars": bars,
            # How far below the recent high it sits — near-high = coiled/consolidating,
            # far below = broken down. The pass/fail check alone throws this away.
            "pct_from_high": (highest - last_close) / highest if highest > 0 else 0.0,
            "sector": None,   # filled in for the final picks only (needs a .info call)
        }

    # ── Scan ───────────────────────────────────────────────────────────────

    def _screen_sync(self) -> List[dict]:
        """Blocking full-universe scan — run via asyncio.to_thread."""
        universe = self._fetch_universe()
        if not universe:
            self.logger.error("Symbol universe is empty — screener cannot run.")
            return []

        self.logger.info("Scanning %d US listed symbols for momentum leaders...", len(universe))
        results: List[dict] = []
        self._reject = {k: 0 for k in
                        ("no_data", "short_history", "dollar_vol", "aptr",
                         "bmu", "sma10", "at_highs", "extension")}
        evaluated = 0
        lost_batches = 0

        for start in range(0, len(universe), self.BATCH_SIZE):
            chunk = universe[start:start + self.BATCH_SIZE]
            data = None
            for attempt in (1, 2):      # one retry — a dropped batch loses 150 symbols
                try:
                    data = yf.download(
                        chunk,
                        period=self.HISTORY_PERIOD,
                        interval="1d",
                        group_by="ticker",
                        auto_adjust=True,
                        progress=False,
                        threads=True,
                    )
                    break
                except Exception as exc:
                    if attempt == 2:
                        lost_batches += 1
                        self.logger.warning("Batch at %d failed twice (%d symbols lost): %s",
                                            start, len(chunk), exc)
                    else:
                        self.logger.debug("Batch at %d attempt 1 failed, retrying: %s", start, exc)
            if data is None or data.empty:
                continue

            multi = isinstance(data.columns, pd.MultiIndex)
            for symbol in chunk:
                try:
                    sub = data[symbol] if multi else data
                except (KeyError, TypeError):
                    self._reject["no_data"] += 1
                    continue
                evaluated += 1
                try:
                    row = self._evaluate(symbol, sub)
                except Exception:
                    self._reject["no_data"] += 1
                    continue
                if row:
                    results.append(row)

            self.logger.info("  scanned %d/%d — %d qualifying so far",
                             min(start + self.BATCH_SIZE, len(universe)),
                             len(universe), len(results))

        r = self._reject
        self.logger.info(
            "Screen funnel | universe=%d evaluated=%d lost_batches=%d || rejected: "
            "no_data=%d short_history=%d $vol=%d APTR=%d BMU=%d <10SMA=%d at_highs=%d extension=%d "
            "|| qualified=%d",
            len(universe), evaluated, lost_batches,
            r["no_data"], r["short_history"], r["dollar_vol"], r["aptr"],
            r["bmu"], r["sma10"], r["at_highs"], r["extension"], len(results),
        )

        self.last_scanned = len(universe)
        self.last_qualified = len(results)
        results.sort(key=lambda r: r["dollar_volume"], reverse=True)
        return results[:self.TOP_N]

    async def _fetch_sectors(self, symbols: List[str]) -> dict[str, str]:
        """Sector lookup for the final picks only — one .info call each, in parallel."""
        async def one(sym: str) -> tuple[str, str]:
            try:
                info = await asyncio.to_thread(lambda: yf.Ticker(sym).info)
                return sym, (info.get("sector") or "—")
            except Exception:
                return sym, "—"

        try:
            pairs = await asyncio.gather(*(one(s) for s in symbols))
            return dict(pairs)
        except Exception as exc:
            self.logger.warning("Sector lookup failed: %s", exc)
            return {}

    async def screen(self) -> List[str]:
        """Run the full scan and return the tickers the bot should trade today."""
        leaders = await asyncio.to_thread(self._screen_sync)
        if not leaders:
            self.logger.warning("Screener found no qualifying momentum leaders.")
            return []

        # Qualified pool is ranked by liquidity; trade the strongest momentum from it.
        picks = sorted(leaders, key=lambda r: r["bmu"], reverse=True)[:self.FINAL_N]
        symbols = [r["symbol"] for r in picks]

        sectors = await self._fetch_sectors(symbols)
        for r in picks:
            r["sector"] = sectors.get(r["symbol"], "—")
        self.last_picks = picks

        self.logger.info("%d names qualified — trading top %d by BMU:", len(leaders), len(symbols))
        for r in picks:
            ext = f"{r['extension']:.1f}" if r["extension"] is not None else "n/a"
            self.logger.info(
                "  %-6s BMU %+6.1f%% | APTR %4.1f%% | $vol %6.0fM | ext %s ATR | $%.2f",
                r["symbol"], r["bmu"] * 100, r["aptr"] * 100,
                r["dollar_volume"] / 1e6, ext, r["price"],
            )

        self._write_cache(symbols)
        return symbols

    # ── Cache ──────────────────────────────────────────────────────────────

    @staticmethod
    def _today() -> str:
        return datetime.now(_ET).strftime("%Y-%m-%d")

    def _write_cache(self, symbols: List[str]) -> None:
        try:
            with open(self.cache_path, "w") as f:
                json.dump({"date": self._today(), "symbols": symbols}, f)
        except Exception as exc:
            self.logger.warning("Could not write screen cache: %s", exc)

    def _cached_today(self) -> List[str] | None:
        if not self.cache_path.exists():
            return None
        try:
            with open(self.cache_path) as f:
                data = json.load(f)
            if data.get("date") == self._today():
                return data.get("symbols") or None
            self.logger.info("Screen cache is stale (%s) — re-screening for %s.",
                             data.get("date"), self._today())
        except Exception as exc:
            self.logger.warning("Could not read screen cache: %s", exc)
        return None

    async def load_or_screen(self) -> List[str]:
        """Today's cached screen if present, otherwise run a fresh scan."""
        cached = self._cached_today()
        if cached:
            self.logger.info("Using cached screen from today: %s", cached)
            return cached
        return await self.screen()
