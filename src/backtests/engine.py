"""
BacktestEngine
──────────────
Replays the existing OpenClaw strategy against 90 days of IBKR historical
data without touching live execution.  Every bar is fed sequentially so the
strategy at bar t never sees data from t+1.

Key design choices
──────────────────
• LLM (openclaw/llama3.2) is called at every entry candidate — same model and
  prompt structure as live, but without news (historical headlines unavailable).
  _compute_signal() acts as a cheap pre-filter to avoid calling Ollama on bars
  with no indicator activity.
• Fill price = next bar's open, NOT the signal bar's close.  This is the most
  important anti-lookahead-bias guard.
• Position sizing is confidence-proportional: 1.5% × LLM confidence per trade.
• Exit rules are identical to live: 2% hard stop, ATR trailing stop, 3×ATR
  take-profit, signal reversal — in that priority order.
• $1.50/trade flat commission applied to every round-trip.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import aiohttp
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

import numpy as np
import pandas as pd
from ib_insync import Stock

from data.data_models import BarData
from data.ibkr_client import IBKRClient
from strategy.indicators import IndicatorCalculator
from strategy.logic import StrategyEngine


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol: str
    direction: str          # "LONG" or "SHORT"
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    confidence: float       # fraction of indicators aligned at entry
    # Market conditions captured at entry — used by tail forensics to compare
    # what the worst 5% and best 5% of trades had in common.
    entry_conditions: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    start_equity: float = 100_000.0
    final_equity: float = 100_000.0
    duration_days: int = 90
    tickers: list[str] = field(default_factory=list)

    # Diagnostic — populated during run() before finalise()
    bars_fetched: dict = field(default_factory=dict)   # {symbol: int}  5m bars per ticker
    bars_replayed: dict = field(default_factory=dict)  # {symbol: int}  bars that reached signal eval
    signals_fired: dict = field(default_factory=dict)  # {symbol: int}  entries attempted
    halted_days: int = 0                               # trading days the circuit breaker fired

    # Computed by finalise()
    total_return: float = 0.0
    annualised_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown_duration_days: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    biggest_win: float = 0.0
    biggest_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_trades: int = 0
    total_commissions: float = 0.0

    def finalise(self) -> None:
        self.total_trades = len(self.trades)

        # Build equity curve
        equity = self.start_equity
        curve = [equity]
        for t in self.trades:
            equity += t.pnl
            curve.append(equity)
        self.equity_curve = curve
        self.final_equity = equity

        if self.total_trades == 0:
            return

        # ── Return metrics ──
        self.total_return = (self.final_equity - self.start_equity) / self.start_equity
        years = self.duration_days / 252.0
        if years > 0 and self.total_return > -1.0:
            self.annualised_return = (1.0 + self.total_return) ** (1.0 / years) - 1.0

        # ── Sharpe / Sortino (trade-level, annualised) ──
        pnls = np.array([t.pnl for t in self.trades], dtype=float)
        if len(pnls) > 1:
            mean_pnl = float(np.mean(pnls))
            std_pnl = float(np.std(pnls)) + 1e-9
            trades_per_day = max(self.total_trades / max(self.duration_days, 1), 0.1)
            annual_factor = math.sqrt(252.0 * trades_per_day)
            self.sharpe_ratio = (mean_pnl / std_pnl) * annual_factor

            downside = pnls[pnls < 0.0]
            downside_std = float(np.std(downside)) + 1e-9 if len(downside) > 0 else std_pnl
            self.sortino_ratio = (mean_pnl / downside_std) * annual_factor

        # ── Max drawdown & average duration ──
        peak = self.start_equity
        max_dd = 0.0
        dd_depths: list[float] = []
        dd_durations: list[int] = []
        in_drawdown = False
        dd_bars = 0

        for eq in self.equity_curve:
            if eq >= peak:
                if in_drawdown:
                    dd_durations.append(dd_bars)
                    dd_bars = 0
                    in_drawdown = False
                peak = eq
            else:
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)
                if not in_drawdown:
                    in_drawdown = True
                dd_bars += 1
                dd_depths.append(dd)

        if in_drawdown:
            dd_durations.append(dd_bars)

        self.max_drawdown = max_dd
        if dd_durations:
            trades_per_day = max(self.total_trades / max(self.duration_days, 1), 0.1)
            self.avg_drawdown_duration_days = (
                sum(dd_durations) / len(dd_durations)
            ) / trades_per_day

        # ── Calmar ──
        if self.max_drawdown > 0:
            self.calmar_ratio = self.annualised_return / self.max_drawdown

        # ── Trade-level stats ──
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        self.win_rate = len(wins) / self.total_trades
        self.avg_win = sum(wins) / len(wins) if wins else 0.0
        self.avg_loss = sum(losses) / len(losses) if losses else 0.0
        self.biggest_win = max(wins) if wins else 0.0
        self.biggest_loss = min(losses) if losses else 0.0
        sum_loss = abs(sum(losses)) if losses else 0.0
        self.profit_factor = sum(wins) / sum_loss if sum_loss > 0 else float("inf")
        self.expectancy = (self.win_rate * self.avg_win) + ((1.0 - self.win_rate) * self.avg_loss)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_utc(ts: Any) -> datetime:
    """Normalise BarData.timestamp to UTC-aware datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _hourly_bars_up_to(bars_1h: list[BarData], cutoff: datetime) -> list[BarData]:
    """
    Return 1h bars whose bar closes BEFORE the cutoff time.
    A 1h bar timestamped at T covers T → T+1h, so it is complete at T+1h.
    We use T+1h < cutoff to avoid lookahead.
    """
    result = [b for b in bars_1h if _to_utc(b.timestamp) + timedelta(hours=1) <= cutoff]
    return result[-60:] if len(result) > 60 else result   # cap lookback at 60 h


# ── Engine ─────────────────────────────────────────────────────────────────────

class BacktestEngine:
    # Mirror the live trading constants exactly
    WARMUP_BARS = 50
    SIGNAL_THRESHOLD = 3      # out of 4 indicators — 3/4 required for quality entries
    STOP_LOSS_PCT = 0.02
    DAILY_LOSS_LIMIT_PCT = 0.005  # halt new entries when day's P&L < -0.5% of equity
    ATR_TRAIL_MULT = 2.5
    TAKE_PROFIT_ATR_MULT = 3.0      # TP1 — first profit at entry + 3×ATR
    TAKE_PROFIT_2_ATR_MULT = 6.0    # TP2 — runner target at entry + 6×ATR
    SCALE_OUT_PCT = 0.60            # sell 60% at TP1, run the remaining 40%
    POSITION_PCT = 0.10       # base position = 10% of equity × confidence per trade
    LLM_CONF_THRESHOLD = 0.60 # min LLM confidence to open a trade
    MIN_HOLD_BARS = 5         # 25 minutes before AI-reversal exit allowed
    COOLDOWN_BARS = 3         # 15-minute cooldown after close
    MAX_LOOKBACK_5M = 200     # rolling window for indicator computation
    IBKR_TIMEFRAME_PAUSE = 5.0   # seconds between bar-size requests for same symbol
    IBKR_SYMBOL_PAUSE = 12.0     # seconds between symbols — avoids pacing with live subs

    def __init__(
        self,
        client: IBKRClient,
        start_equity: float = 100_000.0,
        commission_per_trade: float = 1.50,
        ollama_url: str = "http://openclaw_ollama:11434",
    ) -> None:
        self.client = client
        self.start_equity = start_equity
        self.commission_per_trade = commission_per_trade
        self.ollama_url = ollama_url
        self.logger = logging.getLogger("BacktestEngine")
        self._calc = IndicatorCalculator()
        self._strategy = StrategyEngine()

    # ── Historical data ────────────────────────────────────────────────────────

    async def _fetch_bars(self, symbol: str, duration: str) -> dict[str, list[BarData]]:
        contract = Stock(symbol, self.client.exchange, self.client.currency)
        try:
            qualified = await self.client.ib.qualifyContractsAsync(contract)
        except Exception as exc:
            self.logger.error("qualifyContractsAsync failed for %s: %s", symbol, exc)
            return {}

        if not qualified:
            self.logger.warning("Could not qualify %s — no contract found.", symbol)
            return {}

        result: dict[str, list[BarData]] = {}
        for bar_size in ("5 mins", "1 hour"):
            try:
                bars = await self.client.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=False,   # one-time fetch — no live subscription
                )
                result[bar_size] = [
                    BarData(
                        symbol=symbol,
                        timestamp=bar.date,
                        timeframe=bar_size,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(bar.volume),
                    )
                    for bar in bars
                ]
                self.logger.info(
                    "%s [%s]: fetched %d bars.",
                    symbol, bar_size, len(result[bar_size]),
                )
            except Exception as exc:
                self.logger.error("Historical fetch failed for %s [%s]: %s", symbol, bar_size, exc)
                result[bar_size] = []

            await asyncio.sleep(self.IBKR_TIMEFRAME_PAUSE)

        return result

    # ── Signal (technical-only, no LLM) ───────────────────────────────────────

    def _compute_signal(
        self,
        window_5m: list[BarData],
        window_1h: list[BarData],
        cached_df=None,
        cached_df_1h=None,
    ) -> tuple[str, float]:
        """
        Thin wrapper over StrategyEngine.compute_alignment_signal so the backtest
        and live trading gate entries with the exact same logic.
        cached_df: pre-computed result of calculate_all(window_5m) to avoid recomputing.
        cached_df_1h: pre-computed 1h DataFrame slice — skips calculate_all(window_1h).
        """
        df = cached_df if cached_df is not None else self._calc.calculate_all(window_5m)
        df_1h = cached_df_1h if cached_df_1h is not None else (
            self._calc.calculate_all(window_1h) if len(window_1h) >= 50 else pd.DataFrame()
        )
        # Single source of truth — identical gate used by live trading.
        return self._strategy.compute_alignment_signal(
            df, df_1h,
            signal_threshold=self.SIGNAL_THRESHOLD,
            warmup_bars=self.WARMUP_BARS,
        )

    def _capture_entry_conditions(
        self,
        cached_df,
        bar_et,
        fill_price: float,
        atr: float,
        source: str,
    ) -> dict:
        """Snapshot the market conditions at entry for tail forensics."""
        cond: dict = {
            "atr_pct": round(atr / fill_price * 100, 3) if fill_price > 0 else 0.0,
            "hour_et": f"{bar_et.hour:02d}:{bar_et.minute:02d}",
            "source": source,
            "adx": None,
            "vol_ratio": None,
            "max_bar_range_atr": None,   # biggest single-bar range of last 12 bars vs ATR
        }
        if cached_df is None or cached_df.empty:
            return cond
        row = cached_df.iloc[-1]

        adx_col = next((c for c in cached_df.columns if c.startswith("ADX_")), None)
        if adx_col is not None:
            v = row.get(adx_col)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                cond["adx"] = round(float(v), 1)

        vol, vsma = row.get("volume"), row.get("volume_sma_20")
        try:
            if vol is not None and vsma is not None and float(vsma) > 0:
                cond["vol_ratio"] = round(float(vol) / float(vsma), 2)
        except (TypeError, ValueError):
            pass

        if atr > 0 and {"high", "low"}.issubset(cached_df.columns):
            recent = cached_df.iloc[-12:]
            try:
                max_range = float((recent["high"] - recent["low"]).max())
                cond["max_bar_range_atr"] = round(max_range / atr, 2)
            except (TypeError, ValueError):
                pass
        return cond

    @staticmethod
    def _avg(values: list) -> float | None:
        vals = [v for v in values if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def _log_tail_forensics(self, trades: list[TradeRecord]) -> None:
        """Compare entry conditions of the worst 5% vs best 5% of trades by P&L.
        A condition only matters as a filter when it differs between the tails —
        common to both just means 'the stock was moving'."""
        n = len(trades)
        if n < 20:
            return
        by_pnl = sorted(trades, key=lambda t: t.pnl)
        k = max(3, int(n * 0.05))
        tails = {"WORST": by_pnl[:k], "BEST": by_pnl[-k:][::-1]}

        for label, group in tails.items():
            conds = [t.entry_conditions for t in group if t.entry_conditions]
            llm_share = sum(1 for c in conds if c.get("source") == "llm")
            self.logger.info(
                "Forensics %s %d: avg P&L $%.0f | ATR%%=%s | ADX=%s | vol=%sx | "
                "max_bar/ATR=%s | conf=%.2f | llm=%d/%d",
                label, k,
                sum(t.pnl for t in group) / k,
                self._avg([c.get("atr_pct") for c in conds]),
                self._avg([c.get("adx") for c in conds]),
                self._avg([c.get("vol_ratio") for c in conds]),
                self._avg([c.get("max_bar_range_atr") for c in conds]),
                sum(t.confidence for t in group) / k,
                llm_share, len(conds),
            )
            for t in group:
                c = t.entry_conditions or {}
                self.logger.info(
                    "  %s %s $%+.0f (%s) | entry %s ET | ATR%%=%s ADX=%s vol=%sx range=%s src=%s",
                    label.lower(), t.symbol, t.pnl, t.exit_reason,
                    c.get("hour_et", "?"), c.get("atr_pct"), c.get("adx"),
                    c.get("vol_ratio"), c.get("max_bar_range_atr"), c.get("source", "?"),
                )

    # ── LLM evaluation ─────────────────────────────────────────────────────────

    def _build_technical_context(self, cached_df, df_1h=None) -> str:
        """Build the same technical_context string that live trading sends to the LLM."""
        tech_5m = self._strategy.evaluate_signals(cached_df)
        hourly_trend = ""
        if df_1h is not None and not df_1h.empty:
            hourly_trend = self._strategy.evaluate_hourly_trend(df_1h)
        base = f"{hourly_trend} | 5m: {tech_5m}" if hourly_trend else tech_5m
        regime_ctx = self._strategy.evaluate_regime(cached_df)
        ctx = f"{base} | {regime_ctx}" if regime_ctx else base
        divergence = self._calc.calculate_divergence(cached_df)
        if divergence["strength"] > 0.0:
            ctx = (
                f"{ctx} | Divergence: RSI={divergence['rsi_divergence']} "
                f"MACD={divergence['macd_divergence']} strength={divergence['strength']:.2f}"
            )
        return ctx

    _LLM_SYSTEM_PROMPT = (
        "You are an elite quantitative trading intelligence engine analyzing historical chart data. "
        "No news context is available for this replay — evaluate based on technical indicators only.\n\n"
        "### ANALYTICAL FRAMEWORK\n"
        "Assess: trend (moving averages), momentum (RSI, MACD), mean-reversion (Bollinger Bands), "
        "price vs VWAP, and market regime.\n"
        "REGIME RULES:\n"
        "  - RANGING (ADX<20): indicators are noisy — only trade extreme RSI or clear BB touches; "
        "prefer NEUTRAL unless the setup is very clean.\n"
        "  - DEVELOPING (ADX 20-25): emerging trend — require at least two confirming signals.\n"
        "  - TRENDING (ADX≥25): follow the trend direction; momentum signals carry high weight.\n"
        "VOLUME RULES: Low volume (<0.7× avg) weakens any signal — lower confidence or go NEUTRAL.\n"
        "Rate conviction 0.5 to 1.0 when indicators clearly align. "
        "Output NEUTRAL when indicators conflict or the setup is ambiguous.\n\n"
        "### STRICT OUTPUT PROTOCOL\n"
        "Output ONLY raw, unformatted JSON — no markdown, no explanation:\n"
        "{\"direction\": \"BULLISH\" | \"BEARISH\" | \"NEUTRAL\", \"confidence\": <float>}"
    )

    def _parse_llm_raw(self, raw: str) -> tuple[str, float]:
        if raw.startswith("```"):
            raw = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", raw,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
        parsed = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        if isinstance(parsed, dict):
            direction = str(parsed.get("direction", "NEUTRAL")).upper()
            if direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
                direction = "NEUTRAL"
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
            return direction, confidence
        dm = re.search(r"\b(BULLISH|BEARISH|NEUTRAL)\b", raw, re.IGNORECASE)
        if dm:
            return dm.group(1).upper(), 0.5
        return "NEUTRAL", 0.0

    async def _llm_call(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        technical_context: str,
        model: str,
    ) -> tuple[str, float]:
        user_prompt = (
            f"Ticker: {symbol}\n"
            f"Technical Setup: {technical_context}\n\n"
            "Return the JSON evaluation now."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.4},
        }
        try:
            async with session.post(f"{self.ollama_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
            raw = data.get("message", {}).get("content", "").strip()
            return self._parse_llm_raw(raw)
        except Exception as exc:
            self.logger.warning("%s failed for %s: %s — defaulting to NEUTRAL.", model, symbol, exc)
            return "NEUTRAL", 0.0

    async def _llm_evaluate(
        self,
        symbol: str,
        technical_context: str,
    ) -> tuple[str, float]:
        """
        Two-stage LLM evaluation:
          1. openclaw (llama3.2) — primary assessment
          2. qwen_reviewer (fine-tuned Qwen) — reviews primary and makes final call
        Qwen is only invoked when openclaw returns a non-NEUTRAL signal,
        keeping inference overhead low on quiet bars.
        """
        timeout = aiohttp.ClientTimeout(total=90, connect=10, sock_read=80)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # ── Step 1: primary evaluator ──────────────────────────────────────
            primary_dir, primary_conf = await self._llm_call(
                session, symbol, technical_context, model="openclaw",
            )
            self.logger.debug(
                "Primary (openclaw) %s → %s %.2f", symbol, primary_dir, primary_conf,
            )
            self._openclaw_tally[primary_dir] = self._openclaw_tally.get(primary_dir, 0) + 1

            if primary_dir == "NEUTRAL":
                return "NEUTRAL", 0.0

            # ── Step 2: Qwen reviewer — final decision ─────────────────────────
            review_ctx = (
                f"{technical_context}\n\n"
                f"Primary assessment: {primary_dir} at {primary_conf:.0%} confidence. "
                "Review this setup independently and return your final evaluation."
            )
            final_dir, final_conf = await self._llm_call(
                session, symbol, review_ctx, model="qwen_reviewer",
            )
            self.logger.debug(
                "Reviewer (qwen) %s → %s %.2f", symbol, final_dir, final_conf,
            )
            self._qwen_tally[final_dir] = self._qwen_tally.get(final_dir, 0) + 1
            return final_dir, final_conf

    # ── Symbol replay ──────────────────────────────────────────────────────────

    async def _replay_symbol(
        self,
        symbol: str,
        bars_5m: list[BarData],
        bars_1h: list[BarData],
        starting_equity: float,
        daily_global_pnl: dict[str, float] | None = None,
    ) -> tuple[list[TradeRecord], float, int]:
        """
        Sequential bar-by-bar replay.  Fill price is always bar[i+1].open so
        the signal bar's close is never used as the entry/exit price.
        Returns (trades, final_equity, signals_attempted).
        """
        trades: list[TradeRecord] = []
        equity = starting_equity
        signals_attempted = 0

        # Open-position state
        in_position = False
        direction = "LONG"
        entry_price = 0.0
        entry_time: datetime = datetime.now(timezone.utc)
        quantity = 0
        peak_price = 0.0     # tracks highest (LONG) or lowest (SHORT) seen price
        initial_atr = 0.0
        entry_bar_idx = -1
        last_exit_bar = -1
        entry_confidence = 0.0
        partial_taken = False  # True after the 60% scale-out at TP1
        entry_snapshot: dict = {}  # entry conditions for tail forensics

        # LLM result cache — avoids duplicate calls for identical context strings
        llm_cache: dict[str, tuple[str, float]] = {}
        # Verdict tally — shows why the LLM rejected candidates (pick the right lever)
        llm_verdicts = {"pass": 0, "bull_low": 0, "neutral": 0, "bearish": 0}
        # Per-model tallies — separate openclaw's raw verdict from qwen's veto, so we
        # can tell WHICH model is the wall (openclaw only calls qwen on non-NEUTRAL).
        self._openclaw_tally: dict[str, int] = {}
        self._qwen_tally: dict[str, int] = {}

        # Pre-compute full indicator DataFrames once — sliced per bar, no per-bar recalculation
        full_df_5m = self._calc.calculate_all(bars_5m) if bars_5m else pd.DataFrame()
        full_df_1h = self._calc.calculate_all(bars_1h) if bars_1h else pd.DataFrame()

        n = len(bars_5m)

        for i, bar in enumerate(bars_5m):
            # We always need the next bar for realistic fill
            if i + 1 >= n:
                break

            current_price = float(bar.close)
            current_time = _to_utc(bar.timestamp)
            fill_price = float(bars_5m[i + 1].open)
            bar_et = current_time.astimezone(ZoneInfo("America/New_York"))
            # Live liquidates everything at 15:50 ET and never holds overnight —
            # mirror that so overnight gaps can't appear in backtest results.
            in_pre_close = bar_et.hour == 15 and bar_et.minute >= 50

            # Rolling windows — no future bars (list used only for length checks)
            window_5m = bars_5m[max(0, i - self.MAX_LOOKBACK_5M + 1): i + 1]
            window_1h = _hourly_bars_up_to(bars_1h, current_time)

            # Slice pre-computed DataFrames — O(1), no recalculation
            current_atr = 0.0
            cached_df = None
            df_1h_slice = full_df_1h.iloc[:len(window_1h)] if not full_df_1h.empty else pd.DataFrame()
            if i >= self.WARMUP_BARS - 1 and not full_df_5m.empty:
                start = max(0, i - self.MAX_LOOKBACK_5M + 1)
                cached_df = full_df_5m.iloc[start: i + 1]
                if not cached_df.empty:
                    atr_val = cached_df.iloc[-1].get("atr_14")
                    if atr_val is not None:
                        try:
                            v = float(atr_val)
                            if not math.isnan(v):
                                current_atr = v
                        except (TypeError, ValueError):
                            pass

            # ── Exit checks (evaluated before entry each bar) ──
            if in_position:
                bars_held = i - entry_bar_idx
                exit_reason: str | None = None
                stop_fill: float | None = None   # set when a resting stop fills intrabar
                bar_high = float(bar.high)
                bar_low = float(bar.low)

                is_partial = False   # True when this exit is the TP1 60% scale-out

                # 0. End-of-day liquidation (15:50 ET) — live force-flattens all
                #    positions pre-close and never holds overnight.
                if in_pre_close:
                    exit_reason = "End-of-day liquidation"

                # 1. Hard stop — a resting broker stop, filled intrabar at the stop level
                #    (or the open on a gap). Before TP1 it sits at STOP_LOSS_PCT; after the
                #    scale-out it has been raised to breakeven (entry) so the trade can't
                #    turn into a loss.
                if exit_reason is None:
                    if direction == "LONG":
                        stop_level = entry_price if partial_taken else entry_price * (1 - self.STOP_LOSS_PCT)
                        if bar_low <= stop_level:
                            exit_reason = "Breakeven stop" if partial_taken else "2% stop-loss"
                            stop_fill = min(float(bar.open), stop_level)
                    else:
                        stop_level = entry_price if partial_taken else entry_price * (1 + self.STOP_LOSS_PCT)
                        if bar_high >= stop_level:
                            exit_reason = "Breakeven stop" if partial_taken else "2% stop-loss"
                            stop_fill = max(float(bar.open), stop_level)

                # 2. ATR trailing stop (anchored to initial_atr). After the scale-out the
                #    trail is floored at breakeven so the runner can't give back to a loss.
                if exit_reason is None and initial_atr > 0:
                    if direction == "LONG":
                        if current_price > peak_price:
                            peak_price = current_price
                        trail = peak_price - self.ATR_TRAIL_MULT * initial_atr
                        if partial_taken:
                            trail = max(trail, entry_price)
                        if bar_low <= trail:
                            exit_reason = "ATR trailing stop"
                            stop_fill = min(float(bar.open), trail)
                    else:
                        if current_price < peak_price:
                            peak_price = current_price
                        trail = peak_price + self.ATR_TRAIL_MULT * initial_atr
                        if partial_taken:
                            trail = min(trail, entry_price)
                        if bar_high >= trail:
                            exit_reason = "ATR trailing stop"
                            stop_fill = max(float(bar.open), trail)

                # 3. Take-profit. TP1 (entry + 3×ATR) scales out SCALE_OUT_PCT and keeps
                #    the runner; TP2 (entry + 6×ATR) closes the remainder.
                if exit_reason is None and initial_atr > 0 and bars_held >= self.MIN_HOLD_BARS:
                    tp_mult = self.TAKE_PROFIT_2_ATR_MULT if partial_taken else self.TAKE_PROFIT_ATR_MULT
                    if direction == "LONG":
                        if current_price >= entry_price + tp_mult * initial_atr:
                            exit_reason = "Take-profit"
                            is_partial = not partial_taken
                    else:
                        if current_price <= entry_price - tp_mult * initial_atr:
                            exit_reason = "Take-profit"
                            is_partial = not partial_taken

                # 4. Signal reversal (minimum hold period passed)
                if exit_reason is None and bars_held >= self.MIN_HOLD_BARS:
                    sig, _ = self._compute_signal(window_5m, window_1h, cached_df_1h=df_1h_slice)
                    if direction == "LONG" and sig == "BEARISH":
                        exit_reason = "Signal reversal"
                    elif direction == "SHORT" and sig == "BULLISH":
                        exit_reason = "Signal reversal"

                if exit_reason:
                    # Resting stops fill intrabar at the stop price; signal/TP exits are
                    # decided on the close, so they fill at the next bar's open.
                    exit_price = stop_fill if stop_fill is not None else fill_price

                    # Scale-out: at TP1 sell SCALE_OUT_PCT, keep the rest as a runner with
                    # the stop raised to breakeven. Needs ≥1 share on each side to split.
                    scale_qty = int(quantity * self.SCALE_OUT_PCT)
                    if is_partial and scale_qty >= 1 and (quantity - scale_qty) >= 1:
                        close_qty = scale_qty
                    else:
                        close_qty = quantity   # full close (TP2, stops, reversal, or too small to split)

                    if direction == "LONG":
                        pnl = (exit_price - entry_price) * close_qty - 2.0 * self.commission_per_trade
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - exit_price) * close_qty - 2.0 * self.commission_per_trade
                        pnl_pct = (entry_price - exit_price) / entry_price

                    equity += pnl
                    if daily_global_pnl is not None:
                        day_key = current_time.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
                        daily_global_pnl[day_key] = daily_global_pnl.get(day_key, 0.0) + pnl
                    trades.append(TradeRecord(
                        symbol=symbol,
                        direction=direction,
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=close_qty,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=("Partial take-profit" if close_qty != quantity else exit_reason),
                        confidence=entry_confidence,
                        entry_conditions=entry_snapshot,
                    ))

                    if close_qty != quantity:
                        # Runner stays open: reduce size, raise stop to breakeven.
                        quantity -= close_qty
                        partial_taken = True
                    else:
                        in_position = False
                        partial_taken = False
                        last_exit_bar = i
                    continue

            # ── Entry ──
            if not in_position:
                # Cooldown gate
                if last_exit_bar >= 0 and (i - last_exit_bar) < self.COOLDOWN_BARS:
                    continue

                # Opening 30-minute noise filter (9:30–10:00 ET)
                # IBKR RTH data starts at 9:30, so hour==9 covers the full noisy opening window
                if bar_et.hour == 9:
                    continue

                # No new entries in the pre-close window — live liquidates at 15:50.
                if in_pre_close:
                    continue

                # Daily drawdown circuit breaker — halt new entries once the shared
                # account-level daily P&L crosses -DAILY_LOSS_LIMIT_PCT of equity
                if daily_global_pnl is not None:
                    today_str = bar_et.strftime("%Y-%m-%d")
                    if daily_global_pnl.get(today_str, 0.0) < -(self.DAILY_LOSS_LIMIT_PCT * starting_equity):
                        continue

                # Pre-filter: fast indicator vote — avoids LLM call on bars with no activity
                sig_det, conf_det = self._compute_signal(window_5m, window_1h, cached_df=cached_df, cached_df_1h=df_1h_slice)
                # Live is long-only: the LLM/news step only runs on BULLISH technical
                # setups, so mirror that here (BEARISH setups are never entered).
                if sig_det != "BULLISH":
                    continue
                signals_attempted += 1

                # The LLM decides every entry: technicals only qualify the candidate.
                # Cached result for an identical context, otherwise a fresh call.
                tech_ctx = self._build_technical_context(cached_df, df_1h=df_1h_slice)
                if tech_ctx in llm_cache:
                    llm_sig, llm_conf = llm_cache[tech_ctx]
                else:
                    llm_sig, llm_conf = await self._llm_evaluate(symbol, tech_ctx)
                    llm_cache[tech_ctx] = (llm_sig, llm_conf)

                # Tally the verdict so we can see *why* candidates are rejected:
                # low-confidence bullish (→ lower the gate) vs neutral/bearish
                # (→ the model itself is conservative; retrain).
                if llm_sig == "BULLISH" and llm_conf >= self.LLM_CONF_THRESHOLD:
                    llm_verdicts["pass"] += 1
                elif llm_sig == "BULLISH":
                    llm_verdicts["bull_low"] += 1
                elif llm_sig == "BEARISH":
                    llm_verdicts["bearish"] += 1
                else:
                    llm_verdicts["neutral"] += 1

                # No technical-fallback entries — a NEUTRAL/BEARISH or low-confidence
                # LLM verdict means no trade, exactly like live.
                if llm_sig != "BULLISH" or llm_conf < self.LLM_CONF_THRESHOLD:
                    continue
                conf = llm_conf

                qty = max(1, int((equity * self.POSITION_PCT * conf) / fill_price))
                direction = "LONG"
                entry_price = fill_price
                entry_time = current_time
                quantity = qty
                peak_price = entry_price
                initial_atr = current_atr
                entry_bar_idx = i
                entry_confidence = conf
                partial_taken = False
                entry_snapshot = self._capture_entry_conditions(
                    cached_df, bar_et, fill_price, current_atr, source="llm",
                )
                in_position = True

        # Close any position still open at the last available bar
        if in_position and n > 0:
            last_bar = bars_5m[-1]
            last_price = float(last_bar.close)
            last_time = _to_utc(last_bar.timestamp)

            if direction == "LONG":
                pnl = (last_price - entry_price) * quantity - 2.0 * self.commission_per_trade
                pnl_pct = (last_price - entry_price) / entry_price
            else:
                pnl = (entry_price - last_price) * quantity - 2.0 * self.commission_per_trade
                pnl_pct = (entry_price - last_price) / entry_price

            equity += pnl
            trades.append(TradeRecord(
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                exit_time=last_time,
                entry_price=entry_price,
                exit_price=last_price,
                quantity=quantity,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason="End-of-backtest close",
                confidence=entry_confidence,
                entry_conditions=entry_snapshot,
            ))

        v = llm_verdicts
        evaluated = v["pass"] + v["bull_low"] + v["neutral"] + v["bearish"]
        self.logger.info(
            "%s LLM verdicts on %d candidates | passed=%d | bullish<%.2f=%d | neutral=%d | bearish=%d",
            symbol, evaluated, v["pass"], self.LLM_CONF_THRESHOLD, v["bull_low"], v["neutral"], v["bearish"],
        )
        oc, qw = self._openclaw_tally, self._qwen_tally
        self.logger.info(
            "%s openclaw verdicts | BULLISH=%d | BEARISH=%d | NEUTRAL=%d  →  qwen review of the "
            "non-neutral | BULLISH=%d | BEARISH=%d | NEUTRAL=%d",
            symbol, oc.get("BULLISH", 0), oc.get("BEARISH", 0), oc.get("NEUTRAL", 0),
            qw.get("BULLISH", 0), qw.get("BEARISH", 0), qw.get("NEUTRAL", 0),
        )

        return trades, equity, signals_attempted

    # ── Public entry point ─────────────────────────────────────────────────────

    async def run(
        self,
        tickers: list[str],
        account_equity: float | None = None,
        duration: str = "3 M",
    ) -> BacktestResult:
        start_eq = account_equity if account_equity and account_equity > 0 else self.start_equity
        result = BacktestResult(
            start_equity=start_eq,
            tickers=list(tickers),
            duration_days=90,
        )

        self.logger.info(
            "Backtest starting | tickers=%s | duration=%s | equity=%.2f",
            tickers, duration, start_eq,
        )

        # Fetch then immediately replay each ticker before moving to the next.
        # This surfaces per-ticker results in the terminal as they complete and
        # keeps IBKR pacing safe — the replay time absorbs most of the required
        # inter-request gap; the explicit sleep at the end provides the minimum.
        equity = start_eq
        all_trades: list[TradeRecord] = []
        fetched_any = False
        # Shared daily P&L across all symbols — lets the circuit breaker operate
        # at the account level rather than per-symbol
        daily_global_pnl: dict[str, float] = {}

        for symbol in tickers:
            self.logger.info("Fetching history for %s...", symbol)
            bars = await self._fetch_bars(symbol, duration)
            bars_5m = bars.get("5 mins", [])
            bars_1h = bars.get("1 hour", [])
            n5 = len(bars_5m)
            n1 = len(bars_1h)
            result.bars_fetched[symbol] = n5
            self.logger.info("%s: 5m=%d bars, 1h=%d bars.", symbol, n5, n1)

            if n5 == 0:
                self.logger.warning("Skipping %s — IBKR returned 0 bars (pacing or data issue).", symbol)
                await asyncio.sleep(self.IBKR_SYMBOL_PAUSE)
                continue

            fetched_any = True

            if n5 < self.WARMUP_BARS + 2:
                self.logger.warning(
                    "%s: only %d 5m bars — need at least %d. Skipping.",
                    symbol, n5, self.WARMUP_BARS + 2,
                )
                result.bars_replayed[symbol] = 0
                await asyncio.sleep(self.IBKR_SYMBOL_PAUSE)
                continue

            self.logger.info("Replaying %s (%d bars, LLM active)...", symbol, n5)
            sym_trades, equity, sig_count = await self._replay_symbol(
                symbol, bars_5m, bars_1h, equity,
                daily_global_pnl=daily_global_pnl,
            )
            result.bars_replayed[symbol] = n5
            result.signals_fired[symbol] = sig_count
            all_trades.extend(sym_trades)
            self.logger.info(
                "%s done — %d trades, %d signals, equity $%.2f.",
                symbol, len(sym_trades), sig_count, equity,
            )

            # Minimum IBKR pacing gap before the next symbol's fetch.
            # When LLM replay takes longer than IBKR_SYMBOL_PAUSE the sleep is
            # still useful as a brief yield back to the event loop.
            await asyncio.sleep(self.IBKR_SYMBOL_PAUSE)

        if not fetched_any:
            self.logger.error("No data fetched for any ticker. Backtest aborted.")
            return result

        # Count days where the circuit breaker was active
        limit = self.DAILY_LOSS_LIMIT_PCT * start_eq
        result.halted_days = sum(1 for pnl in daily_global_pnl.values() if pnl < -limit)

        # Sort chronologically and compute all metrics
        all_trades.sort(key=lambda t: t.entry_time)
        result.trades = all_trades
        result.finalise()
        result.total_commissions = 2.0 * self.commission_per_trade * result.total_trades

        self._log_tail_forensics(all_trades)

        self.logger.info(
            "Backtest complete — %d trades, %.2f%% return, Sharpe %.2f, max DD %.2f%%.",
            result.total_trades,
            result.total_return * 100,
            result.sharpe_ratio,
            result.max_drawdown * 100,
        )
        return result
