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
    SIGNAL_THRESHOLD = 3      # out of 4 indicators — 3/4 required for higher win rate
    STOP_LOSS_PCT = 0.02
    ATR_TRAIL_MULT = 2.0
    TAKE_PROFIT_ATR_MULT = 3.0
    POSITION_PCT = 0.015
    MIN_HOLD_BARS = 3         # 15 minutes before AI-reversal exit allowed
    COOLDOWN_BARS = 3         # 15-minute cooldown after close
    LLM_COOLDOWN_BARS = 78    # one full trading session between fresh LLM calls per symbol
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
                    "Backtest fetch: %s [%s] → %d bars",
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
        Pure indicator-alignment signal.  Mirrors what the LLM synthesises from
        the same inputs — five binary votes, threshold = 2 net in either direction.

        Returns (direction, confidence) where confidence is the fraction of
        indicators that voted for the winning side (0.0–1.0).
        cached_df: pre-computed result of calculate_all(window_5m) to avoid recomputing.
        cached_df_1h: pre-computed 1h DataFrame slice — skips calculate_all(window_1h).
        """
        if len(window_5m) < self.WARMUP_BARS:
            return "NEUTRAL", 0.0

        df = cached_df if cached_df is not None else self._calc.calculate_all(window_5m)
        if df is None or df.empty:
            return "NEUTRAL", 0.0

        row = df.iloc[-1]
        score = 0
        components = 0

        def _v(key: str) -> float | None:
            val = row.get(key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return None
            return float(val)

        close = _v("close")

        # ── Hard gate: 1h macro trend must be clear and unambiguous ────────────
        # Any entry that opposes the hourly direction is rejected outright.
        # NEUTRAL hourly (price straddling SMA_50) also blocks all entries.
        hourly_direction = "NEUTRAL"
        df_1h = cached_df_1h if cached_df_1h is not None else (
            self._calc.calculate_all(window_1h) if len(window_1h) >= 50 else pd.DataFrame()
        )
        if not df_1h.empty:
            h = df_1h.iloc[-1]
            h_close = h.get("close")
            h_sma50 = h.get("sma_50")
            if h_close is not None and h_sma50 is not None:
                try:
                    hc, hs = float(h_close), float(h_sma50)
                    if not (math.isnan(hc) or math.isnan(hs)):
                        if hc > hs:
                            hourly_direction = "BULLISH"
                        elif hc < hs:
                            hourly_direction = "BEARISH"
                except (TypeError, ValueError):
                    pass

        if hourly_direction == "NEUTRAL":
            return "NEUTRAL", 0.0

        # ── Volume confirmation gate ─────────────────────────────────────────────
        # Low-volume bars produce unreliable breakouts — skip them
        vol = row.get("volume")
        vol_sma = row.get("volume_sma_20")
        if vol is not None and vol_sma is not None:
            try:
                vol_f, vsma_f = float(vol), float(vol_sma)
                if vsma_f > 0 and not math.isnan(vol_f) and not math.isnan(vsma_f):
                    if vol_f / vsma_f < 0.8:
                        return "NEUTRAL", 0.0
            except (TypeError, ValueError):
                pass

        # ── 5m indicator votes (4 signals) ──────────────────────────────────────
        # 1. RSI momentum
        rsi = _v("rsi_14")
        if rsi is not None:
            components += 1
            score += 1 if rsi > 55 else (-1 if rsi < 45 else 0)

        # 2. MACD crossover
        macd, macd_s = _v("MACD_6_20_9"), _v("MACDs_6_20_9")
        if macd is not None and macd_s is not None:
            components += 1
            score += 1 if macd > macd_s else (-1 if macd < macd_s else 0)

        # 3. Price vs VWAP
        vwap = _v("vwap")
        if close is not None and vwap is not None:
            components += 1
            score += 1 if close > vwap else (-1 if close < vwap else 0)

        # 4. Price vs SMA_5
        sma5 = _v("sma_5")
        if close is not None and sma5 is not None:
            components += 1
            score += 1 if close > sma5 else (-1 if close < sma5 else 0)

        if components == 0:
            return "NEUTRAL", 0.0

        confidence = abs(score) / components

        if score >= self.SIGNAL_THRESHOLD:
            direction = "BULLISH"
        elif score <= -self.SIGNAL_THRESHOLD:
            direction = "BEARISH"
        else:
            return "NEUTRAL", 0.0

        # ── Trend alignment check ────────────────────────────────────────────────
        if direction != hourly_direction:
            return "NEUTRAL", 0.0

        return direction, confidence

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
            self.logger.warning("LLM call (%s) failed for %s: %s — NEUTRAL", model, symbol, exc)
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
            return final_dir, final_conf

    # ── Symbol replay ──────────────────────────────────────────────────────────

    async def _replay_symbol(
        self,
        symbol: str,
        bars_5m: list[BarData],
        bars_1h: list[BarData],
        starting_equity: float,
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

        # LLM rate-limiting: cache results by context string + one-per-session cooldown
        llm_cache: dict[str, tuple[str, float]] = {}
        last_llm_bar = -self.LLM_COOLDOWN_BARS

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

                # 1. Hard stop-loss
                if direction == "LONG":
                    loss_pct = (entry_price - current_price) / entry_price
                else:
                    loss_pct = (current_price - entry_price) / entry_price

                if loss_pct >= self.STOP_LOSS_PCT:
                    exit_reason = "2% stop-loss"

                # 2. ATR trailing stop
                if exit_reason is None and current_atr > 0:
                    if direction == "LONG":
                        if current_price > peak_price:
                            peak_price = current_price
                        if current_price <= peak_price - self.ATR_TRAIL_MULT * current_atr:
                            exit_reason = "ATR trailing stop"
                    else:
                        if current_price < peak_price:
                            peak_price = current_price
                        if current_price >= peak_price + self.ATR_TRAIL_MULT * current_atr:
                            exit_reason = "ATR trailing stop"

                # 3. Take-profit (3× ATR from entry)
                if exit_reason is None and initial_atr > 0 and bars_held >= self.MIN_HOLD_BARS:
                    if direction == "LONG":
                        if current_price >= entry_price + self.TAKE_PROFIT_ATR_MULT * initial_atr:
                            exit_reason = "Take-profit"
                    else:
                        if current_price <= entry_price - self.TAKE_PROFIT_ATR_MULT * initial_atr:
                            exit_reason = "Take-profit"

                # 4. Signal reversal (minimum hold period passed)
                if exit_reason is None and bars_held >= self.MIN_HOLD_BARS:
                    sig, _ = self._compute_signal(window_5m, window_1h, cached_df_1h=df_1h_slice)
                    if direction == "LONG" and sig == "BEARISH":
                        exit_reason = "Signal reversal"
                    elif direction == "SHORT" and sig == "BULLISH":
                        exit_reason = "Signal reversal"

                if exit_reason:
                    exit_price = fill_price
                    if direction == "LONG":
                        pnl = (exit_price - entry_price) * quantity - 2.0 * self.commission_per_trade
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - exit_price) * quantity - 2.0 * self.commission_per_trade
                        pnl_pct = (entry_price - exit_price) / entry_price

                    equity += pnl
                    trades.append(TradeRecord(
                        symbol=symbol,
                        direction=direction,
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=quantity,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        confidence=entry_confidence,
                    ))
                    in_position = False
                    last_exit_bar = i
                    continue

            # ── Entry ──
            if not in_position:
                # Cooldown gate
                if last_exit_bar >= 0 and (i - last_exit_bar) < self.COOLDOWN_BARS:
                    continue

                # Pre-filter: fast indicator vote — avoids LLM call on bars with no activity
                sig_det, conf_det = self._compute_signal(window_5m, window_1h, cached_df=cached_df, cached_df_1h=df_1h_slice)
                if sig_det == "NEUTRAL":
                    continue
                signals_attempted += 1

                # LLM confirmation: cached result → cooldown fresh call → else skip to deterministic
                tech_ctx = self._build_technical_context(cached_df, df_1h=df_1h_slice)
                if tech_ctx in llm_cache:
                    llm_sig, llm_conf = llm_cache[tech_ctx]
                elif (i - last_llm_bar) >= self.LLM_COOLDOWN_BARS:
                    llm_sig, llm_conf = await self._llm_evaluate(symbol, tech_ctx)
                    llm_cache[tech_ctx] = (llm_sig, llm_conf)
                    last_llm_bar = i
                else:
                    llm_sig, llm_conf = "NEUTRAL", 0.0

                # Resolve the effective signal and confidence
                if llm_sig != "NEUTRAL":
                    sig, conf = llm_sig, llm_conf
                else:
                    sig, conf = sig_det, conf_det

                if llm_sig != "NEUTRAL":
                    if conf < 0.60:
                        continue
                else:
                    if conf < 0.55:
                        continue

                qty = max(1, int((equity * self.POSITION_PCT * conf) / fill_price))
                direction = "LONG" if sig == "BULLISH" else "SHORT"
                entry_price = fill_price
                entry_time = current_time
                quantity = qty
                peak_price = entry_price
                initial_atr = current_atr
                entry_bar_idx = i
                entry_confidence = conf
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
            ))

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

        for symbol in tickers:
            self.logger.info("Fetching historical bars for %s...", symbol)
            bars = await self._fetch_bars(symbol, duration)
            bars_5m = bars.get("5 mins", [])
            bars_1h = bars.get("1 hour", [])
            n5 = len(bars_5m)
            n1 = len(bars_1h)
            result.bars_fetched[symbol] = n5
            self.logger.info("%s → 5m=%d bars, 1h=%d bars", symbol, n5, n1)

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

            self.logger.info("Replaying %s — %d bars (LLM active)", symbol, n5)
            sym_trades, equity, sig_count = await self._replay_symbol(
                symbol, bars_5m, bars_1h, equity
            )
            result.bars_replayed[symbol] = n5
            result.signals_fired[symbol] = sig_count
            all_trades.extend(sym_trades)
            self.logger.info(
                "%s done | trades=%d | signals=%d | equity_after=%.2f",
                symbol, len(sym_trades), sig_count, equity,
            )

            # Minimum IBKR pacing gap before the next symbol's fetch.
            # When LLM replay takes longer than IBKR_SYMBOL_PAUSE the sleep is
            # still useful as a brief yield back to the event loop.
            await asyncio.sleep(self.IBKR_SYMBOL_PAUSE)

        if not fetched_any:
            self.logger.error("No data fetched for any ticker. Backtest aborted.")
            return result

        # Sort chronologically and compute all metrics
        all_trades.sort(key=lambda t: t.entry_time)
        result.trades = all_trades
        result.finalise()
        result.total_commissions = 2.0 * self.commission_per_trade * result.total_trades

        self.logger.info(
            "Backtest complete | trades=%d | return=%.2f%% | sharpe=%.2f | max_dd=%.2f%%",
            result.total_trades,
            result.total_return * 100,
            result.sharpe_ratio,
            result.max_drawdown * 100,
        )
        return result
