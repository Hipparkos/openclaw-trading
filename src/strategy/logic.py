from __future__ import annotations

import math

import pandas as pd


class StrategyEngine:
    def compute_alignment_signal(
        self,
        df_5m: pd.DataFrame,
        df_1h: pd.DataFrame | None = None,
        signal_threshold: int = 3,
        warmup_bars: int = 50,
    ) -> tuple[str, float]:
        """Pure indicator-alignment signal — four binary votes (RSI, MACD, price
        vs VWAP, price vs SMA-5) gated by a clear hourly trend and adequate volume.

        Returns (direction, confidence) where confidence is the fraction of
        indicators that voted for the winning side (0.0–1.0). Shared by live
        trading and the backtest so both gate entries identically.
        """
        if df_5m is None or df_5m.empty or len(df_5m) < warmup_bars:
            return "NEUTRAL", 0.0

        row = df_5m.iloc[-1]

        def _v(key: str) -> float | None:
            val = row.get(key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return None
            return float(val)

        close = _v("close")

        # Hard gate: 1h macro trend must be clear and unambiguous. NEUTRAL hourly
        # (price straddling SMA_50) blocks all entries.
        hourly_direction = "NEUTRAL"
        if df_1h is not None and not df_1h.empty:
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

        # Volume confirmation gate — low-volume bars produce unreliable breakouts.
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

        # 5m indicator votes (4 signals)
        score = 0
        components = 0

        rsi = _v("rsi_14")
        if rsi is not None:
            components += 1
            score += 1 if rsi > 55 else (-1 if rsi < 45 else 0)

        macd, macd_s = _v("MACD_6_20_9"), _v("MACDs_6_20_9")
        if macd is not None and macd_s is not None:
            components += 1
            score += 1 if macd > macd_s else (-1 if macd < macd_s else 0)

        vwap = _v("vwap")
        if close is not None and vwap is not None:
            components += 1
            score += 1 if close > vwap else (-1 if close < vwap else 0)

        sma5 = _v("sma_5")
        if close is not None and sma5 is not None:
            components += 1
            score += 1 if close > sma5 else (-1 if close < sma5 else 0)

        if components == 0:
            return "NEUTRAL", 0.0

        confidence = abs(score) / components

        if score >= signal_threshold:
            direction = "BULLISH"
        elif score <= -signal_threshold:
            direction = "BEARISH"
        else:
            return "NEUTRAL", 0.0

        # Entry must agree with the hourly macro trend.
        if direction != hourly_direction:
            return "NEUTRAL", 0.0

        return direction, confidence

    def evaluate_hourly_trend(self, df: pd.DataFrame) -> str:
        """Evaluate the overall trend direction from hourly data.

        Returns a concise trend assessment (BULLISH/BEARISH/NEUTRAL).
        """
        if df is None or df.empty:
            return "Insufficient hourly data."

        latest_row = df.iloc[-1]
        sma_20 = latest_row.get("sma_20")
        close = latest_row.get("close")
        rsi = latest_row.get("rsi_14")

        if sma_20 is None or pd.isna(sma_20) or close is None or pd.isna(close):
            return "Insufficient hourly data."

        try:
            sma_20 = float(sma_20)
            close = float(close)
            rsi = float(rsi) if rsi is not None and not pd.isna(rsi) else None
        except (TypeError, ValueError):
            return "Insufficient hourly data."

        if close > sma_20:
            trend = "BULLISH"
        elif close < sma_20:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        rsi_context = ""
        if rsi is not None:
            if rsi > 60:
                rsi_context = " (RSI strong)"
            elif rsi < 40:
                rsi_context = " (RSI weak)"

        return f"Hourly trend is {trend}{rsi_context}"

    def evaluate_signals(self, df: pd.DataFrame) -> str:
        """Evaluate detailed 5-minute technical signals."""
        if df is None or df.empty:
            return "No market data is available yet."

        latest_row = df.iloc[-1]

        def _format_value(value: object) -> str:
            return "N/A" if value is None or pd.isna(value) else f"{value:.2f}"

        def _get_value(*candidate_columns: str) -> object:
            for column in candidate_columns:
                value = latest_row.get(column)
                if value is not None and not pd.isna(value):
                    return value
            return None

        # Detect Bollinger Band columns dynamically — pandas_ta naming varies by version
        bbl_col = next((c for c in df.columns if c.startswith("BBL_")), None)
        bbu_col = next((c for c in df.columns if c.startswith("BBU_")), None)

        return (
            f"The SMA_5 is {_format_value(latest_row.get('sma_5'))}. "
            f"The Close is {_format_value(latest_row.get('close'))}. "
            f"The RSI_14 is {_format_value(latest_row.get('rsi_14'))}. "
            f"The VWAP is {_format_value(_get_value('vwap'))}. "
            f"The Bollinger Bands are lower {_format_value(latest_row.get(bbl_col)) if bbl_col else 'N/A'} "
            f"and upper {_format_value(latest_row.get(bbu_col)) if bbu_col else 'N/A'}. "
            f"The MACD is {_format_value(latest_row.get('MACD_6_20_9'))} and the Signal is {_format_value(latest_row.get('MACDs_6_20_9'))}."
        )

    def evaluate_regime(self, df: pd.DataFrame) -> str:
        """Return a concise market-regime string for the LLM prompt.

        Covers three dimensions:
          • Trend strength  — ADX: RANGING (<20) / DEVELOPING (20-25) / TRENDING (≥25)
          • Volume context  — current bar volume relative to 20-bar average
          • Session timing  — OPEN / MID_DAY / CLOSE in US Eastern time
        """
        if df is None or df.empty:
            return ""

        latest = df.iloc[-1]
        parts: list[str] = []

        # ── Trend strength (ADX) ────────────────────────────────────────────────
        adx_col = next((c for c in df.columns if c.startswith("ADX_")), None)
        if adx_col:
            adx_val = latest.get(adx_col)
            if adx_val is not None and not pd.isna(adx_val):
                adx = float(adx_val)
                if adx >= 25:
                    label = "TRENDING"
                elif adx >= 20:
                    label = "DEVELOPING"
                else:
                    label = "RANGING"
                parts.append(f"{label} (ADX={adx:.0f})")

        # ── Volume relative to 20-bar average ───────────────────────────────────
        vol = latest.get("volume")
        vol_avg = latest.get("volume_sma_20")
        if (
            vol is not None and vol_avg is not None
            and not pd.isna(vol) and not pd.isna(vol_avg)
            and float(vol_avg) > 0
        ):
            ratio = float(vol) / float(vol_avg)
            if ratio >= 1.5:
                vol_label = f"High ({ratio:.1f}×)"
            elif ratio >= 0.7:
                vol_label = f"Normal ({ratio:.1f}×)"
            else:
                vol_label = f"Low ({ratio:.1f}×)"
            parts.append(f"Volume: {vol_label}")

        return "Regime: " + ", ".join(parts) if parts else ""