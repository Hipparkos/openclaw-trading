from __future__ import annotations

import datetime as _dt

import pandas as pd


class StrategyEngine:
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

        # ── Session timing (US Eastern) ─────────────────────────────────────────
        try:
            from zoneinfo import ZoneInfo  # stdlib Python 3.9+
            ET = ZoneInfo("America/New_York")
            ts = df.index[-1]
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                ts_et = ts.astimezone(ET)
            else:
                ts_et = ts.replace(tzinfo=_dt.timezone.utc).astimezone(ET)
            total_min = ts_et.hour * 60 + ts_et.minute
            if total_min <= 10 * 60:           # up to 10:00 ET
                session = "OPEN"
            elif total_min >= 15 * 60 + 30:    # 15:30 ET onward
                session = "CLOSE"
            else:
                session = "MID_DAY"
            parts.append(f"Session: {session}")
        except Exception:
            pass

        return "Regime: " + ", ".join(parts) if parts else ""