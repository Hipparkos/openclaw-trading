from __future__ import annotations

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

        return (
            f"The SMA_5 is {_format_value(latest_row.get('sma_5'))}. "
            f"The Close is {_format_value(latest_row.get('close'))}. "
            f"The RSI_14 is {_format_value(latest_row.get('rsi_14'))}. "
            f"The VWAP is {_format_value(_get_value('vwap'))}. "
            f"The Bollinger Bands are lower {_format_value(_get_value('BBL_20_2.0', 'BBL_20_2'))} and upper {_format_value(_get_value('BBU_20_2.0', 'BBU_20_2'))}. "
            f"The MACD is {_format_value(latest_row.get('MACD_6_20_9'))} and the Signal is {_format_value(latest_row.get('MACDs_6_20_9'))}."
        )