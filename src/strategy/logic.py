from __future__ import annotations

import pandas as pd


class StrategyEngine:
    def evaluate_signals(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "No market data is available yet."

        latest_row = df.iloc[-1]
        def _format_value(value: object) -> str:
            return "N/A" if value is None or pd.isna(value) else f"{value:.2f}"

        return (
            f"The SMA_5 is {_format_value(latest_row.get('sma_5'))}. "
            f"The Close is {_format_value(latest_row.get('close'))}. "
            f"The MACD is {_format_value(latest_row.get('MACD_6_20_9'))} and the Signal is {_format_value(latest_row.get('MACDs_6_20_9'))}."
        )