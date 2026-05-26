from __future__ import annotations
import logging

import pandas as pd


class StrategyEngine:
    def __init__(self):
        # Add a logger for the strategy
        self.logger = logging.getLogger("StrategyEngine")

    # Entry signals.
    def evaluate_signals(self, df: pd.DataFrame) -> dict:
        symbol = None
        if df is not None and not df.empty and "symbol" in df.columns:
            symbol = df["symbol"].iloc[-1]

        result = {"signal": "HOLD", "setup_type": None, "symbol": symbol}

        required_columns = {"close", "high", "sma_5", "MACD_6_20_9", "MACDs_6_20_9"}
        if df is None or df.empty or len(df) < 2 or not required_columns.issubset(df.columns):
            return result

        working_df = df.sort_index().copy()
        working_df = working_df.dropna(subset=["close", "high", "sma_5", "MACD_6_20_9", "MACDs_6_20_9"])
        if len(working_df) < 2:
            return result

        prev = working_df.iloc[-2]
        current = working_df.iloc[-1]

        fresh_620_cross_up = (
            prev["MACD_6_20_9"] < prev["MACDs_6_20_9"]
            and current["MACD_6_20_9"] > current["MACDs_6_20_9"]
        )

        if not fresh_620_cross_up:
            return result

        dip_buy = prev["close"] < prev["sma_5"] and current["close"] > current["sma_5"]

        current_day = current.name.normalize() if isinstance(current.name, pd.Timestamp) else None
        if current_day is not None:
            day_mask = working_df.index.normalize() == current_day
            intraday_high = working_df.loc[day_mask, "high"].shift(1).cummax().iloc[-1]
        else:
            intraday_high = working_df["high"].shift(1).cummax().iloc[-1]

        breakout = pd.notna(intraday_high) and current["close"] > intraday_high

        if dip_buy or breakout:
            setup = "Dip-Buy" if dip_buy else "Breakout"
            self.logger.info(
                f"\n=== {setup} SIGNAL MATH TRIGGERED FOR {symbol} ==="
                f"\nTime: {current.name}"
                f"\n[MACD Cross] Prev: {prev['MACD_6_20_9']:.4f} < {prev['MACDs_6_20_9']:.4f} | Curr: {current['MACD_6_20_9']:.4f} > {current['MACDs_6_20_9']:.4f}"
                f"\n[Dip-Buy Check] Prev Close: {prev['close']:.2f} (SMA: {prev['sma_5']:.2f}) | Curr Close: {current['close']:.2f} (SMA: {current['sma_5']:.2f})"
                f"\n[Breakout Check] Curr Close: {current['close']:.2f} | Intraday High: {intraday_high:.2f}"
                f"\n================================================="
            )

        if dip_buy:
            result.update({"signal": "BUY", "setup_type": "Dip-Buy"})
        elif breakout:
            result.update({"signal": "BUY", "setup_type": "Breakout"})

        return result