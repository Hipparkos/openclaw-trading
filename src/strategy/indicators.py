from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from data.data_models import BarData


class IndicatorCalculator:
    # Build indicator dataframe.
    def calculate_all(self, bars: list[BarData]) -> pd.DataFrame:
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "symbol": bar.symbol,
                    "timeframe": bar.timeframe,
                }
                for bar in bars
            ]
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

        price_columns = ["open", "high", "low", "close", "volume"]
        df[price_columns] = df[price_columns].astype(float)

        df["rsi_14"] = ta.rsi(df["close"], length=14)

        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            df = df.join(macd)

        macd_620 = ta.macd(df["close"], fast=6, slow=20, signal=9)
        if macd_620 is not None:
            df = df.join(macd_620)

        bbands = ta.bbands(df["close"], length=20, std=2)
        if bbands is not None:
            df = df.join(bbands)

        df["sma_5"] = ta.sma(df["close"], length=5)
        df["sma_8"] = ta.sma(df["close"], length=8)
        df["sma_13"] = ta.sma(df["close"], length=13)
        df["sma_20"] = ta.sma(df["close"], length=20)

        df["ema_5"] = ta.ema(df["close"], length=5)
        df["ema_9"] = ta.ema(df["close"], length=9)
        df["ema_21"] = ta.ema(df["close"], length=21)

        vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        if vwap is not None:
            df["vwap"] = vwap

        return df
