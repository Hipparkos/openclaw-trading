from __future__ import annotations

import pandas as pd
import pandas_ta as ta
import math

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
        df["sma_50"] = ta.sma(df["close"], length=50)

        df["ema_5"] = ta.ema(df["close"], length=5)
        df["ema_9"] = ta.ema(df["close"], length=9)
        df["ema_21"] = ta.ema(df["close"], length=21)

        vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        if vwap is not None:
            df["vwap"] = vwap

        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_result is not None:
            df = df.join(adx_result)

        df["volume_sma_20"] = ta.sma(df["volume"], length=20)

        return df

    def calculate_divergence(self, df: pd.DataFrame, lookback: int = 14) -> dict:
        """
        Detect RSI and MACD divergence over the last `lookback` bars using
        3-bar swing pivots. Returns rsi_divergence, macd_divergence, strength.
        """
        empty = {"rsi_divergence": "NONE", "macd_divergence": "NONE", "strength": 0.0}
        if df is None or len(df) < lookback + 2:
            return empty

        window = df.iloc[-lookback:].reset_index(drop=True)
        price = window["close"].astype(float).values

        # Find 3-bar swing lows and highs
        lows: list[int] = []
        highs: list[int] = []
        for i in range(1, len(price) - 1):
            if price[i] < price[i - 1] and price[i] < price[i + 1]:
                lows.append(i)
            if price[i] > price[i - 1] and price[i] > price[i + 1]:
                highs.append(i)

        rsi_arr = window["rsi_14"].astype(float).values if "rsi_14" in window.columns else None
        macd_col = next((c for c in window.columns if c.startswith("MACD_6_")), None)
        macd_arr = window[macd_col].astype(float).values if macd_col else None

        rsi_div = macd_div = "NONE"

        def _valid(arr: list | None, i: int) -> bool:
            return arr is not None and not math.isnan(arr[i])

        # Bullish: price lower low, oscillator higher low
        if len(lows) >= 2:
            i1, i2 = lows[-2], lows[-1]
            if price[i2] < price[i1]:
                if _valid(rsi_arr, i1) and _valid(rsi_arr, i2) and rsi_arr[i2] > rsi_arr[i1]:
                    rsi_div = "BULLISH"
                if _valid(macd_arr, i1) and _valid(macd_arr, i2) and macd_arr[i2] > macd_arr[i1]:
                    macd_div = "BULLISH"

        # Bearish: price higher high, oscillator lower high
        if len(highs) >= 2:
            i1, i2 = highs[-2], highs[-1]
            if price[i2] > price[i1]:
                if _valid(rsi_arr, i1) and _valid(rsi_arr, i2) and rsi_arr[i2] < rsi_arr[i1]:
                    rsi_div = "BEARISH"
                if _valid(macd_arr, i1) and _valid(macd_arr, i2) and macd_arr[i2] < macd_arr[i1]:
                    macd_div = "BEARISH"

        agreement = rsi_div == macd_div and rsi_div != "NONE"
        strength = 0.85 if agreement else (0.55 if rsi_div != "NONE" or macd_div != "NONE" else 0.0)
        return {"rsi_divergence": rsi_div, "macd_divergence": macd_div, "strength": strength}
