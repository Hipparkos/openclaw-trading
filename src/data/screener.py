import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import yahoo_fin.stock_info as si
import yfinance as yf

_ET = ZoneInfo("America/New_York")


class VolumeGainerScreener:
    def __init__(self):
        self.logger = logging.getLogger("Screener")
        self.cache_path = Path(__file__).resolve().parents[2] / "data" / "screener_cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    async def screen_volume_gainers(self) -> List[str]:
        self.logger.info("Starting volume gainer screener...")

        # Only screen during market hours (avoid API errors when market is closed)
        if not self._is_market_open():
            self.logger.info("Market is currently closed. Skipping live screener.")
            return []

        try:
            self.logger.debug("Calling si.get_day_most_active()...")
            active_df = await asyncio.to_thread(si.get_day_most_active)
            self.logger.debug(f"Raw response type: {type(active_df)}, empty: {active_df.empty if hasattr(active_df, 'empty') else 'N/A'}")

            if active_df is None or (hasattr(active_df, 'empty') and active_df.empty):
                self.logger.warning("Yahoo Finance returned empty data — market may be closed or data unavailable")
                return []

            if not hasattr(active_df, 'columns'):
                self.logger.error(f"Response is not a DataFrame: {type(active_df)}")
                return []

            if "Symbol" not in active_df.columns:
                self.logger.warning(f"Expected 'Symbol' column not found. Available columns: {list(active_df.columns)}")
                return []

            symbols = active_df["Symbol"].head(20).tolist()
            self.logger.info(f"Found {len(symbols)} most active stocks from Yahoo Finance: {symbols}")
        except KeyError as e:
            self.logger.error(f"KeyError in screener (missing data field): {e}. This typically means market is closed or API data is incomplete.")
            return []
        except Exception as e:
            self.logger.error(f"Failed to fetch most active stocks: {type(e).__name__}: {str(e)}", exc_info=True)
            return []

        # Validate: price range, volume, liquidity
        validated = []
        for symbol in symbols:
            if await self._is_daytrading_suitable(symbol):
                validated.append(symbol)
                if len(validated) >= 10:
                    break

        self.logger.info(f"Validated {len(validated)} stocks for day trading: {validated}")

        cache_data = {
            "date": datetime.now().isoformat(),
            "symbols": validated,
        }
        with open(self.cache_path, "w") as f:
            json.dump(cache_data, f)

        return validated

    async def _is_daytrading_suitable(self, symbol: str) -> bool:
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            info = await asyncio.to_thread(self._get_ticker_info, ticker)

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            volume = info.get("volume") or info.get("regularMarketVolume")
            float_shares = info.get("floatShares")
            market_cap = info.get("marketCap")

            if not all([price, volume, float_shares, market_cap]):
                self.logger.debug(
                    f"Missing data for {symbol}: price={price}, vol={volume}, "
                    f"float={float_shares}, mcap={market_cap}"
                )
                return False

            price = float(price)
            volume = float(volume)
            float_shares = float(float_shares)
            market_cap = float(market_cap)

            if not (5 <= price <= 500):
                self.logger.debug(f"{symbol}: price ${price:.2f} out of range")
                return False
            if volume < 1_000_000:
                self.logger.debug(f"{symbol}: volume {volume:,} too low")
                return False
            if float_shares < 10_000_000:
                self.logger.debug(f"{symbol}: float {float_shares:,} too low")
                return False
            if market_cap < 300_000_000:
                self.logger.debug(f"{symbol}: market cap ${market_cap:,} too low")
                return False

            self.logger.info(
                f"✓ {symbol} | Price ${price:.2f} | Volume {volume:,} | "
                f"Float {float_shares:,} | MCap ${market_cap:,.0f}"
            )
            return True

        except Exception as e:
            self.logger.debug(f"Validation failed for {symbol}: {e}")
            return False

    @staticmethod
    def _get_ticker_info(ticker) -> dict:
        try:
            return ticker.info
        except Exception:
            return {}

    @staticmethod
    def _is_market_open() -> bool:
        """Check if US stock market is currently open (Mon-Fri 9:30-16:00 ET)."""
        now = datetime.now(_ET)
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        opens = now.replace(hour=9, minute=30, second=0, microsecond=0)
        closes = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return opens <= now < closes

    async def load_cached_symbols(self) -> List[str]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                cache_date = data.get("date", "").split("T")[0]
                today = datetime.now().isoformat().split("T")[0]

                if cache_date == today:
                    symbols = data.get("symbols", [])
                    self.logger.info(f"Loaded cached symbols (date={cache_date}): {symbols}")
                    return symbols
                else:
                    self.logger.info(f"Cache stale (date={cache_date}, today={today}), re-screening...")
            except Exception as e:
                self.logger.warning(f"Could not load cache: {e}")

        return await self.screen_volume_gainers()
