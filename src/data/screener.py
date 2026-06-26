import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import requests
import yfinance as yf

_ET = ZoneInfo("America/New_York")


class VolumeGainerScreener:
    def __init__(self):
        self.logger = logging.getLogger("Screener")
        self.cache_path = Path(__file__).resolve().parents[2] / "data" / "screener_cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    async def screen_volume_gainers(self) -> List[str]:
        self.logger.info("Running stock screener...")

        # Only screen during market hours (avoid API errors when market is closed)
        if not self._is_market_open():
            self.logger.info("Market closed — skipping screener.")
            return []

        try:
            symbols = await asyncio.to_thread(self._fetch_most_active)
            if not symbols:
                self.logger.warning("Yahoo screener returned no symbols.")
                return []
            self.logger.info(f"Yahoo most-active list ({len(symbols)}): {symbols}")
        except Exception as e:
            self.logger.error(f"Could not fetch most-active stocks: {type(e).__name__}: {e}", exc_info=True)
            return []

        # Validate: price range, volume, liquidity
        validated = []
        for symbol in symbols:
            if await self._is_daytrading_suitable(symbol):
                validated.append(symbol)
                if len(validated) >= 10:
                    break

        self.logger.info(f"Selected {len(validated)} tradable tickers: {validated}")

        cache_data = {
            "date": datetime.now().isoformat(),
            "symbols": validated,
        }
        with open(self.cache_path, "w") as f:
            json.dump(cache_data, f)

        return validated

    def _fetch_most_active(self) -> List[str]:
        """Call Yahoo Finance's screener JSON API directly — no HTML parsing, no fragile library."""
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {
            "formatted": "false",
            "lang": "en-US",
            "region": "US",
            "scrIds": "most_actives",
            "count": 25,
        }
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        quotes = data["finance"]["result"][0]["quotes"]
        return [q["symbol"] for q in quotes if q.get("symbol")]

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
                f"{symbol} passed screen | price ${price:.2f} | vol {volume:,.0f} | "
                f"float {float_shares:,.0f} | mcap ${market_cap:,.0f}"
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
                    self.logger.info(f"Using cached screen from {cache_date}: {symbols}")
                    return symbols
                else:
                    self.logger.info(f"Screen cache is stale ({cache_date}) — re-screening for {today}.")
            except Exception as e:
                self.logger.warning(f"Could not load screen cache: {e}")

        return await self.screen_volume_gainers()
