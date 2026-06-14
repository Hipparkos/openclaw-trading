import asyncio
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import List

import requests
from bs4 import BeautifulSoup
import yfinance as yf


class VolumeGainerScreener:
    def __init__(self):
        self.logger = logging.getLogger("Screener")
        self.cache_path = Path(__file__).resolve().parents[2] / "data" / "screener_cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    async def screen_volume_gainers(self) -> List[str]:
        self.logger.info("Starting volume gainer screener...")

        symbols = await self._scrape_finviz_gainers()
        self.logger.info(f"Found {len(symbols)} candidates from Finviz")

        validated = []
        for symbol in symbols[:20]:
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

    async def _scrape_finviz_gainers(self) -> List[str]:
        try:
            url = (
                "https://finviz.com/screener.ashx?"
                "v=111&"
                "f=sh_price_5to1000,sh_avgvol_o500k&"
                "o=-volume"
            )

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            table = soup.find("table", {"class": "screener_table"})
            if not table:
                self.logger.warning("Could not find screener table on Finviz")
                return []

            symbols = []
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if len(cells) > 0:
                    symbol = cells[0].get_text(strip=True)
                    if symbol and symbol.isalpha() and len(symbol) <= 5:
                        symbols.append(symbol)

            self.logger.info(f"Scraped {len(symbols)} symbols from Finviz")
            return symbols[:20]

        except Exception as e:
            self.logger.error(f"Finviz scrape failed: {e}")
            return []

    async def _is_daytrading_suitable(self, symbol: str) -> bool:
        """Validate stock is suitable for day trading."""
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            info = await asyncio.to_thread(self._get_ticker_info, ticker)

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            volume = info.get("volume") or info.get("regularMarketVolume")
            float_shares = info.get("floatShares")
            market_cap = info.get("marketCap")

            if not all([price, volume, float_shares, market_cap]):
                self.logger.debug(f"Missing data for {symbol}: price={price}, vol={volume}, float={float_shares}, mcap={market_cap}")
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

            self.logger.info(f"✓ {symbol} | Price ${price:.2f} | Volume {volume:,} | Float {float_shares:,} | MCap ${market_cap:,.0f}")
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
