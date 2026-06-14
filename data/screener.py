# src/data/screener.py
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import json

class VolumeGainerScreener:
    def __init__(self):
        self.logger = logging.getLogger("Screener")
        self.cache_path = Path(__file__).resolve().parents[2] / "data" / "screener_cache.json"
    
    async def screen_volume_gainers(self) -> list[str]:
        """Scrape Finviz for top volume gainers, validate with yfinance."""
        self.logger.info("Starting volume gainer screener...")
        
        # Scrape Finviz
        symbols = await self._scrape_finviz_gainers()
        self.logger.info(f"Found {len(symbols)} candidates from Finviz")
        
        # Validate: price range, volume, float
        validated = []
        for symbol in symbols[:20]:  # Get top 20, validate down to 10
            if await self._is_daytrading_suitable(symbol):
                validated.append(symbol)
                if len(validated) >= 10:
                    break
        
        self.logger.info(f"Validated {len(validated)} stocks for day trading")
        
        # Cache results
        cache_data = {
            "date": datetime.now().isoformat(),
            "symbols": validated,
        }
        with open(self.cache_path, "w") as f:
            json.dump(cache_data, f)
        
        return validated
    
    async def _scrape_finviz_gainers(self) -> list[str]:
        try:
            # Finviz screener: Price $5-$1000, Volume > 500k, Market Cap over 2 B, sorted by volume desc
            url = (
                "https://finviz.com/screener.ashx?"
                "v=111&"
                "f=sh_price_5to1000,sh_avgvol_o500k&"
                "cap_midover"
                "o=-volume"
            )
            
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find the screener table
            table = soup.find("table", {"class": "screener_table"})
            if not table:
                self.logger.warning("Could not find screener table on Finviz")
                return []
            
            symbols = []
            for row in table.find_all("tr")[1:]:  # Skip header
                cells = row.find_all("td")
                if len(cells) > 0:
                    symbol = cells[0].get_text(strip=True)
                    if symbol and symbol.isalpha():
                        symbols.append(symbol)
            
            return symbols[:20]
        
        except Exception as e:
            self.logger.error(f"Finviz scrape failed: {e}")
            return []
    
    async def _is_daytrading_suitable(self, symbol: str) -> bool:
        """Validate stock is suitable for day trading."""
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            info = await asyncio.to_thread(getattr, ticker, "info")
            
            # Filters for day trading suitability
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            volume = info.get("volume") or info.get("regularMarketVolume")
            float_shares = info.get("floatShares")
            market_cap = info.get("marketCap")
            
            # Must have: price $5-500, volume > 1M, float > 10M, market cap > 300M
            if not all([price, volume, float_shares, market_cap]):
                return False
            
            if not (5 <= float(price) <= 500):
                return False
            if float(volume) < 1_000_000:
                return False
            if float(float_shares) < 10_000_000:
                return False
            if float(market_cap) < 300_000_000:
                return False
            
            self.logger.info(f"✓ {symbol} | Price ${price:.2f} | Volume {volume:,}")
            return True
        
        except Exception as e:
            self.logger.debug(f"Validation failed for {symbol}: {e}")
            return False
    
    async def load_cached_symbols(self) -> list[str]:
        """Load today's cached symbols, or run screener if stale."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                cache_date = data.get("date", "").split("T")[0]
                today = datetime.now().isoformat().split("T")[0]
                
                if cache_date == today:
                    self.logger.info(f"Loaded cached symbols: {data['symbols']}")
                    return data.get("symbols", [])
            except Exception as e:
                self.logger.warning(f"Could not load cache: {e}")
        
        # Cache miss or stale — run screener
        return await self.screen_volume_gainers()