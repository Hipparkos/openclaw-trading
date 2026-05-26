import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import aiohttp
import yfinance as yf

from data.data_models import NewsData


class NewsClient:
    def __init__(self) -> None:
        self.logger = logging.getLogger("NewsClient")
        self.seen_urls = set()

    async def get_ollama_sentiment(self, headline: str) -> float:
        url = "http://openclaw_ollama:11434/api/generate"
        
        system_prompt = (
            "You are a quantitative financial sentiment analyzer. "
            "Read the provided financial news headline and evaluate its impact on the stock's price. "
            "Score the sentiment on a scale from -1.0 (Extremely Bearish) to 1.0 (Extremely Bullish). "
            "0.0 is completely neutral. "
            "CRITICAL RULE: You must output ONLY the raw float number. No words, no explanations, no formatting."
        )

        payload = {
            "model": "llama3.2",
            "system": system_prompt,
            "prompt": f"Headline: {headline}",
            "stream": False 
        }

        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    raw_response = data.get("response", "").strip()
                    
                    try:
                        return float(raw_response)
                    except ValueError:
                        self.logger.error(f"Ollama returned non-float text: {raw_response}")
                        return 0.0 
                        
        except Exception as e:
            self.logger.error(f"Failed to reach Ollama: {e}")
            return 0.0

    async def fetch_latest_news(self, symbol: str) -> list[NewsData]:
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            news_items = await asyncio.to_thread(getattr, ticker, "news")
        except Exception as e:
            self.logger.warning(f"Failed to fetch yfinance news for {symbol}: {e}")
            return []

        normalized_news: list[NewsData] = []
        if not news_items:
            return normalized_news

        for item in news_items[:3]:
            url = item.get("link", "")
            
            if not url or url in self.seen_urls:
                continue
                
            self.seen_urls.add(url)
            headline = item.get("title", "")
            
            # Feed the headline to local AI
            self.logger.info(f"Asking Ollama to score headline: '{headline}'")
            sentiment_score = await self.get_ollama_sentiment(headline)

            # Convert Yahoo's unix timestamp to standard ISO
            pub_time = item.get("providerPublishTime")
            if pub_time:
                timestamp = datetime.fromtimestamp(pub_time, tz=timezone.utc).isoformat()
            else:
                timestamp = datetime.now(timezone.utc).isoformat()

            normalized_news.append(
                NewsData(
                    symbol=symbol,
                    timestamp=timestamp,
                    headline=headline,
                    sentiment_score=sentiment_score,
                    url=url,
                )
            )

        return normalized_news