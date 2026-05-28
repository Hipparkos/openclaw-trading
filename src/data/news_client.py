import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

import aiohttp
import yfinance as yf

from data.data_models import NewsData


class NewsClient:
    def __init__(self) -> None:
        self.logger = logging.getLogger("NewsClient")

    @staticmethod
    def _first_nonempty_text(item: dict[str, Any], keys: tuple[str, ...]) -> str:
        nested_content = item.get("content")
        if isinstance(nested_content, dict):
            for key in keys:
                value = nested_content.get(key, "")
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if value is not None and not isinstance(value, (dict, list)):
                    text = str(value).strip()
                    if text:
                        return text

        for key in keys:
            value = item.get(key, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value is not None and not isinstance(value, (dict, list)):
                text = str(value).strip()
                if text:
                    return text
        return ""

    @staticmethod
    def _extract_url(item: dict[str, Any]) -> str:
        url = item.get("link", "")
        if isinstance(url, str) and url.strip():
            return url.strip()

        nested_content = item.get("content")
        if isinstance(nested_content, dict):
            nested_url = nested_content.get("canonicalUrl", {})
            if isinstance(nested_url, dict):
                url = nested_url.get("url", "")
                if isinstance(url, str) and url.strip():
                    return url.strip()

            nested_click_through = nested_content.get("clickThroughUrl", {})
            if isinstance(nested_click_through, dict):
                url = nested_click_through.get("url", "")
                if isinstance(url, str) and url.strip():
                    return url.strip()

        canonical_url = item.get("canonicalUrl", {})
        if isinstance(canonical_url, dict):
            url = canonical_url.get("url", "")
            if isinstance(url, str) and url.strip():
                return url.strip()

        click_through = item.get("clickThroughUrl", {})
        if isinstance(click_through, dict):
            url = click_through.get("url", "")
            if isinstance(url, str) and url.strip():
                return url.strip()

        return ""

    async def get_ollama_sentiment(self, headline: str) -> float:
        url = "http://openclaw_ollama:11434/api/generate"
        self.logger.info("Requesting Ollama sentiment for headline: %s", headline)
        
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
            timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_read=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json(content_type=None)
                    
                    raw_response = str(data.get("response", "")).strip()
                    
                    match = re.search(r"-?\d+(?:\.\d+)?", raw_response)
                    if match:
                        sentiment = float(match.group(0))
                        self.logger.info("Ollama sentiment score: %s", sentiment)
                        return sentiment

                    self.logger.error("Ollama returned non-numeric text: %s", raw_response)
                    return 0.0 
                        
        except asyncio.TimeoutError:
            self.logger.error("Timed out waiting for Ollama sentiment response.")
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
        
        if not news_items:
            self.logger.warning(f"Yahoo Finance returned ZERO articles for {symbol}.")
            return normalized_news

        self.logger.info(f"Found {len(news_items)} total articles for {symbol} on Yahoo.")

        for item in news_items[:3]:
            url = self._extract_url(item)
            headline = self._first_nonempty_text(
                item,
                ("title", "headline", "summary", "description", "content"),
            )

            if not headline:
                item_id = item.get("id", "unknown")
                self.logger.warning("Skipping Yahoo news item without headline (id=%s)", item_id)
                continue
            
            # Feed the headline to local AI
            self.logger.info(f"Asking Ollama to score headline: '{headline}'")
            sentiment_score = await self.get_ollama_sentiment(headline)

            self.logger.info(
                f"Title: {headline} | URL: {url} | Sentiment: {sentiment_score}"
            )

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