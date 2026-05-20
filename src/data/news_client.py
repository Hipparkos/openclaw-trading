import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import aiohttp
from dotenv import load_dotenv

from data.data_models import NewsData


class MarketauxClient:
    def __init__(self) -> None:
        load_dotenv()
        self.logger = logging.getLogger("MarketauxClient")
        self.api_token = os.getenv("MARKETAUX_API_TOKEN")
        self.base_url = "https://api.marketaux.com/v1/news/all"

    async def fetch_latest_news(self, symbol: str) -> list[NewsData]:
        if not self.api_token:
            self.logger.warning("MARKETAUX_API_TOKEN is not set.")
            return []

        params = {
            "symbols": symbol,
            "filter_entities": "true",
            "language": "en",
            "limit": 10,
            "api_token": self.api_token,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.base_url, params=params) as response:
                    response.raise_for_status()
                    payload: dict[str, Any] = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            self.logger.warning("Marketaux request failed for %s: %s", symbol, exc)
            return []

        articles = payload.get("data", []) or []
        normalized_news: list[NewsData] = []

        for article in articles:
            if not isinstance(article, dict):
                continue

            sentiment_score = article.get("overall_sentiment_score")
            if sentiment_score is None:
                entities = article.get("entities") or []
                if entities and isinstance(entities, list):
                    first_entity = entities[0] if entities else {}
                    if isinstance(first_entity, dict):
                        sentiment_score = first_entity.get("sentiment_score")

            published_at = article.get("published_at") or datetime.utcnow().isoformat()
            normalized_news.append(
                NewsData(
                    symbol=symbol,
                    timestamp=published_at,
                    headline=str(article.get("title", "")),
                    sentiment_score=sentiment_score,
                    url=str(article.get("url", "")),
                )
            )

        return normalized_news