import asyncio
import json
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
    def _neutral_openclaw_prediction() -> dict[str, Any]:
        return {
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "raw_response": "",
        }

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0

        if confidence < 0.0:
            return 0.0
        if confidence > 1.0:
            return 1.0
        return confidence

    @staticmethod
    def _normalize_direction(value: Any) -> str:
        direction = str(value).strip().upper()
        if direction in {"BULLISH", "BEARISH", "NEUTRAL"}:
            return direction
        return "NEUTRAL"

    def _parse_openclaw_response(self, raw_response: str) -> dict[str, Any]:
        parsed = self._neutral_openclaw_prediction()
        parsed["raw_response"] = raw_response

        cleaned_response = raw_response.strip()
        if not cleaned_response:
            return parsed

        json_payload = cleaned_response
        if json_payload.startswith("```"):
            # Refactored string matching to prevent markdown parser breaks
            marker = "```"
            json_payload = re.sub(
                rf"^{marker}(?:json)?\s*|\s*{marker}$", 
                "", 
                json_payload, 
                flags=re.IGNORECASE | re.DOTALL
            ).strip()

        candidate_object = None
        try:
            candidate_object = json.loads(json_payload)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", json_payload, flags=re.DOTALL)
            if match:
                try:
                    candidate_object = json.loads(match.group(0))
                except json.JSONDecodeError:
                    candidate_object = None

        if isinstance(candidate_object, dict):
            parsed["direction"] = self._normalize_direction(candidate_object.get("direction"))
            parsed["confidence"] = self._clamp_confidence(candidate_object.get("confidence"))
            return parsed

        direction_match = re.search(r"\b(BULLISH|BEARISH|NEUTRAL)\b", cleaned_response, flags=re.IGNORECASE)
        if direction_match:
            parsed["direction"] = self._normalize_direction(direction_match.group(1))

        confidence_match = re.search(r"confidence\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", cleaned_response, flags=re.IGNORECASE)
        if confidence_match:
            parsed["confidence"] = self._clamp_confidence(confidence_match.group(1))
            return parsed

        numeric_match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", cleaned_response)
        if numeric_match and parsed["direction"] != "NEUTRAL":
            parsed["confidence"] = self._clamp_confidence(numeric_match.group(1))

        return parsed

    @staticmethod
    def _prediction_to_score(prediction: dict[str, Any]) -> float:
        direction = str(prediction.get("direction", "NEUTRAL")).upper()
        confidence = NewsClient._clamp_confidence(prediction.get("confidence", 0.0))

        if direction == "BULLISH":
            return confidence
        if direction == "BEARISH":
            return -confidence
        return 0.0

    async def get_openclaw_prediction(
        self,
        session: aiohttp.ClientSession,
        ticker: str,
        technical_context: str,
        headline: str,
    ) -> dict[str, Any]:
        url = "http://openclaw_ollama:11434/api/chat"
        self.logger.info("Requesting OpenClaw pattern evaluation for %s", ticker)

        system_prompt = (
            "You are an elite quantitative trading intelligence engine. "
            "Your sole function is to synthesize technical market structures and fundamental catalysts (news) "
            "to predict immediate directional price momentum.\n\n"
            "### ANALYTICAL FRAMEWORK\n"
            "1. CONTEXT: Evaluate the 'Technical Setup' (moving averages, volume, trend structure).\n"
            "2. CATALYST: Evaluate the 'Headline' for actionable institutional impact (earnings, macro shifts, supply chain).\n"
            "3. ALIGNMENT LOGIC:\n"
            "   - Synergy: If the catalyst confirms the technical trend -> High Confidence (0.7 to 1.0).\n"
            "   - Conflict: If the catalyst contradicts the technical trend -> Low Confidence (0.1 to 0.4).\n"
            "   - Noise: If the headline is irrelevant, vague, or non-actionable -> NEUTRAL (0.0).\n\n"
            "### STRICT OUTPUT PROTOCOL\n"
            "You are a programmatic endpoint. You must output ONLY raw, unformatted JSON. "
            "Do not use markdown blocks (```json), do not include preambles, and do not explain your reasoning. "
            "Your response must be parseable by Python's json.loads() immediately.\n\n"
            "REQUIRED FORMAT:\n"
            "{\n"
            '  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            '  "confidence": <float>\n'
            "}"
        )

        prompt = (
            f"Ticker: {ticker}\n"
            f"Technical Setup: {technical_context or 'Not provided'}\n"
            f"Headline Catalyst: {headline}\n\n"
            "Return the JSON evaluation now."
        )

        payload = {
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json(content_type=None)

            # Updated parsing path matching standard Ollama /api/chat layout
            message_content = data.get("message", {}).get("content", "")
            raw_response = str(message_content).strip()
            
            prediction = self._parse_openclaw_response(raw_response)
            self.logger.info(
                "OpenClaw prediction | ticker=%s | direction=%s | confidence=%.2f",
                ticker,
                prediction["direction"],
                prediction["confidence"],
            )
            return prediction
        except asyncio.TimeoutError:
            self.logger.error("Timed out waiting for OpenClaw prediction for %s.", ticker)
        except Exception as exc:
            self.logger.error("Failed to reach OpenClaw for %s: %s", ticker, exc)

        return self._neutral_openclaw_prediction()

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
            for target_key in ("canonicalUrl", "clickThroughUrl"):
                target_dict = nested_content.get(target_key, {})
                if isinstance(target_dict, dict):
                    url = target_dict.get("url", "")
                    if isinstance(url, str) and url.strip():
                        return url.strip()

        for target_key in ("canonicalUrl", "clickThroughUrl"):
            target_dict = item.get(target_key, {})
            if isinstance(target_dict, dict):
                url = target_dict.get("url", "")
                if isinstance(url, str) and url.strip():
                    return url.strip()

        return ""

    async def get_ollama_sentiment(self, headline: str, session: aiohttp.ClientSession = None) -> float:
        """Helper updated to optionally receive an external connection session."""
        if session:
            prediction = await self.get_openclaw_prediction(
                session=session, ticker="UNKNOWN", technical_context="", headline=headline
            )
            return self._prediction_to_score(prediction)
            
        timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_read=5)
        async with aiohttp.ClientSession(timeout=timeout) as standalone_session:
            prediction = await self.get_openclaw_prediction(
                session=standalone_session, ticker="UNKNOWN", technical_context="", headline=headline
            )
        return self._prediction_to_score(prediction)

    async def fetch_latest_news(self, symbol: str, technical_context: str = "") -> list[NewsData]:
        try:
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            news_items = await asyncio.to_thread(getattr, ticker, "news")
        except Exception as e:
            self.logger.warning(f"Failed to fetch yfinance news for {symbol}: {e}")
            return []

        normalized_news: list[NewsData] = []
        if not news_items:
            return normalized_news

        self.logger.info(f"Found {len(news_items)} total articles for {symbol} on Yahoo.")

        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Step 1: Prepare tasks to fetch predictions concurrently via asyncio.gather
            tasks = []
            valid_items = []
            
            for item in news_items[:3]:
                headline = self._first_nonempty_text(
                    item, ("title", "headline", "summary", "description", "content")
                )
                if not headline:
                    self.logger.warning("Skipping Yahoo news item without headline (id=%s)", item.get("id", "unknown"))
                    continue
                
                valid_items.append((item, headline))
                tasks.append(
                    self.get_openclaw_prediction(
                        session=session,
                        ticker=symbol,
                        technical_context=technical_context,
                        headline=headline
                    )
                )

            if not tasks:
                return normalized_news

            # Step 2: Fire all requests concurrently to avoid thread lockups
            predictions = await asyncio.gather(*tasks)

            # Step 3: Process the results
            for (item, headline), prediction in zip(valid_items, predictions):
                url = self._extract_url(item)
                sentiment_score = self._prediction_to_score(prediction)

                self.logger.info(
                    "Title: %s | URL: %s | Direction: %s | Confidence: %.2f | Sentiment Score: %.2f",
                    headline,
                    url,
                    prediction["direction"],
                    prediction["confidence"],
                    sentiment_score,
                )

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