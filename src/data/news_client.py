import asyncio
import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import yfinance as yf

from data.data_models import NewsData


class NewsClient:
    def __init__(self) -> None:
        self.logger = logging.getLogger("NewsClient")
        self.memory_db_path = Path(__file__).resolve().parents[2] / "data" / "database" / "trade_memory.db"
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_db_lock = threading.Lock()
        self._initialize_memory_store()

    def _initialize_memory_store(self) -> None:
        with self._memory_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    technical_context TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_memory_symbol_created ON trade_memory(symbol, created_at DESC)"
            )
            # Safe migration: add embedding column to existing databases
            try:
                connection.execute("ALTER TABLE trade_memory ADD COLUMN embedding BLOB")
            except Exception:
                pass  # Column already exists

    def _memory_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.memory_db_path)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 2}

    @staticmethod
    def _truncate_text(text: str, limit: int = 220) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[: limit - 3]}..."

    @staticmethod
    def _cosine_similarity(a: bytes, b: bytes) -> float:
        va = np.frombuffer(a, dtype=np.float32)
        vb = np.frombuffer(b, dtype=np.float32)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / norm) if norm > 0.0 else 0.0

    async def _get_embedding(self, text: str, session: aiohttp.ClientSession) -> bytes | None:
        url = "http://openclaw_ollama:11434/api/embeddings"
        payload = {"model": "nomic-embed-text", "prompt": text}
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                data = await response.json(content_type=None)
            vector = data.get("embedding")
            if not vector:
                return None
            return np.array(vector, dtype=np.float32).tobytes()
        except Exception as exc:
            self.logger.warning("Could not fetch embedding: %s", exc)
            return None

    def record_trade_memory(
        self,
        symbol: str,
        technical_context: str,
        prediction: str,
        outcome: str,
        embedding_bytes: bytes | None = None,
    ) -> None:
        with self._memory_db_lock, self._memory_connection() as connection:
            connection.execute(
                """
                INSERT INTO trade_memory (symbol, technical_context, prediction, outcome, created_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    technical_context,
                    prediction,
                    outcome,
                    datetime.now(timezone.utc).isoformat(),
                    embedding_bytes,
                ),
            )

    async def record_trade_memory_async(
        self,
        symbol: str,
        technical_context: str,
        prediction: str,
        outcome: str,
    ) -> None:
        embedding_bytes: bytes | None = None
        try:
            timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_read=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                embedding_bytes = await self._get_embedding(technical_context, session)
        except Exception as exc:
            self.logger.warning("Skipped embedding during trade recording: %s", exc)
        self.record_trade_memory(symbol, technical_context, prediction, outcome, embedding_bytes)

    def _fetch_live_memory_injection(
        self,
        symbol: str,
        technical_context: str,
        headline: str,
        query_embedding: bytes | None = None,
    ) -> str:
        with self._memory_db_lock, self._memory_connection() as connection:
            rows = connection.execute(
                """
                SELECT technical_context, prediction, outcome, created_at, embedding
                FROM trade_memory
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT 20
                """,
                (symbol,),
            ).fetchall()

        if not rows:
            return ""

        scored_rows: list[tuple[float, tuple[Any, ...]]] = []

        if query_embedding is not None:
            # Semantic scoring: cosine similarity for rows that have embeddings,
            # token-overlap fallback for legacy rows without.
            current_tokens = self._tokenize(f"{technical_context} {headline}")
            for row in rows:
                stored_embedding = row[4]
                if stored_embedding is not None:
                    score = self._cosine_similarity(query_embedding, stored_embedding)
                else:
                    row_tokens = self._tokenize(str(row[0]))
                    overlap = len(current_tokens & row_tokens) if current_tokens and row_tokens else 0
                    score = overlap / max(len(current_tokens), 1) * 0.5  # Normalise to <1 so semantic rows rank higher
                scored_rows.append((score, row))
        else:
            current_tokens = self._tokenize(f"{technical_context} {headline}")
            for row in rows:
                row_tokens = self._tokenize(str(row[0]))
                overlap = len(current_tokens & row_tokens) if current_tokens and row_tokens else 0
                scored_rows.append((float(overlap), row))

        scored_rows.sort(key=lambda item: (item[0], str(item[1][3])), reverse=True)

        selected_rows = [row for score, row in scored_rows if score > 0][:3]
        if not selected_rows:
            selected_rows = [row for _, row in scored_rows[:3]]

        if not selected_rows:
            return ""

        memory_lines = ["LIVE MEMORY INJECTION:"]
        for technical_context_row, prediction, outcome, created_at, _embedding in selected_rows:
            memory_lines.append(
                "- Recent similar setup on {symbol} at {created_at}: predicted {prediction}, outcome {outcome}. Context: {context}".format(
                    symbol=symbol,
                    created_at=created_at,
                    prediction=prediction,
                    outcome=outcome,
                    context=self._truncate_text(str(technical_context_row)),
                )
            )

        return "\n".join(memory_lines)

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
        self.logger.info("openclaw: evaluating %s...", ticker)

        query_text = f"{technical_context} {headline}"
        query_embedding = await self._get_embedding(query_text, session)

        memory_injection = await asyncio.to_thread(
            self._fetch_live_memory_injection,
            ticker,
            technical_context,
            headline,
            query_embedding,
        )

        system_prompt = (
            "You are an elite quantitative trading intelligence engine. "
            "Your sole function is to synthesize technical market structures and fundamental catalysts (news) "
            "to predict immediate directional price momentum.\n\n"
            "### ANALYTICAL FRAMEWORK\n"
            "1. CONTEXT: Evaluate the 'Technical Setup' (moving averages, momentum, Bollinger Bands, VWAP, "
            "and market regime). REGIME RULES: RANGING (ADX<20) = noisy, prefer NEUTRAL unless setup is very clean; "
            "TRENDING (ADX≥25) = follow the trend, momentum signals carry high weight. "
            "Low volume (<0.7× avg) weakens any signal.\n"
            "2. CATALYST: Evaluate the 'Headline' for actionable institutional impact (earnings, macro shifts, supply chain).\n"
            "3. ALIGNMENT LOGIC:\n"
            "   - Synergy: rate conviction 0.5 to 1.0 based on how clearly catalyst confirms trend. \n"
            "   - Conflict: rate conviction 0.0 to 0.5 based on how sharply catalyst contradicts trend.\n"
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

        user_prompt = (
            f"Ticker: {ticker}\n"
            f"Technical Setup: {technical_context or 'Not provided'}\n"
            f"Headline Catalyst: {headline}\n\n"
            "Return the JSON evaluation now."
        )

        if memory_injection:
            user_prompt = f"{user_prompt}\n\n{memory_injection}"

        payload = {
            "model": "openclaw",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
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

            message_content = data.get("message", {}).get("content", "")
            raw_response = str(message_content).strip()
            primary = self._parse_openclaw_response(raw_response)
            self.logger.info(
                "openclaw verdict on %s: %s (%.0f%% confidence)",
                ticker, primary["direction"], primary["confidence"] * 100,
            )

            # ── Qwen reviewer: only called on non-NEUTRAL primary signals ──────
            if primary["direction"] == "NEUTRAL":
                return primary

            review_user_prompt = (
                f"Ticker: {ticker}\n"
                f"Technical Setup: {technical_context or 'Not provided'}\n"
                f"Headline Catalyst: {headline}\n"
                f"Primary assessment: {primary['direction']} at {primary['confidence']:.0%} confidence.\n\n"
                "Review this setup independently and return your final evaluation now."
            )
            review_payload = {
                "model": "qwen_reviewer",
                "messages": [
                    {"role": "system", "content": payload["messages"][0]["content"]},
                    {"role": "user",   "content": review_user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            }
            async with session.post(url, json=review_payload) as review_response:
                review_response.raise_for_status()
                review_data = await review_response.json(content_type=None)

            review_raw = str(review_data.get("message", {}).get("content", "")).strip()
            final = self._parse_openclaw_response(review_raw)
            self.logger.info(
                "qwen reviewer on %s: %s (%.0f%% confidence) — final decision",
                ticker, final["direction"], final["confidence"] * 100,
            )
            return final

        except asyncio.TimeoutError:
            self.logger.error("openclaw timed out on %s — skipping.", ticker)
        except Exception as exc:
            self.logger.error("Could not reach openclaw for %s: %s", ticker, exc)

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
            self.logger.warning(f"Could not fetch Yahoo news for {symbol}: {e}")
            return []

        normalized_news: list[NewsData] = []
        if not news_items:
            return normalized_news

        self.logger.info(f"{symbol}: {len(news_items)} Yahoo headlines found — evaluating up to 3 newest.")

        # 180s total: up to 3 articles × 2 LLM calls each × ~25s per call on CPU
        timeout = aiohttp.ClientTimeout(total=180, connect=10, sock_read=90)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            valid_items = []
            for item in news_items[:3]:
                headline = self._first_nonempty_text(
                    item, ("title", "headline", "summary", "description", "content")
                )
                if not headline:
                    self.logger.warning("Skipped a Yahoo item with no headline (id=%s).", item.get("id", "unknown"))
                    continue
                valid_items.append((item, headline))

            if not valid_items:
                return normalized_news

            # Sequential — CPU-only Ollama processes one request at a time anyway;
            # concurrent gather just causes all but the first to time out while queued.
            predictions = []
            for item, headline in valid_items:
                prediction = await self.get_openclaw_prediction(
                    session=session,
                    ticker=symbol,
                    technical_context=technical_context,
                    headline=headline,
                )
                predictions.append(prediction)

            for (item, headline), prediction in zip(valid_items, predictions):
                url = self._extract_url(item)
                sentiment_score = self._prediction_to_score(prediction)

                self.logger.info(
                    "News verdict: %s → %s (confidence %.2f, sentiment %.2f) | %s",
                    headline,
                    prediction["direction"],
                    prediction["confidence"],
                    sentiment_score,
                    url,
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
