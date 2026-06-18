"""
Iterates all bars for each ticker, calls _llm_evaluate() only on bars that pass
the deterministic pre-filter, and saves results to cache/llm_cache.json.
Run this ONCE (or to top-up the cache) before training.

Usage:
    cd /app
    python -m ppo.cache_generator --tickers NVDA AAPL MSFT --duration "3 M"

The cache file is automatically used by BacktestEngine._replay_symbol() to
skip live Ollama calls when running the backtest or training the PPO agent.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ibkr_client import IBKRClient
from backtests.engine import BacktestEngine, _to_utc, _hourly_bars_up_to

CACHE_PATH = Path(__file__).parent.parent.parent / "cache" / "llm_cache.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("CacheGenerator")


async def generate(tickers: list[str], duration: str) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, list] = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            existing = json.load(f)
        logger.info("Loaded %d existing cache entries from %s", len(existing), CACHE_PATH)

    client = IBKRClient()
    await client.connect()
    engine = BacktestEngine(client=client)

    new_entries = 0
    for symbol in tickers:
        logger.info("── %s ─────────────────────────", symbol)
        bars = await engine._fetch_bars(symbol, duration)
        bars_5m = bars.get("5 mins", [])
        bars_1h = bars.get("1 hour", [])

        if len(bars_5m) < engine.WARMUP_BARS + 2:
            logger.warning("%s: only %d bars — skipping", symbol, len(bars_5m))
            await asyncio.sleep(engine.IBKR_SYMBOL_PAUSE)
            continue

        import pandas as pd
        full_df_5m = engine._calc.calculate_all(bars_5m)
        full_df_1h = engine._calc.calculate_all(bars_1h) if bars_1h else pd.DataFrame()

        if full_df_5m is None or full_df_5m.empty:
            logger.warning("%s: indicator calc returned empty df — skipping", symbol)
            await asyncio.sleep(engine.IBKR_SYMBOL_PAUSE)
            continue

        n = len(bars_5m)
        candidates = skipped_cache = 0
        for i in range(engine.WARMUP_BARS, n - 1):
            current_time = _to_utc(bars_5m[i].timestamp)
            window_5m = bars_5m[max(0, i - engine.MAX_LOOKBACK_5M + 1): i + 1]
            window_1h = _hourly_bars_up_to(bars_1h, current_time)

            start = max(0, i - engine.MAX_LOOKBACK_5M + 1)
            cached_df = full_df_5m.iloc[start: i + 1]
            df_1h_slice = full_df_1h.iloc[:len(window_1h)] if not full_df_1h.empty else pd.DataFrame()

            sig, _ = engine._compute_signal(
                window_5m, window_1h, cached_df=cached_df, cached_df_1h=df_1h_slice
            )
            if sig == "NEUTRAL":
                continue
            candidates += 1

            tech_ctx = engine._build_technical_context(cached_df, df_1h=df_1h_slice)
            if tech_ctx in existing:
                skipped_cache += 1
                continue

            direction, confidence = await engine._llm_evaluate(symbol, tech_ctx)
            existing[tech_ctx] = [direction, confidence]
            new_entries += 1

            if new_entries % 25 == 0:
                _save(existing)
                logger.info("  saved checkpoint — total entries: %d", len(existing))

        logger.info(
            "%s done: %d candidate bars, %d LLM calls, %d from cache",
            symbol, candidates, candidates - skipped_cache, skipped_cache,
        )
        await asyncio.sleep(engine.IBKR_SYMBOL_PAUSE)

    _save(existing)
    logger.info(
        "Cache generation complete. Total entries: %d  (new this run: %d)",
        len(existing), new_entries,
    )
    client.disconnect()


def _save(data: dict) -> None:
    tmp = CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(CACHE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM cache for PPO training")
    parser.add_argument("--tickers", nargs="+", required=True, metavar="TICKER")
    parser.add_argument("--duration", default="3 M", help="IBKR duration string (default: '3 M')")
    args = parser.parse_args()
    asyncio.run(generate(args.tickers, args.duration))


if __name__ == "__main__":
    main()
