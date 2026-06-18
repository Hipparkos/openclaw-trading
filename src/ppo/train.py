"""
Prerequisites:
    1. cache/llm_cache.json exists (run cache_generator.py first)
    2. pip install stable-baselines3 gymnasium

Usage:
    cd /app
    python -m ppo.train --tickers NVDA AAPL MSFT --duration "3 M" --timesteps 500000

Output:
    models/ppo_policy.zip  — trained policy, loaded automatically by BacktestEngine
    logs/ppo_tensorboard/  — TensorBoard reward curves (optional: tensorboard --logdir logs/ppo_tensorboard)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ibkr_client import IBKRClient
from backtests.engine import BacktestEngine, _to_utc, _hourly_bars_up_to
from ppo.environment import TickerData, build_static_obs

CACHE_PATH = Path(__file__).parent.parent.parent / "cache" / "llm_cache.json"
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "ppo_policy"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("PPOTrainer")


# ── IBKR fetch ─────────────────────────────────────────────────────────────────

async def _fetch_all(tickers: list[str], duration: str) -> tuple[dict, BacktestEngine]:
    client = IBKRClient()
    await client.connect()
    engine = BacktestEngine(client=client)
    all_bars: dict[str, dict] = {}
    for symbol in tickers:
        logger.info("Fetching bars for %s...", symbol)
        all_bars[symbol] = await engine._fetch_bars(symbol, duration)
        await asyncio.sleep(engine.IBKR_SYMBOL_PAUSE)
    client.disconnect()
    return all_bars, engine


# ── Pre-computation ─────────────────────────────────────────────────────────────

def precompute_ticker(
    symbol: str,
    bars_5m: list,
    bars_1h: list,
    engine: BacktestEngine,
    llm_cache: dict[str, list],
) -> TickerData | None:
    """Convert raw bars into pre-computed TickerData for fast Gym env stepping.

    Runs _compute_signal + _build_technical_context for every bar so the Gym env
    never recalculates indicators during training — each step() is just array lookups.
    """
    if len(bars_5m) < engine.WARMUP_BARS + 2:
        logger.warning("%s: only %d 5m bars — skipping", symbol, len(bars_5m))
        return None

    full_df_5m = engine._calc.calculate_all(bars_5m)
    full_df_1h = engine._calc.calculate_all(bars_1h) if bars_1h else pd.DataFrame()

    if full_df_5m is None or full_df_5m.empty:
        logger.warning("%s: indicator calc returned empty — skipping", symbol)
        return None

    n = len(bars_5m)
    prefilter = np.zeros(n, dtype=np.int8)
    llm_conf_arr = np.zeros(n, dtype=np.float32)
    llm_dir_arr = np.zeros(n, dtype=np.float32)
    static_obs = np.zeros((n, 4), dtype=np.float32)

    candidates = 0
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

        prefilter[i] = 1 if sig == "BULLISH" else -1
        candidates += 1

        tech_ctx = engine._build_technical_context(cached_df, df_1h=df_1h_slice)
        cached = llm_cache.get(tech_ctx)
        if cached:
            raw_dir = str(cached[0]).upper()
            llm_dir_arr[i] = 1.0 if raw_dir == "BULLISH" else (-1.0 if raw_dir == "BEARISH" else 0.0)
            llm_conf_arr[i] = float(cached[1])

        # Static observation features (market-structure, no position state)
        row = full_df_5m.iloc[i]
        h_close, h_sma50 = 0.0, 0.0
        if not df_1h_slice.empty:
            h_row = df_1h_slice.iloc[-1]
            try:
                h_close = float(h_row.get("close") or 0.0)
                h_sma50 = float(h_row.get("sma_50") or 0.0)
            except (TypeError, ValueError):
                pass
        static_obs[i] = build_static_obs(row, h_close, h_sma50)

    logger.info(
        "%s: %d total bars, %d candidate bars (%.1f%%)",
        symbol, n, candidates, 100.0 * candidates / n if n else 0,
    )

    return TickerData(
        symbol=symbol,
        n_bars=n,
        full_df_5m=full_df_5m,
        full_df_1h=full_df_1h,
        prefilter=prefilter,
        llm_conf=llm_conf_arr,
        llm_dir=llm_dir_arr,
        static_obs=static_obs,
    )


# ── Training ────────────────────────────────────────────────────────────────────

def train(tickers: list[str], duration: str, timesteps: int) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        logger.error("stable-baselines3 not installed. Run: pip install stable-baselines3 gymnasium")
        sys.exit(1)

    from ppo.environment import OpenClawEnv

    # Load LLM cache
    if not CACHE_PATH.exists():
        logger.error(
            "LLM cache not found at %s.\n"
            "Run first:  python -m ppo.cache_generator --tickers %s",
            CACHE_PATH, " ".join(tickers),
        )
        sys.exit(1)
    with open(CACHE_PATH) as f:
        llm_cache: dict[str, list] = json.load(f)
    logger.info("Loaded %d LLM cache entries", len(llm_cache))

    # Fetch bars from IBKR
    logger.info("Fetching historical bars from IBKR...")
    all_bars, engine = asyncio.run(_fetch_all(tickers, duration))

    # Pre-compute all ticker data
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ticker_data: list[TickerData] = []
    for symbol in tickers:
        bars = all_bars.get(symbol, {})
        td = precompute_ticker(
            symbol=symbol,
            bars_5m=bars.get("5 mins", []),
            bars_1h=bars.get("1 hour", []),
            engine=engine,
            llm_cache=llm_cache,
        )
        if td is not None:
            ticker_data.append(td)

    if not ticker_data:
        logger.error("No valid ticker data available — aborting")
        sys.exit(1)

    logger.info("Pre-computed %d tickers, starting PPO training...", len(ticker_data))

    env = OpenClawEnv(tickers=ticker_data)
    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="logs/ppo_tensorboard",
    )
    model.learn(total_timesteps=timesteps)
    model.save(str(MODEL_PATH))
    logger.info("Model saved to %s.zip", MODEL_PATH)
    logger.info(
        "To use in backtesting, pass ppo_policy_path='%s.zip' to BacktestEngine.",
        MODEL_PATH,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent on OpenClaw backtest env")
    parser.add_argument("--tickers", nargs="+", required=True, metavar="TICKER")
    parser.add_argument("--duration", default="3 M", help="IBKR duration string (default: '3 M')")
    parser.add_argument("--timesteps", type=int, default=500_000, metavar="N")
    args = parser.parse_args()
    train(args.tickers, args.duration, args.timesteps)


if __name__ == "__main__":
    main()
