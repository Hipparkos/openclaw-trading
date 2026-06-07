import contextlib
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from execution.order_manager import OrderManager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import yaml
from dotenv import load_dotenv

from data.ibkr_client import IBKRClient
from data.news_client import NewsClient
from discord_bot.controller import OpenClawDiscord
from strategy.indicators import IndicatorCalculator
from strategy.logic import StrategyEngine

class ConfigurationError(Exception):
    pass


app = FastAPI(title="OpenClaw Trading API")


class TradeRequest(BaseModel):
    symbol: str
    action: str
    quantity: float


@app.get("/status")
async def get_status(request: Request) -> Dict[str, Any]:
    # Expose live connection status.
    client = request.app.state.client
    settings = request.app.state.settings
    return {
        "ibkr_connected": client.ib.isConnected(),
        "active_tickers": settings["tickers"],
    }


@app.post("/execute")
async def execute_trade(request: Request, trade_request: TradeRequest) -> Dict[str, Any]:
    # Allow external trade execution.
    action = trade_request.action.upper()
    if action not in {"BUY", "SELL"}:
        raise HTTPException(status_code=400, detail="action must be BUY or SELL")

    order_manager = request.app.state.order_manager
    await order_manager.place_market_order(
        trade_request.symbol,
        action,
        trade_request.quantity,
    )
    return {
        "message": "Trade submitted successfully",
        "symbol": trade_request.symbol,
        "action": action,
        "quantity": trade_request.quantity,
    }


def setup_logging() -> None:
    # Keep runtime logs and errors.
    log_dir = Path(__file__).resolve().parents[1] / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            RotatingFileHandler(
                log_dir / "openclaw.log", maxBytes=5 * 1024 * 1024, backupCount=3
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("ib_insync").setLevel(logging.WARNING)


def load_settings(path: Path) -> Dict[str, Any]:
    # Load validated runtime configuration.
    if not path.exists():
        raise ConfigurationError(f"Configuration file missing at: {path}")

    with path.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file) or {}

    if "tickers" not in settings or not settings["tickers"]:
        raise ConfigurationError("Missing or empty 'tickers' list in settings.yaml")

    return settings


async def main() -> None:
    # Coordinate IBKR, API, shutdown.
    setup_logging()
    load_dotenv()
    logger = logging.getLogger("Main")

    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    
    try:
        settings = load_settings(settings_path)
    except ConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    client = IBKRClient(settings)
    order_manager = OrderManager(client)
    news_client = NewsClient()
    discord_ui = OpenClawDiscord(order_manager)
    discord_token = os.getenv("DISCORD_TOKEN")
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    server: Optional[uvicorn.Server] = None
    server_task: Optional[asyncio.Task[Any]] = None
    last_buy_time: Dict[str, datetime] = {}
    last_sell_time: Dict[str, datetime] = {}
    open_trade_memory: Dict[str, Dict[str, Any]] = {}

    app.state.client = client
    app.state.settings = settings
    app.state.order_manager = order_manager

    def graceful_shutdown() -> None:
        # Stop tasks without dropping orders.
        logger.info("Shutdown signal received. Initiating graceful exit...")
        shutdown_event.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, graceful_shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda sig, frame: loop.call_soon_threadsafe(graceful_shutdown))
        signal.signal(signal.SIGTERM, lambda sig, frame: loop.call_soon_threadsafe(graceful_shutdown))

    try:
        logger.info("Starting...")
        if discord_token:
            logger.info("Discord token loaded securely. Starting Discord UI...")
            asyncio.create_task(discord_ui.start(discord_token))
        else:
            logger.warning("DISCORD_TOKEN is missing from the environment; Discord UI will not start.")

        async def _stream_news() -> None:
            logger.info("Starting news stream...")

            async def _build_trade_snapshot(symbol: str) -> tuple[str, float]:
                bars_5m = list(client.data_buffer.get(symbol, {}).get("5 mins", []))
                if len(bars_5m) < 2:
                    return "Insufficient 5-minute market data yet.", 0.0

                indicator_calculator = IndicatorCalculator()
                df = await asyncio.to_thread(indicator_calculator.calculate_all, bars_5m)
                if df.empty:
                    return "Technical dataframe is empty.", 0.0

                strategy_engine = StrategyEngine()
                technical_context = await asyncio.to_thread(strategy_engine.evaluate_signals, df)

                current_price = 0.0
                latest_close = df.iloc[-1].get("close")
                if latest_close is not None and latest_close == latest_close:
                    try:
                        current_price = float(latest_close)
                    except (TypeError, ValueError):
                        current_price = 0.0

                return technical_context, current_price

            def _trade_outcome(entry_price: float, exit_price: float) -> str:
                if entry_price <= 0.0 or exit_price <= 0.0:
                    return "UNKNOWN"

                pct_change = ((exit_price - entry_price) / entry_price) * 100.0
                outcome_label = "Win" if pct_change >= 0.0 else "Loss"
                return f"{pct_change:+.2f}% ({outcome_label})"

            async def _route_trade(symbol: str, signal_direction: str, technical_context: str, current_price: float) -> None:
                now = datetime.now(timezone.utc)
                holding_quantity = order_manager.get_position(symbol)
                is_holding = holding_quantity != 0.0

                if signal_direction == "BULLISH" and not is_holding:
                    last_buy = last_buy_time.get(symbol)
                    last_sell = last_sell_time.get(symbol)

                    if last_buy and now - last_buy < timedelta(minutes=15):
                        logger.info("Skipping BUY for %s due to buy cooldown.", symbol)
                        return

                    if last_sell and now - last_sell < timedelta(minutes=15):
                        logger.info("Skipping BUY for %s due to recent sell cooldown.", symbol)
                        return

                    account_equity = order_manager.get_account_equity()
                    target_capital = account_equity * 0.015 if account_equity > 0.0 else 0.0
                    calculated_shares = (target_capital / current_price) if current_price > 0.0 else 0.0
                    logger.info(
                        "Sizing check | %s | equity=%.2f | target_capital=%.2f | current_price=%.2f | calculated_shares=%.2f | executed_quantity=10",
                        symbol,
                        account_equity,
                        target_capital,
                        current_price,
                        calculated_shares,
                    )

                    await order_manager.execute_trade(symbol, "BUY", 10)
                    last_buy_time[symbol] = now
                    open_trade_memory[symbol] = {
                        "technical_context": technical_context,
                        "prediction": signal_direction,
                        "entry_price": current_price,
                        "entry_time": now,
                        "quantity": 10,
                    }
                    logger.info("BUY executed for %s with fixed quantity 10.", symbol)
                    return

                if signal_direction == "BEARISH" and is_holding:
                    last_buy = last_buy_time.get(symbol)
                    if last_buy and now - last_buy < timedelta(minutes=5):
                        logger.info("Skipping SELL for %s because the minimum hold time has not elapsed.", symbol)
                        return

                    await order_manager.execute_trade(symbol, "SELL", 10)
                    last_sell_time[symbol] = now

                    trade_memory = open_trade_memory.pop(symbol, None)
                    if trade_memory:
                        outcome = _trade_outcome(float(trade_memory.get("entry_price", 0.0)), current_price)
                        news_client.record_trade_memory(
                            symbol=symbol,
                            technical_context=str(trade_memory.get("technical_context", technical_context)),
                            prediction=str(trade_memory.get("prediction", signal_direction)),
                            outcome=outcome,
                        )

                    logger.info("SELL executed for %s with fixed quantity 10.", symbol)
                    return

                logger.info(
                    "No autonomous trade executed for %s | holding=%s | signal=%s",
                    symbol,
                    is_holding,
                    signal_direction,
                )
            
            while not shutdown_event.is_set():
                for symbol in settings["tickers"]:
                    technical_context, current_price = await _build_trade_snapshot(symbol)
                    news_items = await news_client.fetch_latest_news(symbol, technical_context=technical_context)
                    
                    for news_item in news_items:
                        logger.info(
                            "NEWS ALERT [%s] | AI Sentiment: %.2f | %s",
                            news_item.symbol,
                            news_item.sentiment_score,
                            news_item.headline,
                        )

                        sentiment_score = news_item.sentiment_score
                        if sentiment_score is None or sentiment_score != sentiment_score or sentiment_score == 0.0:
                            continue

                        signal_direction = "BULLISH" if sentiment_score > 0 else "BEARISH"

                        await discord_ui.send_trade_signal(
                            symbol=news_item.symbol,
                            market_story=technical_context,
                            llm_prediction="UP" if signal_direction == "BULLISH" else "DOWN",
                        )

                        await _route_trade(
                            symbol=news_item.symbol,
                            signal_direction=signal_direction,
                            technical_context=technical_context,
                            current_price=current_price,
                        )
                        break
                
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                except asyncio.TimeoutError:
                    continue

        asyncio.create_task(_stream_news())
        await client.stream_market_data()

        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        
        await shutdown_event.wait()

    except asyncio.CancelledError:
        logger.info("Async tasks cancelled.")
    except Exception as e:
        logger.error(f"An error occurred in the execution loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up connections...")
        if server is not None:
            server.should_exit = True
        if server_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(server_task, timeout=5)
        with contextlib.suppress(Exception):
            await discord_ui.close()
        order_manager.close() # Clean up order event listeners
        client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch interrupts outside asyncio.
        logging.getLogger("Main").info("Stopped manually.")
        sys.exit(0)