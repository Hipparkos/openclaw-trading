import contextlib
import asyncio
import logging
import os
import signal
import sys
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
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    server: Optional[uvicorn.Server] = None
    server_task: Optional[asyncio.Task[Any]] = None

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
        asyncio.create_task(discord_ui.start(os.getenv("DISCORD_TOKEN")))

        async def _stream_news() -> None:
            logger.info("Starting news stream...")

            async def _build_technical_context(symbol: str) -> str:
                bars_5m = list(client.data_buffer.get(symbol, {}).get("5 mins", []))
                if len(bars_5m) < 2:
                    return "Insufficient 5-minute market data yet."

                indicator_calculator = IndicatorCalculator()
                df = await asyncio.to_thread(indicator_calculator.calculate_all, bars_5m)
                if df.empty:
                    return "Technical dataframe is empty."

                strategy_engine = StrategyEngine()
                signal = await asyncio.to_thread(strategy_engine.evaluate_signals, df)

                latest_row = df.iloc[-1]
                close_value = latest_row.get("close")
                rsi_value = latest_row.get("rsi_14")
                macd_value = latest_row.get("MACD_6_20_9")
                macd_signal_value = latest_row.get("MACDs_6_20_9")

                return (
                    f"5m bars: {len(bars_5m)} | "
                    f"close: {close_value:.2f} | "
                    f"rsi_14: {rsi_value:.2f} | "
                    f"macd_6_20_9: {macd_value:.4f} | "
                    f"macd_signal_6_20_9: {macd_signal_value:.4f} | "
                    f"strategy_signal: {signal.get('signal')} | "
                    f"setup_type: {signal.get('setup_type') or 'None'}"
                )
            
            while not shutdown_event.is_set():
                for symbol in settings["tickers"]:
                    technical_context = await _build_technical_context(symbol)
                    news_items = await news_client.fetch_latest_news(symbol, technical_context=technical_context)
                    
                    for news_item in news_items:
                        logger.info(
                            "NEWS ALERT [%s] | AI Sentiment: %.2f | %s",
                            news_item.symbol,
                            news_item.sentiment_score,
                            news_item.headline,
                        )

                        prediction: Optional[str] = None
                        if news_item.sentiment_score >= 0.5:
                            prediction = "UP"
                        elif news_item.sentiment_score <= -0.5:
                            prediction = "DOWN"

                        if prediction is not None:
                            await discord_ui.send_trade_signal(
                                symbol=news_item.symbol,
                                market_story=technical_context,
                                llm_prediction=prediction,
                            )
                
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                except asyncio.TimeoutError:
                    continue

        asyncio.create_task(_stream_news())
        await client.stream_market_data()
        
        # logger.info("Testing Order Execution routing...")
        # test_ticker = settings["tickers"][0] if settings["tickers"] else "AAPL"
        
        # async def _test_trade():
        #     await asyncio.sleep(5)
        #     await order_manager.place_market_order(test_ticker, "BUY", 1)
            
        # asyncio.create_task(_test_trade())

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