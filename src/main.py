import asyncio
import logging
import signal
import sys
from logging.handlers import RotatingFileHandler
from execution.order_manager import OrderManager
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from data.ibkr_client import IBKRClient
from data.news_client import MarketauxClient


class ConfigurationError(Exception):
    pass


def setup_logging() -> None:
    # Setup logging - configure handlers
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
    # Load settings - read YAML config
    if not path.exists():
        raise ConfigurationError(f"Configuration file missing at: {path}")

    with path.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file) or {}

    if "tickers" not in settings or not settings["tickers"]:
        raise ConfigurationError("Missing or empty 'tickers' list in settings.yaml")

    return settings


async def main() -> None:
    # Main entry - orchestrate client and loop
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
    news_client = MarketauxClient()
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def graceful_shutdown() -> None:
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
        async def _stream_news() -> None:
            while not shutdown_event.is_set():
                for symbol in settings["tickers"]:
                    news_items = await news_client.fetch_latest_news(symbol)
                    for news_item in news_items:
                        logger.info(
                            "NEWS %s | %s | sentiment=%s | %s | %s",
                            news_item.symbol,
                            news_item.timestamp,
                            news_item.sentiment_score,
                            news_item.headline,
                            news_item.url,
                        )

                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=300)
                except asyncio.TimeoutError:
                    continue

        asyncio.create_task(_stream_news())
        await client.stream_market_data()
        
        logger.info("Testing Order Execution routing...")
        test_ticker = settings["tickers"][0] if settings["tickers"] else "AAPL"
        
        async def _test_trade():
            await asyncio.sleep(5)
            await order_manager.place_market_order(test_ticker, "BUY", 1)
            
        asyncio.create_task(_test_trade())
        
        await shutdown_event.wait()

    except asyncio.CancelledError:
        logger.info("Async tasks cancelled.")
    except Exception as e:
        logger.error(f"An error occurred in the execution loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up connections...")
        order_manager.close() # Clean up order event listeners
        client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Failsafe if the interrupt hits outside the event loop
        logging.getLogger("Main").info("Stopped manually.")
        sys.exit(0)