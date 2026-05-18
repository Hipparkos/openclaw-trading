import asyncio
import logging
from typing import Any, Callable, Dict, List

from ib_insync import IB, Stock


class IBKRClient:
    # Initialize client - setup IB and settings
    def __init__(self, settings: Dict[str, Any]) -> None:
        self.logger = logging.getLogger("IBKRClient")
        
        connection = settings.get("connection", {})
        self.host = connection.get("host", "localhost")
        self.port = connection.get("port", 7497)  # TWS 7497 - Gateway 4002
        self.client_id = connection.get("clientId", 1)
        
        # Modularized exchange and currency
        self.exchange = connection.get("exchange", "SMART")
        self.currency = connection.get("currency", "USD")
        
        self.tickers = settings.get("tickers", [])

        self.ib = IB()
        self.max_retries = 5
        self.retry_delay = 3
        self._subscriptions: List[Any] = []
        
        self.ib.disconnectedEvent += self._on_disconnected

    # Handle disconnect - start reconnect loop
    def _on_disconnected(self) -> None:
        self.logger.warning("IBKR connection lost. Attempting runtime recovery...")
        asyncio.create_task(self._reconnect_loop())

    # Reconnect loop - attempt runtime recovery
    async def _reconnect_loop(self) -> None:
        while not self.ib.isConnected():
            try:
                await asyncio.sleep(self.retry_delay)
                self.logger.info("Attempting to reconnect...")
                await self.connect()
                if self.ib.isConnected():
                    self.logger.info("Runtime recovery successful.")
                    await self.stream_market_data()
                    break
            except Exception as e:
                self.logger.error(f"Runtime reconnection failed: {e}")

    # Connect async - reliable connect with retries
    async def connect(self) -> None:
        attempt = 0
        while True:
            try:
                await self.ib.connectAsync(
                    self.host,
                    self.port,
                    clientId=self.client_id,
                    timeout=5,
                )
                self.logger.info(f"Connected to IBKR at {self.host}:{self.port} with clientId {self.client_id}")
                return
            except Exception as exc:
                attempt += 1
                if attempt >= self.max_retries:
                    self.logger.critical("Exhausted all connection retries.")
                    raise RuntimeError("Unable to connect to IBKR") from exc
                
                self.logger.warning(
                    f"Connection failed ({attempt}/{self.max_retries}): {exc}. "
                    f"Retrying in {self.retry_delay}s"
                )
                await asyncio.sleep(self.retry_delay)

    # Bar update handler - log incoming bars
    def _handle_bar_update(self, symbol: str, bar_size: str) -> Callable[[Any, bool], None]:
        def on_update(bars: Any, has_new_bar: bool) -> None:
            if not has_new_bar or not bars:
                return

            bar = bars[-1]
            self.logger.info(
                f"{symbol} {bar_size} "
                f"{bar.date} O:{bar.open:.2f} H:{bar.high:.2f} "
                f"L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume}"
            )

        return on_update

    # Subscribe ticker - qualify and request bars
    async def _subscribe_to_ticker(self, symbol: str) -> None:
        contract = Stock(symbol, self.exchange, self.currency)
        await self.ib.qualifyContractsAsync(contract)

        bar_tasks = []
        for bar_size in ("1 min", "5 mins", "15 mins"):
            bar_tasks.append(self._request_historical_bars(contract, symbol, bar_size))
            
        await asyncio.gather(*bar_tasks)

    # Request bars - fetch and subscribe bars
    async def _request_historical_bars(self, contract: Stock, symbol: str, bar_size: str) -> None:
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=True,
        )
        bars.updateEvent += self._handle_bar_update(symbol, bar_size)
        self._subscriptions.append(bars)
        self.logger.info(f"Fetched {symbol} {bar_size}")

    # Stream market data - start streams concurrently
    async def stream_market_data(self) -> None:
        if not self.ib.isConnected():
            await self.connect()

        self.logger.info("Initializing concurrent market data streams...")
        
        # Concurrently process all tickers to reduce startup latency
        tasks = [self._subscribe_to_ticker(symbol) for symbol in self.tickers]
        await asyncio.gather(*tasks)

    # Disconnect cleanly - stop IB and handlers
    def disconnect(self) -> None:
        self.ib.disconnectedEvent -= self._on_disconnected
        if self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("Disconnected from IBKR cleanly.")