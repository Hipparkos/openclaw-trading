import asyncio
import logging
from typing import Any, Callable, Dict, List

from ib_insync import IB, Stock

from data.data_models import BarData
from strategy.indicators import IndicatorCalculator
from strategy.logic import StrategyEngine


class IBKRClient:
    # Initialize client - setup IB and settings
    def __init__(self, settings: Dict[str, Any]) -> None:
        self.logger = logging.getLogger("IBKRClient")
        
        connection = settings.get("connection", {})
        # Allow host/port to be configured via settings (keeps previous defaults)
        self.host = connection.get("host", "ib_gateway")
        self.port = connection.get("port", 4002)
        self.client_id = connection.get("clientId", 10)
        
        # Modularized exchange and currency
        self.exchange = connection.get("exchange", "SMART")
        self.currency = connection.get("currency", "USD")
        
        self.tickers = settings.get("tickers", [])
        self.data_buffer: Dict[str, Dict[str, List[BarData]]] = {}

        self.ib = IB()
        self.max_retries = 30
        self.retry_delay = 7
        self._subscriptions: List[Any] = []
        
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error
        self._subscriptions: List[Any] = []
        self._is_reconnecting = False

    # Handle IBKR errors
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Any) -> None:
        if errorCode == 162:
            self.logger.warning(
                "IBKR historical data pacing warning (162) for reqId %s. "
                "Close other active IBKR/TWS sessions or reduce request frequency.",
                reqId,
            )
            return

        self.logger.error(
            "IBKR error %s for reqId %s: %s",
            errorCode,
            reqId,
            errorString,
        )

    # Handle disconnect - start reconnect loop
    def _on_disconnected(self) -> None:
        self.logger.warning("IBKR connection lost.")
        if not self._is_reconnecting:
            self.logger.info("Starting reconnection loop...")
            self._is_reconnecting = True
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
                    self._is_reconnecting = False  # Reset flag
                    await self.stream_market_data()
                    break
            except Exception as e:
                self.logger.error(f"Runtime reconnection failed: {e}")

    # Connect async - reliable connect with retries
    async def connect(self) -> None:
        port = 4004

        try:
            self.logger.info(f"Attempting API handshake at {self.host}:{port} with Client ID {self.client_id}...")

            await self.ib.connectAsync(
                self.host,
                port,
                clientId=self.client_id,
                timeout=15,
            )
            self.logger.info(f"SUCCESS: Connected to IBKR at {self.host}:{port}!")
            self.port = port
            self.ib.reqAccountUpdates(True, "")
        except Exception as exc:
            self.logger.error(f"Handshake failed on port {port}: {exc}")
            raise RuntimeError("Unable to connect to IBKR")

    # Bar update handler - log incoming bars
    def _handle_bar_update(self, symbol: str, bar_size: str) -> Callable[[Any, bool], None]:
        def on_update(bars: Any, has_new_bar: bool) -> None:
            if not has_new_bar or not bars:
                return

            bar = bars[-1]
            normalized_bar = BarData(
                symbol=symbol,
                timestamp=bar.date,
                timeframe=bar_size,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
            )
            self.data_buffer.setdefault(symbol, {}).setdefault(bar_size, []).append(normalized_bar)
            
            self.logger.info(
                "%s [%s] | O: %s | H: %s | L: %s | C: %s | V: %s", 
                symbol, 
                bar_size, 
                normalized_bar.open, 
                normalized_bar.high, 
                normalized_bar.low, 
                normalized_bar.close, 
                normalized_bar.volume
            )

            if bar_size == "5 mins" and has_new_bar:
                async def _evaluate_strategy() -> None:
                    try:
                        bars_5m = list(self.data_buffer.get(symbol, {}).get("5 mins", []))
                        if len(bars_5m) < 2:
                            return

                        indicator_calculator = IndicatorCalculator()
                        df = indicator_calculator.calculate_all(bars_5m)

                        strategy_engine = StrategyEngine()
                        signal = strategy_engine.evaluate_signals(df)
                        self.logger.info("Strategy signal: %s", signal)
                    except Exception:
                        self.logger.exception("Strategy evaluation failed for %s", symbol)

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_evaluate_strategy())
                except RuntimeError:
                    try:
                        loop = asyncio.get_event_loop()
                        loop.call_soon_threadsafe(asyncio.create_task, _evaluate_strategy())
                    except Exception:
                        self.logger.exception("Failed scheduling strategy task for %s", symbol)

        return on_update

    # Subscribe ticker - qualify and request bars
    async def _subscribe_to_ticker(self, symbol: str) -> None:
        contract = Stock(symbol, self.exchange, self.currency)
        await self.ib.qualifyContractsAsync(contract)
        self.data_buffer.setdefault(symbol, {})

        bar_tasks = []
        for bar_size in ("1 min", "5 mins", "15 mins"):
            bar_tasks.append(self._request_historical_bars(contract, symbol, bar_size))
            
        await asyncio.gather(*bar_tasks)

    # Request bars - fetch and subscribe bars
    async def _request_historical_bars(self, contract: Stock, symbol: str, bar_size: str) -> None:
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="3 D",
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=False,
            formatDate=1,
            keepUpToDate=True,
        )

        buffered_bars = self.data_buffer.setdefault(symbol, {}).setdefault(bar_size, [])
        try:
            fetched_count = len(bars)
        except Exception:
            fetched_count = len(buffered_bars)
        self.logger.info(
            "Confirmed fetch: %s fetched %d bars for timeframe=%s",
            symbol,
            fetched_count,
            bar_size,
        )

        for bar in bars:
            buffered_bars.append(
                BarData(
                    symbol=symbol,
                    timestamp=bar.date,
                    timeframe=bar_size,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                )
            )
        bars.updateEvent += self._handle_bar_update(symbol, bar_size)
        self._subscriptions.append(bars)
        latest_bar = bars[-1] if bars else None
        if latest_bar is not None:
            self.logger.info(
                "Fetched %s %s bar with open=%s close=%s",
                symbol,
                bar_size,
                latest_bar.open,
                latest_bar.close,
            )
        else:
            self.logger.info("Fetched %s %s bar", symbol, bar_size)

    # Stream market data - start streams concurrently
    async def stream_market_data(self) -> None:
        if not self.ib.isConnected():
            await self.connect()

        self.logger.info("Initializing concurrent market data streams...")
        semaphore = asyncio.Semaphore(3) 

        async def _bounded_subscribe(symbol: str):
            async with semaphore:
                await self._subscribe_to_ticker(symbol)
                await asyncio.sleep(1) # Brief pause between ticker batches

        tasks = [_bounded_subscribe(symbol) for symbol in self.tickers]
        await asyncio.gather(*tasks)

    # Disconnect cleanly - stop IB and handlers
    def disconnect(self) -> None:
        self.ib.disconnectedEvent -= self._on_disconnected
        if self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("Disconnected from IBKR cleanly.")