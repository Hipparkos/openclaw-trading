import asyncio
from typing import Any, Callable, Dict, List

from ib_insync import IB, Stock


class IBKRClient:
    def __init__(self, settings: Dict[str, Any]) -> None:
        connection = settings.get("connection", {})
        self.host = connection.get("host", "localhost")
        self.port = connection.get("port", 4002) # TWS 7497 - Gateway 4002
        self.client_id = connection.get("clientId", 1)
        self.tickers = settings.get("tickers", [])

        # IBKR session
        self.ib = IB()
        self.max_retries = 5
        self.retry_delay = 3
        self._subscriptions: List[Any] = []

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
                print(f"Connected to IBKR at {self.host}:{self.port} with clientId {self.client_id}")
                return
            except Exception as exc:
                attempt += 1
                if attempt >= self.max_retries:
                    raise RuntimeError("Unable to connect to IBKR") from exc
                print(f"Connection failed ({attempt}/{self.max_retries}): {exc}. Retrying in {self.retry_delay}s")
                await asyncio.sleep(self.retry_delay)

    def _handle_bar_update(self, symbol: str, bar_size: str) -> Callable[[Any, bool], None]:
        def on_update(bars: Any, has_new_bar: bool) -> None:
            if not has_new_bar or not bars:
                return

            bar = bars[-1]
            print(
                f"{symbol} {bar_size} "
                f"{bar.date} O:{bar.open:.2f} H:{bar.high:.2f} "
                f"L:{bar.low:.2f} C:{bar.close:.2f} V:{bar.volume}"
            )

        return on_update

    async def stream_market_data(self) -> None:
        if not self.ib.isConnected():
            await self.connect()

        if not self.tickers:
            raise ValueError("No tickers configured")

        for symbol in self.tickers:
            contract = Stock(symbol, "SMART", "USD")
            await self.ib.qualifyContractsAsync(contract)

            for bar_size in ("1 min", "5 mins", "15 mins"):
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
                print(f"Fetched {symbol} {bar_size}")

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            print("Disconnected from IBKR.")