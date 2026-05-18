from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from ib_insync import LimitOrder, MarketOrder, Stock, StopOrder
from data.ibkr_client import IBKRClient


class OrderManager:
    # Initialize manager - bind IB and prepare state
    def __init__(self, ib_client_or_ib: Any) -> None:
        self.ib = getattr(ib_client_or_ib, "ib", ib_client_or_ib)
        self.exchange = getattr(ib_client_or_ib, "exchange", "SMART")
        self.currency = getattr(ib_client_or_ib, "currency", "USD")

        self.order_states: Dict[int, str] = {}
        self.logger = logging.getLogger(__name__)
        self._qualified_contracts: Dict[str, Stock] = {}
        self.logger = logging.getLogger("OrderManager")
        self._setup_logging()

        self.ib.orderStatusEvent += self._on_order_status

    # Configure logging outputs - console + file
    def _setup_logging(self) -> None:
        if self.logger.handlers:
            return

        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        log_dir = Path(__file__).resolve().parents[2] / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "order_manager.log"

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    # Log order updates - update internal state map
    def _on_order_status(self, trade: Any) -> None:
        try:
            order_id = getattr(trade.order, "orderId", None)
            status = getattr(trade.orderStatus, "status", "Unknown")
            symbol = getattr(trade.contract, "symbol", "UNKNOWN")

            if order_id is None:
                return

            previous_status = self.order_states.get(order_id)
            if previous_status != status:
                self.order_states[order_id] = status
                self.logger.info(
                    "Order %s %s: %s -> %s",
                    order_id,
                    symbol,
                    previous_status or "New",
                    status,
                )

            if status in {"Rejected", "Inactive", "ApiCancelled"}:
                self.logger.warning("Order %s %s ended with status %s", order_id, symbol, status)
        except Exception as exc:
            self.logger.exception("Failed to process order status update: %s", exc)

    # Ensure contract valid - qualify with IB
    async def _qualify_stock(self, symbol: str) -> Stock:
        if symbol in self._qualified_contracts:
            return self._qualified_contracts[symbol]
        
        contract = Stock(symbol, self.exchange, self.currency)
        await self.ib.qualifyContractsAsync(contract)

        self._qualified_contracts[symbol] = contract
        return contract

    # Place and track order - submit via IB
    async def _place_order(self, symbol: str, order: Any) -> Any:
        try:
            contract = await self._qualify_stock(symbol)
            trade = self.ib.placeOrder(contract, order)

            self.logger.info(
                "Placed %s order for %s x%s",
                getattr(order, "orderType", order.__class__.__name__),
                symbol,
                getattr(order, "totalQuantity", "?"),
            )
            return trade
        except Exception as exc:
            self.logger.exception("Failed to place order for %s: %s", symbol, exc)
            raise

    # Convenience market order - quick market submit
    async def place_market_order(self, symbol: str, action: str, quantity: float) -> Any:
        order = MarketOrder(action.upper(), quantity)
        return await self._place_order(symbol, order)

    # Convenience limit order - submit with cap price
    async def place_limit_order(self, symbol: str, action: str, quantity: float, price: float) -> Any:
        order = LimitOrder(action.upper(), quantity, price)
        return await self._place_order(symbol, order)

    # Convenience stop order - submit with stop price
    async def place_stop_order(self, symbol: str, action: str, quantity: float, stop_price: float) -> Any:
        order = StopOrder(action.upper(), quantity, stop_price)
        return await self._place_order(symbol, order)

    # Remove event hook - cleanup before exit
    def close(self) -> None:
        try:
            self.ib.orderStatusEvent -= self._on_order_status
        except Exception:
            pass