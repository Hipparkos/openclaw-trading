from __future__ import annotations

import logging
from typing import Any, Dict

from ib_insync import LimitOrder, MarketOrder, Stock, StopOrder
from data.ibkr_client import IBKRClient

class OrderManager:
    # Initialize manager - bind IB and prepare state
    def __init__(self, client: IBKRClient) -> None:
        self.client = client
        self.ib = client.ib
        self.exchange = client.exchange
        self.currency = client.currency

        self.order_states: Dict[int, str] = {}
        self.logger = logging.getLogger(__name__)
        self._qualified_contracts: Dict[str, Stock] = {}

        self.ib.orderStatusEvent += self._on_order_status

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
                    "Order %s (%s): %s → %s",
                    order_id,
                    symbol,
                    previous_status or "New",
                    status,
                )

            if status in {"Rejected", "Inactive", "ApiCancelled"}:
                self.logger.warning("Order %s (%s) was not filled — status %s.", order_id, symbol, status)
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
                "Submitted order: %s x%s",
                symbol,
                getattr(order, "totalQuantity", "?"),
            )
            return trade
        except Exception as exc:
            self.logger.exception("Failed to place order for %s: %s", symbol, exc)
            raise

    def get_all_positions(self) -> dict[str, float]:
        positions: dict[str, float] = {}
        for position in self.ib.portfolio():
            symbol = getattr(position.contract, "symbol", "")
            if not symbol:
                continue
            try:
                qty = float(position.position)
            except (TypeError, ValueError):
                qty = 0.0
            if qty != 0.0:
                positions[symbol] = qty
        return positions

    def get_position(self, symbol: str) -> float:
        for position in self.ib.portfolio():
            if getattr(position.contract, "symbol", "") == symbol:
                try:
                    return float(position.position)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def get_account_equity(self) -> float:
        try:
            for item in self.ib.accountValues():
                if getattr(item, "tag", "") != "NetLiquidation":
                    continue

                value = getattr(item, "value", None)
                if value is None:
                    continue

                return float(str(value).replace(",", ""))
        except Exception as exc:
            self.logger.warning("Could not read account equity: %s", exc)

        return 0.0

    # Convenience market order - quick market submit
    async def place_market_order(self, symbol: str, action: str, quantity: float) -> Any:
        order = MarketOrder(action.upper(), quantity, tif="DAY")
        return await self._place_order(symbol, order)

    async def execute_trade(self, symbol: str, side: str, quantity: float = 10) -> Any:
        return await self.place_market_order(symbol, side, quantity)

    # Convenience limit order - submit with cap price
    async def place_limit_order(self, symbol: str, action: str, quantity: float, price: float) -> Any:
        order = LimitOrder(action.upper(), quantity, price, tif="DAY")
        return await self._place_order(symbol, order)

    # Convenience stop order - submit with stop price
    async def place_stop_order(self, symbol: str, action: str, quantity: float, stop_price: float) -> Any:
        order = StopOrder(action.upper(), quantity, stop_price, tif="DAY")
        return await self._place_order(symbol, order)

    # Cancel a resting order - used to clear a protective stop on early exit
    def cancel_order(self, trade: Any) -> None:
        order = getattr(trade, "order", None)
        if order is not None:
            self.ib.cancelOrder(order)

    def close(self) -> None:
        try:
            self.ib.orderStatusEvent -= self._on_order_status
        except Exception:
            pass