import contextlib
import asyncio
import logging
import os
import signal
import sys
import math
from datetime import datetime, timedelta, timezone, date
from logging.handlers import RotatingFileHandler
from data.screener import VolumeGainerScreener
from execution.order_manager import OrderManager
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import yaml
from dotenv import load_dotenv

from backtests.engine import BacktestEngine
from data.ibkr_client import IBKRClient
from data.news_client import NewsClient
from discord_bot.controller import OpenClawDiscord
from strategy.indicators import IndicatorCalculator
from strategy.logic import StrategyEngine

_ET = ZoneInfo("America/New_York")


class ConfigurationError(Exception):
    pass


app = FastAPI(title="OpenClaw Trading API")


class TradeRequest(BaseModel):
    symbol: str
    action: str
    quantity: float


@app.get("/status")
async def get_status(request: Request) -> Dict[str, Any]:
    client = request.app.state.client
    settings = request.app.state.settings
    return {
        "ibkr_connected": client.ib.isConnected(),
        "active_tickers": settings["tickers"],
        "market_open": is_market_open(),
    }


@app.post("/execute")
async def execute_trade(request: Request, trade_request: TradeRequest) -> Dict[str, Any]:
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
    if not path.exists():
        raise ConfigurationError(f"Configuration file missing at: {path}")

    with path.open("r", encoding="utf-8") as file:
        settings = yaml.safe_load(file) or {}

    if "tickers" not in settings or not settings["tickers"]:
        raise ConfigurationError("Missing or empty 'tickers' list in settings.yaml")

    return settings


def _now_et() -> datetime:
    return datetime.now(_ET)


def is_market_open() -> bool:
    now = _now_et()
    if now.weekday() >= 5:
        return False
    opens = now.replace(hour=9, minute=30, second=0, microsecond=0)
    closes = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return opens <= now < closes


def _is_pre_close() -> bool:
    """True during the 10-minute window before market close (15:50–16:00 ET)."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    pre_close = now.replace(hour=15, minute=50, second=0, microsecond=0)
    closes = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return pre_close <= now < closes


def _is_just_after_close() -> bool:
    """True for the 10-minute window immediately after market close (16:00–16:10 ET)."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    closes = now.replace(hour=16, minute=0, second=0, microsecond=0)
    recap_end = now.replace(hour=16, minute=10, second=0, microsecond=0)
    return closes <= now < recap_end


def _compute_eod_stats(daily_closed_trades: List[Dict[str, Any]], account_equity: float) -> Dict[str, Any]:
    total = len(daily_closed_trades)
    if total == 0:
        return {
            "net_pnl": 0.0,
            "account_equity": account_equity,
            "total_trades": 0,
            "wins": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_confidence_wins": 0.0,
            "avg_confidence_losses": 0.0,
        }

    net_pnl = sum(t["pnl"] for t in daily_closed_trades)
    win_trades = [t for t in daily_closed_trades if t["pnl"] >= 0]
    loss_trades = [t for t in daily_closed_trades if t["pnl"] < 0]

    avg_win = sum(t["pnl"] for t in win_trades) / len(win_trades) if win_trades else 0.0
    avg_loss = sum(t["pnl"] for t in loss_trades) / len(loss_trades) if loss_trades else 0.0
    largest_win = max((t["pnl"] for t in win_trades), default=0.0)
    largest_loss = min((t["pnl"] for t in loss_trades), default=0.0)

    conf_wins = [t["confidence"] for t in win_trades if t["confidence"] > 0]
    conf_losses = [t["confidence"] for t in loss_trades if t["confidence"] > 0]
    avg_conf_win = sum(conf_wins) / len(conf_wins) if conf_wins else 0.0
    avg_conf_loss = sum(conf_losses) / len(conf_losses) if conf_losses else 0.0

    return {
        "net_pnl": net_pnl,
        "account_equity": account_equity,
        "total_trades": total,
        "wins": len(win_trades),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "avg_confidence_wins": avg_conf_win,
        "avg_confidence_losses": avg_conf_loss,
    }


async def main() -> None:
    setup_logging()
    load_dotenv()
    logger = logging.getLogger("Main")

    settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"

    try:
        settings = load_settings(settings_path)
    except ConfigurationError as e:
        logger.error(e)
        sys.exit(1)

    # Run screener once at startup (with brief delay to ensure APIs are ready)
    logger.info("Waiting 2 seconds before screener startup...")
    await asyncio.sleep(2)

    screener = VolumeGainerScreener()
    screened_tickers = await screener.load_cached_symbols()

    if screened_tickers:
        settings["tickers"] = screened_tickers
        logger.info(f"Using screened tickers: {screened_tickers}")
    else:
        fallback_tickers = settings.get("tickers", [])
        logger.warning(f"Screener failed, using fallback from settings.yaml: {fallback_tickers}")

    client = IBKRClient(settings)
    order_manager = OrderManager(client)
    news_client = NewsClient()
    discord_ui = OpenClawDiscord(order_manager)

    # Attach screener and settings to Discord bot for !screener command
    discord_ui.screener = screener
    discord_ui.settings = settings

    async def _subscribe_if_new(symbol: str) -> None:
        if symbol not in client.data_buffer:
            logger.info("Subscribing new ticker from Discord command: %s", symbol)
            await client._subscribe_to_ticker(symbol)
        else:
            logger.debug("Ticker %s already buffered, skipping re-subscription.", symbol)

    discord_ui.on_add_ticker = _subscribe_if_new

    # ── Backtest callbacks ────────────────────────────────────────────────────
    backtest_engine = BacktestEngine(client, start_equity=100_000.0)

    async def _run_backtest(channel_id: int) -> None:
        settings["backtest_mode"] = True
        logger.info("Backtest mode ON — live trading suspended.")
        try:
            account_equity = order_manager.get_account_equity()
            tickers = list(settings.get("tickers", []))
            result = await backtest_engine.run(
                tickers=tickers,
                account_equity=account_equity if account_equity > 0 else None,
            )
            await discord_ui.send_backtest_result(result, channel_id=channel_id)
        except Exception as exc:
            logger.error("Backtest failed: %s", exc, exc_info=True)
            channel = discord_ui.get_channel(channel_id)
            if channel:
                await channel.send(f"Backtest encountered an error: {exc}")
        finally:
            settings["backtest_mode"] = False
            logger.info("Backtest mode OFF — live trading resumed.")

    def _stop_backtest() -> None:
        settings["backtest_mode"] = False
        logger.info("Backtest mode cleared manually.")

    discord_ui.on_backtest_start = _run_backtest
    discord_ui.on_backtest_stop = _stop_backtest

    discord_token = os.getenv("DISCORD_TOKEN")
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    server: Optional[uvicorn.Server] = None
    server_task: Optional[asyncio.Task[Any]] = None
    last_buy_time: Dict[str, datetime] = {}
    last_sell_time: Dict[str, datetime] = {}
    open_trade_memory: Dict[str, Dict[str, Any]] = {}

    # Day-scoped state — reset each trading day
    daily_closed_trades: List[Dict[str, Any]] = []
    eod_recap_sent_date: Optional[date] = None
    eod_liquidation_done_date: Optional[date] = None

    app.state.client = client
    app.state.settings = settings
    app.state.order_manager = order_manager

    def graceful_shutdown() -> None:
        logger.info("Shutdown signal received. Initiating graceful exit...")
        shutdown_event.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, graceful_shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda sig, frame: loop.call_soon_threadsafe(graceful_shutdown))
        signal.signal(signal.SIGTERM, lambda sig, frame: loop.call_soon_threadsafe(graceful_shutdown))

    def _record_closed_trade(entry_price: float, exit_price: float, quantity: float, confidence: float) -> None:
        if entry_price > 0.0 and exit_price > 0.0:
            pnl = (exit_price - entry_price) * quantity
            daily_closed_trades.append({
                "pnl": pnl,
                "confidence": confidence,
            })

    def _get_last_price(symbol: str) -> float:
        bars = list(client.data_buffer.get(symbol, {}).get("5 mins", []))
        if bars:
            try:
                return float(bars[-1].close)
            except (TypeError, ValueError, AttributeError):
                return 0.0
        return 0.0

    async def liquidate_all_positions(reason: str = "Manual") -> List[str]:
        results: List[str] = []
        now = datetime.now(timezone.utc)
        for symbol, quantity in order_manager.get_all_positions().items():
            is_long = quantity > 0
            action = "SELL" if is_long else "BUY"
            close_qty = max(1, int(abs(quantity)))

            await order_manager.execute_trade(symbol, action, close_qty)
            symbol_key = symbol.upper().strip()
            last_sell_time[symbol_key] = now

            trade_memory = open_trade_memory.pop(symbol_key, None)
            entry_price = 0.0
            entry_time_mem = None
            if trade_memory:
                entry_price = float(trade_memory.get("entry_price", 0.0))
                entry_time_mem = trade_memory.get("entry_time")
                exit_price = _get_last_price(symbol) or entry_price
                confidence = float(trade_memory.get("entry_confidence", 0.0))
                _record_closed_trade(entry_price, exit_price, close_qty, confidence)

                if is_long:
                    outcome_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
                else:
                    outcome_pct = ((entry_price - exit_price) / entry_price * 100) if entry_price > 0 else 0.0

                outcome_label = "Win" if outcome_pct >= 0 else "Loss"
                await news_client.record_trade_memory_async(
                    symbol=symbol,
                    technical_context=str(trade_memory.get("technical_context", "")),
                    prediction=str(trade_memory.get("prediction", "BULLISH")),
                    outcome=f"{outcome_pct:+.2f}% ({outcome_label})",
                )
            else:
                exit_price = _get_last_price(symbol) or 0.0

            exit_reason = "End-of-day close" if reason == "End-of-Day" else "Manual close (!closeall)"
            await discord_ui.send_close_alert(
                symbol=symbol,
                is_long=is_long,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=close_qty,
                entry_time=entry_time_mem,
                exit_reason=exit_reason,
                market_story=str(trade_memory.get("technical_context", "")) if trade_memory else "",
            )
            results.append(f"{symbol}: {action} {close_qty} shares [{reason}]")
            logger.info("Closed %s x%d via %s — reason: %s", symbol, close_qty, action, reason)
        return results

    discord_ui.on_manual_sell = liquidate_all_positions

    try:
        logger.info("Starting...")
        if discord_token:
            logger.info("Discord token loaded securely. Starting Discord UI...")
            asyncio.create_task(discord_ui.start(discord_token))
        else:
            logger.warning("DISCORD_TOKEN is missing from the environment; Discord UI will not start.")

        async def _stream_news() -> None:
            logger.info("Starting news stream...")

            async def _build_trade_snapshot(symbol: str) -> tuple[str, float, float]:
                bars_5m = list(client.data_buffer.get(symbol, {}).get("5 mins", []))
                if len(bars_5m) < 2:
                    return "Insufficient 5-minute market data yet.", 0.0, 0.0

                indicator_calculator = IndicatorCalculator()
                df_5m = await asyncio.to_thread(indicator_calculator.calculate_all, bars_5m)
                if df_5m.empty:
                    return "Technical dataframe is empty.", 0.0, 0.0

                strategy_engine = StrategyEngine()
                technical_5m = await asyncio.to_thread(strategy_engine.evaluate_signals, df_5m)

                # Evaluate 1-hour trend context
                bars_1h = list(client.data_buffer.get(symbol, {}).get("1 hour", []))
                hourly_trend = ""
                if len(bars_1h) >= 2:
                    df_1h = await asyncio.to_thread(indicator_calculator.calculate_all, bars_1h)
                    if not df_1h.empty:
                        hourly_trend = await asyncio.to_thread(strategy_engine.evaluate_hourly_trend, df_1h)

                # Combine 1h trend + 5m signals
                technical_context = f"{hourly_trend} | 5m: {technical_5m}" if hourly_trend else technical_5m

                current_price = 0.0
                latest_close = df_5m.iloc[-1].get("close")
                if latest_close is not None and latest_close == latest_close:
                    try:
                        current_price = float(latest_close)
                    except (TypeError, ValueError):
                        current_price = 0.0

                current_atr = 0.0
                latest_atr = df_5m.iloc[-1].get("atr_14", 0.0)
                try:
                    current_atr = float(latest_atr)
                    if not math.isfinite(current_atr):
                        current_atr = 0.0
                except (TypeError, ValueError):
                    current_atr = 0.0

                return technical_context, current_price, current_atr

            def _trade_outcome(entry_price: float, exit_price: float) -> str:
                if entry_price <= 0.0 or exit_price <= 0.0:
                    return "UNKNOWN"
                pct_change = ((exit_price - entry_price) / entry_price) * 100.0
                outcome_label = "Win" if pct_change >= 0.0 else "Loss"
                return f"{pct_change:+.2f}% ({outcome_label})"

            async def _route_trade(
                symbol: str,
                signal_direction: str,
                technical_context: str,
                current_price: float,
                current_atr: float,
                confidence: float,
            ) -> None:
                # Gate all order execution to regular market hours only.
                if not is_market_open():
                    logger.info(
                        "Pre/after-market signal detected for %s | signal=%s | confidence=%.2f | "
                        "Evaluating only — execution gated until 09:30 ET.",
                        symbol, signal_direction, confidence,
                    )
                    return

                now = datetime.now(timezone.utc)
                symbol_key = symbol.upper().strip()
                holding_quantity = order_manager.get_position(symbol)
                is_holding = holding_quantity != 0.0

                if signal_direction == "BULLISH" and not is_holding:
                    last_buy = last_buy_time.get(symbol_key)
                    last_sell = last_sell_time.get(symbol_key)

                    if last_buy and now - last_buy < timedelta(minutes=15):
                        logger.info("Skipping BUY for %s due to buy cooldown.", symbol)
                        return

                    if last_sell and now - last_sell < timedelta(minutes=15):
                        logger.info("Skipping BUY for %s due to recent sell cooldown.", symbol)
                        return

                    account_equity = order_manager.get_account_equity()
                    target_capital = account_equity * 0.015 * confidence if account_equity > 0.0 else 0.0
                    calculated_shares = (target_capital / current_price) if current_price > 0.0 else 0.0
                    trade_size = max(1, int(calculated_shares))
                    logger.info(
                        "Sizing check | %s | equity=%.2f | target_capital=%.2f | current_price=%.2f | "
                        "calculated_shares=%.2f | executed_quantity=%d",
                        symbol, account_equity, target_capital, current_price, calculated_shares, trade_size,
                    )

                    await order_manager.execute_trade(symbol, "BUY", trade_size)
                    last_buy_time[symbol_key] = now
                    open_trade_memory[symbol_key] = {
                        "technical_context": technical_context,
                        "prediction": signal_direction,
                        "entry_price": current_price,
                        "entry_time": now,
                        "quantity": trade_size,
                        "highest_price_seen": current_price,
                        "initial_atr": current_atr,
                        "entry_confidence": confidence,
                    }
                    stop_loss = current_price * 0.98 if current_price > 0.0 else "N/A"
                    target_price = current_price * 1.04 if current_price > 0.0 else "N/A"
                    await discord_ui.send_execution_alert(
                        symbol=symbol,
                        action="BUY",
                        confidence=confidence,
                        market_story=technical_context,
                        entry_price=current_price,
                        target_price=target_price,
                        stop_loss=stop_loss,
                        quantity=trade_size,
                    )
                    logger.info("BUY executed for %s with quantity %d.", symbol, trade_size)
                    return

                if is_holding:
                    trade_memory = open_trade_memory.get(symbol_key)
                    entry_price = 0.0
                    if trade_memory is not None:
                        try:
                            entry_price = float(trade_memory.get("entry_price", 0.0))
                        except (TypeError, ValueError):
                            entry_price = 0.0

                    if entry_price > 0.0 and current_price > 0.0:
                        loss_pct = ((entry_price - current_price) / entry_price) * 100.0
                        if loss_pct >= 2.0:
                            sell_quantity = max(1, int(abs(holding_quantity)))
                            logger.warning(
                                "EMERGENCY STOP-LOSS TRIGGERED for %s | entry_price=%.2f | "
                                "current_price=%.2f | loss_pct=%.2f | quantity=%d",
                                symbol, entry_price, current_price, loss_pct, sell_quantity,
                            )

                            await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                            last_sell_time[symbol_key] = now

                            stopped_trade_memory = open_trade_memory.pop(symbol_key, None)
                            entry_time_mem = stopped_trade_memory.get("entry_time") if stopped_trade_memory else None
                            if stopped_trade_memory:
                                entry_conf = float(stopped_trade_memory.get("entry_confidence", 0.0))
                                _record_closed_trade(entry_price, current_price, sell_quantity, entry_conf)
                                outcome = _trade_outcome(entry_price, current_price)
                                await news_client.record_trade_memory_async(
                                    symbol=symbol,
                                    technical_context=str(stopped_trade_memory.get("technical_context", technical_context)),
                                    prediction=str(stopped_trade_memory.get("prediction", signal_direction)),
                                    outcome=outcome,
                                )

                            is_long = holding_quantity > 0
                            await discord_ui.send_close_alert(
                                symbol=symbol,
                                is_long=is_long,
                                entry_price=entry_price,
                                exit_price=current_price,
                                quantity=sell_quantity,
                                entry_time=entry_time_mem,
                                exit_reason="2% stop-loss triggered",
                                market_story=technical_context,
                            )
                            logger.info("Close executed for %s x%d — 2%% stop-loss.", symbol, sell_quantity)
                            return

                if signal_direction == "BEARISH" and is_holding:
                    last_buy = last_buy_time.get(symbol_key)
                    if last_buy and now - last_buy < timedelta(minutes=5):
                        logger.info("Skipping SELL for %s because the minimum hold time has not elapsed.", symbol)
                        return

                    sell_quantity = max(1, int(abs(holding_quantity)))
                    await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                    last_sell_time[symbol_key] = now

                    trade_memory = open_trade_memory.pop(symbol_key, None)
                    entry_price_mem = 0.0
                    entry_time_mem = None
                    if trade_memory:
                        entry_price_mem = float(trade_memory.get("entry_price", 0.0))
                        entry_time_mem = trade_memory.get("entry_time")
                        entry_conf = float(trade_memory.get("entry_confidence", 0.0))
                        _record_closed_trade(entry_price_mem, current_price, sell_quantity, entry_conf)
                        outcome = _trade_outcome(entry_price_mem, current_price)
                        await news_client.record_trade_memory_async(
                            symbol=symbol,
                            technical_context=str(trade_memory.get("technical_context", technical_context)),
                            prediction=str(trade_memory.get("prediction", signal_direction)),
                            outcome=outcome,
                        )

                    is_long = holding_quantity > 0
                    await discord_ui.send_close_alert(
                        symbol=symbol,
                        is_long=is_long,
                        entry_price=entry_price_mem,
                        exit_price=current_price,
                        quantity=sell_quantity,
                        entry_time=entry_time_mem,
                        exit_reason="AI bearish signal",
                        market_story=technical_context,
                    )
                    logger.info("Close executed for %s x%d — AI bearish signal.", symbol, sell_quantity)
                    return

                logger.info(
                    "No autonomous trade executed for %s | holding=%s | signal=%s",
                    symbol, is_holding, signal_direction,
                )

            while not shutdown_event.is_set():
                # Pause live trading while a backtest is running
                if settings.get("backtest_mode"):
                    await asyncio.sleep(5)
                    continue

                nonlocal eod_recap_sent_date, eod_liquidation_done_date
                today_et = _now_et().date()

                # Reset daily stats at the start of each new trading day
                if eod_recap_sent_date is not None and eod_recap_sent_date != today_et:
                    daily_closed_trades.clear()

                # End of Day recap — sent once in the 10-minute window after close
                if _is_just_after_close() and eod_recap_sent_date != today_et:
                    logger.info("Market closed. Sending EoD recap.")
                    stats = _compute_eod_stats(daily_closed_trades, order_manager.get_account_equity())
                    await discord_ui.send_eod_recap(stats)
                    eod_recap_sent_date = today_et

                # Pre-close forced liquidation — clear all positions 10 min before close
                if _is_pre_close() and eod_liquidation_done_date != today_et:
                    logger.info("Pre-close window reached. Liquidating all positions.")
                    await liquidate_all_positions(reason="End-of-Day")
                    eod_liquidation_done_date = today_et

                for symbol in settings["tickers"]:
                    if settings.get("backtest_mode"):
                        break
                    technical_context, current_price, current_atr = await _build_trade_snapshot(symbol)

                    holding_quantity = order_manager.get_position(symbol)
                    is_holding = holding_quantity != 0.0
                    symbol_key = symbol.upper().strip()

                    # ATR trailing stop and take-profit (active regardless of time, protect open capital)
                    if is_holding:
                        trade_memory = open_trade_memory.get(symbol_key)
                        if trade_memory is not None:
                            try:
                                highest_price_seen = float(trade_memory.get("highest_price_seen", trade_memory.get("entry_price", current_price)))
                            except (TypeError, ValueError):
                                highest_price_seen = float(trade_memory.get("entry_price", current_price) or current_price)

                            if current_price > highest_price_seen:
                                trade_memory["highest_price_seen"] = current_price
                                highest_price_seen = current_price

                            try:
                                entry_price = float(trade_memory.get("entry_price", 0.0))
                            except (TypeError, ValueError):
                                entry_price = 0.0

                            try:
                                initial_atr = float(trade_memory.get("initial_atr", 0.0))
                            except (TypeError, ValueError):
                                initial_atr = 0.0

                            entry_conf = float(trade_memory.get("entry_confidence", 0.0))

                            trailing_stop_level = highest_price_seen - (2 * current_atr)
                            if current_price > 0.0 and current_price <= trailing_stop_level:
                                sell_quantity = max(1, int(abs(holding_quantity)))
                                logger.warning("ATR TRAILING STOP TRIGGERED for %s", symbol)

                                await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                                last_sell_time[symbol_key] = datetime.now(timezone.utc)

                                stopped_trade_memory = open_trade_memory.pop(symbol_key, None)
                                entry_time_mem = stopped_trade_memory.get("entry_time") if stopped_trade_memory else None
                                if stopped_trade_memory:
                                    _record_closed_trade(entry_price, current_price, sell_quantity, entry_conf)
                                    outcome = _trade_outcome(entry_price, current_price)
                                    await news_client.record_trade_memory_async(
                                        symbol=symbol,
                                        technical_context=str(stopped_trade_memory.get("technical_context", technical_context)),
                                        prediction=str(stopped_trade_memory.get("prediction", "BULLISH")),
                                        outcome=outcome,
                                    )

                                is_long = holding_quantity > 0
                                await discord_ui.send_close_alert(
                                    symbol=symbol,
                                    is_long=is_long,
                                    entry_price=entry_price,
                                    exit_price=current_price,
                                    quantity=sell_quantity,
                                    entry_time=entry_time_mem,
                                    exit_reason="ATR trailing stop",
                                    market_story=technical_context,
                                )
                                logger.info("Close executed for %s x%d — ATR trailing stop.", symbol, sell_quantity)
                                continue

                            take_profit_level = entry_price + (3 * initial_atr)
                            if entry_price > 0.0 and initial_atr > 0.0 and current_price >= take_profit_level:
                                sell_quantity = max(1, int(abs(holding_quantity)))
                                logger.warning("TAKE PROFIT TARGET MET for %s", symbol)

                                await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                                last_sell_time[symbol_key] = datetime.now(timezone.utc)

                                stopped_trade_memory = open_trade_memory.pop(symbol_key, None)
                                entry_time_mem = stopped_trade_memory.get("entry_time") if stopped_trade_memory else None
                                if stopped_trade_memory:
                                    _record_closed_trade(entry_price, current_price, sell_quantity, entry_conf)
                                    outcome = _trade_outcome(entry_price, current_price)
                                    await news_client.record_trade_memory_async(
                                        symbol=symbol,
                                        technical_context=str(stopped_trade_memory.get("technical_context", technical_context)),
                                        prediction=str(stopped_trade_memory.get("prediction", "BULLISH")),
                                        outcome=outcome,
                                    )

                                is_long = holding_quantity > 0
                                await discord_ui.send_close_alert(
                                    symbol=symbol,
                                    is_long=is_long,
                                    entry_price=entry_price,
                                    exit_price=current_price,
                                    quantity=sell_quantity,
                                    entry_time=entry_time_mem,
                                    exit_reason="Take-profit target met",
                                    market_story=technical_context,
                                )
                                logger.info("Close executed for %s x%d — take profit.", symbol, sell_quantity)
                                continue

                    # News evaluation runs continuously (pre-market builds context, only executes when open)
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

                        confidence = abs(sentiment_score)
                        if confidence < 0.5:
                            continue

                        signal_direction = "BULLISH" if sentiment_score > 0 else "BEARISH"

                        await _route_trade(
                            symbol=news_item.symbol,
                            signal_direction=signal_direction,
                            technical_context=technical_context,
                            current_price=current_price,
                            current_atr=current_atr,
                            confidence=confidence,
                        )
                        break

                # Sleep in 5-second increments so backtest_mode is noticed within 5 s
                for _ in range(12):
                    if shutdown_event.is_set() or settings.get("backtest_mode"):
                        break
                    await asyncio.sleep(5)

        asyncio.create_task(_stream_news())

        startup_attempts = 12
        for attempt in range(1, startup_attempts + 1):
            try:
                await client.stream_market_data()
                break
            except RuntimeError as exc:
                if shutdown_event.is_set():
                    raise

                if attempt >= startup_attempts:
                    raise

                logger.warning(
                    "IBKR startup handshake failed (%d/%d): %s. Retrying in 5 seconds.",
                    attempt, startup_attempts, exc,
                )
                await asyncio.sleep(5)

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
        order_manager.close()
        client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger("Main").info("Stopped manually.")
        sys.exit(0)
