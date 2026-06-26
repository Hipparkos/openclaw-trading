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

import discord
from backtests.engine import BacktestEngine
from data.ibkr_client import IBKRClient
from data.news_client import NewsClient
from data.trade_db import TradeHistoryDB
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
    logger.info("Starting screener in 2s...")
    await asyncio.sleep(2)

    screener = VolumeGainerScreener()
    screened_tickers = await screener.load_cached_symbols()

    if screened_tickers:
        settings["tickers"] = screened_tickers
        logger.info(f"Trading screened tickers: {screened_tickers}")
    else:
        fallback_tickers = settings.get("tickers", [])
        logger.warning(f"Screener empty — using fallback tickers from settings.yaml: {fallback_tickers}")

    client = IBKRClient(settings)
    order_manager = OrderManager(client)
    news_client = NewsClient()
    discord_ui = OpenClawDiscord(order_manager)

    # Attach screener and settings to Discord bot for !screener command
    discord_ui.screener = screener
    discord_ui.settings = settings

    async def _subscribe_if_new(symbol: str) -> None:
        if symbol not in client.data_buffer:
            logger.info("Adding ticker from Discord: %s", symbol)
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

    # Persistent trade history (survives restarts, used for YTD/MTD/1W stats)
    trade_history_db = TradeHistoryDB()

    # Day-scoped state — reset each trading day
    daily_closed_trades: List[Dict[str, Any]] = []
    # Mutable dict so nested functions can mutate without nonlocal
    circuit_breaker = {"halted": False}
    # Risk/exit constants — kept identical to BacktestEngine so live mirrors the backtest
    DAILY_LOSS_LIMIT_PCT = 0.005  # halt new entries when day P&L < -0.5% equity
    ATR_TRAIL_MULT = 2.5          # trailing stop = peak − 2.5 × entry-time ATR
    TAKE_PROFIT_ATR_MULT = 3.0    # take-profit = entry + 3.0 × entry-time ATR
    MIN_HOLD_MINUTES = 25         # 5 × 5m bars before take-profit / reversal exits
    eod_recap_sent_date: Optional[date] = None
    eod_liquidation_done_date: Optional[date] = None

    app.state.client = client
    app.state.settings = settings
    app.state.order_manager = order_manager

    def graceful_shutdown() -> None:
        logger.info("Shutdown signal received — exiting gracefully...")
        shutdown_event.set()

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, graceful_shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda sig, frame: loop.call_soon_threadsafe(graceful_shutdown))
        signal.signal(signal.SIGTERM, lambda sig, frame: loop.call_soon_threadsafe(graceful_shutdown))

    def _record_closed_trade(
        entry_price: float,
        exit_price: float,
        quantity: float,
        confidence: float,
        *,
        symbol: str = "",
        entry_time: datetime | None = None,
    ) -> None:
        if entry_price > 0.0 and exit_price > 0.0:
            pnl = (exit_price - entry_price) * quantity
            daily_closed_trades.append({"pnl": pnl, "confidence": confidence})
            if symbol:
                trade_history_db.record(
                    symbol=symbol,
                    pnl=pnl,
                    confidence=confidence,
                    entry_time=entry_time,
                    exit_time=datetime.now(timezone.utc),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                )
            # Check circuit breaker after recording the trade
            if not circuit_breaker["halted"]:
                daily_net = sum(t["pnl"] for t in daily_closed_trades)
                account_eq = order_manager.get_account_equity()
                if account_eq > 0 and daily_net < -(DAILY_LOSS_LIMIT_PCT * account_eq):
                    circuit_breaker["halted"] = True
                    logger.warning(
                        "Circuit breaker tripped — daily loss $%.2f exceeds %.1f%% limit. "
                        "No new entries for the rest of the session.",
                        daily_net, DAILY_LOSS_LIMIT_PCT * 100,
                    )
                    asyncio.create_task(discord_ui.send_circuit_breaker_alert(
                        daily_loss=daily_net,
                        trade_count=len(daily_closed_trades),
                        equity=account_eq,
                        limit_pct=DAILY_LOSS_LIMIT_PCT,
                    ))

    def _get_last_price(symbol: str) -> float:
        bars = list(client.data_buffer.get(symbol, {}).get("5 mins", []))
        if bars:
            try:
                return float(bars[-1].close)
            except (TypeError, ValueError, AttributeError):
                return 0.0
        return 0.0

    def _cancel_resting_stop(trade_memory: dict | None) -> None:
        # Clear the protective stop order when we close a position by other means,
        # so it can't fire later against a fresh position in the same symbol.
        if not trade_memory:
            return
        stop_trade = trade_memory.get("stop_order")
        if stop_trade is not None:
            try:
                order_manager.cancel_order(stop_trade)
            except Exception as exc:
                logger.warning("Could not cancel resting stop: %s", exc)

    async def liquidate_all_positions(reason: str = "Manual") -> List[str]:
        results: List[str] = []
        now = datetime.now(timezone.utc)
        for symbol, quantity in order_manager.get_all_positions().items():
            is_long = quantity > 0
            action = "SELL" if is_long else "BUY"
            close_qty = max(1, int(abs(quantity)))

            symbol_key = symbol.upper().strip()
            # Cancel the protective stop before closing so the two can't both fill.
            trade_memory = open_trade_memory.pop(symbol_key, None)
            _cancel_resting_stop(trade_memory)

            await order_manager.execute_trade(symbol, action, close_qty)
            last_sell_time[symbol_key] = now
            entry_price = 0.0
            entry_time_mem = None
            if trade_memory:
                entry_price = float(trade_memory.get("entry_price", 0.0))
                entry_time_mem = trade_memory.get("entry_time")
                exit_price = _get_last_price(symbol) or entry_price
                confidence = float(trade_memory.get("entry_confidence", 0.0))
                _record_closed_trade(entry_price, exit_price, close_qty, confidence,
                                     symbol=symbol, entry_time=entry_time_mem)

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
            logger.info("Closed %s x%d via %s — %s.", symbol, close_qty, action, reason)
        return results

    discord_ui.on_manual_sell = liquidate_all_positions
    discord_ui.circuit_breaker = circuit_breaker

    try:
        logger.info("OpenClaw starting up...")
        if discord_token:
            logger.info("Discord token found — starting Discord UI...")
            asyncio.create_task(discord_ui.start(discord_token))
        else:
            logger.warning("DISCORD_TOKEN missing — Discord UI disabled.")

        async def _stream_news() -> None:
            logger.info("Starting news/signal loop...")

            async def _build_trade_snapshot(symbol: str) -> tuple[str, float, float, str]:
                bars_5m = list(client.data_buffer.get(symbol, {}).get("5 mins", []))
                if len(bars_5m) < 2:
                    return "Insufficient 5-minute market data yet.", 0.0, 0.0, "NEUTRAL", "NEUTRAL", 0.0

                indicator_calculator = IndicatorCalculator()
                df_5m = await asyncio.to_thread(indicator_calculator.calculate_all, bars_5m)
                if df_5m.empty:
                    return "Technical dataframe is empty.", 0.0, 0.0, "NEUTRAL", "NEUTRAL", 0.0

                strategy_engine = StrategyEngine()
                technical_5m = await asyncio.to_thread(strategy_engine.evaluate_signals, df_5m)

                # Evaluate 1-hour trend context
                bars_1h = list(client.data_buffer.get(symbol, {}).get("1 hour", []))
                hourly_trend = ""
                hourly_direction = "NEUTRAL"
                df_1h = None
                if len(bars_1h) >= 2:
                    df_1h = await asyncio.to_thread(indicator_calculator.calculate_all, bars_1h)
                    if not df_1h.empty:
                        hourly_trend = await asyncio.to_thread(strategy_engine.evaluate_hourly_trend, df_1h)
                        h_row = df_1h.iloc[-1]
                        h_close = h_row.get("close")
                        h_sma50 = h_row.get("sma_50")
                        if h_close is not None and h_sma50 is not None:
                            try:
                                if float(h_close) > float(h_sma50):
                                    hourly_direction = "BULLISH"
                                elif float(h_close) < float(h_sma50):
                                    hourly_direction = "BEARISH"
                            except (TypeError, ValueError):
                                pass

                # Market regime: ADX trend strength, volume, session timing
                regime_ctx = await asyncio.to_thread(strategy_engine.evaluate_regime, df_5m)

                # Combine all context layers
                technical_context = f"{hourly_trend} | 5m: {technical_5m}" if hourly_trend else technical_5m
                if regime_ctx:
                    technical_context = f"{technical_context} | {regime_ctx}"

                divergence = await asyncio.to_thread(indicator_calculator.calculate_divergence, df_5m)
                if divergence["strength"] > 0.0:
                    technical_context = (
                        f"{technical_context} | Divergence: RSI={divergence['rsi_divergence']} "
                        f"MACD={divergence['macd_divergence']} strength={divergence['strength']:.2f}"
                    )

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

                # Technical alignment gate — same logic the backtest uses to qualify
                # an entry (hourly trend + volume + 3/4 indicator vote).
                tech_dir, tech_conf = await asyncio.to_thread(
                    strategy_engine.compute_alignment_signal, df_5m, df_1h
                )

                return (technical_context, current_price, current_atr,
                        hourly_direction, tech_dir, tech_conf)

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
                        "%s signal %s (%.2f) outside market hours — noted, no order until 09:30 ET.",
                        symbol, signal_direction, confidence,
                    )
                    return

                now = datetime.now(timezone.utc)
                symbol_key = symbol.upper().strip()
                holding_quantity = order_manager.get_position(symbol)
                is_holding = holding_quantity != 0.0

                if signal_direction == "BULLISH" and not is_holding:
                    if circuit_breaker["halted"]:
                        logger.info("Circuit breaker on — no BUY for %s.", symbol)
                        return

                    # Opening 30-minute noise filter (9:30–10:00 ET) — mirrors the
                    # backtest, which skips entries while the hour is 9 in RTH data.
                    if datetime.now(_ET).hour == 9:
                        logger.info("%s: BUY skipped (opening 30-min filter).", symbol)
                        return

                    last_buy = last_buy_time.get(symbol_key)
                    last_sell = last_sell_time.get(symbol_key)

                    if last_buy and now - last_buy < timedelta(minutes=15):
                        logger.info("%s: BUY skipped (buy cooldown).", symbol)
                        return

                    if last_sell and now - last_sell < timedelta(minutes=15):
                        logger.info("%s: BUY skipped (recent sell cooldown).", symbol)
                        return

                    # Need a valid price to size the order and set the protective stop.
                    if current_price <= 0.0:
                        logger.warning("%s: BUY skipped — no valid price yet.", symbol)
                        return

                    account_equity = order_manager.get_account_equity()
                    target_capital = account_equity * 0.015 * confidence if account_equity > 0.0 else 0.0
                    calculated_shares = (target_capital / current_price) if current_price > 0.0 else 0.0
                    trade_size = max(1, int(calculated_shares))
                    logger.info(
                        "%s sizing | equity $%.2f | target $%.2f | price $%.2f | shares %.2f → %d",
                        symbol, account_equity, target_capital, current_price, calculated_shares, trade_size,
                    )

                    # Enter with a bracket order: a market BUY plus an attached resting
                    # 2% stop. The stop activates only when the BUY fills (so it can never
                    # rest naked) and fills near the stop intrabar — matching the backtest's
                    # stop model instead of relying on this loop to catch the breach late.
                    stop_price = round(current_price * 0.98, 2) if current_price > 0.0 else 0.0
                    stop_trade = None
                    if stop_price > 0.0:
                        try:
                            _, stop_trade = await order_manager.place_bracket_buy(
                                symbol, trade_size, stop_price
                            )
                            logger.info("BUY %s x%d with resting stop at $%.2f (−2%%).",
                                        symbol, trade_size, stop_price)
                        except Exception as exc:
                            logger.warning("Bracket BUY failed for %s: %s — using plain market order.",
                                           symbol, exc)
                            await order_manager.execute_trade(symbol, "BUY", trade_size)
                    else:
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
                        "stop_order": stop_trade,
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
                    logger.info("BUY filled: %s x%d.", symbol, trade_size)
                    return

                if is_holding:
                    trade_memory = open_trade_memory.get(symbol_key)
                    entry_price = 0.0
                    if trade_memory is not None:
                        try:
                            entry_price = float(trade_memory.get("entry_price", 0.0))
                        except (TypeError, ValueError):
                            entry_price = 0.0

                    # Fallback stop only — when no resting broker stop is active (e.g. it
                    # failed to place). Normally the broker's resting stop handles the 2%
                    # exit and is reconciled at the top of the polling loop.
                    has_resting_stop = bool(trade_memory and trade_memory.get("stop_order") is not None)
                    if entry_price > 0.0 and current_price > 0.0 and not has_resting_stop:
                        loss_pct = ((entry_price - current_price) / entry_price) * 100.0
                        if loss_pct >= 2.0:
                            sell_quantity = max(1, int(abs(holding_quantity)))
                            logger.warning(
                                "Fallback stop-loss: %s | entry $%.2f → now $%.2f (−%.2f%%) | selling %d.",
                                symbol, entry_price, current_price, loss_pct, sell_quantity,
                            )

                            await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                            last_sell_time[symbol_key] = now

                            stopped_trade_memory = open_trade_memory.pop(symbol_key, None)
                            entry_time_mem = stopped_trade_memory.get("entry_time") if stopped_trade_memory else None
                            if stopped_trade_memory:
                                entry_conf = float(stopped_trade_memory.get("entry_confidence", 0.0))
                                _record_closed_trade(entry_price, current_price, sell_quantity, entry_conf,
                                                     symbol=symbol, entry_time=entry_time_mem)
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
                            logger.info("Closed %s x%d — 2%% stop-loss.", symbol, sell_quantity)
                            return

                if signal_direction == "BEARISH" and is_holding:
                    last_buy = last_buy_time.get(symbol_key)
                    if last_buy and now - last_buy < timedelta(minutes=MIN_HOLD_MINUTES):
                        logger.info("%s: SELL skipped (min hold not met).", symbol)
                        return

                    sell_quantity = max(1, int(abs(holding_quantity)))
                    # Cancel the protective stop before closing so the two can't both fill.
                    trade_memory = open_trade_memory.pop(symbol_key, None)
                    _cancel_resting_stop(trade_memory)

                    await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                    last_sell_time[symbol_key] = now
                    entry_price_mem = 0.0
                    entry_time_mem = None
                    if trade_memory:
                        entry_price_mem = float(trade_memory.get("entry_price", 0.0))
                        entry_time_mem = trade_memory.get("entry_time")
                        entry_conf = float(trade_memory.get("entry_confidence", 0.0))
                        _record_closed_trade(entry_price_mem, current_price, sell_quantity, entry_conf,
                                             symbol=symbol, entry_time=entry_time_mem)
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
                    logger.info("Closed %s x%d — AI bearish signal.", symbol, sell_quantity)
                    return

                logger.info(
                    "No trade for %s (holding=%s, signal=%s).",
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
                    circuit_breaker["halted"] = False

                # End of Day recap — sent once in the 10-minute window after close
                if _is_just_after_close() and eod_recap_sent_date != today_et:
                    logger.info("Market closed — sending end-of-day recap.")
                    stats = _compute_eod_stats(daily_closed_trades, order_manager.get_account_equity())
                    await discord_ui.send_eod_recap(stats)
                    eod_recap_sent_date = today_et

                # Pre-close forced liquidation — clear all positions 10 min before close
                if _is_pre_close() and eod_liquidation_done_date != today_et:
                    logger.info("Pre-close — liquidating all positions.")
                    await liquidate_all_positions(reason="End-of-Day")
                    eod_liquidation_done_date = today_et

                scan = {"held": 0, "bull_setup": 0, "hourly_neutral": 0}
                for symbol in settings["tickers"]:
                    if settings.get("backtest_mode"):
                        break
                    (technical_context, current_price, current_atr,
                     hourly_direction, tech_dir, tech_conf) = await _build_trade_snapshot(symbol)

                    holding_quantity = order_manager.get_position(symbol)
                    is_holding = holding_quantity != 0.0
                    symbol_key = symbol.upper().strip()

                    if is_holding:
                        scan["held"] += 1
                    if tech_dir == "BULLISH":
                        scan["bull_setup"] += 1
                    if hourly_direction == "NEUTRAL":
                        scan["hourly_neutral"] += 1

                    # Reconcile broker-side stop fills: if the resting stop order has
                    # filled, the position is flat but we still hold trade memory. Record
                    # the close at the actual fill price and clear the slot.
                    if not is_holding and symbol_key in open_trade_memory:
                        pending_mem = open_trade_memory.get(symbol_key)
                        stop_trade = pending_mem.get("stop_order") if pending_mem else None
                        stop_filled = False
                        if stop_trade is not None:
                            try:
                                stop_filled = getattr(stop_trade.orderStatus, "status", "") == "Filled"
                            except Exception:
                                stop_filled = False
                        if stop_filled:
                            open_trade_memory.pop(symbol_key, None)
                            entry_price = float(pending_mem.get("entry_price", 0.0))
                            entry_time_mem = pending_mem.get("entry_time")
                            entry_conf = float(pending_mem.get("entry_confidence", 0.0))
                            closed_qty = max(1, int(pending_mem.get("quantity", 1)))
                            try:
                                fill_px = float(getattr(stop_trade.orderStatus, "avgFillPrice", 0.0)) or 0.0
                            except Exception:
                                fill_px = 0.0
                            exit_px = fill_px if fill_px > 0.0 else (entry_price * 0.98 if entry_price > 0.0 else current_price)
                            last_sell_time[symbol_key] = datetime.now(timezone.utc)
                            _record_closed_trade(entry_price, exit_px, closed_qty, entry_conf,
                                                 symbol=symbol, entry_time=entry_time_mem)
                            outcome = _trade_outcome(entry_price, exit_px)
                            await news_client.record_trade_memory_async(
                                symbol=symbol,
                                technical_context=str(pending_mem.get("technical_context", "")),
                                prediction=str(pending_mem.get("prediction", "BULLISH")),
                                outcome=outcome,
                            )
                            await discord_ui.send_close_alert(
                                symbol=symbol,
                                is_long=True,
                                entry_price=entry_price,
                                exit_price=exit_px,
                                quantity=closed_qty,
                                entry_time=entry_time_mem,
                                exit_reason="2% stop-loss (resting order filled)",
                                market_story=str(pending_mem.get("technical_context", "")),
                            )
                            logger.info("Resting stop filled for %s x%d — 2%% stop-loss.", symbol, closed_qty)
                            continue

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

                            # Trailing stop anchored to entry-time ATR (not current ATR),
                            # so the distance can't widen mid-trade — mirrors the backtest.
                            trailing_stop_level = highest_price_seen - (ATR_TRAIL_MULT * initial_atr)
                            if initial_atr > 0.0 and current_price > 0.0 and current_price <= trailing_stop_level:
                                sell_quantity = max(1, int(abs(holding_quantity)))
                                logger.warning("ATR trailing stop hit: %s.", symbol)

                                # Cancel the protective stop before closing so the two can't both fill.
                                stopped_trade_memory = open_trade_memory.pop(symbol_key, None)
                                _cancel_resting_stop(stopped_trade_memory)

                                await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                                last_sell_time[symbol_key] = datetime.now(timezone.utc)
                                entry_time_mem = stopped_trade_memory.get("entry_time") if stopped_trade_memory else None
                                if stopped_trade_memory:
                                    _record_closed_trade(entry_price, current_price, sell_quantity, entry_conf,
                                                         symbol=symbol, entry_time=entry_time_mem)
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
                                logger.info("Closed %s x%d — ATR trailing stop.", symbol, sell_quantity)
                                continue

                            # Take-profit, gated by the same 25-min minimum hold as the backtest.
                            entry_time_tp = trade_memory.get("entry_time")
                            held_long_enough = (
                                entry_time_tp is not None
                                and datetime.now(timezone.utc) - entry_time_tp >= timedelta(minutes=MIN_HOLD_MINUTES)
                            )
                            take_profit_level = entry_price + (TAKE_PROFIT_ATR_MULT * initial_atr)
                            if (held_long_enough and entry_price > 0.0 and initial_atr > 0.0
                                    and current_price >= take_profit_level):
                                sell_quantity = max(1, int(abs(holding_quantity)))
                                logger.warning("Take-profit hit: %s.", symbol)

                                # Cancel the protective stop before closing so the two can't both fill.
                                stopped_trade_memory = open_trade_memory.pop(symbol_key, None)
                                _cancel_resting_stop(stopped_trade_memory)

                                await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                                last_sell_time[symbol_key] = datetime.now(timezone.utc)
                                entry_time_mem = stopped_trade_memory.get("entry_time") if stopped_trade_memory else None
                                if stopped_trade_memory:
                                    _record_closed_trade(entry_price, current_price, sell_quantity, entry_conf,
                                                         symbol=symbol, entry_time=entry_time_mem)
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
                                logger.info("Closed %s x%d — take-profit.", symbol, sell_quantity)
                                continue

                    # ── Entry / exit decision (mirrors the backtest) ─────────────
                    # Held position: exit on a technical signal reversal — exactly like
                    # the backtest, which uses the cheap indicator vote (NOT a fresh
                    # LLM/news call) for reversals. Stops / take-profit / trailing are
                    # handled above. Never running the LLM on an open position is also
                    # what stops a single held name from monopolising the CPU and
                    # starving the rest of the watchlist.
                    if is_holding:
                        if tech_dir == "BEARISH":
                            await _route_trade(
                                symbol=symbol,
                                signal_direction="BEARISH",
                                technical_context=technical_context,
                                current_price=current_price,
                                current_atr=current_atr,
                                confidence=tech_conf,
                            )
                        continue

                    # Flat: the technical gate must qualify before we spend any LLM/news
                    # call. Flat-with-no-setup tickers are skipped cheaply, so the whole
                    # watchlist stays responsive every cycle.
                    if tech_dir != "BULLISH":
                        continue

                    news_items = await news_client.fetch_latest_news(symbol, technical_context=technical_context)

                    # Strongest directional news verdict across the fetched headlines.
                    llm_dir, llm_conf = "NEUTRAL", 0.0
                    for news_item in news_items:
                        logger.info(
                            "%s news | sentiment %.2f | %s",
                            news_item.symbol, news_item.sentiment_score, news_item.headline,
                        )
                        score = news_item.sentiment_score
                        if score is None or score != score or score == 0.0:
                            continue
                        if abs(score) > llm_conf:
                            llm_conf = abs(score)
                            llm_dir = "BULLISH" if score > 0 else "BEARISH"

                    # Flat with a bullish technical setup. Resolve the final signal the
                    # same way the backtest does: the news/LLM decides when it has an
                    # opinion (≥0.60); otherwise fall back to the technical vote (≥0.55).
                    if llm_dir != "NEUTRAL":
                        final_dir, final_conf, min_conf = llm_dir, llm_conf, 0.60
                    else:
                        final_dir, final_conf, min_conf = tech_dir, tech_conf, 0.55

                    if final_dir != "BULLISH":
                        logger.info("%s: entry skipped — news vetoed setup (verdict %s).", symbol, final_dir)
                        continue
                    if final_conf < min_conf:
                        logger.info("%s: entry skipped — confidence %.2f < %.2f.", symbol, final_conf, min_conf)
                        continue

                    await _route_trade(
                        symbol=symbol,
                        signal_direction="BULLISH",
                        technical_context=technical_context,
                        current_price=current_price,
                        current_atr=current_atr,
                        confidence=final_conf,
                    )

                if not settings.get("backtest_mode"):
                    logger.info(
                        "Scan: %d tickers | %d held | %d bullish setups | %d hourly-NEUTRAL (no 1h SMA-50?)",
                        len(settings["tickers"]), scan["held"], scan["bull_setup"], scan["hourly_neutral"],
                    )

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
                    "IBKR connect failed (%d/%d): %s — retrying in 5s.",
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
        logger.error(f"Execution loop error: {e}", exc_info=True)
    finally:
        logger.info("Shutting down — cleaning up connections...")
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
