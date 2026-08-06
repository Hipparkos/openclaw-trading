import contextlib
import asyncio
import logging
import os
import signal
import sys
import math
from datetime import datetime, timedelta, timezone, date
from logging.handlers import RotatingFileHandler
from data.screener import MomentumScreener
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


def _compute_eod_stats(trades: List[Dict[str, Any]], account_equity: float) -> Dict[str, Any]:
    total = len(trades)
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

    net_pnl = sum(t["pnl"] for t in trades)
    win_trades = [t for t in trades if t["pnl"] >= 0]
    loss_trades = [t for t in trades if t["pnl"] < 0]

    avg_win = sum(t["pnl"] for t in win_trades) / len(win_trades) if win_trades else 0.0
    avg_loss = sum(t["pnl"] for t in loss_trades) / len(loss_trades) if loss_trades else 0.0
    largest_win = max((t["pnl"] for t in win_trades), default=0.0)
    largest_loss = min((t["pnl"] for t in loss_trades), default=0.0)

    # confidence is nullable in the DB — coerce so a legacy NULL row can't crash the recap
    conf_wins = [c for t in win_trades if (c := t["confidence"] or 0.0) > 0]
    conf_losses = [c for t in loss_trades if (c := t["confidence"] or 0.0) > 0]
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

    # Fast, non-blocking startup: use today's cached screen if present, otherwise the
    # settings.yaml fallback — so Discord and IBKR come online immediately. The full
    # (multi-minute) universe scan is deferred to the daily re-screen in the main loop,
    # which only fires once Discord is online.
    screener = MomentumScreener()
    screened_tickers = screener._cached_today()   # instant — reads cache file, no scan

    if screened_tickers:
        settings["tickers"] = screened_tickers
        logger.info(f"Using today's cached screen: {screened_tickers}")
    else:
        logger.info(
            "No fresh screen yet — starting on fallback tickers %s; momentum scan runs once Discord is online.",
            settings.get("tickers", []),
        )

    client = IBKRClient(settings)
    order_manager = OrderManager(client)
    news_client = NewsClient()
    discord_ui = OpenClawDiscord(order_manager)

    # Attach screener and settings to Discord bot for !screener command
    discord_ui.screener = screener
    discord_ui.settings = settings

    async def _subscribe_if_new(symbol: str) -> None:
        symbol = symbol.upper().strip()
        # Make sure the symbol is on the list the trading loop iterates. Being
        # data-subscribed alone isn't enough — the loop reads settings["tickers"].
        watchlist = settings.setdefault("tickers", [])
        if symbol not in watchlist:
            watchlist.append(symbol)

        if symbol in client.data_buffer:
            logger.info("Ticker %s already subscribed; confirmed on the trading list.", symbol)
            return

        # Runs as a detached task from the Discord command, so swallow nothing —
        # a silent failure here is exactly why a manually added ticker never starts.
        try:
            logger.info("Subscribing market data for %s (added via Discord)...", symbol)
            await client._subscribe_to_ticker(symbol)
            bars_5m = len(client.data_buffer.get(symbol, {}).get("5 mins", []))
            bars_1h = len(client.data_buffer.get(symbol, {}).get("1 hour", []))
            logger.info("%s now monitored — %d 5m / %d 1h bars loaded.", symbol, bars_5m, bars_1h)
        except Exception as exc:
            logger.exception("Failed to subscribe %s (added via Discord): %s", symbol, exc)

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
    # Mutable dict so nested functions can mutate without nonlocal
    circuit_breaker = {"halted": False}
    # Risk/exit constants — kept identical to BacktestEngine so live mirrors the backtest
    DAILY_LOSS_LIMIT_PCT = 0.015  # halt new entries when day P&L < -1.5% equity
    # Intraday geometry: on a 5-minute ATR these multipliers must be reachable
    # inside one session, since every position is flattened at 15:50 ET. A 6×ATR
    # stop / 12×ATR TP1 put the target ~43% away on a volatile name — unreachable,
    # so trades could only ever end at the stop or the EoD close.
    ATR_TRAIL_MULT = 2.5          # trailing stop = peak − 2.5 × entry-time ATR
    STOP_ATR_MULT = 2.0          # hard stop = entry − 2 × entry-time ATR (1:1.5 vs 3×ATR TP1)
    STOP_LOSS_PCT = 0.02         # fallback stop when ATR is unavailable
    TAKE_PROFIT_ATR_MULT = 3.0    # TP1 — first profit at entry + 3 × entry-time ATR
    TAKE_PROFIT_2_ATR_MULT = 6.0  # TP2 — runner target at entry + 6 × entry-time ATR
    SCALE_OUT_PCT = 0.60          # sell 60% at TP1, run the remaining 40%
    MIN_HOLD_MINUTES = 25         # 5 × 5m bars before take-profit / reversal exits
    # Kept well below DAILY_LOSS_LIMIT_PCT so one stop-out can't halt the day:
    # 1.5% / 0.25% = 6 full-risk losers before the circuit breaker trips.
    RISK_PER_TRADE = 0.0025       # size so each trade risks 0.25% of equity to the stop
    MAX_POSITION_PCT = 0.10       # hard ceiling on position size (caps the risk-parity result)
    LLM_CONF_THRESHOLD = 0.70     # min LLM confidence to open a trade
    MAX_GROSS_EXPOSURE_PCT = 0.90 # no new BUYs once open positions ≥ 90% of equity
    SCREEN_HOUR_ET = 9            # daily re-screen fires from 09:00 ET (pre-open)
    eod_recap_sent_date: Optional[date] = None
    eod_liquidation_done_date: Optional[date] = None
    # If the startup screen produced a list, mark today as done so the loop doesn't
    # immediately re-scan; if it came back empty, leave it unset so the loop retries.
    screened_date: Optional[date] = _now_et().date() if screened_tickers else None

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

    def _todays_trades() -> List[Dict[str, Any]]:
        # Today's closed trades, read from the persistent DB rather than kept in
        # memory — the container restarts on every redeploy (restart: always), and
        # an in-memory tally would silently reset to zero mid-session.
        start_of_day = _now_et().replace(hour=0, minute=0, second=0, microsecond=0)
        return trade_history_db.get_trades(since=start_of_day.astimezone(timezone.utc))

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
                todays = _todays_trades()
                daily_net = sum(t["pnl"] for t in todays)
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
                        trade_count=len(todays),
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

            async def _finalize_close(
                symbol: str,
                mem: dict | None,
                entry_price: float,
                exit_price: float,
                quantity: int,
                exit_reason: str,
                prediction_default: str,
                is_long: bool,
                market_story: str,
            ) -> None:
                # Shared close tail: record the trade + AI memory (only when we have
                # the entry memory) and send the Discord close alert. Factored out of
                # the identical blocks in the stop / trailing / take-profit / reversal /
                # reconciliation exits.
                entry_time_mem = mem.get("entry_time") if mem else None
                if mem:
                    entry_conf = float(mem.get("entry_confidence", 0.0))
                    _record_closed_trade(entry_price, exit_price, quantity, entry_conf,
                                         symbol=symbol, entry_time=entry_time_mem)
                    await news_client.record_trade_memory_async(
                        symbol=symbol,
                        technical_context=str(mem.get("technical_context", market_story)),
                        prediction=str(mem.get("prediction", prediction_default)),
                        outcome=_trade_outcome(entry_price, exit_price),
                    )
                await discord_ui.send_close_alert(
                    symbol=symbol,
                    is_long=is_long,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    entry_time=entry_time_mem,
                    exit_reason=exit_reason,
                    market_story=market_story,
                )

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

                    # No new entries in the pre-close window (15:50+) — everything is
                    # liquidated at 15:50, so a late entry would ride into the close
                    # or hold overnight. Mirrors the backtest.
                    if _is_pre_close():
                        logger.info("%s: BUY skipped (pre-close window).", symbol)
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

                    # Gross-exposure cap — don't open new positions once the book is
                    # already ~fully deployed, so 10% sizing can't over-commit equity.
                    gross_exposure = order_manager.get_gross_position_value()
                    if account_equity > 0.0 and gross_exposure >= MAX_GROSS_EXPOSURE_PCT * account_equity:
                        logger.info(
                            "%s: BUY skipped — open exposure %.0f%% ≥ %.0f%% cap.",
                            symbol, (gross_exposure / account_equity) * 100, MAX_GROSS_EXPOSURE_PCT * 100,
                        )
                        return

                    # Constant-risk sizing: risk RISK_PER_TRADE of equity to the ATR stop,
                    # so a stop-out costs the same whatever the stock's volatility. Capped
                    # at MAX_POSITION_PCT of equity (also the fallback when ATR is 0).
                    max_shares = (account_equity * MAX_POSITION_PCT) / current_price if account_equity > 0.0 else 0.0
                    if account_equity > 0.0 and current_atr > 0.0:
                        risk_shares = (account_equity * RISK_PER_TRADE) / (STOP_ATR_MULT * current_atr)
                        trade_size = max(1, int(min(risk_shares, max_shares)))
                    else:
                        trade_size = max(1, int(max_shares))
                    logger.info(
                        "%s sizing | equity $%.2f | ATR $%.3f | risk %.1f%% | cap %.0f%% | shares → %d",
                        symbol, account_equity, current_atr, RISK_PER_TRADE * 100,
                        MAX_POSITION_PCT * 100, trade_size,
                    )

                    # Enter with a bracket order: a market BUY plus an attached resting
                    # stop at entry − STOP_ATR_MULT × ATR (fixed 2% only if ATR is 0). The
                    # stop activates only when the BUY fills (so it can never rest naked)
                    # and fills near the stop intrabar — matching the backtest's stop model.
                    if current_price > 0.0 and current_atr > 0.0:
                        stop_price = round(current_price - STOP_ATR_MULT * current_atr, 2)
                    elif current_price > 0.0:
                        stop_price = round(current_price * (1 - STOP_LOSS_PCT), 2)
                    else:
                        stop_price = 0.0
                    stop_trade = None
                    if stop_price > 0.0:
                        try:
                            _, stop_trade = await order_manager.place_bracket_buy(
                                symbol, trade_size, stop_price
                            )
                            logger.info("BUY %s x%d with resting stop at $%.2f.",
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
                        "partial_taken": False,
                    }
                    # Report the stop/target actually in force, not a hardcoded %: the
                    # resting stop is at stop_price, TP1 is entry + 3 × entry-ATR.
                    stop_loss = stop_price if stop_price > 0.0 else "N/A"
                    target_price = (current_price + TAKE_PROFIT_ATR_MULT * current_atr) \
                        if (current_price > 0.0 and current_atr > 0.0) else "N/A"
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
                    # failed to place). Normally the broker's resting stop handles the
                    # exit and is reconciled at the top of the polling loop.
                    has_resting_stop = bool(trade_memory and trade_memory.get("stop_order") is not None)
                    mem_atr = float(trade_memory.get("initial_atr", 0.0)) if trade_memory else 0.0
                    if entry_price > 0.0 and current_price > 0.0 and not has_resting_stop:
                        stop_level = (entry_price - STOP_ATR_MULT * mem_atr) if mem_atr > 0.0 \
                            else entry_price * (1 - STOP_LOSS_PCT)
                        if current_price <= stop_level:
                            loss_pct = ((entry_price - current_price) / entry_price) * 100.0
                            sell_quantity = max(1, int(abs(holding_quantity)))
                            logger.warning(
                                "Fallback stop-loss: %s | entry $%.2f → now $%.2f (−%.2f%%) | selling %d.",
                                symbol, entry_price, current_price, loss_pct, sell_quantity,
                            )

                            await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                            last_sell_time[symbol_key] = now

                            stopped_trade_memory = open_trade_memory.pop(symbol_key, None)
                            await _finalize_close(
                                symbol, stopped_trade_memory, entry_price, current_price, sell_quantity,
                                "ATR stop (fallback)", signal_direction, holding_quantity > 0, technical_context,
                            )
                            logger.info("Closed %s x%d — ATR stop (fallback).", symbol, sell_quantity)
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
                    entry_price_mem = float(trade_memory.get("entry_price", 0.0)) if trade_memory else 0.0
                    await _finalize_close(
                        symbol, trade_memory, entry_price_mem, current_price, sell_quantity,
                        "AI bearish signal", signal_direction, holding_quantity > 0, technical_context,
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

                nonlocal eod_recap_sent_date, eod_liquidation_done_date, screened_date
                today_et = _now_et().date()

                # Reset the circuit breaker at the start of each new trading day.
                # Daily P&L needs no reset — it is queried per-day from the DB.
                if eod_recap_sent_date is not None and eod_recap_sent_date != today_et:
                    circuit_breaker["halted"] = False

                # Daily pre-market re-screen — refresh the momentum-leader watchlist
                # once per trading day from 09:00 ET, before the 09:30 open. Gated on
                # Discord being online (when a token is configured) so the bot is up
                # and can post results before the multi-minute scan starts.
                now_et = _now_et()
                discord_online = discord_ui.is_ready() if discord_token else True
                if (screened_date != today_et and now_et.weekday() < 5
                        and now_et.hour >= SCREEN_HOUR_ET and discord_online):
                    screened_date = today_et
                    logger.info("Daily re-screen — scanning for today's momentum leaders...")
                    try:
                        fresh = await screener.screen()
                        if fresh:
                            settings["tickers"] = fresh
                            logger.info("Watchlist updated (%d tickers): %s", len(fresh), fresh)
                            for sym in fresh:
                                await _subscribe_if_new(sym)
                        else:
                            logger.warning("Re-screen returned nothing — keeping current watchlist.")
                        await discord_ui.send_screener_results(
                            screener.last_picks, screener.last_qualified, screener.last_scanned,
                        )
                    except Exception as exc:
                        logger.exception("Daily re-screen failed: %s", exc)

                # End of Day recap — sent once in the 10-minute window after close
                if _is_just_after_close() and eod_recap_sent_date != today_et:
                    logger.info("Market closed — sending end-of-day recap.")
                    stats = _compute_eod_stats(_todays_trades(), order_manager.get_account_equity())
                    await discord_ui.send_eod_recap(stats)
                    eod_recap_sent_date = today_et

                # Pre-close forced liquidation — clear all positions 10 min before close
                if _is_pre_close() and eod_liquidation_done_date != today_et:
                    logger.info("Pre-close — liquidating all positions.")
                    await liquidate_all_positions(reason="End-of-Day")
                    eod_liquidation_done_date = today_et

                scan = {"held": 0, "bull_setup": 0, "hourly_neutral": 0}
                # Snapshot the list so a Discord !add or a re-screen mid-cycle can't
                # disrupt iteration; new tickers are picked up on the next pass.
                # Open positions are always included even if the daily re-screen
                # dropped them from the watchlist — otherwise their stops, targets
                # and trailing exits would stop being evaluated while still held.
                active_tickers = list(dict.fromkeys(
                    list(settings["tickers"]) + list(order_manager.get_all_positions().keys())
                ))
                for symbol in active_tickers:
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
                            was_partial = bool(pending_mem.get("partial_taken"))
                            try:
                                fill_px = float(getattr(stop_trade.orderStatus, "avgFillPrice", 0.0)) or 0.0
                            except Exception:
                                fill_px = 0.0
                            # After TP1 the resting stop sits at breakeven (entry); before
                            # it, at entry − STOP_ATR_MULT × ATR (fixed 2% if ATR unknown).
                            mem_atr = float(pending_mem.get("initial_atr", 0.0))
                            if was_partial:
                                stop_ref = entry_price
                            elif mem_atr > 0.0:
                                stop_ref = entry_price - STOP_ATR_MULT * mem_atr
                            else:
                                stop_ref = entry_price * (1 - STOP_LOSS_PCT)
                            exit_px = fill_px if fill_px > 0.0 else (stop_ref if entry_price > 0.0 else current_price)
                            last_sell_time[symbol_key] = datetime.now(timezone.utc)
                            await _finalize_close(
                                symbol, pending_mem, entry_price, exit_px, closed_qty,
                                ("Breakeven stop (resting order filled)" if was_partial
                                 else "ATR stop (resting order filled)"),
                                "BULLISH", True, str(pending_mem.get("technical_context", "")),
                            )
                            logger.info("Resting stop filled for %s x%d — %s.", symbol, closed_qty,
                                        "breakeven" if was_partial else "ATR stop")
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
                            # After the TP1 scale-out the trail is floored at breakeven so
                            # the runner can never give back into a loss.
                            trailing_stop_level = highest_price_seen - (ATR_TRAIL_MULT * initial_atr)
                            if trade_memory.get("partial_taken"):
                                trailing_stop_level = max(trailing_stop_level, entry_price)
                            if initial_atr > 0.0 and current_price > 0.0 and current_price <= trailing_stop_level:
                                sell_quantity = max(1, int(abs(holding_quantity)))
                                logger.warning("ATR trailing stop hit: %s.", symbol)

                                # Cancel the protective stop before closing so the two can't both fill.
                                stopped_trade_memory = open_trade_memory.pop(symbol_key, None)
                                _cancel_resting_stop(stopped_trade_memory)

                                await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                                last_sell_time[symbol_key] = datetime.now(timezone.utc)
                                await _finalize_close(
                                    symbol, stopped_trade_memory, entry_price, current_price, sell_quantity,
                                    "ATR trailing stop", "BULLISH", holding_quantity > 0, technical_context,
                                )
                                logger.info("Closed %s x%d — ATR trailing stop.", symbol, sell_quantity)
                                continue

                            # Take-profit, gated by the same 25-min minimum hold as the backtest.
                            # TP1 (3×ATR) scales out 60% and raises the stop to breakeven;
                            # TP2 (6×ATR) closes the runner.
                            entry_time_tp = trade_memory.get("entry_time")
                            held_long_enough = (
                                entry_time_tp is not None
                                and datetime.now(timezone.utc) - entry_time_tp >= timedelta(minutes=MIN_HOLD_MINUTES)
                            )
                            partial_taken = bool(trade_memory.get("partial_taken", False))
                            tp_mult = TAKE_PROFIT_2_ATR_MULT if partial_taken else TAKE_PROFIT_ATR_MULT
                            take_profit_level = entry_price + (tp_mult * initial_atr)
                            if (held_long_enough and entry_price > 0.0 and initial_atr > 0.0
                                    and current_price >= take_profit_level):
                                full_qty = max(1, int(abs(holding_quantity)))
                                scale_qty = int(full_qty * SCALE_OUT_PCT)

                                # ── TP1: scale out 60%, keep a runner, stop → breakeven ──
                                if not partial_taken and scale_qty >= 1 and (full_qty - scale_qty) >= 1:
                                    remaining = full_qty - scale_qty
                                    logger.warning("Take-profit 1 hit: %s — scaling out %d/%d.", symbol, scale_qty, full_qty)

                                    await order_manager.execute_trade(symbol, "SELL", scale_qty)
                                    last_sell_time[symbol_key] = datetime.now(timezone.utc)

                                    # Replace the protective stop with a breakeven stop on the runner.
                                    _cancel_resting_stop(trade_memory)
                                    breakeven_stop = None
                                    try:
                                        breakeven_stop = await order_manager.place_stop_order(
                                            symbol, "SELL", remaining, round(entry_price, 2)
                                        )
                                    except Exception as exc:
                                        logger.warning("Could not place breakeven stop for %s: %s", symbol, exc)

                                    trade_memory["partial_taken"] = True
                                    trade_memory["quantity"] = remaining
                                    trade_memory["stop_order"] = breakeven_stop

                                    _record_closed_trade(entry_price, current_price, scale_qty, entry_conf,
                                                         symbol=symbol, entry_time=entry_time_tp)
                                    await discord_ui.send_close_alert(
                                        symbol=symbol,
                                        is_long=holding_quantity > 0,
                                        entry_price=entry_price,
                                        exit_price=current_price,
                                        quantity=scale_qty,
                                        entry_time=entry_time_tp,
                                        exit_reason="Partial take-profit (60%) — stop to breakeven",
                                        market_story=technical_context,
                                    )
                                    logger.info("Partial TP: %s sold %d, runner %d on breakeven stop.", symbol, scale_qty, remaining)
                                    continue

                                # ── TP2 (or position too small to split): close the remainder ──
                                sell_quantity = full_qty
                                logger.warning("Take-profit 2 hit: %s — closing runner.", symbol)
                                stopped_trade_memory = open_trade_memory.pop(symbol_key, None)
                                _cancel_resting_stop(stopped_trade_memory)

                                await order_manager.execute_trade(symbol, "SELL", sell_quantity)
                                last_sell_time[symbol_key] = datetime.now(timezone.utc)
                                await _finalize_close(
                                    symbol, stopped_trade_memory, entry_price, current_price, sell_quantity,
                                    "Take-profit target met (runner)", "BULLISH", holding_quantity > 0, technical_context,
                                )
                                logger.info("Closed %s x%d — take-profit 2.", symbol, sell_quantity)
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

                    # No headlines → the LLM still evaluates the bare technical setup
                    # (same "Not provided" convention it was trained on), so the LLM
                    # rules on every entry even for news-quiet tickers.
                    if not news_items:
                        prediction = await news_client.evaluate_setup(symbol, technical_context)
                        llm_dir = str(prediction.get("direction", "NEUTRAL")).upper()
                        llm_conf = float(prediction.get("confidence", 0.0))

                    # The LLM is the decision maker — no technical-fallback entries.
                    # Technicals qualify the candidate; only a confident BULLISH
                    # verdict from the LLM opens a position (mirrors the backtest).
                    if llm_dir != "BULLISH":
                        logger.info("%s: entry skipped — LLM verdict %s.", symbol, llm_dir)
                        continue
                    if llm_conf < LLM_CONF_THRESHOLD:
                        logger.info("%s: entry skipped — LLM confidence %.2f < %.2f.",
                                    symbol, llm_conf, LLM_CONF_THRESHOLD)
                        continue

                    await _route_trade(
                        symbol=symbol,
                        signal_direction="BULLISH",
                        technical_context=technical_context,
                        current_price=current_price,
                        current_atr=current_atr,
                        confidence=llm_conf,
                    )

                if not settings.get("backtest_mode"):
                    logger.info(
                        "Scan: %d tickers | %d held | %d bullish setups | %d hourly-NEUTRAL (no 1h SMA-50?)",
                        len(active_tickers), scan["held"], scan["bull_setup"], scan["hourly_neutral"],
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
