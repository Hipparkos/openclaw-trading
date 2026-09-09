# OpenClaw

An autonomous day-trading system that screens the US equity market every day,
evaluates candidates with a technical gate and two locally-hosted fine-tuned
language models, and manages entries and exits with ATR-based risk controls —
reporting everything through Discord. Built as a school project.


## Contents

- [How it works](#how-it-works)
- [Architecture](#architecture)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running it](#running-it)
- [Discord commands](#discord-commands)
- [Project layout](#project-layout)
- [Status & limitations](#status--limitations)

## How it works

The system runs as a daily cycle with distinct stages.

1. **Screen** (09:00 ET). Downloads the official NASDAQ Trader listing files
   and builds its universe from common stocks on NASDAQ, NYSE and NYSE Arca —
   funds, preferred shares, warrants and debt instruments excluded. Each
   symbol is scored on turnover, volatility, three-month momentum, trend, and
   distance from its recent high; a stock whose entire move came from a single
   exceptional day is rejected as a liquidity event rather than a trend. The
   strongest ~25 of roughly 7,000 scanned names become the day's watchlist.
   ([`screener.py`](src/data/screener.py))

2. **Qualify** (every 5 minutes, market hours). A candidate must show a clear
   bullish hourly trend, volume at 80%+ of its recent average, and a net score
   of at least 3 (of 4) from RSI, MACD, VWAP and a short moving average. This
   stage opens no positions — it only decides which setups are worth an LLM
   call. Backtest and live share this exact function, so the two can't
   silently diverge. ([`logic.py`](src/strategy/logic.py))

3. **Decide.** Two language models running locally via Ollama — **Llama 3.2**
   as the first analyst, a fine-tuned **Qwen** as reviewer — evaluate the
   setup together with any news and the three most similar past setups for
   that same symbol from the trade-memory database. A position opens only on
   a BULLISH verdict at ≥0.70 confidence. ([`news_client.py`](src/data/news_client.py))

4. **Execute.** Entry is a bracket order — a market buy with an attached
   broker-side stop that only activates once the buy fills, so a naked stop
   can never rest. Position size is constant-risk: sized so a stop-out always
   costs the same fraction of equity, whatever the stock's volatility.
   ([`order_manager.py`](src/execution/order_manager.py))

5. **Manage.** Stop at 2×ATR, 60% scaled out at 3×ATR (stop then moves to
   breakeven), the rest closed at 6×ATR, trailing stop at 2.5×ATR. Every
   position is flattened by 15:50 ET — nothing is held overnight.
   ([`main.py`](src/main.py))

6. **Report.** Every closed trade is written to SQLite; Discord gets
   execution alerts, close alerts, an end-of-day recap, and a live-updating
   dashboard message. ([`controller.py`](src/discord_bot/controller.py),
   [`dashboard.py`](src/discord_bot/dashboard.py))

## Architecture

```
                    ┌──────────────┐
  NASDAQ Trader ───▶│   Screener   │──▶ today's ~25 tickers
                    └──────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────┐
│                  Main loop                          │
│                                                     │
│   IBKR bars ──▶ Technical gate ──▶ News + LLMs     │
│   (5m/1h)      (logic.py, shared    (Llama 3.2 →    │
│                 w/ backtest)         Qwen review)   │
│                       │                    │        │
│                       ▼                    ▼        │
│                 no candidate          BULLISH ≥0.70 │
│                                             │       │
│                                             ▼       │
│                                    Bracket order,   │
│                                    ATR stop/targets │
└───────────────────┬───────────────────┬─────────────┘
                     ▼                   ▼
              trade_history.db   Discord (alerts,
              trade_memory.db     EoD recap,
                                   live dashboard)
```

`ib-gateway`, `ollama` and the `openclaw` agent run as separate containers
(`docker-compose.yml`); the agent connects to IBKR over the API port and to
Ollama over HTTP.

## Setup

**Requirements:** Docker and Docker Compose. An Interactive Brokers account
(paper trading is the default and strongly recommended) with IB Gateway
credentials. A Discord bot token if you want the reporting/dashboard side —
the trading loop itself doesn't require Discord.

1. Clone the repo and copy the environment template:

   ```bash
   cp .env.example .env
   ```

2. Fill in `.env` — see [Configuration](#configuration) for what each value
   means.

3. Copy the example configs and adjust the watchlist / risk limits if needed:

   ```bash
   cp config/settings.example.yaml config/settings.yaml
   cp config/risk_limits.example.yaml config/risk_limits.yaml
   ```

4. Build and start:

   ```bash
   docker compose up -d --build
   ```

## Configuration

| Variable | In | Meaning |
|---|---|---|
| `IB_USER`, `IB_PASSWORD` | `.env` | IBKR login used by `ib-gateway`. Paper and live trading share the same login — `TRADING_MODE` in `docker-compose.yml` selects which. |
| `IB_ACCOUNT` | `.env` | Your IB account ID (`DU...` for paper). |
| `MARKETAUX_API_TOKEN` | `.env` | News headlines for the LLM decision step. |
| `DISCORD_TOKEN`, `DISCORD_CHANNEL_ID` | `.env` | Bot token and the channel alerts/dashboard post to. Optional — the bot still trades without Discord configured. |
| `config/settings.yaml` | file | IBKR connection details. |
| `config/risk_limits.yaml` | file | Position sizing and stop/target parameters. |

Risk and exit constants (`STOP_ATR_MULT`, `TAKE_PROFIT_ATR_MULT`,
`RISK_PER_TRADE`, `DAILY_LOSS_LIMIT_PCT`, …) live as class constants at the
top of [`main.py`](src/main.py) and are mirrored exactly in
[`engine.py`](src/backtests/engine.py) so live and backtest never quietly
diverge. Change them in both places together.

## Running it

**Live / paper trading** starts automatically once the containers are up.
It trades only during regular market hours and holds nothing overnight.

## Discord commands

| Command | Does |
|---|---|
| `!dashboard` | Post the live-updating panel (Positions / Decisions / Watchlist) — refreshes itself every 30s. |
| `!stats` | YTD / MTD / 1-week performance chart. |
| `!positions`, `!equity` | Current holdings and account equity. |
| `!screener` | Force a fresh momentum scan now. |
| `!add TICKER`, `!remove TICKER` | Adjust today's watchlist manually. |
| `!halt`, `!resume` | Manually trip or clear the circuit breaker. |
| `!closeall` | Liquidate every open position. |

## Project layout

```
src/
  main.py                  live trading loop — the orchestrator
  strategy/
    logic.py                technical alignment gate (shared with backtest)
    indicators.py            indicator calculation (RSI, MACD, VWAP, ATR, …)
  data/
    screener.py              daily momentum universe scan
    ibkr_client.py            market data subscriptions
    news_client.py            LLM decision calls + trade-memory recall
    trade_db.py               persistent trade history (SQLite)
  execution/
    order_manager.py          order placement, bracket stops, position queries
  backtests/
    engine.py                 replay engine — mirrors main.py's strategy exactly
  discord_bot/
    controller.py             commands, alerts, EoD recap
    dashboard.py               live-updating dashboard panel
    stats_chart.py             matplotlib performance charts
config/                       settings.yaml, risk_limits.yaml (gitignored)
data/
  database/                   trade_history.db, trade_memory.db (gitignored)
  logs/                       (gitignored)
```
