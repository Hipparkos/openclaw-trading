import asyncio
import os
import logging
from datetime import datetime, timezone
from typing import Any
import discord
from discord.ext import commands

class TradeApprovalView(discord.ui.View):
    def __init__(self, order_manager, symbol, side):
        super().__init__(timeout=300)
        self.order_manager = order_manager
        self.symbol = symbol
        self.side = side

    @discord.ui.button(label="APPROVE", style=discord.ButtonStyle.green)
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(view=self)
        
        await interaction.followup.send(f"Executing {self.side} order for {self.symbol}")
        try:
            await self.order_manager.execute_trade(self.symbol, self.side)
        except Exception as e:
            await interaction.followup.send(f"Transmission failure: {e}")

    @discord.ui.button(label="REJECT", style=discord.ButtonStyle.red)
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(view=self)
        await interaction.followup.send("Signal rejected. Order cancelled.")


class TradingCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.order_manager = bot.order_manager

    def _format_currency(self, value):
        try:
            return f"${float(value):,.2f}"
        except (TypeError, ValueError):
            return "$0.00"

    @commands.command(name="positions")
    async def positions(self, ctx):
        try:
            active_positions = []
            for item in self.order_manager.ib.portfolio():
                symbol = getattr(item.contract, "symbol", "UNKNOWN")
                shares = float(getattr(item, "position", 0.0) or 0.0)
                if shares == 0.0:
                    continue

                average_cost = getattr(item, "averageCost", 0.0)
                market_value = getattr(item, "marketValue", 0.0)
                active_positions.append(
                    f"{symbol} | Shares: {shares:,.0f} | Avg Cost: {self._format_currency(average_cost)} | Market Value: {self._format_currency(market_value)}"
                )

            embed = discord.Embed(title="Active Positions", color=0x2F3136)
            if active_positions:
                embed.description = "\n".join(active_positions)
            else:
                embed.description = "No active positions held."

            await ctx.send(embed=embed)
        except Exception as exc:
            logging.error("Failed to retrieve positions: %s", exc)
            embed = discord.Embed(title="Active Positions", description="Unable to retrieve positions right now.", color=0x992D22)
            await ctx.send(embed=embed)

    @commands.command(name="equity")
    async def equity(self, ctx):
        try:
            account_equity = self.order_manager.get_account_equity()
            embed = discord.Embed(title="Account Equity", color=0x2F3136)
            embed.description = self._format_currency(account_equity)
            await ctx.send(embed=embed)
        except Exception as exc:
            logging.error("Failed to retrieve account equity: %s", exc)
            embed = discord.Embed(title="Account Equity", description="Unable to retrieve account equity right now.", color=0x992D22)
            await ctx.send(embed=embed)

    @commands.command(name="backteston")
    async def backteston(self, ctx):
        """Suspend live trading and run a 90-day historical backtest."""
        if not callable(getattr(self.bot, "on_backtest_start", None)):
            await ctx.send("Backtest engine not available yet.")
            return

        if self.bot.settings and self.bot.settings.get("backtest_mode"):
            await ctx.send("A backtest is already running. Use `!backtestoff` to stop it first.")
            return

        embed = discord.Embed(
            title="BACKTEST INITIATED",
            description=(
                "Live trading **suspended**.\n"
                "Fetching 90 days of historical data and replaying strategy...\n"
                "This may take 2–5 minutes depending on ticker count."
            ),
            color=0x00AAFF,
        )
        await ctx.send(embed=embed)

        asyncio.create_task(self.bot.on_backtest_start(ctx.channel.id))

    @commands.command(name="backtestoff")
    async def backtestoff(self, ctx):
        """Clear backtest state and resume live IBKR execution."""
        if callable(getattr(self.bot, "on_backtest_stop", None)):
            self.bot.on_backtest_stop()

        embed = discord.Embed(
            title="LIVE TRADING RESUMED",
            description="Backtest mode cleared. Bot is back to live execution.",
            color=0x00FF00,
        )
        await ctx.send(embed=embed)

    @commands.command(name="ppocache")
    async def ppocache(self, ctx, *, tickers: str = None):
        """Generate the LLM cache used for PPO training."""
        if not tickers:
            await ctx.send("Usage: `!ppocache NVDA AAPL MSFT` (space or comma separated)")
            return
        if not callable(getattr(self.bot, "on_ppo_cache", None)):
            await ctx.send("PPO cache generator not available.")
            return
        symbols = [s.strip().upper() for s in tickers.replace(",", " ").split() if s.strip()]
        embed = discord.Embed(
            title="PPO CACHE GENERATION STARTED",
            description=(
                f"Tickers: **{', '.join(symbols)}**\n"
                "Calling the LLM for each candidate bar — takes **10–60 min** depending on bar count.\n"
                "You'll get a message here when it's done."
            ),
            color=0xFFAA00,
        )
        await ctx.send(embed=embed)
        asyncio.create_task(self.bot.on_ppo_cache(symbols, ctx.channel.id))

    @commands.command(name="ppotrain")
    async def ppotrain(self, ctx, *, tickers: str = None):
        """Generate LLM cache and train the PPO agent end-to-end."""
        if not tickers:
            await ctx.send("Usage: `!ppotrain NVDA AAPL MSFT` (space or comma separated)")
            return
        if not callable(getattr(self.bot, "on_ppo_train", None)):
            await ctx.send("PPO trainer not available.")
            return
        symbols = [s.strip().upper() for s in tickers.replace(",", " ").split() if s.strip()]
        embed = discord.Embed(
            title="PPO TRAINING STARTED",
            description=(
                f"Tickers: **{', '.join(symbols)}**\n"
                "**Step 1** — LLM cache generation\n"
                "**Step 2** — Fetch historical bars from IBKR\n"
                "**Step 3** — Train PPO agent (500k steps)\n\n"
                "Total time: **1–3 hours**. You'll get a notification when done."
            ),
            color=0xFF6600,
        )
        await ctx.send(embed=embed)
        asyncio.create_task(self.bot.on_ppo_train(symbols, ctx.channel.id))

    @commands.command(name="closeall")
    async def closeall(self, ctx):
        if not callable(getattr(self.bot, "on_manual_sell", None)):
            await ctx.send("Manual liquidation is not available right now.")
            return

        embed = discord.Embed(
            title="MANUAL LIQUIDATION INITIATED",
            description="Closing all open positions...",
            color=0xFF6600,
        )
        await ctx.send(embed=embed)

        try:
            results = await self.bot.on_manual_sell()
            if results:
                embed = discord.Embed(title="Liquidation Complete", color=0xFF6600)
                embed.description = "\n".join(results)
            else:
                embed = discord.Embed(
                    title="No Open Positions",
                    description="Portfolio is already flat.",
                    color=0x2F3136,
                )
            await ctx.send(embed=embed)
        except Exception as exc:
            logging.error("!sellall failed: %s", exc)
            await ctx.send("Liquidation encountered an error. Check the logs.")

    @commands.command(name="add")
    async def add_ticker(self, ctx, *, tickers: str = None):
        if not self.bot.settings:
            await ctx.send("Bot settings not available yet.")
            return
        if not tickers:
            await ctx.send("Usage: `!add AAPL` or `!add AAPL, NOW, MSFT`")
            return

        symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
        if not symbols:
            await ctx.send("Usage: `!add AAPL` or `!add AAPL, NOW, MSFT`")
            return

        watchlist: list = self.bot.settings.setdefault("tickers", [])
        added = []
        skipped = []

        for symbol in symbols:
            if symbol in watchlist:
                skipped.append(symbol)
            else:
                watchlist.append(symbol)
                added.append(symbol)

        embed = discord.Embed(
            title="Watchlist Updated",
            color=0x00FF00 if added else 0xAAAAAA,
        )
        if added:
            embed.add_field(name="Added", value=" | ".join(f"`{s}`" for s in added), inline=False)
        if skipped:
            embed.add_field(name="Already on watchlist", value=" | ".join(f"`{s}`" for s in skipped), inline=False)
        embed.add_field(name="Current Watchlist", value=" | ".join(watchlist), inline=False)

        if added and callable(getattr(self.bot, "on_add_ticker", None)):
            for symbol in added:
                asyncio.create_task(self.bot.on_add_ticker(symbol))
            embed.add_field(
                name="Data Feed",
                value=f"Fetching data for {len(added)} ticker(s), ready in ~30s",
                inline=False,
            )

        await ctx.send(embed=embed)

    @commands.command(name="remove")
    async def remove_ticker(self, ctx, *, tickers: str = None):
        if not self.bot.settings:
            await ctx.send("Bot settings not available yet.")
            return
        if not tickers:
            await ctx.send("Usage: `!remove AAPL` or `!remove AAPL, NOW, MSFT`")
            return

        symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
        if not symbols:
            await ctx.send("Usage: `!remove AAPL` or `!remove AAPL, NOW, MSFT`")
            return

        watchlist: list = self.bot.settings.get("tickers", [])
        removed = []
        not_found = []

        for symbol in symbols:
            if symbol in watchlist:
                watchlist.remove(symbol)
                removed.append(symbol)
            else:
                not_found.append(symbol)

        embed = discord.Embed(
            title="Watchlist Updated",
            color=0xFF6600 if removed else 0xAAAAAA,
        )
        if removed:
            embed.add_field(name="Removed", value=" | ".join(f"`{s}`" for s in removed), inline=False)
        if not_found:
            embed.add_field(name="Not on watchlist", value=" | ".join(f"`{s}`" for s in not_found), inline=False)
        remaining = " | ".join(watchlist) if watchlist else "_(empty)_"
        embed.add_field(name="Current Watchlist", value=remaining, inline=False)

        await ctx.send(embed=embed)

    @commands.command(name="status")
    async def status(self, ctx):
        try:
            is_connected = self.order_manager.ib.isConnected()
            gateway_status = "Connected" if is_connected else "Disconnected"

            embed = discord.Embed(title="OpenClaw Status", color=0x2F3136)
            embed.add_field(name="Gateway Status", value=gateway_status, inline=False)
            embed.add_field(name="Active AI Engine", value="OpenClaw V2", inline=False)

            await ctx.send(embed=embed)
        except Exception as exc:
            logging.error("Failed to retrieve gateway status: %s", exc)
            embed = discord.Embed(title="OpenClaw Status", description="Unable to retrieve status right now.", color=0x992D22)
            await ctx.send(embed=embed)

    @commands.command(name="screener")
    async def screener(self, ctx):
        """Run the volume gainer screener and update trading tickers."""
        if not self.bot.screener or not self.bot.settings:
            await ctx.send("❌ Screener not available. Bot may still be initializing.")
            return

        embed = discord.Embed(
            title="🔍 VOLUME GAINER SCREENER",
            description="Scanning market for top volume gainers...",
            color=0x00AAFF,
        )
        status_msg = await ctx.send(embed=embed)

        try:
            screened_tickers = await self.bot.screener.screen_volume_gainers()

            if not screened_tickers:
                embed = discord.Embed(
                    title="🔍 SCREENER RESULTS",
                    description="❌ No suitable stocks found. Market may be closed or no gainers meet criteria.",
                    color=0xFF6600,
                )
                await status_msg.edit(embed=embed)
                return

            # Update bot's active tickers
            self.bot.settings["tickers"] = screened_tickers
            logging.info(f"Discord !screener command updated tickers: {screened_tickers}")

            # Subscribe IBKR data feeds for any tickers not already buffered
            if callable(getattr(self.bot, "on_add_ticker", None)):
                for t in screened_tickers:
                    asyncio.create_task(self.bot.on_add_ticker(t))

            # Show results
            embed = discord.Embed(
                title="🔍 SCREENER RESULTS",
                description=f"✅ Found {len(screened_tickers)} suitable stocks for day trading",
                color=0x00FF00,
            )
            embed.add_field(
                name="Updated Trading Tickers",
                value=" | ".join(screened_tickers),
                inline=False,
            )
            embed.add_field(
                name="Status",
                value="Bot will now trade only these tickers until next manual scan or daily reset.",
                inline=False,
            )
            await status_msg.edit(embed=embed)

        except Exception as exc:
            logging.error("!screener command failed: %s", exc)
            embed = discord.Embed(
                title="🔍 SCREENER ERROR",
                description=f"❌ Screener encountered an error: {str(exc)}",
                color=0x992D22,
            )
            await status_msg.edit(embed=embed)


class OpenClawDiscord(commands.Bot):
    def __init__(self, order_manager):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.order_manager = order_manager
        self.on_manual_sell = None    # set by main after startup
        self.on_add_ticker = None     # set by main after startup
        self.on_backtest_start = None # set by main after startup
        self.on_backtest_stop = None  # set by main after startup
        self.on_ppo_cache = None      # set by main after startup
        self.on_ppo_train = None      # set by main after startup
        self.screener = None          # set by main after startup
        self.settings = None          # set by main after startup
        
        try:
            self.channel_id = int(os.getenv("DISCORD_CHANNEL_ID", 0))
        except (ValueError, TypeError):
            logging.error("DISCORD_CHANNEL_ID in your .env file is missing or not a valid number.")
            self.channel_id = None

    async def setup_hook(self):
        await self.add_cog(TradingCommands(self))

    async def on_ready(self):
        logging.info(f"OpenClaw UI online. Hooked as: {self.user}")

    async def _get_target_channel(self):
        if not self.channel_id:
            return None

        channel = self.get_channel(self.channel_id)
        if channel is not None:
            return channel

        try:
            return await self.fetch_channel(self.channel_id)
        except Exception as exc:
            logging.error("Unable to resolve Discord channel %s: %s", self.channel_id, exc)
            return None

    async def send_trade_signal(self, symbol, market_story, llm_prediction):
        if not self.channel_id:
            logging.error("Trade signal suppressed: Missing target Discord Channel ID configuration.")
            return
        channel = await self._get_target_channel()
        if channel is None:
            logging.error("Trade signal suppressed: Discord channel could not be resolved.")
            return


        is_bullish = llm_prediction.upper() == "UP"
        embed = discord.Embed(
            title=f"SIGNAL DETECTED: {symbol}", 
            color=0x00ff00 if is_bullish else 0xff0000
        )
        embed.add_field(name="Llama 3.2 Prediction", value=f"**{llm_prediction}**", inline=False)
        embed.add_field(name="The Market Story", value=f"```{market_story}```", inline=False)

        side = "BUY" if is_bullish else "SELL"
        view = TradeApprovalView(self.order_manager, symbol, side)
        
        await channel.send(embed=embed, view=view)

    async def send_eod_recap(self, stats: dict) -> None:
        channel = await self._get_target_channel()
        if channel is None:
            logging.error("EoD recap suppressed: Discord channel could not be resolved.")
            return

        net_pnl = stats.get("net_pnl", 0.0)
        pnl_sign = "+" if net_pnl >= 0 else ""
        color = 0x00FF00 if net_pnl >= 0 else 0xFF0000

        embed = discord.Embed(title="End of Day Recap", color=color)

        embed.add_field(name="Net P&L", value=f"${pnl_sign}{net_pnl:,.2f}", inline=True)
        embed.add_field(name="Account Liquidity", value=f"${stats.get('account_equity', 0.0):,.2f}", inline=True)
        embed.add_field(name="​", value="​", inline=True)

        total = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)
        win_ratio = (wins / total * 100) if total > 0 else 0.0
        embed.add_field(name="Total Trades", value=str(total), inline=True)
        embed.add_field(name="Win Ratio", value=f"{win_ratio:.1f}%", inline=True)
        embed.add_field(name="​", value="​", inline=True)

        embed.add_field(name="Avg Win", value=f"${stats.get('avg_win', 0.0):,.2f}", inline=True)
        embed.add_field(name="Avg Loss", value=f"${stats.get('avg_loss', 0.0):,.2f}", inline=True)
        embed.add_field(name="​", value="​", inline=True)

        embed.add_field(name="Largest Win", value=f"${stats.get('largest_win', 0.0):,.2f}", inline=True)
        embed.add_field(name="Largest Loss", value=f"${stats.get('largest_loss', 0.0):,.2f}", inline=True)
        embed.add_field(name="​", value="​", inline=True)

        embed.add_field(name="AI Confidence — Wins", value=f"{stats.get('avg_confidence_wins', 0.0):.2f}", inline=True)
        embed.add_field(name="AI Confidence — Losses", value=f"{stats.get('avg_confidence_losses', 0.0):.2f}", inline=True)

        await channel.send(embed=embed)

    async def send_close_alert(
        self,
        symbol: str,
        is_long: bool,
        entry_price: float,
        exit_price: float,
        quantity: int,
        entry_time: datetime | None,
        exit_reason: str,
        market_story: str,
    ) -> None:
        channel = await self._get_target_channel()
        if channel is None:
            return

        direction_label = "CLOSED LONG" if is_long else "CLOSED SHORT"

        if is_long:
            pnl_dollars = (exit_price - entry_price) * quantity
            pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0.0
        else:
            pnl_dollars = (entry_price - exit_price) * quantity
            pnl_pct = ((entry_price - exit_price) / entry_price * 100) if entry_price > 0 else 0.0

        is_win = pnl_dollars >= 0
        color = 0x00FF00 if is_win else 0xFF0000
        pnl_sign = "+" if pnl_dollars >= 0 else ""

        if entry_time is not None:
            try:
                delta = datetime.now(timezone.utc) - entry_time
                total_seconds = int(delta.total_seconds())
                hours, remainder = divmod(max(total_seconds, 0), 3600)
                minutes = remainder // 60
                duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            except Exception:
                duration_str = "Unknown"
        else:
            duration_str = "Unknown"

        embed = discord.Embed(title=f"{direction_label}  —  {symbol}", color=color)
        embed.add_field(name="Realized P&L", value=f"`{pnl_sign}${pnl_dollars:,.2f}  ({pnl_sign}{pnl_pct:.2f}%)`", inline=False)
        embed.add_field(name="Entry Price", value=f"${entry_price:,.2f}", inline=True)
        embed.add_field(name="Exit Price", value=f"${exit_price:,.2f}", inline=True)
        embed.add_field(name="Duration", value=duration_str, inline=True)
        embed.add_field(name="Exit Trigger", value=exit_reason, inline=True)
        embed.add_field(name="Quantity", value=f"{quantity:,}", inline=True)
        embed.add_field(name="​", value="​", inline=True)
        embed.add_field(name="Market Context", value=f"```{market_story}```", inline=False)

        await channel.send(embed=embed)

    async def send_execution_alert(self, symbol, action, confidence, market_story, entry_price, target_price, stop_loss, quantity):
        if not self.channel_id:
            logging.error("Execution alert suppressed: Missing target Discord Channel ID configuration.")
            return

        channel = await self._get_target_channel()
        if channel is None:
            logging.error("Execution alert suppressed: Discord channel could not be resolved.")
            return

        color = 0x00ff00 if str(action).upper() == "BUY" else 0xff0000
        embed = discord.Embed(
            title=f"EXECUTION ALERT: {symbol}",
            color=color,
        )
        def _format_currency(value):
            try:
                return f"${float(value):,.2f}"
            except (TypeError, ValueError):
                return str(value)

        embed.add_field(name="Action", value=str(action).upper(), inline=True)
        embed.add_field(name="Confidence", value=f"{float(confidence):.2f}", inline=True)
        embed.add_field(name="Entry Price", value=_format_currency(entry_price), inline=True)
        embed.add_field(name="Target Price", value=_format_currency(target_price), inline=True)
        embed.add_field(name="Stop Loss", value=_format_currency(stop_loss), inline=True)
        embed.add_field(name="Quantity", value=f"{int(quantity):,}", inline=True)
        embed.add_field(name="Execution Rationale", value=f"```{market_story}```", inline=False)

        await channel.send(embed=embed)

    async def send_backtest_result(self, result: Any, channel_id: int | None = None) -> None:
        if channel_id:
            try:
                channel = self.get_channel(channel_id) or await self.fetch_channel(channel_id)
            except Exception:
                channel = await self._get_target_channel()
        else:
            channel = await self._get_target_channel()

        if channel is None:
            logging.error("Backtest result suppressed: channel not found.")
            return

        # ── Color: green if Sharpe > 1.0 and positive return, amber if marginal, red if poor ──
        if result.sharpe_ratio >= 1.0 and result.total_return > 0:
            color = 0x00C851      # green
            verdict = "VIABLE"
        elif result.sharpe_ratio >= 0.5 or result.total_return > 0:
            color = 0xFF8800      # amber
            verdict = "MARGINAL"
        else:
            color = 0xCC0000      # red
            verdict = "UNDERPERFORMING"

        tickers_str = " · ".join(result.tickers) if result.tickers else "N/A"
        pnl_dollars = result.final_equity - result.start_equity
        pnl_sign = "+" if pnl_dollars >= 0 else ""
        ret_sign = "+" if result.total_return >= 0 else ""

        embed = discord.Embed(
            title=f"BACKTEST RESULTS  —  {result.duration_days}-Day Replay  [{verdict}]",
            description=(
                f"**Tickers:** {tickers_str}\n"
                f"**Starting Equity:** `${result.start_equity:,.2f}`   →   "
                f"**Final Equity:** `${result.final_equity:,.2f}`  "
                f"(`{pnl_sign}${pnl_dollars:,.2f}`)\n"
                f"⚠️ *Signal: LLM (technical indicators only — no news for historical replay)*"
            ),
            color=color,
        )

        # ── Diagnostic section — only shown when 0 trades produced ──
        if result.total_trades == 0:
            bars_report = result.bars_fetched if hasattr(result, "bars_fetched") else {}
            signals_report = result.signals_fired if hasattr(result, "signals_fired") else {}

            total_bars = sum(bars_report.values())
            total_signals = sum(signals_report.values())

            if total_bars == 0:
                diag_title = "❌ DATA FETCH FAILED"
                diag_body = (
                    "IBKR returned **0 bars** for every ticker.\n"
                    "**Most likely cause:** IBKR pacing limit hit. The live bot already\n"
                    "holds streaming subscriptions (up to 48 open connections). Adding\n"
                    "backtest requests pushed over the 60-req/10-min limit.\n\n"
                    "**Fix:** Wait 10 minutes, then retry `!backteston`. The new\n"
                    "engine now uses 12-second gaps between symbols to avoid this."
                )
            elif total_signals == 0:
                diag_title = "⚠️ BARS FETCHED — NO SIGNALS FIRED"
                fetch_lines = "\n".join(
                    f"`{sym}`: {n:,} bars" for sym, n in bars_report.items()
                )
                diag_body = (
                    f"Data was fetched successfully:\n{fetch_lines}\n\n"
                    "But no signals passed the entry threshold. This can happen when\n"
                    "indicator alignment (RSI + MACD + VWAP + SMA5 + 1h trend) rarely\n"
                    "reaches 2-out-of-5 agreement. Check docker logs for signal scores."
                )
            else:
                diag_title = "⚠️ SIGNALS FIRED — NO TRADES RECORDED"
                fetch_lines = "\n".join(
                    f"`{sym}`: {bars_report.get(sym, 0):,} bars, "
                    f"{signals_report.get(sym, 0)} signals"
                    for sym in bars_report
                )
                diag_body = (
                    f"{fetch_lines}\n\n"
                    "Signals fired but no entries recorded — check the confidence\n"
                    "threshold and cooldown logic in engine.py."
                )

            embed.add_field(name="​", value=f"**— {diag_title} —**", inline=False)
            embed.add_field(name="Diagnostics", value=diag_body, inline=False)
            embed.set_footer(text="Check docker logs for per-ticker bar counts and signal scores.")
            await channel.send(embed=embed)
            return

        # ── Section 1: Return Metrics ──
        embed.add_field(name="​", value="**— RETURN METRICS —**", inline=False)
        embed.add_field(
            name="Total Return",
            value=f"`{ret_sign}{result.total_return * 100:.2f}%`",
            inline=True,
        )
        embed.add_field(
            name="Annualised Return",
            value=f"`{ret_sign}{result.annualised_return * 100:.2f}%`",
            inline=True,
        )
        embed.add_field(name="​", value="​", inline=True)

        # ── Section 2: Risk & Drawdown ──
        embed.add_field(name="​", value="**— RISK & DRAWDOWN —**", inline=False)

        sharpe_icon = "✅" if result.sharpe_ratio >= 1.0 else ("⚠️" if result.sharpe_ratio >= 0.5 else "❌")
        embed.add_field(
            name=f"Sharpe Ratio {sharpe_icon}",
            value=f"`{result.sharpe_ratio:.3f}`",
            inline=True,
        )
        sortino_icon = "✅" if result.sortino_ratio >= 1.0 else ("⚠️" if result.sortino_ratio >= 0.5 else "❌")
        embed.add_field(
            name=f"Sortino Ratio {sortino_icon}",
            value=f"`{result.sortino_ratio:.3f}`",
            inline=True,
        )
        embed.add_field(name="​", value="​", inline=True)

        dd_icon = "✅" if result.max_drawdown < 0.10 else ("⚠️" if result.max_drawdown < 0.20 else "❌")
        embed.add_field(
            name=f"Max Drawdown {dd_icon}",
            value=f"`{result.max_drawdown * 100:.2f}%`",
            inline=True,
        )
        embed.add_field(
            name="Avg Drawdown Duration",
            value=f"`{result.avg_drawdown_duration_days:.1f} days`",
            inline=True,
        )
        embed.add_field(name="​", value="​", inline=True)

        # ── Section 3: Trade Execution ──
        embed.add_field(name="​", value="**— TRADE EXECUTION —**", inline=False)

        wr_icon = "✅" if result.win_rate >= 0.50 else "⚠️"
        pf_icon = "✅" if result.profit_factor >= 1.5 else ("⚠️" if result.profit_factor >= 1.0 else "❌")
        exp_icon = "✅" if result.expectancy > 0 else "❌"
        pf_str = f"`{result.profit_factor:.2f}`" if result.profit_factor != float("inf") else "`∞`"

        embed.add_field(name="Total Trades", value=f"`{result.total_trades}`", inline=True)
        embed.add_field(
            name=f"Win Rate {wr_icon}",
            value=f"`{result.win_rate * 100:.1f}%`",
            inline=True,
        )
        embed.add_field(name="​", value="​", inline=True)

        embed.add_field(
            name="Avg Win",
            value=f"`+${result.avg_win:,.2f}`",
            inline=True,
        )
        embed.add_field(
            name="Avg Loss",
            value=f"`-${abs(result.avg_loss):,.2f}`",
            inline=True,
        )
        embed.add_field(
            name="Biggest Win",
            value=f"`+${result.biggest_win:,.2f}`",
            inline=True,
        )
        embed.add_field(
            name="Biggest Loss",
            value=f"`-${abs(result.biggest_loss):,.2f}`",
            inline=True,
        )
        embed.add_field(name="​", value="​", inline=True)

        embed.add_field(
            name=f"Profit Factor {pf_icon}",
            value=pf_str,
            inline=True,
        )
        embed.add_field(
            name=f"Expectancy / Trade {exp_icon}",
            value=f"`{'+' if result.expectancy >= 0 else ''}${result.expectancy:,.2f}`",
            inline=True,
        )
        embed.add_field(
            name="Total Commissions",
            value=f"`-${result.total_commissions:,.2f}`",
            inline=True,
        )
        embed.add_field(name="​", value="​", inline=True)

        embed.set_footer(text=(
            "Fill price = next bar open  ·  $1.50/trade commission applied  ·  "
            "Signal: technical alignment only  ·  Past results ≠ future performance"
        ))

        await channel.send(embed=embed)