import asyncio
import os
import logging
from datetime import datetime, timezone
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
    async def add_ticker(self, ctx, ticker: str = None):
        if not self.bot.settings:
            await ctx.send("Bot settings not available yet.")
            return
        if not ticker:
            await ctx.send("Usage: `!add TICKER` — e.g. `!add AAPL`")
            return

        symbol = ticker.upper().strip()
        tickers: list = self.bot.settings.setdefault("tickers", [])

        if symbol in tickers:
            embed = discord.Embed(
                title="Already Watching",
                description=f"`{symbol}` is already on the watchlist.",
                color=0xAAAAAA,
            )
        else:
            tickers.append(symbol)
            embed = discord.Embed(
                title="Ticker Added",
                description=f"`{symbol}` added to the watchlist.",
                color=0x00FF00,
            )
            embed.add_field(name="Current Watchlist", value=" | ".join(tickers), inline=False)

            if callable(getattr(self.bot, "on_add_ticker", None)):
                asyncio.create_task(self.bot.on_add_ticker(symbol))
                embed.add_field(name="Data Feed", value="Fetching historical data, ready in ~30s", inline=False)

        await ctx.send(embed=embed)

    @commands.command(name="remove")
    async def remove_ticker(self, ctx, ticker: str = None):
        if not self.bot.settings:
            await ctx.send("Bot settings not available yet.")
            return
        if not ticker:
            await ctx.send("Usage: `!remove TICKER` — e.g. `!remove AAPL`")
            return

        symbol = ticker.upper().strip()
        tickers: list = self.bot.settings.get("tickers", [])

        if symbol not in tickers:
            embed = discord.Embed(
                title="Not Found",
                description=f"`{symbol}` is not on the watchlist.",
                color=0xAAAAAA,
            )
        else:
            tickers.remove(symbol)
            embed = discord.Embed(
                title="Ticker Removed",
                description=f"`{symbol}` removed from the watchlist.",
                color=0xFF6600,
            )
            remaining = " | ".join(tickers) if tickers else "_(empty)_"
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
        self.on_manual_sell = None   # set by main after startup
        self.on_add_ticker = None    # set by main after startup
        self.screener = None         # set by main after startup
        self.settings = None         # set by main after startup
        
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