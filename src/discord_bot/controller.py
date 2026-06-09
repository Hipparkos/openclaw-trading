import os
import logging
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
            for position in self.order_manager.ib.positions():
                shares = float(getattr(position, "position", 0.0) or 0.0)
                if shares == 0.0:
                    continue

                symbol = getattr(position.contract, "symbol", "UNKNOWN")
                average_cost = getattr(position, "avgCost", 0.0)
                active_positions.append(
                    f"**{symbol}** | Shares: {shares:,.2f} | Average Cost: {self._format_currency(average_cost)}"
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


class OpenClawDiscord(commands.Bot):
    def __init__(self, order_manager):
        intents = discord.Intents.default()
        intents.message_content = True  
        super().__init__(command_prefix="!", intents=intents)
        self.order_manager = order_manager
        
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

    async def send_execution_alert(self, symbol, action, confidence, market_story):
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
        embed.add_field(name="Action", value=str(action).upper(), inline=True)
        embed.add_field(name="Confidence", value=f"{float(confidence):.2f}", inline=True)
        embed.add_field(name="Market Story", value=f"```{market_story}```", inline=False)

        await channel.send(embed=embed)