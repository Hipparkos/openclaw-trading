import discord
from discord.ext import commands
import os
import logging

class TradeApprovalView(discord.ui.View):
    def __init__(self, order_manager, symbol, side):
        super().__init__(timeout=300)
        self.order_manager = order_manager
        self.symbol = symbol
        self.side = side

    @discord.ui.button(label="APPROVE", style=discord.ButtonStyle.green)
    async def approve_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message(f"Executing {self.side} on {self.symbol}.")
        # Trigger the physical trade execution
        await self.order_manager.execute_trade(self.symbol, self.side)
        self.stop()

    @discord.ui.button(label="REJECT", style=discord.ButtonStyle.red)
    async def reject_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message(f"Trade cancelled by user.")
        self.stop()

class OpenClawDiscord(commands.Bot):
    def __init__(self, order_manager):
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)
        self.order_manager = order_manager
        self.channel_id = int(os.getenv("DISCORD_CHANNEL_ID"))

    async def on_ready(self):
        logging.info(f"OpenClaw connected to Discord as {self.user}")

    async def send_trade_signal(self, symbol, market_story, llm_prediction):
        channel = self.get_channel(self.channel_id)
        
        embed = discord.Embed(title=f"SIGNAL DETECTED: {symbol}", color=0x00ff00 if llm_prediction == "UP" else 0xff0000)
        embed.add_field(name="Llama 3.2 Prediction", value=f"**{llm_prediction}**", inline=False)
        embed.add_field(name="The Market Story", value=f"```{market_story}```", inline=False)
        
        # Attach the interactive buttons
        view = TradeApprovalView(self.order_manager, symbol, "BUY" if llm_prediction == "UP" else "SELL")
        await channel.send(embed=embed, view=view)