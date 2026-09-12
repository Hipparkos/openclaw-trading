# Live-updating Discord dashboard: one message, edited on a timer.

# Discord renders message edits in place for everyone viewing the channel, so a
# single pinned message becomes a continuously-updating panel without any web
# server, port, or auth. See DashboardView for the tab switching and
# DashboardRefresher for the edit loop.

# Data flow: main.py owns the live state (order_manager, open_trade_memory,
# screener, circuit_breaker) and feeds a DecisionLog as it evaluates symbols.
# This module only reads that state and renders it — it never mutates trading
# state itself.

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque

import discord
from discord.ext import tasks

logger = logging.getLogger("Dashboard")

_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "database" / "dashboard_state.json"
_MAX_DECISIONS = 40


# ── Decision log ─────────────────────────────────────────────────────────────

@dataclass
class DecisionEntry:
    symbol: str
    gate: str
    llm: str
    outcome: str
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionLog:

    def __init__(self, maxlen: int = _MAX_DECISIONS) -> None:
        self._entries: Deque[DecisionEntry] = deque(maxlen=maxlen)

    def record(self, symbol: str, gate: str, llm: str, outcome: str) -> None:
        self._entries.appendleft(DecisionEntry(symbol=symbol, gate=gate, llm=llm, outcome=outcome))

    def recent(self) -> list[DecisionEntry]:
        return list(self._entries)

def _load_state() -> dict:
    try:
        return json.loads(_STATE_PATH.read_text())
    except Exception:
        return {}


def _save_state(channel_id: int, message_id: int) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps({"channel_id": channel_id, "message_id": message_id}))
    except Exception as exc:
        logger.warning("Could not persist dashboard message location: %s", exc)


# ── Rendering ─────────────────────────────────────────────────────────────────

_COLOR = 0x0E5C55


def _fmt_money(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def _fmt_duration(entry_time: Any) -> str:
    if entry_time is None:
        return "—"
    try:
        delta = datetime.now(timezone.utc) - entry_time
        total = int(delta.total_seconds())
        h, rem = divmod(max(total, 0), 3600)
        m = rem // 60
        return f"{h}h{m:02d}m" if h else f"{m}m"
    except Exception:
        return "—"


def build_positions_embed(
    order_manager: Any,
    open_trade_memory: dict[str, dict],
    todays_trades: list[dict],
    circuit_breaker: dict | None,
) -> discord.Embed:
    embed = discord.Embed(title="Dashboard — Positions", color=_COLOR)

    try:
        equity = order_manager.get_account_equity()
    except Exception:
        equity = 0.0
    try:
        gross_exposure = order_manager.get_gross_position_value()
    except Exception:
        gross_exposure = 0.0
    exposure_pct = (gross_exposure / equity * 100) if equity > 0 else 0.0

    day_pnl = sum(float(t.get("pnl", 0.0)) for t in todays_trades)
    day_pnl_pct = (day_pnl / (equity - day_pnl) * 100) if (equity - day_pnl) > 0 else 0.0

    halted = bool(circuit_breaker and circuit_breaker.get("halted"))
    status_line = "HALTED" if halted else "live"

    embed.description = (
        f"{status_line}\n"
        f"**Equity** `${equity:,.2f}`   "
        f"**Day P&L** `{_fmt_money(day_pnl)} ({day_pnl_pct:+.2f}%)`   "
        f"**Exposure** `{exposure_pct:.1f}%`"
    )

    try:
        portfolio = list(order_manager.ib.portfolio())
    except Exception:
        portfolio = []

    rows = []
    for item in portfolio:
        symbol = getattr(item.contract, "symbol", "?")
        qty = float(getattr(item, "position", 0.0) or 0.0)
        if qty == 0.0:
            continue
        avg_cost = float(getattr(item, "averageCost", 0.0) or 0.0)
        market_value = float(getattr(item, "marketValue", 0.0) or 0.0)
        now_price = (market_value / qty) if qty else 0.0
        pnl = market_value - avg_cost * qty
        pnl_pct = (pnl / (avg_cost * qty) * 100) if avg_cost * qty else 0.0

        mem = open_trade_memory.get(symbol.upper().strip())
        held_for = _fmt_duration(mem.get("entry_time")) if mem else "—"

        rows.append(
            f"{symbol:<6}{qty:>7,.0f}  {now_price:>9,.2f}  "
            f"{_fmt_money(pnl):>11}  {pnl_pct:>+6.1f}%  {held_for:>6}"
        )

    if rows:
        header = f"{'SYM':<6}{'QTY':>7}  {'PRICE':>9}  {'P&L':>11}  {'%':>7}  {'HELD':>6}"
        table = "\n".join([header] + rows[:15])
        embed.add_field(name=f"Open positions ({len(rows)})", value=f"```{table}```", inline=False)
    else:
        embed.add_field(name="Open positions", value="_flat — no positions held_", inline=False)

    wins = sum(1 for t in todays_trades if float(t.get("pnl", 0.0)) >= 0)
    losses = len(todays_trades) - wins
    embed.set_footer(
        text=f"{len(todays_trades)} trades today · {wins}W/{losses}L  ·  "
             f"updated {datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S')}"
    )
    return embed


def build_decisions_embed(decision_log: DecisionLog) -> discord.Embed:
    embed = discord.Embed(title="Dashboard — Decisions", color=_COLOR)
    entries = decision_log.recent()

    if not entries:
        embed.description = "_No evaluations recorded yet this session._"
        return embed

    header = f"{'SYM':<6}{'GATE':<15}{'LLM':<14}{'OUTCOME'}"
    lines = [header]
    for e in entries[:18]:
        lines.append(f"{e.symbol:<6}{e.gate[:14]:<15}{e.llm[:13]:<14}{e.outcome}")
    table = "\n".join(lines)
    embed.description = f"```{table}```"
    embed.set_footer(text="Most recent evaluation per row · newest first · this session only")
    return embed


def build_watchlist_embed(screener: Any, settings: dict | None) -> discord.Embed:
    embed = discord.Embed(title="Dashboard — Watchlist", color=_COLOR)
    picks = getattr(screener, "last_picks", None) or []

    if not picks:
        tickers = (settings or {}).get("tickers", [])
        embed.description = (
            "_No screener run recorded yet — showing configured tickers:_\n"
            f"`{', '.join(tickers) if tickers else '(none)'}`"
        )
        return embed

    header = f"{'SYM':<7}{'BMU':>7}{'APTR':>6}  SECTOR"
    lines = [header]
    for p in picks[:20]:
        lines.append(
            f"{p['symbol']:<7}{p['bmu'] * 100:>+6.1f}%{p['aptr'] * 100:>5.1f}%  "
            f"{str(p.get('sector') or '-')[:16]}"
        )
    table = "\n".join(lines)
    embed.description = f"```{table}```"

    qualified = getattr(screener, "last_qualified", 0)
    scanned = getattr(screener, "last_scanned", 0)
    embed.set_footer(text=f"{qualified} qualified of {scanned:,} scanned  ·  trading top {len(picks)}")
    return embed


# ── View (tabs) ───────────────────────────────────────────────────────────────

_LIQUIDATE_IDLE_LABEL = "Liquidation"
_LIQUIDATE_ARMED_LABEL = "Confirm Liquidation?"


class DashboardView(discord.ui.View):
    """Tabs on row 0; trading controls on row 1. Pause/Resume mutate the
    shared circuit_breaker dict directly — the same dict main.py's !halt/
    !resume commands already mutate, so either surface reflects the other.
    Liquidation needs a second click to confirm: one accidental tap on a
    phone shouldn't be able to close every position."""

    def __init__(self, state: "DashboardState") -> None:
        super().__init__(timeout=None)
        self.state = state
        self._liquidation_armed = False
        self._sync_control_styles(state.current_tab)

    def _sync_control_styles(self, active_tab: str, *, reset_liquidation: bool = True) -> None:
        """Keep button appearance honest against live state: highlights the
        active tab, disables Pause/Resume to match whether trading is
        currently halted (which may have changed via !halt/!resume rather
        than these buttons), and — unless told not to — clears any armed
        liquidation confirmation back to its resting state."""
        halted = bool(self.state.circuit_breaker and self.state.circuit_breaker.get("halted"))
        if reset_liquidation:
            self._liquidation_armed = False

        for child in self.children:
            if not isinstance(child, discord.ui.Button):
                continue
            if child.custom_id == "dash_pause":
                child.disabled = halted
            elif child.custom_id == "dash_resume":
                child.disabled = not halted
            elif child.custom_id == "dash_liquidate":
                if reset_liquidation:
                    child.label = _LIQUIDATE_IDLE_LABEL
            else:   # the three tab buttons
                child.style = (
                    discord.ButtonStyle.primary if child.label == active_tab
                    else discord.ButtonStyle.secondary
                )

    def render(self) -> discord.Embed:
        s = self.state
        if s.current_tab == "Decisions":
            return build_decisions_embed(s.decision_log)
        if s.current_tab == "Watchlist":
            return build_watchlist_embed(s.screener, s.settings)
        return build_positions_embed(
            s.order_manager, s.open_trade_memory, s.get_todays_trades(), s.circuit_breaker,
        )

    async def _switch(self, interaction: discord.Interaction, tab: str) -> None:
        self.state.current_tab = tab
        self._sync_control_styles(tab)
        await interaction.response.edit_message(embed=self.render(), view=self)

    # ── Tabs (row 0) ──────────────────────────────────────────────────────

    @discord.ui.button(label="Positions", style=discord.ButtonStyle.primary, custom_id="dash_positions", row=0)
    async def btn_positions(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await self._switch(interaction, "Positions")

    @discord.ui.button(label="Decisions", style=discord.ButtonStyle.secondary, custom_id="dash_decisions", row=0)
    async def btn_decisions(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await self._switch(interaction, "Decisions")

    @discord.ui.button(label="Watchlist", style=discord.ButtonStyle.secondary, custom_id="dash_watchlist", row=0)
    async def btn_watchlist(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        await self._switch(interaction, "Watchlist")

    # ── Controls (row 1) ─────────────────────────────────────────────────

    @discord.ui.button(label="Pause", style=discord.ButtonStyle.secondary, custom_id="dash_pause", row=1)
    async def btn_pause(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        if self.state.circuit_breaker is not None:
            self.state.circuit_breaker["halted"] = True
        self._sync_control_styles(self.state.current_tab)
        await interaction.response.edit_message(embed=self.render(), view=self)
        await interaction.followup.send(
            "Trading **paused** — no new entries. Open positions are still managed (stops/targets remain active).",
            ephemeral=True,
        )

    @discord.ui.button(label="Resume", style=discord.ButtonStyle.success, custom_id="dash_resume", row=1)
    async def btn_resume(self, interaction: discord.Interaction, _: discord.ui.Button) -> None:
        if self.state.circuit_breaker is not None:
            self.state.circuit_breaker["halted"] = False
        self._sync_control_styles(self.state.current_tab)
        await interaction.response.edit_message(embed=self.render(), view=self)
        await interaction.followup.send("Trading **resumed** — new entries allowed again.", ephemeral=True)

    @discord.ui.button(label=_LIQUIDATE_IDLE_LABEL, style=discord.ButtonStyle.danger, custom_id="dash_liquidate", row=1)
    async def btn_liquidate(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._liquidation_armed:
            # First click: arm and wait for a deliberate second click. Any
            # other button (including the 30s auto-refresh) clears this.
            self._liquidation_armed = True
            self._sync_control_styles(self.state.current_tab, reset_liquidation=False)
            button.label = _LIQUIDATE_ARMED_LABEL
            await interaction.response.edit_message(embed=self.render(), view=self)
            return

        # Second click: confirmed. Reset the button before executing so a
        # slow or failed liquidation can't leave it stuck in "confirm" state.
        self._liquidation_armed = False
        button.label = _LIQUIDATE_IDLE_LABEL
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        await interaction.response.edit_message(view=self)

        results: list[str] = []
        error: str | None = None
        if callable(self.state.on_liquidate):
            try:
                results = await self.state.on_liquidate() or []
            except Exception as exc:
                error = str(exc)
        else:
            error = "Liquidation handler is not wired up yet."

        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = False
        self._sync_control_styles(self.state.current_tab)
        await interaction.edit_original_response(embed=self.render(), view=self)

        if error:
            await interaction.followup.send(f"Liquidation failed: {error}", ephemeral=True)
        elif results:
            await interaction.followup.send("**Liquidated:**\n" + "\n".join(results), ephemeral=True)
        else:
            await interaction.followup.send("No open positions — nothing to liquidate.", ephemeral=True)


# ── State + refresh loop ───────────────────────────────────────────────────────

@dataclass
class DashboardState:
    order_manager: Any
    screener: Any
    settings: dict | None
    circuit_breaker: dict | None
    open_trade_memory: dict
    get_todays_trades: Callable[[], list[dict]]
    on_liquidate: Callable[[], Any] | None = None   # async () -> list[str]; closes every open position
    decision_log: DecisionLog = field(default_factory=DecisionLog)
    current_tab: str = "Positions"


class DashboardController:

    def __init__(self, bot: discord.Client, state: DashboardState) -> None:
        self.bot = bot
        self.state = state
        self.view = DashboardView(state)
        self.channel_id: int | None = None
        self.message_id: int | None = None
        self._loop = tasks.loop(seconds=30)(self._tick)

    def register_persistent_view(self) -> None:
        self.bot.add_view(self.view)

    async def post_new(self, channel: discord.abc.Messageable) -> discord.Message:
        message = await channel.send(embed=self.view.render(), view=self.view)
        try:
            await message.pin()
        except Exception:
            pass
        self.channel_id = message.channel.id
        self.message_id = message.id
        _save_state(self.channel_id, self.message_id)
        if not self._loop.is_running():
            self._loop.start()
        return message

    async def resume_if_known(self) -> bool:
        saved = _load_state()
        channel_id, message_id = saved.get("channel_id"), saved.get("message_id")
        if not channel_id or not message_id:
            return False
        try:
            channel = self.bot.get_channel(channel_id) or await self.bot.fetch_channel(channel_id)
            await channel.fetch_message(message_id)   # just confirms it still exists
        except Exception as exc:
            logger.info("Saved dashboard message no longer reachable (%s) — will post a new one.", exc)
            return False
        self.channel_id, self.message_id = channel_id, message_id
        if not self._loop.is_running():
            self._loop.start()
        return True

    async def _tick(self) -> None:
        if not self.channel_id or not self.message_id:
            return
        try:
            channel = self.bot.get_channel(self.channel_id) or await self.bot.fetch_channel(self.channel_id)
            message = await channel.fetch_message(self.message_id)
            # Re-sync Pause/Resume against the live circuit breaker (it can
            # change via !halt/!resume too) and clear a forgotten liquidation
            # confirmation rather than leaving it armed indefinitely.
            self.view._sync_control_styles(self.state.current_tab)
            await message.edit(embed=self.view.render(), view=self.view)
        except discord.NotFound:
            logger.warning("Dashboard message was deleted — stopping refresh loop. Run !dashboard to repost.")
            self._loop.stop()
            self.channel_id = self.message_id = None
        except Exception as exc:
            logger.warning("Dashboard refresh failed (will retry next tick): %s", exc)

    def stop(self) -> None:
        if self._loop.is_running():
            self._loop.stop()
