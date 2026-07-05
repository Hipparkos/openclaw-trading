from __future__ import annotations

import io
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
# Render "$" literally — don't let matplotlib parse $...$ as math mode, which
# italicised the text between two dollar signs and swallowed the "$" characters.
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


_BG       = "#0d1117"
_PANEL_BG = "#161b22"
_TEAL     = "#00d4b8"
_GREEN    = "#26a269"
_RED      = "#e01e37"
_MUTED    = "#8b949e"
_WHITE    = "#e6edf3"
_GRID     = "#21262d"
_BORDER   = "#30363d"


def generate_stats_chart(trades: list[dict], period_label: str,
                         account_equity: float = 0.0) -> io.BytesIO:
    """Return a PNG BytesIO of an IBKR-style performance chart.

    account_equity is the current account equity; it's used to express net P&L
    as a percentage of the equity at the start of the period.
    """

    pnls = [float(t["pnl"]) for t in trades]
    net_pnl   = sum(pnls)
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    win_rate  = len(wins) / len(pnls) * 100 if pnls else 0.0
    avg_win   = sum(wins)   / len(wins)   if wins   else 0.0
    avg_loss  = sum(losses) / len(losses) if losses else 0.0
    pf_denom  = abs(sum(losses))
    profit_factor = abs(sum(wins)) / pf_denom if pf_denom > 0 else float("inf")

    # Cumulative P&L per trade
    cum_pnl: list[float] = []
    exit_times: list[datetime] = []
    running = 0.0
    for t in trades:
        running += float(t["pnl"])
        cum_pnl.append(running)
        try:
            dt = datetime.fromisoformat(t["exit_time"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            exit_times.append(dt)
        except Exception:
            exit_times.append(datetime.now(timezone.utc))

    # Daily P&L totals
    daily: dict[str, float] = defaultdict(float)
    for t in trades:
        try:
            dt = datetime.fromisoformat(t["exit_time"])
        except Exception:
            dt = datetime.now(timezone.utc)
        daily[dt.strftime("%Y-%m-%d")] += float(t["pnl"])

    day_labels = sorted(daily.keys())
    day_values = [daily[d] for d in day_labels]
    day_dates  = [datetime.strptime(d, "%Y-%m-%d") for d in day_labels]

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 7), facecolor=_BG)

    pnl_color = _GREEN if net_pnl >= 0 else _RED
    pnl_sign  = "+" if net_pnl >= 0 else ""

    # Net P&L as % growth of the account over the period (vs equity at period start).
    start_equity = account_equity - net_pnl
    pnl_pct = (net_pnl / start_equity * 100) if start_equity > 0 else 0.0
    pct_str = f"  ({pnl_pct:.2f}%)" if account_equity > 0 else ""

    fig.text(0.03, 0.96, period_label,
             fontsize=14, color=_WHITE, fontweight="bold", va="top")
    fig.text(0.03, 0.91,
             f"Net P&L  {pnl_sign}${net_pnl:,.2f}{pct_str}",
             fontsize=12, color=pnl_color, fontweight="bold", va="top")
    stats_line = (
        f"Trades: {len(pnls)}   "
        f"Win Rate: {win_rate:.1f}%   "
        f"Avg Win: +${avg_win:,.2f}   "
        f"Avg Loss: -${abs(avg_loss):,.2f}   "
        f"Profit Factor: {profit_factor:.2f}"
    )
    fig.text(0.03, 0.87, stats_line,
             fontsize=9, color=_MUTED, va="top")

    if not trades:
        fig.text(0.5, 0.5, "No trades in this period",
                 ha="center", va="center", fontsize=14, color=_MUTED)
        return _save(fig)

    # ── Top panel: cumulative P&L curve ────────────────────────────────────────
    # Plotted by TRADE ORDER (not calendar time) so overnight / weekend / closed-
    # market gaps don't draw a misleading straight line across dead time. Date
    # ticks are placed at each day's first trade so the timeline is still readable.
    ax1 = fig.add_axes([0.06, 0.45, 0.91, 0.36], facecolor=_PANEL_BG)
    if cum_pnl:
        xs = list(range(len(cum_pnl)))
        ax1.plot(xs, cum_pnl, color=_TEAL, linewidth=1.6, zorder=3)
        fill_col = _GREEN if cum_pnl[-1] >= 0 else _RED
        ax1.fill_between(xs, 0, cum_pnl, alpha=0.13, color=fill_col, zorder=2)
        ax1.set_xlim(0, max(len(cum_pnl) - 1, 1))
    ax1.axhline(0, color=_MUTED, linewidth=0.6, linestyle="--", alpha=0.5)
    _style_axis(ax1)

    # Date ticks at the first trade of each new day (thinned if there are many days).
    day_pos: list[int] = []
    day_lab: list[str] = []
    last_day = None
    for idx, dt in enumerate(exit_times):
        key = dt.strftime("%Y-%m-%d")
        if key != last_day:
            day_pos.append(idx)
            day_lab.append(dt.strftime("%b %d"))
            last_day = key
    if len(day_pos) > 12:
        stride = (len(day_pos) // 10) + 1
        day_pos, day_lab = day_pos[::stride], day_lab[::stride]
    ax1.set_xticks(day_pos)
    ax1.set_xticklabels(day_lab, rotation=25, ha="right")

    ax1.set_ylabel("Cumulative P&L ($)", color=_MUTED, fontsize=8)
    _dollar_axis(ax1)

    # ── Bottom panel: daily bars ────────────────────────────────────────────────
    ax2 = fig.add_axes([0.06, 0.11, 0.91, 0.27], facecolor=_PANEL_BG)
    if day_dates:
        bar_colors = [_GREEN if v >= 0 else _RED for v in day_values]
        ax2.bar(day_dates, day_values, color=bar_colors, width=0.6, zorder=3)
    ax2.axhline(0, color=_MUTED, linewidth=0.6, linestyle="--", alpha=0.5)
    _style_axis(ax2)
    ax2.set_ylabel("Daily P&L ($)", color=_MUTED, fontsize=8)
    _dollar_axis(ax2)

    if day_dates:
        span = (max(day_dates) - min(day_dates)).days
        if span > 35:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
        else:
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            if span > 10:
                ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
        plt.setp(ax2.get_xticklabels(), rotation=25, ha="right")

    return _save(fig)


def _dollar_axis(ax: plt.Axes) -> None:
    """Format an axis' y-ticks as whole-dollar amounts."""
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    for spine in ax.spines.values():
        spine.set_color(_BORDER)
    ax.grid(axis="y", color=_GRID, linewidth=0.5, zorder=0)


def _save(fig: plt.Figure) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf
