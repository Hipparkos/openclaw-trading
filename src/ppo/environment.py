"""
OpenClaw Gym environment and shared observation utilities for PPO training.

build_static_obs / build_obs / TickerData are importable without gymnasium.
OpenClawEnv requires: pip install gymnasium stable-baselines3
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STATE_DIM = 9  # observation vector length (shared constant for env + engine inference)


def _sf(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


@dataclass
class TickerData:
    """Pre-processed per-ticker arrays for zero-overhead Gym env stepping."""
    symbol: str
    n_bars: int
    full_df_5m: pd.DataFrame
    full_df_1h: pd.DataFrame
    prefilter: np.ndarray   # int8: 0=NEUTRAL, 1=BULLISH, -1=BEARISH
    llm_conf: np.ndarray    # float32, 0.0–1.0
    llm_dir: np.ndarray     # float32, -1 / 0 / +1
    static_obs: np.ndarray  # float32, shape (n_bars, 4): adx, vol_ratio, rsi, h_dir


def build_static_obs(row: pd.Series, h_close: float, h_sma50: float) -> np.ndarray:
    """Compute the 4 static market-structure features for one bar.

    Returns float32 array: [adx_norm, vol_ratio, rsi_norm, h_dir]
    """
    adx_col = next((c for c in row.index if c.startswith("ADX_")), None)
    adx = min(_sf(row.get(adx_col)) / 60.0, 1.0) if adx_col else 0.0

    vol = _sf(row.get("volume"))
    vol_avg = _sf(row.get("volume_sma_20"), 1.0)
    vol_ratio = min(vol / vol_avg if vol_avg > 0 else 1.0, 3.0) / 3.0

    rsi = _sf(row.get("rsi_14")) / 100.0

    h_dir = 0.0
    if h_sma50 > 0:
        h_dir = 1.0 if h_close > h_sma50 else (-1.0 if h_close < h_sma50 else 0.0)

    return np.array([adx, vol_ratio, rsi, h_dir], dtype=np.float32)


def build_obs(
    static: np.ndarray,
    llm_conf: float,
    llm_dir: float,
    in_position: bool,
    unrealized_pct: float,
    recent_pnl_norm: float,
) -> np.ndarray:
    """Assemble the full 9-feature observation vector.

    Layout:
      [0] llm_conf        0.0–1.0
      [1] llm_dir         -1/0/+1
      [2] adx_norm        0.0–1.0  (ADX/60)
      [3] vol_ratio       0.0–1.0  (vol/avg, capped at 3×, /3)
      [4] rsi_norm        0.0–1.0  (RSI/100)
      [5] h_dir           -1/0/+1  (hourly SMA_50 gate)
      [6] in_position     0.0 or 1.0
      [7] unrealized_pct  -1.0..+1.0  (pnl / 10%, clamped)
      [8] recent_pnl_norm -1.0..+1.0  (avg last-3 trades, normalised)
    """
    obs = np.empty(STATE_DIM, dtype=np.float32)
    obs[0] = llm_conf
    obs[1] = llm_dir
    obs[2:6] = static                                       # adx, vol_ratio, rsi, h_dir
    obs[6] = 1.0 if in_position else 0.0
    obs[7] = max(-1.0, min(1.0, unrealized_pct / 0.10))
    obs[8] = recent_pnl_norm
    return obs


# ── Gym environment ────────────────────────────────────────────────────────────

try:
    import gymnasium as gym
    from gymnasium import spaces

    class OpenClawEnv(gym.Env):
        """
        Bar-by-bar Gym environment for PPO training on the OpenClaw strategy.

        Each episode steps through one full ticker's 5-minute bar sequence.
        All per-bar signals are pre-computed in TickerData so each step() is
        just array lookups — no pandas recalculation during training.

        Action space: Discrete(3)  — 0=skip, 1=long, 2=short
        The pre-filter gate is enforced inside step(): PPO can only execute when
        the deterministic pre-filter already sees a signal in the same direction.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            tickers: list[TickerData],
            commission: float = 1.50,
            position_pct: float = 0.015,
            start_equity: float = 100_000.0,
            stop_loss_pct: float = 0.02,
            atr_trail_mult: float = 2.0,
            take_profit_atr_mult: float = 3.0,
            min_hold_bars: int = 3,
            cooldown_bars: int = 3,
        ):
            super().__init__()
            if not tickers:
                raise ValueError("Need at least one TickerData")
            self.tickers = tickers
            self.commission = commission
            self.position_pct = position_pct
            self.start_equity = start_equity
            self.stop_loss_pct = stop_loss_pct
            self.atr_trail_mult = atr_trail_mult
            self.take_profit_atr_mult = take_profit_atr_mult
            self.min_hold_bars = min_hold_bars
            self.cooldown_bars = cooldown_bars

            self.observation_space = spaces.Box(
                low=np.full(STATE_DIM, -1.0, dtype=np.float32),
                high=np.full(STATE_DIM,  3.0, dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(3)

            self._ep = -1
            self._reset_ep()

        # ── Episode state ──────────────────────────────────────────────────────

        def _td(self) -> TickerData:
            return self.tickers[self._ep % len(self.tickers)]

        def _reset_ep(self) -> None:
            self._i = 0
            self.equity = self.start_equity
            self.in_pos = False
            self.direction = "LONG"
            self.entry_price = 0.0
            self.qty = 0
            self.peak = 0.0
            self.initial_atr = 0.0
            self.entry_bar = -1
            self.last_exit = -1
            self.recent_pnls: list[float] = []

        def _make_obs(self) -> np.ndarray:
            td = self._td()
            i = min(self._i, td.n_bars - 1)
            static = td.static_obs[i]
            llm_conf = float(td.llm_conf[i])
            llm_dir = float(td.llm_dir[i])

            unreal = 0.0
            if self.in_pos and self.entry_price > 0:
                cp = _sf(td.full_df_5m.iloc[i].get("close", self.entry_price))
                unreal = (
                    (cp - self.entry_price) / self.entry_price
                    if self.direction == "LONG"
                    else (self.entry_price - cp) / self.entry_price
                )

            recent = 0.0
            if self.recent_pnls:
                avg = sum(self.recent_pnls[-3:]) / min(len(self.recent_pnls), 3)
                recent = max(-1.0, min(1.0, avg / (self.start_equity * 0.01)))

            return build_obs(static, llm_conf, llm_dir, self.in_pos, unreal, recent)

        # ── Gym interface ──────────────────────────────────────────────────────

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._ep += 1
            self._reset_ep()
            return self._make_obs(), {}

        def step(self, action: int):
            td = self._td()
            i = self._i
            n = td.n_bars
            reward = 0.0

            # Episode boundary
            if i + 1 >= n:
                if self.in_pos:
                    lp = _sf(td.full_df_5m.iloc[-1].get("close", self.entry_price))
                    pnl = (
                        (lp - self.entry_price) * self.qty
                        if self.direction == "LONG"
                        else (self.entry_price - lp) * self.qty
                    ) - 2 * self.commission
                    self.equity += pnl
                    reward = pnl
                self._i += 1
                return self._make_obs(), reward, True, False, {"equity": self.equity}

            df = td.full_df_5m
            row = df.iloc[i]
            fp = _sf(df.iloc[i + 1].get("open", row.get("close", 1.0)))
            cp = _sf(row.get("close", fp))
            atr = _sf(row.get("atr_14"))

            # ── Exit ──────────────────────────────────────────────────────────
            if self.in_pos:
                bh = i - self.entry_bar
                reason = None

                loss = (
                    (self.entry_price - cp) / self.entry_price
                    if self.direction == "LONG"
                    else (cp - self.entry_price) / self.entry_price
                )
                if loss >= self.stop_loss_pct:
                    reason = "stop"

                if reason is None and atr > 0:
                    if self.direction == "LONG":
                        self.peak = max(self.peak, cp)
                        if cp <= self.peak - self.atr_trail_mult * atr:
                            reason = "trail"
                    else:
                        self.peak = min(self.peak, cp)
                        if cp >= self.peak + self.atr_trail_mult * atr:
                            reason = "trail"

                if reason is None and self.initial_atr > 0 and bh >= self.min_hold_bars:
                    if self.direction == "LONG" and cp >= self.entry_price + self.take_profit_atr_mult * self.initial_atr:
                        reason = "tp"
                    elif self.direction == "SHORT" and cp <= self.entry_price - self.take_profit_atr_mult * self.initial_atr:
                        reason = "tp"

                if reason:
                    pnl = (
                        (fp - self.entry_price) * self.qty
                        if self.direction == "LONG"
                        else (self.entry_price - fp) * self.qty
                    ) - 2 * self.commission
                    self.equity += pnl
                    reward = pnl
                    self.recent_pnls.append(pnl)
                    self.in_pos = False
                    self.last_exit = i

            # ── Entry (only when PPO requests it and pre-filter agrees) ───────
            if not self.in_pos and action != 0:
                pre = int(td.prefilter[i])
                in_cool = self.last_exit >= 0 and (i - self.last_exit) < self.cooldown_bars
                if pre != 0 and not in_cool:
                    desired = 1 if action == 1 else -1   # 1=BULLISH, -1=BEARISH
                    if desired == pre:
                        conf = max(float(td.llm_conf[i]), 0.5)
                        self.qty = max(1, int((self.equity * self.position_pct * conf) / fp))
                        self.direction = "LONG" if desired == 1 else "SHORT"
                        self.entry_price = fp
                        self.peak = fp
                        self.initial_atr = atr
                        self.entry_bar = i
                        self.in_pos = True

            self._i += 1
            return self._make_obs(), reward, self._i >= n - 1, False, {"equity": self.equity}

        def render(self):
            pass
except ImportError:
    OpenClawEnv = None
