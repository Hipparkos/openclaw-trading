from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class TradeHistoryDB:
    def __init__(self) -> None:
        self.db_path = Path(__file__).resolve().parents[2] / "data" / "database" / "trade_history.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT    NOT NULL,
                    direction   TEXT    NOT NULL DEFAULT 'LONG',
                    entry_time  TEXT,
                    exit_time   TEXT    NOT NULL,
                    entry_price REAL,
                    exit_price  REAL,
                    quantity    REAL,
                    pnl         REAL    NOT NULL,
                    confidence  REAL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exit_time ON trades(exit_time)"
            )

    def record(
        self,
        symbol: str,
        pnl: float,
        confidence: float,
        direction: str = "LONG",
        entry_time: datetime | None = None,
        exit_time: datetime | None = None,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        quantity: float = 0.0,
    ) -> None:
        now = datetime.now(timezone.utc)
        exit_ts = (exit_time or now).isoformat()
        entry_ts = entry_time.isoformat() if entry_time else None
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO trades
                       (symbol, direction, entry_time, exit_time,
                        entry_price, exit_price, quantity, pnl, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, direction, entry_ts, exit_ts,
                     entry_price, exit_price, quantity, pnl, confidence),
                )

    def get_trades(
        self,
        since: datetime,
        until: datetime | None = None,
    ) -> list[dict]:
        end = (until or datetime.now(timezone.utc)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM trades
                   WHERE exit_time >= ? AND exit_time <= ?
                   ORDER BY exit_time ASC""",
                (since.isoformat(), end),
            ).fetchall()
        return [dict(r) for r in rows]
