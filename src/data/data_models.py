from dataclasses import dataclass
from datetime import datetime


@dataclass
class BarData:
    symbol: str
    timestamp: datetime | str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class NewsData:
    symbol: str
    timestamp: datetime | str
    headline: str
    sentiment_score: float | None
    url: str