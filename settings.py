"""
Portfolio Analytics — Global Configuration
==========================================

Centralized parameters for data acquisition, strategy rules,
and analysis settings used across all notebooks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DataConfig:
    """Data source configuration."""

    START_DATE: str = "2019-01-01"
    END_DATE: str = "2024-01-01"
    RISK_FREE_RATE: float = 0.045
    TRADING_DAYS_PER_YEAR: int = 252
    RANDOM_SEED: int = 42

    FACTOR_TICKERS: Dict[str, List[str]] = field(default_factory=lambda: {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'],
        'Financials': ['JPM', 'BAC', 'GS', 'WFC', 'MS'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABT'],
        'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
        'Energy': ['XOM', 'CVX', 'COP'],
        'Industrials': ['CAT', 'BA', 'HON', 'GE'],
        'Utilities_REITs': ['NEE', 'DUK', 'AMT', 'SPG'],
    })

    PAIRS: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('KO', 'PEP'),
        ('XOM', 'CVX'),
        ('GLD', 'GDX'),
        ('JPM', 'BAC'),
        ('MSFT', 'AAPL'),
    ])

    OPTIONS_UNDERLYING: str = "SPY"

    FRED_SERIES: Dict[str, str] = field(default_factory=lambda: {
        'risk_free_rate': 'DGS10',
        'fed_funds': 'DFF',
        'cpi': 'CPIAUCSL',
        'vix': '^VIX',
    })


@dataclass
class StrategyConfig:
    """Trading strategy parameters."""

    ZSCORE_ENTRY: float = 2.0
    ZSCORE_EXIT: float = 0.5
    ZSCORE_STOP: float = 3.0
    LOOKBACK_WINDOW: int = 252
    HALF_LIFE_MAX: int = 126

    NUM_QUINTILES: int = 5
    REBALANCE_FREQ: str = 'M'
    MOMENTUM_SKIP: int = 21
    MOMENTUM_LOOKBACK: int = 252
    VOL_WINDOW_SHORT: int = 21
    VOL_WINDOW_LONG: int = 63


DATA = DataConfig()
STRATEGY = StrategyConfig()
