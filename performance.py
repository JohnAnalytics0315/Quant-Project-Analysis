"""
Performance analytics for portfolio evaluation.

Implements standard risk-adjusted return metrics used in quantitative
finance and portfolio management, following industry conventions for
annualization and downside-risk measurement.

All public methods are static, so no instantiation is needed:
    >>> from utils.performance import PerformanceAnalytics as pa
    >>> pa.sharpe_ratio(daily_returns, rf=0.0)
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


class PerformanceAnalytics:
    """Portfolio performance measurement toolkit.

    Provides static methods for computing risk-adjusted return metrics,
    drawdown analysis, and multi-strategy comparison tables. Every
    method operates on a returns Series or array and uses NumPy
    vectorized operations for speed.

    Example:
        >>> returns = pd.Series(np.random.randn(252) * 0.01)
        >>> sharpe = PerformanceAnalytics.sharpe_ratio(returns)
        >>> summary = PerformanceAnalytics.summary_table({'MyStrat': returns})
    """

    @staticmethod
    def compute_returns(
        prices: pd.DataFrame,
        method: str = "simple",
    ) -> pd.DataFrame:
        """Compute period returns from a price DataFrame.

        Args:
            prices: DataFrame of asset prices with a ``DatetimeIndex``.
            method: ``'simple'`` for arithmetic returns
                ``(P_t / P_{t-1} - 1)``, or ``'log'`` for logarithmic
                returns ``ln(P_t / P_{t-1})``.

        Returns:
            DataFrame of returns (first row dropped).
        """
        if method == "log":
            return np.log(prices / prices.shift(1)).dropna()
        return prices.pct_change().dropna()

    @staticmethod
    def annualized_return(
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: int = 252,
    ) -> float:
        """Compound annualized growth rate (CAGR).

        Args:
            returns: Period returns.
            periods_per_year: Number of periods in one year (252 for
                daily, 12 for monthly).

        Returns:
            CAGR as a decimal.
        """
        total = float(np.prod(1 + np.asarray(returns)))
        n_years = len(returns) / periods_per_year
        if n_years <= 0:
            return 0.0
        return total ** (1.0 / n_years) - 1.0

    @staticmethod
    def annualized_volatility(
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: int = 252,
    ) -> float:
        """Annualized standard deviation of returns.

        Args:
            returns: Period returns.
            periods_per_year: Annualization factor.

        Returns:
            Annualized volatility as a decimal.
        """
        return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))

    @staticmethod
    def sharpe_ratio(
        returns: Union[pd.Series, np.ndarray],
        rf: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Annualized Sharpe ratio (Sharpe, 1966).

        .. math::
            SR = \\frac{\\bar{r} - r_f}{\\sigma} \\sqrt{N}

        Args:
            returns: Period returns.
            rf: Risk-free rate *per period* (same frequency as returns).
            periods_per_year: Annualization factor.

        Returns:
            Annualized Sharpe ratio.
        """
        excess = np.asarray(returns) - rf
        sigma = float(np.std(excess, ddof=1))
        if sigma == 0:
            return 0.0
        return float(np.mean(excess) / sigma * np.sqrt(periods_per_year))

    @staticmethod
    def sortino_ratio(
        returns: Union[pd.Series, np.ndarray],
        rf: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Annualized Sortino ratio (downside-deviation denominator).

        Only negative excess returns contribute to the risk measure,
        making it more appropriate for asymmetric return distributions.

        Args:
            returns: Period returns.
            rf: Risk-free rate per period.
            periods_per_year: Annualization factor.

        Returns:
            Annualized Sortino ratio (``inf`` if no downside).
        """
        excess = np.asarray(returns) - rf
        downside = excess[excess < 0]
        if len(downside) == 0 or float(np.std(downside, ddof=1)) == 0:
            return np.inf if float(np.mean(excess)) > 0 else 0.0
        return float(
            np.mean(excess) / np.std(downside, ddof=1) * np.sqrt(periods_per_year)
        )

    @staticmethod
    def calmar_ratio(
        returns: Union[pd.Series, np.ndarray],
        periods_per_year: int = 252,
    ) -> float:
        """Calmar ratio: annualized return divided by maximum drawdown.

        Args:
            returns: Period returns.
            periods_per_year: Annualization factor.

        Returns:
            Calmar ratio (``inf`` if max drawdown is zero).
        """
        ann_ret = PerformanceAnalytics.annualized_return(returns, periods_per_year)
        mdd = abs(PerformanceAnalytics.max_drawdown(returns))
        return ann_ret / mdd if mdd > 0 else np.inf

    @staticmethod
    def information_ratio(
        returns: pd.Series,
        benchmark: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Information ratio relative to a benchmark.

        .. math::
            IR = \\frac{\\bar{r}_a}{\\sigma_a} \\sqrt{N}

        where :math:`r_a = r - r_b` is the active return.

        Args:
            returns: Strategy period returns.
            benchmark: Benchmark period returns (same index).
            periods_per_year: Annualization factor.

        Returns:
            Annualized information ratio.
        """
        active = returns - benchmark
        sigma = float(active.std())
        if sigma == 0:
            return 0.0
        return float(np.sqrt(periods_per_year) * active.mean() / sigma)

    @staticmethod
    def max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
        """Maximum peak-to-trough drawdown.

        Args:
            returns: Period returns.

        Returns:
            Maximum drawdown as a negative decimal (e.g. ``-0.25``
            for a 25 % drawdown).
        """
        cum = np.cumprod(1 + np.asarray(returns))
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        return float(np.min(dd))

    @staticmethod
    def drawdown_series(returns: pd.Series) -> pd.Series:
        """Full drawdown time series.

        Args:
            returns: Period returns with a ``DatetimeIndex``.

        Returns:
            Series of drawdowns (negative values indicate loss from
            the running peak).
        """
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        return (cum - running_max) / running_max

    @staticmethod
    def var(
        returns: Union[pd.Series, np.ndarray],
        level: float = 0.05,
    ) -> float:
        """Historical Value at Risk.

        Args:
            returns: Period returns.
            level: Left-tail probability (default ``0.05`` for 95 % VaR).

        Returns:
            VaR as a (typically negative) decimal.
        """
        return float(np.percentile(returns, level * 100))

    @staticmethod
    def cvar(
        returns: Union[pd.Series, np.ndarray],
        level: float = 0.05,
    ) -> float:
        """Conditional VaR / Expected Shortfall.

        Average return in the worst ``level``-fraction of observations.

        Args:
            returns: Period returns.
            level: Left-tail probability.

        Returns:
            CVaR as a (typically negative) decimal.
        """
        threshold = PerformanceAnalytics.var(returns, level)
        arr = np.asarray(returns)
        tail = arr[arr <= threshold]
        return float(np.mean(tail)) if len(tail) > 0 else threshold

    @staticmethod
    def hit_rate(returns: Union[pd.Series, np.ndarray]) -> float:
        """Fraction of positive-return periods (win rate).

        Args:
            returns: Period returns.

        Returns:
            Win rate as a decimal in ``[0, 1]``.
        """
        return float(np.mean(np.asarray(returns) > 0))

    @staticmethod
    def profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
        """Ratio of gross profits to gross losses.

        Args:
            returns: Period returns.

        Returns:
            Profit factor (> 1 is profitable, ``inf`` if no losses).
        """
        arr = np.asarray(returns)
        gains = arr[arr > 0].sum()
        losses = abs(arr[arr < 0].sum())
        return gains / losses if losses > 0 else np.inf

    @staticmethod
    def tail_ratio(
        returns: Union[pd.Series, np.ndarray],
        level: float = 0.05,
    ) -> float:
        """Ratio of the right tail to the left tail.

        Compares the magnitude of the 95th percentile to the 5th
        percentile. Values > 1 indicate positive skew in the tails.

        Args:
            returns: Period returns.
            level: Percentile level for the tails.

        Returns:
            Tail ratio.
        """
        right = float(np.percentile(returns, (1 - level) * 100))
        left = abs(float(np.percentile(returns, level * 100)))
        return right / left if left > 0 else np.inf

    @staticmethod
    def summary_table(
        returns_dict: Dict[str, pd.Series],
        rf: float = 0.0,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """Multi-strategy performance comparison table.

        Args:
            returns_dict: Mapping of ``{strategy_name: returns_series}``.
            rf: Risk-free rate per period.
            periods_per_year: Annualization factor.

        Returns:
            DataFrame with one row per strategy and columns for all
            major risk-return metrics.
        """
        pa = PerformanceAnalytics
        records = []
        for name, rets in returns_dict.items():
            records.append(
                {
                    "Strategy": name,
                    "Ann. Return": pa.annualized_return(rets, periods_per_year),
                    "Ann. Volatility": pa.annualized_volatility(rets, periods_per_year),
                    "Sharpe": pa.sharpe_ratio(rets, rf, periods_per_year),
                    "Sortino": pa.sortino_ratio(rets, rf, periods_per_year),
                    "Calmar": pa.calmar_ratio(rets, periods_per_year),
                    "Max Drawdown": pa.max_drawdown(rets),
                    "VaR 95%": pa.var(rets, 0.05),
                    "CVaR 95%": pa.cvar(rets, 0.05),
                    "Hit Rate": pa.hit_rate(rets),
                    "Profit Factor": pa.profit_factor(rets),
                }
            )
        return pd.DataFrame(records).set_index("Strategy")

