"""
Risk analytics for portfolio construction and monitoring.

Provides covariance estimation (sample, shrinkage, EWMA), portfolio
risk decomposition, stress testing, and rolling risk metrics used in
professional portfolio management.

Example:
    >>> from utils.risk import RiskMetrics
    >>> cov = RiskMetrics.covariance_matrix(returns, method='ledoit_wolf')
    >>> beta = RiskMetrics.beta(stock_returns, market_returns)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class RiskMetrics:
    """Portfolio risk measurement and decomposition toolkit.

    All methods are static for composability. Covariance estimators
    support sample, Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004), and
    exponentially weighted moving average (RiskMetrics / J.P. Morgan,
    1996) approaches.

    Example:
        >>> cov_lw = RiskMetrics.covariance_matrix(daily_returns, method='ledoit_wolf')
        >>> vol_series = RiskMetrics.rolling_volatility(returns['AAPL'], window=63)
    """

    @staticmethod
    def covariance_matrix(
        returns: pd.DataFrame,
        method: str = "sample",
        ewma_span: int = 63,
    ) -> pd.DataFrame:
        """Estimate the asset covariance matrix.

        Args:
            returns: DataFrame of asset returns (columns = assets).
            method: Estimation method — one of:

                * ``'sample'``: Maximum-likelihood sample covariance.
                * ``'ledoit_wolf'``: Ledoit-Wolf shrinkage toward a
                  structured target (requires ``scikit-learn``).
                * ``'ewma'``: Exponentially weighted covariance with
                  *ewma_span* half-life.

            ewma_span: Span parameter for EWMA (ignored by other
                methods).

        Returns:
            Square covariance DataFrame (annualized, 252 trading days).

        Raises:
            ValueError: If *method* is not recognized.
        """
        if method == "sample":
            return returns.cov() * 252

        if method == "ledoit_wolf":
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf().fit(returns.values)
            cov = pd.DataFrame(
                lw.covariance_ * 252,
                index=returns.columns,
                columns=returns.columns,
            )
            return cov

        if method == "ewma":
            ewma_cov = returns.ewm(span=ewma_span).cov()
            last_date = returns.index[-1]
            cov = ewma_cov.loc[last_date] * 252
            return cov

        raise ValueError(
            f"Unknown method '{method}'. Use 'sample', 'ledoit_wolf', or 'ewma'."
        )

    @staticmethod
    def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """Sample Pearson correlation matrix.

        Args:
            returns: DataFrame of asset returns.

        Returns:
            Correlation DataFrame with values in ``[-1, 1]``.
        """
        return returns.corr()

    @staticmethod
    def beta(
        returns: pd.Series,
        market: pd.Series,
    ) -> float:
        """Market beta (OLS slope of asset on market).

        .. math::
            \\beta = \\frac{\\text{Cov}(r_i, r_m)}{\\text{Var}(r_m)}

        Args:
            returns: Asset return series.
            market: Market return series (same index).

        Returns:
            Beta coefficient.
        """
        aligned = pd.concat([returns, market], axis=1).dropna()
        if len(aligned) < 2:
            return np.nan
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else np.nan

    @staticmethod
    def tracking_error(
        returns: pd.Series,
        benchmark: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Annualized tracking error relative to a benchmark.

        Args:
            returns: Strategy returns.
            benchmark: Benchmark returns.
            periods_per_year: Annualization factor.

        Returns:
            Annualized tracking error as a decimal.
        """
        active = (returns - benchmark).dropna()
        return float(active.std() * np.sqrt(periods_per_year))

    @staticmethod
    def rolling_volatility(
        returns: pd.Series,
        window: int = 63,
        periods_per_year: int = 252,
    ) -> pd.Series:
        """Rolling annualized volatility.

        Args:
            returns: Period returns.
            window: Rolling window in periods.
            periods_per_year: Annualization factor.

        Returns:
            Series of annualized rolling volatilities.
        """
        return returns.rolling(window).std() * np.sqrt(periods_per_year)

    @staticmethod
    def rolling_beta(
        returns: pd.Series,
        market: pd.Series,
        window: int = 252,
    ) -> pd.Series:
        """Rolling market beta.

        Args:
            returns: Asset returns.
            market: Market returns.
            window: Rolling window size.

        Returns:
            Series of rolling beta values.
        """
        cov = returns.rolling(window).cov(market)
        var = market.rolling(window).var()
        return cov / var

    @staticmethod
    def drawdown_series(returns: pd.Series) -> pd.Series:
        """Complete drawdown time series.

        Args:
            returns: Period returns with a ``DatetimeIndex``.

        Returns:
            Series of drawdown values (non-positive).
        """
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        return (cum - running_max) / running_max

    @staticmethod
    def drawdown_details(returns: pd.Series) -> pd.DataFrame:
        """Table of individual drawdown episodes.

        Identifies every peak-to-trough-to-recovery cycle and reports
        the depth, duration, and recovery time of each.

        Args:
            returns: Period returns.

        Returns:
            DataFrame with columns ``['start', 'trough', 'end',
            'depth', 'duration_days', 'recovery_days']``, sorted by
            depth (worst first).
        """
        dd = RiskMetrics.drawdown_series(returns)
        is_dd = dd < 0

        episodes: List[Dict] = []
        in_dd = False
        start = None

        for i, (date, val) in enumerate(dd.items()):
            if val < 0 and not in_dd:
                in_dd = True
                start = dd.index[max(0, i - 1)]
                trough_date = date
                trough_val = val
            elif val < 0 and in_dd:
                if val < trough_val:
                    trough_date = date
                    trough_val = val
            elif val >= 0 and in_dd:
                in_dd = False
                episodes.append(
                    {
                        "start": start,
                        "trough": trough_date,
                        "end": date,
                        "depth": trough_val,
                        "duration_days": (date - start).days,
                        "recovery_days": (date - trough_date).days,
                    }
                )

        if in_dd and start is not None:
            episodes.append(
                {
                    "start": start,
                    "trough": trough_date,
                    "end": dd.index[-1],
                    "depth": trough_val,
                    "duration_days": (dd.index[-1] - start).days,
                    "recovery_days": np.nan,
                }
            )

        df = pd.DataFrame(episodes)
        if not df.empty:
            df = df.sort_values("depth").reset_index(drop=True)
        return df

    @staticmethod
    def risk_contribution(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """Marginal risk contribution of each asset.

        Decomposes total portfolio volatility into per-asset
        contributions such that they sum to total portfolio variance.

        .. math::
            RC_i = w_i \\cdot (\\Sigma w)_i

        Args:
            weights: Asset weight vector (sums to 1).
            cov_matrix: Covariance matrix (annualized).

        Returns:
            Array of risk contributions (same length as weights).
        """
        port_var = weights @ cov_matrix @ weights
        marginal = cov_matrix @ weights
        return weights * marginal / np.sqrt(port_var)

    @staticmethod
    def risk_parity_weights(
        cov_matrix: np.ndarray,
        budget: Optional[np.ndarray] = None,
        max_iter: int = 1000,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """Compute risk-parity (equal risk contribution) weights.

        Uses the iterative Spinu (2013) algorithm to find weights where
        each asset's risk contribution equals the risk budget.

        Args:
            cov_matrix: Annualized covariance matrix.
            budget: Target risk budget per asset. Defaults to equal
                (``1/n`` each).
            max_iter: Maximum iterations.
            tol: Convergence tolerance.

        Returns:
            Weight vector summing to 1.
        """
        n = cov_matrix.shape[0]
        if budget is None:
            budget = np.ones(n) / n

        w = np.ones(n) / n
        for _ in range(max_iter):
            marginal = cov_matrix @ w
            rc = w * marginal
            rc_total = rc.sum()
            target = budget * rc_total
            w_new = w * (target / rc) ** 0.5
            w_new /= w_new.sum()
            if np.max(np.abs(w_new - w)) < tol:
                break
            w = w_new

        return w

    @staticmethod
    def stress_test(
        returns: pd.DataFrame,
        scenarios: Dict[str, Tuple[str, str]],
    ) -> pd.DataFrame:
        """Run historical stress-test scenarios.

        Evaluates portfolio performance during predefined crisis
        periods and reports cumulative return, max drawdown, and
        annualized volatility for each scenario.

        Args:
            returns: DataFrame of asset or portfolio returns.
            scenarios: Mapping of ``{scenario_name: (start_date,
                end_date)}``, e.g.
                ``{'COVID Crash': ('2020-02-19', '2020-03-23')}``.

        Returns:
            DataFrame with one row per scenario and columns for each
            asset's cumulative return, max drawdown, and volatility.
        """
        records = []
        for name, (start, end) in scenarios.items():
            mask = (returns.index >= start) & (returns.index <= end)
            period = returns.loc[mask]
            if period.empty:
                continue

            cum_ret = (1 + period).prod() - 1
            record = {"Scenario": name, "Period": f"{start} to {end}"}

            if isinstance(cum_ret, pd.Series):
                for col in cum_ret.index:
                    record[f"{col} Return"] = cum_ret[col]
            else:
                record["Return"] = float(cum_ret)

            records.append(record)

        return pd.DataFrame(records).set_index("Scenario") if records else pd.DataFrame()

    @staticmethod
    def parametric_var(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        mean_returns: np.ndarray,
        level: float = 0.05,
        horizon_days: int = 1,
    ) -> float:
        """Parametric (variance-covariance) Value at Risk.

        Assumes returns are normally distributed.

        Args:
            weights: Portfolio weight vector.
            cov_matrix: Daily covariance matrix.
            mean_returns: Expected daily return vector.
            level: Left-tail probability (default 5 %).
            horizon_days: VaR horizon in trading days.

        Returns:
            Portfolio VaR as a negative decimal.
        """
        from scipy.stats import norm

        port_mean = float(weights @ mean_returns) * horizon_days
        port_vol = float(
            np.sqrt(weights @ cov_matrix @ weights * horizon_days)
        )
        return port_mean + norm.ppf(level) * port_vol

