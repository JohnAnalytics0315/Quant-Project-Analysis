"""
Cross-sectional factor construction and evaluation.

Implements the standard quantitative equity workflow: signal
construction, cross-sectional normalization, composite scoring,
and factor performance measurement via Information Coefficients.

Example:
    >>> from utils.factors import FactorEngine as fe
    >>> mom = fe.momentum(prices, lookback=252, skip=21)
    >>> z = fe.cross_sectional_zscore(mom)
    >>> ic = fe.information_coefficient(z, forward_returns)
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, spearmanr


class FactorEngine:
    """Cross-sectional equity factor construction toolkit.

    Provides static methods for building, normalizing, and evaluating
    common alpha factors (momentum, value, quality, low volatility,
    mean reversion) following the methodology of Fama-French (1993),
    Carhart (1997), and Barra.

    Example:
        >>> mom = FactorEngine.momentum(prices, lookback=252, skip=21)
        >>> zscore = FactorEngine.cross_sectional_zscore(mom)
    """

    @staticmethod
    def momentum(
        prices: pd.DataFrame,
        lookback: int = 252,
        skip: int = 21,
    ) -> pd.DataFrame:
        """Momentum factor: trailing return excluding the most recent month.

        Computes the ``(lookback - skip)``-day return ending ``skip``
        days before each date, following Jegadeesh & Titman (1993).

        Args:
            prices: DataFrame of asset prices.
            lookback: Total lookback window in trading days (default
                12 months = 252).
            skip: Number of recent days to exclude (default 1 month =
                21) to avoid short-term reversal contamination.

        Returns:
            DataFrame of momentum signals (same shape as *prices*,
            with leading NaNs).
        """
        lagged = prices.shift(skip)
        return lagged / prices.shift(lookback) - 1

    @staticmethod
    def volatility_factor(
        returns: pd.DataFrame,
        window: int = 63,
    ) -> pd.DataFrame:
        """Low-volatility factor: inverse of rolling realized volatility.

        Lower volatility receives a higher score, following the low-vol
        anomaly (Baker, Bradley & Wurgler, 2011).

        Args:
            returns: DataFrame of daily returns.
            window: Rolling window in trading days (default 63 ~ 3
                months).

        Returns:
            DataFrame of inverse-volatility signals.
        """
        vol = returns.rolling(window).std()
        return -vol

    @staticmethod
    def mean_reversion(
        returns: pd.DataFrame,
        window: int = 5,
    ) -> pd.DataFrame:
        """Short-term mean-reversion factor: negative recent return.

        Stocks that have recently declined are scored higher, capturing
        the short-term reversal effect documented by Jegadeesh (1990).

        Args:
            returns: DataFrame of daily returns.
            window: Lookback window in trading days.

        Returns:
            DataFrame of reversal signals.
        """
        return -returns.rolling(window).sum()

    @staticmethod
    def value_factor(
        prices: pd.DataFrame,
        lookback: int = 252,
    ) -> pd.DataFrame:
        """Price-based value proxy: inverse of trailing return.

        In the absence of fundamental data (P/E, P/B), uses inverse
        of the 12-month return as a crude value signal — stocks that
        have appreciated less are scored as cheaper.

        Args:
            prices: DataFrame of asset prices.
            lookback: Lookback in trading days.

        Returns:
            DataFrame of value signals.
        """
        trailing_ret = prices / prices.shift(lookback) - 1
        return -trailing_ret

    @staticmethod
    def quality_factor(
        returns: pd.DataFrame,
        window: int = 252,
    ) -> pd.DataFrame:
        """Quality factor: inverse of return volatility.

        Proxies for earnings stability when fundamental data is
        unavailable. Stocks with more stable returns are scored higher.

        Args:
            returns: DataFrame of daily returns.
            window: Lookback window in trading days.

        Returns:
            DataFrame of quality signals.
        """
        return -returns.rolling(window).std()

    @staticmethod
    def rsi(
        prices: pd.DataFrame,
        window: int = 14,
    ) -> pd.DataFrame:
        """Relative Strength Index (Wilder, 1978).

        Args:
            prices: DataFrame of asset prices.
            window: RSI lookback period.

        Returns:
            RSI values in ``[0, 100]``.
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def volume_ratio(
        volume: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 60,
    ) -> pd.DataFrame:
        """Volume ratio: short-term vs long-term average volume.

        Values > 1 indicate above-average recent trading activity.

        Args:
            volume: DataFrame of daily volumes.
            short_window: Short-term average window.
            long_window: Long-term average window.

        Returns:
            Volume ratio signals.
        """
        short_avg = volume.rolling(short_window).mean()
        long_avg = volume.rolling(long_window).mean()
        return short_avg / long_avg.replace(0, np.nan)

    @staticmethod
    def cross_sectional_zscore(
        signals: pd.DataFrame,
        method: str = "rank",
    ) -> pd.DataFrame:
        """Cross-sectional normalization of factor signals.

        For each date, transforms the cross-section of raw signals into
        comparable z-scores.

        Args:
            signals: DataFrame of raw signals (rows = dates, columns =
                assets).
            method: Normalization approach:

                * ``'rank'``: Rank-based inverse-normal transform
                  (robust to outliers). Each asset's rank is mapped to
                  ``U(0,1)`` then to ``N(0,1)`` via the inverse CDF.
                * ``'standard'``: Subtract cross-sectional mean and
                  divide by cross-sectional standard deviation.

        Returns:
            DataFrame of z-scores (same shape as *signals*).
        """
        if method == "rank":
            def _rank_zscore(row: pd.Series) -> pd.Series:
                valid = row.dropna()
                if len(valid) < 3:
                    return row * np.nan
                ranks = rankdata(valid, method="average")
                uniform = (ranks - 0.5) / len(ranks)
                z = norm.ppf(uniform)
                result = row.copy() * np.nan
                result[valid.index] = z
                return result

            return signals.apply(_rank_zscore, axis=1)

        if method == "standard":
            mu = signals.mean(axis=1)
            sigma = signals.std(axis=1)
            return signals.sub(mu, axis=0).div(sigma.replace(0, np.nan), axis=0)

        raise ValueError(f"Unknown method '{method}'. Use 'rank' or 'standard'.")

    @staticmethod
    def composite_score(
        signals_dict: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Weighted composite factor score.

        Combines multiple z-scored signals into a single composite by
        weighted average. All signals are first aligned on their
        intersection of dates and assets.

        Args:
            signals_dict: Mapping of ``{factor_name: z-score_df}``.
            weights: Mapping of ``{factor_name: weight}``. Defaults to
                equal weight if ``None``.

        Returns:
            DataFrame of composite z-scores.
        """
        names = list(signals_dict.keys())
        if weights is None:
            weights = {n: 1.0 / len(names) for n in names}

        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

        composite = None
        for name, df in signals_dict.items():
            w = weights.get(name, 0)
            contribution = df * w
            composite = contribution if composite is None else composite.add(
                contribution, fill_value=0,
            )

        return composite

    @staticmethod
    def assign_quintiles(
        scores: pd.DataFrame,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """Assign assets to quantile bins cross-sectionally.

        Args:
            scores: DataFrame of factor scores.
            n_quantiles: Number of quantile groups.

        Returns:
            DataFrame of integer labels (1 = top, *n_quantiles* =
            bottom).
        """
        def _qcut_row(row: pd.Series) -> pd.Series:
            valid = row.dropna()
            if len(valid) < n_quantiles:
                return row * np.nan
            try:
                labels = pd.qcut(valid, n_quantiles, labels=False) + 1
            except ValueError:
                labels = pd.qcut(valid.rank(method="first"), n_quantiles, labels=False) + 1
            result = row.copy() * np.nan
            result[valid.index] = labels
            return result

        return scores.apply(_qcut_row, axis=1)

    @staticmethod
    def information_coefficient(
        signals: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> pd.Series:
        """Cross-sectional Spearman IC time series.

        For each date, computes the rank correlation between the
        signal and subsequent realized returns.

        Args:
            signals: Factor z-scores at time *t*.
            forward_returns: Realized returns from *t* to *t + horizon*.

        Returns:
            Series of IC values indexed by date.
        """
        common_dates = signals.index.intersection(forward_returns.index)
        common_cols = signals.columns.intersection(forward_returns.columns)

        ics = {}
        for date in common_dates:
            sig = signals.loc[date, common_cols].dropna()
            ret = forward_returns.loc[date, common_cols].dropna()
            overlap = sig.index.intersection(ret.index)
            if len(overlap) < 5:
                continue
            rho, _ = spearmanr(sig[overlap], ret[overlap])
            ics[date] = rho

        return pd.Series(ics, name="IC")

    @staticmethod
    def factor_decay(
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        horizons: List[int] = None,
    ) -> pd.DataFrame:
        """IC decay analysis across multiple forward horizons.

        Measures how quickly the predictive power of a signal decays
        as the holding period lengthens.

        Args:
            signals: Factor z-scores (monthly or daily).
            returns: Daily asset returns.
            horizons: List of forward horizons in trading days.
                Defaults to ``[21, 42, 63, 126, 252]``.

        Returns:
            DataFrame with columns ``['Horizon', 'Mean IC', 'IC Std',
            'IC_IR', 'Hit Rate']``.
        """
        if horizons is None:
            horizons = [21, 42, 63, 126, 252]

        records = []
        for h in horizons:
            fwd = returns.rolling(h).sum().shift(-h)
            ic_series = FactorEngine.information_coefficient(signals, fwd)
            if ic_series.empty:
                continue
            records.append(
                {
                    "Horizon (days)": h,
                    "Mean IC": ic_series.mean(),
                    "IC Std": ic_series.std(),
                    "IC_IR": (
                        ic_series.mean() / ic_series.std()
                        if ic_series.std() > 0
                        else 0
                    ),
                    "Hit Rate": (ic_series > 0).mean(),
                }
            )

        return pd.DataFrame(records)

    @staticmethod
    def long_short_returns(
        forward_returns: pd.DataFrame,
        quintiles: pd.DataFrame,
        n_quantiles: int = 5,
    ) -> pd.Series:
        """Compute long-short portfolio returns from quintile assignments.

        Goes long the top quintile (Q1) and short the bottom quintile.

        Args:
            forward_returns: Forward-period asset returns.
            quintiles: Quintile assignment DataFrame (1 = top).
            n_quantiles: Number of quantile groups used.

        Returns:
            Series of long-short portfolio returns.
        """
        common = forward_returns.index.intersection(quintiles.index)
        ls_returns = {}

        for date in common:
            q = quintiles.loc[date].dropna()
            r = forward_returns.loc[date].dropna()
            overlap = q.index.intersection(r.index)

            longs = overlap[q[overlap] == 1]
            shorts = overlap[q[overlap] == n_quantiles]

            if len(longs) > 0 and len(shorts) > 0:
                ls_returns[date] = r[longs].mean() - r[shorts].mean()

        return pd.Series(ls_returns, name="L/S Return")

