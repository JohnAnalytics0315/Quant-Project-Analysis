"""
Data loading utilities with multi-source API integration.

Supports Yahoo Finance (yfinance), FRED (Federal Reserve Economic Data),
and synthetic data generation for testing and fallback. Implements disk
and memory caching to minimize redundant API calls.

Typical usage:
    >>> loader = DataLoader(cache_dir='./data')
    >>> prices = loader.fetch_prices(['AAPL', 'MSFT'], '2020-01-01', '2024-01-01')
    >>> rf = loader.get_risk_free_rate('2020-01-01', '2024-01-01')
"""

import logging
import hashlib
from typing import Optional, List, Dict, Union, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Multi-source financial data loader with caching and fallback.

    Fetches real market data from Yahoo Finance and FRED, with automatic
    fallback to synthetic data generation when API access is unavailable.
    Supports both in-memory and Parquet-based disk caching.

    Args:
        cache_dir: Directory for Parquet disk cache. ``None`` disables disk
            caching (in-memory cache still active for the session).

    Attributes:
        cache_dir: Resolved cache directory path, or ``None``.

    Example:
        >>> loader = DataLoader(cache_dir='./data')
        >>> prices = loader.fetch_prices(
        ...     ['AAPL', 'MSFT', 'GOOGL'],
        ...     start='2020-01-01',
        ...     end='2024-01-01',
        ... )
        >>> print(f"Downloaded {prices.shape[1]} tickers, {len(prices)} days")
        Downloaded 3 tickers, 1007 days
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch_prices(
        self,
        tickers: List[str],
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Download adjusted close prices from Yahoo Finance.

        Attempts a live download via ``yfinance``. On failure, falls back
        to correlated geometric Brownian motion (GBM) synthetic data so
        that downstream analysis always receives a valid DataFrame.

        Args:
            tickers: List of ticker symbols (e.g. ``['AAPL', 'MSFT']``).
            start: Start date in ``YYYY-MM-DD`` format.
            end: End date in ``YYYY-MM-DD`` format.
            use_cache: If ``True``, check memory and disk caches first.

        Returns:
            DataFrame with a ``DatetimeIndex`` and one column per ticker.

        Raises:
            ValueError: If *tickers* is empty.
        """
        if not tickers:
            raise ValueError("tickers list must not be empty")

        cache_key = self._make_cache_key(tickers, start, end)

        if use_cache and cache_key in self._cache:
            logger.info("Returning prices from memory cache")
            return self._cache[cache_key]

        if use_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            if cache_file.exists():
                logger.info("Loading prices from disk cache: %s", cache_file)
                df = pd.read_parquet(cache_file)
                self._cache[cache_key] = df
                return df

        try:
            import yfinance as yf

            logger.info(
                "Downloading %d tickers from Yahoo Finance (%s to %s)...",
                len(tickers), start, end,
            )
            data = yf.download(
                tickers, start=start, end=end, progress=False, auto_adjust=True,
            )

            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"].dropna(axis=1)
            else:
                prices = data[["Close"]].dropna()
                prices.columns = tickers[:1]

            if prices.empty:
                raise ValueError("Empty result from Yahoo Finance")

            self._store_cache(cache_key, prices)
            logger.info(
                "Downloaded %d tickers, %d trading days",
                prices.shape[1], len(prices),
            )
            return prices

        except Exception as exc:
            logger.warning("Yahoo Finance download failed: %s", exc)
            logger.info("Falling back to synthetic GBM data")
            return self.generate_synthetic_prices(
                n_assets=len(tickers),
                n_days=252 * 5,
                seed=42,
                columns=tickers,
                start_date=start,
            )

    def fetch_options_chain(
        self,
        ticker: str,
        max_expirations: int = 10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download current options chains from Yahoo Finance.

        Args:
            ticker: Underlying ticker symbol (e.g. ``'SPY'``).
            max_expirations: Maximum number of expiration dates to fetch.

        Returns:
            Tuple of ``(calls_df, puts_df)`` with an ``expiration`` column.
        """
        import yfinance as yf

        asset = yf.Ticker(ticker)
        expirations = asset.options[:max_expirations]

        all_calls, all_puts = [], []
        for exp in expirations:
            chain = asset.option_chain(exp)
            calls = chain.calls.copy()
            calls["expiration"] = exp
            puts = chain.puts.copy()
            puts["expiration"] = exp
            all_calls.append(calls)
            all_puts.append(puts)

        return pd.concat(all_calls, ignore_index=True), pd.concat(all_puts, ignore_index=True)

    def fetch_fred_series(
        self,
        series_id: str,
        start: str,
        end: str,
    ) -> pd.Series:
        """Download a single time series from FRED.

        Uses ``pandas_datareader`` to query the Federal Reserve Economic
        Data API. Returns an empty Series on failure so callers can apply
        a fallback without exception handling.

        Args:
            series_id: FRED series identifier (e.g. ``'DGS10'``,
                ``'T10YIE'``, ``'VIXCLS'``).
            start: Start date string (``YYYY-MM-DD``).
            end: End date string (``YYYY-MM-DD``).

        Returns:
            ``pd.Series`` with a ``DatetimeIndex``. Empty on failure.
        """
        try:
            import pandas_datareader.data as web

            logger.info("Fetching FRED series '%s' (%s to %s)", series_id, start, end)
            data = web.DataReader(series_id, "fred", start, end)
            return data.iloc[:, 0].dropna()
        except Exception as exc:
            logger.warning("FRED download failed for '%s': %s", series_id, exc)
            return pd.Series(dtype=float)

    def get_risk_free_rate(
        self,
        start: str,
        end: str,
        fallback: float = 0.045,
    ) -> pd.Series:
        """Get the daily risk-free rate, with a constant fallback.

        Attempts to download the 10-Year Treasury Constant Maturity Rate
        (``DGS10``) from FRED and converts it to a daily rate. If the
        download fails, returns a constant daily rate derived from
        *fallback*.

        Args:
            start: Start date string.
            end: End date string.
            fallback: Annualized risk-free rate used when FRED is
                unavailable (default ``4.5%``).

        Returns:
            Daily risk-free rate as a decimal (annualized rate / 252).
        """
        series = self.fetch_fred_series("DGS10", start, end)
        if not series.empty:
            logger.info("Using FRED DGS10 for risk-free rate")
            return series / 100 / 252

        logger.info("Using constant risk-free rate: %.2f%%", fallback * 100)
        idx = pd.bdate_range(start, end)
        return pd.Series(fallback / 252, index=idx, name="rf")

    def get_macro_indicators(
        self,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Download a panel of macro indicators from FRED.

        Fetches the 10-Year Treasury yield, breakeven inflation, VIX,
        and the Fed Funds effective rate. Columns with no data are
        dropped silently.

        Args:
            start: Start date string.
            end: End date string.

        Returns:
            DataFrame with one column per indicator (daily frequency).
        """
        indicators = {
            "Treasury_10Y": "DGS10",
            "Breakeven_Inflation": "T10YIE",
            "Fed_Funds": "DFF",
        }
        frames: Dict[str, pd.Series] = {}
        for label, sid in indicators.items():
            s = self.fetch_fred_series(sid, start, end)
            if not s.empty:
                frames[label] = s

        if frames:
            return pd.DataFrame(frames)
        logger.warning("No macro indicators retrieved from FRED")
        return pd.DataFrame()

    @staticmethod
    def generate_synthetic_prices(
        n_assets: int,
        n_days: int = 1260,
        seed: int = 42,
        columns: Optional[List[str]] = None,
        start_date: str = "2019-01-01",
        mu_range: Tuple[float, float] = (0.03, 0.15),
        sigma_range: Tuple[float, float] = (0.15, 0.45),
    ) -> pd.DataFrame:
        """Generate synthetic prices via correlated geometric Brownian motion.

        Produces realistic multi-asset price paths with a random
        correlation structure, suitable as a fallback when live data is
        unavailable or for unit-testing analytics.

        Args:
            n_assets: Number of assets to simulate.
            n_days: Number of trading days (default 5 years).
            seed: NumPy random seed for reproducibility.
            columns: Column names; auto-generated (``Asset_1``, ...) if
                ``None``.
            start_date: Start date for the business-day index.
            mu_range: ``(low, high)`` bounds for annualized drift.
            sigma_range: ``(low, high)`` bounds for annualized volatility.

        Returns:
            DataFrame of simulated adjusted-close prices (initial = 100).
        """
        rng = np.random.RandomState(seed)

        if columns is None:
            columns = [f"Asset_{i + 1}" for i in range(n_assets)]

        dt = 1.0 / 252
        mu = rng.uniform(*mu_range, size=n_assets)
        sigma = rng.uniform(*sigma_range, size=n_assets)

        k = max(n_assets, 3)
        A = rng.randn(n_assets, k) * 0.3
        np.fill_diagonal(A[:min(n_assets, k), :min(n_assets, k)], 1.0)
        cov = A @ A.T
        d_inv = np.diag(1.0 / np.sqrt(np.diag(cov)))
        corr = d_inv @ cov @ d_inv
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-8)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = np.linalg.cholesky(corr)

        Z = rng.randn(n_days, n_assets)
        corr_Z = Z @ L.T

        log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * corr_Z
        prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))

        dates = pd.bdate_range(start_date, periods=n_days)
        return pd.DataFrame(
            prices, index=dates, columns=columns[:n_assets],
        )

    @staticmethod
    def _make_cache_key(tickers: List[str], start: str, end: str) -> str:
        """Deterministic, filesystem-safe cache key."""
        raw = f"{'|'.join(sorted(tickers))}_{start}_{end}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _store_cache(self, key: str, df: pd.DataFrame) -> None:
        """Write to both memory and (optionally) disk cache."""
        self._cache[key] = df
        if self.cache_dir:
            path = self.cache_dir / f"{key}.parquet"
            df.to_parquet(path)
            logger.info("Cached to disk: %s", path)

