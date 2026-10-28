"""
Publication-quality financial visualization.

Provides a consistent, professional aesthetic for quantitative finance
charts — cumulative returns, drawdowns, rolling metrics, heatmaps,
and factor exposure plots — with automatic dark/light mode support.

Example:
    >>> from utils.visualization import QuantPlotter
    >>> QuantPlotter.set_style()
    >>> fig = QuantPlotter.plot_cumulative_returns({'Strategy': returns})
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes


PALETTE = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent1": "#F18F01",
    "accent2": "#C73E1D",
    "accent3": "#3B1F2B",
    "accent4": "#44BBA4",
    "positive": "#2E8B57",
    "negative": "#DC143C",
    "neutral": "#6C757D",
    "background": "#FAFBFC",
    "grid": "#E8ECEF",
}

SERIES_COLORS = [
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["accent1"],
    PALETTE["accent2"],
    PALETTE["accent4"],
    PALETTE["accent3"],
    "#7B68EE",
    "#20B2AA",
]


class QuantPlotter:
    """Professional financial chart generator.

    Provides static methods that return ``matplotlib.figure.Figure``
    objects for common quantitative finance visualizations. Call
    :meth:`set_style` once at the start of a notebook to apply the
    professional theme globally.

    Example:
        >>> QuantPlotter.set_style()
        >>> fig = QuantPlotter.plot_cumulative_returns(
        ...     {'Momentum': mom_returns, 'Value': val_returns},
        ...     title='Factor Performance',
        ... )
        >>> fig.savefig('performance.png', dpi=150, bbox_inches='tight')
    """

    @staticmethod
    def set_style() -> None:
        """Apply the professional chart style globally.

        Sets font sizes, grid style, background colors, and other
        ``rcParams`` for a clean, institutional look.
        """
        plt.rcParams.update(
            {
                "figure.figsize": (12, 6),
                "figure.facecolor": PALETTE["background"],
                "figure.dpi": 100,
                "axes.facecolor": PALETTE["background"],
                "axes.edgecolor": PALETTE["grid"],
                "axes.grid": True,
                "axes.axisbelow": True,
                "axes.labelsize": 11,
                "axes.titlesize": 14,
                "axes.titleweight": "bold",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "grid.color": PALETTE["grid"],
                "grid.linewidth": 0.8,
                "grid.alpha": 0.7,
                "lines.linewidth": 1.8,
                "lines.antialiased": True,
                "font.size": 10,
                "font.family": "sans-serif",
                "legend.framealpha": 0.9,
                "legend.edgecolor": PALETTE["grid"],
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
            }
        )

    @staticmethod
    def plot_cumulative_returns(
        returns_dict: Dict[str, pd.Series],
        title: str = "Cumulative Returns",
        log_scale: bool = False,
        figsize: Tuple[int, int] = (12, 6),
    ) -> Figure:
        """Multi-strategy cumulative return chart.

        Args:
            returns_dict: ``{name: return_series}`` mapping.
            title: Chart title.
            log_scale: Use log scale on the y-axis.
            figsize: Figure dimensions ``(width, height)``.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        for i, (name, returns) in enumerate(returns_dict.items()):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            cum = (1 + returns).cumprod()
            ax.plot(cum.index, cum.values, label=name, color=color)

        if log_scale:
            ax.set_yscale("log")

        ax.set_title(title)
        ax.set_ylabel("Growth of $1")
        ax.legend(loc="upper left")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_drawdown(
        returns: pd.Series,
        title: str = "Underwater Chart",
        color: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4),
    ) -> Figure:
        """Drawdown (underwater) chart.

        Args:
            returns: Strategy returns.
            title: Chart title.
            color: Fill color. Defaults to ``PALETTE['negative']``.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        color = color or PALETTE["negative"]

        cum = (1 + returns).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()

        ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.35)
        ax.plot(dd.index, dd.values, color=color, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8)

        ax.set_title(title)
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_rolling_metrics(
        returns: pd.Series,
        window: int = 252,
        title: str = "Rolling Performance",
        figsize: Tuple[int, int] = (12, 8),
    ) -> Figure:
        """Rolling Sharpe ratio and volatility (dual panel).

        Args:
            returns: Strategy returns.
            window: Rolling window size.
            title: Chart title.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure with two subplots.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

        ax1.plot(rolling_sharpe.index, rolling_sharpe.values, color=PALETTE["primary"])
        ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax1.axhline(
            rolling_sharpe.mean(),
            color=PALETTE["accent1"],
            linewidth=1,
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {rolling_sharpe.mean():.2f}",
        )
        ax1.set_title(f"{title} — {window}-Day Rolling Sharpe Ratio")
        ax1.set_ylabel("Sharpe Ratio")
        ax1.legend(loc="upper right")

        rolling_vol = rolling_std * np.sqrt(252)
        ax2.plot(rolling_vol.index, rolling_vol.values, color=PALETTE["secondary"])
        ax2.axhline(
            rolling_vol.mean(),
            color=PALETTE["accent1"],
            linewidth=1,
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {rolling_vol.mean():.1%}",
        )
        ax2.set_title(f"{window}-Day Rolling Volatility")
        ax2.set_ylabel("Annualized Volatility")
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax2.legend(loc="upper right")

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_correlation_heatmap(
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix",
        annotate: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        cmap: str = "RdBu_r",
    ) -> Figure:
        """Annotated correlation heatmap.

        Args:
            corr_matrix: Square correlation DataFrame.
            title: Chart title.
            annotate: Show correlation values in cells.
            figsize: Figure size; auto-scaled if ``None``.
            cmap: Matplotlib colormap name.

        Returns:
            Matplotlib Figure.
        """
        n = len(corr_matrix)
        if figsize is None:
            size = max(6, n * 0.8)
            figsize = (size, size)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr_matrix.index)

        if annotate and n <= 15:
            for i in range(n):
                for j in range(n):
                    val = corr_matrix.iloc[i, j]
                    color = "white" if abs(val) > 0.6 else "black"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color=color, fontsize=8,
                    )

        fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_factor_exposure(
        loadings: pd.Series,
        title: str = "Factor Exposures",
        figsize: Tuple[int, int] = (10, 5),
    ) -> Figure:
        """Horizontal bar chart of factor loadings / exposures.

        Args:
            loadings: Series with factor names as index and loadings
                as values.
            title: Chart title.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = [
            PALETTE["positive"] if v >= 0 else PALETTE["negative"]
            for v in loadings.values
        ]
        ax.barh(loadings.index, loadings.values, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Loading")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_weights(
        weights_dict: Dict[str, pd.Series],
        title: str = "Portfolio Weights Comparison",
        figsize: Tuple[int, int] = (14, 6),
    ) -> Figure:
        """Grouped bar chart comparing portfolio weights across methods.

        Args:
            weights_dict: ``{method_name: weights_series}`` where each
                Series is indexed by asset name.
            title: Chart title.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        df = pd.DataFrame(weights_dict)
        n_assets = len(df)
        n_methods = len(df.columns)
        bar_width = 0.8 / n_methods
        x = np.arange(n_assets)

        for i, col in enumerate(df.columns):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            ax.bar(
                x + i * bar_width, df[col].values,
                width=bar_width, label=col, color=color, edgecolor="white",
            )

        ax.set_xticks(x + bar_width * (n_methods - 1) / 2)
        ax.set_xticklabels(df.index, rotation=45, ha="right")
        ax.set_ylabel("Weight")
        ax.set_title(title)
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_performance_dashboard(
        returns_dict: Dict[str, pd.Series],
        title: str = "Performance Dashboard",
        figsize: Tuple[int, int] = (16, 12),
    ) -> Figure:
        """Four-panel performance dashboard.

        Panels: cumulative returns, drawdowns, rolling Sharpe, and
        return distribution.

        Args:
            returns_dict: ``{strategy_name: returns}``.
            title: Overall figure title.
            figsize: Figure dimensions.

        Returns:
            Matplotlib Figure with four subplots.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        ax = axes[0, 0]
        for i, (name, rets) in enumerate(returns_dict.items()):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            cum = (1 + rets).cumprod()
            ax.plot(cum.index, cum.values, label=name, color=color)
        ax.set_title("Cumulative Returns")
        ax.set_ylabel("Growth of $1")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        for i, (name, rets) in enumerate(returns_dict.items()):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            cum = (1 + rets).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            ax.plot(dd.index, dd.values, label=name, color=color, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Drawdowns")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        ax = axes[1, 0]
        for i, (name, rets) in enumerate(returns_dict.items()):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            rm = rets.rolling(252).mean()
            rs = rets.rolling(252).std()
            sharpe = (rm / rs) * np.sqrt(252)
            ax.plot(sharpe.index, sharpe.values, label=name, color=color)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Rolling 252-Day Sharpe")
        ax.set_ylabel("Sharpe Ratio")

        ax = axes[1, 1]
        for i, (name, rets) in enumerate(returns_dict.items()):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            ax.hist(
                rets.values, bins=50, alpha=0.5,
                label=name, color=color, edgecolor="white",
            )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Return Distribution")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

        fig.tight_layout()
        return fig

