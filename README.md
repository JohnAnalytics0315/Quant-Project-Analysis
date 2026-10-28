# Quantitative Trading & Portfolio Analytics

**A suite of quantitative finance analyses using real market data**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## Overview

This repository contains three end-to-end quantitative finance research notebooks spanning factor investing, statistical arbitrage, and derivatives modeling. Each notebook downloads real market data via the Yahoo Finance API, applies institutional-grade analytical methods, and produces publication-quality visualizations. A synthetic-data fallback ensures full reproducibility in any environment.

---

## Table of Contents

| Notebook | Description |
|:---------|:------------|
| [Factor-Based Portfolio Construction](01_Factor_Based_Portfolio_Construction.ipynb) | Constructs momentum, value, quality, and low-volatility factors across 30 U.S. equities; builds quintile long-short portfolios with Fama-MacBeth regression analysis |
| [Statistical Arbitrage & Pairs Trading](02_Statistical_Arbitrage_Pairs_Trading.ipynb) | Tests cointegration on real stock pairs (KO/PEP, XOM/CVX, GLD/GDX), fits Ornstein-Uhlenbeck models, and validates strategies with Monte Carlo simulation |
| [Volatility Surface Modeling](03_Volatility_Surface_Modeling.ipynb) | Extracts implied volatilities from live SPY options, fits SVI and SABR models, constructs Greeks surfaces, and simulates delta hedging |

---

## Project Structure

```
Quant-Project-Analysis/
├── 01_Factor_Based_Portfolio_Construction.ipynb
├── 02_Statistical_Arbitrage_Pairs_Trading.ipynb
├── 03_Volatility_Surface_Modeling.ipynb
├── config/
│   ├── __init__.py
│   └── settings.py            — Centralized configuration (tickers, parameters)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         — Multi-source data acquisition (yfinance, FRED)
│   ├── performance.py         — Risk-return analytics (Sharpe, Sortino, VaR, CVaR)
│   ├── risk.py                — Covariance estimation, risk decomposition, stress testing
│   ├── factors.py             — Cross-sectional factor construction & IC analysis
│   └── visualization.py       — Publication-quality plotting & style management
├── data/                      — Cached datasets (auto-generated, git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Key Highlights

- **Real market data** — Every notebook downloads live data from Yahoo Finance; a synthetic fallback guarantees offline reproducibility
- **Multi-API integration** — Combines yfinance (equities, options) with FRED (risk-free rates, macro indicators) via `pandas-datareader`
- **Rigorous statistical testing** — Newey-West corrected t-statistics, Engle-Granger and Johansen cointegration tests, and walk-forward validation
- **Publication-quality output** — Every chart is presentation-ready with proper labels, annotations, and consistent styling
- **Production-grade code** — Type-hinted functions, docstrings, centralized configuration, and modular utilities

---

## Methodology

### 1. Factor-Based Portfolio Construction

Implements the Fama-French cross-sectional framework on 30 S&P 500 constituents across 7 sectors. Four factors (momentum, value, quality, low volatility) are computed as rank-based inverse-normal z-scores each month. Quintile portfolios are formed on a composite score, and factor premia are estimated via Fama-MacBeth regressions with Newey-West standard errors. Out-of-sample validation uses a 60/40 split with Nelder-Mead weight optimization.

### 2. Statistical Arbitrage & Pairs Trading

Tests five real equity pairs for cointegration using both Engle-Granger (two-step) and Johansen (multivariate) procedures. Confirmed cointegrated spreads are modeled as Ornstein-Uhlenbeck processes with estimated mean-reversion speed and half-life. A z-score-based trading strategy (entry at |z| > 2, exit at |z| < 0.5, stop-loss at |z| > 3) is backtested and validated with 1,000-path Monte Carlo simulation and Markov regime detection.

### 3. Volatility Surface Modeling & Options Analytics

Pulls live SPY options chains and extracts implied volatilities via Brent's root-finding method applied to the Black-Scholes formula. The resulting volatility surface is fit with both the SVI (Gatheral, 2004) and SABR (Hagan, 2002) parametric models, with calendar and butterfly arbitrage checks. Includes full Greeks surface computation, delta hedging simulation (single-path and 1,000-path Monte Carlo), and put-call parity verification.

---

## Tech Stack

| Category | Libraries |
|:---------|:----------|
| **Core** | NumPy, Pandas, SciPy |
| **Data** | yfinance, pandas-datareader (FRED) |
| **Statistics** | statsmodels (Fama-MacBeth, Markov Switching, cointegration tests) |
| **Visualization** | Matplotlib, Seaborn |
| **Volatility Models** | arch (GARCH) |
| **Environment** | Jupyter Notebook, Python 3.10+ |

---

## Quick Start

```bash
git clone https://github.com/JohnAnalytics0315/Quant-Project-Analysis.git
cd Quant-Project-Analysis

pip install -r requirements.txt

jupyter notebook
```

Open any notebook and Run All — data downloads automatically. No API keys required.

---

## Results Summary

| Analysis | Key Metric | Value |
|:---------|:-----------|:------|
| Factor Portfolio | Long-short annualized Sharpe | Run notebook for live results |
| Pairs Trading (KO/PEP) | Cointegration p-value | Run notebook for live results |
| Volatility Surface | SVI fit RMSE (avg across slices) | Run notebook for live results |

> Results depend on the data window. Run the notebooks to generate current figures.

---

## Future Enhancements

- **ML alpha signal research** — Walk-forward cross-validation with gradient boosting, SHAP explainability, and permutation-based overfitting diagnostics
- **Risk parity & portfolio optimization** — Markowitz, Black-Litterman, HRP, and risk parity comparison across multi-asset ETFs
- **Live trading integration** — Connect to Interactive Brokers or Alpaca API for paper trading
- **Alternative data** — Incorporate sentiment (news/social media NLP), satellite imagery, or web traffic signals
- **Real-time dashboard** — Streamlit or Dash app for live portfolio monitoring

---

## Author

**Arjohn Calig-onan**
[GitHub](https://github.com/JohnAnalytics0315)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
