# Portfolio Risk & Simulation Toolkit

A concise Python toolkit for estimating correlations, fitting return
distributions, and running portfolio simulations using copulas and Monte
Carlo.

## Features

-   Download market data with **yfinance**
-   Compute **rolling median correlations** and apply **quantile-based
    stress shifts**
-   Fit several probability distributions (Normal, Student-t, Lognormal,
    Pareto, etc.)
-   Build **Gaussian copula--based simulations**
-   Generate random portfolio weights and evaluate:
    -   Expected return\
    -   Volatility\
    -   Max Sharpe portfolio\
    -   Global Minimum Variance\
-   Utility risk metrics (Sortino ratio, max drawdown)

## Main Functions

### `cov_matrix_shift()`

Computes median rolling correlations and applies a sign-aware quantile
shift to stress correlations.

### `distribution_fit()`

Fits several distributions and returns ranking using AIC/BIC + KS test.

### `copule_gaussian()`

Transforms Gaussian samples into target distribution samples using a
copula.

### `PortfolioStat()`

Simulates portfolios, returns weights, returns, and volatilities.

### `MaxSharpeRate()` / `GMV()`

Identifies optimal portfolio configurations.

## How to Use

    tickers = ["GOOGL", "MSFT", "GC=F", "TSLA", "AAPL"]
    corr_shifted, corr_med = cov_matrix_shift(tickers, "2025-08-01", moving_window=30)
    returns = data_returns(tickers, "2025-05-01", "2025-08-01")

## Dependencies

-   pandas\
-   numpy\
-   yfinance\
-   scipy\
-   sklearn\
-   matplotlib

## Author

Clément --- Portfolio analysis & risk modeling toolkit.

