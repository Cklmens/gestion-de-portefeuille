#!/usr/bin/env python3
"""
portfolio_oop.py

Single-file OOP implementation for portfolio data ingestion, distribution fitting,
correlation/covariance estimation with quantile shift, copula-based simulation,
portfolio sampling and simple optimizers (Max Sharpe from simulated portfolios,
Global Minimum Variance via quadratic optimization).

Requirements:
    pip install pandas numpy yfinance scipy scikit-learn matplotlib

Author: (adapted from user's code)
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as sts
from scipy import optimize
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# -------------------------
# Utilities
# -------------------------
def ensure_positive_definite(mat: np.ndarray, eps: float = 1e-8, max_tries: int = 10) -> np.ndarray:
    """
    Make a symmetric matrix positive definite by adding eps to diagonal repeatedly.
    Returns a PD matrix suitable for Cholesky.
    """
    A = (mat + mat.T) / 2.0
    k = 0
    while k < max_tries:
        try:
            np.linalg.cholesky(A)
            return A
        except np.linalg.LinAlgError:
            eig_min = np.min(np.linalg.eigvalsh(A))
            adjust = max(eps, -eig_min + eps)
            A += np.eye(A.shape[0]) * adjust
            k += 1
    raise np.linalg.LinAlgError("Could not make matrix positive definite after tries")


def random_weights(n: int) -> np.ndarray:
    """Generate random weights summing to 1 (integers 1..9 normalized)."""
    w = np.random.randint(1, 10, size=n).astype(float)
    return w / w.sum()


# -------------------------
# Data class
# -------------------------
class PortfolioData:
    """
    Download prices and compute returns.
    """
    def __init__(self, tickers: List[str], start: str, end: str):
        self.tickers = tickers
        self.start = start
        self.end = end
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None

    def download(self) -> pd.DataFrame:
        df = yf.download(self.tickers, start=self.start, end=self.end)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df = df.dropna(how='all')  # drop rows with all NaN
        self.prices = df
        self.returns = df.pct_change().dropna()
        return self.returns


# -------------------------
# Covariance / Correlation estimator
# -------------------------
class CovarianceEstimator:
    """
    Estimate rolling correlation median and a quantile-shifted correlation matrix,
    then build covariance from returns standard deviations.
    """
    def __init__(self, returns: pd.DataFrame, moving_window: int = 30, picking: int = 1, quantile_shift: float = 0.05):
        if returns is None:
            raise ValueError("returns must not be None")
        self.returns = returns.iloc[::picking].reset_index(drop=True)
        self.window = int(moving_window)
        self.quantile_shift = float(quantile_shift)
        self.assets = list(self.returns.columns)
        self.n = len(self.assets)

    def rolling_pair_quantile(self, s1: str, s2: str, q: float) -> float:
        series = self.returns[s1].rolling(self.window).corr(self.returns[s2]).dropna()
        if len(series) == 0:
            return 0.0
        return float(series.quantile(q))

    def corr_median(self) -> pd.DataFrame:
        corr = pd.DataFrame(np.eye(self.n), index=self.assets, columns=self.assets)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                s1, s2 = self.assets[i], self.assets[j]
                rho50 = self.rolling_pair_quantile(s1, s2, 0.5)
                corr.loc[s1, s2] = corr.loc[s2, s1] = rho50
        return corr

    def corr_shifted(self) -> pd.DataFrame:
        corr_s = pd.DataFrame(np.eye(self.n), index=self.assets, columns=self.assets)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                s1, s2 = self.assets[i], self.assets[j]
                rho50 = self.rolling_pair_quantile(s1, s2, 0.5)
                if rho50 < 0:
                    q = 0.5 + self.quantile_shift
                else:
                    q = 0.5 - self.quantile_shift
                rho_s = self.rolling_pair_quantile(s1, s2, q)
                corr_s.loc[s1, s2] = corr_s.loc[s2, s1] = rho_s
        np.fill_diagonal(corr_s.values, 1.0)
        return corr_s

    def covariance_from_corr(self, corr: pd.DataFrame) -> pd.DataFrame:
        std = self.returns.std().values
        D = np.diag(std)
        cov = D @ corr.values @ D
        return pd.DataFrame(cov, index=self.assets, columns=self.assets)


# -------------------------
# Distribution fitter
# -------------------------
class DistributionFitter:
    """
    Fit a set of candidate distributions to a 1D data array and rank them.
    """
    CANDIDATES = [
        'norm', 't', 'lognorm', 'pareto', 'logistic', 'genextreme', 'invgauss', 'invgamma'
    ]

    def __init__(self, data: pd.Series):
        self.data = data.dropna().values
        self.n = len(self.data)

    def fit_all(self, bins: int = 100) -> List[Dict[str, Any]]:
        if self.n == 0:
            return []
        y, xedges = np.histogram(self.data, bins=bins, density=True)
        x = (xedges[:-1] + xedges[1:]) / 2.0
        results = []

        for name in self.CANDIDATES:
            try:
                dist = getattr(sts, name)
            except AttributeError:
                continue
            try:
                params = dist.fit(self.data)
                *shape, loc, scale = params
                pdf = dist.pdf(x, *shape, loc=loc, scale=scale)
                if np.any(np.isnan(pdf)) or np.any(np.isinf(pdf)):
                    continue
                sse = np.sum((y - pdf) ** 2)
                r2 = r2_score(y, pdf)
                loglik = np.sum(dist.logpdf(self.data, *shape, loc=loc, scale=scale))
                k = len(params)
                AIC = -2 * loglik + 2 * k
                BIC = -2 * loglik + k * np.log(self.n)
                ks_stat, ks_p = sts.kstest(self.data, name, args=params)
                results.append({
                    "name": name,
                    "params": params,
                    "sse": sse,
                    "r2": r2,
                    "AIC": AIC,
                    "BIC": BIC,
                    "KS_stat": ks_stat,
                    "KS_pvalue": ks_p,
                    "KS_pass": ks_p > 0.05
                })
            except Exception:
                continue

        results_sorted = sorted(results, key=lambda d: (d["AIC"], d["BIC"]))
        return results_sorted

    def best(self) -> Optional[Dict[str, Any]]:
        allr = self.fit_all()
        return allr[0] if len(allr) > 0 else None


# -------------------------
# Simulator using Gaussian copula
# -------------------------
class PortfolioSimulator:
    """
    Simulate correlated returns using a Gaussian copula, then map to marginal laws.
    data_info: list of dicts: {"asset": name, "law": name, "params": params}
    corr: correlation matrix (pandas DataFrame or np.ndarray)
    """
    def __init__(self, data_info: List[Dict[str, Any]], corr: np.ndarray, horizon: int = 63, sims: int = 20000,
                 random_state: Optional[int] = None):
        self.data_info = data_info
        self.corr = np.asarray(corr)
        self.horizon = horizon
        self.sims = sims
        self.rng = np.random.default_rng(random_state)
        if self.corr.shape[0] != len(data_info):
            raise ValueError("Dimension mismatch between corr and data_info")

    def _cholesky(self) -> np.ndarray:
        # ensure PD
        corr_pd = ensure_positive_definite(self.corr)
        return np.linalg.cholesky(corr_pd)

    def simulate_once(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate `sims` random portfolios by:
         - generating correlated gaussian draws (assets x horizon)
         - mapping marginals via the estimated distributions (copula)
         - sampling a random weight for each simulation and computing portfolio mean & std
        Returns:
            W: (sims, n_assets) weights
            returns_mean: (sims,)
            returns_std: (sims,)
        """
        n = len(self.data_info)
        L = self._cholesky()
        W = np.zeros((self.sims, n))
        r_means = np.zeros(self.sims)
        r_stds = np.zeros(self.sims)

        # We'll vectorize across horizon: generate all Gaussian draws needed in blocks to avoid massive mem use
        # For each simulation, we'll draw horizon samples for each asset (vectorized across sims)
        # Draw shape: (sims, horizon, n) of standard normals, then transform per asset via cholesky
        # To reduce memory, generate in chunks if sims too large.
        chunk = 5000
        idx = 0
        while idx < self.sims:
            cnt = min(chunk, self.sims - idx)
            # Raw normals: shape (cnt, horizon, n)
            Z = self.rng.standard_normal(size=(cnt, self.horizon, n))
            # Introduce correlations per time step: for each t, multiply (cnt, n) by L.T
            # Equivalent: for each sim, compute for all t the correlated normals
            # We'll reshape to (cnt * horizon, n), multiply by L.T
            Z2 = Z.reshape(-1, n) @ L.T  # (cnt * horizon, n)
            Z2 = Z2.reshape(cnt, self.horizon, n)
            # Now map marginals: for each asset i, transform column via cdf->ppf
            mapped = np.zeros_like(Z2)  # (cnt, horizon, n)
            for i, info in enumerate(self.data_info):
                law = info["law"]
                params = info["params"]
                # CDF of normal
                U = sts.norm.cdf(Z2[:, :, i])
                # ppf of target law
                dist = getattr(sts, law)
                # params may include shape/loc/scale; pass as positional + loc, scale
                try:
                    mapped[:, :, i] = dist.ppf(U, *params)
                except Exception:
                    # fallback to using empirical quantiles if ppf fails
                    mapped[:, :, i] = np.quantile(info.get("empirical", np.zeros(1)), U.flatten()).reshape(cnt, self.horizon)

            # For each sim in chunk, sample random weight and compute portfolio returns series
            for k in range(cnt):
                w = random_weights(n)
                port_series = (w @ mapped[k].T)  # horizon-length series
                W[idx + k] = w
                r_means[idx + k] = np.mean(port_series)
                r_stds[idx + k] = np.std(port_series)
            idx += cnt

        return W, r_means, r_stds


# -------------------------
# Optimizer utilities
# -------------------------
class PortfolioOptimizer:
    """
    Simple optimizers:
      - Max Sharpe from simulated portfolios (takes simulation outputs)
      - GMV via quadratic programming (cvx can be used; here we use scipy minimize)
    """
    @staticmethod
    def max_sharpe_from_simulation(W: np.ndarray, means: np.ndarray, sigmas: np.ndarray, rf: float = 0.0) -> Dict[str, Any]:
        sharpe = (means - rf) / (sigmas + 1e-12)
        idx = np.nanargmax(sharpe)
        return {"weights": W[idx], "return": means[idx], "sigma": sigmas[idx], "sharpe": sharpe[idx]}

    @staticmethod
    def gmv_from_cov(cov: np.ndarray, bounds: Optional[Tuple[Tuple[float, float], ...]] = None) -> Dict[str, Any]:
        n = cov.shape[0]
        cov = np.asarray(cov)

        def fun(w):
            return w @ cov @ w

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        x0 = np.repeat(1.0 / n, n)
        if bounds is None:
            bounds = tuple((0.0, 1.0) for _ in range(n))

        res = optimize.minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons)
        if not res.success:
            raise RuntimeError("GMV optimization failed: " + str(res.message))
        w = res.x
        return {"weights": w, "sigma": np.sqrt(w @ cov @ w)}

    @staticmethod
    def efficient_frontier(cov: np.ndarray, returns: np.ndarray, points: int = 20):
        # Simple frontier by solving min variance for target returns
        n = len(returns)
        cov = np.asarray(cov)

        def min_variance_for_return(target):
            # Optimize variance w.r.t constraint w^T mu = target and sum w = 1, w>=0
            mu = returns

            def fun(w):
                return w @ cov @ w

            cons = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: w @ mu - target},
            )
            x0 = np.repeat(1.0 / n, n)
            bounds = tuple((0.0, 1.0) for _ in range(n))
            res = optimize.minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons)
            return res

        targets = np.linspace(np.min(returns), np.max(returns), points)
        sols = []
        for t in targets:
            try:
                r = min_variance_for_return(t)
                if r.success:
                    sols.append({"target": t, "weights": r.x, "sigma": np.sqrt(r.x @ cov @ r.x)})
            except Exception:
                continue
        return sols


# -------------------------
# Metrics
# -------------------------
class RiskMetrics:
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        cumulative = np.cumprod(1 + np.asarray(returns))
        cumulative_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - cumulative_max) / cumulative_max
        return float(drawdowns.min())

    @staticmethod
    def sortino(returns: np.ndarray, rf: float = 0.0) -> float:
        excess = np.asarray(returns) - rf
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float(np.inf)
        downside_dev = np.sqrt(np.mean(downside ** 2))
        return float(excess.mean() / downside_dev)


# -------------------------
# Main example / CLI
# -------------------------
def main():
    # ---- User config ----
    TICKERS = ["GOOGL", "MSFT", "TSLA", "AAPL", "AMZN"]
    DATE_START = "2025-05-01"
    DATE_END = "2025-08-01"
    MOVING_WINDOW = 30
    PICKING = 1
    QUANTILE_SHIFT = 0.05
    HORIZON = 63
    SIMS = 10000  # lower for quick demo; increase if you want more precision
    RF = 0.0

    print("Downloading data...")
    pd_data = PortfolioData(tickers=TICKERS, start=DATE_START, end=DATE_END)
    returns = pd_data.download()
    print(f"Downloaded returns shape: {returns.shape}")

    # Covariance estimation
    cov_est = CovarianceEstimator(returns=returns, moving_window=MOVING_WINDOW, picking=PICKING, quantile_shift=QUANTILE_SHIFT)
    corr_med = cov_est.corr_median()
    corr_shift = cov_est.corr_shifted()
    cov_med = cov_est.covariance_from_corr(corr_med)
    cov_shifted = cov_est.covariance_from_corr(corr_shift)

    print("Estimated correlation (median):")
    print(corr_med.round(3))

    # Fit distributions per asset
    print("Fitting marginal distributions (this can be slow)...")
    data_info = []
    for col in returns.columns:
        fitter = DistributionFitter(returns[col])
        best = fitter.best()
        if best is None:
            # fallback to empirical distribution description:
            emp = np.array(returns[col].dropna())
            # approximate by normal if nothing else
            mu, sigma = emp.mean(), emp.std()
            best = {"name": "norm", "params": (mu, sigma)}
            best["params"] = (mu, sigma)  # Note: norm.fit returns (loc, scale) but our code expects (shape..., loc, scale)
            # adapt to standard expected param order: (loc, scale) -> we will set properly below
            params = (mu, sigma)
            law = "norm"
        else:
            law = best["name"]
            params = best["params"]

        # For safety, ensure params arranged as ( *shape , loc, scale )
        # Also keep empirical distribution for fallback mapping if ppf fails.
        data_info.append({
            "asset": col,
            "law": law,
            "params": params,
            "empirical": returns[col].dropna().values
        })
        print(f" - {col}: best law {law}")

    # Simulator
    print("Simulating portfolios with Gaussian copula...")
    simulator = PortfolioSimulator(data_info=data_info, corr=corr_shift.values, horizon=HORIZON, sims=SIMS, random_state=42)
    W, means, sigmas = simulator.simulate_once()
    print("Simulation finished.")

    # Optimizers / results
    print("Selecting max Sharpe from simulated candidates...")
    best_sharpe = PortfolioOptimizer.max_sharpe_from_simulation(W, means, sigmas, rf=RF)
    print("Max Sharpe candidate:")
    print(f"  Sharpe: {best_sharpe['sharpe']:.4f}, Return: {best_sharpe['return']:.4f}, Sigma: {best_sharpe['sigma']:.4f}")
    print(f"  Weights: {dict(zip(returns.columns, best_sharpe['weights'].round(3)))}")

    print("Computing GMV from shifted covariance...")
    try:
        gmv = PortfolioOptimizer.gmv_from_cov(cov_shifted.values)
        print("GMV weights:")
        print(dict(zip(returns.columns, gmv["weights"].round(4))))
        print(f"GMV sigma: {gmv['sigma']:.4f}")
    except Exception as e:
        print("GMV optimization failed:", e)

    # Plot simulated efficient cloud and highlight selected points
    try:
        plt.figure(figsize=(8, 5))
        plt.scatter(sigmas, means, alpha=0.2, s=6, label="Simulated portfolios")
        plt.scatter([best_sharpe["sigma"]], [best_sharpe["return"]], color='red', label="MaxSharpe", zorder=10)
        plt.scatter([gmv["sigma"]], [(np.dot(gmv["weights"], [np.mean(data) for data in returns.T]))], color='green', label="GMV (sigma)", zorder=10)
        plt.xlabel("Volatility (std)")
        plt.ylabel("Mean return")
        plt.title("Portfolio cloud (simulated)")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    print("Done.")


if __name__ == "__main__":
    main()
