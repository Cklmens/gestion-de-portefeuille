#%%
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats as sts
from sklearn.metrics import r2_score

#%% 

def cov_matrix_shift(
        tickers, date_end,
        moving_window: int,
        years_back: int = 5,
        quantile_shift: float = 0.05,
        picking: int = 1,
        bound: float = 0.999
    ):

    # --- Convertir date si string ---
    date_end = pd.to_datetime(date_end)
    date_start = date_end - pd.DateOffset(years=years_back)
    # Reconvertir en string ISO pour yfinance
    date_start = date_start.strftime("%Y-%m-%d")
    date_end = date_end.strftime("%Y-%m-%d")

    # --- Télécharger les prix ---
    data = yf.download(tickers, start=date_start, end=date_end)["Close"]

    # --- Calcul des rendements ---
    returns = data.pct_change().dropna()

    # --- Subsampling ---
    if picking > 1:
        returns = returns.iloc[::picking].reset_index(drop=True)

    assets = list(returns.columns)
    n = len(assets)

    # --- Rolling correlations median ---
    corr_median = pd.DataFrame(np.eye(n), index=assets, columns=assets)

    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = assets[i], assets[j]

            series = returns[s1].rolling(moving_window).corr(returns[s2]).dropna()
            rho50 = series.quantile(0.5) if len(series) > 0 else 0.0

            corr_median.loc[s1, s2] = rho50
            corr_median.loc[s2, s1] = rho50

    # --- Apply “sign-aware” quantile shift ---
    corr_shifted = corr_median.copy()

    for i in range(n):
        for j in range(i + 1, n):

            s1, s2 = assets[i], assets[j]

            series = returns[s1].rolling(moving_window).corr(returns[s2]).dropna()

            rho50 = series.quantile(0.5) if len(series) > 0 else 0.0
            if rho50<0:
                rho_s=series.quantile(0.5+quantile_shift) if len(series) > 0 else 0.0
            else:
                rho_s=series.quantile(0.5-quantile_shift) if len(series) > 0 else 0.0

            corr_shifted.loc[s1, s2] = rho_s
            corr_shifted.loc[s2, s1] = rho_s


    np.fill_diagonal(corr_shifted.values, 1.0)

    # --- Covariance matrices ---
    std = returns.std().values
    D = np.diag(std)

    cov_median = D @ corr_median.values @ D
    cov_shifted = D @ corr_shifted.values @ D

    cov_median = pd.DataFrame(cov_median, index=assets, columns=assets)
    cov_shifted = pd.DataFrame(cov_shifted, index=assets, columns=assets)

    return  corr_shifted, corr_median


def get_quantile(data):
    """
    Return key quantiles : 5%, 25%, 50%, 75%, 95%.
    """
    q = data.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return q

def data_returns(tickers, date1, date2):
    """
    Download adjusted close prices for a list of tickers
    and return a DataFrame of daily returns.
    """
    # Download all tickers
    data = yf.download(tickers, start=date1, end=date2)["Close"]

    # Convert to DataFrame if only one ticker
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Compute returns for all tickers in one vectorized operation
    returns = data.pct_change().dropna()

    return returns
 
def copule_gaussian(normal_data, target_law, target_param):
    # Normal to Uniform  
    Unif=sts.norm.cdf(normal_data)
    *arg, loc, scale = target_param
    target_name=target_law
    # scipy distribution object
    dist = getattr(sts, target_name)

    target_data = dist.ppf(Unif, *arg, loc=loc, scale=scale)
    return target_data

 
def distribution_fit(data):
    """
    Fit several relevant financial distributions to 'data'.
    Compute SSE, R², AIC, BIC, KS test and KS_pass (boolean).
    Return sorted list by AIC (best model first).
    """

    list_loi = [
        'norm','t','lognorm','pareto','logistic','genextreme','invgauss','invgamma'
    ]

    # Histogram for SSE/R² evaluation
    y, x = np.histogram(data, bins=100, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    n = len(data)
    results = []
 
    for loi_name in list_loi:
        try:
            dist = getattr(sts, loi_name)
            params = dist.fit(data)

            *shape, loc, scale = params

            # PDF estimation on histogram bins
            pdf = dist.pdf(x, *shape, loc=loc, scale=scale)
            if np.isnan(pdf).any() or np.isinf(pdf).any():
                continue
            # SSE
            sse = np.sum((y - pdf)**2)
            # R²
            r2 = r2_score(y, pdf)
            # Log-likelihood
            loglik = np.sum(dist.logpdf(data, *shape, loc=loc, scale=scale))
            # Parameters count
            k = len(params)
            # AIC / BIC
            AIC = -2 * loglik + 2 * k
            BIC = -2 * loglik + k * np.log(n)
            # KS test
            ks_stat, ks_pvalue = sts.kstest(data, loi_name, args=params)

            # Pass/fail condition for Kolmogorov–Smirnov
            KS_pass = ks_pvalue > 0.05   # threshold at 5%

            results.append({
                "name": loi_name,
                "params": params,
                "sse": sse,
                "r2": r2,
                "AIC": AIC,
                "BIC": BIC,
                "KS_stat": ks_stat,
                "KS_pvalue": ks_pvalue,
                "KS_pass": KS_pass
            })

        except Exception:
            continue

    # Ranking by AIC then BIC
    results_sorted = sorted(results, key=lambda d: (d["AIC"], d["BIC"]))

    return results_sorted

def Pond(n):
    W=np.random.randint(1,10,n)  # Generation de n entiers compris entre 1 et 10
    W= W/np.sum(W) #  Normalisation pour aoir la somme de spondérations égale à 1
    return W

def Retrieve_historical_data(Tickers, start_date, end_date):
    pass

#%%
# Faire de loi sous forme d moving avg et prendre le plus recurrent 
# penser a une corr comme fonction du temps 

###################################################################################
# Risk Measures
################################################################################

def max_drawdown(returns):
    """
    Computes the Maximum Drawdown based on daily returns.
    """
    returns = np.asarray(returns)

    cumulative = np.cumprod(1 + returns)
    cumulative_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - cumulative_max) / cumulative_max
    return drawdowns.min()  # negative value

def sortino(returns, rf=0):
    """
    Computes Sortino ratio:
    (mean(r - rf)) / downside_deviation
    """
    returns = np.asarray(returns)
    excess = returns - rf

    # downside deviation (semi-volatility)
    downside = excess[excess < 0]
    downside_dev = np.sqrt(np.mean(downside**2))

    return excess.mean() / downside_dev


def simulate_correlated_data(corr, n_samples=63, random_state=None):
    corr = np.asarray(corr)
    # Décomposition de Cholesky
    L = np.linalg.cholesky(corr)

    rng = np.random.default_rng(random_state)

    # Génère des variables normales indépendantes
    Z = rng.standard_normal(size=(n_samples, corr.shape[0]))

    # Introduit les corrélations
    X = Z @ L.T
    return X.T
def PortfolioStat(data_information,corr, sim=100000):
    #Simulate normal correlated random variables
    N=len(data_information["law"])
    M=63
    simulate_data=simulate_correlated_data(corr)
    data_list = []
    for i in range(N):
        asset = copule_gaussian(
            simulate_data[i, :],
            data_information["law"][i],
            data_information["params"][i]
        )
        data_list.append(pd.DataFrame(asset))
    data = pd.concat(data_list, axis=1).reset_index(drop=True)
    data.columns=[data_information["asset"]]
    n= data.shape[1]
    W=np.zeros((sim,n))
    moy=np.zeros(sim)
    theta=np.zeros(sim)
    for i in range(sim):
        wi= Pond(n) # POndération de chaque portefeuille
        moy[i]=np.mean(np.dot(wi,data.T))    # Rentabilité espérée du portefeuille
        theta[i]  = np.std(np.dot(wi,data.T))  #Volatilité de chaque portefeuille
        W[i]= wi   # incrémenter au fur et à mesure une matrice des pondérations
         # incrémentation des moyennes
         # incrémentations des variances
    results= list()    #liste qui contiendra la matrice des pondérations, le vecteur rentabilité, et le vecteur variance pour chaque simulation
    results.append(W) 
    results.append(moy)
    results.append(theta)
    plt.scatter(results[2],results[1])

    return results
    

#%%
tickers=["GOOGL", "MSFT", "GC=F", "TSLA",'AAPL']
date_start='2025-05-01'
date_end='2025-08-01'

Matrice_corr=cov_matrix_shift(tickers, date_end= date_end,moving_window=30,picking=2)
#%%


data=data_returns(tickers=tickers,date1=date_start,date2=date_end)

data_information={}
data_information["asset"]=[]
data_information["law"]=[]
data_information["params"]=[]

# Matrice_corr=cov_matrix_shift(data, date_end= date_end,moving_window=30,picking=2)

for asset in data.columns:
   data_information["asset"].append(asset)
   data_information["law"].append(distribution_fit(data[asset])[0]["name"])
   data_information["params"].append(distribution_fit(data[asset])[0]["params"])


# %%

r=PortfolioStat(data_information,corr=Matrice_corr[0])



# %%
# -------------------------------------------------------------------
# Ratio de Sharpe maximal
# portfolio = (weights_list, returns_list, sigma_list)
# -------------------------------------------------------------------
def MaxSharpeRate(portfolio, rf):
    w, r, s = portfolio
    
    sharpe = (r - rf) / s
    idx = np.argmax(sharpe)

    return {
        "weights": w[idx],
        "return": r[idx],
        "sigma": s[idx],
        "sharpe": sharpe[idx]
    }


# -------------------------------------------------------------------
# Global Minimum Variance
# -------------------------------------------------------------------
def GMV(portfolio):
    w, r, s = portfolio

    idx = np.argmin(s)

    return {
        "weights": w[idx],
        "return": r[idx],
        "sigma": s[idx]
    }


# -------------------------------------------------------------------
# Beta via covariance / variance
# -------------------------------------------------------------------
def beta(Rm, Rp):
    return np.cov(Rp, Rm)[0, 1] / np.var(Rm)


# -------------------------------------------------------------------
# Treynor Ratio
# -------------------------------------------------------------------
def treynor(Rp, Rm, rf):
    b = beta(Rm, Rp)
    return (np.mean(Rp) - rf) / b


# -------------------------------------------------------------------
# Alpha de Jensen
# -------------------------------------------------------------------
def alphaJensen(Rp, Rm, rf):
    b = beta(Rm, Rp)
    return np.mean(Rp) - (rf + b * (np.mean(Rm) - rf))


# -------------------------------------------------------------------
# Ratio de Sharpe simple
# -------------------------------------------------------------------
def SharpeRate(r, sigma, rf):
    return (r - rf) / sigma

# %%
MaxSharpeRate(r, 0)
# %%
