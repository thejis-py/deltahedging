# %% [markdown]
# # 01 Explore data
# Goal: check the data quality and see how real returns differ from GBM,
# before building the hedging backtest.

# %%
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path.cwd().parent          # notebook is run from notebooks/
sys.path.insert(0, str(ROOT))

from src.data import get_prices
from src.simulation import estimate_gbm_params, simulate_gbm

TRADING_DAYS = 252
plt.rcParams["figure.figsize"] = (10, 4)

# %% [markdown]
# ## 1. Parameters

# %%
SYMBOL = "AOT"
START = "2023-01-01T00:00"
END = "2024-12-31T00:00"       # use a fixed end date so results are reproducible
INTERVAL = "1d"

# %% [markdown]
# ## 2. Load data
# Uses the cached file in `data/raw/` if it exists, otherwise calls the API.

# %%
try:
    df = get_prices(SYMBOL, START, END, INTERVAL)
    source = "Settrade API / cache"
except Exception as e:
    print("Could not load real data:", e)
    print("Using SYNTHETIC GBM data instead. Do not draw conclusions from it.")
    dates = pd.bdate_range(START[:10], periods=500, tz="Asia/Bangkok")
    fake = simulate_gbm(35, 0.05, 0.25, T=500 / TRADING_DAYS,
                        n_steps=499, n_paths=1, seed=1)[0]
    df = pd.DataFrame({"close": fake}, index=dates)
    source = "SYNTHETIC"

print("source:", source)
df.head()

# %% [markdown]
# ## 3. Basic checks
# Look for missing values, duplicate dates, zero or negative prices, and big date gaps.

# %%
print("rows            :", len(df))
print("first / last    :", df.index.min().date(), "/", df.index.max().date())
print("missing values  :\n", df.isna().sum())
print("duplicate dates :", df.index.duplicated().sum())
print("close <= 0      :", (df["close"] <= 0).sum())

gap_days = df.index.to_series().diff().dt.days
print("\nGaps longer than 4 days (holidays are normal, long gaps are not):")
print(gap_days[gap_days > 4])

df.describe()

# %% [markdown]
# ## 4. Price (and volume)

# %%
fig, ax = plt.subplots()
df["close"].plot(ax=ax)
ax.set_title(f"{SYMBOL} close ({source})")
ax.set_ylabel("price")
plt.show()

if "volume" in df.columns:
    df["volume"].plot(title="Volume", alpha=0.7)
    plt.show()

# %% [markdown]
# ## 5. Returns and summary statistics
# Daily log returns. Kurtosis is "excess" kurtosis (normal distribution = 0).

# %%
ret = np.log(df["close"]).diff().dropna()

summary = {
    "symbol": SYMBOL,
    "source": source,
    "n_returns": int(len(ret)),
    "mean_daily": float(ret.mean()),
    "std_daily": float(ret.std()),
    "annual_vol": float(ret.std() * np.sqrt(TRADING_DAYS)),
    "skew": float(ret.skew()),
    "excess_kurtosis": float(ret.kurt()),
    "min_daily": float(ret.min()),
    "max_daily": float(ret.max()),
}
pd.Series(summary)

# %% [markdown]
# ## 6. Are returns normal?
# GBM assumes they are. Fat tails (kurtosis > 0) mean big moves happen more often than GBM says.

# %%
jb_stat, jb_p = stats.jarque_bera(ret)
print(f"Jarque-Bera p-value = {jb_p:.4g}  (small = not normal)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(ret, bins=50, density=True, alpha=0.6, label="real")
x = np.linspace(ret.min(), ret.max(), 200)
axes[0].plot(x, stats.norm.pdf(x, ret.mean(), ret.std()), label="normal")
axes[0].set_title("Return histogram")
axes[0].legend()

stats.probplot(ret, dist="norm", plot=axes[1])
axes[1].set_title("QQ plot (points off the line at the ends = fat tails)")
plt.show()

# %% [markdown]
# ## 7. Volatility over time
# GBM assumes constant volatility. If this line moves a lot, a single sigma is a rough approximation.

# %%
roll_vol = ret.rolling(21).std() * np.sqrt(TRADING_DAYS)

fig, ax = plt.subplots()
roll_vol.plot(ax=ax, label="21-day rolling vol")
ax.axhline(summary["annual_vol"], color="red", linestyle="--", label="full-sample vol")
ax.set_title("Annualised volatility")
ax.legend()
plt.show()

# %% [markdown]
# ## 8. Drawdown

# %%
drawdown = df["close"] / df["close"].cummax() - 1
drawdown.plot(title=f"Drawdown (max = {drawdown.min():.1%})")
plt.show()

# %% [markdown]
# ## 9. Largest daily moves
# Check these dates by hand. A big drop on an ex-dividend or split date means the prices are
# probably **not adjusted**, and the returns (and the volatility) would be wrong.

# %%
big = ret.reindex(ret.abs().sort_values(ascending=False).index)[:10]
big.to_frame("log_return")

# %% [markdown]
# ## 10. Real returns vs GBM
# Simulate GBM with the same estimated parameters and compare how often large moves happen.

# %%
mu, sigma = estimate_gbm_params(df["close"])
n = len(ret)
sim = simulate_gbm(df["close"].iloc[0], mu, sigma, T=n / TRADING_DAYS,
                   n_steps=n, n_paths=200, seed=1)
sim_ret = np.diff(np.log(sim), axis=1)

def tail_fraction(x, z=3):
    x = np.asarray(x)
    return (np.abs(x - x.mean()) > z * x.std()).mean()

print(f"estimated mu = {mu:.3f}, sigma = {sigma:.3f}")
print(f"Share of days with a move > 3 std:")
print(f"  real data : {tail_fraction(ret):.2%}")
print(f"  GBM       : {tail_fraction(sim_ret):.2%}   (normal theory: 0.27%)")

plt.hist(sim_ret.ravel(), bins=80, density=True, alpha=0.5, label="GBM")
plt.hist(ret, bins=80, density=True, alpha=0.5, label="real")
plt.yscale("log")
plt.title("Return distribution (log y-axis shows the tails)")
plt.legend()
plt.show()

# %% [markdown]
# ## 11. Save summary for later notebooks

# %%
out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)
summary["mu_est"] = float(mu)
summary["sigma_est"] = float(sigma)
(out_dir / f"explore_{SYMBOL}.json").write_text(json.dumps(summary, indent=2))
print("saved to", out_dir / f"explore_{SYMBOL}.json")

# %% [markdown]
# ## Notes
# - Is the annual vol reasonable for this stock?
# - Are there gaps, duplicates or suspicious big moves (unadjusted prices)?
# - Is the kurtosis well above 0? If so, test the hedge on bootstrapped paths as well as GBM.
# - Does the rolling vol change a lot? If so, a constant sigma in the hedge model will be wrong at times.

#jupytext --to ipynb notebooks/01_explore_data.py