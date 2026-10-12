# %% [markdown]
# # 02 Results
# Compare hedging strategies: how much risk do they remove, and what do they cost?
# Input: results/exp01_paths.csv (written by backtest.py).
# P&L is from the issuer's view: positive = good.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

EXP = "exp01"
RESULTS = ROOT / "results"
plt.rcParams["figure.figsize"] = (10, 4)

# %% [markdown]
# ## 1. Load results and check the format

# %%
df = pd.read_csv(RESULTS / f"{EXP}_paths.csv")

needed = {"strategy", "pnl", "cost", "n_trades"}
missing = needed - set(df.columns)
if missing:
    raise ValueError(f"results file is missing columns: {missing}")

if "scenario" not in df.columns:
    df["scenario"] = "default"

print("rows       :", len(df))
print("scenarios  :", list(df["scenario"].unique()))
print("strategies :", list(df["strategy"].unique()))
print("NaN pnl    :", df["pnl"].isna().sum())
df.head()

# %% [markdown]
# ## 2. Experiment settings
# Printed from the config so the results are not separated from their settings.

# %%
cfg_path = ROOT / "configs" / f"{EXP}.yaml"
print(cfg_path.read_text() if cfg_path.exists() else "config file not found")

# %% [markdown]
# ## 3. Summary table
# - `std_pnl`: hedging error (lower = better hedge)
# - `var5`: 5th percentile of P&L (95% of paths do better than this)
# - `cvar5`: average P&L of the worst 5% of paths
# - `mean_cost`: average transaction cost per path

# %%
summary = df.groupby(["scenario", "strategy"], sort=False).agg(
    n_paths=("pnl", "size"),
    mean_pnl=("pnl", "mean"),
    std_pnl=("pnl", "std"),
    var5=("pnl", lambda x: x.quantile(0.05)),
    cvar5=("pnl", lambda x: x[x <= x.quantile(0.05)].mean()),
    worst=("pnl", "min"),
    mean_cost=("cost", "mean"),
    mean_trades=("n_trades", "mean"),
)
summary.round(4)

# %% [markdown]
# ## 4. How much risk does each hedge remove?
# `risk_reduction = 1 - std(strategy) / std(no_hedge)`. 1.0 means a perfect hedge, 0 means no better than no hedge.

# %%
if "no_hedge" in df["strategy"].values:
    base = summary.xs("no_hedge", level="strategy")["std_pnl"]
    reduction = 1 - summary["std_pnl"] / summary.index.get_level_values("scenario").map(base).values
    reduction.name = "risk_reduction"
    print(reduction.round(3).to_string())
else:
    print("No 'no_hedge' strategy in the results, so there is no baseline.")

# %% [markdown]
# ## 5. P&L distributions
# The x-axis is cut at the 1st and 99th percentile so that `no_hedge` does not squash the others.

# %%
scenarios = list(df["scenario"].unique())
fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 4), squeeze=False)

for ax, sc in zip(axes[0], scenarios):
    d = df[df["scenario"] == sc]
    lo, hi = d["pnl"].quantile([0.01, 0.99])
    bins = np.linspace(lo, hi, 50)
    for name, g in d.groupby("strategy", sort=False):
        ax.hist(g["pnl"], bins=bins, histtype="step", linewidth=1.5, label=name)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"P&L distribution ({sc})")
    ax.set_xlabel("P&L")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Box plot

# %%
fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 4), squeeze=False)

for ax, sc in zip(axes[0], scenarios):
    d = df[df["scenario"] == sc]
    names = list(d["strategy"].unique())
    ax.boxplot([d.loc[d["strategy"] == n, "pnl"] for n in names],
               labels=names, showfliers=False)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"P&L by strategy ({sc})")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Risk vs cost
# Each point is one strategy. Bottom-left is best (low risk, low cost).
# Trading more usually lowers risk but raises cost, and this plot shows that trade-off.

# %%
fig, ax = plt.subplots()
markers = ["o", "s", "^", "D", "v"]

for i, sc in enumerate(scenarios):
    s = summary.xs(sc, level="scenario")
    ax.scatter(s["mean_cost"], s["std_pnl"], marker=markers[i % len(markers)], label=sc)
    for name, row in s.iterrows():
        ax.annotate(name, (row["mean_cost"], row["std_pnl"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")

ax.set_xlabel("mean transaction cost per path")
ax.set_ylabel("std of P&L (hedging error)")
ax.set_title("Risk vs cost")
ax.legend()
plt.show()

# %% [markdown]
# ## 8. Number of trades vs cost

# %%
trades = df.groupby(["scenario", "strategy"], sort=False)[["n_trades", "cost"]].mean()
trades["cost_per_trade"] = trades["cost"] / trades["n_trades"].replace(0, np.nan)
trades.round(4)

# %% [markdown]
# ## 9. Worst outcomes
# Look at the tail. A good average can hide a few paths with very large losses.

# %%
for sc in scenarios:
    d = df[df["scenario"] == sc]
    print(f"--- {sc}: 5 worst P&L per strategy ---")
    worst5 = (d.sort_values("pnl").groupby("strategy", sort=False).head(5)
                .sort_values(["strategy", "pnl"]))
    cols = [c for c in ["strategy", "path", "pnl", "cost", "n_trades"] if c in worst5.columns]
    print(worst5[cols].to_string(index=False))
    print()

# %% [markdown]
# ## 10. Save the summary table

# %%
out = RESULTS / f"{EXP}_summary.csv"
summary.to_csv(out)
print("saved to", out)

# %% [markdown]
# ## Notes (fill in after reading the results)
# - Which strategy has the lowest `std_pnl`? Is it still best after costs?
# - Does `mean_pnl` sit below zero by about the cost? If it is far from that, check the premium and the P&L code.
# - Is `cvar5` much worse than `var5`? That points to fat-tail losses.
# - Do the GBM and bootstrap scenarios rank strategies the same way?