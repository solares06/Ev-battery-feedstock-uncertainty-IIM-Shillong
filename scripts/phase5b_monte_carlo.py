"""
Phase 5b: Monte Carlo Scenario Generation
==========================================
Samples N=10,000 simulated EOL battery volume paths by:
    1. Estimating per-year forecast σ from Prophet's 80% confidence interval
    2. Drawing annual EV sales from N(μ, σ) independently for each year
    3. Applying vehicle-class disaggregation (fixed IEA-derived shares)
    4. Convolving each draw through class-specific Weibull hazard functions

This replaces the deterministic 3-scenario (Pessimistic/Base/Optimistic)
approach with proper probabilistic uncertainty quantification, as specified
in the original project brief.

Outputs:
    data/processed/mc_eol_paths_raw.csv   - (10000 × 16) total EOL per year
    data/processed/mc_sales_samples.csv   - (10000 × 16) sampled annual sales
    outputs/mc_spaghetti_plot.png         - MC path visualization
"""

import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import time

os.makedirs("outputs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("=" * 60)
print("Phase 5b: Monte Carlo Scenario Generation")
print("=" * 60)

# ============================================================
# 1. LOAD & PREPARE FORECAST DATA
# ============================================================

forecast_df = pd.read_csv("data/processed/ev_sales_forecast_2035.csv")
forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
forecast_df["Year"] = forecast_df["Date"].dt.year

# Estimate monthly standard deviation from Prophet's 80% CI
# For a normal distribution: 80% CI = mean ± 1.2816σ
# => σ = (upper − lower) / (2 × 1.2816)
forecast_df["sigma_monthly"] = (
    (forecast_df["yhat_upper"] - forecast_df["yhat_lower"]) / (2 * 1.2816)
).clip(lower=0)

# Aggregate to annual level
# Mean: sum of monthly yhat values
# Sigma: sqrt(sum of monthly σ²) - assumes monthly forecast errors are independent
MAX_YEAR = 2035
annual_agg = (
    forecast_df.groupby("Year")
    .agg(
        mean_sales=("Forecast_Registrations", "sum"),
        sigma_sales=("sigma_monthly", lambda x: np.sqrt((x**2).sum())),
    )
    .reset_index()
)
annual_agg = annual_agg[annual_agg["Year"] <= MAX_YEAR].copy()
annual_agg["mean_sales"] = annual_agg["mean_sales"].clip(lower=0)

years = annual_agg["Year"].values
n_sale_years = len(years)
all_target_years = np.arange(years.min(), MAX_YEAR + 1)
n_target_years = len(all_target_years)

print(f"Forecast range: {years.min()}–{years.max()} ({n_sale_years} years)")
print(f"\nAnnual forecast uncertainty (σ as % of mean):")
for _, row in annual_agg.iterrows():
    pct = row["sigma_sales"] / row["mean_sales"] * 100 if row["mean_sales"] > 0 else 0
    print(
        f"  {int(row['Year'])}: mean={row['mean_sales']:>12,.0f}  σ={row['sigma_sales']:>10,.0f}  ({pct:.1f}%)"
    )

# ============================================================
# 2. LOAD AUXILIARY DATA
# ============================================================

# Vehicle class shares (from IEA BEV data, 2020-2025)
iea_df = pd.read_csv("data/processed/iea_india_ev_sales.csv")
iea_bev = iea_df[
    (iea_df["powertrain"] == "BEV")
    & (iea_df["mode"].isin(["2 and 3 wheelers", "Cars", "Buses", "Vans", "Trucks"]))
]
iea_recent = iea_bev[(iea_bev["year"] >= 2020) & (iea_bev["year"] <= 2025)]
class_shares = (
    iea_recent.groupby("mode")["value"].sum() / iea_recent["value"].sum()
).to_dict()

print("\nVehicle class shares (IEA 2020-2025):")
for cls, share in class_shares.items():
    print(f"  {cls}: {share:.2%}")

# Weibull survival parameters (literature-backed)
weibull_params = {
    "2 and 3 wheelers": {"shape": 3.5, "scale": 5.0},
    "Cars": {"shape": 3.0, "scale": 10.0},
    "Buses": {"shape": 2.5, "scale": 7.0},
    "Vans": {"shape": 3.0, "scale": 8.0},
    "Trucks": {"shape": 2.5, "scale": 8.0},
}

# ============================================================
# 3. PRE-COMPUTE WEIBULL CONVOLUTION MATRICES
# ============================================================
# For each vehicle class, M[i,j] = P(battery sold in year[i] reaches EOL in year[j])
#                                 = F(age) − F(age−1), where age = year[j] − year[i]

conv_matrices = {}
for cls, params in weibull_params.items():
    beta, lam = params["shape"], params["scale"]
    M = np.zeros((n_sale_years, n_target_years))
    for i, sy in enumerate(years):
        for j, ty in enumerate(all_target_years):
            age = ty - sy
            if age <= 0:
                continue
            M[i, j] = weibull_min.cdf(age, beta, scale=lam) - weibull_min.cdf(
                age - 1, beta, scale=lam
            )
    conv_matrices[cls] = M

# Combined convolution matrix (weighted sum across all vehicle classes)
combined_M = np.zeros((n_sale_years, n_target_years))
for cls, share in class_shares.items():
    combined_M += share * conv_matrices[cls]

# ============================================================
# 4. MONTE CARLO SAMPLING (Fully Vectorized)
# ============================================================

N_SAMPLES = 10_000
np.random.seed(42)

mean_sales = annual_agg["mean_sales"].values  # (n_sale_years,)
sigma_sales = annual_agg["sigma_sales"].values  # (n_sale_years,)

print(f"\nSampling {N_SAMPLES:,} annual sales paths...")
start_time = time.time()

# Draw all N samples at once: shape (N_SAMPLES, n_sale_years)
sampled_sales = np.maximum(
    np.random.normal(
        loc=mean_sales[np.newaxis, :],
        scale=sigma_sales[np.newaxis, :],
        size=(N_SAMPLES, n_sale_years),
    ),
    0,  # Sales cannot be negative
)

# Weibull convolution via matrix multiply:
# (N_SAMPLES, n_sale_years) @ (n_sale_years, n_target_years) → (N_SAMPLES, n_target_years)
eol_paths = sampled_sales @ combined_M

elapsed = time.time() - start_time
print(f"Completed {N_SAMPLES:,} simulations in {elapsed:.2f} seconds")

# ============================================================
# 5. SAVE RAW MC DATA
# ============================================================

mc_eol_df = pd.DataFrame(eol_paths, columns=[str(y) for y in all_target_years])
mc_eol_df.to_csv("data/processed/mc_eol_paths_raw.csv", index=False)

mc_sales_df = pd.DataFrame(sampled_sales, columns=[str(y) for y in years])
mc_sales_df.to_csv("data/processed/mc_sales_samples.csv", index=False)

print(f"\nSaved raw MC data:")
print(f"  EOL paths:     data/processed/mc_eol_paths_raw.csv  ({mc_eol_df.shape})")
print(f"  Sales samples: data/processed/mc_sales_samples.csv  ({mc_sales_df.shape})")

# ============================================================
# 6. SUMMARY STATISTICS
# ============================================================

print("\n--- MC Summary: Total EOL Batteries by Year ---")
print(
    f"{'Year':>6} | {'Mean':>14} | {'Std Dev':>12} | {'P5':>14} | {'P50':>14} | {'P95':>14}"
)
print("-" * 82)
for yr_idx, yr in enumerate(all_target_years):
    if yr in [2025, 2027, 2028, 2030, 2032, 2035]:
        vals = eol_paths[:, yr_idx]
        print(
            f"{yr:>6} | {vals.mean():>14,.0f} | {vals.std():>12,.0f} | "
            f"{np.percentile(vals, 5):>14,.0f} | {np.percentile(vals, 50):>14,.0f} | "
            f"{np.percentile(vals, 95):>14,.0f}"
        )

# ============================================================
# 7. VISUALIZATION: MC Spaghetti Plot with Percentile Bands
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

# Background: 300 random MC paths
rng_plot = np.random.RandomState(0)
subset_idx = rng_plot.choice(N_SAMPLES, size=300, replace=False)
for idx in subset_idx:
    ax.plot(
        all_target_years, eol_paths[idx], color="steelblue", alpha=0.03, linewidth=0.5
    )

# Percentile bands
p5 = np.percentile(eol_paths, 5, axis=0)
p25 = np.percentile(eol_paths, 25, axis=0)
p50 = np.percentile(eol_paths, 50, axis=0)
p75 = np.percentile(eol_paths, 75, axis=0)
p95 = np.percentile(eol_paths, 95, axis=0)

ax.fill_between(
    all_target_years,
    p5,
    p95,
    alpha=0.15,
    color="steelblue",
    label="5th–95th percentile",
)
ax.fill_between(
    all_target_years,
    p25,
    p75,
    alpha=0.3,
    color="steelblue",
    label="25th–75th percentile",
)
ax.plot(all_target_years, p50, color="darkblue", linewidth=2.5, label="Median (P50)")

ax.set_title(
    f"Monte Carlo Simulation: EOL Battery Volume Paths (N={N_SAMPLES:,})",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Total EOL Batteries", fontsize=12)
ax.legend(loc="upper left", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.3)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
ax.set_xlim(2020, 2035)

plt.tight_layout()
plt.savefig("outputs/mc_spaghetti_plot.png", dpi=300)
plt.close()

print("\nGenerated: outputs/mc_spaghetti_plot.png")
print("\n" + "=" * 60)
print("✓ Phase 5b complete.")
print("  Proceed to Phase 5c for scenario reduction (k-means clustering).")
print("=" * 60)
