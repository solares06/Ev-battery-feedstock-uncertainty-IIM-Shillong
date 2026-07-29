import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

# Load Prophet forecast (monthly, national)
forecast_df = pd.read_csv("data/processed/ev_sales_forecast_2035.csv")
forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
forecast_df["Year"] = forecast_df["Date"].dt.year

# Aggregate forecast to annual level
annual_forecast = (
    forecast_df.groupby("Year")["Forecast_Registrations"].sum().reset_index()
)
annual_forecast.columns = ["Year", "Total_Sales"]
# Clip negative forecasts (Prophet can sometimes go slightly negative)
annual_forecast["Total_Sales"] = annual_forecast["Total_Sales"].clip(lower=0)

# Load IEA India EV sales by vehicle class
iea_df = pd.read_csv("data/processed/iea_india_ev_sales.csv")
# Filter for BEV only (primary battery recycling concern) and relevant modes
iea_bev = iea_df[
    (iea_df["powertrain"] == "BEV")
    & (iea_df["mode"].isin(["2 and 3 wheelers", "Cars", "Buses", "Vans", "Trucks"]))
]

# ============================================================
# 2. COMPUTE VEHICLE CLASS SHARES FROM IEA (2020-2025)
# ============================================================

iea_recent = iea_bev[(iea_bev["year"] >= 2020) & (iea_bev["year"] <= 2025)]
class_totals = iea_recent.groupby("mode")["value"].sum()
total_all = class_totals.sum()
class_shares = (class_totals / total_all).to_dict()

print("Vehicle Class Shares (from IEA 2020-2025):")
for cls, share in class_shares.items():
    print(f"  {cls}: {share:.2%}")

# ============================================================
# 3. SPLIT PROPHET FORECAST BY VEHICLE CLASS
# ============================================================

class_sales = pd.DataFrame({"Year": annual_forecast["Year"]})
for cls, share in class_shares.items():
    class_sales[cls] = (annual_forecast["Total_Sales"] * share).astype(int)

print("\nAnnual Sales by Vehicle Class (sample):")
print(class_sales.head(10).to_string(index=False))

# ============================================================
# 4. WEIBULL SURVIVAL PARAMETERS (Literature-backed)
# ============================================================

weibull_params = {
    "2 and 3 wheelers": {"shape": 3.5, "scale": 5.0},
    "Cars": {"shape": 3.0, "scale": 10.0},
    "Buses": {"shape": 2.5, "scale": 7.0},
    "Vans": {"shape": 3.0, "scale": 8.0},
    "Trucks": {"shape": 2.5, "scale": 8.0},
}

# ============================================================
# 5. EOL CONVOLUTION
# ============================================================
# For each cohort sold in year t, compute the incremental
# fraction that fails (reaches EOL) in year t+k.
# f(k) = F(k) - F(k-1), where F is the Weibull CDF.

years = class_sales["Year"].values
max_year = 2035
all_target_years = np.arange(years.min(), max_year + 1)

eol_by_class = {}

for cls in class_shares.keys():
    beta = weibull_params[cls]["shape"]
    lam = weibull_params[cls]["scale"]

    eol_annual = np.zeros(len(all_target_years))

    for i, sale_year in enumerate(years):
        if sale_year > max_year:
            continue
        sales_count = class_sales.loc[class_sales["Year"] == sale_year, cls].values[0]

        for j, target_year in enumerate(all_target_years):
            age = target_year - sale_year
            if age <= 0:
                continue
            # Incremental failure: F(age) - F(age-1)
            cdf_now = weibull_min.cdf(age, beta, scale=lam)
            cdf_prev = weibull_min.cdf(age - 1, beta, scale=lam)
            incremental_failure = cdf_now - cdf_prev
            eol_annual[j] += sales_count * incremental_failure

    eol_by_class[cls] = eol_annual

# Build output dataframe
eol_df = pd.DataFrame({"Year": all_target_years})
for cls in class_shares.keys():
    eol_df[cls] = eol_by_class[cls].astype(int)
eol_df["Total_EOL"] = eol_df[list(class_shares.keys())].sum(axis=1)

# ============================================================
# 6. SANITY CHECK
# ============================================================

cumulative_sales = class_sales[list(class_shares.keys())].sum().sum()
cumulative_eol = eol_df["Total_EOL"].sum()
print(f"\nSanity Check:")
print(f"  Cumulative Sales (2020-2035): {cumulative_sales:,.0f}")
print(f"  Cumulative EOL (2020-2035):   {cumulative_eol:,.0f}")
print(f"  Ratio (EOL/Sales):            {cumulative_eol/cumulative_sales:.2%}")
assert cumulative_eol <= cumulative_sales, "ERROR: EOL exceeds sales!"
print("  ✓ Sanity check passed (EOL ≤ Sales)")

# ============================================================
# 7. SAVE OUTPUT
# ============================================================

eol_df.to_csv("data/processed/eol_battery_projection.csv", index=False)
print(f"\nSaved EOL projection to data/processed/eol_battery_projection.csv")

# ============================================================
# 8. VISUALIZATION - Stacked Area Chart
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

classes = list(class_shares.keys())
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

ax.stackplot(
    eol_df["Year"],
    [eol_df[cls] for cls in classes],
    labels=classes,
    colors=colors,
    alpha=0.85,
)

ax.set_title(
    "Projected End-of-Life EV Batteries in India (2020–2035)",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Number of EOL Batteries", fontsize=12)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_xlim(2020, 2035)

# Add total annotation for 2030 and 2035
for yr in [2030, 2035]:
    total = eol_df.loc[eol_df["Year"] == yr, "Total_EOL"].values[0]
    ax.annotate(
        f"{total:,.0f}",
        xy=(yr, total),
        xytext=(yr - 1, total * 1.1),
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black"),
    )

plt.tight_layout()
plt.savefig("outputs/eol_projection_plot.png", dpi=300)
os.system("cp outputs/eol_projection_plot.png docs/")
plt.close()

print("Generated EOL projection plot.")

# Print key results
print("\n" + "=" * 60)
print("KEY RESULTS")
print("=" * 60)
for yr in [2025, 2028, 2030, 2033, 2035]:
    row = eol_df[eol_df["Year"] == yr]
    if not row.empty:
        print(f"  {yr}: {row['Total_EOL'].values[0]:>12,.0f} EOL batteries")
