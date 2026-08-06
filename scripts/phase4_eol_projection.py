import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/phase4_eol", exist_ok=True)
os.makedirs("data/processed/model_outputs", exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

# Load Prophet forecast (monthly, national)
forecast_df = pd.read_csv("data/processed/model_outputs/ev_sales_forecast_2035.csv")
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
iea_df = pd.read_csv("data/processed/iea/iea_india_ev_sales.csv")
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

eol_df.to_csv("data/processed/model_outputs/eol_battery_projection.csv", index=False)
print(f"\nSaved EOL projection to data/processed/model_outputs/eol_battery_projection.csv")

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
    "Projected End-of-Life EV Battery Pack Units in India (2020–2035)",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Number of Retired Battery Pack Units", fontsize=12)
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
plt.savefig("outputs/phase4_eol/eol_projection_plot.png", dpi=300)
plt.close()

print("Generated EOL projection plot (Unit Count).")

# ============================================================
# 9. EOL CAPACITY IN GWh (Energy-Based View)
# ============================================================
# Average battery capacity per vehicle class (kWh per pack).
# Sources: BNEF, JMK Research, OEM spec sheets (Tata, Ather, BYD).

avg_kwh_per_pack = {
    "2 and 3 wheelers": 2.5,   # e-scooters ~2–3 kWh, e-rickshaws ~3.5 kWh
    "Cars": 35.0,               # Tata Nexon ~30.2, MG ZS ~44.5, avg ~35
    "Buses": 250.0,             # Tata Starbus ~200–300 kWh
    "Vans": 30.0,               # Light commercial EVs ~25–35 kWh
    "Trucks": 150.0,            # Medium-duty e-trucks ~100–200 kWh
}

eol_gwh_df = pd.DataFrame({"Year": all_target_years})
for cls in class_shares.keys():
    # Convert units to GWh: (num_packs * kWh_per_pack) / 1e6
    eol_gwh_df[cls] = (eol_df[cls] * avg_kwh_per_pack[cls]) / 1e6

eol_gwh_df["Total_EOL_GWh"] = eol_gwh_df[list(class_shares.keys())].sum(axis=1)

# Save GWh projection
eol_gwh_df.to_csv("data/processed/model_outputs/eol_battery_projection_gwh.csv", index=False)
print("Saved EOL GWh projection to data/processed/model_outputs/eol_battery_projection_gwh.csv")

# ============================================================
# 10. VISUALIZATION - Stacked Area Chart (GWh)
# ============================================================

fig2, ax2 = plt.subplots(figsize=(12, 7))

ax2.stackplot(
    eol_gwh_df["Year"],
    [eol_gwh_df[cls] for cls in classes],
    labels=classes,
    colors=colors,
    alpha=0.85,
)

ax2.set_title(
    "Projected End-of-Life EV Battery Capacity in India (2020–2035)",
    fontsize=14,
    fontweight="bold",
)
ax2.set_xlabel("Year", fontsize=12)
ax2.set_ylabel("Retired Battery Capacity (GWh)", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.set_xlim(2020, 2035)

# Add total annotation for 2030 and 2035
for yr in [2030, 2035]:
    total = eol_gwh_df.loc[eol_gwh_df["Year"] == yr, "Total_EOL_GWh"].values[0]
    ax2.annotate(
        f"{total:,.1f} GWh",
        xy=(yr, total),
        xytext=(yr - 1.5, total * 1.12),
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black"),
    )

plt.tight_layout()
plt.savefig("outputs/phase4_eol/eol_projection_gwh_plot.png", dpi=300)
plt.close()

print("Generated EOL projection plot (GWh Capacity).")

# ============================================================
# 11. KEY RESULTS SUMMARY (Units + GWh)
# ============================================================

print("\n" + "=" * 70)
print("KEY RESULTS — EOL Battery Units")
print("=" * 70)
for yr in [2025, 2028, 2030, 2033, 2035]:
    row = eol_df[eol_df["Year"] == yr]
    if not row.empty:
        print(f"  {yr}: {row['Total_EOL'].values[0]:>12,.0f} battery packs")

print("\n" + "=" * 70)
print("KEY RESULTS — EOL Battery Capacity (GWh)")
print("=" * 70)
for yr in [2025, 2028, 2030, 2033, 2035]:
    row = eol_gwh_df[eol_gwh_df["Year"] == yr]
    if not row.empty:
        print(f"  {yr}: {row['Total_EOL_GWh'].values[0]:>12.2f} GWh")

print("\n" + "=" * 70)
print("COMPOSITION COMPARISON (2035)")
print("=" * 70)
yr_2035 = 2035
print(f"  {'Vehicle Class':<22} {'% of Units':>12} {'% of GWh':>12}")
print(f"  {'-'*22} {'-'*12} {'-'*12}")
for cls in classes:
    unit_share = eol_df.loc[eol_df["Year"] == yr_2035, cls].values[0] / eol_df.loc[eol_df["Year"] == yr_2035, "Total_EOL"].values[0] * 100
    gwh_share = eol_gwh_df.loc[eol_gwh_df["Year"] == yr_2035, cls].values[0] / eol_gwh_df.loc[eol_gwh_df["Year"] == yr_2035, "Total_EOL_GWh"].values[0] * 100
    print(f"  {cls:<22} {unit_share:>11.1f}% {gwh_share:>11.1f}%")
