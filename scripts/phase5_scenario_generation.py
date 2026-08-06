import pandas as pd
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/phase5_scenarios", exist_ok=True)
os.makedirs("data/processed/model_outputs", exist_ok=True)

# ============================================================
# 1. LOAD DATA & SETUP
# ============================================================

# Load Prophet forecast
forecast_df = pd.read_csv("data/processed/model_outputs/ev_sales_forecast_2035.csv")
forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
forecast_df["Year"] = forecast_df["Date"].dt.year

# We will use yhat_lower (Pessimistic), Forecast_Registrations (Base), yhat_upper (Optimistic)
annual_forecast = (
    forecast_df.groupby("Year")
    .agg({"yhat_lower": "sum", "Forecast_Registrations": "sum", "yhat_upper": "sum"})
    .reset_index()
)

# Clip negative values
for col in ["yhat_lower", "Forecast_Registrations", "yhat_upper"]:
    annual_forecast[col] = annual_forecast[col].clip(lower=0)

# Load IEA class shares
iea_df = pd.read_csv("data/processed/iea/iea_india_ev_sales.csv")
iea_bev = iea_df[
    (iea_df["powertrain"] == "BEV")
    & (iea_df["mode"].isin(["2 and 3 wheelers", "Cars", "Buses", "Vans", "Trucks"]))
]
iea_recent = iea_bev[(iea_bev["year"] >= 2020) & (iea_bev["year"] <= 2025)]
class_shares = (
    iea_recent.groupby("mode")["value"].sum() / iea_recent["value"].sum()
).to_dict()

# Load Chemistry Mix
chem_df = pd.read_csv("data/processed/chemistry/chemistry_mix.csv")
chem_df["Date"] = pd.to_datetime(chem_df["Date"])
chem_df["Year"] = chem_df["Date"].dt.year
annual_chem = (
    chem_df.groupby("Year").first().reset_index()[["Year", "LFP_Share", "NMC_Share"]]
)

# Weibull params
weibull_params = {
    "2 and 3 wheelers": {"shape": 3.5, "scale": 5.0},
    "Cars": {"shape": 3.0, "scale": 10.0},
    "Buses": {"shape": 2.5, "scale": 7.0},
    "Vans": {"shape": 3.0, "scale": 8.0},
    "Trucks": {"shape": 2.5, "scale": 8.0},
}

# ============================================================
# 2. RUN SCENARIO CONVOLUTIONS
# ============================================================

scenarios = {
    "Pessimistic": {"col": "yhat_lower", "prob": 0.25},
    "Base": {"col": "Forecast_Registrations", "prob": 0.50},
    "Optimistic": {"col": "yhat_upper", "prob": 0.25},
}

max_year = 2035
all_target_years = np.arange(annual_forecast["Year"].min(), max_year + 1)
master_records = []

for sc_name, sc_info in scenarios.items():
    forecast_col = sc_info["col"]
    prob = sc_info["prob"]

    for cls, share in class_shares.items():
        beta = weibull_params[cls]["shape"]
        lam = weibull_params[cls]["scale"]

        for sale_year in annual_forecast["Year"]:
            if sale_year > max_year:
                continue

            # Annual sales for this class in this scenario
            total_sales = annual_forecast.loc[
                annual_forecast["Year"] == sale_year, forecast_col
            ].values[0]
            class_sales = int(total_sales * share)

            # Get chemistry for this sales cohort
            chem_row = annual_chem[annual_chem["Year"] == sale_year]
            if not chem_row.empty:
                lfp_ratio = chem_row["LFP_Share"].values[0]
                nmc_ratio = chem_row["NMC_Share"].values[0]
            else:
                # Fallback to last known year (2026)
                lfp_ratio = annual_chem["LFP_Share"].iloc[-1]
                nmc_ratio = annual_chem["NMC_Share"].iloc[-1]

            for target_year in all_target_years:
                age = target_year - sale_year
                if age <= 0:
                    continue

                # Fraction failing this year
                cdf_now = weibull_min.cdf(age, beta, scale=lam)
                cdf_prev = weibull_min.cdf(age - 1, beta, scale=lam)
                fail_frac = cdf_now - cdf_prev

                eol_volume = class_sales * fail_frac

                if eol_volume > 0.1:  # Skip negligible fractions
                    # LFP
                    master_records.append(
                        {
                            "Year": target_year,
                            "Scenario": sc_name,
                            "Probability": prob,
                            "Vehicle_Class": cls,
                            "Chemistry": "LFP",
                            "Cohort_Year": sale_year,
                            "EOL_Volume": eol_volume * lfp_ratio,
                        }
                    )
                    # NMC
                    master_records.append(
                        {
                            "Year": target_year,
                            "Scenario": sc_name,
                            "Probability": prob,
                            "Vehicle_Class": cls,
                            "Chemistry": "NMC",
                            "Cohort_Year": sale_year,
                            "EOL_Volume": eol_volume * nmc_ratio,
                        }
                    )

master_df = pd.DataFrame(master_records)
# Aggregate over cohorts
smip_df = (
    master_df.groupby(
        ["Year", "Scenario", "Probability", "Vehicle_Class", "Chemistry"]
    )["EOL_Volume"]
    .sum()
    .reset_index()
)
smip_df["EOL_Volume"] = smip_df["EOL_Volume"].astype(int)

# ============================================================
# 3. SAVE MASTER DATASET FOR SMIP SOLVER
# ============================================================

smip_df.to_csv("data/processed/model_outputs/smip_feedstock_scenarios.csv", index=False)
print(
    "Saved SMIP master scenario dataset to data/processed/smip_feedstock_scenarios.csv"
)

# ============================================================
# 4. PLOT SCENARIO FUNNEL
# ============================================================

plot_df = smip_df.groupby(["Year", "Scenario"])["EOL_Volume"].sum().unstack()

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    plot_df.index,
    plot_df["Base"],
    label="Base Scenario (50% prob)",
    color="blue",
    linewidth=3,
)
ax.fill_between(
    plot_df.index,
    plot_df["Pessimistic"],
    plot_df["Optimistic"],
    color="lightblue",
    alpha=0.4,
    label="Pessimistic - Optimistic Bounds (25% prob each)",
)

ax.set_title(
    "EV Battery EOL Volume Scenarios (2020-2035)", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Number of EOL Batteries", fontsize=12)
ax.legend(loc="upper left", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)

# Format y-axis to millions
import matplotlib.ticker as ticker

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ",")))

plt.tight_layout()
plt.savefig("outputs/phase5_scenarios/scenario_funnel_plot.png", dpi=300)
os.system("cp outputs/phase5_scenarios/scenario_funnel_plot.png docs/")
plt.close()

print("Generated scenario funnel plot.")
