import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import os

df = pd.read_csv("data/processed/vahan_registrations.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Aggregate to national level
national_df = df.groupby("Date")["Registrations"].sum().reset_index()
national_df = national_df.sort_values("Date").set_index("Date")

os.makedirs("outputs", exist_ok=True)

# 1. Plot Total Registrations
plt.figure(figsize=(12, 6))
plt.plot(
    national_df.index,
    national_df["Registrations"],
    marker="o",
    linestyle="-",
    color="b",
)
plt.title("Total EV Registrations in India (2020 - 2026)")
plt.xlabel("Date")
plt.ylabel("Monthly Registrations")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("outputs/total_registrations.png", dpi=300)
plt.close()

# 2. Seasonal Decomposition
try:
    decomposition = seasonal_decompose(
        national_df["Registrations"], model="additive", period=12
    )
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig("outputs/seasonal_decomposition.png", dpi=300)
    plt.close()
    print("Saved plots to outputs/")
except Exception as e:
    print(f"Decomposition failed: {e}")

# 3. ADF Stationarity Test
result = adfuller(national_df["Registrations"].dropna())
print("\n--- Augmented Dickey-Fuller Test Results ---")
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"\t{key}: {value:.4f}")

if result[1] <= 0.05:
    print("=> Conclusion: The time series is STATIONARY (reject the null hypothesis).")
else:
    print(
        "=> Conclusion: The time series is NON-STATIONARY (fail to reject the null hypothesis)."
    )

# Check differenced series
diff_result = adfuller(national_df["Registrations"].diff().dropna())
print("\n--- ADF Test on First Difference ---")
print(f"p-value: {diff_result[1]:.4f}")
if diff_result[1] <= 0.05:
    print(
        "=> Conclusion: The first difference is STATIONARY. (Indicates ARIMA model should use d=1)"
    )
else:
    print("=> Conclusion: The first difference is STILL non-stationary.")
