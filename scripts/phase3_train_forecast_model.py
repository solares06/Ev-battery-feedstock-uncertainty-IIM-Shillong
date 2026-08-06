import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import os
import json
from prophet.serialize import model_to_json

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/phase3_forecast", exist_ok=True)

# 1. Data Preparation
vahan_df = pd.read_csv("data/processed/vahan/vahan_registrations.csv")
vahan_df["Date"] = pd.to_datetime(vahan_df["Date"])

# Aggregate to National Level
national_df = vahan_df.groupby("Date")["Registrations"].sum().reset_index()

# Load Policy Regressors
policy_df = pd.read_csv("data/processed/policy/policy_regressors.csv")
policy_df["Date"] = pd.to_datetime(policy_df["Date"])

# Merge
df = pd.merge(national_df, policy_df, on="Date", how="left")

# Prepare for Prophet (requires 'ds' and 'y' columns)
df = df.rename(columns={"Date": "ds", "Registrations": "y"})

# 2. Train/Test Split
# Train: Jan 2020 - Dec 2025
# Test: Jan 2026 - Jul 2026
train_df = df[df["ds"] < "2026-01-01"]
test_df = df[(df["ds"] >= "2026-01-01") & (df["ds"] <= "2026-07-01")]

# 3. Model Training
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
)

# Add regressors
m.add_regressor("FAME_II_Active")
m.add_regressor("PM_EDRIVE_Active")
m.add_regressor("State_Subsidy_Active")

m.fit(train_df)

# Validation on Holdout
forecast_test = m.predict(test_df.drop(columns=["y"]))
mape = mean_absolute_percentage_error(test_df["y"], forecast_test["yhat"])
rmse = np.sqrt(mean_squared_error(test_df["y"], forecast_test["yhat"]))
print(f"Validation MAPE: {mape:.2%}")
print(f"Validation RMSE: {rmse:.2f}")

# 4. Long-Term Forecasting to 2035
future = m.make_future_dataframe(
    periods=12 * 10, freq="MS"
)  # 10 years from Dec 2025 -> ~2035
# Merge future with full policy dummy dataframe
future_with_regressors = pd.merge(
    future, policy_df, left_on="ds", right_on="Date", how="left"
)
# Drop extra Date column
if "Date" in future_with_regressors.columns:
    future_with_regressors = future_with_regressors.drop(columns=["Date"])

# Any missing policy values (e.g. past 2026) can be safely assumed as 0 for FAME/PM-EDRIVE
future_with_regressors.fillna(0, inplace=True)

forecast = m.predict(future_with_regressors)

# Format output
forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
    columns={"ds": "Date", "yhat": "Forecast_Registrations"}
)
forecast_out.to_csv("data/processed/model_outputs/ev_sales_forecast_2035.csv", index=False)

# Save Model
with open("models/prophet_sales_model.json", "w") as fout:
    json.dump(model_to_json(m), fout)

# 5. Plotting
fig = m.plot(forecast)
plt.scatter(
    test_df["ds"], test_df["y"], color="red", label="Holdout Test Data", zorder=5
)
plt.title("Prophet Forecast of EV Registrations (2020 - 2035)")
plt.xlabel("Year")
plt.ylabel("Registrations")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/phase3_forecast/forecast_validation.png", dpi=300)
os.system("cp outputs/phase3_forecast/forecast_validation.png docs/")
plt.close()

print("Phase 3 forecasting completed successfully.")
