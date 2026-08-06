"""
Phase 8: Validation against NITI Aayog Benchmark
=================================================
Validates the cumulative 2030 battery projection (GWh) against the
NITI Aayog 'Advanced Chemistry Cell Battery Reuse and Recycling Market in India'
report, which projects an addressable market of ~128 GWh by 2030.

Methodology:
    1. Load 2030 EOL volume by vehicle class (from Phase 4 or Phase 5c).
    2. Apply standard battery capacities (kWh) by class to convert counts to GWh.
    3. Compare expected GWh against the 128 GWh benchmark.
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("Phase 8: NITI Aayog Benchmark Validation")
print("=" * 60)

# 1. Load MC reduced scenarios (probabilistic expected value)
try:
    smip_df = pd.read_csv("data/processed/model_outputs/smip_scenarios_mc_reduced.csv")
    print("Loaded reduced MC scenarios.")
except FileNotFoundError:
    print("MC scenarios not found. Falling back to deterministic scenarios.")
    smip_df = pd.read_csv("data/processed/model_outputs/smip_feedstock_scenarios.csv")

# 2. Extract 2030 data and calculate probability-weighted volume
df_2030 = smip_df[smip_df["Year"] == 2030]
# Calculate expected value per vehicle class
expected_volumes = (
    df_2030.groupby(["Vehicle_Class", "Scenario"])
    .agg(volume=("EOL_Volume", "sum"), prob=("Probability", "first"))
    .reset_index()
)
expected_volumes["expected_vol"] = expected_volumes["volume"] * expected_volumes["prob"]
class_volumes = (
    expected_volumes.groupby("Vehicle_Class")["expected_vol"].sum().to_dict()
)

print("\n--- 2030 Expected EOL Batteries (Count) ---")
for cls, vol in class_volumes.items():
    print(f"  {cls}: {vol:,.0f}")
print(f"  Total: {sum(class_volumes.values()):,.0f}")

# 3. Convert to GWh
# Standard assumptions for average pack size (kWh)
# Source: NITI Aayog / IEA proxy estimates
capacities_kwh = {
    "2 and 3 wheelers": 3.5,
    "Cars": 30.0,
    "Vans": 40.0,
    "Buses": 150.0,
    "Trucks": 150.0,
}

print("\n--- 2030 Expected EOL Batteries (GWh) ---")
total_gwh = 0
for cls, vol in class_volumes.items():
    gwh = (vol * capacities_kwh.get(cls, 30.0)) / 1_000_000  # kWh to GWh
    total_gwh += gwh
    print(f"  {cls}: {gwh:,.2f} GWh")

print(f"  --------------------------")
print(f"  Total Projected: {total_gwh:,.2f} GWh")

# 4. Compare with NITI Aayog Benchmark
NITI_BENCHMARK_GWH = 128.0
diff = total_gwh - NITI_BENCHMARK_GWH
pct_diff = (diff / NITI_BENCHMARK_GWH) * 100

print(f"\n--- Benchmark Validation ---")
print(f"  NITI Aayog 2030 Benchmark: ~{NITI_BENCHMARK_GWH} GWh")
print(f"  Model Projection:          ~{total_gwh:,.2f} GWh")
print(f"  Difference:                {diff:+.2f} GWh ({pct_diff:+.1f}%)")

if abs(pct_diff) <= 25.0:
    print(
        "\n✓ VALIDATION PASSED: The projection is well within the acceptable bounds (±25%) of the NITI Aayog estimate."
    )
else:
    print(
        "\n✗ VALIDATION WARNING: The projection diverges significantly from the NITI Aayog estimate. Check market assumptions (e.g., car share ratio)."
    )

print("=" * 60)
