import pandas as pd
import pulp
import os

os.makedirs("outputs/phase6_smip", exist_ok=True)

print("Initializing SMIP Model Formulation...")

# ==============================================================================
# 1. LOAD DATA & SPATIAL APPORTIONMENT (Target Year: 2030)
# ==============================================================================
scenarios_df = pd.read_csv("data/processed/model_outputs/smip_scenarios_mc_reduced.csv")

# We focus the optimization on the year 2030 network design
df_2030 = scenarios_df[scenarios_df["Year"] == 2030].copy()

# Dummy Spatial Zones (I) & Demand Shares (based on typical EV spread in India)
# In reality, this would be 28 states.
zones = ["North", "South", "East", "West", "Central"]
zone_shares = {
    "North": 0.25,
    "South": 0.35,
    "East": 0.10,
    "West": 0.20,
    "Central": 0.10,
}

# Potential Recycling Facilities (J)
facilities = ["Delhi_NCR", "Chennai_Hub", "Pune_Hub"]

# Scenario Probabilities (Dynamic from MC Reduction)
scen_probs_df = scenarios_df[["Scenario", "Probability"]].drop_duplicates()
scen_probs = dict(zip(scen_probs_df["Scenario"], scen_probs_df["Probability"]))
scenarios = list(scen_probs.keys())

# Chemistries (C)
chemistries = ["LFP", "NMC"]

# Generate Supply Data: dict indexed by (Zone, Scenario, Chemistry) -> Volume
supply_data = {}
for s in scenarios:
    for c in chemistries:
        # Sum total EOL volume for this scenario and chemistry across all vehicle classes
        vol = df_2030[(df_2030["Scenario"] == s) & (df_2030["Chemistry"] == c)][
            "EOL_Volume"
        ].sum()
        for i in zones:
            supply_data[(i, s, c)] = vol * zone_shares[i]

# ==============================================================================
# 2. DEFINE COST PARAMETERS (Illustrative parameters for prototype)
# ==============================================================================

# Fixed cost to open a facility ($)
fixed_cost = {"Delhi_NCR": 5000000, "Chennai_Hub": 4500000, "Pune_Hub": 4800000}

# Unit capacity expansion cost ($ / battery processed)
unit_cap_cost = 5.0

# Transportation Cost: dict (Zone, Facility) -> cost/battery
# (Dummy distances: closer zones have lower transport costs)
trans_cost = {
    "North": {"Delhi_NCR": 1.0, "Chennai_Hub": 8.0, "Pune_Hub": 5.0},
    "South": {"Delhi_NCR": 8.0, "Chennai_Hub": 1.0, "Pune_Hub": 4.0},
    "East": {"Delhi_NCR": 6.0, "Chennai_Hub": 6.0, "Pune_Hub": 7.0},
    "West": {"Delhi_NCR": 5.0, "Chennai_Hub": 4.0, "Pune_Hub": 1.0},
    "Central": {"Delhi_NCR": 3.0, "Chennai_Hub": 4.0, "Pune_Hub": 3.0},
}

# Informal Sector Penalty: Environmental/Lost Opportunity cost per battery
penalty_cost = 25.0

# Revenue from Black Mass extraction per battery
# NMC yields high-value Cobalt/Nickel. LFP yields lower-value Lithium/Iron.
revenue = {"NMC": 35.0, "LFP": 8.0}

# ==============================================================================
# 3. INITIALIZE PuLP MODEL
# ==============================================================================
model = pulp.LpProblem("EV_Battery_Reverse_Supply_Chain_SMIP", pulp.LpMinimize)

# ==============================================================================
# 4. DECISION VARIABLES
# ==============================================================================

# First-Stage Variables
# y_j: Binary, 1 if facility j is opened
y = pulp.LpVariable.dicts("Open_Facility", facilities, cat="Binary")

# cap_j: Continuous, Capacity of facility j
cap = pulp.LpVariable.dicts("Capacity", facilities, lowBound=0, cat="Continuous")

# Second-Stage Variables
# x_ijs^c: Flow from zone i to facility j under scenario s for chemistry c
x = pulp.LpVariable.dicts(
    "Flow",
    [
        (i, j, s, c)
        for i in zones
        for j in facilities
        for s in scenarios
        for c in chemistries
    ],
    lowBound=0,
    cat="Continuous",
)

# z_is^c: Flow lost to informal sector from zone i under scenario s for chemistry c
z = pulp.LpVariable.dicts(
    "Informal_Loss",
    [(i, s, c) for i in zones for s in scenarios for c in chemistries],
    lowBound=0,
    cat="Continuous",
)

# ==============================================================================
# 5. OBJECTIVE FUNCTION (Minimize Expected Total Cost)
# ==============================================================================

# First-stage cost: Fixed Facility Costs + Capacity Setup Costs
first_stage_cost = pulp.lpSum(
    [fixed_cost[j] * y[j] + unit_cap_cost * cap[j] for j in facilities]
)

# Second-stage cost (Expected across scenarios)
expected_second_stage_cost = pulp.lpSum(
    scen_probs[s]
    * (
        # Transport Costs
        pulp.lpSum(
            trans_cost[i][j] * x[(i, j, s, c)]
            for i in zones
            for j in facilities
            for c in chemistries
        )
        # Informal Sector Penalties
        + pulp.lpSum(penalty_cost * z[(i, s, c)] for i in zones for c in chemistries)
        # Revenues (Subtracted, since we are minimizing cost)
        - pulp.lpSum(
            revenue[c] * x[(i, j, s, c)]
            for i in zones
            for j in facilities
            for c in chemistries
        )
    )
    for s in scenarios
)

model += first_stage_cost + expected_second_stage_cost, "Total_Expected_Cost"

# ==============================================================================
# 6. CONSTRAINTS
# ==============================================================================

# Constraint 1: Supply Balance (All batteries generated must go to a facility OR the informal sector)
for i in zones:
    for s in scenarios:
        for c in chemistries:
            model += (
                pulp.lpSum(x[(i, j, s, c)] for j in facilities) + z[(i, s, c)]
                == supply_data[(i, s, c)],
                f"Supply_Balance_{i}_{s}_{c}",
            )

# Constraint 2: Capacity Limits (Total flow into facility j cannot exceed cap_j)
for j in facilities:
    for s in scenarios:
        model += (
            pulp.lpSum(x[(i, j, s, c)] for i in zones for c in chemistries) <= cap[j],
            f"Capacity_Limit_{j}_{s}",
        )

# Constraint 3: Logical Big-M (Cannot build capacity at j if y_j = 0)
# M is set to a very large number (e.g., total supply in the optimistic scenario)
M = df_2030.groupby("Scenario")["EOL_Volume"].sum().max()
for j in facilities:
    model += (cap[j] <= M * y[j], f"Logical_Capacity_{j}")

# ==============================================================================
# 7. SOLVE
# ==============================================================================
print("Solving SMIP Model...")
model.solve(pulp.PULP_CBC_CMD(msg=0))

# ==============================================================================
# 8. OUTPUT RESULTS
# ==============================================================================
status = pulp.LpStatus[model.status]
print(f"Model Status: {status}")

with open("outputs/phase6_smip/smip_results.txt", "w") as f:
    f.write(f"=== SMIP OPTIMIZATION RESULTS ===\n")
    f.write(f"Status: {status}\n")
    f.write(f"Objective Value (Expected Cost): ${pulp.value(model.objective):,.2f}\n\n")

    f.write("--- FIRST STAGE DECISIONS (Here-and-Now) ---\n")
    for j in facilities:
        if pulp.value(y[j]) > 0.5:
            f.write(
                f"[OPEN] Facility: {j} | Designed Capacity: {pulp.value(cap[j]):,.0f} batteries\n"
            )
        else:
            f.write(f"[CLOSED] Facility: {j}\n")

    f.write("\n--- SECOND STAGE RECOURSE SUMMARY ---\n")
    for s in scenarios:
        f.write(f"\nScenario: {s} (Prob: {scen_probs[s]})\n")

        total_formal = sum(
            pulp.value(x[(i, j, s, c)])
            for i in zones
            for j in facilities
            for c in chemistries
        )
        total_informal = sum(
            pulp.value(z[(i, s, c)]) for i in zones for c in chemistries
        )

        f.write(f"  Processed in Formal Recycling: {total_formal:,.0f} batteries\n")
        f.write(f"  Lost to Informal Sector:       {total_informal:,.0f} batteries\n")

print("SMIP formulation solved and results exported to outputs/phase6_smip/smip_results.txt.")
