import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("Phase 7: State-Level SMIP with Realistic INR Costs")
print("=" * 60)

# ==============================================================================
# 1. COMPUTE REAL STATE-LEVEL SHARES FROM VAHAN DATA
# ==============================================================================
vahan_df = pd.read_csv('data/processed/vahan_registrations.csv')
state_totals = vahan_df.groupby('State')['Registrations'].sum().sort_values(ascending=False)

# Top 10 States
top10_states = state_totals.head(10)
# Normalize shares so they sum to 1.0 (we attribute 100% of feedstock to these 10)
top10_shares = (top10_states / top10_states.sum()).to_dict()

# Clean state names for readability
state_short = {
    'UTTAR PRADESH': 'UP', 'MAHARASHTRA': 'MH', 'TAMIL NADU': 'TN',
    'GUJARAT': 'GJ', 'KARNATAKA': 'KA', 'MADHYA PRADESH': 'MP',
    'RAJASTHAN': 'RJ', 'BIHAR': 'BR', 'WEST BENGAL': 'WB', 'TELANGANA': 'TG'
}

supply_regions = list(top10_shares.keys())
region_shares = top10_shares

print("\nSupply Regions (Top 10 EV States):")
for state, share in region_shares.items():
    print(f"  {state} ({state_short[state]}): {share:.1%}")

# ==============================================================================
# 2. FACILITY CANDIDATE LOCATIONS & PARAMETERS (INR)
# ==============================================================================
facilities = ['Delhi_NCR', 'Chennai', 'Pune', 'Hyderabad', 'Ahmedabad', 'Kolkata']

# Fixed cost to open a facility (INR)
# ~₹30-50 Crore for mid-scale hydrometallurgical plant
fixed_cost = {
    'Delhi_NCR': 40_00_00_000,   # ₹40 Cr
    'Chennai':   35_00_00_000,   # ₹35 Cr
    'Pune':      38_00_00_000,   # ₹38 Cr
    'Hyderabad': 32_00_00_000,   # ₹32 Cr
    'Ahmedabad': 36_00_00_000,   # ₹36 Cr
    'Kolkata':   30_00_00_000,   # ₹30 Cr
}

# Unit capacity cost (INR/battery)
unit_cap_cost = 400  # ₹400/battery

# Informal Sector Penalty (INR/battery) — environmental + lost material
penalty_cost = 2000  # ₹2,000/battery

# Revenue from Black Mass (INR/battery)
revenue = {'NMC': 2800, 'LFP': 650}  # NMC is much more valuable

# ==============================================================================
# 3. INTER-STATE DISTANCE MATRIX (Approximate km between state capitals & facilities)
# ==============================================================================
# Transport cost = ₹0.08 per battery per km (approx ₹4-5/km/tonne, ~50 batteries/tonne)
cost_per_km = 0.08

# Approximate road distances (km) from state capital to each facility city
distances = {
    'UTTAR PRADESH':  {'Delhi_NCR': 450,  'Chennai': 2200, 'Pune': 1400, 'Hyderabad': 1250, 'Ahmedabad': 900,  'Kolkata': 1000},
    'MAHARASHTRA':    {'Delhi_NCR': 1400, 'Chennai': 1200, 'Pune': 150,  'Hyderabad': 550,  'Ahmedabad': 530,  'Kolkata': 1900},
    'TAMIL NADU':     {'Delhi_NCR': 2200, 'Chennai': 0,    'Pune': 1200, 'Hyderabad': 630,  'Ahmedabad': 1800, 'Kolkata': 1700},
    'GUJARAT':        {'Delhi_NCR': 950,  'Chennai': 1800, 'Pune': 530,  'Hyderabad': 1100, 'Ahmedabad': 0,    'Kolkata': 2100},
    'KARNATAKA':      {'Delhi_NCR': 2100, 'Chennai': 350,  'Pune': 850,  'Hyderabad': 570,  'Ahmedabad': 1500, 'Kolkata': 1900},
    'MADHYA PRADESH': {'Delhi_NCR': 780,  'Chennai': 1700, 'Pune': 850,  'Hyderabad': 750,  'Ahmedabad': 600,  'Kolkata': 1300},
    'RAJASTHAN':      {'Delhi_NCR': 400,  'Chennai': 2300, 'Pune': 1200, 'Hyderabad': 1400, 'Ahmedabad': 650,  'Kolkata': 1600},
    'BIHAR':          {'Delhi_NCR': 1000, 'Chennai': 1900, 'Pune': 1700, 'Hyderabad': 1550, 'Ahmedabad': 1600, 'Kolkata': 500},
    'WEST BENGAL':    {'Delhi_NCR': 1500, 'Chennai': 1700, 'Pune': 1900, 'Hyderabad': 1500, 'Ahmedabad': 2100, 'Kolkata': 0},
    'TELANGANA':      {'Delhi_NCR': 1500, 'Chennai': 630,  'Pune': 550,  'Hyderabad': 0,    'Ahmedabad': 1100, 'Kolkata': 1500},
}

# Convert to ₹/battery transport cost
trans_cost = {}
for state in supply_regions:
    trans_cost[state] = {}
    for fac in facilities:
        trans_cost[state][fac] = distances[state][fac] * cost_per_km

# ==============================================================================
# 4. LOAD SCENARIO DATA (2030)
# ==============================================================================
scenarios_df = pd.read_csv('data/processed/smip_feedstock_scenarios.csv')
df_2030 = scenarios_df[scenarios_df['Year'] == 2030].copy()

scen_probs = {'Pessimistic': 0.25, 'Base': 0.50, 'Optimistic': 0.25}
scenario_list = list(scen_probs.keys())
chemistries = ['LFP', 'NMC']

# Supply data: (State, Scenario, Chemistry) -> EOL Volume
supply_data = {}
for s in scenario_list:
    for c in chemistries:
        total_vol = df_2030[(df_2030['Scenario'] == s) & (df_2030['Chemistry'] == c)]['EOL_Volume'].sum()
        for state in supply_regions:
            supply_data[(state, s, c)] = total_vol * region_shares[state]

# ==============================================================================
# 5. BUILD PuLP MODEL
# ==============================================================================
print("\nFormulating SMIP Model...")
model = pulp.LpProblem("EV_Battery_SMIP_State_Level", pulp.LpMinimize)

# First-Stage Variables
y = pulp.LpVariable.dicts("Open", facilities, cat='Binary')
cap = pulp.LpVariable.dicts("Cap", facilities, lowBound=0, cat='Continuous')

# Second-Stage Variables
x = pulp.LpVariable.dicts("Flow",
    [(i, j, s, c) for i in supply_regions for j in facilities for s in scenario_list for c in chemistries],
    lowBound=0, cat='Continuous')

z = pulp.LpVariable.dicts("Informal",
    [(i, s, c) for i in supply_regions for s in scenario_list for c in chemistries],
    lowBound=0, cat='Continuous')

# Objective Function
first_stage = pulp.lpSum([fixed_cost[j] * y[j] + unit_cap_cost * cap[j] for j in facilities])

expected_recourse = pulp.lpSum(
    scen_probs[s] * (
        pulp.lpSum(trans_cost[i][j] * x[(i, j, s, c)] for i in supply_regions for j in facilities for c in chemistries)
        + pulp.lpSum(penalty_cost * z[(i, s, c)] for i in supply_regions for c in chemistries)
        - pulp.lpSum(revenue[c] * x[(i, j, s, c)] for i in supply_regions for j in facilities for c in chemistries)
    )
    for s in scenario_list
)

model += first_stage + expected_recourse, "Total_Expected_Cost_INR"

# Constraints
# Supply Balance
for i in supply_regions:
    for s in scenario_list:
        for c in chemistries:
            model += (
                pulp.lpSum(x[(i, j, s, c)] for j in facilities) + z[(i, s, c)] == supply_data[(i, s, c)],
                f"Supply_{i}_{s}_{c}"
            )

# Capacity
for j in facilities:
    for s in scenario_list:
        model += (
            pulp.lpSum(x[(i, j, s, c)] for i in supply_regions for c in chemistries) <= cap[j],
            f"Cap_{j}_{s}"
        )

# Big-M Logical
M = df_2030[df_2030['Scenario'] == 'Optimistic']['EOL_Volume'].sum()
for j in facilities:
    model += (cap[j] <= M * y[j], f"Logic_{j}")

# ==============================================================================
# 6. SOLVE
# ==============================================================================
print("Solving...")
model.solve(pulp.PULP_CBC_CMD(msg=0))
status = pulp.LpStatus[model.status]
print(f"Status: {status}")
print(f"Objective Value: ₹{pulp.value(model.objective):,.0f}")

# ==============================================================================
# 7. OUTPUT RESULTS
# ==============================================================================
results_lines = []
results_lines.append("=" * 70)
results_lines.append("STATE-LEVEL SMIP OPTIMIZATION RESULTS (Year 2030, INR)")
results_lines.append("=" * 70)
results_lines.append(f"Status: {status}")
results_lines.append(f"Objective Value (Expected Cost): ₹{pulp.value(model.objective):,.0f}")
results_lines.append(f"  = ₹{pulp.value(model.objective)/1e7:.2f} Crore\n")

results_lines.append("--- FIRST STAGE: FACILITY DECISIONS ---")
facility_caps = {}
for j in facilities:
    opened = pulp.value(y[j]) > 0.5
    capacity = pulp.value(cap[j])
    facility_caps[j] = capacity if opened else 0
    marker = "[OPEN]" if opened else "[CLOSED]"
    results_lines.append(f"  {marker} {j}: Capacity = {capacity:,.0f} batteries")

results_lines.append("\n--- SECOND STAGE: SCENARIO RECOURSE ---")
for s in scenario_list:
    results_lines.append(f"\n  Scenario: {s} (Prob: {scen_probs[s]})")
    total_formal = sum(pulp.value(x[(i, j, s, c)]) for i in supply_regions for j in facilities for c in chemistries)
    total_informal = sum(pulp.value(z[(i, s, c)]) for i in supply_regions for c in chemistries)
    results_lines.append(f"    Formal Recycling:     {total_formal:>15,.0f} batteries")
    results_lines.append(f"    Informal Sector Loss: {total_informal:>15,.0f} batteries")
    results_lines.append(f"    Informal Loss Rate:   {total_informal / (total_formal + total_informal) * 100:.1f}%")

    # Top flows
    results_lines.append(f"    Top State→Facility Flows:")
    flows = []
    for i in supply_regions:
        for j in facilities:
            flow = sum(pulp.value(x[(i, j, s, c)]) for c in chemistries)
            if flow > 1000:
                flows.append((i, j, flow))
    flows.sort(key=lambda x: x[2], reverse=True)
    for state, fac, flow in flows[:8]:
        results_lines.append(f"      {state_short[state]:>2} → {fac:<14}: {flow:>12,.0f}")

results_text = "\n".join(results_lines)
with open('outputs/smip_results_state_level.txt', 'w') as f:
    f.write(results_text)
print(results_text)

# ==============================================================================
# 8. VISUALIZATION: Facility Capacity Bar Chart
# ==============================================================================
opened_facs = {j: v for j, v in facility_caps.items() if v > 0}

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
bars = ax.bar(opened_facs.keys(), [v / 1e6 for v in opened_facs.values()], color=colors[:len(opened_facs)])
ax.set_title('Optimal Recycling Facility Capacities (2030)', fontsize=14, fontweight='bold')
ax.set_ylabel('Capacity (Millions of Batteries)', fontsize=12)
ax.set_xlabel('Facility Location', fontsize=12)
for bar, val in zip(bars, opened_facs.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{val/1e6:.1f}M', ha='center', fontsize=10, fontweight='bold')
ax.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/facility_decisions_chart.png', dpi=300)
os.system('cp outputs/facility_decisions_chart.png docs/')
plt.close()

print("\nPhase 7 complete. Results saved to outputs/smip_results_state_level.txt")
