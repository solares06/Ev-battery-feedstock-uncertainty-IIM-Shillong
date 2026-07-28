import pandas as pd
import numpy as np
import pulp
import matplotlib.pyplot as plt
import os
import copy

os.makedirs('outputs', exist_ok=True)

print("=" * 60)
print("Phase 8: Sensitivity Analysis")
print("=" * 60)

# ==============================================================================
# REUSABLE SMIP SOLVER FUNCTION
# ==============================================================================

def solve_smip(supply_regions, region_shares, facilities, fixed_cost, unit_cap_cost,
               trans_cost, penalty_cost, revenue, supply_data, scen_probs, chemistries):
    """Solve the SMIP and return key metrics."""
    scenario_list = list(scen_probs.keys())
    
    model = pulp.LpProblem("Sensitivity_SMIP", pulp.LpMinimize)
    
    y = pulp.LpVariable.dicts("Open", facilities, cat='Binary')
    cap = pulp.LpVariable.dicts("Cap", facilities, lowBound=0, cat='Continuous')
    x = pulp.LpVariable.dicts("Flow",
        [(i, j, s, c) for i in supply_regions for j in facilities for s in scenario_list for c in chemistries],
        lowBound=0, cat='Continuous')
    z = pulp.LpVariable.dicts("Informal",
        [(i, s, c) for i in supply_regions for s in scenario_list for c in chemistries],
        lowBound=0, cat='Continuous')
    
    first_stage = pulp.lpSum([fixed_cost[j] * y[j] + unit_cap_cost * cap[j] for j in facilities])
    expected_recourse = pulp.lpSum(
        scen_probs[s] * (
            pulp.lpSum(trans_cost[i][j] * x[(i, j, s, c)] for i in supply_regions for j in facilities for c in chemistries)
            + pulp.lpSum(penalty_cost * z[(i, s, c)] for i in supply_regions for c in chemistries)
            - pulp.lpSum(revenue[c] * x[(i, j, s, c)] for i in supply_regions for j in facilities for c in chemistries)
        )
        for s in scenario_list
    )
    model += first_stage + expected_recourse
    
    for i in supply_regions:
        for s in scenario_list:
            for c in chemistries:
                model += pulp.lpSum(x[(i, j, s, c)] for j in facilities) + z[(i, s, c)] == supply_data[(i, s, c)]
    
    for j in facilities:
        for s in scenario_list:
            model += pulp.lpSum(x[(i, j, s, c)] for i in supply_regions for c in chemistries) <= cap[j]
    
    M_val = max(supply_data.values()) * len(supply_regions) * 10
    for j in facilities:
        model += cap[j] <= M_val * y[j]
    
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[model.status] != 'Optimal':
        return None
    
    # Extract results
    opened = [j for j in facilities if pulp.value(y[j]) > 0.5]
    total_cap = sum(pulp.value(cap[j]) for j in facilities)
    obj = pulp.value(model.objective)
    
    # Average informal loss across scenarios
    total_supply = sum(supply_data.values()) / len(scenario_list)
    base_informal = sum(pulp.value(z[(i, 'Base', c)]) for i in supply_regions for c in chemistries)
    informal_pct = (base_informal / (total_supply / len(chemistries) if total_supply > 0 else 1)) * 100
    
    return {
        'obj': obj,
        'n_open': len(opened),
        'opened': opened,
        'total_cap': total_cap,
        'informal_base': base_informal,
        'informal_pct': informal_pct
    }

# ==============================================================================
# LOAD BASE PARAMETERS (Same as Phase 7)
# ==============================================================================
vahan_df = pd.read_csv('data/processed/vahan_registrations.csv')
state_totals = vahan_df.groupby('State')['Registrations'].sum().sort_values(ascending=False)
top10 = state_totals.head(10)
region_shares = (top10 / top10.sum()).to_dict()
supply_regions = list(region_shares.keys())

facilities = ['Delhi_NCR', 'Chennai', 'Pune', 'Hyderabad', 'Ahmedabad', 'Kolkata']

base_fixed_cost = {
    'Delhi_NCR': 40_00_00_000, 'Chennai': 35_00_00_000, 'Pune': 38_00_00_000,
    'Hyderabad': 32_00_00_000, 'Ahmedabad': 36_00_00_000, 'Kolkata': 30_00_00_000,
}
base_unit_cap = 400
base_penalty = 2000
base_revenue = {'NMC': 2800, 'LFP': 650}

cost_per_km = 0.08
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
base_trans_cost = {}
for state in supply_regions:
    base_trans_cost[state] = {fac: distances[state][fac] * cost_per_km for fac in facilities}

scenarios_df = pd.read_csv('data/processed/smip_feedstock_scenarios.csv')
df_2030 = scenarios_df[scenarios_df['Year'] == 2030].copy()
scen_probs = {'Pessimistic': 0.25, 'Base': 0.50, 'Optimistic': 0.25}
chemistries = ['LFP', 'NMC']

base_supply = {}
for s in scen_probs:
    for c in chemistries:
        total_vol = df_2030[(df_2030['Scenario'] == s) & (df_2030['Chemistry'] == c)]['EOL_Volume'].sum()
        for state in supply_regions:
            base_supply[(state, s, c)] = total_vol * region_shares[state]

# ==============================================================================
# SENSITIVITY 1: NMC Revenue (What if NMC prices crash?)
# ==============================================================================
print("\n--- Sensitivity 1: NMC Revenue ---")
nmc_values = [500, 1000, 1500, 2000, 2500, 2800, 3500, 4500]
nmc_results = []
for nmc_rev in nmc_values:
    rev = {'NMC': nmc_rev, 'LFP': 650}
    res = solve_smip(supply_regions, region_shares, facilities, base_fixed_cost,
                     base_unit_cap, base_trans_cost, base_penalty, rev, base_supply, scen_probs, chemistries)
    if res:
        nmc_results.append({'NMC_Revenue': nmc_rev, 'Obj_Crore': res['obj']/1e7, 
                           'Facilities_Open': res['n_open'], 'Opened': ', '.join(res['opened']),
                           'Informal_Loss_Pct': res['informal_pct']})
        print(f"  NMC ₹{nmc_rev}: Obj=₹{res['obj']/1e7:.0f}Cr, Open={res['n_open']}, Informal={res['informal_pct']:.1f}%")

nmc_df = pd.DataFrame(nmc_results)

# ==============================================================================
# SENSITIVITY 2: Informal Sector Penalty (What if enforcement weakens?)
# ==============================================================================
print("\n--- Sensitivity 2: Informal Sector Penalty ---")
penalty_values = [100, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]
penalty_results = []
for pen in penalty_values:
    res = solve_smip(supply_regions, region_shares, facilities, base_fixed_cost,
                     base_unit_cap, base_trans_cost, pen, base_revenue, base_supply, scen_probs, chemistries)
    if res:
        penalty_results.append({'Penalty': pen, 'Obj_Crore': res['obj']/1e7,
                               'Facilities_Open': res['n_open'], 'Opened': ', '.join(res['opened']),
                               'Informal_Loss_Pct': res['informal_pct']})
        print(f"  Penalty ₹{pen}: Obj=₹{res['obj']/1e7:.0f}Cr, Open={res['n_open']}, Informal={res['informal_pct']:.1f}%")

penalty_df = pd.DataFrame(penalty_results)

# ==============================================================================
# SENSITIVITY 3: Fixed Facility Cost (What if building plants is more expensive?)
# ==============================================================================
print("\n--- Sensitivity 3: Fixed Facility Cost Multiplier ---")
fixed_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
fixed_results = []
for mult in fixed_multipliers:
    scaled_fixed = {j: int(v * mult) for j, v in base_fixed_cost.items()}
    res = solve_smip(supply_regions, region_shares, facilities, scaled_fixed,
                     base_unit_cap, base_trans_cost, base_penalty, base_revenue, base_supply, scen_probs, chemistries)
    if res:
        fixed_results.append({'Multiplier': mult, 'Obj_Crore': res['obj']/1e7,
                             'Facilities_Open': res['n_open'], 'Opened': ', '.join(res['opened']),
                             'Informal_Loss_Pct': res['informal_pct']})
        print(f"  Fixed Cost x{mult}: Obj=₹{res['obj']/1e7:.0f}Cr, Open={res['n_open']}, Informal={res['informal_pct']:.1f}%")

fixed_df = pd.DataFrame(fixed_results)

# ==============================================================================
# SENSITIVITY 4: Transport Cost (What if fuel/logistics costs increase?)
# ==============================================================================
print("\n--- Sensitivity 4: Transport Cost Multiplier ---")
trans_multipliers = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
trans_results = []
for mult in trans_multipliers:
    scaled_trans = {}
    for state in supply_regions:
        scaled_trans[state] = {fac: base_trans_cost[state][fac] * mult for fac in facilities}
    res = solve_smip(supply_regions, region_shares, facilities, base_fixed_cost,
                     base_unit_cap, scaled_trans, base_penalty, base_revenue, base_supply, scen_probs, chemistries)
    if res:
        trans_results.append({'Multiplier': mult, 'Obj_Crore': res['obj']/1e7,
                             'Facilities_Open': res['n_open'], 'Opened': ', '.join(res['opened']),
                             'Informal_Loss_Pct': res['informal_pct']})
        print(f"  Transport x{mult}: Obj=₹{res['obj']/1e7:.0f}Cr, Open={res['n_open']}, Informal={res['informal_pct']:.1f}%")

trans_df = pd.DataFrame(trans_results)

# ==============================================================================
# VISUALIZATION: 2x2 Sensitivity Dashboard
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sensitivity Analysis of SMIP Model Parameters', fontsize=16, fontweight='bold')

# Plot 1: NMC Revenue vs Objective
ax1 = axes[0, 0]
ax1.plot(nmc_df['NMC_Revenue'], nmc_df['Obj_Crore'], 'o-', color='#FF6B6B', linewidth=2, markersize=8)
ax1.set_xlabel('NMC Revenue (₹/battery)')
ax1.set_ylabel('Expected Cost (₹ Crore)')
ax1.set_title('NMC Revenue Impact')
ax1.axvline(x=2800, color='gray', linestyle='--', alpha=0.5, label='Base Case')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.3)

# Plot 2: Informal Penalty vs Facilities Open
ax2 = axes[0, 1]
ax2.plot(penalty_df['Penalty'], penalty_df['Facilities_Open'], 's-', color='#4ECDC4', linewidth=2, markersize=8)
ax2.set_xlabel('Informal Sector Penalty (₹/battery)')
ax2.set_ylabel('Number of Facilities Opened')
ax2.set_title('Informal Penalty Impact on Network Size')
ax2.axvline(x=2000, color='gray', linestyle='--', alpha=0.5, label='Base Case')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_yticks(range(0, 7))

# Plot 3: Fixed Cost Multiplier vs Objective
ax3 = axes[1, 0]
ax3.plot(fixed_df['Multiplier'], fixed_df['Obj_Crore'], 'D-', color='#45B7D1', linewidth=2, markersize=8)
ax3.set_xlabel('Fixed Cost Multiplier')
ax3.set_ylabel('Expected Cost (₹ Crore)')
ax3.set_title('Fixed Facility Cost Impact')
ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Base Case')
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.3)

# Plot 4: Transport Cost vs Facilities Open
ax4 = axes[1, 1]
ax4.plot(trans_df['Multiplier'], trans_df['Facilities_Open'], '^-', color='#96CEB4', linewidth=2, markersize=8)
ax4.set_xlabel('Transport Cost Multiplier')
ax4.set_ylabel('Number of Facilities Opened')
ax4.set_title('Transport Cost Impact on Network Size')
ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Base Case')
ax4.legend()
ax4.grid(True, linestyle='--', alpha=0.3)
ax4.set_yticks(range(0, 7))

plt.tight_layout()
plt.savefig('outputs/sensitivity_dashboard.png', dpi=300)
os.system('cp outputs/sensitivity_dashboard.png docs/')
plt.close()

# Save tables to CSV
nmc_df.to_csv('outputs/sensitivity_nmc_revenue.csv', index=False)
penalty_df.to_csv('outputs/sensitivity_penalty.csv', index=False)
fixed_df.to_csv('outputs/sensitivity_fixed_cost.csv', index=False)
trans_df.to_csv('outputs/sensitivity_transport.csv', index=False)

print("\n✓ All sensitivity analyses complete.")
print("  Dashboard: outputs/sensitivity_dashboard.png")
print("  Tables:    outputs/sensitivity_*.csv")
