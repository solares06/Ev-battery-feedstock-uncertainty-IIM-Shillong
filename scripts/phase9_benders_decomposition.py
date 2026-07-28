import pandas as pd
import numpy as np
import pulp
import time
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 70)
print("Phase 9: Benders Decomposition for SMIP")
print("=" * 70)

# ==============================================================================
# 1. LOAD DATA (Same as Phase 7)
# ==============================================================================
vahan_df = pd.read_csv('data/processed/vahan_registrations.csv')
state_totals = vahan_df.groupby('State')['Registrations'].sum().sort_values(ascending=False)
top10 = state_totals.head(10)
region_shares = (top10 / top10.sum()).to_dict()
supply_regions = list(region_shares.keys())

facilities = ['Delhi_NCR', 'Chennai', 'Pune', 'Hyderabad', 'Ahmedabad', 'Kolkata']

fixed_cost = {
    'Delhi_NCR': 40_00_00_000, 'Chennai': 35_00_00_000, 'Pune': 38_00_00_000,
    'Hyderabad': 32_00_00_000, 'Ahmedabad': 36_00_00_000, 'Kolkata': 30_00_00_000,
}
unit_cap_cost = 400
penalty_cost = 2000
revenue = {'NMC': 2800, 'LFP': 650}

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
trans_cost = {}
for state in supply_regions:
    trans_cost[state] = {fac: distances[state][fac] * cost_per_km for fac in facilities}

scenarios_df = pd.read_csv('data/processed/smip_feedstock_scenarios.csv')
df_2030 = scenarios_df[scenarios_df['Year'] == 2030].copy()
scen_probs = {'Pessimistic': 0.25, 'Base': 0.50, 'Optimistic': 0.25}
scenario_list = list(scen_probs.keys())
chemistries = ['LFP', 'NMC']

supply_data = {}
for s in scenario_list:
    for c in chemistries:
        total_vol = df_2030[(df_2030['Scenario'] == s) & (df_2030['Chemistry'] == c)]['EOL_Volume'].sum()
        for state in supply_regions:
            supply_data[(state, s, c)] = total_vol * region_shares[state]

# Big-M for capacity
M_val = df_2030[df_2030['Scenario'] == 'Optimistic']['EOL_Volume'].sum() * 2

# ==============================================================================
# 2. BENDERS DECOMPOSITION IMPLEMENTATION
# ==============================================================================

def solve_master(cuts, iteration):
    """
    Solve the Benders Master Problem.
    Master contains: first-stage variables (y, cap) + surrogate variable theta.
    Benders optimality cuts are added as constraints on theta.
    """
    master = pulp.LpProblem(f"Benders_Master_Iter{iteration}", pulp.LpMinimize)
    
    y = pulp.LpVariable.dicts("y", facilities, cat='Binary')
    cap = pulp.LpVariable.dicts("cap", facilities, lowBound=0, cat='Continuous')
    # Theta: surrogate for expected second-stage (recourse) cost
    theta = pulp.LpVariable("theta", lowBound=-1e15, cat='Continuous')
    
    # Objective: first-stage cost + theta
    master += (
        pulp.lpSum([fixed_cost[j] * y[j] + unit_cap_cost * cap[j] for j in facilities]) + theta,
        "Master_Obj"
    )
    
    # Logical Big-M: cap_j <= M * y_j
    for j in facilities:
        master += cap[j] <= M_val * y[j], f"Logic_{j}"
    
    # Benders Optimality Cuts from previous iterations
    for k, cut in enumerate(cuts):
        # cut = {'constant': ..., 'y_coeffs': {j: ...}, 'cap_coeffs': {j: ...}}
        master += (
            theta >= cut['constant'] 
            + pulp.lpSum(cut['y_coeffs'][j] * y[j] for j in facilities)
            + pulp.lpSum(cut['cap_coeffs'][j] * cap[j] for j in facilities),
            f"Benders_Cut_{k}"
        )
    
    master.solve(pulp.PULP_CBC_CMD(msg=0))
    
    y_val = {j: pulp.value(y[j]) for j in facilities}
    cap_val = {j: pulp.value(cap[j]) for j in facilities}
    theta_val = pulp.value(theta)
    obj_val = pulp.value(master.objective)
    
    return y_val, cap_val, theta_val, obj_val


def solve_subproblem(y_fixed, cap_fixed, scenario):
    """
    Solve one Benders Subproblem for a given scenario.
    Given fixed first-stage (y*, cap*), solve the second-stage LP.
    Returns: objective value, and dual variables for cut generation.
    """
    s = scenario
    sub = pulp.LpProblem(f"Benders_Sub_{s}", pulp.LpMinimize)
    
    # Second-stage variables
    x = pulp.LpVariable.dicts("x",
        [(i, j, c) for i in supply_regions for j in facilities for c in chemistries],
        lowBound=0, cat='Continuous')
    z = pulp.LpVariable.dicts("z",
        [(i, c) for i in supply_regions for c in chemistries],
        lowBound=0, cat='Continuous')
    
    # Objective for this scenario (no probability weighting here; master handles it)
    sub += (
        pulp.lpSum(trans_cost[i][j] * x[(i, j, c)] for i in supply_regions for j in facilities for c in chemistries)
        + pulp.lpSum(penalty_cost * z[(i, c)] for i in supply_regions for c in chemistries)
        - pulp.lpSum(revenue[c] * x[(i, j, c)] for i in supply_regions for j in facilities for c in chemistries),
        "Sub_Obj"
    )
    
    # Supply balance constraints (dual: pi)
    # Sanitize names: replace spaces with underscores for PuLP constraint keys
    for i in supply_regions:
        for c in chemistries:
            safe_name = f"Supply_{i.replace(' ', '_')}_{c}"
            sub += (
                pulp.lpSum(x[(i, j, c)] for j in facilities) + z[(i, c)] == supply_data[(i, s, c)],
                safe_name
            )
    
    # Capacity constraints (dual: mu)
    for j in facilities:
        safe_name = f"Cap_{j}"
        sub += (
            pulp.lpSum(x[(i, j, c)] for i in supply_regions for c in chemistries) <= cap_fixed[j],
            safe_name
        )
    
    sub.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[sub.status] != 'Optimal':
        return None, None, None
    
    obj = pulp.value(sub.objective)
    
    # Extract dual values
    pi_vals = {}
    for i in supply_regions:
        for c in chemistries:
            safe_name = f"Supply_{i.replace(' ', '_')}_{c}"
            pi_vals[(i, c)] = sub.constraints[safe_name].pi if sub.constraints[safe_name].pi is not None else 0
    
    mu_vals = {}
    for j in facilities:
        safe_name = f"Cap_{j}"
        mu_vals[j] = sub.constraints[safe_name].pi if sub.constraints[safe_name].pi is not None else 0
    
    return obj, pi_vals, mu_vals


def generate_benders_cut(scenarios_results, y_fixed, cap_fixed):
    """
    Generate a single aggregated optimality cut from all scenario subproblems.
    
    The cut has the form:
        theta >= constant + sum_j(y_coeff_j * y_j) + sum_j(cap_coeff_j * cap_j)
    """
    constant = 0.0
    y_coeffs = {j: 0.0 for j in facilities}
    cap_coeffs = {j: 0.0 for j in facilities}
    
    for s, (obj, pi_vals, mu_vals) in scenarios_results.items():
        prob = scen_probs[s]
        
        # Constant term: sum of pi * supply_data (RHS of supply constraints)
        rhs_sum = sum(pi_vals[(i, c)] * supply_data[(i, s, c)] for i in supply_regions for c in chemistries)
        # Plus sum of mu * cap_fixed (RHS of capacity constraints, but cap is a variable)
        # The cut coefficients for cap come from mu
        constant += prob * rhs_sum
        
        for j in facilities:
            # mu_j is the dual of the capacity constraint: flow <= cap_j
            # In the cut, cap_j's coefficient is prob * mu_j
            cap_coeffs[j] += prob * mu_vals[j]
    
    # Adjust constant: subtract the cap terms evaluated at current cap_fixed
    # Because the cut should be: theta >= constant_pure + cap_coeffs * cap
    # And constant_pure = constant - cap_coeffs * cap_fixed
    adjusted_constant = constant - sum(cap_coeffs[j] * cap_fixed[j] for j in facilities)
    
    return {
        'constant': adjusted_constant,
        'y_coeffs': y_coeffs,  # y doesn't appear directly in subproblem RHS
        'cap_coeffs': cap_coeffs,
    }


# ==============================================================================
# 3. BENDERS ITERATION LOOP
# ==============================================================================

MAX_ITER = 50
TOLERANCE = 1e-4  # Relative gap tolerance

cuts = []
upper_bound = float('inf')
lower_bound = float('-inf')
convergence_log = []

print(f"\n{'Iter':>4} | {'Lower Bound (₹ Cr)':>18} | {'Upper Bound (₹ Cr)':>18} | {'Gap (%)':>10} | {'Cuts':>5}")
print("-" * 75)

start_time = time.time()

for iteration in range(1, MAX_ITER + 1):
    # Step 1: Solve Master Problem
    y_val, cap_val, theta_val, master_obj = solve_master(cuts, iteration)
    lower_bound = master_obj
    
    # Step 2: Solve Subproblems for each scenario
    total_recourse = 0.0
    scenarios_results = {}
    
    for s in scenario_list:
        obj_s, pi_s, mu_s = solve_subproblem(y_val, cap_val, s)
        if obj_s is None:
            print(f"  Subproblem {s} infeasible at iteration {iteration}!")
            break
        scenarios_results[s] = (obj_s, pi_s, mu_s)
        total_recourse += scen_probs[s] * obj_s
    
    # Compute Upper Bound
    first_stage_cost = sum(fixed_cost[j] * y_val[j] + unit_cap_cost * cap_val[j] for j in facilities)
    current_ub = first_stage_cost + total_recourse
    upper_bound = min(upper_bound, current_ub)
    
    # Convergence check
    if abs(upper_bound) > 1e-6:
        gap = abs(upper_bound - lower_bound) / abs(upper_bound) * 100
    else:
        gap = abs(upper_bound - lower_bound) * 100
    
    convergence_log.append({
        'Iteration': iteration,
        'Lower_Bound': lower_bound / 1e7,
        'Upper_Bound': upper_bound / 1e7,
        'Gap_Pct': gap,
        'Cuts': len(cuts)
    })
    
    print(f"{iteration:>4} | {lower_bound/1e7:>18,.2f} | {upper_bound/1e7:>18,.2f} | {gap:>9.4f}% | {len(cuts):>5}")
    
    if gap < TOLERANCE:
        print(f"\n✓ Converged in {iteration} iterations! Gap = {gap:.6f}%")
        break
    
    # Step 3: Generate Benders Cut and add to master
    cut = generate_benders_cut(scenarios_results, y_val, cap_val)
    cuts.append(cut)

elapsed = time.time() - start_time

# ==============================================================================
# 4. FINAL RESULTS
# ==============================================================================
opened_facs = [j for j in facilities if y_val[j] > 0.5]

results_lines = []
results_lines.append("=" * 70)
results_lines.append("BENDERS DECOMPOSITION RESULTS")
results_lines.append("=" * 70)
results_lines.append(f"Converged: {'Yes' if gap < TOLERANCE else 'No'}")
results_lines.append(f"Iterations: {iteration}")
results_lines.append(f"Final Gap: {gap:.6f}%")
results_lines.append(f"Elapsed Time: {elapsed:.2f} seconds")
results_lines.append(f"Optimal Objective: ₹{upper_bound:,.0f} (₹{upper_bound/1e7:.2f} Crore)")
results_lines.append(f"Number of Benders Cuts Generated: {len(cuts)}")
results_lines.append("")
results_lines.append("--- FACILITY DECISIONS ---")
for j in facilities:
    marker = "[OPEN]" if y_val[j] > 0.5 else "[CLOSED]"
    results_lines.append(f"  {marker} {j}: Capacity = {cap_val[j]:,.0f}")
results_lines.append("")
results_lines.append("--- CONVERGENCE LOG ---")
results_lines.append(f"{'Iter':>4} | {'LB (₹ Cr)':>12} | {'UB (₹ Cr)':>12} | {'Gap (%)':>10}")
for row in convergence_log:
    results_lines.append(f"{row['Iteration']:>4} | {row['Lower_Bound']:>12,.2f} | {row['Upper_Bound']:>12,.2f} | {row['Gap_Pct']:>9.4f}%")

results_text = "\n".join(results_lines)
with open('outputs/benders_results.txt', 'w') as f:
    f.write(results_text)
print("\n" + results_text)

# ==============================================================================
# 5. CONVERGENCE PLOT
# ==============================================================================
conv_df = pd.DataFrame(convergence_log)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Benders Decomposition Convergence', fontsize=14, fontweight='bold')

# Bounds convergence
ax1.plot(conv_df['Iteration'], conv_df['Lower_Bound'], 'o-', color='blue', label='Lower Bound (Master)', linewidth=2)
ax1.plot(conv_df['Iteration'], conv_df['Upper_Bound'], 's-', color='red', label='Upper Bound (Subproblems)', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Objective Value (₹ Crore)')
ax1.set_title('Upper & Lower Bound Convergence')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.3)

# Gap convergence
ax2.plot(conv_df['Iteration'], conv_df['Gap_Pct'], 'D-', color='green', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Optimality Gap (%)')
ax2.set_title('Optimality Gap Convergence')
ax2.axhline(y=TOLERANCE, color='red', linestyle='--', alpha=0.5, label=f'Tolerance ({TOLERANCE}%)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/benders_convergence.png', dpi=300)
os.system('cp outputs/benders_convergence.png docs/')
plt.close()

# Save convergence log
conv_df.to_csv('outputs/benders_convergence_log.csv', index=False)

print(f"\n✓ Phase 9 complete. Results in outputs/benders_results.txt")
