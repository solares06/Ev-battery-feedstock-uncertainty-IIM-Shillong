"""
Phase 5c: Scenario Reduction via K-Means Clustering
====================================================
Reduces the 10,000 Monte Carlo EOL paths from Phase 5b down to K
representative scenarios using K-means clustering with medoid selection.

Methodology:
    1. Standardize MC paths for clustering (zero-mean, unit-variance per year)
    2. Evaluate K from 3 to 12 using silhouette score
    3. Select optimal K (constrained to 5–10 per project brief)
    4. Run K-means; select nearest actual MC sample to each centroid
    5. Assign probability = cluster_size / N_total
    6. Disaggregate representative scenarios by vehicle class & battery chemistry
    7. Output in SMIP-ready format identical to existing pipeline

Output Format (one row per Year × Scenario × Vehicle_Class × Chemistry):
    Year, Scenario, Probability, Vehicle_Class, Chemistry, EOL_Volume

Outputs:
    data/processed/model_outputs/smip_scenarios_mc_reduced.csv   - SMIP-ready scenario set
    outputs/phase5_scenarios/scenario_k_selection.png               - Silhouette & Elbow plot
    outputs/phase5_scenarios/representative_scenarios_overlay.png   - Representative paths on MC cloud
"""

import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

os.makedirs("outputs/phase5_scenarios", exist_ok=True)
os.makedirs("data/processed/model_outputs", exist_ok=True)

print("=" * 60)
print("Phase 5c: Scenario Reduction (K-Means Clustering)")
print("=" * 60)

# ============================================================
# 1. LOAD MC DATA FROM PHASE 5b
# ============================================================

mc_eol_raw = pd.read_csv("data/processed/model_outputs/mc_eol_paths_raw.csv")
mc_sales_raw = pd.read_csv("data/processed/model_outputs/mc_sales_samples.csv")

eol_paths = mc_eol_raw.values
sales_samples = mc_sales_raw.values
N_SAMPLES = eol_paths.shape[0]

target_years = np.array([int(c) for c in mc_eol_raw.columns])
sale_years = np.array([int(c) for c in mc_sales_raw.columns])

print(f"Loaded {N_SAMPLES:,} MC paths")
print(
    f"  Target years: {target_years.min()}–{target_years.max()} ({len(target_years)} years)"
)
print(
    f"  Sale years:   {sale_years.min()}–{sale_years.max()} ({len(sale_years)} years)"
)

# ============================================================
# 2. LOAD AUXILIARY DATA
# ============================================================

# Vehicle class shares (IEA)
iea_df = pd.read_csv("data/processed/iea/iea_india_ev_sales.csv")
iea_bev = iea_df[
    (iea_df["powertrain"] == "BEV")
    & (iea_df["mode"].isin(["2 and 3 wheelers", "Cars", "Buses", "Vans", "Trucks"]))
]
iea_recent = iea_bev[(iea_bev["year"] >= 2020) & (iea_bev["year"] <= 2025)]
class_shares = (
    iea_recent.groupby("mode")["value"].sum() / iea_recent["value"].sum()
).to_dict()

# Weibull parameters (same as Phase 4/5b)
weibull_params = {
    "2 and 3 wheelers": {"shape": 3.5, "scale": 5.0},
    "Cars": {"shape": 3.0, "scale": 10.0},
    "Buses": {"shape": 2.5, "scale": 7.0},
    "Vans": {"shape": 3.0, "scale": 8.0},
    "Trucks": {"shape": 2.5, "scale": 8.0},
}

# Pre-compute per-class convolution matrices
conv_matrices = {}
for cls, params in weibull_params.items():
    beta, lam = params["shape"], params["scale"]
    M = np.zeros((len(sale_years), len(target_years)))
    for i, sy in enumerate(sale_years):
        for j, ty in enumerate(target_years):
            age = ty - sy
            if age <= 0:
                continue
            M[i, j] = weibull_min.cdf(age, beta, scale=lam) - weibull_min.cdf(
                age - 1, beta, scale=lam
            )
    conv_matrices[cls] = M

# Chemistry mix by cohort year (for disaggregation)
chem_df = pd.read_csv("data/processed/chemistry/chemistry_mix.csv")
chem_df["Date"] = pd.to_datetime(chem_df["Date"])
chem_df["Year"] = chem_df["Date"].dt.year
annual_chem = (
    chem_df.groupby("Year").first().reset_index()[["Year", "LFP_Share", "NMC_Share"]]
)

# Build LFP share vector aligned to sale_years
# For years beyond chemistry_mix.csv coverage, use last known value
last_known_lfp = annual_chem["LFP_Share"].iloc[-1]
lfp_shares = np.array(
    [
        (
            annual_chem.loc[annual_chem["Year"] == y, "LFP_Share"].values[0]
            if y in annual_chem["Year"].values
            else last_known_lfp
        )
        for y in sale_years
    ]
)

print(f"\nChemistry mix (LFP share by cohort year):")
for i, sy in enumerate(sale_years):
    if sy in [2020, 2023, 2026, 2030, 2035]:
        print(f"  {sy}: LFP={lfp_shares[i]:.0%}, NMC={1-lfp_shares[i]:.0%}")

# ============================================================
# 3. K-MEANS CLUSTERING WITH SILHOUETTE ANALYSIS
# ============================================================

# Standardize EOL paths so each year contributes equally to clustering
scaler = StandardScaler()
eol_scaled = scaler.fit_transform(eol_paths)

K_MIN, K_MAX = 3, 12
K_range = range(K_MIN, K_MAX + 1)
silhouette_scores = []
inertias = []

print(f"\nEvaluating K = {K_MIN}..{K_MAX}...")
print(f"{'K':>4} | {'Silhouette':>11} | {'Inertia':>14}")
print("-" * 35)

for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(eol_scaled)
    sil = silhouette_score(
        eol_scaled, labels, sample_size=min(5000, N_SAMPLES), random_state=42
    )
    silhouette_scores.append(sil)
    inertias.append(km.inertia_)
    print(f"{k:>4} | {sil:>11.4f} | {km.inertia_:>14,.0f}")

# Select K: best silhouette, constrained to [5, 10] per project brief
best_k_idx = np.argmax(silhouette_scores)
K_OPTIMAL = list(K_range)[best_k_idx]
if K_OPTIMAL < 5:
    K_OPTIMAL = 5
elif K_OPTIMAL > 10:
    K_OPTIMAL = 10

print(
    f"\n→ Selected K = {K_OPTIMAL} (silhouette = {silhouette_scores[K_OPTIMAL - K_MIN]:.4f})"
)

# ============================================================
# 4. FINAL CLUSTERING & MEDOID SELECTION
# ============================================================

km_final = KMeans(n_clusters=K_OPTIMAL, n_init=20, random_state=42, max_iter=500)
labels = km_final.fit_predict(eol_scaled)

# For each cluster, find the actual MC sample nearest to the centroid
# (this is a medoid approximation using K-means centroids)
centroids = km_final.cluster_centers_
representative_indices = []

for k in range(K_OPTIMAL):
    cluster_mask = labels == k
    cluster_points = eol_scaled[cluster_mask]
    cluster_original_indices = np.where(cluster_mask)[0]

    # Euclidean distance from each member to its centroid
    dists = cdist(cluster_points, centroids[k : k + 1, :], metric="euclidean").ravel()
    nearest_in_cluster = np.argmin(dists)
    representative_indices.append(cluster_original_indices[nearest_in_cluster])

# Probabilities = cluster size / total
cluster_sizes = np.bincount(labels, minlength=K_OPTIMAL)
probabilities = cluster_sizes / N_SAMPLES

# Sort scenarios by 2030 EOL volume (ascending) for readable naming
yr_2030_idx = (
    np.where(target_years == 2030)[0][0]
    if 2030 in target_years
    else len(target_years) // 2
)
sort_order = np.argsort([eol_paths[idx, yr_2030_idx] for idx in representative_indices])
representative_indices = [representative_indices[i] for i in sort_order]
probabilities = probabilities[sort_order]
cluster_sizes_sorted = cluster_sizes[sort_order]

print(f"\n--- {K_OPTIMAL} Representative Scenarios (sorted by 2030 EOL volume) ---")
print(
    f"{'Scenario':>10} | {'Prob':>7} | {'Cluster':>8} | {'2025 EOL':>14} | {'2030 EOL':>14} | {'2035 EOL':>14}"
)
print("-" * 80)

yr_2025_idx = np.where(target_years == 2025)[0][0] if 2025 in target_years else 5
yr_2035_idx = np.where(target_years == 2035)[0][0] if 2035 in target_years else -1

for k, (idx, prob) in enumerate(zip(representative_indices, probabilities)):
    print(
        f"{'S'+str(k+1):>10} | {prob:>6.1%} | {cluster_sizes_sorted[k]:>7,} | "
        f"{eol_paths[idx, yr_2025_idx]:>14,.0f} | {eol_paths[idx, yr_2030_idx]:>14,.0f} | "
        f"{eol_paths[idx, yr_2035_idx]:>14,.0f}"
    )

print(f"\nSum of probabilities: {probabilities.sum():.4f}")

# ============================================================
# 5. DISAGGREGATE BY VEHICLE CLASS & CHEMISTRY
# ============================================================

print("\nDisaggregating representative scenarios by vehicle class and chemistry...")

records = []
for k, idx in enumerate(representative_indices):
    scenario_name = f"S{k+1}"
    prob = round(float(probabilities[k]), 4)
    sample_sales = sales_samples[idx]  # (n_sale_years,)

    for cls, share in class_shares.items():
        cls_sales = sample_sales * share  # (n_sale_years,)
        M_cls = conv_matrices[cls]

        # Total EOL per target year for this class
        eol_cls_total = cls_sales @ M_cls  # (n_target_years,)

        # LFP EOL: weight each cohort's contribution by its sale-year LFP share
        eol_cls_lfp = (cls_sales * lfp_shares) @ M_cls
        eol_cls_nmc = eol_cls_total - eol_cls_lfp

        for j, ty in enumerate(target_years):
            if eol_cls_lfp[j] > 0.5:
                records.append(
                    {
                        "Year": int(ty),
                        "Scenario": scenario_name,
                        "Probability": prob,
                        "Vehicle_Class": cls,
                        "Chemistry": "LFP",
                        "EOL_Volume": int(eol_cls_lfp[j]),
                    }
                )
            if eol_cls_nmc[j] > 0.5:
                records.append(
                    {
                        "Year": int(ty),
                        "Scenario": scenario_name,
                        "Probability": prob,
                        "Vehicle_Class": cls,
                        "Chemistry": "NMC",
                        "EOL_Volume": int(eol_cls_nmc[j]),
                    }
                )

smip_df = pd.DataFrame(records)

# ============================================================
# 6. VALIDATION
# ============================================================

print("\n--- Validation ---")
prob_check = smip_df.groupby("Scenario")["Probability"].first().sum()
print(
    f"  Sum of scenario probabilities: {prob_check:.4f} {'✓' if abs(prob_check - 1.0) < 0.01 else '✗'}"
)
print(f"  Total records: {len(smip_df):,}")
print(f"  Unique scenarios: {smip_df['Scenario'].nunique()}")
print(f"  Year range: {smip_df['Year'].min()}–{smip_df['Year'].max()}")
print(f"  Vehicle classes: {smip_df['Vehicle_Class'].nunique()}")
print(f"  Chemistries: {list(smip_df['Chemistry'].unique())}")

# Cross-check 2030 total vs Phase 4 base projection
eol_2030_expected = (
    smip_df[smip_df["Year"] == 2030]
    .groupby("Scenario")
    .apply(
        lambda g: (g["EOL_Volume"] * g["Probability"].iloc[0]).sum(),
        include_groups=False,
    )
    .sum()
)
print(f"  Expected 2030 total EOL (prob-weighted): {eol_2030_expected:,.0f}")

# ============================================================
# 7. SAVE OUTPUT
# ============================================================

output_file = "data/processed/model_outputs/smip_scenarios_mc_reduced.csv"
smip_df.to_csv(output_file, index=False)
print(f"\nSaved {len(smip_df):,} records to {output_file}")

# Also save a compact scenario summary
summary_records = []
for k, (idx, prob) in enumerate(zip(representative_indices, probabilities)):
    row = {
        "Scenario": f"S{k+1}",
        "Probability": f"{prob:.1%}",
        "Cluster_Size": int(cluster_sizes_sorted[k]),
    }
    for yr_idx, yr in enumerate(target_years):
        if yr in [2025, 2028, 2030, 2033, 2035]:
            row[f"EOL_{yr}"] = int(eol_paths[idx, yr_idx])
    summary_records.append(row)
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv("outputs/phase5_scenarios/mc_scenario_summary.csv", index=False)
print("Saved scenario summary to outputs/phase5_scenarios/mc_scenario_summary.csv")

# ============================================================
# 8. VISUALIZATIONS
# ============================================================

# 8a. Silhouette & Elbow Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Scenario Reduction: Optimal K Selection", fontsize=14, fontweight="bold")

ax1.plot(
    list(K_range), silhouette_scores, "o-", color="#FF6B6B", linewidth=2, markersize=8
)
ax1.axvline(
    x=K_OPTIMAL,
    color="gray",
    linestyle="--",
    alpha=0.7,
    label=f"Selected K={K_OPTIMAL}",
)
ax1.set_xlabel("Number of Scenarios (K)", fontsize=11)
ax1.set_ylabel("Silhouette Score", fontsize=11)
ax1.set_title("Silhouette Analysis")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.set_xticks(list(K_range))

ax2.plot(list(K_range), inertias, "s-", color="#4ECDC4", linewidth=2, markersize=8)
ax2.axvline(
    x=K_OPTIMAL,
    color="gray",
    linestyle="--",
    alpha=0.7,
    label=f"Selected K={K_OPTIMAL}",
)
ax2.set_xlabel("Number of Scenarios (K)", fontsize=11)
ax2.set_ylabel("Inertia (WCSS)", fontsize=11)
ax2.set_title("Elbow Method")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.set_xticks(list(K_range))

plt.tight_layout()
plt.savefig("outputs/phase5_scenarios/scenario_k_selection.png", dpi=300)
os.system("cp outputs/phase5_scenarios/scenario_k_selection.png docs/")
plt.close()

# 8b. Representative Scenarios Overlaid on MC Cloud
fig, ax = plt.subplots(figsize=(14, 7))

# Background: 300 random MC paths
rng = np.random.RandomState(0)
subset = rng.choice(N_SAMPLES, 300, replace=False)
for s_idx in subset:
    ax.plot(target_years, eol_paths[s_idx], color="lightgray", alpha=0.1, linewidth=0.5)

# Percentile envelope
p5 = np.percentile(eol_paths, 5, axis=0)
p95 = np.percentile(eol_paths, 95, axis=0)
ax.fill_between(
    target_years, p5, p95, alpha=0.08, color="steelblue", label="5th–95th percentile"
)

# Highlight representative scenarios
colors = plt.cm.tab10(np.linspace(0, 0.9, K_OPTIMAL))
for k, idx in enumerate(representative_indices):
    ax.plot(
        target_years,
        eol_paths[idx],
        color=colors[k],
        linewidth=2.5,
        label=f"S{k+1} (p={probabilities[k]:.0%})",
        zorder=5,
    )
    # Annotate 2030 value
    ax.plot(
        2030, eol_paths[idx, yr_2030_idx], "o", color=colors[k], markersize=8, zorder=6
    )

ax.set_title(
    f"Reduced Scenario Set: {K_OPTIMAL} Representatives from {N_SAMPLES:,} MC Paths",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Total EOL Batteries", fontsize=12)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(True, linestyle="--", alpha=0.3)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M"))
ax.set_xlim(2020, 2035)

plt.tight_layout()
plt.savefig("outputs/phase5_scenarios/representative_scenarios_overlay.png", dpi=300)
os.system("cp outputs/phase5_scenarios/representative_scenarios_overlay.png docs/")
plt.close()

print("\nGenerated visualizations:")
print("  outputs/phase5_scenarios/scenario_k_selection.png")
print("  outputs/phase5_scenarios/representative_scenarios_overlay.png")
print(f"\n" + "=" * 60)
print(f"✓ Phase 5c complete.")
print(f"  {K_OPTIMAL} representative scenarios saved to {output_file}")
print(f"  Ready to feed into downstream SMIP model.")
print("=" * 60)
