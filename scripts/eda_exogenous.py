import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

# 1. EDA for Chemistry Mix
chem_df = pd.read_csv('data/processed/chemistry_mix.csv')
chem_df['Date'] = pd.to_datetime(chem_df['Date'])

plt.figure(figsize=(10, 6))
plt.plot(chem_df['Date'], chem_df['LFP_Share'] * 100, label='LFP Share (%)', color='orange', linewidth=2)
plt.plot(chem_df['Date'], chem_df['NMC_Share'] * 100, label='NMC Share (%)', color='blue', linewidth=2)
plt.title('Assumed Battery Chemistry Mix (India)')
plt.ylabel('Market Share (%)')
plt.xlabel('Year')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('outputs/chemistry_mix_trend.png', dpi=300)
plt.close()

# 2. EDA for Policy Regressors
pol_df = pd.read_csv('data/processed/policy_regressors.csv')
pol_df['Date'] = pd.to_datetime(pol_df['Date'])

plt.figure(figsize=(10, 6))
# We plot them with slight offsets so they don't overlap entirely if active simultaneously
plt.fill_between(pol_df['Date'], 0, pol_df['FAME_II_Active'], label='FAME-II Active', color='green', alpha=0.3)
plt.fill_between(pol_df['Date'], 0, pol_df['PM_EDRIVE_Active'], label='PM E-DRIVE Active', color='red', alpha=0.3)
plt.fill_between(pol_df['Date'], 0, pol_df['State_Subsidy_Active'], label='State Subsidies Active', color='purple', alpha=0.1)

plt.title('EV Policy Timelines (Regressors)')
plt.yticks([0, 1], ['Inactive', 'Active'])
plt.xlabel('Year')
plt.legend(loc='center left')
plt.tight_layout()
plt.savefig('outputs/policy_timeline.png', dpi=300)
plt.close()

print("Generated EDA plots for exogenous variables in outputs/ folder.")
