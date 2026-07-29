# AI-Enabled Reverse Supply Chain Framework for EV Battery Recycling in India

This repository contains the codebase and data for an AI-enabled, two-stage Stochastic Mixed-Integer Programming (SMIP) framework designed to optimize reverse supply chain networks for EV battery recycling in India. By leveraging Prophet time-series forecasting, Weibull survival convolutions, Monte Carlo simulation, and K-Means clustering on VAHAN registration data, the framework models the impending surge of End-of-Life (EOL) lithium-ion batteries. It then determines the optimal location, scale, and routing of formal recycling infrastructure under uncertainty, competing directly against the informal sector through economic penalty mechanisms. 

## Folder Structure
- `data/` : Contains raw datasets (VAHAN extracts, macroeconomic data) and processed outputs used by the models.
- `docs/` : Contains project methodology documentation, the main thesis paper, and exploratory data analysis reports.
- `outputs/` : Generated artifacts from the optimization models, including diagnostic plots, sensitivity CSVs, and solver logs.
- `scripts/` : The sequential Python scripts that perform the entire end-to-end data processing, forecasting, simulation, and optimization pipeline.

## Setup Instructions
1. Ensure Python 3.9+ is installed.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Reproducing the Pipeline
To reproduce the full pipeline from raw data to final Benders decomposition results, run the scripts in the following exact sequence from the root directory:

**Data Preprocessing & Forecasting (Phases 0–3):**
1. `python scripts/phase0_process_vahan_data.py`
2. `python scripts/phase1_process_exogenous.py`
3. `python scripts/phase2_eda_vahan.py`
4. `python scripts/phase2b_eda_exogenous.py`
5. `python scripts/phase3_train_forecast_model.py`

**Stochastic Simulation (Phases 4–5):**
6. `python scripts/phase4_eol_projection.py`
7. `python scripts/phase5_scenario_generation.py`
8. `python scripts/phase5b_monte_carlo.py`
9. `python scripts/phase5c_scenario_reduction.py`

**Optimization & Analysis (Phases 6–9):**
10. `python scripts/phase6_smip_model.py`
11. `python scripts/phase7_state_level_smip.py`
12. `python scripts/phase8_sensitivity_analysis.py`
13. `python scripts/phase8b_niti_validation.py`
14. `python scripts/phase9_benders_decomposition.py`

For a full technical explanation of each phase, refer to [docs/project_documentation.md](docs/project_documentation.md).
