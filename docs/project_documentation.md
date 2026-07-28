# AI-Enabled Reverse Supply Chain Framework for EV Battery Recycling in India

This document tracks the technical implementation, dataset details, and methodologies for the feedstock forecasting pipeline of the EV Battery Recycling project (a two-stage stochastic mixed-integer programming model).

---

## Phase 0: Environment & Project Setup

**Objective:** Establish a robust data science environment and folder structure for the pipeline.

**Technical Procedures:**
- Created a standard project hierarchy:
  - `data/raw/vahan/`: Raw EV registration Excel files.
  - `data/raw/exogenous/`: Raw datasets for macroeconomic factors and battery data.
  - `data/processed/`: Cleaned and merged CSV files ready for modeling.
  - `scripts/`: Python scripts for data processing and EDA.
  - `outputs/`: Generated plots and visualizations.
- **Python Stack:** Initialized a virtual environment (`venv`) and installed core dependencies via `requirements.txt` (`pandas`, `numpy`, `statsmodels`, `prophet`, `lifelines`, `scikit-learn`, `lightgbm`, `openpyxl`).

---

## Phase 1: Data Collection

**Objective:** Gather real-world historical EV sales data and exogenous variables required for time-series forecasting.

### 1.1 VAHAN EV Registration Data
- **Source:** Ministry of Road Transport & Highways (MoRTH) VAHAN Dashboard.
- **Data Range:** January 2020 – July 2026.
- **Structure:** 7 Excel files (`reportTable.xlsx` to `reportTable (6).xlsx`), each containing month-wise, state-wise EV registrations.

### 1.2 IEA Global EV Data
- **Source:** International Energy Agency (IEA) Global EV Data Explorer.
- **File:** `EV data by country 2026.xlsx`.
- **Procedure:** Filtered the global dataset to extract India-specific metrics using a Python script.
- **Outputs generated:**
  - `iea_india_ev_sales.csv`: Historical EV sales data for India.
  - `iea_india_battery_deployment.csv`: Historical battery deployment (in GWh) for India.

### 1.3 Exogenous Policy Regressors
- **Context:** Sudden spikes in EV adoption are highly correlated with government subsidies. 
- **Procedure:** Generated a time-series CSV (`policy_regressors.csv`) based on official Ministry of Heavy Industries (MHI) timelines.
  - **FAME-II Active:** Boolean flag for April 2019 – March 2024.
  - **PM E-DRIVE Active:** Boolean flag for October 2024 – March 2026.
  - **State Subsidies Active:** Boolean flag (proxy from July 2021) for major state-level EV policies (Delhi, Maharashtra, etc.).

### 1.4 Battery Chemistry Mix Assumptions
- **Context:** The ratio of NMC (Nickel Manganese Cobalt) to LFP (Lithium Iron Phosphate) drastically impacts recycling economics. The IEA dataset does not split chemistry by country.
- **Procedure:** Generated `chemistry_mix.csv` based on Indian industry literature (e.g., JMK Research, BNEF). Since India's market is dominated by 2W/3W vehicles, the assumption encodes a transition from 30% LFP in 2020 to ~80% LFP by 2026+.

---

## Phase 2: Exploratory Data Analysis (EDA)

**Objective:** Clean the raw VAHAN data, visualize adoption trends, and test for time-series stationarity to inform the Phase 3 forecasting model architecture.

### 2.1 Raw Data Processing
- **Script:** `scripts/process_vahan_data.py`
- **Procedure:** 
  - Iterated over the 7 VAHAN Excel files.
  - Extracted the Year from the header, State from the index, and mapped the monthly columns.
  - Melted the data into a long-format DataFrame and cleaned state strings and numeric formats (removed commas, handled NaNs).
- **Output:** `data/processed/vahan_registrations.csv` (2,844 records).

### 2.2 Visualizations & Seasonal Decomposition
- **Script:** `scripts/eda_vahan.py`
- **Procedure:** Aggregated the dataset nationally to plot total monthly registrations. Used `statsmodels.tsa.seasonal_decompose` to break the series into Trend, Seasonal, and Residual components.
- **Observation:** The data exhibits a massive, non-linear upward trend and strong seasonality (spikes near financial year-end / festive seasons).

### 2.3 Stationarity Checks (Augmented Dickey-Fuller Test)
- **Procedure:** Ran the ADF test on the national aggregated registrations to determine integration order ($d$).
- **Results:**
  - **Level Series (Original):** p-value = 0.9948 (Non-stationary)
  - **First-Differenced Series:** p-value = 0.0000 (Stationary)
- **Conclusion:** The time series must be differenced once to achieve stationarity. This confirms that an integrated model like **SARIMAX (with $d=1$)** or **Prophet** is the mathematically appropriate choice for Phase 3.
