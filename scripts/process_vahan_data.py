import pandas as pd
import numpy as np
import glob
import os
import re

input_dir = "data/raw/vahan"
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

files = glob.glob(os.path.join(input_dir, "reportTable*.xlsx"))
print(f"Found {len(files)} files to process.")

all_data = []

for file in files:
    # Read just the first row to get the year from title
    try:
        df_head = pd.read_excel(file, header=None, nrows=1)
        title = df_head.iloc[0, 0]
        year_match = re.search(r"\((\d{4})\)", str(title))
        if not year_match:
            print(f"Could not find year in title of {file}")
            continue
        year = year_match.group(1)

        # Read the entire sheet
        df = pd.read_excel(file, header=None)

        # Month headers are in row index 3
        months_row = df.iloc[3].values

        # Valid rows are where column 0 has a valid Serial Number (S.No)
        valid_rows_mask = pd.to_numeric(df.iloc[4:, 0], errors="coerce").notna()

        states = df.iloc[4:, 1][valid_rows_mask].values

        month_data = {}
        for col_idx, month in enumerate(months_row):
            if pd.isna(month):
                continue
            month_str = str(month).strip().upper()
            if month_str in [
                "JAN",
                "FEB",
                "MAR",
                "APR",
                "MAY",
                "JUN",
                "JUL",
                "AUG",
                "SEP",
                "OCT",
                "NOV",
                "DEC",
            ]:
                vals = df.iloc[4:, col_idx][valid_rows_mask].values
                # Clean commas and convert to numeric
                vals = [
                    str(x).replace(",", "").strip() if pd.notna(x) else "0"
                    for x in vals
                ]
                vals = pd.Series(vals)
                vals = pd.to_numeric(vals, errors="coerce").fillna(0).values
                month_data[month_str] = vals

        df_year = pd.DataFrame({"State": states})
        for m, v in month_data.items():
            df_year[m] = v

        df_year["Year"] = year

        # Melt dataframe to long format
        df_melt = df_year.melt(
            id_vars=["State", "Year"], var_name="Month", value_name="Registrations"
        )
        all_data.append(df_melt)
        print(f"Processed {file} for Year {year}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

# Combine all years
final_df = pd.concat(all_data, ignore_index=True)

# Clean State names
final_df["State"] = final_df["State"].str.strip()

# Map month string to number for datetime conversion
month_map = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
final_df["Month_Num"] = final_df["Month"].map(month_map)

# Create Datetime column
final_df["Date"] = pd.to_datetime(
    final_df["Year"].astype(str) + "-" + final_df["Month_Num"].astype(str) + "-01"
)

# Sort chronologically by state
final_df = final_df.sort_values(by=["State", "Date"]).reset_index(drop=True)

# Save processed data
output_file = os.path.join(output_dir, "vahan_registrations.csv")
final_df.to_csv(output_file, index=False)

print("\n--- Data Processing Complete ---")
print(f"Total Records: {len(final_df)}")
print(f"Date Range: {final_df['Date'].min().date()} to {final_df['Date'].max().date()}")
print("\nSample Data:")
print(final_df.head(12))
