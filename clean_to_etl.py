import pandas as pd
from pathlib import Path

input_path = Path("data/climate_merged_master.csv")
df = pd.read_csv(input_path)

df = df[df["country"].isin(["United States", "Canada"])]
df = df[df["year"] >= 1990]

keep_cols = ["country", "year", "co2", "co2_per_capita", "temp_anomaly"]
df = df[keep_cols].sort_values(["country", "year"])

output_path = Path("data/etl_cleaned_dataset.csv")
df.to_csv(output_path, index=False)

print("Saved cleaned ETL dataset to:", output_path)
