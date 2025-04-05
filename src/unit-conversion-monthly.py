import pandas as pd
from calendar import monthrange

#!/usr/bin/env python3
"""
This script converts precipitation values from mm/hr to mm.
For monthly data, the conversion is made by multiplying each value by the
number of hours in its respective month (computed from the date).
It assumes the input file is a CSV with at least a “date” column (in a parseable format)
and a “value” column holding mm/hr.
The output is written as a new CSV file.
"""


def convert_mmhr_to_mm(input_file: str, output_file: str) -> None:
  # Read the CSV file. Adjust the delimiter or file format as needed.
  df = pd.read_csv(input_file)
  
  # Ensure the "date" column is in datetime format
  df["date"] = pd.to_datetime(df["date"])
  
  # Compute the number of hours in the month for each date
  # monthrange(year, month) returns (first_weekday, num_days)
  df["hours_in_month"] = df["date"].apply(lambda d: monthrange(d.year, d.month)[1] * 24)
  
  # Convert mm/hr to mm: multiply value by number of hours in the month
  df["value"] = df["value"] * df["hours_in_month"]
  
  # Write the results to a new CSV file; drop the helper column if desired.
  df.drop(columns=["hours_in_month"], inplace=True)
  df["date"] = df["date"].dt.strftime("%Y-%m-%d")
  
  df.to_csv(output_file, index=False)
  print(f"Conversion complete. Output saved to {output_file}")

if __name__ == "__main__":
  input_path = "./data/beirut-accumulated-10k-v7-monthly.csv"
  output_path = "./data/beirut-accumulated-10k-v7-monthly_converted.csv"
  convert_mmhr_to_mm(input_path, output_path)