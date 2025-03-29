import pandas as pd

#!/usr/bin/env python3

def main():
  # Path to the CSV file with 30-minute precipitation readings (in mm/hr)
  csv_path = "./data/beirut-hourly-corrected.csv"
  
  # Read the CSV file. Ensure that the timestamp column is parsed as datetime.
  # Adjust the column names if needed.
  df = pd.read_csv(csv_path, parse_dates=['date'])
  # df['value'] = df['value'] * 3

  # Assuming each hourly value represents the precipitation accumulated during that hour,
  # the daily total in mm is simply the sum of the hourly values.
  daily_precip = df.groupby(df['date'].dt.floor("d"))["value"].sum().reset_index()
  # daily_precip_min = df.groupby(df['date'].dt.floor("d"))["value"].min().reset_index()
  # daily_precip_mean = df.groupby(df['date'].dt.floor("d"))["value"].mean().reset_index()
  
  # Save the daily accumulated precipitation to a new CSV file.
  daily_precip.to_csv("./data/beirut-daily-prcp.csv", index=False)
  # daily_precip_min.to_csv("./data/beirut-daily-temps-min.csv", index=False)
  # daily_precip_mean.to_csv("./data/beirut-daily-temps-mean.csv", index=False)
  print("Daily precipitation accumulation has been saved to 'beirut-daily-prcp.csv'.")
  # Convert hourly precipitation (mm/hr) to daily precipitation (mm) by summing the hourly values for each day.
  # Compute monthly precipitation accumulation from the daily totals.
  # daily_precip['month'] = pd.to_datetime(daily_precip['date']).dt.to_period('M').dt.to_timestamp().dt.strftime("%Y-%m-%d")
  
  # Save the monthly accumulated precipitation to a new CSV file.
  # monthly_precip = daily_precip.groupby('month')['value'].sum().reset_index()
  # monthly_precip.to_csv("./data/beirut-monthly-temps.csv", index=False)
  # print("Monthly precipitation accumulation has been saved to 'beirut-monthly-temps.csv'.")

if __name__ == "__main__":
  main()