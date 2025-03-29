import pandas as pd

#!/usr/bin/env python3

def main():
  # Path to the CSV file with 30-minute precipitation readings (in mm/hr)
  csv_path = "data/beirut-imerg-downtown-30mns-2.csv"

  
  # Read the CSV file. Ensure that the timestamp column is parsed as datetime.
  # Adjust the column names if needed.
  df = pd.read_csv(csv_path, parse_dates=['date'])
  
  # Create a new column for the hour part from the timestamp.
  df['hour'] = df['date'].dt.floor('h')

  # For each hour, calculate the mean precipitation reading out of the two half-hour periods.
  hourly_precip = df.groupby('hour')['value'].sum().reset_index()
  hourly_precip['value'] = hourly_precip['value'] / 2

  # Create a new column for the date part from the hourly timestamp.
  hourly_precip['date'] = pd.to_datetime(hourly_precip['hour']).dt.floor("d")

  # Group by date and sum the hourly precipitation (in mm) for each day.
  daily_precip = hourly_precip.groupby(hourly_precip['date'])["value"].sum().reset_index()
  # Save the daily accumulated precipitation to a new CSV file.
  daily_precip.to_csv("./data/beirut-daily-precipitation.csv", index=False)
  print("Daily precipitation accumulation has been saved to 'beirut-daily-precipitation.csv'.")

  # Save the hourly accumulated precipitation to a new CSV file.
  hourly_precip['date'] = hourly_precip['hour']
  hourly_precip = hourly_precip.drop(columns='hour')
  # Reorder the columns
  hourly_precip = hourly_precip[['date', 'value']]
  hourly_precip.to_csv("./data/beirut-hourly-precipitation.csv", index=False)
  
  # Compute monthly precipitation accumulation from the daily totals.
  daily_precip['month'] = pd.to_datetime(daily_precip['date']).dt.to_period('M').dt.to_timestamp().dt.strftime("%Y-%m-%d")
  print("Hourly precipitation accumulation has been saved to 'beirut-hourly-precipitation.csv'.")
  
  # Save the monthly accumulated precipitation to a new CSV file.
  monthly_precip = daily_precip.groupby('month')['value'].sum().reset_index()
  monthly_precip.to_csv("./data/beirut-monthly-precipitation.csv", index=False)
  print("Monthly precipitation accumulation has been saved to 'beirut-monthly-precipitation.csv'.")

if __name__ == "__main__":
  main()