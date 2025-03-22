import pandas as pd
import numpy as np
import os

# Folder path
folder_path = 'data'

# List of CSV files to read
csv_files = [
  ('40100-refined.csv', 'meteostat'),
  # ('beirut-lst-day-rhia.csv', 'lst-day'),
  # ('beirut-lst-night-rhia.csv', 'lst-night'),
  # ('beirut-daily-temps-max.csv', 'temperature-max'),
  # ('beirut-daily-temps-min.csv', 'temperature-min'),
  # ('beirut-daily-temps-mean.csv', 'temperature-mean'),
  # ('beirut-ndvi-rhia-daily.csv', 'ndvi'),
  # ('beirut-ndwi-rhia-daily.csv', 'ndwi'),
  # ('soil-m.csv', 'soil-moisture'),
  # ('soil-t.csv', 'soil-temperature'),
  #('gpm-bey.csv', 'gpm-precipitation'),
  ('fldas-evap.csv', 'evapotranspiration'),
  ('fldas-runoff.csv', 'runoff'),
  # ('airs-bey.csv', 'airs-precipitation'),
]

csv_files_shift_date = [
  ('fldas-radio-temp-max.csv', 'radio-temp-min', 0),
  ('fldas-sat.csv', 'sat', -1),
  ('fldas-soil-moist.csv', 'soil-moist', -1),
  ('fldas-soil-temp.csv', 'soil-temp', 0),
]

# Initialize a list to store dataframes
dfs = []

# Read each CSV file
for file in csv_files:
  file_path = os.path.join(folder_path, file[0])
  df = pd.read_csv(file_path)
  
  # Convert date column to datetime
  if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    
    # Get a prefix from the filename (without extension)
    prefix = file[1]
    
    # Rename all columns except 'date' to include the prefix
    df.columns = [col if col == 'date' else f"{prefix}_{col}" for col in df.columns]
  else:
    print(f"Warning: {file} does not have a 'date' column")
    continue
  
  dfs.append(df)
  print(f"Read {file}: {df.shape[0]} rows, {df.shape[1]} columns")


# Read each CSV file with date shift
for file in csv_files_shift_date:
  file_path = os.path.join(folder_path, file[0])
  df = pd.read_csv(file_path)
  # df.drop(columns=['system:index', '.geo'], inplace=True)
  
  # Convert date column to datetime
  if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    
    # Get a prefix from the filename (without extension)
    prefix = file[1]
    
    # Rename all columns except 'date' to include the prefix
    df.columns = [col if col == 'date' else f"{prefix}_{col}" for col in df.columns]
    
    # Shift the date column by a number of days
    shift_days = file[2]
    df['date'] = df['date'] + pd.DateOffset(days=shift_days)
  else:
    print(f"Warning: {file} does not have a 'date' column")
    continue
  
  dfs.append(df)
  print(f"Read {file}: {df.shape[0]} rows, {df.shape[1]} columns")

# Merge all dataframes on the date column
if dfs:
  merged_df = dfs[0]
  for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on='date', how='outer')
    print(merged_df.head())
  
  # Sort by date
  merged_df = merged_df.sort_values('date')

  # Fill missing values with NaN
  merged_df = merged_df.dropna()
 
  # Drop 50% of the rows where the target variable is zero
  meteostat_value = 'meteostat_value'
  zero_rows = merged_df[merged_df[meteostat_value] == 0].index
  drop_indices = np.random.choice(zero_rows, int(len(zero_rows) * 0.0), replace=False)
  merged_df = merged_df.drop(drop_indices)

  # Save the merged dataframe
  merged_df.to_csv('merged_data.csv', index=False)
  
  print(f"Final merged dataframe shape: {merged_df.shape}")
  print(f"Number of unique dates: {merged_df['date'].nunique()}")
else:
  print("No dataframes to merge")