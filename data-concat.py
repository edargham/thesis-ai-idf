import pandas as pd
import numpy as np
import os

# Folder path
folder_path = 'data'

# List of CSV files to read
csv_files = [
  ('40100.csv', 'meteostat'),
  ('beirut-lst-day-rhia.csv', 'lst-day'),
  ('beirut-lst-night-rhia.csv', 'lst-night'),
  ('beirut-ndvi-rhia-daily.csv', 'ndvi'),
  ('beirut-ndwi-rhia-daily.csv', 'ndwi'),
  ('gpm-bey.csv', 'gpm')
]

# csv_files_shift_date = [
#   ('beirut-landsat-combined-ndwi.csv', 'landsat-ndwi', -42),
#   ('beirut-landsat-combined-ndvi.csv', 'landsat-ndvi', -63)
# ]

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
# for file in csv_files_shift_date:
#   file_path = os.path.join(folder_path, file[0])
#   df = pd.read_csv(file_path)
#   df.drop(columns=['system:index', '.geo'], inplace=True)
  
#   # Convert date column to datetime
#   if 'date' in df.columns:
#     df['date'] = pd.to_datetime(df['date'])
    
#     # Get a prefix from the filename (without extension)
#     prefix = file[1]
    
#     # Rename all columns except 'date' to include the prefix
#     df.columns = [col if col == 'date' else f"{prefix}_{col}" for col in df.columns]
    
#     # Shift the date column by a number of days
#     shift_days = file[2]
#     df['date'] = df['date'] + pd.DateOffset(days=shift_days)
#   else:
#     print(f"Warning: {file} does not have a 'date' column")
#     continue
  
#   dfs.append(df)
#   print(f"Read {file}: {df.shape[0]} rows, {df.shape[1]} columns")

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

  
  # Save the merged dataframe
  merged_df.to_csv('merged_data.csv', index=False)
  
  print(f"Final merged dataframe shape: {merged_df.shape}")
  print(f"Number of unique dates: {merged_df['date'].nunique()}")
else:
  print("No dataframes to merge")