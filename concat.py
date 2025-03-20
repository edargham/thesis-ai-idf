import pandas as pd

# Load the CSV files
landsat7_df = pd.read_csv('data/beirut-landsat7-ndwi.csv')
landsat8_df = pd.read_csv('data/beirut-landsat8-ndwi.csv')

# Concatenate the dataframes
combined_df = pd.concat([landsat7_df, landsat8_df])

# Sort by date in ascending order
combined_df = combined_df.sort_values(by='date')

# Save the combined dataframe to a new CSV file
combined_df.to_csv('data/beirut-landsat-combined-ndwi.csv', index=False)