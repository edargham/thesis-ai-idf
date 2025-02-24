import pandas as pd
import numpy as np
from math import sqrt

# Load CSV files from the folder "data"
accumulated_path = "data/beirut-daily-precipitation.csv"
meteostat_path = "data/beirut-meteostat.csv"

df_accumulated = pd.read_csv(accumulated_path)
df_meteostat = pd.read_csv(meteostat_path)

# Replace null values with zeros in "value" column for meteostat
df_meteostat['value'] = df_meteostat['value'].fillna(0)

# Select rows from both datasets where the date column matches and both "value" columns are non-zero
merged_df = pd.merge(df_accumulated, df_meteostat, on='date', suffixes=('_acc', '_met'))
merged_df = merged_df[(merged_df['value_met'] != 0)]
df_accumulated = merged_df[['date', 'value_acc']].rename(columns={'value_acc': 'value'})
df_meteostat = merged_df[['date', 'value_met']].rename(columns={'value_met': 'value'})

# To compare using RMSE, we need to ensure the two arrays align.
# For simplicity, we'll assume they align by index. If necessary, you can merge them on a common key.

# Align indices by taking the intersection of indices if lengths differ
common_index = df_accumulated.index.intersection(df_meteostat.index)
values_accumulated = merged_df[['date', 'value_acc']]
values_accumulated['value'] = values_accumulated['value_acc']
values_accumulated = values_accumulated.drop(columns='value_acc')
print(values_accumulated)
values_accumulated.to_csv('data/comp-accumulated.csv', index=False)

values_meteostat = merged_df[['date', 'value_met']]
values_meteostat['value'] = values_meteostat['value_met']
values_meteostat = values_meteostat.drop(columns='value_met')
print(values_meteostat)
values_meteostat.to_csv('data/comp-meteostat.csv', index=False)

# Compute RMSE
mse = np.mean((values_accumulated['value'] - values_meteostat['value']) ** 2)
rmse = sqrt(mse)
print("MSE:", mse)
print("RMSE:", rmse)

# Compare the sum of the values
sum_accumulated = values_accumulated['value'].sum()
sum_meteostat = values_meteostat['value'].sum()
print("Sum of accumulated values:", sum_accumulated)
print("Sum of meteostat values:", sum_meteostat)
print("Difference in sums:", abs(sum_accumulated - sum_meteostat))

# Compute R2 score
mean_accumulated = values_accumulated['value'].mean()
mean_meteostat = values_meteostat['value'].mean()
ss_tot = np.sum((values_accumulated['value'] - mean_accumulated) ** 2)
ss_res = np.sum(abs(values_accumulated['value'] - values_meteostat['value']) ** 2)
r2 = 1 - (ss_res / ss_tot)
print("R2 score:", r2)

# Compute correlation coefficient
correlation = values_accumulated['value'].corr(values_meteostat['value'])
print("Correlation coefficient:", correlation)

# Compute MAE
mae = np.mean(np.abs(values_accumulated['value'] - values_meteostat['value']))
print("MAE:", mae)
