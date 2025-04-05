import numpy as np
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df_solar = pd.read_csv("modis-data/modis-angle.csv")
df_solar['date'] = pd.to_datetime(df_solar['date'])
df_solar.set_index('date', inplace=True)
print('len angle', len(df_solar))

df_650 = pd.read_csv("modis-data/modis-terra-650.csv")
df_650['date'] = pd.to_datetime(df_650['date'])
df_650.set_index('date', inplace=True)
print('len 650nm', len(df_650))

df_gpm = pd.read_csv("modis-data/gpm-bey.csv")
df_gpm['date'] = pd.to_datetime(df_gpm['date'])
df_gpm.set_index('date', inplace=True)
print('len gpm', len(df_gpm))

df_tb_d = pd.read_csv("modis-data/modis-temp-day.csv")
df_tb_d['date'] = pd.to_datetime(df_tb_d['date'])
df_tb_d.set_index('date', inplace=True)
print('len temp d', len(df_tb_d))

df_tb_n = pd.read_csv("modis-data/modis-temp-night.csv")
df_tb_n['date'] = pd.to_datetime(df_tb_n['date'])
df_tb_n.set_index('date', inplace=True)
print('len temp n', len(df_tb_n))

df_ctp_d = pd.read_csv("modis-data/modis-pressure-day.csv")
df_ctp_d['date'] = pd.to_datetime(df_ctp_d['date'])
df_ctp_d.set_index('date', inplace=True)
print('len ctp d', len(df_ctp_d))

df_ctp_n = pd.read_csv("modis-data/modis-pressure-night.csv")
df_ctp_n['date'] = pd.to_datetime(df_ctp_n['date'])
df_ctp_n.set_index('date', inplace=True)
print('len ctp', len(df_ctp_n))

df_vpr = pd.read_csv("modis-data/modis-vapor.csv")
df_vpr['date'] = pd.to_datetime(df_vpr['date'])
df_vpr.set_index('date', inplace=True)
print('len vpr', len(df_vpr))

df_thck = pd.read_csv("modis-data/modis-thk.csv")
df_thck['date'] = pd.to_datetime(df_thck['date'])
df_thck.set_index('date', inplace=True)
print('len thck', len(df_thck))

df_meteostat = pd.read_csv("modis-data/40100-refined.csv")
df_meteostat['date'] = pd.to_datetime(df_meteostat['date'])
df_meteostat.set_index('date', inplace=True)
print('len meteo', len(df_meteostat))

merged_df = pd.merge(df_solar, df_meteostat, on='date', suffixes=('_angle', '_met'))
merged_df = pd.merge(merged_df, df_tb_d, on='date', suffixes=('_met', '_tb-day'))
merged_df = pd.merge(merged_df, df_ctp_d, on='date', suffixes=('_tb-day', '_ctp-day'))
merged_df = pd.merge(merged_df, df_tb_n, on='date', suffixes=('_ctp-day', '_tb-night'))
merged_df = pd.merge(merged_df, df_ctp_n, on='date', suffixes=('_tb-night', '_ctp-night'))
merged_df = pd.merge(merged_df, df_vpr, on='date', suffixes=('_ctp-night', '_vpr'))
merged_df = pd.merge(merged_df, df_thck, on='date', suffixes=('_vpr', '_thck'))
# merged_df = pd.merge(merged_df, df_650, on='date', suffixes=('_thck', '_650'))
merged_df = pd.merge(merged_df, df_gpm, on='date', suffixes=('_thck', '_prcp'))

merged_df.loc[merged_df['vpr'].astype(np.int32) == -9999, 'vpr'] = merged_df['vpr'].mean()
merged_df = merged_df[merged_df['vpr'] >= 0.75]
merged_df['vpr'] = merged_df['vpr'] / np.cos(np.radians(merged_df['angle']))

# merged_df.loc[merged_df['b650nm'].isna(), 'b650nm'] = merged_df['b650nm'].mean()
# # merged_df['b650nm'] = merged_df['b650nm'] * np.cos(np.radians(merged_df['angle']))
# merged_df = merged_df[merged_df['b650nm'] >= 0.75]

merged_df.loc[merged_df['tb-day'].astype(np.int32) == -9999, 'tb-day'] = merged_df['tb-day'].mean()
# merged_df = merged_df[merged_df['tb-day'] <= 295]

merged_df.loc[merged_df['tb-night'].astype(np.int32) == -9999, 'tb-night'] = merged_df['tb-night'].mean()
# merged_df = merged_df[merged_df['tb-night'] <= 295]

merged_df.loc[merged_df['ctp-day'].astype(np.int32) == -9999, 'ctp-day'] = merged_df['ctp-day'].mean()

merged_df.loc[merged_df['ctp-night'].astype(np.int32) == -9999, 'ctp-night'] = merged_df['ctp-night'].mean()

merged_df.loc[merged_df['thck'].astype(np.int32) == -9999, 'thck'] = merged_df['thck'].mean()

merged_df.drop(columns=['angle', 'prcp'], inplace=True)
print(merged_df.head(10))

print(f"Total samples: {len(merged_df)}")

X = merged_df.drop(columns=['value'])
y = merged_df['value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=False)

model = XGBRegressor(
    n_estimators=111,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.65,
    colsample_bytree=0.7,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred[y_pred < 0] = 0
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

test_pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]
print("\nTest Set Correlation:")
print(f"Pearson correlation coefficient (test set): {test_pearson_corr:.4f}")

print("\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

y_pred_train = model.predict(X_train)
y_pred_train[y_pred_train < 0] = 0
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

train_pearson_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
print("\nTrain Set Correlation:")
print(f"Pearson correlation coefficient (train set): {train_pearson_corr:.4f}")

print("\nModel Performance on Training Data:")
print(f"Mean Squared Error: {mse_train}")
print(f"Root Mean Squared Error: {rmse_train}")
print(f"Mean Absolute Error: {mae_train}")
print(f"R2 Score: {r2_train}")

y_total_pred = np.concatenate((y_pred_train, y_pred))

# Total Correlation Analysis
pearson_corr = np.corrcoef(y, y_total_pred)[0, 1]

print("\nCorrelation Analysis:")
print(f"Pearson correlation coefficient: {pearson_corr:.4f}")


out_df = pd.DataFrame({
    'date': merged_df.index,
    'value': y_total_pred,
    'actual': y
})

out_df.to_csv('modis-data/predicted.csv', index=True)