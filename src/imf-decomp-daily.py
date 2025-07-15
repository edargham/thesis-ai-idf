import os
import pandas as pd
import numpy as np

# Read the daily CSV file
file_path = os.path.join(
    os.path.dirname(__file__), "data", "gpm-bey-daily.csv"
)
df = pd.read_csv(file_path, parse_dates=["date"])
# No conversion to intensity here - work with precipitation values directly


# Function to apply IMF decomposition for precipitation amounts
def apply_imf_decomp_precip(precip_T, t, T, n=(1/3)):
    """
    Apply IMF decomposition for precipitation amounts.

    Parameters:
    precip_T (float): Rainfall amount for duration T in mm
    t (float): Target duration in hours
    T (float): Original duration in hours
    n (float): Empirical parameter for Bell's ratio

    Returns:
    float: Rainfall amount for duration t in mm
    """
    if precip_T == 0:
        return 0

    # For precipitation amounts, the relationship is P_t / P_T = (t/T)^(1-n)
    ratio = (t / T) ** (1 - n)
    precip_t = precip_T * ratio

    return precip_t


# Create DataFrames for each target duration
df_1hr = pd.DataFrame(columns=["date", "value"])
df_90min = pd.DataFrame(columns=["date", "value"])
df_2hr = pd.DataFrame(columns=["date", "value"])
df_3hr = pd.DataFrame(columns=["date", "value"])
df_6hr = pd.DataFrame(columns=["date", "value"])
df_12hr = pd.DataFrame(columns=["date", "value"])
df_15hr = pd.DataFrame(columns=["date", "value"])  # New 15hr DataFrame
df_18hr = pd.DataFrame(columns=["date", "value"])  # New 18hr DataFrame

# For each 24-hour interval, create higher resolution data
# Calculate size of output arrays
n_rows = len(df)
n_1hr = n_rows * 24
n_90min = n_rows * 16  # 24 hours / 1.5 hours = 16 intervals
n_2hr = n_rows * 12
n_3hr = n_rows * 8
n_6hr = n_rows * 4
n_12hr = n_rows * 2
n_15hr = n_rows  # One 15hr interval per day
n_18hr = n_rows  # One 18hr interval per day

# Pre-allocate arrays
dates_1hr = np.zeros(n_1hr, dtype="datetime64[ns]")
values_1hr = np.zeros(n_1hr)
dates_90min = np.zeros(n_90min, dtype="datetime64[ns]")
values_90min = np.zeros(n_90min)
dates_2hr = np.zeros(n_2hr, dtype="datetime64[ns]")
values_2hr = np.zeros(n_2hr)
dates_3hr = np.zeros(n_3hr, dtype="datetime64[ns]")
values_3hr = np.zeros(n_3hr)
dates_6hr = np.zeros(n_6hr, dtype="datetime64[ns]")
values_6hr = np.zeros(n_6hr)
dates_12hr = np.zeros(n_12hr, dtype="datetime64[ns]")
values_12hr = np.zeros(n_12hr)
dates_15hr = np.zeros(n_15hr, dtype="datetime64[ns]")
values_15hr = np.zeros(n_15hr)
dates_18hr = np.zeros(n_18hr, dtype="datetime64[ns]")
values_18hr = np.zeros(n_18hr)

# Extract dates and precipitation values as arrays
dates = df["date"].values
precip_values = df["value"].values

# Calculate all precipitation amounts at once using IMF decomposition
precip_1hr = np.vectorize(lambda x: apply_imf_decomp_precip(x, 1, 24))(
    precip_values
)
precip_90min = np.vectorize(lambda x: apply_imf_decomp_precip(x, 1.5, 24))(
    precip_values
)
precip_2hr = np.vectorize(lambda x: apply_imf_decomp_precip(x, 2, 24))(
    precip_values
)
precip_3hr = np.vectorize(lambda x: apply_imf_decomp_precip(x, 3, 24))(
    precip_values
)
precip_6hr = np.vectorize(lambda x: apply_imf_decomp_precip(x, 6, 24))(
    precip_values
)
precip_12hr = np.vectorize(lambda x: apply_imf_decomp_precip(x, 12, 24))(
    precip_values
)
precip_15hr = np.vectorize(lambda x: apply_imf_decomp_precip(x, 15, 24))(
    precip_values
)
precip_18hr = np.vectorize(lambda x: apply_imf_decomp_precip(x, 18, 24))(
    precip_values
)

# Fill arrays for 1 hour intervals
for i in range(24):
    idx = np.arange(i, n_1hr, 24)
    dates_1hr[idx] = dates + np.timedelta64(i * 60, "m")
    values_1hr[idx] = precip_1hr

# Fill arrays for 90 minute intervals
for i in range(16):
    idx = np.arange(i, n_90min, 16)
    dates_90min[idx] = dates + np.timedelta64(i * 90, "m")
    values_90min[idx] = precip_90min

# Fill arrays for 2 hour intervals
for i in range(12):
    idx = np.arange(i, n_2hr, 12)
    dates_2hr[idx] = dates + np.timedelta64(i * 120, "m")
    values_2hr[idx] = precip_2hr

# Fill arrays for 3 hour intervals
for i in range(8):
    idx = np.arange(i, n_3hr, 8)
    dates_3hr[idx] = dates + np.timedelta64(i * 180, "m")
    values_3hr[idx] = precip_3hr

# Fill arrays for 6 hour intervals
for i in range(4):
    idx = np.arange(i, n_6hr, 4)
    dates_6hr[idx] = dates + np.timedelta64(i * 360, "m")
    values_6hr[idx] = precip_6hr

# Fill arrays for 12 hour intervals
for i in range(2):
    idx = np.arange(i, n_12hr, 2)
    dates_12hr[idx] = dates + np.timedelta64(i * 720, "m")
    values_12hr[idx] = precip_12hr

# Fill arrays for 15 hour intervals
for i in range(1):  # Just one interval per day
    dates_15hr = dates + np.timedelta64(i * 15 * 60, "m")
    values_15hr = precip_15hr

# Fill arrays for 18 hour intervals
for i in range(1):  # Just one interval per day
    dates_18hr = dates + np.timedelta64(i * 18 * 60, "m")
    values_18hr = precip_18hr

# Create DataFrames
df_1hr = pd.DataFrame({"date": dates_1hr, "value": values_1hr})
df_90min = pd.DataFrame({"date": dates_90min, "value": values_90min})
df_2hr = pd.DataFrame({"date": dates_2hr, "value": values_2hr})
df_3hr = pd.DataFrame({"date": dates_3hr, "value": values_3hr})
df_6hr = pd.DataFrame({"date": dates_6hr, "value": values_6hr})
df_12hr = pd.DataFrame({"date": dates_12hr, "value": values_12hr})
df_15hr = pd.DataFrame({"date": dates_15hr, "value": values_15hr})
df_18hr = pd.DataFrame({"date": dates_18hr, "value": values_18hr})

# Print the max date for each dataframe
print(f"Original daily data max date: {df['date'].max()}")
print(f"1hr data max date: {df_1hr['date'].max()}")
print(f"90min data max date: {df_90min['date'].max()}")
print(f"2hr data max date: {df_2hr['date'].max()}")
print(f"3hr data max date: {df_3hr['date'].max()}")
print(f"6hr data max date: {df_6hr['date'].max()}")
print(f"12hr data max date: {df_12hr['date'].max()}")
print(f"15hr data max date: {df_15hr['date'].max()}")
print(f"18hr data max date: {df_18hr['date'].max()}")

# Print some statistics
print("\nStatistics:")
print(f"Original daily data shape: {df.shape}")
print(f"1hr data shape: {df_1hr.shape}")
print(f"90min data shape: {df_90min.shape}")
print(f"2hr data shape: {df_2hr.shape}")
print(f"3hr data shape: {df_3hr.shape}")
print(f"6hr data shape: {df_6hr.shape}")
print(f"12hr data shape: {df_12hr.shape}")
print(f"15hr data shape: {df_15hr.shape}")
print(f"18hr data shape: {df_18hr.shape}")

# Save the results
df_1hr.to_csv("./data/gpm-bey-1hr.csv", index=False)
df_90min.to_csv("./data/gpm-bey-90min.csv", index=False)
df_2hr.to_csv("./data/gpm-bey-2hr.csv", index=False)
df_3hr.to_csv("./data/gpm-bey-3hr.csv", index=False)
df_6hr.to_csv("./data/gpm-bey-6hr.csv", index=False)
df_12hr.to_csv("./data/gpm-bey-12hr.csv", index=False)
df_15hr.to_csv("./data/gpm-bey-15hr.csv", index=False)
df_18hr.to_csv("./data/gpm-bey-18hr.csv", index=False)

print("\nDisaggregation complete. Files saved:")
print("- gpm-bey-1hr.csv")
print("- gpm-bey-90min.csv")
print("- gpm-bey-2hr.csv")
print("- gpm-bey-3hr.csv")
print("- gpm-bey-6hr.csv")
print("- gpm-bey-12hr.csv")
print("- gpm-bey-15hr.csv")
print("- gpm-bey-18hr.csv")
print("- gpm-bey-15hr.csv")
print("- gpm-bey-18hr.csv")
