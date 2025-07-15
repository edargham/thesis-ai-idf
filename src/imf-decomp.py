import os
import pandas as pd
import numpy as np

# Read the CSV file
file_path = os.path.join(
    os.path.dirname(__file__), "data", "gpm-bey-30mns.csv"
)
df = pd.read_csv(file_path, parse_dates=["date"])
df["value"] = df["value"] #/ 0.5


# Function to apply IMF decomposition for intensities
def apply_imf_decomp_intensity(intensity_T, t, T, n=(1/3)):
    """
    Apply IMF decomposition for intensities.

    Parameters:
    intensity_T (float): Rainfall intensity for duration T in mm/hr
    t (float): Target duration in minutes
    T (float): Original duration in minutes
    n (float): Empirical parameter for Bell's ratio

    Returns:
    float: Rainfall intensity for duration t in mm/hr
    """
    if intensity_T == 0:
        return 0

    # For intensities, the relationship is I_t / I_T = (T/t)^(1-n)
    ratio = (T / t) ** (n)
    intensity_t = intensity_T * ratio

    return intensity_t


# Create DataFrames for each target duration
df_5min = pd.DataFrame(columns=["date", "value"])
df_10min = pd.DataFrame(columns=["date", "value"])
df_15min = pd.DataFrame(columns=["date", "value"])

# For each 30-minute interval, create higher resolution data
# Calculate size of output arrays
n_rows = len(df)
n_5min = n_rows * 6
n_10min = n_rows * 3
n_15min = n_rows * 2

# Pre-allocate arrays
dates_5min = np.zeros(n_5min, dtype="datetime64[ns]")
values_5min = np.zeros(n_5min)
dates_10min = np.zeros(n_10min, dtype="datetime64[ns]")
values_10min = np.zeros(n_10min)
dates_15min = np.zeros(n_15min, dtype="datetime64[ns]")
values_15min = np.zeros(n_15min)

# Extract dates and intensities as arrays
dates = df["date"].values
intensities = df["value"].values

# Calculate all intensities at once using IMF decomposition
intensity_5min = np.vectorize(lambda x: apply_imf_decomp_intensity(x, 5, 30))(
    intensities
)
intensity_10min = np.vectorize(lambda x: apply_imf_decomp_intensity(x, 10, 30))(
    intensities
)
intensity_15min = np.vectorize(lambda x: apply_imf_decomp_intensity(x, 15, 30))(
    intensities
)

# Fill arrays
for i in range(6):
    idx = np.arange(i, n_5min, 6)
    dates_5min[idx] = dates + np.timedelta64(i * 5, "m")
    values_5min[idx] = intensity_5min

for i in range(3):
    idx = np.arange(i, n_10min, 3)
    dates_10min[idx] = dates + np.timedelta64(i * 10, "m")
    values_10min[idx] = intensity_10min

for i in range(2):
    idx = np.arange(i, n_15min, 2)
    dates_15min[idx] = dates + np.timedelta64(i * 15, "m")
    values_15min[idx] = intensity_15min

# Create DataFrames
df_5min = pd.DataFrame({"date": dates_5min, "value": values_5min})
df_10min = pd.DataFrame({"date": dates_10min, "value": values_10min})
df_15min = pd.DataFrame({"date": dates_15min, "value": values_15min})

# Print the max date for each dataframe
print(f"Original 30min data max date: {df['date'].max()}")
print(f"5min data max date: {df_5min['date'].max()}")
print(f"10min data max date: {df_10min['date'].max()}")
print(f"15min data max date: {df_15min['date'].max()}")

# Save the results
df_5min.to_csv("./data/gpm-bey-5mns.csv", index=False)
df_10min.to_csv("./data/gpm-bey-10mns.csv", index=False)
df_15min.to_csv("./data/gpm-bey-15mns.csv", index=False)

print("Disaggregation complete. Files saved.")
