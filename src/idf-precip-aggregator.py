import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_path_5mns = os.path.join(
        os.path.dirname(__file__), "data", "gpm-bey-5mns.csv"
    )
    data_path_10mns = os.path.join(
        os.path.dirname(__file__), "data", "gpm-bey-10mns.csv"
    )
    data_path_15mns = os.path.join(
        os.path.dirname(__file__), "data", "gpm-bey-15mns.csv"
    )
    data_path_30mns = os.path.join(
        os.path.dirname(__file__), "data", "gpm-bey-30mns.csv"
    )
    data_path_1h = os.path.join(
        os.path.dirname(__file__), "data", "gpm-gsmap-bey-hourly.csv"
    )
    data_path_3h = os.path.join(os.path.dirname(__file__), "data", "trmm-bey-3hrs.csv")
    data_path_24h = os.path.join(os.path.dirname(__file__), "data", "gpm-bey-daily.csv")

    df_5mns = pd.read_csv(data_path_5mns)
    df_10mns = pd.read_csv(data_path_10mns)
    df_15mns = pd.read_csv(data_path_15mns)
    df_30mns = pd.read_csv(data_path_30mns)
    df_30mns["value"] = df_30mns["value"] / 0.5

    # df_hr is the the sum divided by 2 for every hour in the 30mns data
    df_1h = df_30mns.copy()
    df_1h["hour"] = pd.to_datetime(df_1h["date"].values).ceil("h")
    df_1h = df_1h.groupby("hour")["value"].mean().reset_index()
    df_1h.rename(columns={"hour": "date"}, inplace=True)
    # df_1h = pd.read_csv(data_path_1h)

    # Get 3h value from 1h value by resampling with 3-hour intervals (window=3, stride=3)
    temp_df = df_1h.copy()
    temp_df.set_index("date", inplace=True)
    temp_df = temp_df.resample("3h", label="right", closed="right").mean()
    df_3h = temp_df.reset_index()
    # df_3h = pd.read_csv(data_path_3h)

    df_24h = pd.read_csv(data_path_24h)

    # Convert the time columns to datetime
    df_5mns["date"] = pd.to_datetime(df_5mns["date"])
    df_10mns["date"] = pd.to_datetime(df_10mns["date"])
    df_15mns["date"] = pd.to_datetime(df_15mns["date"])
    df_30mns["date"] = pd.to_datetime(df_30mns["date"])
    df_1h["date"] = pd.to_datetime(df_1h["date"])
    df_3h["date"] = pd.to_datetime(df_3h["date"])
    df_24h["date"] = pd.to_datetime(df_24h["date"])

    # Set the time columns as the index
    df_5mns.set_index("date", inplace=True)
    df_10mns.set_index("date", inplace=True)
    df_15mns.set_index("date", inplace=True)
    df_30mns.set_index("date", inplace=True)
    df_1h.set_index("date", inplace=True)
    df_3h.set_index("date", inplace=True)
    df_24h.set_index("date", inplace=True)

    # Merge the dataframes on the index
    df = pd.concat([df_5mns, df_10mns, df_15mns, df_30mns, df_1h, df_3h, df_24h], axis=1)

    df.columns = ["5mns", "10mns", "15mns", "30mns", "1h", "3h", "24h"]

    # Drop all the rows with dates after 2019-12-31 23:59:59
    df = df[df.index < "2019-12-31 23:59:59"]

    # Fill NaN values with 0
    df.fillna(0, inplace=True)

    # Save the new dataframe to a new XSLX file.
    output_path = os.path.join(
        os.path.dirname(__file__), "results", "bey-aggregated-final"
    )
    # df.to_excel(f"{output_path}.xlsx", index=True)
    df.to_csv(f"{output_path}.csv", index=True)
    print(f"Aggregated data saved to {output_path}")


# Find annual maximums and convert to intensities (mm/hr)
df["year"] = df.index.year

df_intensity = df.copy()

# Convert rainfall depth to intensity (mm/hr)
df_intensity["5mns"] = df["5mns"]  # 5 mins = 1/12 hours
df_intensity["10mns"] = df["10mns"]  # 10 mins = 1/6 hours
df_intensity["15mns"] = df["15mns"]  # 15 mins = 1/4 hours
df_intensity["30mns"] = df["30mns"]  # 30 mins = 0.5 hours
df_intensity["1h"] = df["1h"]  # / 1  # 1 hour (already in mm/hr)
df_intensity["3h"] = df["3h"]  # / 3  # 3 hours
df_intensity["24h"] = df["24h"] / 24  # 24 hours

# Reorder columns to put 'year' first
columns = ['year'] + [col for col in df_intensity.columns if col != 'year']
df_intensity = df_intensity[columns]
df_intensity.to_csv(os.path.join(os.path.dirname(__file__), "results", "historical_intensity.csv"), index=True)

# Store intensity values in separate columns for IDF analysis
df_intensity["5mns_intensity"] = df_intensity["5mns"].copy()
df_intensity["10mns_intensity"] = df_intensity["10mns"].copy()
df_intensity["15mns_intensity"] = df_intensity["15mns"].copy()
df_intensity["30mns_intensity"] = df_intensity["30mns"].copy()
df_intensity["1h_intensity"] = df_intensity["1h"].copy()
df_intensity["3h_intensity"] = df_intensity["3h"].copy()
df_intensity["24h_intensity"] = df_intensity["24h"].copy()

# Get annual maximum intensities first
annual_max_intensity = (
    df_intensity[["year", "5mns", "10mns", "15mns", "30mns", "1h", "3h", "24h"]].groupby("year").max()
)

output_path = os.path.join(
    os.path.dirname(__file__), "results", "annual_max_intensity"
)
annual_max_intensity.to_csv(f"{output_path}.csv", index=True)

# Define return periods and corresponding probabilities
return_periods = np.array([2, 5, 10, 25, 50, 100])
probabilities = 1 - 1 / return_periods

# Dictionary to store Gumbel parameters for each duration
gumbel_params = {}
emperical_params = {}
durations = ["5mns", "10mns", "15mns", "30mns", "1h", "3h", "24h"]
duration_hours = [5, 10, 15, 30, 60, 180, 1440]

# Fit Gumbel distribution and calculate intensities
intensities_gum = np.zeros((len(return_periods), len(durations)))
intensities_wbl = np.zeros((len(return_periods), len(durations)))

for j, dur in enumerate(durations):
    loc, scale = stats.gumbel_r.fit(annual_max_intensity[dur])
    gumbel_params[dur] = (loc, scale)  # Now storing location and scale
    for i, prob in enumerate(probabilities):
        intensities_gum[i, j] = stats.gumbel_r.ppf(q=prob, loc=loc, scale=scale)


# Plot the IDF curves similar to ann.py
plt.figure(figsize=(14, 10))

# Define colors for different return periods
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

plt.tight_layout()
output_dir = os.path.join(os.path.dirname(__file__), "figures")
plt.savefig(os.path.join(output_dir, "idf_gumbel.png"), dpi=300)

# Original IDF curve plot (simplified version)
plt.figure(figsize=(10, 6))
for i, rp in enumerate(return_periods):
    plt.plot(duration_hours, intensities_gum[i], marker="o", label=f"{rp}-year")

plt.xlabel("Duration (minutes)")
plt.ylabel("Rainfall Intensity (mm/hr)")
plt.title("Intensity-Duration-Frequency (IDF) Curves\nGumbel Distribution Method")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_dir = os.path.join(os.path.dirname(__file__), "figures")
plt.savefig(os.path.join(output_dir, "idf_curves_trad.png"), dpi=300)

# Save the IDF data
idf_data = pd.DataFrame(
    intensities_gum, index=return_periods, columns=[f"{d} mins" for d in duration_hours]
)
idf_data.index.name = "Return Period (years)"

output_dir = os.path.join(os.path.dirname(__file__), "results")
idf_data.to_csv(os.path.join(output_dir, "idf_data.csv"))

print("Gumbel Distribution Parameters:")
for dur in durations:
    shape, loc = gumbel_params[dur]
    print(f"{dur}: shape = {shape:.4f}, location = {loc:.4f}")
