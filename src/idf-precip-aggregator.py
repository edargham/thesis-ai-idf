import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_path_30mns = os.path.join(
        os.path.dirname(__file__), "data", "gpm-nasa-30mns.csv"
    )
    # data_path_1h = os.path.join(
    #     os.path.dirname(__file__),
    #     'data',
    #     'gpm-gsmap-bey-hourly.csv'
    # )
    data_path_3h = os.path.join(os.path.dirname(__file__), "data", "trmm-bey-3hrs.csv")
    data_path_24h = os.path.join(os.path.dirname(__file__), "data", "gpm-bey-daily.csv")

    # Read the data
    df_30mns = pd.read_csv(data_path_30mns)

    # df_hr is the the sum divided by 2 for every hour in the 30mns data
    df_1h = df_30mns.copy()
    df_1h["hour"] = pd.to_datetime(df_1h["date"].values).ceil("h")
    df_1h = df_1h.groupby("hour")["value"].mean().reset_index()
    df_1h.rename(columns={"hour": "date"}, inplace=True)

    # Get 3h value from 1h value by resampling with 3-hour intervals (window=3, stride=3)
    temp_df = df_1h.copy()
    temp_df.set_index("date", inplace=True)
    temp_df = temp_df.resample("3h", label="right", closed="right").sum()
    df_3h = temp_df.reset_index()

    # df_3h = pd.read_csv(data_path_3h)
    df_24h = pd.read_csv(data_path_24h)

    # Convert the time columns to datetime
    df_30mns["date"] = pd.to_datetime(df_30mns["date"])
    df_1h["date"] = pd.to_datetime(df_1h["date"])
    df_3h["date"] = pd.to_datetime(df_3h["date"])
    df_24h["date"] = pd.to_datetime(df_24h["date"])

    # Set the time columns as the index
    df_30mns.set_index("date", inplace=True)
    df_1h.set_index("date", inplace=True)
    df_3h.set_index("date", inplace=True)
    df_24h.set_index("date", inplace=True)

    # Merge the dataframes on the index
    df = pd.concat([df_30mns, df_1h, df_3h, df_24h], axis=1)

    df.columns = ["30mns", "1h", "3h", "24h"]

    # Drop all the rows with dates after 2019-12-31 23:59:59
    df = df[df.index < "2019-12-31 23:59:59"]

    # Drop all the rows with dates before 1998-01-01 00:00:00
    # df = df[df.index >= '1998-01-01 00:00:00']

    df["30mns"] = df["30mns"] * 0.5

    # Fill NaN values with 0
    df.fillna(0, inplace=True)

    # Save the new dataframe to a new XSLX file.
    output_path = os.path.join(
        os.path.dirname(__file__), "data", "bey-aggregated-final"
    )
    df.to_excel(f"{output_path}.xlsx", index=True)
    df.to_csv(f"{output_path}.csv", index=True)
    print(f"Aggregated data saved to {output_path}")


# Find annual maximums and convert to intensities (mm/hr)
df["year"] = df.index.year

# Calculate rainfall intensities in mm/hr
# Define catchment area (10 km²)
catchment_area_km2 = 10
catchment_area_m2 = catchment_area_km2 * 1000  # Convert to m²

df_intensity = df.copy()
# Convert rainfall depth to intensity (mm/hr)
df_intensity["30mns"] = df["30mns"] / 0.5  # 30 mins = 0.5 hours
df_intensity["1h"] = df["1h"] / 1  # 1 hour (already in mm/hr)
df_intensity["3h"] = df["3h"] / 3  # 3 hours
df_intensity["24h"] = df["24h"] / 24  # 24 hours

# Calculate precipitation with catchment area consideration (mm/hr)
df_intensity["30mns_catchment"] = df_intensity["30mns"]  # mm/hr stays mm/hr
df_intensity["1h_catchment"] = df_intensity["1h"] 
df_intensity["3h_catchment"] = df_intensity["3h"]
df_intensity["24h_catchment"] = df_intensity["24h"]

# Calculate volume in m³/hr for reference
df_intensity["30mns"] = df_intensity["30mns"] * catchment_area_m2 * 0.001  # m³/hr
df_intensity["1h"] = df_intensity["1h"] * catchment_area_m2 * 0.001
df_intensity["3h"] = df_intensity["3h"] * catchment_area_m2 * 0.001
df_intensity["24h"] = df_intensity["24h"] * catchment_area_m2 * 0.001

# Get annual maximum intensities
annual_max_intensity = df_intensity.groupby("year").max()

# Define return periods and corresponding probabilities
return_periods = np.array([2, 5, 10, 25, 50, 100])
probabilities = 1 - 1 / return_periods

# Dictionary to store Gumbel parameters for each duration
gumbel_params = {}
durations = ["30mns", "1h", "3h", "24h"]
duration_hours = [0.5, 1, 3, 24]

# Fit Gumbel distribution and calculate intensities
intensities = np.zeros((len(return_periods), len(durations)))
for j, dur in enumerate(durations):
    shape, loc = stats.gumbel_l.fit(annual_max_intensity[dur])
    gumbel_params[dur] = (shape, loc)  # Now storing shape, location, scale
    for i, prob in enumerate(probabilities):
        intensities[i, j] = stats.gumbel_l.ppf(prob, shape, loc)

# Create IDF curve plot
plt.figure(figsize=(10, 6))
for i, rp in enumerate(return_periods):
    plt.plot(duration_hours, intensities[i], marker="o", label=f"T = {rp} years")

# plt.xscale('log')
# plt.yscale('log')
plt.xlabel("Duration (hours)")
plt.ylabel("Intensity (mm/hr)")
plt.title("Intensity-Duration-Frequency (IDF) Curve")
plt.grid(True, which="both", ls="-")
plt.legend()

output_dir = os.path.join(os.path.dirname(__file__), "data")
plt.savefig(os.path.join(output_dir, "idf_curve.png"))
plt.show()

# Save the IDF data
idf_data = pd.DataFrame(
    intensities, index=return_periods, columns=[f"{d} hours" for d in duration_hours]
)
idf_data.index.name = "Return Period (years)"
idf_data.to_csv(os.path.join(output_dir, "idf_data.csv"))

print("Gumbel Distribution Parameters:")
for dur in durations:
    loc, scale = gumbel_params[dur]
    print(f"{dur}: location = {loc:.4f}, scale = {scale:.4f}")
