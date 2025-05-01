import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


if __name__ == '__main__':
    data_path_30mns = os.path.join(
        os.path.dirname(__file__),
        'data',
        'gpm-bey-30mns.csv'
    )
    # data_path_1h = os.path.join(
    #     os.path.dirname(__file__),
    #     'data',
    #     'gpm-gsmap-bey-hourly.csv'
    # )
    data_path_3h = os.path.join(
        os.path.dirname(__file__),
        'data',
        'trmm-bey-3hrs.csv'
    )
    data_path_24h = os.path.join(
        os.path.dirname(__file__),
        'data',
        'gpm-bey-daily.csv'
    )

    # Read the data
    df_30mns = pd.read_csv(data_path_30mns)
    # df_1h = pd.read_csv(data_path_1h)

    # df_hr is the the sum divided by 2 for every hour in the 30mns data
    df_1h = df_30mns.copy()
    df_1h['value'] = df_1h['value'] / 2
    df_1h['date'] = pd.to_datetime(df_1h['date']).dt.ceil('h')
    df_1h.set_index('date', inplace=True)
    df_1h = df_1h.groupby(level=0).sum()

    df_3h = pd.read_csv(data_path_3h)
    df_24h = pd.read_csv(data_path_24h)

    # Convert the time columns to datetime
    df_30mns['date'] = pd.to_datetime(df_30mns['date'])
    df_3h['date'] = pd.to_datetime(df_3h['date'])
    df_24h['date'] = pd.to_datetime(df_24h['date'])

    # Set the time columns as the index
    df_30mns.set_index('date', inplace=True)
    # df_1h.set_index('date', inplace=True)
    df_3h.set_index('date', inplace=True)
    df_24h.set_index('date', inplace=True)

    # Merge the dataframes on the index
    df = pd.concat([
        df_30mns,
        df_1h,
        df_3h,
        df_24h
    ], axis=1)

    df.columns = [
        '30mns',
        '1h',
        '3h',
        '24h'
    ]

    # Drop all the rows with dates after 2019-12-31 23:59:59
    df = df[df.index < '2019-12-31 23:59:59']
    
    df['3h'] = df['3h'] * 3  # 3 hours = 3 hours
    df['30mns'] = df['30mns'] * 0.5  # 30 minutes = 0.5 hours

    # Fill NaN values with 0
    df.fillna(0, inplace=True)

    # Save the new dataframe to a new XSLX file.
    output_path = os.path.join(
        os.path.dirname(__file__),
        'data',
        'bey-aggregated-final.xlsx'
    )
    df.to_excel(output_path, index=True)
    print(f"Aggregated data saved to {output_path}")


# Find annual maximums and convert to intensities (mm/hr)
df["year"] = df.index.year

df_intensity = df.copy()

# Convert rainfall depth to intensity (mm/hr)
df_intensity["30mns"] = df["30mns"]  # 30 mins = 0.5 hours
df_intensity["1h"] = df["1h"] #/ 1  # 1 hour (already in mm/hr)
df_intensity["3h"] = df["3h"] #/ 3  # 3 hours
df_intensity["24h"] = df["24h"] / 24  # 24 hours

# Store intensity values in separate columns for IDF analysis
df_intensity["30mns_intensity"] = df_intensity["30mns"].copy()
df_intensity["1h_intensity"] = df_intensity["1h"].copy()
df_intensity["3h_intensity"] = df_intensity["3h"].copy()
df_intensity["24h_intensity"] = df_intensity["24h"].copy()

# Get annual maximum intensities first
annual_max_intensity = df_intensity[["year", "30mns", "1h", "3h", "24h"]].groupby("year").max()

# Define return periods and corresponding probabilities
return_periods = np.array([2, 5, 10, 25, 50, 100])
probabilities = 1 - 1 / return_periods

# Dictionary to store Gumbel parameters for each duration
gumbel_params = {}
durations = ["30mns", "1h", "3h", "24h"]
duration_hours = [30, 60, 180, 1440]

# # Fit GEV distribution and calculate intensities
intensities = np.zeros((len(return_periods), len(durations)))
for j, dur in enumerate(durations):
    shape, loc, scale = stats.genextreme.fit(annual_max_intensity[dur])
    gumbel_params[dur] = (shape, loc)  # Now storing shape, location, scale
    for i, prob in enumerate(probabilities):
        intensities[i, j] = stats.genextreme.ppf(c=shape, loc=loc, scale=scale, q=prob)

# Fit Gumbel distribution and calculate intensities
# intensities = np.zeros((len(return_periods), len(durations)))
# for j, dur in enumerate(durations):
#     loc, scale = stats.gumbel_r.fit(annual_max_intensity[dur])
#     gumbel_params[dur] = (loc, scale)  # Now storing location and scale
#     for i, prob in enumerate(probabilities):
#         intensities[i, j] = stats.gumbel_r.ppf(q=prob, loc=loc, scale=scale)

# Create IDF curve plot
plt.figure(figsize=(10, 6))
for i, rp in enumerate(return_periods):
    plt.plot(duration_hours, intensities[i], label=f"T = {rp} years")

plt.xlabel("Duration (minutes)")
plt.ylabel("Intensity (mm/hr)")
plt.title("Intensity-Duration-Frequency (IDF) Curve")
plt.grid(True, which="both", ls="-")
plt.legend()

output_dir = os.path.join(os.path.dirname(__file__), "data")
plt.savefig(os.path.join(output_dir, "idf_curve.png"))
plt.show()

# Save the IDF data
idf_data = pd.DataFrame(
    intensities, index=return_periods, columns=[f"{d} mins" for d in duration_hours]
)
idf_data.index.name = "Return Period (years)"
idf_data.to_csv(os.path.join(output_dir, "idf_data.csv"))

print("Gumbel Distribution Parameters:")
for dur in durations:
    shape, loc = gumbel_params[dur]
    print(f"{dur}: shape = {shape:.4f}, location = {loc:.4f}")
