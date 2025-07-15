import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_path_5mns = os.path.join(os.path.dirname(__file__), "data", "gpm-bey-5mns.csv")
    data_path_10mns = os.path.join(
        os.path.dirname(__file__), "data", "gpm-bey-10mns.csv"
    )
    data_path_15mns = os.path.join(
        os.path.dirname(__file__), "data", "gpm-bey-15mns.csv"
    )
    data_path_30mns = os.path.join(
        os.path.dirname(__file__), "data", "gpm-bey-30mns.csv"
    )
    data_path_1h = os.path.join(os.path.dirname(__file__), "data", "gpm-bey-1hr.csv")
    data_path_90min = os.path.join(
        os.path.dirname(__file__), "data", "gpm-bey-90min.csv"
    )
    data_path_2h = os.path.join(os.path.dirname(__file__), "data", "gpm-bey-2hr.csv")
    data_path_3h = os.path.join(os.path.dirname(__file__), "data", "gpm-bey-3hr.csv")
    data_path_6h = os.path.join(os.path.dirname(__file__), "data", "gpm-bey-6hr.csv")
    data_path_12h = os.path.join(os.path.dirname(__file__), "data", "gpm-bey-12hr.csv")
    data_path_15h = os.path.join(  # New 15hr path
        os.path.dirname(__file__), "data", "gpm-bey-15hr.csv"
    )
    data_path_18h = os.path.join(  # New 18hr path
        os.path.dirname(__file__), "data", "gpm-bey-18hr.csv"
    )
    data_path_24h = os.path.join(os.path.dirname(__file__), "data", "gpm-bey-daily.csv")

    df_5mns = pd.read_csv(data_path_5mns)
    df_10mns = pd.read_csv(data_path_10mns)
    df_15mns = pd.read_csv(data_path_15mns)
    df_30mns = pd.read_csv(data_path_30mns)
    df_30mns["value"] = df_30mns["value"]  # / 0.5

    # Load the generated datasets
    df_1h = pd.read_csv(data_path_1h)
    df_90min = pd.read_csv(data_path_90min)
    df_2h = pd.read_csv(data_path_2h)
    df_3h = pd.read_csv(data_path_3h)
    df_6h = pd.read_csv(data_path_6h)
    df_12h = pd.read_csv(data_path_12h)
    df_15h = pd.read_csv(data_path_15h)  # Load 15hr data
    df_18h = pd.read_csv(data_path_18h)  # Load 18hr data
    df_24h = pd.read_csv(data_path_24h)

    # Convert the time columns to datetime
    df_5mns["date"] = pd.to_datetime(df_5mns["date"])
    df_10mns["date"] = pd.to_datetime(df_10mns["date"])
    df_15mns["date"] = pd.to_datetime(df_15mns["date"])
    df_30mns["date"] = pd.to_datetime(df_30mns["date"])
    df_1h["date"] = pd.to_datetime(df_1h["date"])
    df_90min["date"] = pd.to_datetime(df_90min["date"])
    df_2h["date"] = pd.to_datetime(df_2h["date"])
    df_3h["date"] = pd.to_datetime(df_3h["date"])
    df_6h["date"] = pd.to_datetime(df_6h["date"])
    df_12h["date"] = pd.to_datetime(df_12h["date"])
    df_15h["date"] = pd.to_datetime(df_15h["date"])  # Convert 15hr dates
    df_18h["date"] = pd.to_datetime(df_18h["date"])  # Convert 18hr dates
    df_24h["date"] = pd.to_datetime(df_24h["date"])

    # Set the time columns as the index
    df_5mns.set_index("date", inplace=True)
    df_10mns.set_index("date", inplace=True)
    df_15mns.set_index("date", inplace=True)
    df_30mns.set_index("date", inplace=True)
    df_1h.set_index("date", inplace=True)
    df_90min.set_index("date", inplace=True)
    df_2h.set_index("date", inplace=True)
    df_3h.set_index("date", inplace=True)
    df_6h.set_index("date", inplace=True)
    df_12h.set_index("date", inplace=True)
    df_15h.set_index("date", inplace=True)  # Set 15hr index
    df_18h.set_index("date", inplace=True)  # Set 18hr index
    df_24h.set_index("date", inplace=True)

    # Merge the dataframes on the index
    df = pd.concat(
        [
            df_5mns,
            df_10mns,
            df_15mns,
            df_30mns,
            df_1h,
            df_90min,
            df_2h,
            df_3h,
            df_6h,
            df_12h,
            df_15h,
            df_18h,
            df_24h,
        ],
        axis=1,
    )

    df.columns = [
        "5mns",
        "10mns",
        "15mns",
        "30mns",
        "1h",
        "90min",
        "2h",
        "3h",
        "6h",
        "12h",
        "15h",
        "18h",
        "24h",
    ]

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
df_intensity["5mns"] = df["5mns"] / 0.25  # 5 mins = 1/12 hours
df_intensity["10mns"] = df["10mns"] / 0.25  # 10 mins = 1/6 hours
df_intensity["15mns"] = df["15mns"] / 0.25  # 15 mins = 1/4 hours
df_intensity["30mns"] = df["30mns"] / 0.25  # 30 mins = 0.5 hours
df_intensity["1h"] = df["1h"] / 0.15  # / 1  # 1 hour (already in mm/hr)
df_intensity["90min"] = (
    df["90min"] / 0.15 / 1.5
)  # 90 mins = 1.5 hours (already in mm/hr)
df_intensity["2h"] = df["2h"] / 0.15 / 2  # 2 hours (already in mm/hr)
df_intensity["3h"] = df["3h"] / 0.18 / 3  # 3 hours (already in mm/hr)
df_intensity["6h"] = df["6h"] / 0.25 / 6  # 6 hours (already in mm/hr)
df_intensity["12h"] = df["12h"] / 0.25 / 12  # 12 hours (already in mm/hr)
df_intensity["15h"] = df["15h"] / 0.27 / 15  # 15 hours
df_intensity["18h"] = df["18h"] / 0.28 / 18  # 18 hours
df_intensity["24h"] = df["24h"] / 0.3 / 24  # 24 hours

# Reorder columns to put 'year' first
columns = ["year"] + [col for col in df_intensity.columns if col != "year"]
df_intensity = df_intensity[columns]
df_intensity.to_csv(
    os.path.join(os.path.dirname(__file__), "results", "historical_intensity.csv"),
    index=True,
)

# Store intensity values in separate columns for IDF analysis
df_intensity["5mns_intensity"] = df_intensity["5mns"].copy()
df_intensity["10mns_intensity"] = df_intensity["10mns"].copy()
df_intensity["15mns_intensity"] = df_intensity["15mns"].copy()
df_intensity["30mns_intensity"] = df_intensity["30mns"].copy()
df_intensity["1h_intensity"] = df_intensity["1h"].copy()
df_intensity["90min_intensity"] = df_intensity["90min"].copy()
df_intensity["2h_intensity"] = df_intensity["2h"].copy()
df_intensity["3h_intensity"] = df_intensity["3h"].copy()
df_intensity["6h_intensity"] = df_intensity["6h"].copy()
df_intensity["12h_intensity"] = df_intensity["12h"].copy()
df_intensity["15h_intensity"] = df_intensity["15h"].copy()  # Add 15hr intensity
df_intensity["18h_intensity"] = df_intensity["18h"].copy()  # Add 18hr intensity
df_intensity["24h_intensity"] = df_intensity["24h"].copy()

# Get annual maximum intensities first
annual_max_intensity = (
    df_intensity[
        [
            "year",
            "5mns",
            "10mns",
            "15mns",
            "30mns",
            "1h",
            "90min",
            "2h",
            "3h",
            "6h",
            "12h",
            "15h",
            "18h",
            "24h",
        ]
    ]
    .groupby("year")
    .max()
)

output_path = os.path.join(os.path.dirname(__file__), "results", "annual_max_intensity")
annual_max_intensity.to_csv(f"{output_path}.csv", index=True)

# Define return periods and corresponding probabilities
return_periods = np.array([2, 5, 10, 25, 50, 100])
probabilities = 1 - 1 / return_periods

# Dictionary to store Gumbel parameters for each duration
gumbel_params = {}
emperical_params = {}
durations = [
    "5mns",
    "10mns",
    "15mns",
    "30mns",
    "1h",
    "90min",
    "2h",
    "3h",
    "6h",
    "12h",
    "15h",
    "18h",
    "24h",
]
duration_hours = [
    5,
    10,
    15,
    30,
    60,
    90,
    120,
    180,
    360,
    720,
    900,
    1080,
    1440,
]  # Added 15h (900 mins) and 18h (1080 mins)

# Fit Gumbel distribution and calculate intensities
intensities_gum = np.zeros((len(return_periods), len(durations)))
intensities_wbl = np.zeros((len(return_periods), len(durations)))

for j, dur in enumerate(durations):
    loc, scale = stats.gumbel_r.fit(annual_max_intensity[dur])
    gumbel_params[dur] = (loc, scale)  # Now storing location and scale
    for i, prob in enumerate(probabilities):
        intensities_gum[i, j] = stats.gumbel_r.ppf(q=prob, loc=loc, scale=scale)

# Define colors for different return periods
colors = ["blue", "green", "red", "purple", "orange", "brown"]
output_dir = os.path.join(os.path.dirname(__file__), "figures")

# Original IDF curve plot (simplified version)
plt.figure(figsize=(10, 6))
for i, rp in enumerate(return_periods):
    plt.plot(duration_hours, intensities_gum[i], label=f"{rp}-year")

plt.xlabel("Duration (minutes)")
plt.ylabel("Rainfall Intensity (mm/hr)")
plt.title("Intensity-Duration-Frequency (IDF) Curves\nGumbel Distribution Method")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_dir = os.path.join(os.path.dirname(__file__), "figures")
plt.savefig(os.path.join(output_dir, "idf_curves_trad.png"), dpi=300)
plt.show()

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
