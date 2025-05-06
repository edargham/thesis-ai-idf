import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


# Calculate Nash-Sutcliffe Efficiency (NSE)
def nash_sutcliffe_efficiency(observed, simulated):
    """
    Compute Nash-Sutcliffe Efficiency (NSE).

    Parameters:
        observed (array-like): Array of observed values.
        simulated (array-like): Array of simulated values.

    Returns:
        float: Nash-Sutcliffe Efficiency coefficient.
    """
    observed = np.array(observed)
    simulated = np.array(simulated)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else np.nan


input_file = os.path.join(
    os.path.dirname(__file__), "..", "results", "annual_max_intensity.csv"
)

df = pd.read_csv(input_file)

data_30mns = df["30mns"].values
data_1h = df["1h"].values
data_3h = df["3h"].values
data24h = df["24h"].values

# Create DataFrames for each duration
df_30mns = pd.DataFrame({"duration": 30, "intensity": data_30mns})

df_1h = pd.DataFrame({"duration": 60, "intensity": data_1h})

df_3h = pd.DataFrame({"duration": 180, "intensity": data_3h})

df_24h = pd.DataFrame({"duration": 1440, "intensity": data24h})

# Combine all DataFrames
combined_df = pd.concat([df_30mns, df_1h, df_3h, df_24h], ignore_index=True)

# Transform the data to make the relationship linear
# For IDF relationships, a log-log transformation is common
combined_df["log_duration"] = np.log(combined_df["duration"])
combined_df["log_intensity"] = np.log(combined_df["intensity"])

# Split the data into training and testing sets
X = combined_df[["log_duration"]]
y = combined_df["log_intensity"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=368683
)

# Standard scale the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Fit linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

# Evaluate the model on test data
y_pred_scaled = model.predict(X_test_scaled)
y_pred = np.exp(scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten())
y_actual = np.exp(y_test.values)

rmse = root_mean_squared_error(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
nse = nash_sutcliffe_efficiency(y_actual, y_pred)

print("Model Evaluation on Test Data:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"NSE: {nse:.2f}")

# Generate IDF curves using frequency factors for different return periods
return_periods = [2, 5, 10, 25, 50, 100]
frequency_factors = {2: 0.85, 5: 1.15, 10: 1.35, 25: 1.60, 50: 1.80, 100: 2.00}

# Generate smooth curves
durations = np.linspace(30, 1440, 30)  # From 10 minutes to 24 hours
log_durations = np.log(durations).reshape(-1, 1)
log_durations_scaled = scaler_X.transform(log_durations)

# Base prediction
base_log_intensities_scaled = model.predict(log_durations_scaled)
base_log_intensities = scaler_y.inverse_transform(
    base_log_intensities_scaled.reshape(-1, 1)
).flatten()
base_intensities = np.exp(base_log_intensities)

# Plot IDF curves
plt.figure(figsize=(10, 6))

for return_period in return_periods:
    intensities_rp = base_intensities * frequency_factors[return_period]
    plt.plot(durations, intensities_rp, label=f"{return_period}-year return period")

plt.xlabel("Duration (minutes)")
plt.ylabel("Intensity (mm/h)")
plt.title("Intensity-Duration-Frequency (IDF) Curves using Linear Regression")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "idf_curves_lr.png"))
