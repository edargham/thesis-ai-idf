import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
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

data_5mns = df["5mns"].values
data_10mns = df["10mns"].values
data_15mns = df["15mns"].values
data_30mns = df["30mns"].values
data_1h = df["1h"].values
data_90min = df["90min"].values
data_2h = df["2h"].values
data_3h = df["3h"].values
data_6h = df["6h"].values
data_12h = df["12h"].values
data_15h = df["15h"].values
data_18h = df["18h"].values
data24h = df["24h"].values

# Create DataFrames for each duration
df_5mns = pd.DataFrame({"duration": 5, "intensity": data_5mns})
df_10mns = pd.DataFrame({"duration": 10, "intensity": data_10mns})
df_15mns = pd.DataFrame({"duration": 15, "intensity": data_15mns})
df_30mns = pd.DataFrame({"duration": 30, "intensity": data_30mns})
df_1h = pd.DataFrame({"duration": 60, "intensity": data_1h})
df_90min = pd.DataFrame({"duration": 90, "intensity": data_90min})
df_2h = pd.DataFrame({"duration": 120, "intensity": data_2h})
df_3h = pd.DataFrame({"duration": 180, "intensity": data_3h})
df_6h = pd.DataFrame({"duration": 360, "intensity": data_6h})
df_12h = pd.DataFrame({"duration": 720, "intensity": data_12h})
df_15h = pd.DataFrame({"duration": 900, "intensity": data_15h})
df_18h = pd.DataFrame({"duration": 1080, "intensity": data_18h})
df_24h = pd.DataFrame({"duration": 1440, "intensity": data24h})

# Combine all DataFrames
combined_df = pd.concat(
    [df_5mns, df_10mns, df_15mns, df_30mns, df_1h, df_90min, df_2h, df_3h, 
     df_6h, df_12h, df_15h, df_18h, df_24h], ignore_index=True
)

# Transform the data to make the relationship linear
# For IDF relationships, a log-log transformation is common
combined_df["log_duration"] = np.log(combined_df["duration"])
combined_df["log_intensity"] = np.log(combined_df["intensity"])

# Split the data into training and testing sets
X = combined_df[["log_duration"]]
y = combined_df["log_intensity"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=368683
)

# Standard scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

grid_search_params = {
    "C": [0.1, 1.0, 10.0],
    "epsilon": [0.01, 0.1, 0.2],
    "gamma": [0.01, 0.1, 1.0],
    "kernel": ["rbf"],
}

tuner = GridSearchCV(
    SVR(),
    grid_search_params,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1,
)
tuner.fit(X_train_scaled, y_train_scaled)
print("Best parameters from grid search:", tuner.best_params_)
print("Best score from grid search:", tuner.best_score_)

# Fit linear regression model
model = SVR(**(tuner.best_params_))
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

# Load gumbel IDF data for comparison
gumbel_idf = pd.read_csv(os.path.join(
    os.path.dirname(__file__), "..", "results", "idf_data.csv"))

# Generate smooth curves
durations = np.linspace(5, 1440, 1440//5)  # From 5 minutes to 24 hours
log_durations = np.log(durations).reshape(-1, 1)
log_durations_scaled = scaler_X.transform(log_durations)

# Base prediction
base_log_intensities_scaled = model.predict(log_durations_scaled)
base_log_intensities = scaler_y.inverse_transform(
    base_log_intensities_scaled.reshape(-1, 1)
).flatten()
base_intensities = np.exp(base_log_intensities)

# Create dictionary to store IDF curves for different return periods
idf_curves = {}

# Generate IDF curves for each return period
for return_period in return_periods:
    intensities_rp = base_intensities * frequency_factors[return_period]
    idf_curves[return_period] = intensities_rp

# Save IDF curves to CSV for all standard durations from duration_mapping
standard_durations_minutes = [5, 10, 15, 30, 60, 90, 120, 180, 360, 720, 900, 1080, 1440]

# Generate curves for all standard durations
standard_idf_curves = {}
for return_period in return_periods:
    standard_intensities = []
    for duration in standard_durations_minutes:
        # Find the closest duration in our predictions
        duration_idx = np.abs(durations - duration).argmin()
        standard_intensities.append(idf_curves[return_period][duration_idx])
    standard_idf_curves[return_period] = standard_intensities

# Save standard IDF curves to CSV
idf_df_data = {'Duration (minutes)': standard_durations_minutes}
for rp in return_periods:
    idf_df_data[f'{rp}-year'] = standard_idf_curves[rp]

idf_df = pd.DataFrame(idf_df_data)
csv_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "idf_curves_SVM.csv"
)
idf_df.to_csv(csv_path, index=False)
print(f"IDF curves data saved to: {csv_path}")

# Define duration mapping for column names
duration_mapping = {
    0: "5 mins",
    1: "10 mins", 
    2: "15 mins",
    3: "30 mins",
    4: "60 mins",
    5: "90 mins",
    6: "120 mins",
    7: "180 mins",
    8: "360 mins",
    9: "720 mins",
    10: "900 mins",
    11: "1080 mins",
    12: "1440 mins"
}

# Calculate metrics for each return period
rmse_values = []
mae_values = []
r2_values = []
nse_values = []


for rp in return_periods:
    gumbel_row = gumbel_idf[gumbel_idf["Return Period (years)"] == rp].iloc[0]
    
    # Extract values from gumbel data for this return period
    y_true = []
    y_pred = []
    
    for i, duration in enumerate(standard_durations_minutes):
        gumbel_col = duration_mapping[i]
        y_true.append(gumbel_row[gumbel_col])
        
        # Use the precomputed values from standard_idf_curves
        y_pred.append(standard_idf_curves[rp][i])
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nse = nash_sutcliffe_efficiency(y_true, y_pred)
    
    rmse_values.append(rmse)
    mae_values.append(mae)
    r2_values.append(r2)
    nse_values.append(nse)

# Display metrics
metrics_df = pd.DataFrame({
    'Return Period': return_periods,
    'RMSE': [round(x, 4) for x in rmse_values],
    'MAE': [round(x, 4) for x in mae_values],
    'R2': [round(x, 4) for x in r2_values],
    'NSE': [round(x, 4) for x in nse_values]
})
print("\nModel Performance Metrics by Return Period:")
print(metrics_df)

# Calculate overall metrics
overall_rmse = np.mean(rmse_values)
overall_mae = np.mean(mae_values)
overall_r2 = np.mean(r2_values)
overall_nse = np.mean(nse_values)

print(f"\nOverall RMSE: {overall_rmse:.4f}")
print(f"Overall MAE: {overall_mae:.4f}")
print(f"Overall R2: {overall_r2:.4f}")
print(f"Overall NSE: {overall_nse:.4f}")

# Create a figure to compare model predictions with gumbel data
plt.figure(figsize=(10, 6))

# Define colors for different return periods
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# Plot both model predictions and gumbel data for comparison
for i, rp in enumerate(return_periods):
    # Model prediction (solid line)
    plt.plot(durations, idf_curves[rp], '-', color=colors[i], 
             linewidth=2, label=f"SVM T = {rp} years")
    
    # gumbel data (dashed line with markers)
    gumbel_row = gumbel_idf[gumbel_idf["Return Period (years)"] == rp].iloc[0]
    gumbel_values = [gumbel_row[duration_mapping[j]] for j in range(len(standard_durations_minutes))]
    plt.plot(standard_durations_minutes, gumbel_values, '--', color=colors[i], linewidth=1.5, label=f"Gumbel T = {rp} years")

# plt.xscale('log')
plt.xlabel('Duration (minutes)', fontsize=12)
plt.ylabel('Intensity (mm/hr)', fontsize=12)
plt.title('IDF Curves Comparison: SVM vs Gumbel', fontsize=14)
plt.grid(True, which="both", ls="-")

# Add metrics as text
plt.text(0.02, 0.98, f"RMSE: {overall_rmse:.4f}\nMAE: {overall_mae:.4f}\nR²: {overall_r2:.4f}\nNSE: {overall_nse:.4f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust legend to avoid crowding
plt.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "figures", "idf_comparison_svm.png"),
    dpi=300,
)
print(f"Comparison plot saved to: {os.path.join(os.path.dirname(__file__), '..', 'figures', 'idf_comparison_svm.png')}")

# Original IDF curve plot
plt.figure(figsize=(10, 6))
for return_period in return_periods:
    plt.plot(durations, idf_curves[return_period], label=f"{return_period}-year return period")

plt.xlabel("Duration (minutes)")
plt.ylabel("Intensity (mm/h)")
plt.title("Intensity-Duration-Frequency (IDF) Curves using SVR")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "figures", "idf_curves_svm.png"),
    dpi=300,
)

print(f"IDF curves plot saved to: {os.path.join(os.path.dirname(__file__), '..', 'figures', 'idf_curves_svm.png')}")
