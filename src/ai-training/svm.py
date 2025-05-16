import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
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
data_3h = df["3h"].values
data24h = df["24h"].values

# Create DataFrames for each duration
df_5mns = pd.DataFrame({"duration": 5, "intensity": data_5mns})
df_10mns = pd.DataFrame({"duration": 10, "intensity": data_10mns})
df_15mns = pd.DataFrame({"duration": 15, "intensity": data_15mns})
df_30mns = pd.DataFrame({"duration": 30, "intensity": data_30mns})
df_1h = pd.DataFrame({"duration": 60, "intensity": data_1h})
df_3h = pd.DataFrame({"duration": 180, "intensity": data_3h})
df_24h = pd.DataFrame({"duration": 1440, "intensity": data24h})

# Combine all DataFrames
combined_df = pd.concat(
    [df_5mns, df_10mns, df_15mns, df_30mns, df_1h, df_3h, df_24h], ignore_index=True
)

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

# Load empirical IDF data for comparison
empirical_idf = pd.read_csv(os.path.join(
    os.path.dirname(__file__), "..", "results", "empirical_idf_data.csv"))

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

# Sample specific durations for comparison with empirical data
specific_durations = [5, 10, 15, 30, 60, 180, 1440]
duration_mapping = {
    0: "5 mins",
    1: "10 mins", 
    2: "15 mins",
    3: "30 mins",
    4: "60 mins",
    5: "180 mins",
    6: "1440 mins"
}

# Calculate metrics for each return period
rmse_values = []
mae_values = []
r2_values = []


for rp in return_periods:
    empirical_row = empirical_idf[empirical_idf["Return Period (years)"] == rp].iloc[0]
    
    # Extract values from empirical data for this return period
    y_true = []
    y_pred = []
    
    for i, duration in enumerate(specific_durations):
        empirical_col = duration_mapping[i]
        y_true.append(empirical_row[empirical_col])
        
        # Find the closest duration in our predictions
        duration_idx = np.abs(durations - duration).argmin()
        y_pred.append(idf_curves[rp][duration_idx])
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    rmse_values.append(rmse)
    mae_values.append(mae)
    r2_values.append(r2)

# Display metrics
metrics_df = pd.DataFrame({
    'Return Period': return_periods,
    'RMSE': [round(x, 4) for x in rmse_values],
    'MAE': [round(x, 4) for x in mae_values],
    'R2': [round(x, 4) for x in r2_values]
})
print("\nModel Performance Metrics by Return Period:")
print(metrics_df)

# Calculate overall metrics
overall_rmse = np.mean(rmse_values)
overall_mae = np.mean(mae_values)
overall_r2 = np.mean(r2_values)

print(f"\nOverall RMSE: {overall_rmse:.4f}")
print(f"Overall MAE: {overall_mae:.4f}")
print(f"Overall R2: {overall_r2:.4f}")

# Create a figure to compare model predictions with empirical data
plt.figure(figsize=(14, 10))

# Define colors for different return periods
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# Plot both model predictions and empirical data for comparison
for i, rp in enumerate(return_periods):
    # Model prediction (solid line)
    plt.plot(durations, idf_curves[rp], '-', color=colors[i], 
             linewidth=2, label=f"Model T = {rp} years")
    
    # Empirical data (dashed line with markers)
    empirical_row = empirical_idf[empirical_idf["Return Period (years)"] == rp].iloc[0]
    empirical_values = [empirical_row[duration_mapping[j]] for j in range(len(specific_durations))]
    plt.plot(specific_durations, empirical_values, '--', color=colors[i], 
             marker='o', markersize=5, linewidth=1.5, label=f"Empirical T = {rp} years")

plt.xscale('log')
plt.xlabel('Duration (minutes)', fontsize=12)
plt.ylabel('Intensity (mm/hr)', fontsize=12)
plt.title('IDF Curves Comparison: SVM vs Empirical', fontsize=14)
plt.grid(True, which="both", ls="-")

# Add metrics as text
plt.text(0.02, 0.98, f"RMSE: {overall_rmse:.4f}\nMAE: {overall_mae:.4f}\nR²: {overall_r2:.4f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust legend to avoid crowding
plt.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "figures", "idf_comparison_svm.png"),
    dpi=300,
)

# Original IDF curve plot
plt.figure(figsize=(10, 6))
for return_period in return_periods:
    plt.plot(durations, idf_curves[return_period], label=f"{return_period}-year return period")

plt.xlabel("Duration (minutes)")
plt.ylabel("Intensity (mm/h)")
plt.xscale("log")
plt.title("Intensity-Duration-Frequency (IDF) Curves using SVR")
plt.grid(True, which="both", ls="-")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "figures", "idf_curves_svm.png"),
    dpi=300,
)
