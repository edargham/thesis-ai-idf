import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


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

np.random.seed(368683)
torch.manual_seed(368683)

# Load the data - change input file to annual_max_intensity.csv
input_file = os.path.join(
    os.path.dirname(__file__), "..", "results", "annual_max_intensity.csv"
)

df = pd.read_csv(input_file)

# Extract intensity data for all durations
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
data_24h = df["24h"].values

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
df_24h = pd.DataFrame({"duration": 1440, "intensity": data_24h})

# Combine all DataFrames
combined_df = pd.concat(
    [df_5mns, df_10mns, df_15mns, df_30mns, df_1h, df_90min, df_2h, df_3h,
     df_6h, df_12h, df_15h, df_18h, df_24h], ignore_index=True
)

# Transform the data to make the relationship linear
# For IDF relationships, a log-log transformation is common
combined_df["log_duration"] = np.log(combined_df["duration"])
combined_df["log_intensity"] = np.log(combined_df["intensity"])

# Load empirical IDF data for comparison and training targets
idf_data = pd.read_csv(os.path.join(
    os.path.dirname(__file__), "..", "results", "idf_data.csv"))

# Create dataset for training
durations_minutes = [5, 10, 15, 30, 60, 90, 120, 180, 360, 720, 900, 1080, 1440]
return_periods = [2, 5, 10, 25, 50, 100]

# Create a dataset with duration-return period-intensity triplets from IDF data
training_data = []

for rp in return_periods:
    row = idf_data[idf_data["Return Period (years)"] == rp].iloc[0]
    for i, duration in enumerate(durations_minutes):
        column_name = f"{duration} mins" if duration != 60 else "60 mins"
        intensity = row[column_name]
        training_data.append({
            "duration": duration,
            "return_period": rp,
            "intensity": intensity
        })

idf_training_df = pd.DataFrame(training_data)

# Create a PyTorch dataset for the IDF relationship
class IDFDataset(Dataset):
    def __init__(
        self,
        dataframe,
        duration_col="duration",
        rp_col="return_period",
        intensity_col="intensity",
    ):
        self.dataframe = dataframe

        # Ensure data is properly converted to numpy arrays first
        duration_values = dataframe[duration_col].values.astype(np.float32)
        rp_values = dataframe[rp_col].values.astype(np.float32)
        intensity_values = dataframe[intensity_col].values.astype(np.float32)

        # Add small constant to avoid log(0) and apply log transform
        epsilon = 1e-6
        self.X_duration = np.log(duration_values + epsilon).reshape(-1, 1)
        self.X_rp = np.log(rp_values + epsilon).reshape(-1, 1)
        self.y = np.log(intensity_values + epsilon).reshape(-1, 1)

        # Scale features
        self.scaler_duration = MinMaxScaler()
        self.scaler_rp = MinMaxScaler()
        self.scaler_intensity = MinMaxScaler()

        self.X_duration = self.scaler_duration.fit_transform(self.X_duration)
        self.X_rp = self.scaler_rp.fit_transform(self.X_rp)
        self.y = self.scaler_intensity.fit_transform(self.y)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = np.concatenate([self.X_duration[idx], self.X_rp[idx]], axis=0)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.float32
        )

    def inverse_transform_intensity(self, y_scaled):
        """Convert scaled log-intensity back to original intensity."""
        y_log = self.scaler_intensity.inverse_transform(y_scaled)
        return np.exp(y_log) - 1e-6

    def get_scalers(self):
        """Return scalers for later use."""
        return self.scaler_duration, self.scaler_rp, self.scaler_intensity


# Create dataset and split into train/test
idf_dataset = IDFDataset(idf_training_df)
train_size = int(0.5 * len(idf_dataset))
test_size = len(idf_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    idf_dataset, [train_size, test_size]
)

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model for IDF curve generation
class IDFModel(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[24, 24]):
        super(IDFModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, x):
        return self.layers(x)


# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IDFModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 2500
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    # Evaluation phase after each epoch
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.6f}, "
            f"Test Loss: {test_loss / len(test_loader):.6f}"
        )

# Final evaluation
model.eval()
test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader):.6f}")

# Get scalers for inverse transformation
scaler_duration, scaler_rp, scaler_intensity = idf_dataset.get_scalers()


# Generate IDF curves for specific return periods
def generate_idf_curve(model, return_periods, durations_minutes):
    model.eval()
    idf_curves = {}

    for rp in return_periods:
        intensities = []
        for dur in durations_minutes:
            # Apply log and scaling transformations
            log_dur = np.log(dur + 1e-6).reshape(-1, 1)
            log_rp = np.log(rp + 1e-6).reshape(-1, 1)

            scaled_dur = scaler_duration.transform(log_dur)
            scaled_rp = scaler_rp.transform(log_rp)

            # Prepare input for model
            x = np.concatenate([scaled_dur, scaled_rp], axis=1)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

            # Get prediction
            with torch.no_grad():
                y_scaled = model(x_tensor)

            # Convert back to original scale
            intensity = idf_dataset.inverse_transform_intensity(y_scaled.cpu().numpy())
            intensities.append(float(intensity))

        idf_curves[rp] = intensities

    return idf_curves


# Generate IDF curves
standard_durations_minutes = [5, 10, 15, 30, 60, 90, 120, 180, 360, 720, 900, 1080, 1440]

idf_curves = generate_idf_curve(model, return_periods, standard_durations_minutes)

# Convert to DataFrame for easier handling
idf_df = pd.DataFrame({"Duration_minutes": standard_durations_minutes})

for rp in return_periods:
    idf_df[f"T{rp}"] = idf_curves[rp]

print("\nGenerated IDF Curve Data:")
print(idf_df)

# Create a nicer formatted table for display
display_df = pd.DataFrame(
    {
        "Return_Period": return_periods,
        "5min": [idf_curves[rp][0] for rp in return_periods],
        "10min": [idf_curves[rp][1] for rp in return_periods],
        "15min": [idf_curves[rp][2] for rp in return_periods],
        "30min": [idf_curves[rp][3] for rp in return_periods],
        "60min": [idf_curves[rp][4] for rp in return_periods],
        "90min": [idf_curves[rp][5] for rp in return_periods],
        "120min": [idf_curves[rp][6] for rp in return_periods],
        "180min": [idf_curves[rp][7] for rp in return_periods],
        "360min": [idf_curves[rp][8] for rp in return_periods],
        "720min": [idf_curves[rp][9] for rp in return_periods],
        "900min": [idf_curves[rp][10] for rp in return_periods],
        "1080min": [idf_curves[rp][11] for rp in return_periods],
        "1440min": [idf_curves[rp][12] for rp in return_periods],
    }
)

print("\nIDF Table - Intensity (mm/hr):")
print(display_df)

# Map durations in our model to the columns in empirical data
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

# Generate smooth curves
smooth_durations_minutes = np.linspace(5, 1440, 1440//5)  # From 5 minutes to 24 hours

# Generate smooth IDF curves for different return periods
smooth_idf_curves = {}

for rp in return_periods:
    intensities = []
    for dur in smooth_durations_minutes:
        # Apply log and scaling transformations
        log_dur = np.log(dur + 1e-6).reshape(-1, 1)
        log_rp = np.log(rp + 1e-6).reshape(-1, 1)

        scaled_dur = scaler_duration.transform(log_dur)
        scaled_rp = scaler_rp.transform(log_rp)

        # Prepare input for model
        x = np.concatenate([scaled_dur, scaled_rp], axis=1)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

        # Get prediction
        with torch.no_grad():
            y_scaled = model(x_tensor)

        # Convert back to original scale
        intensity = idf_dataset.inverse_transform_intensity(y_scaled.cpu().numpy())
        intensities.append(float(intensity))

    smooth_idf_curves[rp] = intensities

# Save IDF curves to CSV for standard durations only
idf_df_data = {'Duration (minutes)': standard_durations_minutes}
for rp in return_periods:
    idf_df_data[f'{rp}-year'] = idf_curves[rp]

idf_df = pd.DataFrame(idf_df_data)
csv_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "idf_curves_ANN.csv"
)
idf_df.to_csv(csv_path, index=False)
print(f"IDF curves data saved to: {csv_path}")

# Calculate metrics (RMSE, MAE, R2, NSE) for each return period
rmse_values = []
mae_values = []
r2_values = []
nse_values = []

for rp in return_periods:
    empirical_row = idf_data[idf_data["Return Period (years)"] == rp].iloc[0]
    
    # Extract values from empirical data and model predictions for this return period
    y_true = []
    y_pred = []
    
    for i in range(len(standard_durations_minutes)):
        empirical_col = duration_mapping[i]
        y_true.append(empirical_row[empirical_col])
        y_pred.append(idf_curves[rp][i])
    
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

# Create a figure to compare model predictions with empirical data
plt.figure(figsize=(10, 6))

# Define colors for different return periods
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

# Plot both model predictions and empirical data for comparison
for i, rp in enumerate(return_periods):
    # Model prediction (solid line) - use smooth curves
    plt.plot(smooth_durations_minutes, smooth_idf_curves[rp], '-', color=colors[i], 
             linewidth=2, label=f"ANN T = {rp} years")
    
    # Empirical data (dashed line with markers)
    empirical_row = idf_data[idf_data["Return Period (years)"] == rp].iloc[0]
    empirical_values = [empirical_row[duration_mapping[j]] for j in range(len(standard_durations_minutes))]
    plt.plot(standard_durations_minutes, empirical_values, '--', color=colors[i], linewidth=1.5, 
             label=f"Gumbel T = {rp} years")

plt.xlabel('Duration (minutes)', fontsize=12)
plt.ylabel('Intensity (mm/hr)', fontsize=12)
plt.title('IDF Curves Comparison: Neural Network vs Gumbel', fontsize=14)
plt.grid(True, which="both", ls="-")

# Add metrics as text
plt.text(0.02, 0.98, f"RMSE: {overall_rmse:.4f}\nMAE: {overall_mae:.4f}\nR²: {overall_r2:.4f}\nNSE: {overall_nse:.4f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust legend to avoid crowding
plt.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "figures", "idf_comparison_ann.png"),
    dpi=300,
)
print(f"Comparison plot saved to: {os.path.join(os.path.dirname(__file__), '..', 'figures', 'idf_comparison_ann.png')}")

# Original IDF curve plot - using smooth curves
plt.figure(figsize=(10, 6))
for i, rp in enumerate(return_periods):
    plt.plot(smooth_durations_minutes, smooth_idf_curves[rp], color=colors[i], 
             label=f"{rp}-year return period")

plt.xlabel("Duration (minutes)")
plt.ylabel("Rainfall Intensity (mm/hr)")
plt.title("Intensity-Duration-Frequency (IDF) Curves\nGenerated by Neural Network")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "figures", "idf_curves_ann.png"), 
    dpi=300, 
    bbox_inches="tight"
)
print(f"IDF curves plot saved to: {os.path.join(os.path.dirname(__file__), '..', 'figures', 'idf_curves_ann.png')}")
# Add metrics as text
plt.text(0.02, 0.98, f"RMSE: {overall_rmse:.4f}\nMAE: {overall_mae:.4f}\nR²: {overall_r2:.4f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adjust legend to avoid crowding
plt.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "figures", "idf_comparison_ann.png"),
    dpi=300,
)
print(f"Comparison plot saved to: {os.path.join(os.path.dirname(__file__), '..', 'figures', 'idf_comparison_ann.png')}")

# Original IDF curve plot - using smooth curves
plt.figure(figsize=(10, 6))
for i, rp in enumerate(return_periods):
    plt.plot(smooth_durations_minutes, smooth_idf_curves[rp], color=colors[i], 
             label=f"{rp}-year return period")

plt.xlabel("Duration (minutes)")
plt.ylabel("Rainfall Intensity (mm/hr)")
plt.title("Intensity-Duration-Frequency (IDF) Curves\nGenerated by Neural Network")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "figures", "idf_curves_ann.png"), 
    dpi=300, 
    bbox_inches="tight"
)
print(f"IDF curves plot saved to: {os.path.join(os.path.dirname(__file__), '..', 'figures', 'idf_curves_ann.png')}")
