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

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
if torch.mps.is_available():
    torch.mps.manual_seed(42)

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

checkpoint_path = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "ann_best.pth"
)

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

# Load empirical IDF data for comparison only (NOT for training to avoid data leak)
idf_data = pd.read_csv(os.path.join(
    os.path.dirname(__file__), "..", "results", "idf_data.csv"))

# Create dataset for training using annual maximum intensity data
durations_minutes = [5, 10, 15, 30, 60, 90, 120, 180, 360, 720, 900, 1080, 1440]
return_periods = [2, 5, 10, 25, 50, 100]

# Create dataset for training using same approach as SVM
# This eliminates the circular dependency on return period estimation

# Create a PyTorch dataset for the IDF relationship using SVM approach
class IDFDataset(Dataset):
    def __init__(
        self,
        annual_max_data,
        duration_col="duration",
        intensity_col="intensity",
    ):
        """
        Dataset for IDF curve modeling using the same approach as SVM.
        
        This dataset matches the SVM approach exactly:
        - Features (X): Duration only (like SVM)
        - Labels (Y): Actual historical intensities from annual_max_intensity.csv
        - Training: Model learns duration → intensity relationship
        - Return Period Handling: Applied via frequency factors (like SVM)
        
        Args:
            annual_max_data (pd.DataFrame): Annual maximum rainfall data
            duration_col (str): Column name for duration values
            intensity_col (str): Column name for intensity values
        """
        self.prepare_data(annual_max_data)

    def prepare_data(self, annual_max_data):
        """
        Prepare data using the same approach as SVM model.
        
        This method creates training data exactly like the SVM approach:
        - Features (X): Duration only (no return periods)
        - Labels (Y): Actual historical intensities from annual_max_data
        - Training: Model learns duration → intensity relationship
        """
        annual_max_data = annual_max_data.dropna()
        
        # Extract data for each duration (same as SVM approach)
        data_5mns = annual_max_data["5mns"].values
        data_10mns = annual_max_data["10mns"].values
        data_15mns = annual_max_data["15mns"].values
        data_30mns = annual_max_data["30mns"].values
        data_1h = annual_max_data["1h"].values
        data_90min = annual_max_data["90min"].values
        data_2h = annual_max_data["2h"].values
        data_3h = annual_max_data["3h"].values
        data_6h = annual_max_data["6h"].values
        data_12h = annual_max_data["12h"].values
        data_15h = annual_max_data["15h"].values
        data_18h = annual_max_data["18h"].values
        data_24h = annual_max_data["24h"].values
        
        # Create DataFrames for each duration (same as SVM)
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
        
        # Combine all DataFrames (same as SVM)
        combined_df = pd.concat([
            df_5mns, df_10mns, df_15mns, df_30mns, df_1h, df_90min, df_2h, df_3h,
            df_6h, df_12h, df_15h, df_18h, df_24h
        ], ignore_index=True)
        
        # Remove any invalid data
        combined_df = combined_df.dropna()
        combined_df = combined_df[(combined_df["duration"] > 0) & (combined_df["intensity"] > 0)]
        
        print(f"Created {len(combined_df)} duration-intensity pairs (same as SVM)")
        
        # Transform the data to make the relationship linear (same as SVM)
        epsilon = 1e-6
        combined_df["log_duration"] = np.log(combined_df["duration"] + epsilon)
        combined_df["log_intensity"] = np.log(combined_df["intensity"] + epsilon)
        
        # Prepare features and targets (same as SVM)
        duration_values = combined_df["log_duration"].values.astype(np.float32)
        intensity_values = combined_df["log_intensity"].values.astype(np.float32)
        
        self.X_duration = duration_values.reshape(-1, 1)  # Only duration (like SVM)
        self.y = intensity_values.reshape(-1, 1)
        
        # Scale features
        self.scaler_duration = MinMaxScaler()
        self.scaler_intensity = MinMaxScaler()
        
        self.X_duration = self.scaler_duration.fit_transform(self.X_duration)
        self.y = self.scaler_intensity.fit_transform(self.y)
        
        print(f"Features range: [{self.X_duration.min():.4f}, {self.X_duration.max():.4f}]")
        print(f"Targets range: [{self.y.min():.4f}, {self.y.max():.4f}]")

    def __len__(self):
        return len(self.X_duration)

    def __getitem__(self, idx):
        # Only duration as input (like SVM)
        features = self.X_duration[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.float32
        )

    def inverse_transform_intensity(self, y_scaled):
        """Convert scaled log-intensity back to original intensity."""
        y_log = self.scaler_intensity.inverse_transform(y_scaled)
        return np.exp(y_log) - 1e-6

    def get_scalers(self):
        """Return scalers for later use."""
        return self.scaler_duration, self.scaler_intensity


# Create dataset and split into train/test
idf_dataset = IDFDataset(df)  # Use annual_max_data directly
train_size = int(0.5 * len(idf_dataset))  # Use 50% for training (same as SVM)
test_size = len(idf_dataset) - train_size

# Set generator for reproducible splits (same seed as SVM)
generator = torch.Generator()
generator.manual_seed(368683)  # Same seed as SVM

train_dataset, test_dataset = torch.utils.data.random_split(
    idf_dataset, [train_size, test_size], generator=generator
)

# Create data loaders
batch_size = 16  # Smaller batch size for small dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
if torch.mps.is_available():
    torch.mps.manual_seed(42)

# Define the neural network model for IDF curve generation
class IDFModel(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[32, 64, 32]):  # input_size=1 for duration only
        super(IDFModel, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.05))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.05))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = IDFModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=5e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=150, T_mult=2, eta_min=1e-6
)

# Learning rate scheduler
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50, verbose=False)

# Early stopping parameters
best_test_loss = float('inf')
patience = 200
patience_counter = 0

# Train the model
num_epochs = 3000
train_losses = []
test_losses = []

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
    
    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)
    
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    
    # Learning rate scheduling
    scheduler.step(avg_test_loss)
    
    # Early stopping
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        patience_counter = 0
        torch.save(model.state_dict(), checkpoint_path)
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, "
            f"Test Loss: {avg_test_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

# Load best model
model.load_state_dict(torch.load(checkpoint_path))
print(f"Best test loss: {best_test_loss:.6f}")

# Final evaluation
model.eval()
test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

print(f"Final Test Loss: {test_loss / len(test_loader):.6f}")

# Get scalers for inverse transformation
scaler_duration, scaler_intensity = idf_dataset.get_scalers()  # Only 2 scalers now


# Generate IDF curves for specific return periods
def generate_idf_curve(model, return_periods, durations_minutes):
    """
    Generate IDF curves using the same approach as SVM with frequency factors.
    
    This function creates intensity predictions using the SVM approach:
    1. Generate base predictions for durations (no return periods)
    2. Apply frequency factors to scale for different return periods
    3. This matches the SVM methodology exactly
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Frequency factors (same as SVM)
    frequency_factors = {2: 0.85, 5: 1.15, 10: 1.35, 25: 1.60, 50: 1.80, 100: 2.00}
    
    # Generate base predictions for all durations
    base_intensities = []
    for dur in durations_minutes:
        # Apply log and scaling transformations (duration only)
        log_dur = np.log(dur + 1e-6).reshape(-1, 1)
        scaled_dur = scaler_duration.transform(log_dur)
        
        # Prepare input for model (duration only)
        x_tensor = torch.tensor(scaled_dur, dtype=torch.float32).to(device)
        
        # Get prediction
        with torch.no_grad():
            y_scaled = model(x_tensor)
        
        # Convert back to original scale
        intensity = idf_dataset.inverse_transform_intensity(y_scaled.cpu().numpy())
        base_intensities.append(float(intensity))
    
    # Apply frequency factors for different return periods
    idf_curves = {}
    for rp in return_periods:
        factor = frequency_factors[rp]
        scaled_intensities = [intensity * factor for intensity in base_intensities]
        idf_curves[rp] = scaled_intensities
    
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

# Generate smooth IDF curves for different return periods using SVM approach
smooth_idf_curves = {}

# Frequency factors (same as SVM)
frequency_factors = {2: 0.85, 5: 1.15, 10: 1.35, 25: 1.60, 50: 1.80, 100: 2.00}

# Generate base predictions for all durations
base_intensities = []
for dur in smooth_durations_minutes:
    # Apply log and scaling transformations (duration only)
    log_dur = np.log(dur + 1e-6).reshape(-1, 1)
    scaled_dur = scaler_duration.transform(log_dur)
    
    # Prepare input for model (duration only)
    x_tensor = torch.tensor(scaled_dur, dtype=torch.float32).to(device)
    
    # Get prediction
    with torch.no_grad():
        y_scaled = model(x_tensor)
    
    # Convert back to original scale
    intensity = idf_dataset.inverse_transform_intensity(y_scaled.cpu().numpy())
    base_intensities.append(float(intensity))

# Apply frequency factors for different return periods
for rp in return_periods:
    factor = frequency_factors[rp]
    scaled_intensities = [intensity * factor for intensity in base_intensities]
    smooth_idf_curves[rp] = scaled_intensities

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

# Check if metrics file exists
metrics_file_path = os.path.join(os.path.dirname(__file__), "..", "results", "model_performance_metrics.csv")

# Create metrics row for ANN model
ann_metrics = {
    'Model': 'ANN',
    'RMSE': overall_rmse,
    'MAE': overall_mae,
    'R2': overall_r2,
    'NSE': overall_nse
}

if os.path.exists(metrics_file_path):
    # Load existing metrics file and append ANN metrics
    overall_df = pd.read_csv(metrics_file_path)
    
    # Check if ANN model already exists in the dataframe
    if 'ANN' in overall_df['Model'].values:
        # Update existing ANN entry - first remove the old entry, then add the new one
        overall_df = overall_df[overall_df['Model'] != 'ANN']
        overall_df = pd.concat([overall_df, pd.DataFrame([ann_metrics])], ignore_index=True)
    else:
        # Append ANN metrics as a new row
        overall_df = pd.concat([overall_df, pd.DataFrame([ann_metrics])], ignore_index=True)
else:
    # Create new dataframe with ANN metrics
    overall_df = pd.DataFrame([ann_metrics])

# Save metrics to CSV
overall_df.to_csv(metrics_file_path, index=False)
print(f"Model performance metrics saved to: {metrics_file_path}")

# Additional robustness check - calculate metrics using all data points
all_y_true = []
all_y_pred = []

for rp in return_periods:
    empirical_row = idf_data[idf_data["Return Period (years)"] == rp].iloc[0]
    for i in range(len(standard_durations_minutes)):
        empirical_col = duration_mapping[i]
        all_y_true.append(empirical_row[empirical_col])
        all_y_pred.append(idf_curves[rp][i])

# Calculate overall metrics using all data points
overall_rmse_all = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
overall_mae_all = mean_absolute_error(all_y_true, all_y_pred)
overall_r2_all = r2_score(all_y_true, all_y_pred)
overall_nse_all = nash_sutcliffe_efficiency(all_y_true, all_y_pred)

print("\nOverall Metrics (All Data Points):")
print(f"RMSE: {overall_rmse_all:.4f}")
print(f"MAE: {overall_mae_all:.4f}")
print(f"R2: {overall_r2_all:.4f}")
print(f"NSE: {overall_nse_all:.4f}")

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
plt.text(0.02, 0.98, f"RMSE: {overall_rmse_all:.4f}\nMAE: {overall_mae_all:.4f}\nR²: {overall_r2_all:.4f}\nNSE: {overall_nse_all:.4f}",
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
