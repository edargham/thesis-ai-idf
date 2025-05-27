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

np.random.seed(368683)
torch.manual_seed(368683)

# Load the data
data_path = os.path.join(
    os.path.dirname(__file__), "..", "results", "historical_intensity.csv"
)

df = pd.read_csv(data_path)

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])
print(f"Data spans from {df['date'].min()} to {df['date'].max()}")
print(f"Total data points: {len(df)}")

# Add year column if not present
if "year" not in df.columns:
    df["year"] = df["date"].dt.year

# Extract annual maximum intensities for each duration
durations = ["5mns", "10mns", "15mns", "30mns", "1h", "3h", "24h"]
years = df["year"].unique()
print(f"Years covered: {min(years)} to {max(years)} ({len(years)} years)")

annual_maxima = pd.DataFrame(columns=["year"] + durations)
annual_maxima["year"] = sorted(years)

for duration in durations:
    for year in years:
        year_data = df[df["year"] == year][duration]
        annual_maxima.loc[annual_maxima["year"] == year, duration] = year_data.max()

print("\nAnnual Maximum Intensities:")
print(annual_maxima.head())


# Calculate empirical return periods using Weibull formula
def calculate_empirical_return_periods(annual_max_data):
    """Calculate empirical return periods for each data point using Weibull plotting position."""
    data_with_rp = []

    for duration in durations:
        # Sort data in descending order
        sorted_data = annual_max_data.sort_values(
            by=duration, ascending=False
        ).reset_index(drop=True)
        n = len(sorted_data)

        # Calculate return periods using Weibull formula: T = (n+1)/m
        # where n is the record length and m is the rank
        sorted_data["rank"] = range(1, n + 1)
        sorted_data["return_period"] = (n + 1) / sorted_data["rank"]

        # Keep only relevant columns
        result = sorted_data[["year", duration, "return_period"]].copy()
        result["duration"] = duration
        result.rename(columns={duration: "intensity"}, inplace=True)

        data_with_rp.append(result)

    # Combine data for all durations
    return pd.concat(data_with_rp, ignore_index=True)


# Create dataset with empirical return periods
empirical_data = calculate_empirical_return_periods(annual_maxima)
print("\nEmpirical Return Periods:")
print(empirical_data.head())

# Map duration strings to numeric hours for model training
duration_mapping = {
    "5mns": 5 / 60,
    "10mns": 10 / 60,
    "15mns": 0.25,
    "30mns": 0.5,
    "1h": 1.0,
    "3h": 3.0,
    "24h": 24.0,
}
empirical_data["duration_hours"] = empirical_data["duration"].map(duration_mapping)


# Create a PyTorch dataset for the IDF relationship
class IDFDataset(Dataset):
    def __init__(
        self,
        dataframe,
        duration_col="duration_hours",
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
idf_dataset = IDFDataset(empirical_data)
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
def generate_idf_curve(model, return_periods, durations_hours):
    model.eval()
    idf_curves = {}

    for rp in return_periods:
        intensities = []
        for dur in durations_hours:
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
return_periods = [2, 5, 10, 25, 50, 100]
durations_hours = [5/60, 10/60, 0.25, 0.5, 1, 3, 24]  # 5min, 10min, 15min, 30min, 1h, 3h, 24h
durations_minutes = [5, 10, 15, 30, 60, 180, 1440]  # For plotting

idf_curves = generate_idf_curve(model, return_periods, durations_hours)

# Convert to DataFrame for easier handling
idf_df = pd.DataFrame({"Duration_hours": durations_hours})

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
        "1h": [idf_curves[rp][4] for rp in return_periods],
        "3h": [idf_curves[rp][5] for rp in return_periods],
        "24h": [idf_curves[rp][6] for rp in return_periods],
    }
)

print("\nIDF Table - Intensity (mm/hr):")
print(display_df)

# Load empirical IDF data for comparison
empirical_idf = pd.read_csv(os.path.join(
    os.path.dirname(__file__), "..", "results", "idf_data.csv"))

# Map durations in our model to the columns in empirical data
duration_mapping = {
    0: "5 mins",
    1: "10 mins", 
    2: "15 mins",
    3: "30 mins",
    4: "60 mins",
    5: "180 mins",
    6: "1440 mins"
}

# Calculate metrics (RMSE, MAE, R2) for each return period

rmse_values = []
mae_values = []
r2_values = []

for rp in return_periods:
    empirical_row = empirical_idf[empirical_idf["Return Period (years)"] == rp].iloc[0]
    
    # Extract values from empirical data and model predictions for this return period
    y_true = []
    y_pred = []
    
    for i in range(len(durations_hours)):
        empirical_col = duration_mapping[i]
        y_true.append(empirical_row[empirical_col])
        y_pred.append(idf_curves[rp][i])
    
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
    plt.plot(durations_minutes, idf_curves[rp], '-', color=colors[i], 
             linewidth=2, label=f"Model T = {rp} years")
    
    # Empirical data (dashed line with markers)
    empirical_row = empirical_idf[empirical_idf["Return Period (years)"] == rp].iloc[0]
    empirical_values = [empirical_row[duration_mapping[j]] for j in range(len(durations_hours))]
    plt.plot(durations_minutes, empirical_values, '--', color=colors[i], 
             marker='o', markersize=5, linewidth=1.5, label=f"Empirical T = {rp} years")

plt.xscale('log')
plt.xlabel('Duration (minutes)', fontsize=12)
plt.ylabel('Intensity (mm/hr)', fontsize=12)
plt.title('IDF Curves Comparison: Neural Network vs Empirical', fontsize=14)
plt.grid(True, which="both", ls="-")

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

# Original IDF curve plot
plt.figure(figsize=(10, 6))
for rp in return_periods:
    plt.plot(durations_hours, idf_curves[rp], marker="o", label=f"{rp}-year")

plt.xlabel("Duration (hours)")
plt.ylabel("Rainfall Intensity (mm/hr)")
plt.title("Intensity-Duration-Frequency (IDF) Curves\nGenerated by Neural Network")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "figures", "neural_network_idf_curves.png"), 
    dpi=300, 
    bbox_inches="tight"
)
