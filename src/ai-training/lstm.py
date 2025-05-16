import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(36683)
torch.manual_seed(36683)

datapath = os.path.join(
    os.path.dirname(__file__), "..", "results", "historical_intensity.csv"
)

# Load and examine the data
data = pd.read_csv(datapath)
data["date"] = pd.to_datetime(data["date"])
print(f"Data spans from {data['date'].min()} to {data['date'].max()}")
print(f"Total years: {data['year'].nunique()}")

# Define features
intensity_columns = ["5mns", "10mns", "15mns", "30mns", "1h", "3h", "24h"]

# Normalize the data to improve training
scaler = MinMaxScaler()
normalized_intensities = scaler.fit_transform(data[intensity_columns])
data_normalized = data.copy()
data_normalized[intensity_columns] = normalized_intensities


# Create sequences for LSTM training
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)


seq_length = 24  # Use two weeks of data to predict the next day
X, y = create_sequences(normalized_intensities, seq_length)

# Split into train and test sets
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)


class RainfallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RainfallLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# Set hyperparameters
input_size = len(intensity_columns)  # Number of features
hidden_size = 24
num_layers = 2
output_size = len(intensity_columns)
learning_rate = 0.001
num_epochs = 200
batch_size = 65536

# Initialize model
model = RainfallLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=False
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")
print("Beginning training...")

# Training loop
# Create checkpoint directory if it doesn't exist
checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

best_loss = float("inf")

# Create learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# Check if final checkpoint or best model exists
final_checkpoint_path = os.path.join(checkpoint_dir, f"idf-lstm-epoch_{num_epochs}.pt")
best_model_path = os.path.join(checkpoint_dir, "idf-lstm_best.pt")

if os.path.exists(final_checkpoint_path):
    print("Training already completed. Loading best model.")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    # Check for the latest checkpoint
    start_epoch = 0
    checkpoints = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("idf-lstm-epoch_") and f.endswith(".pt")
    ]

    if checkpoints:
        # Extract epoch numbers from checkpoint filenames and find the latest
        latest_epoch = max([int(f.split("_")[1].split(".")[0]) for f in checkpoints])
        latest_checkpoint_path = os.path.join(
            checkpoint_dir, f"idf-lstm-epoch_{latest_epoch}.pt"
        )

        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting training from scratch")

    # Create test data loader for evaluation
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        # Evaluate test loss
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                test_loss += criterion(outputs, batch_y).item()

        test_loss /= len(test_loader)

        # Update learning rate based on test loss
        scheduler.step(test_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Save checkpoint every 50 epochs and at the end
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"idf-lstm-epoch_{epoch + 1}.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": epoch_loss,
                    "test_loss": test_loss,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model based on test loss
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_path = os.path.join(checkpoint_dir, "idf-lstm_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": epoch_loss,
                    "test_loss": best_loss,
                },
                best_model_path,
            )
            print(f"Best model saved with test loss: {best_loss:.4f}")

    print("Training complete")

# Evaluate the model
# Process predictions in batches to avoid memory issues
model.eval()
batch_size = 65536  # Adjust based on your memory constraints
all_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size)

all_predictions = []
with torch.no_grad():
    for (batch_X,) in all_loader:
        batch_X = batch_X.to(device)
        batch_pred = model(batch_X).cpu().numpy()
        all_predictions.append(batch_pred)

all_predictions = np.vstack(all_predictions)

# Inverse transform predictions to get actual values
all_predictions_denorm = scaler.inverse_transform(all_predictions)

# Create a DataFrame with predictions and corresponding dates/years
prediction_dates = data["date"][seq_length:].reset_index(drop=True)
prediction_years = data["year"][seq_length:].reset_index(drop=True)

predictions_df = pd.DataFrame(all_predictions_denorm, columns=intensity_columns)
predictions_df["date"] = prediction_dates
predictions_df["year"] = prediction_years

# Calculate annual maximum intensities
annual_max = predictions_df.groupby("year")[intensity_columns].max()


def calculate_return_period_intensities(annual_max_series):
    """Calculate intensities for different return periods using the Weibull plotting position formula"""
    n = len(annual_max_series)
    if n == 0:
        return {}

    # Sort in descending order
    sorted_intensities = sorted(annual_max_series, reverse=True)

    # Calculate empirical return periods using Weibull formula: T = (n+1)/m
    # where n is the number of years and m is the rank
    ranks = np.arange(1, n + 1)
    empirical_return_periods = (n + 1) / ranks

    # Interpolate to get intensities for desired return periods
    return_periods = [2, 5, 10, 25, 50, 100]
    result = {}

    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    log_intensities = np.log(np.array(sorted_intensities) + epsilon)
    log_rp = np.log(empirical_return_periods + epsilon)

    # Use all data points for a more robust extrapolation model
    # Fit a polynomial of degree 1 (line) to log-log data
    coef = np.polyfit(log_rp, log_intensities, 1)

    # Sort the empirical data for proper interpolation
    interp_order = np.argsort(empirical_return_periods)
    sorted_rp = empirical_return_periods[interp_order]
    sorted_intensity = np.array(sorted_intensities)[interp_order]

    # Create results with a proper mix of model and interpolation
    for rp in return_periods:
        # Use the fitted model for prediction
        log_intensity = coef[0] * np.log(rp + epsilon) + coef[1]
        model_value = np.exp(log_intensity) - epsilon

        # Always start with the model value
        result[rp] = model_value

        # For return periods within the observed range, blend model prediction with interpolation
        if min(empirical_return_periods) <= rp <= max(empirical_return_periods):
            interp_value = np.interp(rp, sorted_rp, sorted_intensity)
            # Blend model and interpolation with varying weights based on return period
            # Use more model influence for longer return periods
            weight = min(0.5 + (rp / 200), 0.9)  # Weight increases with return period
            result[rp] = weight * model_value + (1 - weight) * interp_value

    return result


# Calculate IDF data for each duration
idf_data = {}
for duration in intensity_columns:
    idf_data[duration] = calculate_return_period_intensities(annual_max[duration])

# Load empirical IDF data for comparison
empirical_idf = pd.read_csv(
    "/home/edargham/devenv/thesis-ai-idf/src/results/empirical_idf_data.csv"
)

# Map model durations to empirical durations
duration_mapping = {
    "5mns": "5 mins",
    "10mns": "10 mins",
    "15mns": "15 mins",
    "30mns": "30 mins",
    "1h": "60 mins",
    "3h": "180 mins",
    "24h": "1440 mins",
}


# Set up durations in hours for plotting
durations = {
    "5mns": 5,  # 5 minutes
    "10mns": 10,  # 10 minutes
    "15mns": 15,  # 15 minutes
    "30mns": 30,  # 30 minutes
    "1h": 60,  # 60 minutes
    "3h": 180,  # 180 minutes
    "24h": 1440,  # 1440 minutes
}

# Generate IDF curves
plt.figure(figsize=(12, 8))
return_periods = [2, 5, 10, 25, 50, 100]
duration_values = list(durations.values())

# Calculate metrics (RMSE, MAE, R2) for each return period
rmse_values = []
mae_values = []
r2_values = []

for rp in return_periods:
    empirical_row = empirical_idf[empirical_idf["Return Period (years)"] == rp].iloc[0]

    # Extract values from empirical data and model predictions for this return period
    y_true = []
    y_pred = []

    for model_col in intensity_columns:
        empirical_col = duration_mapping[model_col]
        y_true.append(empirical_row[empirical_col])
        y_pred.append(idf_data[model_col][rp])

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    rmse_values.append(rmse)
    mae_values.append(mae)
    r2_values.append(r2)

# Display metrics
metrics_df = pd.DataFrame(
    {
        "Return Period": return_periods,
        "RMSE": [round(x, 4) for x in rmse_values],
        "MAE": [round(x, 4) for x in mae_values],
        "R2": [round(x, 4) for x in r2_values],
    }
)
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
colors = ["blue", "green", "red", "purple", "orange", "brown"]

# Plot both model predictions and empirical data for comparison
for i, rp in enumerate(return_periods):
    # Model prediction (solid line)
    model_intensities = [idf_data[model_col][rp] for model_col in intensity_columns]
    plt.plot(
        duration_values,
        model_intensities,
        "-",
        color=colors[i],
        linewidth=2,
        label=f"Model T = {rp} years",
    )

    # Empirical data (dashed line with markers)
    empirical_row = empirical_idf[empirical_idf["Return Period (years)"] == rp].iloc[0]
    empirical_values = [
        empirical_row[duration_mapping[model_col]] for model_col in intensity_columns
    ]
    plt.plot(
        duration_values,
        empirical_values,
        "--",
        color=colors[i],
        marker="o",
        markersize=5,
        linewidth=1.5,
        label=f"Empirical T = {rp} years",
    )

plt.xscale("log")
plt.xlabel("Duration (minutes)", fontsize=12)
plt.ylabel("Intensity (mm/hr)", fontsize=12)
plt.title("IDF Curves Comparison: Model vs Empirical", fontsize=14)
plt.grid(True, which="both", ls="-")

# Add metrics as text
plt.text(
    0.02,
    0.98,
    f"RMSE: {overall_rmse:.4f}\nMAE: {overall_mae:.4f}\nR²: {overall_r2:.4f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Adjust legend to avoid crowding
plt.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "figures", "idf_comparison_lstm.png"),
    dpi=300,
)
