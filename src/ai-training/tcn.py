import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

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
# data["24h"] = data["24h"] / 0.5  # Convert daily to hourly

# Normalize the data to improve training
scaler = MinMaxScaler()
normalized_intensities = scaler.fit_transform(data[intensity_columns])
data_normalized = data.copy()
data_normalized[intensity_columns] = normalized_intensities


# Create sequences for TCN training
def create_sequences(data_series, seq_length):
    x, y = [], []
    for i in range(len(data_series) - seq_length):
        x.append(data_series[i : i + seq_length])
        y.append(data_series[i + seq_length])
    return np.array(x), np.array(y)


# Configuration dictionary for each duration
duration_config = {
    "5mns": {
        "seq_length": 100,
        "lr": 0.001,
        "num_epochs": 200,
        "batch_size": 4096,
        "num_channels": [32, 64],
    },
    "10mns": {
        "seq_length": 100,
        "lr": 0.001,
        "num_epochs": 200,
        "batch_size": 4096,
        "num_channels": [32, 64],
    },
    "15mns": {
        "seq_length": 100,
        "lr": 0.001,
        "num_epochs": 250,
        "batch_size": 4096,
        "num_channels": [32, 64],
    },
    "30mns": {
        "seq_length": 96,
        "lr": 0.001,
        "num_epochs": 200,
        "batch_size": 16,
        "num_channels": [32, 64],
    },
    "1h": {
        "seq_length": 92,
        "lr": 0.001,
        "num_epochs": 200,
        "batch_size": 16,
        "num_channels": [32, 64],
    },
    "3h": {
        "seq_length": 48,
        "lr": 0.001,
        "num_epochs": 180,
        "batch_size": 16,
        "num_channels": [32, 64, 128],
    },
    "24h": {
        "seq_length": 64,
        "lr": 0.001,
        "num_epochs": 200,
        "batch_size": 16,
        "num_channels": [32, 64, 128],
    },
}


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class RainfallTCN(nn.Module):
    def __init__(
        self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2
    ):
        super(RainfallTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.temporal_blocks = nn.Sequential(*layers)

        # Add self-attention layer
        hidden_dim = num_channels[-1]
        num_heads = min(4, hidden_dim // 4)  # Ensure divisible by head count
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Output layer
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # TCN expects input: [batch, channels, seq_len]
        features = self.temporal_blocks(x)

        # Transpose to [batch, seq_len, channels] for attention
        features = features.transpose(1, 2)

        # Apply self-attention
        attn_output, _ = self.self_attention(features, features, features)
        features = features + attn_output  # Residual connection
        features = self.norm(features)

        # Use the last time step for prediction
        out = features[:, -1, :]
        out = self.linear(out)
        return out


# Train individual models for each duration
models = {}
predictions = {}
annual_max_dict = {}
checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx, duration in enumerate(intensity_columns):
    print(f"\nTraining model for {duration}...")

    # Get duration-specific configuration
    config = duration_config[duration]
    seq_length = config["seq_length"]
    learning_rate = config["lr"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_channels = config["num_channels"]

    print(
        f"Using config: seq_length={seq_length}, lr={learning_rate}, epochs={num_epochs}, batch_size={batch_size}, num_channels={num_channels}"
    )

    # Get data for this duration
    # Filter out non-zero values from the original data
    non_zero_mask = data[duration].values > 0
    duration_data = normalized_intensities[non_zero_mask, idx].reshape(-1, 1)

    # Create sequences
    X, y = create_sequences(duration_data, seq_length)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.04, random_state=36683, shuffle=True
    )

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).transpose(1, 2)  # Shape: [batch, 1, seq_len]
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test).transpose(1, 2)
    y_test = torch.FloatTensor(y_test)

    # Create model for this duration (1 input feature, 1 output)
    model = RainfallTCN(
        input_size=1,
        output_size=1,
        num_channels=num_channels,
        kernel_size=3,
        dropout=0.2,
    )
    model.to(device)

    # Setup optimizer and criterion with duration-specific learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.85, patience=10
    )

    # Training setup with duration-specific batch size
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )

    # Training loop with duration-specific epochs
    best_loss = float("inf")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_X = X_test[i : i + batch_size].to(device)
                batch_y = y_test[i : i + batch_size].to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * len(batch_X)

        test_loss /= len(X_test)
        scheduler.step(test_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.6f}, "
                f"Test Loss: {test_loss:.6f} (LR: {optimizer.param_groups[0]['lr']:.6f})"
            )

        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, f"tcn_{duration}_best.pt"),
            )

    # Store the model
    models[duration] = model

    # Generate predictions for this duration
    model.eval()
    with torch.no_grad():
        pred_X = torch.FloatTensor(X).transpose(1, 2).to(device)
        duration_preds = []

        for i in range(0, len(pred_X), batch_size):
            batch_pred = model(pred_X[i : i + batch_size]).cpu().numpy()
            duration_preds.extend(batch_pred)

        predictions[duration] = np.array(duration_preds).flatten()

    # Calculate annual maximum for this duration immediately
    duration_pred = predictions[duration]

    # Get years corresponding to these predictions (using duration-specific seq_length)
    non_zero_mask = data[duration].values > 0
    years = data["year"][non_zero_mask].values[seq_length:]

    # Check if lengths match after accounting for sequence creation
    if len(years) != len(duration_pred):
        print(
            f"Warning: Length mismatch for {duration}. Years: {len(years)}, Predictions: {len(duration_pred)}"
        )
        continue

    # Denormalize predictions for this duration only
    denorm_array = np.zeros((len(duration_pred), len(intensity_columns)))
    denorm_array[:, idx] = duration_pred
    denorm_preds = scaler.inverse_transform(denorm_array)[:, idx]

    # Group by year and find maximum for each year
    year_dict = {}
    for year in np.unique(years):
        year_mask = years == year
        year_dict[year] = np.max(denorm_preds[year_mask])

    annual_max_dict[duration] = year_dict

# Convert to DataFrame with all years
all_years = sorted(set().union(*[set(d.keys()) for d in annual_max_dict.values()]))
annual_max = pd.DataFrame(index=all_years)

for duration, year_values in annual_max_dict.items():
    annual_max[duration] = pd.Series(year_values)


def calculate_return_period_intensities(annual_max_series):
    """Calculate intensities for different return periods using the Weibull plotting position formula"""
    n = len(annual_max_series)
    if n == 0:
        return {}

    # Filter out any potential NaN or negative values
    valid_intensities = [x for x in annual_max_series if np.isfinite(x) and x > 0]
    if not valid_intensities:
        # If no valid data, return zeros for all return periods
        return {rp: 0.0 for rp in [2, 5, 10, 25, 50, 100]}

    # Sort in descending order
    sorted_intensities = sorted(valid_intensities, reverse=True)

    n = len(sorted_intensities)  # Recalculate n after filtering

    # Calculate empirical return periods using Weibull formula: T = (n+1)/m
    # where n is the number of years and m is the rank
    ranks = np.arange(1, n + 1)
    empirical_return_periods = (n + 1) / ranks

    # Interpolate to get intensities for desired return periods
    return_periods = [2, 5, 10, 25, 50, 100]
    result = {}

    # Add small epsilon to avoid log(0)
    epsilon = 1e-6

    # Ensure all values are positive before taking log
    intensity_array = np.maximum(np.array(sorted_intensities), epsilon)
    log_intensities = np.log(intensity_array)
    log_rp = np.log(empirical_return_periods)

    try:
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
            log_intensity = coef[0] * np.log(max(rp, epsilon)) + coef[1]
            model_value = np.exp(log_intensity)

            # Always start with the model value
            result[rp] = model_value

            # For return periods within the observed range, blend model prediction with interpolation
            if min(empirical_return_periods) <= rp <= max(empirical_return_periods):
                interp_value = np.interp(rp, sorted_rp, sorted_intensity)
                # Blend model and interpolation with varying weights based on return period
                # Use more model influence for longer return periods
                weight = min(
                    0.5 + (rp / 200), 0.9
                )  # Weight increases with return period
                result[rp] = weight * model_value + (1 - weight) * interp_value

    except (ValueError, np.linalg.LinAlgError):
        # Fallback for any numerical errors - use simple linear interpolation
        for rp in return_periods:
            if rp <= max(empirical_return_periods):
                # Direct interpolation for return periods in range
                result[rp] = np.interp(rp, sorted_rp, sorted_intensity)
            else:
                # Simple extrapolation for higher return periods
                result[rp] = sorted_intensities[0] * (rp / sorted_rp[-1]) ** 0.2

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

duration_values = {
    "5mns": 5,
    "10mns": 10,
    "15mns": 15,
    "30mns": 30,
    "1h": 60,
    "3h": 180,
    "24h": 1440,
}

return_periods = [2, 5, 10, 25, 50, 100]

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
        empirical_value = empirical_row[empirical_col]
        model_value = idf_data[model_col][rp]

        # Only include valid pairs of values
        if np.isfinite(empirical_value) and np.isfinite(model_value):
            y_true.append(empirical_value)
            y_pred.append(model_value)

    # Check if we have enough valid data to calculate metrics
    if len(y_true) > 1:
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    else:
        # Default values if not enough data
        rmse = np.nan
        mae = np.nan
        r2 = np.nan

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
        duration_values.values(),
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
        duration_values.values(),
        empirical_values,
        "--",
        color=colors[i],
        marker="o",
        markersize=5,
        linewidth=1.5,
        label=f"Empirical T = {rp} years",
    )

plt.xlabel("Duration (minutes)", fontsize=12)
plt.ylabel("Intensity (mm/hr)", fontsize=12)
plt.title("IDF Curves Comparison: TCN Model vs Empirical", fontsize=14)
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
    os.path.join(os.path.dirname(__file__), "..", "figures", "idf_comparison_tcn.png"),
    dpi=300,
)

# Create a table of IDF values
idf_table = pd.DataFrame(index=return_periods, columns=intensity_columns)
for rp in return_periods:
    for duration in intensity_columns:
        idf_table.loc[rp, duration] = round(idf_data[duration][rp], 2)

print("\nIDF Table (Intensity in mm/hr):")
print(idf_table)
