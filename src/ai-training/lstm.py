import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

datapath = os.path.join(
    os.path.dirname(__file__), "..", "results", "historical_intensity.csv"
)

# Load and examine the data
data = pd.read_csv(datapath)
data['date'] = pd.to_datetime(data['date'])
print(f"Data spans from {data['date'].min()} to {data['date'].max()}")
print(f"Total years: {data['year'].nunique()}")

# Define features
intensity_columns = ['30mns', '1h', '3h', '24h']

# Normalize the data to improve training
scaler = StandardScaler()
normalized_intensities = scaler.fit_transform(data[intensity_columns])
data_normalized = data.copy()
data_normalized[intensity_columns] = normalized_intensities

# Create sequences for LSTM training
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 28  # Use two weeks of data to predict the next day
X, y = create_sequences(normalized_intensities, seq_length)

# Split into train and test sets
train_size = int(0.75 * len(X))
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
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
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
hidden_size = 48
num_layers = 2
output_size = len(intensity_columns)
learning_rate = 0.001
num_epochs = 750
batch_size = 8192

# Initialize model
model = RainfallLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")
print("Beginning training...")

# Training loop
for epoch in range(num_epochs):
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
    
    # Print statistics every 10 epochs
    # if (epoch+1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

print('Training complete')

# Evaluate the model
# Process predictions in batches to avoid memory issues
model.eval()
batch_size = 8192  # Adjust based on your memory constraints
all_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size)

all_predictions = []
with torch.no_grad():
    for batch_X, in all_loader:
        batch_X = batch_X.to(device)
        batch_pred = model(batch_X).cpu().numpy()
        all_predictions.append(batch_pred)
    
all_predictions = np.vstack(all_predictions)

# Inverse transform predictions to get actual values
all_predictions_denorm = scaler.inverse_transform(all_predictions)

# Create a DataFrame with predictions and corresponding dates/years
prediction_dates = data['date'][seq_length:].reset_index(drop=True)
prediction_years = data['year'][seq_length:].reset_index(drop=True)

predictions_df = pd.DataFrame(all_predictions_denorm, columns=intensity_columns)
predictions_df['date'] = prediction_dates
predictions_df['year'] = prediction_years

# Calculate annual maximum intensities
annual_max = predictions_df.groupby('year')[intensity_columns].max()

def calculate_return_period_intensities(annual_max_series):
    """Calculate intensities for different return periods using the Weibull plotting position formula"""
    n = len(annual_max_series)
    if n == 0:
        return {}
        
    # Sort in descending order
    sorted_intensities = sorted(annual_max_series, reverse=True)
    
    # Calculate empirical return periods using Weibull formula: T = (n+1)/m
    # where n is the number of years and m is the rank
    ranks = np.arange(1, n+1)
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

# Set up durations in hours for plotting
durations = {
    '30mns': 0.5,
    '1h': 1,
    '3h': 3,
    '24h': 24
}

# Generate IDF curves
plt.figure(figsize=(12, 8))
return_periods = [2, 5, 10, 25, 50, 100]
duration_values = list(durations.values())

for rp in return_periods:
    intensities = [idf_data[duration][rp] for duration in intensity_columns]
    plt.plot(duration_values, intensities, marker='o', linewidth=2, label=f'T = {rp} years')

plt.xlabel('Duration (hours)', fontsize=12)
plt.ylabel('Intensity (mm/hr)', fontsize=12)
plt.title('Intensity-Duration-Frequency (IDF) Curves', fontsize=14)
plt.grid(True, which="both", ls="-")
plt.legend(fontsize=10)

# Add points at standard durations
for duration, hours in durations.items():
    for rp in return_periods:
        plt.scatter(hours, idf_data[duration][rp], color='black', s=30, zorder=5)

plt.tight_layout()
plt.savefig('idf_curves.png', dpi=300)

# Create a table of IDF values
idf_table = pd.DataFrame(index=return_periods, columns=intensity_columns)
for rp in return_periods:
    for duration in intensity_columns:
        idf_table.loc[rp, duration] = round(idf_data[duration][rp], 2)

print("\nIDF Table (Intensity in mm/hr):")
print(idf_table)