import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


class LSTMPredictor(nn.Module):
    def __init__(
        self, num_inputs: int, num_hidden: int, num_outputs: int, device: str = "cuda"
    ):
        super(LSTMPredictor, self).__init__()
        self.device = device

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.lstm = nn.LSTM(self.num_inputs, self.num_hidden, batch_first=True)
        self.fc = nn.Linear(self.num_hidden, self.num_outputs)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #h0 = torch.zeros(32, self.num_inputs, x.size(0), self.num_hidden).to(self.device)
        lstm_out, _ = self.lstm(x)#, h0)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

    def fit(self, train_loader, val_loader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs=100):
        for epoch in range(num_epochs):
            # Set the model to training mode
            self.train()
            train_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader)
            
            self.eval()  # Set the model to evaluation mode
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    window_size = 48
    stride = 1
    num_epochs = 100
    batch_size = 64
    output_size = 1

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "historical_intensity.csv"
    )

    df = pd.read_csv(data_path)

    # Shift the 24hr readings one day into the future
    # Assuming each row represents a 30-minute interval, so a day is 48 rows
    df["24h"] = df["24h"].shift(48)

    # After shifting, the last 48 rows will have NaN values
    # Replace those NaN values with zeros instead of dropping rows
    df.fillna({'24h': 0}, inplace=True)
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.drop(columns=["year", "date"]))

    # Calculate the number of windows
    num_windows = (len(df_scaled) - window_size - output_size) // stride + 1
    feature_count = df_scaled.shape[1]

    # Pre-allocate arrays with the correct shape
    x_data = np.zeros((num_windows, window_size, feature_count))
    y_data = np.zeros((num_windows, feature_count))

    # Fill arrays efficiently in a vectorized way
    for i in range(num_windows):
        start_idx = i * stride
        # Store window_size time steps with all features for each window
        x_data[i] = df_scaled[start_idx:start_idx + window_size]
        # Only store the 24h column (index 0 after shifting) for prediction
        y_data[i] = df_scaled[start_idx + window_size:start_idx + window_size + output_size, :]

    print(f"X data shape: {x_data.shape}")
    print(f"Y data shape: {y_data.shape}")

    # Convert to PyTorch tensors and move to device
    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    
    # Define train/validation split (80% train, 20% validation)
    train_size = int(0.7 * len(x_data))
    
    # Split the data temporally (no shuffling)
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]
    x_val = x_data[train_size:]
    y_val = y_data[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Sanity check
    for inputs, targets in train_loader:
        print(f"Training inputs shape: {inputs.shape}")
        print(f"Training targets shape: {targets.shape}")
        break
        
    for inputs, targets in val_loader:
        print(f"Validation inputs shape: {inputs.shape}")
        print(f"Validation targets shape: {targets.shape}")
        break

    # Initialize the model
    num_inputs = feature_count
    num_hidden = 98
    num_outputs = feature_count
    model = LSTMPredictor(num_inputs, num_hidden, num_outputs, device=device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train the model
    model.fit(train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # Evaluate on validation set


