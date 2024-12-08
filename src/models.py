import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import TWTemperatureDataset
from src.utils import ToTensor, create_sequences

class LSTMModel(nn.Module):
    """
    LSTM-based model for time series prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
        # Take the last time step's output
        last_hidden_state = lstm_out[:, -1, :]  # shape: (batch_size, hidden_dim)

        # Pass through the fully connected layer
        output = self.fc(last_hidden_state)  # shape: (batch_size, output_dim)
        return output

class GRUModel(nn.Module):
    """
    GRU-based model for time series prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden_state = gru_out[:, -1, :]
        out = self.fc(last_hidden_state)
        return out

# Testing the models
if __name__ == "__main__":
    FILEPATH = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/datasets/taiwan_clean_dataset.csv"
    TARGET_COLUMN = "AverageTemperature"
    # Hyperparameters
    input_dim = 19        # Number of features
    hidden_dim = 64       # LSTM hidden state size
    num_layers = 2        # Number of LSTM layers
    output_dim = 1        # Single output (AverageTemperature)
    batch_size = 32
    num_epochs = 2
    learning_rate = 1e-3
    dropout = 0.2
    device = 'cpu'
    sequences = 12

    dataset = TWTemperatureDataset(filepath=FILEPATH, target_column=TARGET_COLUMN, transforms=[ToTensor()])
    in_size = len(dataset.get_feature_names())

    x_sequences, y_sequences = create_sequences(dataset, sequences)
    # Print shapes for confirmation
    print(f"Input sequences shape: {x_sequences.shape}")  # Expected: (n_sequences, seq_length, n_features)
    print(f"Target shape: {y_sequences.shape}")  # Expected: (n_sequences, 1)

    sequenced_dataset = TensorDataset(x_sequences, y_sequences)
    train_loader = DataLoader(sequenced_dataset, batch_size, shuffle=False)

    # Instantiate the model
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    criterion = nn.MSELoss()  # Loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')