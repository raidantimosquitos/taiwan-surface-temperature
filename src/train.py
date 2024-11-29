import os

import json
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit

from config import CHECKPOINTS_DIR
from src.utils import save_model_checkpoint, create_directory, plot_losses, ToTensor
from src.models import LSTMModel, GRUModel
from src.dataset import TWTemperatureDataset


def train(model, train_loader, cross_val_loader, criterion, optimizer, num_epochs, device, logs_dir):
    print(f'Training with {device} device.')
    model.to(device)
    best_val_loss = float("inf")
    best_model_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")

    # Initialize dictionary to track losses
    loss_history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Cross-validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in cross_val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(cross_val_loader)
        train_loss = running_loss / len(train_loader)
        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]: Training Loss: {train_loss:.4f}; Validation Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model found! Validation Loss: {val_loss:.4f}")
            save_model_checkpoint(model, best_model_path)

    # Save loss history to logs and plot it
    directory_logs = os.path.join(logs_dir, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    create_directory(directory_logs)
    logs_file = os.path.join(directory_logs, "loss_history.json")
    with open(logs_file, "w") as f:
        json.dump(loss_history, f)
    print(f"Loss history saved to {logs_file}")
    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")
    
    return directory_logs

# Testing the code
if __name__ == "__main__":
    # Define constants
    FILEPATH = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/datasets/berkeley-earth-surface-temp-dataset/TaiwanLandTemperaturesByCity.csv"
    CHECKPOINTS_DIR = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/checkpoints"
    LOGS_DIR = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/logs"
    TARGET_COLUMN = "AverageTemperature"
    INPUT_WINDOW = 2
    BATCH_SIZE = 32
    N_SPLITS = 5
    LR = 1e-3
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the dataset
    dataset = TWTemperatureDataset(filepath=FILEPATH, target_column=TARGET_COLUMN, input_window=INPUT_WINDOW, transforms=[ToTensor()])
    in_size = len(dataset.get_feature_names())

    # TimeSeriesSplit for train-crossval-test
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Reserve the last fold for testing
    splits = list(tscv.split(dataset))
    train_indices, cross_val_indices = splits[-2]
    _, test_indices = splits[-1]

    train_data = Subset(dataset, train_indices)
    cross_val_data = Subset(dataset, cross_val_indices)
    test_data = Subset(dataset, test_indices)

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle to preserve temporal order
    crossval_loader = DataLoader(cross_val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model, criterion, optimizer
    model = LSTMModel(input_dim=in_size, hidden_dim=2*in_size, num_layers=2, output_dim=1)
    criterion = torch.nn.MSELoss()  # Using MSE for regression
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # train the model
    directory_logs = train(model, train_loader, crossval_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE, logs_dir=LOGS_DIR)
  
    plot_losses(os.path.join(directory_logs, "loss_history.json"), directory_logs)

    pass