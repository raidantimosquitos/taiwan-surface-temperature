import os
import json
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import save_model_checkpoint, create_directory, plot_losses, ToTensor
from models import LSTMModel, GRUModel
from dataset import TWTemperatureDataset


def train(model, train_loader, cross_val_loader, criterion, optimizer, num_epochs, device, logs_dir):
    model.to(device)
    best_val_loss = float("inf")
    best_model_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")

    # Initialize dictionary to track losses
    loss_history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Cross-validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in cross_val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
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

    # Save loss history to logs
    logs_file = os.path.join(logs_dir, "loss_history.json")
    with open(logs_file, "w") as f:
        json.dump(loss_history, f)
    print(f"Loss history saved to {logs_file}")
    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

# Testing the code
if __name__ == "__main__":
    # Define constants
    FILEPATH = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/datasets/berkeley-earth-surface-temp-dataset/TaiwanLandTemperaturesByCity.csv"
    CHECKPOINTS_DIR = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/checkpoints"
    LOGS_DIR = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/logs"
    TARGET_COLUMN = "AverageTemperature"
    INPUT_WINDOW = 2
    BATCH_SIZE = 32
    LR = 1e-3
    NUM_EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Training with {DEVICE} device.')

    # Create the dataset
    dataset = TWTemperatureDataset(filepath=FILEPATH, target_column=TARGET_COLUMN, input_window=INPUT_WINDOW, transforms=[ToTensor()])
    in_size = len(dataset.get_feature_names())

    # Split dataset into train-crossval-test
    train_data, temp_data = train_test_split(dataset, test_size=0.4, random_state=1234)
    val_data, test_data = train_test_split(temp_data, test_size=0.2, random_state=1234)

    # Create PyTorch dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    crossval_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    # Model setup
    model = LSTMModel(input_dim=in_size, hidden_dim=2*in_size, num_layers=2, output_dim = 1)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # train the model
    directory_logs = os.path.join(LOGS_DIR, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    create_directory(directory_logs)
    train(model, train_loader, crossval_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE, logs_dir=directory_logs)
    plot_losses(os.path.join(directory_logs, "loss_history.json"), directory_logs)

    pass