import os

import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

class AddLagFeatures:
    def __init__(self, lag: int, target_columns: list):
        """
        Add lag features for specific target columns.
        :param lag: Number of lags to generate
        :param target_columns: List of column names to apply lag features
        """
        self.lag = lag
        self.target_columns = target_columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.target_columns:
            for i in range(1, self.lag + 1):
                df[f"{col}_lag_{i}"] = df[col].shift(i)
        return df.dropna()

class ToTensor:
    def __call__(self, x, y):
        """
        Converts data to PyTorch tensors.
        :param x: Input features (NumPy array or DataFrame slice)
        :param y: Target value
        :return: Tuple of tensors
        """
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.float32)
        return x_tensor, y_tensor

def create_directory(path):
    """
    Creates a directory if it doesn't exist.
    
    Parameters:
    - path (str): Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_model_checkpoint(model, filepath):
    """
    Save the model checkpoint.
    
    Parameters:
    - model (torch.nn.Module): The PyTorch model to save
    - filepath (str): Path to save the model checkpoint
    """
    torch.save(model.state_dict(), filepath)

def load_model_checkpoint(model, filepath):
    """
    Load a model checkpoint.
    
    Parameters:
    - model (torch.nn.Module): The PyTorch model to load the weights into
    - filepath (str): Path to the saved model checkpoint
    """
    model.load_state_dict(torch.load(filepath))

def plot_losses(json_file_path, output_dir):
    """
    Reads loss history from a JSON file and generates a plot of training and validation losses.

    Args:
        json_file_path (str): Path to the JSON file containing loss history.
        output_dir (str): Directory to save the loss plot.
    """
    # Load the loss history from the JSON file
    with open(json_file_path, "r") as f:
        loss_history = json.load(f)

    # Generate the plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history["train_loss"], label="Training Loss", marker="o")
    plt.plot(loss_history["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid()

    # Save the plot to the output directory
    plot_path = os.path.join(output_dir, "loss_plot.png")
    if plot_path:
        plt.savefig(plot_path)
        print(f"Loss Plot saved to {plot_path}")
    else:
        plt.show()

# Testing the functions
if __name__ == '__main__':
    pass