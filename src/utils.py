import os
import shutil
import torch
import numpy as np
import pandas as pd

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
    """
        Converts data to PyTorch tensors.
        :param x: Input features (NumPy array or DataFrame slice)
        :param y: Target value
        :return: Tuple of tensors
    """
    def __call__(self, x, y):
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

def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy of model predictions.
    
    Parameters:
    - predictions (torch.Tensor): Model predictions
    - targets (torch.Tensor): True labels
    
    Returns:
    - float: Accuracy of predictions
    """
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

# Testing the functions
if __name__ == '__main__':
    pass