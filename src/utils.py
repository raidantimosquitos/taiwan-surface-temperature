import os
import shutil
import torch
import numpy as np

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