import os

import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

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
    
def create_sequences(dataset, seq_length):
    """
    Create sequences and corresponding targets from a PyTorch dataset.
    
    :param dataset: A PyTorch Dataset object with features and targets (already as tensors).
    :param seq_length: The length of each sequence.
    :return: Tensors for sequences (x) and targets (y).
    """
    sequences = []
    targets = []

    # Iterate through the dataset to extract sequences
    for i in range(len(dataset) - seq_length):
        # Extract sequence of features (stacked along sequence dimension)
        x_seq = torch.stack([dataset[j][0] for j in range(i, i + seq_length)])
        # Extract target for the sequence (single target at the end of the sequence)
        y_target = dataset[i + seq_length][1]

        sequences.append(x_seq)
        targets.append(y_target)

    # Convert lists to tensors (for efficient batching)
    sequences = torch.stack(sequences)
    targets = torch.stack(targets)

    return sequences, targets


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
    plt.plot(loss_history["train_loss"], label="Training Loss")
    plt.plot(loss_history["val_loss"], label="Validation Loss")
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