import os

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.models import LSTMModel
from src.dataset import TWTemperatureDataset
from src.utils import ToTensor

class ModelEvaluation:
    """
    Class for evaluating a trained model on a given dataset.
    """
    def __init__(self, model, checkpoint_path, device='cpu'):
        """
        Initialize the ModelEvaluation object.

        Args:
            model (nn.Module): The trained PyTorch model to be evaluated.
            checkpoint_path (str): Path to the model checkpoint.
            device (str): Device to perform evaluation ('cpu' or 'cuda').
        """
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._load_model()

    def _load_model(self):
        """Load the best-performing model from the checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {self.checkpoint_path}")

    def evaluate(self, test_loader, test_dates):
        """
        Evaluate the model on a test dataset.

        Args:
            test_loader (DataLoader): DataLoader for the test dataset.
            test_dates (list): List of datetime objects corresponding to the test data.

        Returns:
            dict: Dictionary with evaluation metrics (e.g., MSE, MAE).
        """
        print(f'Evaluating with {self.device} device.')

        criterion = torch.nn.MSELoss()
        mse_loss = 0.0
        predictions, targets = [], []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(features)

                # Collect predictions and targets for further analysis
                predictions.append(outputs.cpu())
                targets.append(labels.cpu())

                # Compute loss
                mse_loss += criterion(outputs, labels).item()

        mse_loss /= len(test_loader)

        # Convert to tensors
        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

        print(f"Evaluation Complete. MSE Loss: {mse_loss:.4f}")

        return {"mse": mse_loss, "predictions": predictions, "targets": targets, "dates": test_dates}

    def plot_results(self, predictions, targets, dates, save_path=None):
        """
        Plot predictions vs. ground truth with datetime on x-axis.

        Args:
            predictions (numpy.ndarray): Predicted values from the model.
            targets (numpy.ndarray): Ground truth values.
            dates (list): List of datetime objects for the x-axis.
            save_path (str, optional): Path to save the plot. If None, the plot is shown.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(dates, targets, label='Ground Truth', color='blue')
        plt.plot(dates, predictions, label='Predictions', color='orange', alpha=0.7)

        plt.title('Model Predictions vs Ground Truth')
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.xticks(rotation=45)

        # Format x-axis to show dates properly
        # Format the x-axis to display only years (or customize as needed)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Show every year
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as Year (YYYY)

        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

# Main evaluation script
if __name__ == "__main__":
    # Paths and constants
    FILEPATH = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/datasets/berkeley-earth-surface-temp-dataset/TaiwanLandTemperaturesByCity.csv"
    CHECKPOINTS_DIR = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
    TARGET_COLUMN = "AverageTemperature"
    INPUT_WINDOW = 2
    N_SPLITS = 5
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset and create test DataLoader
    dataset = TWTemperatureDataset(
        filepath=FILEPATH,
        target_column=TARGET_COLUMN,
        input_window=INPUT_WINDOW,
        transforms=[ToTensor()]
    )
    in_size = len(dataset.get_feature_names())

    # TimeSeriesSplit for train-crossval-test
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Reserve the last fold for testing
    splits = list(tscv.split(dataset))
    _, test_indices = splits[-1]

    test_data = Subset(dataset, test_indices)

    # Data loaders
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle to preserve temporal order
    dates = dataset.get_dates()
    test_dates = [dates[x] for x in test_indices]

    # Initialize and evaluate the model
    model = LSTMModel(input_dim= in_size, hidden_dim=2*in_size, num_layers=2, output_dim=1)
    evaluator = ModelEvaluation(model=model, checkpoint_path=BEST_MODEL_PATH, device=DEVICE)
    evaluation_results = evaluator.evaluate(test_loader, test_dates)

    # Plot results
    plot_path = os.path.join(CHECKPOINTS_DIR, "predictions_vs_ground_truth.png")
    evaluator.plot_results(
        predictions=evaluation_results["predictions"],
        targets=evaluation_results["targets"],
        dates=evaluation_results["dates"],
        save_path=plot_path
    )
