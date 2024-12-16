import os
from dateutil.relativedelta import relativedelta

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

class ForecastFutureValues:
    """
    Class for evaluating a trained model on a given dataset.
    """
    def __init__(self, model, checkpoint_path, device='cpu'):
        """
        Initialize the ForecastFutureValues object.

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

    def evaluate(self, test_loader, test_dates, mean_y=None, std_y=None):
        """
        Evaluate the model on a test dataset.

        Args:
            test_loader (DataLoader): DataLoader for the test dataset.
            test_dates (list): List of datetime objects corresponding to the test data.
            mean_y (float): Mean of the target variable from training (for denormalization).
            std_y (float): Std of the target variable from training (for denormalization).

        Returns:
            dict: Dictionary with evaluation metrics (e.g., MSE, MAE).
        """
        print(f'Evaluating with {self.device} device.')

        criterion = torch.nn.MSELoss()
        rmse_loss = 0.0
        r2 = 0.0
        predictions, targets = [], []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(features)

                # Collect predictions and targets for further analysis
                predictions.append(outputs.cpu())
                targets.append(labels.cpu())

                # Compute loss and R2 score
                rmse_loss += np.sqrt(criterion(outputs.cpu(), labels.cpu()).item())
                r2 += r2_score(outputs.cpu(), labels.cpu())

        rmse_loss /= len(test_loader)
        r2 /= len(test_loader)

        # Convert to numpy array
        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        
        if mean_y is not None and std_y is not None:
            predictions = [(val * std_y + mean_y) for val in predictions]
            targets = [(val * std_y + mean_y) for val in targets]

        print(f"Evaluation Complete. RMSE Loss: {rmse_loss:.4f}, R^2 score: {r2:.4f}")

        return {"rmse": rmse_loss, "r2_score": r2, "predictions": predictions, "targets": targets, "dates": test_dates}

    def forecast(self, test_data, test_dates, num_forecast_steps=12, mean_y=None, std_y=None):
        """
        Forecast future values using the trained model.

        Args:
            test_data (TensorDataset): The sequenced test dataset of shape [n_samples x n_features x samp_per_sequence].
            test_dates (list): List of datetime objects corresponding to the test data.
            num_forecast_steps (int): Number of steps to forecast.
            mean_y (float): Mean of the target variable from training (for denormalization).
            std_y (float): Std of the target variable from training (for denormalization).

        Returns:
            dict: Forecasted values and corresponding future dates (as strings).
        """
        print("Forecasting future values...")

        # Extract the last sequence from the test dataset as the starting input
        current_sequence = test_data.tensors[0][-1].unsqueeze(0).to(self.device)  # Shape: (1, seq_len, input_dim)
        current_target = test_data.tensors[1][-36:].unsqueeze(0).to(self.device)

        # Initialize forecasted values and future dates
        forecasted_values = []
        future_dates = []

        # Get the last known date as the starting point for future predictions
        initial_date = test_dates[-1]
        current_input = current_sequence.clone()
        current_target = current_target.clone()
        with torch.no_grad():
            for step in range(num_forecast_steps):
                # Make a prediction for the next step
                predicted_value = self.model(current_input).cpu()
                np_pred = predicted_value.numpy()[0,0]

                # Denormalize the predicted value (if needed)
                if mean_y is not None and std_y is not None:
                    np_pred = np_pred * std_y + mean_y

                # Add the forecasted value and corresponding future date
                forecasted_values.append(np_pred)
                future_date = initial_date + relativedelta(months= step + 1)
                future_dates.append(future_date.replace(day=1))  # Ensure it's the first day of the month

        # Return the forecasted values and future dates
        return {"forecasted_values": forecasted_values, "future_dates": future_dates}