import os

import torch
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

        print(f"Evaluation Complete. RMSE Loss: {rmse_loss:.4f}, R^2 score: {r2:.4f}")

        return {"rmse": rmse_loss, "r2_score": r2, "predictions": predictions, "targets": targets, "dates": test_dates}

    def forecast(self, test_data, test_dates, num_forecast_steps=12):
        """
        Forecast future values using the trained model.

        Args:
            test_data (TensorDataset): The test dataset (features and targets).
            test_dates (list): List of date strings corresponding to the test data.
            num_forecast_steps (int): Number of steps to forecast.

        Returns:
            dict: Forecasted values and corresponding future dates (as strings).
        """
        print("Forecasting future values...")

        # Extract the last sequence from test data
        sequence_to_forecast = test_data.tensors[0][-1].cpu().numpy()  # Last sequence in test data
        forecasted_values = []

        with torch.no_grad():
            for step in range(num_forecast_steps):
                # Prepare the input sequence tensor
                sequence_tensor = torch.as_tensor(sequence_to_forecast).unsqueeze(0).float().to(self.device)
                predicted_value = self.model(sequence_tensor).cpu().numpy()[0, 0]

                forecasted_values.append(predicted_value)

                # Update sequence: Shift features and add prediction as the new target
                sequence_to_forecast = np.roll(sequence_to_forecast, shift=-1, axis=0)
                sequence_to_forecast[-1, -1] = predicted_value  # Update the target value

        forecasted_values = (forecasted_values - np.mean(forecasted_values))/np.std(forecasted_values)
        # Generate future dates
        last_date = test_dates[-2]
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_forecast_steps + 1)]

        return {"forecasted_values": forecasted_values, "future_dates": future_dates}