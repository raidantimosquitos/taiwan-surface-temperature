import os
import torch
from torch.utils.data import DataLoader
from dataset import TWTemperatureDataset
from models import LSTMModel
from utils import ToTensor, load_model_checkpoint

# Define the evaluation class (optional for extensibility)
class ModelEvaluation:
    def __init__(self, checkpoint_dir, logs_dir, dataset_path, target_column, input_window, device):
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.input_window = input_window
        self.device = device

    def load_test_data(self):
        """
        Prepares the test dataset and dataloader.
        """
        print("Loading dataset...")
        dataset = TWTemperatureDataset(
            filepath=self.dataset_path,
            target_column=self.target_column,
            input_window=self.input_window,
            transforms=[ToTensor()],
        )
        test_data = dataset[len(dataset) * 4 // 5 :]  # Last 20% as test set
        self.feature_names = dataset.get_feature_names()
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        return test_loader

    def load_model(self, model_class, best_model_path):
        """
        Loads the best model from the checkpoint directory.
        """
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        model = load_model_checkpoint(model_class, best_model_path)
        print("Model loaded successfully.")
        return model

    def run_evaluation(self, model, test_loader):
        """
        Runs evaluation and computes metrics.
        """
        print("Evaluating model...")
        predictions, true_values = evaluate_model(model, test_loader, self.device)
        metrics = evaluate_forecast(predictions, true_values)
        print(f"Test Metrics: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

        print("Generating forecast plot...")
        plot_forecast(predictions, true_values, self.logs_dir)
        print(f"Forecast plot saved in {self.logs_dir}")

# Main evaluation script
if __name__ == "__main__":
    # Paths and settings
    CHECKPOINTS_DIR = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/checkpoints"
    LOGS_DIR = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/logs"
    FILEPATH = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/datasets/berkeley-earth-surface-temp-dataset/TaiwanLandTemperaturesByCity.csv"
    TARGET_COLUMN = "AverageTemperature"
    INPUT_WINDOW = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model hyperparameters (ensure they match training)
    INPUT_DIM = 6  # Adjust to match dataset's feature count
    HIDDEN_DIM = 2 * INPUT_DIM
    NUM_LAYERS = 2
    OUTPUT_DIM = 1

    # Initialize evaluation object
    evaluator = ModelEvaluation(
        checkpoint_dir=CHECKPOINTS_DIR,
        logs_dir=LOGS_DIR,
        dataset_path=FILEPATH,
        target_column=TARGET_COLUMN,
        input_window=INPUT_WINDOW,
        device=DEVICE,
    )

    # Load test data
    test_loader = evaluator.load_test_data()

    # Load the trained model
    model = evaluator.load_model(
        LSTMModel, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM
    )

    # Run evaluation
    evaluator.run_evaluation(model, test_loader)
