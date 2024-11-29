import os

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit

from config import FILEPATH, CHECKPOINTS_DIR, LOGS_DIR, TARGET_COLUMN, INPUT_WINDOW, BATCH_SIZE, N_SPLITS, LR, NUM_EPOCHS, DEVICE
from src.dataset import TWTemperatureDataset
from src.utils import ToTensor, plot_losses
from src.train import train
from src.models import LSTMModel
from src.test import ModelEvaluation

def main():
    # Create the dataset
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # train the model
    directory_logs = train(model, train_loader, crossval_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE, logs_dir=LOGS_DIR)
    plot_losses(os.path.join(directory_logs, "loss_history.json"), directory_logs)

     # Data loaders
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle to preserve temporal order
    dates = dataset.get_dates()
    test_dates = [dates[x] for x in test_indices]

    # Initialize and evaluate the model
    best_model_path = os.path.join(CHECKPOINTS_DIR, 'best_model.pth')
    model = LSTMModel(input_dim= in_size, hidden_dim=2*in_size, num_layers=2, output_dim=1)
    evaluator = ModelEvaluation(model=model, checkpoint_path=best_model_path, device=DEVICE)
    evaluation_results = evaluator.evaluate(test_loader, test_dates)

    # Plot evaluation results
    plot_path = os.path.join(directory_logs, "predictions_vs_ground_truth.png")
    evaluator.plot_results(
        predictions=evaluation_results["predictions"],
        targets=evaluation_results["targets"],
        dates=evaluation_results["dates"],
        save_path=plot_path
    )

if __name__ == '__main__':
    main()
    pass