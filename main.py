import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from tabulate import tabulate

from config import FILEPATH, CHECKPOINTS_DIR, LOGS_DIR, TARGET_COLUMN, BATCH_SIZE, N_SPLITS, LR, NUM_EPOCHS, DEVICE, SEQUENCE_NO
from src.dataset import TWTemperatureDataset
from src.utils import ToTensor, plot_losses, create_sequences, create_directory
from src.train import train
from src.models import LSTMModel
from src.test import ForecastFutureValues

def main():
    # Create the dataset
    dataset = TWTemperatureDataset(
        filepath=FILEPATH, 
        target_column=TARGET_COLUMN,
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

    # Create sequences to feed LSTM model 
    xtrain_sequences, ytrain_sequences = create_sequences(train_data, SEQUENCE_NO)
    ytrain_sequences = (ytrain_sequences - ytrain_sequences.mean())/ytrain_sequences.std()

    xcv_sequences, ycv_sequences = create_sequences(cross_val_data, SEQUENCE_NO)
    ycv_sequences = (ycv_sequences - ycv_sequences.mean())/ycv_sequences.std()

    sequenced_train_data = TensorDataset(xtrain_sequences, ytrain_sequences)
    sequenced_cv_data = TensorDataset(xcv_sequences, ycv_sequences)

    # Data loaders
    train_loader = DataLoader(sequenced_train_data, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle to preserve temporal order
    crossval_loader = DataLoader(sequenced_cv_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model, criterion, optimizer
    model = LSTMModel(input_dim=in_size, hidden_dim=64, num_layers=1, output_dim=1)
    criterion = torch.nn.MSELoss()  # Using MSE for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    # train the model
    directory_logs = train(model, train_loader, crossval_loader, criterion, optimizer, 
                           num_epochs=NUM_EPOCHS, device=DEVICE, logs_dir=LOGS_DIR, 
                           checkpoints_dir=CHECKPOINTS_DIR, scheduler=scheduler)
    plot_losses(os.path.join(directory_logs, "loss_history.json"), directory_logs)

    test_data = Subset(dataset, test_indices)

    xtest_sequences, ytest_sequences = create_sequences(test_data, SEQUENCE_NO)
    ytest_sequences = (ytest_sequences - ytest_sequences.mean())/ytest_sequences.std()
    sequenced_test_data = TensorDataset(xtest_sequences, ytest_sequences)

    # Data loaders
    test_loader = DataLoader(sequenced_test_data, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle to preserve temporal order
    dates = dataset.get_dates()
    test_dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates[-len(sequenced_test_data):]]
    city_eval_results, city_forecast_results = {}, {}

    # Initialize and evaluate the model
    best_model_path = os.path.join(CHECKPOINTS_DIR, 'best_model.pth')
    evaluator = ForecastFutureValues(model=model, checkpoint_path=best_model_path, device=DEVICE)
    city_eval_results['Taiwan'] = evaluator.evaluate(test_loader, test_dates)

    # Train, evaluate and forecast models for each CityGroup
    city_groups = ['North-East', 'North-West', 'South-East', 'South-West']
    
    for city_group in city_groups:
        print(f"\nTraining for city group: {city_group}...")

        # Get city-specific data
        city_dataset = dataset.get_CityGroup(city_group)
        city_in_size = len(city_dataset.get_feature_names())
        city_train_idxs, city_test_idxs = list(tscv.split(city_dataset))[-1]

        city_train_data = Subset(city_dataset, city_train_idxs)
        city_test_data = Subset(city_dataset, city_test_idxs)

        xtrain_city, ytrain_city = create_sequences(city_train_data, SEQUENCE_NO)
        ytrain_city = (ytrain_city - ytrain_city.mean())/ytrain_city.std()

        xtest_city, ytest_city = create_sequences(city_test_data, SEQUENCE_NO)
        ytest_city = (ytest_city - ytest_city.mean())/ytest_city.std()

        citytrain_sequenced_data = TensorDataset(xtrain_city, ytrain_city)
        citytest_sequenced_data = TensorDataset(xtest_city, ytest_city)

        # Data loaders
        citytrain_loader = DataLoader(citytrain_sequenced_data, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle to preserve temporal order
        citytest_loader = DataLoader(citytest_sequenced_data, batch_size=BATCH_SIZE, shuffle=False)

        # Test dates
        city_dates = city_dataset.get_dates()
        city_test_dates = [datetime.strptime(date, "%Y-%m-%d") for date in city_dates[-len(citytest_sequenced_data):]]

        # Training loop
        city_model = LSTMModel(input_dim= city_in_size, hidden_dim=64, num_layers=1, output_dim=1)
        city_optimizer = torch.optim.Adam(city_model.parameters(), lr=LR, weight_decay=1e-4)
        city_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(city_optimizer)

        logs = os.path.join(directory_logs, city_group)
        check_dir = os.path.join(CHECKPOINTS_DIR, city_group)
        create_directory(check_dir)
        best_model_path = os.path.join(check_dir, 'best_model.pth')

        city_logs = train(city_model, citytrain_loader, citytest_loader, criterion, 
                          city_optimizer, num_epochs=NUM_EPOCHS, device=DEVICE, 
                          logs_dir=logs, checkpoints_dir=check_dir, scheduler=city_scheduler)
        plot_losses(os.path.join(city_logs, "loss_history.json"), city_logs)

        print(f'Evaluate and forecast for CityGroup {city_group}')
        city_evaluator = ForecastFutureValues(model=city_model, checkpoint_path=best_model_path, device=DEVICE)
        city_eval_results[city_group] = city_evaluator.evaluate(citytest_loader, city_test_dates)
        city_forecast_results[city_group] = city_evaluator.forecast(citytest_sequenced_data, city_test_dates)

    return directory_logs, city_eval_results, city_forecast_results      

if __name__ == '__main__':
    directory_logs, test_res, forecast_res = main()
    
    plt.figure(num=6, figsize=(20, 15))
    
    aux_dict = {'targets': [], 'dates': []}
    for i, (key, values) in enumerate(test_res.items()):
        if i > 0:
            # do something with key and value
            aux_dict['targets'].append(values['targets'])
            aux_dict['dates'].append(values['dates'])
            plt.subplot(2, 2, i)
            plt.plot(values['dates'], values['targets'], label='Ground Truth', color='blue')
            plt.plot(values['dates'], values['predictions'], label='Predictions', color='orange', alpha=0.7)
            plt.xticks(rotation=45)
            plt.title(f'Model Predictions vs Ground Truth ({key})')
            plt.xlabel('Date')
            plt.ylabel('Temperature')
            # Format x-axis to show dates properly
            # Format the x-axis to display only years (or customize as needed)
            plt.gca().xaxis.set_major_locator(mdates.YearLocator(), )  # Show every year
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as Year (YYYY)
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(10))
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(directory_logs, 'ground_truth_vs_predictions.png'))

    plt.figure(num=7, figsize=(20, 15))
    j = 0
    for key, values in forecast_res.items():
        plt.subplot(2, 2, j+1)
        plt.plot()
        plt.plot(aux_dict['dates'][j][-36:-1], aux_dict['targets'][j][-36:-1], label="Historical Data", color="blue")
        plt.plot(
            values["future_dates"],
            values["forecasted_values"],
            label="Forecasted Values",
            color="red"
        )
        plt.title(f"12-Month Temperature Forecast for CityGroup {key}")
        plt.xlabel("Date")
        plt.ylabel("Temperature")
        plt.legend()
        # Format x-axis to show dates properly
        # Format the x-axis to display only years (or customize as needed)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Show every year
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as Year (YYYY)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.savefig(os.path.join(directory_logs, 'forecasts.png'))
        j += 1

    key_list = list(test_res.keys())
    headers = ['Model', 'RMSE Loss', 'R^2 Score']
    aux_list = [[i, test_res[i]['rmse'], test_res[i]['r2_score']] for i in key_list]
    table = tabulate(aux_list, headers=headers, tablefmt='orgtbl')
    print('\nSummary of the metrics: ')
    print(table)