import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load and preprocess data
file_path = "/Users/blue/joanna_test/ML_final/taiwan-surface-temperature/datasets/taiwan_clean_dataset.csv"
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])
data.sort_values('date', inplace=True)

# Step 2: Split data into training and testing sets
train = data[(data['date'] >= '1841-01-01') & (data['date'] <= '1979-12-01')]
test = data[(data['date'] >= '1980-01-01') & (data['date'] <= '2013-09-01')]

# Define features and target
features = [col for col in data.columns if col != 'AverageTemperature' and col != 'date']
target = 'AverageTemperature'

# Split test data into 4 subsets based on city groups
city_groups = ['CityGroup_North-East', 'CityGroup_North-West',
               'CityGroup_South-East', 'CityGroup_South-West']

test_sets = {group: test[test[group] == 1] for group in city_groups}

# Step 3: Train and evaluate models
def train_and_evaluate(models, train, test_sets, features, target):
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        X_train, y_train = train[features], train[target]
        model.fit(X_train, y_train)

        results[name] = {}
        for group, test_set in test_sets.items():
            X_test, y_test = test_set[features], test_set[target]
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            results[name][group] = {'RMSE': rmse, 'MAE': mae, 'y_test': y_test, 'y_pred': y_pred}

            plt.figure(figsize=(10, 5))
            plt.plot(test_set['date'], y_test, label='Actual', color='blue', linewidth=2)
            plt.plot(test_set['date'], y_pred, label='Predicted', color='green', linestyle='--', linewidth=2)
            plt.title(f"{name} Predictions - {group}")
            plt.xlabel("Date")
            plt.ylabel("Average Temperature")
            plt.legend()
            plt.grid()
            plt.show()
            print(f"{name} on {group}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    return results

# Step 4: Filter results by date range
def filter_test_sets_by_date(test_sets, start_date, end_date):
    return {group: test_set[(test_set['date'] >= start_date) & (test_set['date'] <= end_date)]
            for group, test_set in test_sets.items()}

def plot_filtered_results(results, test_sets, model_name, start_date, end_date):
    filtered_test_sets = filter_test_sets_by_date(test_sets, start_date, end_date)
    for group, test_subset in filtered_test_sets.items():
        if test_subset.empty:
            print(f"No data for {group} between {start_date} and {end_date}.")
            continue
        y_test = test_subset['AverageTemperature']
        y_pred = results[model_name][group]['y_pred'][:len(test_subset)]
        dates = test_subset['date']

        plt.figure(figsize=(10, 5))
        plt.plot(dates, y_test, label='Actual', color='blue', linewidth=2)
        plt.plot(dates, y_pred, label='Predicted', color='green', linestyle='--', linewidth=2)
        plt.title(f"{model_name} Predictions ({start_date} to {end_date}) - {group}")
        plt.xlabel("Date")
        plt.ylabel("Average Temperature")
        plt.legend()
        plt.grid()
        plt.show()

# Step 5: Run the process
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = train_and_evaluate(models, train, test_sets, features, target)

# Step 6: Plot results for 2010-2012
start_date, end_date = '2010-01-01', '2012-12-31'
plot_filtered_results(results, test_sets, 'Linear Regression', start_date, end_date)
plot_filtered_results(results, test_sets, 'Random Forest', start_date, end_date)
