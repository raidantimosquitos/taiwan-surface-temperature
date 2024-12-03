import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def main():
    # Load and preprocess dataset
    df = pd.read_csv("datasets/berkeley-earth-surface-temp-dataset/TaiwanLandTemperaturesByCity.csv")
    df['dt'] = pd.to_datetime(df['dt'], format='%Y-%m-%d')
    df['Year'] = df['dt'].dt.year
    df['Month'] = df['dt'].dt.month
    df = df.drop(columns=['dt', 'Unnamed: 0'])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df[['Latitude', 'Longitude', 'Year', 'Month']] = imputer.fit_transform(df[['Latitude', 'Longitude', 'Year', 'Month']])
    df['AverageTemperature'] = df['AverageTemperature'].fillna(df['AverageTemperature'].mean())

    # Label encode city names
    label_encoder = LabelEncoder()
    df['City'] = label_encoder.fit_transform(df['City'])
    city_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    np.save('city_mapping.npy', city_mapping)

    # Prepare features and target
    X = df[['Latitude', 'Longitude', 'Year', 'Month', 'City']]
    y = df['AverageTemperature']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {'objective': 'reg:squarederror', 'learning_rate': 0.05, 'max_depth': 6, 'eval_metric': 'rmse'}
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round=10000, evals=evals, early_stopping_rounds=50, verbose_eval=True)

    # Evaluate model
    y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Save model
    model.save_model('xgboost_model.json')

if __name__ == "__main__":
    main()