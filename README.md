# Study of Taiwan Land Surface Temperature

This project predicts the **average temperature** for various cities in Taiwan based on their geographic location (latitude and longitude), the time of year (year and month), and the city name. The model is trained using XGBoost, a powerful gradient boosting algorithm, and the predictions are made using a trained model.

## Why XGBoost for Temperature Prediction?

XGBoost (Extreme Gradient Boosting) is a highly efficient and flexible gradient boosting algorithm that excels in supervised learning tasks. It is based on decision trees and builds models through an iterative process where each new tree corrects the errors of the previous trees. XGBoost uses a technique called boosting, which allows it to combine the outputs of multiple weak learners (decision trees) to form a strong and accurate predictive model:

1. **Effective for Tabular Data**: XGBoost excels in handling structured, tabular data, making it an ideal choice for this dataset, which contains both numerical (latitude, longitude, year, month) and categorical (city) features.

2. **Captures Complex Relationships**: Temperature prediction involves complex, nonlinear relationships between geographic and temporal variables. XGBoost's gradient boosting framework can effectively capture these interactions and dependencies, ensuring accurate predictions.

3. **Robustness to Overfitting**: With built-in regularization (L1 and L2), XGBoost controls overfitting, which is crucial for preventing the model from learning noise from the data and ensuring good generalization to unseen cities and time periods.

4. **Efficient Handling of Missing Data**: XGBoost handles missing values natively, eliminating the need for extensive data imputation, which is particularly helpful when dealing with real-world datasets that may have incomplete records.

5. **Scalability and Speed**: XGBoost is optimized for speed and memory efficiency, enabling it to handle larger datasets and make predictions quickly without significant performance loss. This makes it ideal for efficiently processing large geographical and temporal datasets.

## Files Overview

### 1. **train.py**

The `train.py` script is used to train an XGBoost model on the dataset, which includes information such as the city name, geographic coordinates, and time. The steps followed in this script are:
- Data preprocessing (handling missing values and encoding categorical data)
- Training the XGBoost model
- Evaluating model performance using Root Mean Squared Error (RMSE)
- Saving the trained model for later use

### 2. **predict.py**

The `predict.py` script allows the user to input data about a city (name, latitude, longitude, year, and month) and uses the saved XGBoost model to predict the average temperature for that city at the specified time. It includes:
- A user input prompt for entering data
- A temperature prediction function using the trained XGBoost model
- Display of the predicted temperature for the given input

## Requirements

- Python 3.x
- XGBoost
- NumPy
- Pandas
- scikit-learn
