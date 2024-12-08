import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.dates import DateFormatter

# 載入資料
file_path = "/Users/blue/joanna_test/ML_final/taiwan-surface-temperature/datasets/taiwan_clean_dataset.csv"
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])
data.sort_values('date', inplace=True)

# 訓練與測試資料集切分
train = data[(data['date'] >= '1841-01-01') & (data['date'] <= '1979-12-01')]
test = data[(data['date'] >= '1980-01-01') & (data['date'] <= '2013-09-01')]

# 定義特徵與目標
features = [col for col in data.columns if col not in ['date', 'AverageTemperature']]
target = 'AverageTemperature'

# 測試集依據 CityGroup 切分
city_groups = ['CityGroup_North-East', 'CityGroup_North-West', 'CityGroup_South-East', 'CityGroup_South-West']
test_sets = {group: test[test[group] == 1] for group in city_groups}

# 訓練模型並預測
results = {}
for group, test_group in test_sets.items():
    print(f"Processing for {group}...")
    
    x_train = train[features]
    y_train = train[target]
    x_test = test_group[features]
    y_test = test_group[target]

    # 訓練隨機森林模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    
    # 預測與評估
    y_pred = rf.predict(x_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    # 儲存結果
    results[group] = {'rmse': rmse, 'r2': r2, 'y_test': y_test, 'y_pred': y_pred, 'dates': test_group['date']}
    
    # 繪製預測結果圖
    plt.figure(figsize=(12, 6))
    plt.plot(test_group['date'], y_test, label="Ground Truth", color="blue")
    plt.plot(test_group['date'], y_pred, label="Predictions", color="orange", linestyle="--")
    plt.title(f"{group} Predictions (RMSE: {rmse:.2f}, R²: {r2:.2f})")
    plt.xlabel("Date")
    plt.ylabel("Average Temperature")
    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"./{group}_predictions.png")
    plt.show()

# 總結結果
print("\nSummary of results:")
for group, metrics in results.items():
    print(f"{group}: RMSE = {metrics['rmse']:.2f}, R² = {metrics['r2']:.2f}")

# 預測未來12個月
future_features = train[features].iloc[-12:]  # 使用最近12個月的特徵
future_predictions = rf.predict(future_features)

# 畫未來12個月的預測圖
plt.figure(figsize=(12, 6))
plt.plot(train['date'].iloc[-12:], train[target].iloc[-12:], label="Historical Data", color="blue")
plt.plot(train['date'].iloc[-12:], future_predictions, label="Forecast", color="red", linestyle="--")
plt.title("12-Month Temperature Forecast")
plt.xlabel("Date")
plt.ylabel("Average Temperature")
plt.legend()
plt.grid()
plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./future_forecast.png")
plt.show()
