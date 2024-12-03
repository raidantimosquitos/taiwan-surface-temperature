import numpy as np
import xgboost as xgb

def load_city_mapping():
    return np.load('city_mapping.npy', allow_pickle=True).item()

def get_available_cities():
    return sorted(load_city_mapping().keys())

def user_input_prompt():
    cities = get_available_cities()
    print("Available cities:")
    for i in range(0, len(cities), 4):
        print("  ".join(cities[i:i + 4]))

    city_name = input("\nEnter the city name: ").strip()

    if city_name not in cities:
        print(f"City '{city_name}' not available.")
        return None

    try:
        latitude = float(input("Enter the latitude: ").strip())
        longitude = float(input("Enter the longitude: ").strip())
        year = int(input("Enter the year: ").strip())
        month = int(input("Enter the month (1-12): ").strip())
    except ValueError:
        print("Invalid input.")
        return None

    return city_name, latitude, longitude, year, month

def predict_temperature(model, city_name, latitude, longitude, year, month):
    city_mapping = load_city_mapping()
    city_label = city_mapping.get(city_name, -1)
    
    if city_label == -1:
        print(f"City '{city_name}' not found!")
        return None

    features = np.array([[latitude, longitude, year, month, city_label]])
    feature_names = ['Latitude', 'Longitude', 'Year', 'Month', 'City']
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)

    return model.predict(dmatrix)[0]

def main():
    model = xgb.Booster()
    model.load_model('xgboost_model.json')

    user_input = user_input_prompt()

    if user_input:
        city_name, latitude, longitude, year, month = user_input
        predicted_temp = predict_temperature(model, city_name, latitude, longitude, year, month)
        if predicted_temp is not None:
            print(f"\nPredicted temperature for {city_name} in {year}-{month}: {predicted_temp:.2f} °C")

if __name__ == "__main__":
    main()