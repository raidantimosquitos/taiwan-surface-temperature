import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by:
    - One-hot encoding the 'city' column.
    - Normalizing 'average temperature' and 'average temperature uncertainty'.
    - Dropping the 'country' column.

    :param df: Original DataFrame
    :return: Preprocessed DataFrame
    """
    # Drop the 'Country' column if it exists
    if "Country" in df.columns:
        df = df.drop(columns=["Country"])

    # One-hot encode the 'City' column
    encoder = OneHotEncoder(sparse_output=False)
    city_encoded = encoder.fit_transform(df[["City"]])
    city_encoded_df = pd.DataFrame(
        city_encoded, columns=encoder.get_feature_names_out(["City"]), index=df.index
    )
    df = pd.concat([df.drop(columns=["City"]), city_encoded_df], axis=1)

    # Normalize 'AverageTemperature' and 'AverageTemperatureUncertainty'
    scaler = MinMaxScaler()
    normalized_features = ["AverageTemperature", "AverageTemperatureUncertainty"]
    df[normalized_features] = scaler.fit_transform(df[normalized_features])

    return df

if __name__ == "__main__":
     # Define constants
    FILEPATH = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/datasets/berkeley-earth-surface-temp-dataset/TaiwanLandTemperaturesByCity.csv"

    # Load the dataframe
    df = pd.read_csv(FILEPATH)

    # Test the dataset
    print("\nDataframe before pre-processing:")
    print(df.head(10))

    # Apply transformation
    df_transf = preprocess_data(df)
    # Test the dataset
    print("\nDataframe after pre-processing:")
    print("Preprocessing summary one-hot encoding City, dropping Country and normalizing AverageTemperature and AverageTemperatureUncertainty features")
    print(df_transf.head(10))

    pass
