import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_preprocess import preprocess_data
from utils import ToTensor, AddLagFeatures

class TWTemperatureDataset(Dataset):
    def __init__(self, filepath: str, target_column: str, input_window: int, transforms=None):
        """
        :param filepath: Path to the CSV file
        :param target_column: Name of the column to predict
        :param input_window: Number of past observations to use as input
        :param transforms: List of transformations to apply
        """
        # Load and preprocess data
        df = pd.read_csv(filepath)
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.sort_values(by="dt")
        df.set_index("dt", inplace=True)
        df = preprocess_data(df)

        # Apply lag features
        lag_transform = AddLagFeatures(input_window, target_columns=["AverageTemperature", "AverageTemperatureUncertainty"])
        df = lag_transform.transform(df)

        # Store data and configuration
        self.data = df
        self.target_column = target_column
        self.input_window = input_window
        self.transforms = transforms or [ToTensor()]

    def __len__(self):
        return len(self.data) - self.input_window

    def __getitem__(self, idx):
        # Extract input features (lagged data) and target
        x = self.data.iloc[idx:idx + self.input_window].drop(columns=[self.target_column]).values
        y = self.data.iloc[idx + self.input_window][self.target_column]

        # Convert x to numeric
        x = x.astype(np.float32)

        # Apply transformations sequentially
        for transform in self.transforms:
            x, y = transform(x, y)

        return x, y
    
    def get_feature_names(self):
        """
        Returns the names of the features used as input.
        """
        # Exclude the target column from feature names
        return [col for col in self.data.columns if col != self.target_column]

# Testing the functionality
if __name__ == '__main__':
    # Define constant
    FILEPATH = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/datasets/berkeley-earth-surface-temp-dataset/TaiwanLandTemperaturesByCity.csv"
    TARGET_COLUMN = "AverageTemperature"
    INPUT_WINDOW = 2
    BATCH_SIZE = 32

    # Create the dataset
    dataset = TWTemperatureDataset(filepath=FILEPATH, target_column=TARGET_COLUMN, input_window=INPUT_WINDOW, transforms=[ToTensor()])

    # Test the dataset
    print("New Input features: ", dataset.get_feature_names())
    print("\nExample Data Point:")
    x, y = dataset[0]
    print("Input (x):", x)
    print("Target (y):", y)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for batch_x, batch_y in dataloader:
        print("Batch Input Shape:", batch_x.shape)
        print("Batch Target Shape:", batch_y.shape)
        break
    
    pass