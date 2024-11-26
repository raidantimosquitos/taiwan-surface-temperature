import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TWTemperatureDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target_column: str, input_window: int, transforms=None):
        """
        :param data: Preprocessed DataFrame
        :param target_column: Name of the column to predict
        :param input_window: Number of past observations to use as input
        :param transforms: List of transformations to apply
        """
        self.data = data
        self.target_column = target_column
        self.input_window = input_window
        self.transforms = transforms or []

    def __len__(self):
        return len(self.data) - self.input_window

    def __getitem__(self, idx):
        # Extract input features (lagged data) and target
        x = self.data.iloc[idx:idx + self.input_window].drop(columns=[self.target_column]).values
        y = self.data.iloc[idx + self.input_window][self.target_column]

        # Apply transformations sequentially
        for transform in self.transforms:
            if isinstance(transform, ToTensor):
                x, y = transform(x, y)  # Transform input and target to tensors
            else:
                self.data = transform.transform(self.data)

        return x, y
    
# Normalization transformation class
class Normalize:
    def __init__(self, columns):
        """
        :param columns: List of columns to normalize
        """
        self.columns = columns
        self.min_values = {}
        self.max_values = {}

    def fit(self, df):
        """
        Calculate min and max for normalization.
        :param df: DataFrame
        """
        for col in self.columns:
            self.min_values[col] = df[col].min()
            self.max_values[col] = df[col].max()

    def transform(self, df):
        """
        Normalize the columns.
        :param df: DataFrame
        :return: Transformed DataFrame
        """
        for col in self.columns:
            df[col] = (df[col] - self.min_values[col]) / (self.max_values[col] - self.min_values[col])
        return df

# Class add lag features
class AddLagFeatures:
    def __init__(self, column, lags):
        """
        :param column: Column name to generate lag features for
        :param lags: Number of lag features to create
        """
        self.column = column
        self.lags = lags

    def transform(self, df):
        """
        Add lag features to the DataFrame.
        :param df: DataFrame
        :return: Transformed DataFrame with lag features
        """
        for lag in range(1, self.lags + 1):
            df[f'{self.column}_lag_{lag}'] = df[self.column].shift(lag)
        return df.dropna()  # Drop rows with NaN values created by lagging

# ToTensor transformation class. Returns input samples as torch tensors.
class ToTensor:
    def __call__(self, x, y):
        """
        Converts data to PyTorch tensors.
        :param x: Input features (NumPy array or DataFrame slice)
        :param y: Target value
        :return: Tuple of tensors
        """
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file, convert 'date-time' to a datetime object,
    set it as the index, and arrange the data in temporal order.

    :param filepath: Path to the CSV file
    :return: Temporally sorted DataFrame
    """
    # Load the data
    df = pd.read_csv(filepath)

    # Convert 'dt' column to datetime objects
    df['dt'] = pd.to_datetime(df['dt'])

    # Sort data by 'dt' in ascending order
    df = df.sort_values(by='dt')

    # Drop Country, City and NA features
    df = df.drop(columns='Country')
    df = df.drop(columns='City')
    df = df.dropna()

    # Set 'dt' as the index
    df.set_index('dt', inplace=True)

    return df  

# Testing the functionality
if __name__ == '__main__':
    # Step 1: Load the data
    filepath = "datasets/berkeley-earth-surface-temp-dataset/TaiwanLandTemperaturesByCity.csv"
    df = load_data(filepath)

    print("Loaded dataset:")
    print(df.head())

    # Step 2: Define preprocessing transformations
    target_column = "AverageTemperature"

    # Normalization
    normalize = Normalize(columns=["AverageTemperature", "AverageTemperatureUncertainty"])
    normalize.fit(df)
    df = normalize.transform(df)
    
    numeric_columns = df.select_dtypes(include=["number"]).columns
    print(numeric_columns)
    
    print("\nAfter Normalization:")
    print(df.head())

    # Add lag features (e.g., last 3 time steps)
    lag_features = AddLagFeatures(column="AverageTemperature", lags=3)
    df = lag_features.transform(df)

    print("\nAfter Adding Lag Features:")
    print(df.head())

    # Step 3: Create the PyTorch Dataset
    input_window = 5  # Use the last 5 time steps as input
    transforms = [ToTensor()]

    dataset = TWTemperatureDataset(data=df, target_column=target_column, input_window=input_window, transforms=transforms)

    # Step 4: Test the dataset
    print("\nExample Data Point:")
    x, y = dataset[0]
    print("Input (x):", x)
    print("Target (y):", y)

    # Step 5: Use the Dataset with a DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Example: Iterate over one batch
    print("\nBatch Example:")
    for batch_x, batch_y in dataloader:
        print("Batch Input Shape:", batch_x.shape)
        print("Batch Target Shape:", batch_y.shape)
        break

    pass