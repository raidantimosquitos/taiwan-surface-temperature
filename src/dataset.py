import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from src.utils import ToTensor

class TWTemperatureDataset(Dataset):
    def __init__(self, filepath: str = None, dataframe: pd.DataFrame = None, target_column: str = None, transforms=None):
        """
        :param filepath: Path to the CSV file
        :param dataframe: Preloaded DataFrame to use instead of loading from file
        :param target_column: Name of the column to predict
        :param transforms: List of transformations to apply
        """
        # Ensure either filepath or dataframe is provided
        if filepath is None and dataframe is None:
            raise ValueError("Either `filepath` or `dataframe` must be provided.")

        if filepath:
            df = pd.read_csv(filepath)
        else:
            df = dataframe.copy()  # Avoid modifying the original DataFrame

        if 'date' in df.columns.values.tolist():
            df.set_index('date', inplace=True)
      
        # Store data and configuration
        self.data = df
        self.target_column = target_column

        self.dates = self.data.index
        self.transforms = transforms or [ToTensor()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract input features and target
        row = self.data.iloc[idx]
        x = row.drop(labels=[self.target_column]).values  # Exclude the target column
        y = row[self.target_column]

        # Ensure x contains numeric data
        x = np.array(x, dtype=np.float32)

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
    
    def get_dates(self):
        """
        Returns the dates corresponding to the target values.
        """
        return self.dates.tolist()
    
    def get_mean_std(self, column):
        """
        Returns the mean and standard deviation of the target value
        """
        self.mean = self.data[column].mean()
        self.std = self.data[column].std()

        return self.mean, self.std
    
    def get_CityGroup(self, citygroup_name):
        """
        Returns a dataset filtered for a specific CityGroup.

        Args:
            citygroup_name (str): The name of the CityGroup to filter by.

        Returns:
            TWTemperatureDataset: A new dataset object containing only the rows
                                for the specified CityGroup.
        """
        name = 'CityGroup_' + citygroup_name
        if name not in self.data.columns:
            raise ValueError("`CityGroup` column not found in the dataset.")
        
        # Filter the DataFrame by the specified CityGroup
        citygroup_data = self.data[self.data[name] == 1]
        feat_to_del = [feat_name for feat_name in citygroup_data.columns.values.tolist() if (('CityGroup_' in feat_name) or ('long_' in feat_name) or ('lat_' in feat_name))]

        citygroup_data = citygroup_data.drop(columns = feat_to_del)
        
        # Return a new instance of the dataset with the filtered data
        return TWTemperatureDataset(dataframe=citygroup_data, target_column=self.target_column, transforms=[ToTensor()])


# Testing the functionality
if __name__ == '__main__':
    # Define constant
    FILEPATH = "/home/lucash/NTUST_GIMT/2024_Fall_Semester/Machine_Learning/taiwan-surface-temperature/datasets/taiwan_clean_dataset.csv"
    TARGET_COLUMN = "AverageTemperature"
    BATCH_SIZE = 32

    # Create the dataset
    dataset = TWTemperatureDataset(filepath=FILEPATH, target_column=TARGET_COLUMN, transforms=[ToTensor()])

    # Test the dataset
    print("New Input features: ", dataset.get_feature_names())
    print("\nExample Data Point:")
    x, y = dataset[0]
    date = dataset.get_dates()[0]
    print("Date of input (index): ", date)
    print("Input (x):", x)
    print("Target (y):", y)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for batch_x, batch_y in dataloader:
        print("Batch Input Shape:", batch_x.shape)
        print("Batch Target Shape:", batch_y.shape)
        break
    
    pass