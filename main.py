import pandas as pd
import os

for dirname, _, filenames in os.walk('/datasets/berkeley-earth-surface-temp-dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# adjusting display options to make sure all columns are visible and up to 100 rows can be displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', None)

df_cities = pd.read_csv('datasets/berkeley-earth-surface-temp-dataset/GlobalLandTemperaturesByCity.csv')
df_tw = df_cities[df_cities['Country'] == 'Taiwan']

# preview of the dataset
print(df_tw)

# size of the dataset
print(df_tw.shape)

# number of records per city (30 cities in total)
print()
print(df_tw['City'].value_counts())

# display the number of empty records (84 temperature records in total)
print()
print(df_tw.isnull().sum())

# check if any duplicates within our data
print()
print('Duplicates: ', df_tw.duplicated().sum())

# print description of dataset
print(df_tw.describe().T)

# After all above we can observe that our dataset has 30 cities/regions in 
# Taiwan with total 62106 samples, no duplicates and 84 missing values in 
# average temperature and average temperature uncertainty features.