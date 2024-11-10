import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = d + '/datasets/berkeley-earth-surface-temp-dataset/'

# adjusting display options to make sure all columns are visible and up to 100 rows can be displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', None)

df_cities = pd.read_csv(datasets_dir + 'GlobalLandTemperaturesByCity.csv')
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

df_aux = df_tw.loc[df_tw['dt'] == '2013-01-01']
df_aux = df_aux.drop(columns=['dt', 'AverageTemperature', 'AverageTemperatureUncertainty','Country'])

# Compute average temperature for all records per city in Taiwan
average_temp = []
for city in df_aux['City'].values:
    mean_temp = df_tw.loc[df_tw['City'] == city, 'AverageTemperature'].mean()
    average_temp.append(mean_temp.astype(np.float32))

df_aux['AverageTemperature'] = average_temp

df_aux['Latitude'] = df_aux['Latitude'].str.replace(r'N', '', regex=True)
df_aux['Latitude'] = df_aux['Latitude'].astype(np.float32)

df_aux['Longitude'] = df_aux['Longitude'].str.replace(r'E', '', regex=True)
df_aux['Longitude'] = df_aux['Longitude'].astype(np.float32)
df_aux = df_aux.reset_index(drop=True)

print()
print(df_aux)

# Create a Geopandas GeoDataFrame for plotting
gdf_cities = gpd.GeoDataFrame(df_aux, 
                              geometry=gpd.points_from_xy(df_aux['Longitude'], df_aux['Latitude']))

# Plot the cities with colors based on average temperatures
fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot of cities, with color representing temperature
scatter = ax.scatter(df_aux['Longitude'], df_aux['Latitude'], 
                     c=df_aux['AverageTemperature'], cmap='coolwarm', s=100, edgecolor='black')

# Annotate each city
for i, row in df_aux.iterrows():
    ax.text(row['Longitude'], row['Latitude'], row['City'], fontsize=8, ha='right')

# Add a colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Average Temperature (°C)')

# Set axis limits to focus on Taiwan
ax.set_xlim(119, 123)
ax.set_ylim(23, 25)
ax.set_title('Average Temperatures in Taiwanese Cities')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# From the plot you can appreciate how the coordinates are clustered in four groups,
# this is because in the original dataset there were only two decimal digit accuracy 
# to specify latitude and longitude. In the end merging the cities that are close to
# each other in a small country as Taiwan is.

df_aux['Group'] = df_aux.groupby(['Latitude', 'Longitude']).ngroup()

print()
print('Group 0: South West cities')
print(df_aux.loc[df_aux['Group'] == 0])

print('Group 1: South East cities')
print(df_aux.loc[df_aux['Group'] == 1])

print('Group 2: Central cities')
print(df_aux.loc[df_aux['Group'] == 2])

print('Group 3: North cities')
print(df_aux.loc[df_aux['Group'] == 3])

# However after checking on Google Maps, it seems something is wrong with the
# coordinates in the dataset, cities like Nantou and Taitung should not be in
# the same group.