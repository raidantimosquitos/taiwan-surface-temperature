import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = d + '/datasets/'
output_dir = d + '/data-summary/'

# adjusting display options to make sure all columns are visible and up to 100 rows can be displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', None)

df_cities = pd.read_csv(datasets_dir + 'berkeley-earth-surface-temp-dataset/GlobalLandTemperaturesByCity.csv')

# dataframe with corrected latitude-longitude values and city names
df_coords = pd.read_csv(datasets_dir + 'worldcities-coordinates/taiwan-city-names-and-coordinates.csv')
df_tw = df_cities[df_cities['Country'] == 'Taiwan']

# assign correct coordinates and city names to df_tw dataframe
for index, row in df_coords.iterrows():
    city_name = row['City']
    corrected_cn = row['Name-equivalent']
    lat = row['Latitude']
    long = row['Longitude']

    # changing coordinates
    df_tw.loc[df_tw['City'] == city_name, 'Latitude'] = lat
    df_tw.loc[df_tw['City'] == city_name, 'Longitude'] = long
    df_tw.loc[df_tw['City'] == city_name, 'City'] = corrected_cn

df_tw = df_tw.reset_index(drop=True)

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

# output the corrected dataset
df_tw.to_csv(output_dir + 'TaiwanLandTemperaturesByCity.csv')

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
df_aux['Latitude'] = df_aux['Latitude'].astype(np.float32)
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
ax.set_xlim(120, 122)
ax.set_ylim(22, 25.5)
ax.set_title('Average Temperatures in Taiwanese Cities')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()