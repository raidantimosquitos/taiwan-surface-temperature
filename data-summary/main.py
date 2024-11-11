import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy import stats

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

# Assuming df_cities is already loaded

# Extract the year from the date
df_tw['Year'] = df_tw['dt'].apply(lambda x: x[:4])

# Group by year and calculate the average temperature and uncertainty
tw_yearly = df_tw.groupby('Year').agg({
    'AverageTemperature': 'mean',
    'AverageTemperatureUncertainty': 'mean'
}).reset_index()

# Extract the values
years = tw_yearly['Year'].astype(int).values
mean_temp_tw = tw_yearly['AverageTemperature'].values
mean_temp_tw_uncertainty = tw_yearly['AverageTemperatureUncertainty'].values

# Calculate the trend line (linear regression)
slope, intercept, r_value, p_value, std_err = stats.linregress(years, mean_temp_tw)
trend_line = slope * years + intercept

# Create traces for the plot
trace0 = go.Scatter(
    x=years, 
    y=mean_temp_tw + mean_temp_tw_uncertainty,
    fill=None,
    mode='lines',
    name='Uncertainty top',
    line=dict(color='rgb(0, 255, 255)')
)

trace1 = go.Scatter(
    x=years, 
    y=mean_temp_tw - mean_temp_tw_uncertainty,
    fill='tonexty',
    mode='lines',
    name='Uncertainty bot',
    line=dict(color='rgb(0, 255, 255)')
)

trace2 = go.Scatter(
    x=years, 
    y=mean_temp_tw,
    name='Average Temperature',
    line=dict(color='rgb(199, 121, 093)')
)

# Add the trend line
trace3 = go.Scatter(
    x=years,
    y=trend_line,
    name='Trend Line',
    line=dict(color='rgb(255, 0, 0)', dash='dash')
)

# Add a vertical line to indicate the start of global warming (around 1970)
global_warming_start_year = 1970

trace4 = go.Scatter(
    x=[global_warming_start_year, global_warming_start_year],
    y=[min(mean_temp_tw - mean_temp_tw_uncertainty), max(mean_temp_tw + mean_temp_tw_uncertainty)],
    mode='lines',
    name='Start of Global Warming',
    line=dict(color='rgb(0, 100, 0)', dash='dot')
)

data = [trace0, trace1, trace2, trace3, trace4]

layout = go.Layout(
    xaxis=dict(title='Year'),
    yaxis=dict(title='Average Temperature, °C'),
    title='Average Land Temperature in Taiwan with Trend Line and Global Warming Start',
    showlegend=True
)

fig = go.Figure(data=data, layout=layout)
fig.show()