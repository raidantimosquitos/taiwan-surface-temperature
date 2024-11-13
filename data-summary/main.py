import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy import stats
import seaborn as sns

d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = d + '/datasets/'
output_dir = d + '/data-summary/'

# adjusting display options to make sure all columns are visible and up to 100 rows can be displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', None)

df_tw = pd.read_csv(datasets_dir + '/berkeley-earth-surface-temp-dataset/TaiwanLandTemperaturesByCity.csv')

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

# Convert 'dt' to datetime format
df_tw['dt'] = pd.to_datetime(df_tw['dt'])

# Grouping data by seasons
df_tw['Month'] = df_tw['dt'].dt.month

# Plot
plt.figure(figsize=(16, 10))
sns.lineplot(data=df_tw, x='Month', y='AverageTemperature', hue='City', marker='o')
plt.title('Seasonal Temperature Patterns in Taiwanese Cities', fontsize=18)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Temperature (°C)', fontsize=14)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Rename the dt dataframe column
df_tw = df_tw.rename(columns={'dt': 'date'})

# Create new dataframe with date time features and seasons
df_features = df_tw.copy()
df_features['date'] = pd.to_datetime(df_features['date'])
df_features['year'] = df_features['date'].dt.year
df_features['month'] = df_features['date'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    
df_features['season'] = df_features['date'].dt.month.apply(get_season)

# Box Plot for temperatures distribution in Taiwan per season
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_features, x='season', y='AverageTemperature', palette='coolwarm')
plt.title('Temperature Distribution by Season', fontsize=18)
plt.xlabel('Season', fontsize=14)
plt.ylabel('Average Temperature (°C)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Ensure 'season' column is categorical with specific order
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
df_features['season'] = pd.Categorical(df_features['season'], categories=season_order, ordered=True)

# Compute the average temperature by season
seasonal_avg_temp = df_features.groupby('season')['AverageTemperature'].mean().reset_index()

# Bar plot for average temperature by season
plt.figure(figsize=(10, 6))
sns.barplot(data=seasonal_avg_temp, x='season', y='AverageTemperature', palette='coolwarm')
plt.title('Average Temperature by Season', fontsize=18)
plt.xlabel('Season', fontsize=14)
plt.ylabel('Average Temperature (°C)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

# Create one extra column in the dataframe to indicate weather the temperature is above or
# below the average of that season.
baseline_avg_temp = df_features.groupby('season')['AverageTemperature'].mean()
df_features['Anomaly'] = df_features.apply(lambda row: row['AverageTemperature'] - baseline_avg_temp[row['season']], axis=1)

# Compute average anomalies by season
seasonal_anomalies = df_features.groupby('season')['Anomaly'].mean().reset_index()

# Plot anomalies
plt.figure(figsize=(10, 6))
sns.barplot(data=seasonal_anomalies, x='season', y='Anomaly', palette='coolwarm')
plt.title('Average Temperature Anomalies by Season', fontsize=18)
plt.xlabel('Season', fontsize=14)
plt.ylabel('Temperature Anomaly (°C)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


# Compute average temperature for each season and year
seasonal_temp_trend = df_features.groupby(['Year', 'season'])['AverageTemperature'].mean().reset_index()

# Plot
plt.figure(figsize=(16, 10))
sns.lineplot(data=seasonal_temp_trend, x='Year', y='AverageTemperature', hue='season', marker='o')
plt.title('Seasonal Temperature Trends Over Time (1841-2013)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Temperature (°C)', fontsize=14)
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot missing values heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(df_features.isnull(), cbar=False, cmap='summer', yticklabels=False)
plt.title('Missing Data Heatmap')
plt.show()