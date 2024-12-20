import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

# Load the NetCDF data
data_path = '/Users/yashnilmohanty/Desktop/combined_data.nc'
ds = xr.open_dataset(data_path)

# Define feature and target variables
static_features = ['Elevation', 'slope', 'VegTyp', 'aspect_ratio']
time_features = [
    'aorcFallTemperature', 'aorcFallRain', 'aorcFallHumidity', 'aorcFallLongwave', 'aorcFallShortwave',
    'aorcSpringTemperature', 'aorcSpringRain', 'aorcSpringHumidity', 'aorcSpringLongwave', 'aorcSpringShortwave',
    'aorcSummerTemperature', 'aorcSummerRain', 'aorcSummerHumidity', 'aorcSummerLongwave', 'aorcSummerShortwave',
    'aorcWinterTemperature', 'aorcWinterRain', 'aorcWinterHumidity', 'aorcWinterLongwave', 'aorcWinterShortwave',
    'sweSpring', 'sweWinter'
]
target_var = 'DOD'

# Prepare Static Data
static_data = []
for var in static_features:
    static_data.append(ds[var].values.flatten())  # Flatten Geo2D data
X_static = np.array(static_data).T  # Transpose to shape (n_samples, n_features)

# Prepare Time-Series Data
n_years = 15
n_seasons = 4
time_data = []

for var in time_features:
    # Stack the seasonal data for all years
    time_data.append(ds[var].values.reshape(n_years, -1))  # Reshape to (n_years, n_samples_per_year)

X_time = np.stack(time_data, axis=-1)  # Shape: (n_years, n_samples_per_year, n_features)

# Average over seasons for simplicity (optional)
time_series_avg = X_time.mean(axis=0)  # Shape: (n_samples, n_features)

# Target Variable (flatten to align with static and time-series data)
y = ds[target_var].values.flatten()

# Combine static and time-series data into a DataFrame
static_df = pd.DataFrame(X_static, columns=static_features)
time_series_df = pd.DataFrame(time_series_avg, columns=time_features)
combined_df = pd.concat([static_df, time_series_df], axis=1)
combined_df['DOD'] = y  # Add the target variable

# Compute Correlation Matrix
correlation_matrix = combined_df.corr()

# Plot Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
