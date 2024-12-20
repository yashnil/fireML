import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the NetCDF data
data_path = '/Users/yashnilmohanty/Desktop/combined_data.nc'
ds = xr.open_dataset(data_path)

# Variables to include in the correlation
variables = [
    'Elevation', 'slope', 'VegTyp', 'aspect_ratio',
    'aorcFallTemperature', 'aorcFallRain', 'aorcFallHumidity', 'aorcFallLongwave', 'aorcFallShortwave',
    'aorcSpringTemperature', 'aorcSpringRain', 'aorcSpringHumidity', 'aorcSpringLongwave', 'aorcSpringShortwave',
    'aorcSummerTemperature', 'aorcSummerRain', 'aorcSummerHumidity', 'aorcSummerLongwave', 'aorcSummerShortwave',
    'aorcWinterTemperature', 'aorcWinterRain', 'aorcWinterHumidity', 'aorcWinterLongwave', 'aorcWinterShortwave',
    'sweSpring', 'sweWinter', 'DOD'
]

# Aggregate data spatially (mean over all coordinates)
temporal_data = ds[variables].mean(dim='ncoords')

# Convert to a Pandas DataFrame for easier manipulation
temporal_df = temporal_data.to_dataframe()

# Compute the correlation matrix across time (rows = years)
correlation_matrix_time = temporal_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_time, annot=False, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Temporal Correlation Heatmap')
plt.show()
