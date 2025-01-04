import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the processed NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/preprocessed_combined_data.nc'  # Update with your actual file path
data = xr.open_dataset(file_path)

# List of variables
variables = [
    'Elevation', 'slope', 'VegTyp', 'aspect_ratio',
    'aorcFallTemperature', 'aorcFallRain', 'aorcFallHumidity', 'aorcFallLongwave', 'aorcFallShortwave',
    'aorcSpringTemperature', 'aorcSpringRain', 'aorcSpringHumidity', 'aorcSpringLongwave', 'aorcSpringShortwave',
    'aorcSummerTemperature', 'aorcSummerRain', 'aorcSummerHumidity', 'aorcSummerLongwave', 'aorcSummerShortwave',
    'aorcWinterTemperature', 'aorcWinterRain', 'aorcWinterHumidity', 'aorcWinterLongwave', 'aorcWinterShortwave',
    'sweSpring', 'sweWinter', 'DOD'
]

# Determine the number of spatial points
num_pixels = data.dims['npixels']

# Separate variables based on their time dimension
year_variables = [var for var in variables if 'year' in data[var].dims]
nyears_variables = [var for var in variables if 'nyears' in data[var].dims]

# Compute the spatial mean across years for `year` variables
spatial_year_data = {
    var: data[var].mean(dim='year').values.flatten() for var in year_variables
}

# Ensure `nyears` variables are reduced to 1D arrays
spatial_nyears_data = {
    var: data[var].values.flatten() for var in nyears_variables
}

# Combine the two dictionaries
spatial_data = {**spatial_year_data, **spatial_nyears_data}

# Ensure all arrays have the same length
for var, values in spatial_data.items():
    if len(values) != num_pixels:
        print(f"Variable '{var}' has length {len(values)}, expected {num_pixels}. Fixing...")
        spatial_data[var] = values[:num_pixels]  # Trim excess or pad with NaNs
        spatial_data[var] = spatial_data[var].tolist()  # Convert to list for uniformity

# Create a DataFrame for spatial data (rows: coordinates, columns: variables)
spatial_df = pd.DataFrame(spatial_data)

# Compute the correlation matrix
correlation_matrix = spatial_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
plt.title('Spatial Correlation Heatmap of Variables')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Show the plot
plt.show()
