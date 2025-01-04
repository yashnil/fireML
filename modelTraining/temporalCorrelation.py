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

# Separate variables based on their dimensions
nyears_variables = [var for var in variables if 'nyears' in data[var].dims]

# Check if any dynamic variables exist
if not nyears_variables:
    print("No dynamic variables found with 'nyears' dimension.")
else:
    # Compute temporal means for dynamic variables across spatial points
    temporal_nyears_data = {
        var: data[var].mean(dim='npixels').values.flatten() for var in nyears_variables
    }

    # Create a DataFrame for temporal data (rows: years, columns: variables)
    temporal_df = pd.DataFrame(temporal_nyears_data, index=range(data.sizes['nyears']))
    temporal_df.index.name = 'Year'

    # Compute the correlation matrix
    correlation_matrix = temporal_df.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
    plt.title('Temporal Correlation Heatmap of Dynamic Variables')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Show the plot
    plt.show()
