
import xarray as xr

class NetCDFDataAccessor:
    def __init__(self, file_path):
        # Open the NetCDF file
        self.dataset = xr.open_dataset(file_path)
    
    def get_variable(self, var_name):
        """Fetches a specific variable from the NetCDF file."""
        if var_name in self.dataset.variables:
            return self.dataset[var_name]
        else:
            raise ValueError(f"Variable '{var_name}' not found in the dataset.")

    def list_variables(self):
        """Lists all available variables in the NetCDF file."""
        return list(self.dataset.variables.keys())

# Example usage (for testing purposes)
if __name__ == "__main__":
    file_path = "/Users/yashnilmohanty/Desktop/combined_data.nc"  # Replace with your file path
    accessor = NetCDFDataAccessor(file_path)
    
    # List all variables
    print("Available Variables:")
    print(accessor.list_variables())

    # Access a specific variable
    try:
        fall_humidity = accessor.get_variable('aorcFallHumidity')
        print("\nVariable 'aorcFallHumidity':")
        print(fall_humidity)
    except ValueError as e:
        print(e)

    # Access another variable
    try:
        elevation = accessor.get_variable('Elevation')
        print("\nVariable 'Elevation':")
        print(elevation)
    except ValueError as e:
        print(e)
