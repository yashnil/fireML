import xarray as xr
import numpy as np

# Open the netCDF file
data = xr.open_dataset("/Users/yashnilmohanty/Desktop/fire.nc")

# Extract the burn_fraction variable
burn_fraction = data["burn_fraction"]

# Calculate the number of pixels where burn fraction > 1
burn_frac_gt_1 = (burn_fraction > 1).sum().item()
# Print the number of pixels where burn fraction > 1
print(f"Number of pixels with burn_fraction > 1: {burn_frac_gt_1}")

# Calculate the number of pixels where burn fraction > 0
burn_frac_gt_0 = (burn_fraction > 0).sum().item()
print(f"Number of pixels with burn_fraction > 0: {burn_frac_gt_0}")

# Calculate the number of pixels where burn fraction == 0
burn_frac_eq_0 = (burn_fraction == 0).sum().item()
print(f"Number of pixels with burn_fraction == 0: {burn_frac_eq_0}")

# Calculate the ratio
burn_frac_ratio = burn_frac_gt_1 / burn_frac_gt_0 if burn_frac_gt_0 > 0 else 0
print(f"Ratio of pixels (BurnFrac > 1) / (BurnFrac > 0): {burn_frac_ratio:.4f}")

# Apply constraint if ratio is < 2%
if burn_frac_ratio < 0.02:
    # Limit burn fraction to a maximum of 1
    burn_fraction = burn_fraction.where(burn_fraction <= 1, other=1)
    print("Constraint applied: Annual burn fraction limited to 1 where necessary.")
else:
    print("No constraint applied.")

# Save the modified burn_fraction back to a new file
data["burn_fraction"] = burn_fraction  # Replace the original variable
data.to_netcdf("/Users/yashnilmohanty/Desktop/fire_modified.nc")
print("Modified dataset saved as fire_modified.nc")
