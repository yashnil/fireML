import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

########################################
# 1) Plot Cumulative Burn Sum (Spatial)
########################################
def plot_cumulative_burn_sum_spatial(
    final_dataset_path,
    lat_var="latitude",
    lon_var="longitude",
    burn_var="burn_fraction"
):
    """
    Plot a spatial map of cumulative burn sum over 15 years.
    Uses a reverse inferno colormap (black -> purple -> red -> orange -> yellow).
    """
    ds = xr.open_dataset(final_dataset_path)
    
    lat_1d = ds[lat_var].values  # (pixel,)
    lon_1d = ds[lon_var].values  # (pixel,)

    if burn_var not in ds:
        raise ValueError(f"{burn_var} not found in dataset.")

    # Compute cumulative burn fraction over 15 years
    burn_2d = ds[burn_var].values  # shape (year, pixel)
    burn_sum_1d = np.nansum(burn_2d, axis=0)  # Sum across years (pixel,)

    if len(burn_sum_1d) != len(lat_1d):
        raise ValueError(f"Burn Sum length ({len(burn_sum_1d)}) != lat/lon length ({len(lat_1d)}).")

    # Define reverse inferno colormap (yellow = low, black = high)
    cmap = plt.cm.inferno.reversed()

    plt.figure(figsize=(8,6))
    plt.scatter(
        lon_1d, lat_1d, 
        c=burn_sum_1d, s=2, marker='s', 
        cmap=cmap, alpha=1, edgecolors='none'
    )
    plt.colorbar(label='Cumulative Burn Fraction (15 years)')
    plt.title("Cumulative Burn Sum (Second Dimensionality Reduction Dataset)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

########################################
# 2) Annual Burned vs. Unburned Pixels per Elevation Bin
########################################
def plot_burned_unburned_pixel_counts_by_elevation(
    final_dataset_path,
    burn_var="burn_fraction",
    elev_var="Elevation",
    bin_size=500,
    burn_threshold=0.7
):
    """
    Plots the count of burned and unburned pixels for each elevation bin.
    A pixel is classified as burned if its cumulative burn fraction exceeds burn_threshold.
    Bars are **side-by-side** (Burned = Red, Unburned = Blue).
    """
    ds = xr.open_dataset(final_dataset_path)

    if burn_var not in ds:
        print(f"Error: missing {burn_var} in dataset.")
        return

    burn_2d = ds[burn_var].values  # shape (year, pixel)
    burn_sum_1d = np.nansum(burn_2d, axis=0)  # Sum across years (pixel,)

    # Load elevation data (pixel,)
    elev_1d = ds[elev_var].values  # (pixel,)

    min_elev = elev_1d.min()
    max_elev = elev_1d.max()

    # Define elevation bins
    bin_edges = np.arange(min_elev, max_elev + bin_size, bin_size)
    bin_indices = np.digitize(elev_1d, bin_edges)

    burned_counts = []
    unburned_counts = []
    x_labels = []

    for i in range(1, len(bin_edges)):
        bin_mask = bin_indices == i
        burned_pixels = np.sum(bin_mask & (burn_sum_1d >= burn_threshold))
        unburned_pixels = np.sum(bin_mask & (burn_sum_1d < burn_threshold))

        burned_counts.append(burned_pixels)
        unburned_counts.append(unburned_pixels)
        x_labels.append(f"{int(bin_edges[i-1])}–{int(bin_edges[i])} m")

    # Define bar width
    bar_width = 0.4
    x_pos = np.arange(len(x_labels))

    # Create side-by-side bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(x_pos - bar_width/2, unburned_counts, width=bar_width, color="blue", alpha=0.7, label="Unburned Pixels")
    plt.bar(x_pos + bar_width/2, burned_counts, width=bar_width, color="red", alpha=0.7, label="Burned Pixels")

    plt.xticks(x_pos, x_labels, rotation=45)
    plt.xlabel("Elevation Interval (m)")
    plt.ylabel("Pixel Count")
    plt.title("Burned vs. Unburned Pixels by Elevation Bin (Second Dimensionality Reduction Dataset)")
    plt.legend()
    plt.tight_layout()
    plt.show()

########################################
# 3) Percentage of Burned Pixels per Elevation Bin
########################################
def plot_percent_burned_pixels_by_elevation(
    final_dataset_path,
    burn_var="burn_fraction",
    elev_var="Elevation",
    bin_size=500,
    burn_threshold=0.7
):
    """
    Plots the percentage of pixels in each elevation bin that are classified as burned.
    A pixel is considered burned if its cumulative burn fraction (sum across 15 years) exceeds burn_threshold.
    """
    ds = xr.open_dataset(final_dataset_path)

    if burn_var not in ds:
        print(f"Error: missing {burn_var} in dataset.")
        return

    burn_2d = ds[burn_var].values  # shape (year, pixel)
    burn_sum_1d = np.nansum(burn_2d, axis=0)  # Cumulative burn over years (pixel,)

    # Load elevation data (pixel,)
    elev_1d = ds[elev_var].values  # (pixel,)

    min_elev = elev_1d.min()
    max_elev = elev_1d.max()

    # Define elevation bins
    bin_edges = np.arange(min_elev, max_elev + bin_size, bin_size)
    bin_indices = np.digitize(elev_1d, bin_edges)

    percent_burned = []
    x_labels = []

    for i in range(1, len(bin_edges)):
        bin_mask = bin_indices == i
        total_pixels = np.sum(bin_mask)
        burned_pixels = np.sum(bin_mask & (burn_sum_1d >= burn_threshold))

        # Compute percentage (avoid division by zero)
        burned_percent = (burned_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        percent_burned.append(burned_percent)
        x_labels.append(f"{int(bin_edges[i-1])}–{int(bin_edges[i])} m")

    # Create bar chart
    plt.figure(figsize=(10,5))
    plt.bar(x_labels, percent_burned, color="red", alpha=0.7)

    plt.xticks(rotation=45)
    plt.xlabel("Elevation Interval (m)")
    plt.ylabel("Percentage of Burned Pixels (%)")
    plt.title("Percent of Pixels Burned Across Elevation Bins (Second Dimensionality Reduction Dataset)")
    plt.tight_layout()
    plt.show()

########################################
# Example usage
########################################
if __name__ == "__main__":
    final_dataset2 = "/Users/yashnilmohanty/Desktop/final_dataset2.nc"

    # 1) Cumulative Burn Sum Spatial Plot
    plot_cumulative_burn_sum_spatial(final_dataset2)

    # 2) Burned vs. Unburned Pixels by Elevation
    plot_burned_unburned_pixel_counts_by_elevation(final_dataset2, bin_size=500, burn_threshold=0.7)

    # 3) Percent Burned Pixels by Elevation
    plot_percent_burned_pixels_by_elevation(final_dataset2, bin_size=500, burn_threshold=0.7)
