# visualizations.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

########################################
# Helper: Ensure Elevation is (pixel,) only
########################################
def _get_elevation_1d(ds, elev_var="Elevation"):
    """
    Returns a 1D array of shape (pixel,) for 'Elevation'.
    If ds[elev_var] is (year, pixel), automatically average across 'year'
    so we end up with one static elevation per pixel.
    """
    if elev_var not in ds:
        raise ValueError(f"{elev_var} not found in dataset.")
    elev_data = ds[elev_var]
    # If dimension is (year, pixel), reduce to (pixel,)
    if "year" in elev_data.dims:
        elev_data = elev_data.mean(dim="year")
    return elev_data.values  # shape should now be (pixel,)

########################################
# 1) Plot Terrain Height (Spatial)
########################################
def plot_terrain_spatial(
    final_dataset_path,
    lat_var="latitude",
    lon_var="longitude",
    elev_var="Elevation"
):
    """
    Plot a spatial map of terrain height for these pixels (after filtering).
    We'll treat Elevation as shape (pixel,), averaging across year if needed.
    """
    ds = xr.open_dataset(final_dataset_path)
    
    lat_1d = ds[lat_var].values  # (pixel,)
    lon_1d = ds[lon_var].values  # (pixel,)
    
    # get 1D elevation
    elev_1d = _get_elevation_1d(ds, elev_var=elev_var)
    
    if len(elev_1d) != len(lat_1d):
        raise ValueError(f"Elevation length ({len(elev_1d)}) != lat/lon length ({len(lat_1d)}).")
    
    plt.figure(figsize=(8,6))
    plt.scatter(
        lon_1d, lat_1d, 
        c=elev_1d, s=2, marker='s', 
        cmap='terrain', alpha=1, edgecolors='none'
    )
    plt.colorbar(label='Elevation (m)')
    plt.title("Terrain Height (After Filtering)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

########################################
# 2) Boxplot: DoD vs Burn Fraction (10% intervals)
########################################
def boxplot_dod_vs_burn_intervals(
    final_dataset_path,
    burn_var="burn_fraction",
    dod_var="DOD",
    n_bins=10
):
    """
    Boxplot of DoD vs. BurnFrac in e.g. 10% intervals [0..0.1], [0.1..0.2], ...
    Using all (year, pixel) combos => (year*pixel).
    """
    ds = xr.open_dataset(final_dataset_path)

    if burn_var not in ds or dod_var not in ds:
        print(f"Variables {burn_var} or {dod_var} not found in dataset.")
        return

    burn_2d = ds[burn_var].values  # shape (year, pixel)
    dod_2d  = ds[dod_var].values   # shape (year, pixel)

    # Flatten => (year*pixel,)
    burn_1d = burn_2d.flatten()
    dod_1d  = dod_2d.flatten()

    # remove NaNs
    mask = ~np.isnan(burn_1d) & ~np.isnan(dod_1d)
    burn_1d = burn_1d[mask]
    dod_1d  = dod_1d[mask]

    # bin edges => 0..1 in n_bins intervals
    bin_edges = np.linspace(0, 1, n_bins+1)
    bin_indices = np.digitize(burn_1d, bin_edges)  # range 1..n_bins
    boxplot_data = []
    x_labels = []

    for i in range(1, n_bins+1):
        this_bin_dod = dod_1d[bin_indices == i]
        boxplot_data.append(this_bin_dod)
        label_str = f"{bin_edges[i-1]:.1f}–{bin_edges[i]:.1f}"
        x_labels.append(label_str)

    plt.figure(figsize=(10,5))
    plt.boxplot(boxplot_data, showmeans=True)
    plt.xticks(range(1, n_bins+1), x_labels, rotation=45)
    plt.xlabel("Burn Fraction Interval")
    plt.ylabel("Day of Snow Disappearance (DoD)")
    plt.title("Boxplot: DoD vs. Burn Fraction (10% intervals)")
    plt.tight_layout()
    plt.show()

########################################
# 3a) Boxplot: DoD vs. Elevation intervals
########################################
def boxplot_dod_vs_elev_intervals(
    final_dataset_path,
    dod_var="DOD",
    elev_var="Elevation",
    bin_size=500
):
    """
    Plot DoD as boxplots for each elevation interval (e.g. every 500m).
    Uses all (year, pixel) combos => (year*pixel) for DoD,
    replicating elevation so shape matches.
    """
    ds = xr.open_dataset(final_dataset_path)

    if dod_var not in ds:
        print(f"Error: missing {dod_var} in dataset.")
        return

    # (year, pixel)
    dod_2d = ds[dod_var].values
    n_year, n_pixel = dod_2d.shape
    dod_1d = dod_2d.flatten()

    # Elevation => shape (pixel,)
    try:
        elev_1d = _get_elevation_1d(ds, elev_var=elev_var)
    except ValueError as e:
        print(e)
        return

    if len(elev_1d) != n_pixel:
        raise ValueError(f"Elevation length {len(elev_1d)} != pixel {n_pixel}")

    # replicate to match (year*pixel,)
    elev_rep = np.repeat(elev_1d, n_year)

    # remove NaNs
    mask = (~np.isnan(dod_1d)) & (~np.isnan(elev_rep))
    dod_1d = dod_1d[mask]
    elev_rep = elev_rep[mask]

    min_elev = elev_rep.min()
    max_elev = elev_rep.max()

    # bin edges
    bin_edges = np.arange(min_elev, max_elev + bin_size, bin_size)
    bin_indices = np.digitize(elev_rep, bin_edges)

    boxplot_data = []
    x_labels = []

    for i in range(1, len(bin_edges)):
        this_bin_dod = dod_1d[bin_indices == i]
        boxplot_data.append(this_bin_dod)
        low_edge = bin_edges[i-1]
        high_edge= bin_edges[i]
        x_labels.append(f"{int(low_edge)}–{int(high_edge)} m")

    plt.figure(figsize=(10,5))
    plt.boxplot(boxplot_data, showmeans=True)
    plt.xticks(np.arange(1, len(bin_edges)), x_labels, rotation=45)
    plt.xlabel("Elevation Interval (m)")
    plt.ylabel("Day of Snow Disappearance (DoD)")
    plt.title(f"DoD vs. Elevation intervals ({bin_size}m bins)")
    plt.tight_layout()
    plt.show()

########################################
# 3b) Boxplot: peak SWE vs. Elevation intervals
########################################
def boxplot_peak_swe_vs_elev_intervals(
    final_dataset_path,
    peak_var="peakValue",
    elev_var="Elevation",
    bin_size=500
):
    """
    Plot peak SWE as boxplots for each elevation interval (e.g. each 500m).
    Uses all (year, pixel) combos => flatten (year*pixel).
    """
    ds = xr.open_dataset(final_dataset_path)

    if peak_var not in ds:
        print(f"Error: missing {peak_var} in dataset.")
        return

    peak_2d = ds[peak_var].values   # shape (year, pixel)
    n_year, n_pixel = peak_2d.shape
    peak_1d = peak_2d.flatten()

    # Elevation => shape (pixel,)
    try:
        elev_1d = _get_elevation_1d(ds, elev_var=elev_var)
    except ValueError as e:
        print(e)
        return

    if len(elev_1d) != n_pixel:
        raise ValueError(f"Elevation length {len(elev_1d)} != pixel {n_pixel}")

    elev_rep = np.repeat(elev_1d, n_year)

    # remove NaNs
    mask = (~np.isnan(peak_1d)) & (~np.isnan(elev_rep))
    peak_1d = peak_1d[mask]
    elev_rep = elev_rep[mask]

    min_elev = elev_rep.min()
    max_elev = elev_rep.max()

    bin_edges = np.arange(min_elev, max_elev + bin_size, bin_size)
    bin_indices = np.digitize(elev_rep, bin_edges)

    boxplot_data = []
    x_labels = []

    for i in range(1, len(bin_edges)):
        this_bin_peak = peak_1d[bin_indices == i]
        boxplot_data.append(this_bin_peak)
        low_edge = bin_edges[i-1]
        high_edge= bin_edges[i]
        x_labels.append(f"{int(low_edge)}–{int(high_edge)} m")

    plt.figure(figsize=(10,5))
    plt.boxplot(boxplot_data, showmeans=True)
    plt.xticks(np.arange(1, len(bin_edges)), x_labels, rotation=45)
    plt.xlabel("Elevation Interval (m)")
    plt.ylabel("Peak SWE")
    plt.title(f"Peak SWE vs. Elevation intervals ({bin_size}m bins)")
    plt.tight_layout()
    plt.show()

########################################
# 4) Boxplot: Winter SWE vs. Elevation intervals
########################################
def boxplot_winter_swe_vs_elev_intervals(
    final_dataset_path,
    winter_var="sweWinter",
    elev_var="Elevation",
    bin_size=500
):
    """
    Plot wintertime mean SWE as boxplots for each elevation interval.
    We use all (year, pixel) combos => flatten shape (year*pixel).
    """
    ds = xr.open_dataset(final_dataset_path)

    if winter_var not in ds:
        print(f"Error: missing {winter_var} in dataset.")
        return

    swe_2d = ds[winter_var].values  # shape (year, pixel)
    n_year, n_pixel = swe_2d.shape
    swe_1d = swe_2d.flatten()

    # Elevation => shape (pixel,)
    try:
        elev_1d = _get_elevation_1d(ds, elev_var=elev_var)
    except ValueError as e:
        print(e)
        return

    if len(elev_1d) != n_pixel:
        raise ValueError(f"Elevation length {len(elev_1d)} != pixel {n_pixel}")

    elev_rep = np.repeat(elev_1d, n_year)

    # remove NaNs
    mask = (~np.isnan(swe_1d)) & (~np.isnan(elev_rep))
    swe_1d = swe_1d[mask]
    elev_rep = elev_rep[mask]

    min_elev = elev_rep.min()
    max_elev = elev_rep.max()

    bin_edges = np.arange(min_elev, max_elev + bin_size, bin_size)
    bin_indices = np.digitize(elev_rep, bin_edges)

    boxplot_data = []
    x_labels = []

    for i in range(1, len(bin_edges)):
        this_bin_swe = swe_1d[bin_indices == i]
        boxplot_data.append(this_bin_swe)
        low_edge = bin_edges[i-1]
        high_edge= bin_edges[i]
        x_labels.append(f"{int(low_edge)}–{int(high_edge)} m")

    plt.figure(figsize=(10,5))
    plt.boxplot(boxplot_data, showmeans=True)
    plt.xticks(np.arange(1, len(bin_edges)), x_labels, rotation=45)
    plt.xlabel("Elevation Interval (m)")
    plt.ylabel("Winter SWE")
    plt.title(f"Winter SWE vs. Elevation intervals ({bin_size}m bins)")
    plt.tight_layout()
    plt.show()

########################################
# 5) Plot Peak SWE (Spatial)
########################################
def plot_peak_swe_spatial(
    final_dataset_path,
    lat_var="latitude",
    lon_var="longitude",
    peak_swe_var="peakValue"
):
    """
    Plot a spatial map of peak SWE (max annual SWE) for each pixel.
    We'll treat Peak SWE as shape (year, pixel) and average across years.
    """
    ds = xr.open_dataset(final_dataset_path)
    
    lat_1d = ds[lat_var].values  # (pixel,)
    lon_1d = ds[lon_var].values  # (pixel,)

    if peak_swe_var not in ds:
        raise ValueError(f"{peak_swe_var} not found in dataset.")

    # Compute average peak SWE over years
    peak_swe_2d = ds[peak_swe_var].values  # shape (year, pixel)
    peak_swe_1d = np.nanmean(peak_swe_2d, axis=0)  # shape (pixel,)

    if len(peak_swe_1d) != len(lat_1d):
        raise ValueError(f"Peak SWE length ({len(peak_swe_1d)}) != lat/lon length ({len(lat_1d)}).")

    plt.figure(figsize=(8,6))
    plt.scatter(
        lon_1d, lat_1d, 
        c=peak_swe_1d, s=2, marker='s', 
        cmap='Blues', alpha=1, edgecolors='none'
    )
    plt.colorbar(label='Peak SWE (mm)')
    plt.title("Peak SWE (After Filtering)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

########################################
# 6) Bar Chart: Burned vs. Unburned Pixels by Elevation Bins
########################################
def plot_burned_vs_unburned_elevation_bins(
    final_dataset_path,
    burn_var="burn_fraction",
    elev_var="Elevation",
    bin_size=500,
    burn_threshold=0.7
):
    """
    Plots the number of burned and unburned pixels for different elevation bins.
    A pixel is considered burned if its cumulative burn fraction (sum across 15 years) exceeds burn_threshold.
    """
    ds = xr.open_dataset(final_dataset_path)

    if burn_var not in ds:
        print(f"Error: missing {burn_var} in dataset.")
        return

    # Load burn fraction data (year, pixel)
    burn_2d = ds[burn_var].values  # shape (year, pixel)
    burn_sum_1d = np.nansum(burn_2d, axis=0)  # Cumulative burn over years (pixel,)

    # Load elevation data (pixel,)
    try:
        elev_1d = _get_elevation_1d(ds, elev_var=elev_var)
    except ValueError as e:
        print(e)
        return

    # Classify pixels as burned/unburned
    burned_mask = burn_sum_1d >= burn_threshold
    unburned_mask = ~burned_mask

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
        burned_counts.append(np.sum(bin_mask & burned_mask))
        unburned_counts.append(np.sum(bin_mask & unburned_mask))
        x_labels.append(f"{int(bin_edges[i-1])}–{int(bin_edges[i])} m")

    # Create bar chart
    plt.figure(figsize=(10,5))
    bar_width = 0.4
    x = np.arange(len(x_labels))

    plt.bar(x - bar_width/2, burned_counts, bar_width, label="Burned Pixels", color="red", alpha=0.7)
    plt.bar(x + bar_width/2, unburned_counts, bar_width, label="Unburned Pixels", color="blue", alpha=0.7)

    plt.xticks(x, x_labels, rotation=45)
    plt.xlabel("Elevation Interval (m)")
    plt.ylabel("Number of Pixels")
    plt.title("Burned vs. Unburned Pixels Across Elevation Bins")
    plt.legend()
    plt.tight_layout()
    plt.show()

########################################
# 7) Percent of Burned Pixels by Elevation Bins
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

    # Load burn fraction data (year, pixel)
    burn_2d = ds[burn_var].values  # shape (year, pixel)
    burn_sum_1d = np.nansum(burn_2d, axis=0)  # Cumulative burn over years (pixel,)

    # Load elevation data (pixel,)
    try:
        elev_1d = _get_elevation_1d(ds, elev_var=elev_var)
    except ValueError as e:
        print(e)
        return

    # Classify pixels as burned/unburned
    burned_mask = burn_sum_1d >= burn_threshold

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
        burned_pixels = np.sum(bin_mask & burned_mask)
        
        # Compute percentage (avoid division by zero)
        if total_pixels > 0:
            burned_percent = (burned_pixels / total_pixels) * 100
        else:
            burned_percent = 0
        
        percent_burned.append(burned_percent)
        x_labels.append(f"{int(bin_edges[i-1])}–{int(bin_edges[i])} m")

    # Create bar chart
    plt.figure(figsize=(10,5))
    plt.bar(x_labels, percent_burned, color="red", alpha=0.7)

    plt.xticks(rotation=45)
    plt.xlabel("Elevation Interval (m)")
    plt.ylabel("Percentage of Burned Pixels (%)")
    plt.title("Percent of Pixels Burned Across Elevation Bins")
    plt.tight_layout()
    plt.show()



########################################
# Example usage
########################################
if __name__ == "__main__":
    final_dataset = "/Users/yashnilmohanty/Desktop/final_dataset.nc"

    # 1) Terrain
    plot_terrain_spatial(final_dataset)

    # 2) DoD vs. Burn fraction (10% bins)
    boxplot_dod_vs_burn_intervals(final_dataset)

    # 3) DoD vs. Elevation (500m bins)
    boxplot_dod_vs_elev_intervals(final_dataset, bin_size=500)

    # 4) Peak SWE vs. Elevation
    boxplot_peak_swe_vs_elev_intervals(final_dataset, bin_size=500)

    # 5) Winter SWE vs. Elevation
    boxplot_winter_swe_vs_elev_intervals(final_dataset, bin_size=500)

    # 6) Peak SWE Spatial Map
    plot_peak_swe_spatial(final_dataset)

    # 7) Burned vs. Unburned Pixels by Elevation Bins
    plot_burned_vs_unburned_elevation_bins(final_dataset, bin_size=500, burn_threshold=0.7)

    # 8) Percent of Burned Pixels by Elevation Bins
    plot_percent_burned_pixels_by_elevation(final_dataset, bin_size=500, burn_threshold=0.7)
