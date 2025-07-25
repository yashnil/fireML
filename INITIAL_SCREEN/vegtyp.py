
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def plot_vegtyp_vs_elevation(final_dataset_path, elev_var="Elevation", veg_var="VegTyp", bin_size=500):
    """
    Plot a stacked bar chart showing the distribution of vegetation types (VegTyp) across elevation bins.
    
    Parameters:
    - final_dataset_path: Path to the NetCDF dataset.
    - elev_var: Elevation variable name.
    - veg_var: Vegetation type variable name.
    - bin_size: Elevation bin width (default: 500m).
    """

    # Open dataset
    ds = xr.open_dataset(final_dataset_path)

    # Extract data
    if elev_var not in ds or veg_var not in ds:
        raise ValueError(f"Error: Missing {elev_var} or {veg_var} in dataset.")

    elev_1d = ds[elev_var].mean(dim="year").values  # (pixel,)
    veg_2d = ds[veg_var].values  # (year, pixel)

    # Ensure dimensions match
    n_years, n_pixels = veg_2d.shape
    veg_1d = veg_2d.flatten()  # Flatten to (year * pixel,)

    # Replicate elevation to match (year * pixel)
    elev_rep = np.repeat(elev_1d, n_years)

    # Remove NaN values
    mask = (~np.isnan(elev_rep)) & (~np.isnan(veg_1d))
    elev_rep = elev_rep[mask]
    veg_1d = veg_1d[mask]

    # Define elevation bins
    min_elev = np.floor(elev_rep.min() / bin_size) * bin_size
    max_elev = np.ceil(elev_rep.max() / bin_size) * bin_size
    bin_edges = np.arange(min_elev, max_elev + bin_size, bin_size)

    # Assign pixels to elevation bins
    bin_indices = np.digitize(elev_rep, bin_edges) - 1  # Adjust for indexing

    # Unique vegetation types
    unique_vegtypes = np.unique(veg_1d)
    veg_counts = {veg_type: np.zeros(len(bin_edges) - 1) for veg_type in unique_vegtypes}

    # Count vegetation occurrences in each elevation bin
    for veg, bin_idx in zip(veg_1d, bin_indices):
        if 0 <= bin_idx < len(bin_edges) - 1:  # Ensure valid bin
            veg_counts[veg][bin_idx] += 1

    # Convert to stacked bar chart format
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}m" for i in range(len(bin_edges)-1)]
    bottom = np.zeros(len(bin_labels))

    plt.figure(figsize=(12, 6))
    
    for veg_type, counts in veg_counts.items():
        plt.bar(bin_labels, counts, bottom=bottom, label=f"Veg {int(veg_type)}")
        bottom += counts  # Stack bars

    plt.xlabel("Elevation Interval (m)")
    plt.ylabel("Count of Pixels")
    plt.title("Vegetation Type Distribution Across Elevation Bins")
    plt.xticks(rotation=45)
    plt.legend(title="VegTyp", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()

# Example usage
final_dataset = "/Users/yashnilmohanty/Desktop/final_dataset.nc"
plot_vegtyp_vs_elevation(final_dataset)
