
#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd

def generate_veg_elev_tables_visual_screen2(
    dataset_path="/Users/yashnilmohanty/Desktop/final_dataset2.nc",
    veg_var="VegTyp",
    elev_var="Elevation",
    burn_var="burn_fraction",
    burn_threshold=0.7,
    elev_bins=None,
    pixel_area_km2=1.0,
    csv_fraction="veg_elev_fraction_screen2.csv",
    html_fraction="veg_elev_fraction_screen2.html",
    csv_counts="veg_elev_counts_screen2.csv",
    html_counts="veg_elev_counts_screen2.html"
):
    """
    Generates TWO VegType vs. Elevation tables for burned pixels using final_dataset2 (Second Screening):
      1) A fraction table => each cell is (n / N)
         (n = # burned pixels of that VegType & ElevBin, N = total burned).
      2) A counts table => each cell is just n (# burned pixels).

    Both tables are saved as CSV & HTML with color gradients.

    Also prints:
      - N (total burned pixels)
      - sum of fraction table (should ~1)
      - total burned area = pixel_area_km2 * N * mean_burn_fraction
    """

    # 1) Load dataset
    ds = xr.open_dataset(dataset_path)
    print(f"Loaded dataset: {dataset_path}")

    # Extract vegetation data
    veg_data = ds[veg_var]
    if "year" in veg_data.dims:
        # shape: (year, pixel) => pick first year
        veg_types = veg_data.isel(year=0).values
    else:
        # shape: (pixel,)
        veg_types = veg_data.values

    # Extract elevation data
    elev_data = ds[elev_var]
    if "year" in elev_data.dims:
        elevations = elev_data.isel(year=0).values
    else:
        elevations = elev_data.values

    # Extract burn fraction data => shape: (year, pixel)
    burn_2d = ds[burn_var].values
    # 2) Compute cumulative burn fraction => shape (pixel,)
    burn_sum = np.nansum(burn_2d, axis=0)

    # 3) Classify burned pixels
    burned_mask = burn_sum > burn_threshold

    # Basic shape checks
    if burned_mask.shape != veg_types.shape:
        raise ValueError(f"Shape mismatch: burned_mask={burned_mask.shape}, veg_types={veg_types.shape}")
    if burned_mask.shape != elevations.shape:
        raise ValueError(f"Shape mismatch: burned_mask={burned_mask.shape}, elevations={elevations.shape}")

    N = burned_mask.sum()  # total burned pixels
    if N == 0:
        print("No pixels exceed the burn threshold. Table will be empty.")
        return

    mean_burn_fraction = np.nanmean(burn_sum[burned_mask])

    # 4) Define elevation bins if not provided
    if elev_bins is None:
        elev_bins = np.arange(0, 5000, 500)  # 0-500, 500-1000, ...
    elev_labels = [f"{int(elev_bins[i])}-{int(elev_bins[i+1])}"
                   for i in range(len(elev_bins) - 1)]

    # Identify unique vegetation types
    unique_veg = np.unique(veg_types)

    # Build a frequency table (# of burned pixels)
    counts_table = pd.DataFrame(0.0, index=unique_veg, columns=elev_labels)

    # Filter for burned
    burned_veg = veg_types[burned_mask]
    burned_elev = elevations[burned_mask]

    # Digitize elevation
    bin_indices = np.digitize(burned_elev, elev_bins) - 1

    for veg, bin_idx in zip(burned_veg, bin_indices):
        if 0 <= bin_idx < len(elev_labels):
            counts_table.loc[veg, elev_labels[bin_idx]] += 1

    # Convert counts -> fraction (n/N)
    fraction_table = counts_table / N

    # Validate sum of fraction table
    fraction_sum = fraction_table.values.sum()

    # 5) Compute total burned area
    total_burned_area = pixel_area_km2 * N * mean_burn_fraction

    # Save counts table as CSV
    counts_table.to_csv(csv_counts, float_format="%.0f")
    print(f"Saved counts table to CSV: {csv_counts}")

    # Style the counts table with a color gradient
    styled_counts = counts_table.style.background_gradient(cmap="Blues")
    styled_counts.set_caption("VegType × Elevation Burn - Counts Table (Screening #2)")

    styled_counts.to_html(html_counts, float_format="%.0f")
    print(f"Saved counts table to HTML: {html_counts}")

    # Save fraction table as CSV
    fraction_table.to_csv(csv_fraction, float_format="%.4f")
    print(f"Saved fraction table to CSV: {csv_fraction}")

    # Style the fraction table
    styled_fraction = fraction_table.style.background_gradient(cmap="viridis")
    styled_fraction.set_caption("VegType × Elevation Burn - Fraction Table (Screening #2)")

    styled_fraction.to_html(html_fraction, float_format="%.4f")
    print(f"Saved fraction table to HTML: {html_fraction}")

    # Display summary in terminal
    print("\nVegType × Elevation Burn Tables => saved as CSV & HTML (counts & fraction).")
    print(f"Total burned pixels (N) = {N}")
    print(f"Mean burn fraction (burned pixels only) = {mean_burn_fraction:.4f}")
    print(f"Sum of fraction table = {fraction_sum:.4f} (should be ~1.0)")
    print(f"Total burned area = {total_burned_area:.2f} km² "
          f"(pixel_area={pixel_area_km2}, N={N}, mean_burn_fraction={mean_burn_fraction:.4f})\n")

if __name__ == "__main__":
    generate_veg_elev_tables_visual_screen2(
        dataset_path="/Users/yashnilmohanty/Desktop/final_dataset2.nc",
        veg_var="VegTyp",
        elev_var="Elevation",
        burn_var="burn_fraction",
        burn_threshold=0.7,
        elev_bins=np.arange(0, 5000, 500),
        pixel_area_km2=1.0,
        csv_fraction="veg_elev_fraction_screen2.csv",
        html_fraction="veg_elev_fraction_screen2.html",
        csv_counts="veg_elev_counts_screen2.csv",
        html_counts="veg_elev_counts_screen2.html"
    )

