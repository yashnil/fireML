#!/usr/bin/env python3
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import ranksums  # for Wilcoxon rank-sum tests
import numpy.random as npr  # for random sampling in the final new step
from typing import List, Tuple, Optional

############################################################
# 0) LOAD COORDINATES
############################################################
from obtainCoordinates import coords  # shape(n_pixels, 2)

############################################################
# 1) Pre-2004 Burn
############################################################
def compute_pre2004_burn(coords, path_pattern, year_start=2001, year_end=2003):
    """
    Summation from 2001..2003 for each pixel bounding box.
    """
    import xarray as xr
    n_pixels = len(coords)
    pre_burn = np.zeros((n_pixels,), dtype=np.float32)

    years = range(year_start, year_end + 1)
    months = range(1, 13)

    for yr in years:
        for mm in months:
            file_path = path_pattern.format(year=yr, month=mm)
            ds_nc = xr.open_dataset(file_path)
            burn_2d = ds_nc["MTBS_BurnFraction"].values
            lat_2d = ds_nc["XLAT_M"].values
            lon_2d = ds_nc["XLONG_M"].values

            # Flatten
            flat_burn = burn_2d.ravel()
            flat_lat  = lat_2d.ravel()
            flat_lon  = lon_2d.ravel()

            # Clamp fraction > 1 => 1
            flat_burn = np.minimum(flat_burn, 1.0)

            # Remove NaNs
            valid_mask = np.isfinite(flat_burn) & np.isfinite(flat_lat) & np.isfinite(flat_lon)
            fb  = flat_burn[valid_mask]
            fla = flat_lat[valid_mask]
            flo = flat_lon[valid_mask]

            # Accumulate for each pixel bounding box
            for i, (coord_lat, coord_lon) in enumerate(coords):
                lat_min, lat_max = coord_lat - 0.005, coord_lat + 0.005
                lon_min, lon_max = coord_lon - 0.005, coord_lon + 0.005
                in_box = ((fla >= lat_min) & (fla <= lat_max) &
                          (flo >= lon_min) & (flo <= lon_max))
                box_burn = fb[in_box]
                mean_frac = np.mean(box_burn) if len(box_burn) > 0 else 0.0
                pre_burn[i] += mean_frac

            ds_nc.close()

    # final clamp => <= 1.0
    pre_burn = np.minimum(pre_burn, 1.0)
    return pre_burn

############################################################
# 2) cumsum with initial
############################################################
def compute_burn_cumsum_with_initial(ds, pre_burn):
    """
    ds["burn_fraction"] => shape (year=15, pixel).
    Add pre_burn to year=0, then cumsum across years.
    """
    burn_2d = ds["burn_fraction"].values
    n_years, n_pixels = burn_2d.shape
    cumsum_2d = np.zeros((n_years, n_pixels), dtype=np.float32)

    cumsum_2d[0, :] = pre_burn + burn_2d[0, :]
    for y in range(1, n_years):
        cumsum_2d[y, :] = cumsum_2d[y - 1, :] + burn_2d[y, :]

    return cumsum_2d

############################################################
# 3) Gather features (including burn_fraction)
############################################################
def gather_spatiotemporal_features(ds, target_var="DOD"):
    """
    Collect all data_vars except the target, lat/lon placeholders, etc.
    We keep 'burn_fraction' as a predictor.
    """
    exclude_vars = {
        target_var.lower(),
        'lat', 'lon', 'latitude', 'longitude',
        'pixel', 'year', 'ncoords_vector', 'nyears_vector'
    }
    all_feats = {}
    n_years = ds.dims['year']

    for var_name in ds.data_vars:
        if var_name.lower() in exclude_vars:
            continue
        
        da = ds[var_name]
        dims = set(da.dims)
        # If it's (year, pixel), keep as is
        if dims == {'year', 'pixel'}:
            arr2d = da.values
            all_feats[var_name] = arr2d
        # If it's (pixel), replicate across all years
        elif dims == {'pixel'}:
            arr1d = da.values
            arr2d = np.tile(arr1d, (n_years,1))
            all_feats[var_name] = arr2d

    return all_feats

def flatten_spatiotemporal(ds, target_var="DOD"):
    feat_dict = gather_spatiotemporal_features(ds, target_var=target_var)
    feat_names = sorted(feat_dict.keys())

    X_cols = []
    for fname in feat_names:
        arr2d = feat_dict[fname]
        arr1d = arr2d.ravel(order='C')
        X_cols.append(arr1d)
    X_all = np.column_stack(X_cols)

    dod_2d = ds[target_var].values
    y_all = dod_2d.ravel(order='C')

    valid_mask = (
        ~np.isnan(X_all).any(axis=1) &
        ~np.isnan(y_all)
    )

    return X_all, y_all, feat_names, valid_mask

############################################################
# 4) define 4 categories c0..c3 from cumsum_2d
############################################################
def define_4cats_cumsum(cumsum_2d):
    cat_2d = np.zeros(cumsum_2d.shape, dtype=int)
    c0 = cumsum_2d < 0.25
    c1 = (cumsum_2d >= 0.25) & (cumsum_2d < 0.5)
    c2 = (cumsum_2d >= 0.5) & (cumsum_2d < 0.75)
    c3 = (cumsum_2d >= 0.75)
    cat_2d[c0] = 0
    cat_2d[c1] = 1
    cat_2d[c2] = 2
    cat_2d[c3] = 3
    return cat_2d

############################################################
# 5) Some plotting tools for scatter, hist, etc.
############################################################
def plot_scatter(y_true, y_pred, title="Scatter"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--', label='1:1 line')

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true, y_pred)

    plt.title(f"{title}\nRMSE={rmse:.2f}, Bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD")
    plt.ylabel("Observed DoD")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bias_hist(y_true, y_pred, title="Bias Histogram", x_min=-100, x_max=100):
    residuals = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=50, range=(x_min,x_max), alpha=0.7)

    mean_bias = np.mean(residuals)
    std_bias  = np.std(residuals)

    plt.axvline(mean_bias, color='k', linestyle='dashed', linewidth=2)
    plt.title(f"{title}\nMean={mean_bias:.2f}, Std={std_bias:.2f}")
    plt.xlabel("Bias (Pred - Obs)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_scatter_by_cat(y_true, y_pred, cat, title="Scatter by Category"):
    plt.figure(figsize=(6,6))
    cat_colors = {0:'red', 1:'green', 2:'blue', 3:'orange'}

    for cval in cat_colors.keys():
        mask = (cat == cval)
        if np.any(mask):
            plt.scatter(
                y_pred[mask],
                y_true[mask],
                alpha=0.4,
                color=cat_colors[cval],
                label=f"cat={cval}"
            )

    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--', label='1:1 line')

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true, y_pred)

    plt.title(f"{title}\nRMSE={rmse:.2f}, Bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD")
    plt.ylabel("Observed DoD")
    plt.legend()
    plt.tight_layout()
    plt.show()

############################################################
# 6) Additional Visualization
#    (A) Original pixel-level bias map w/ high-res background
#    (B) Boxplot by elev/veg
#    (C) [NEW] Simple "grid" pixel-level bias map
#    (D) [NEW] Mean predicted/observed DoD maps
############################################################
def produce_pixel_bias_map_hr_background(ds, pixel_idx, y_true, y_pred,
        lat_var="latitude", lon_var="longitude",
        title="Pixel Bias: High-Res Background"):
    """
    Original function that uses Cartopy (Stamen terrain tiles) for a high-res background.
    We'll keep this here for completeness, but we won't call it if we prefer a simpler approach.
    """
    from matplotlib.colors import TwoSlopeNorm
    tiler = cimgt.Stamen('terrain')
    n_pixels = ds.dims["pixel"]
    sum_bias = np.zeros(n_pixels, dtype=float)
    count    = np.zeros(n_pixels, dtype=float)

    residuals = y_pred - y_true
    for i, px in enumerate(pixel_idx):
        sum_bias[px] += residuals[i]
        count[px]    += 1

    mean_bias = np.full(n_pixels, np.nan, dtype=float)
    mask = (count > 0)
    mean_bias[mask] = sum_bias[mask] / count[mask]

    lat_full = ds[lat_var].values
    lon_full = ds[lon_var].values

    # Flatten lat/lon if stored as 2D
    if lat_full.ndim == 2:
        lat_1d_all = lat_full[0, :]
        lon_1d_all = lon_full[0, :]
    else:
        lat_1d_all = lat_full
        lon_1d_all = lon_full

    max_abs = np.nanmax(np.abs(mean_bias))
    if np.isnan(max_abs):
        max_abs = 1.0

    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    fig = plt.figure(figsize=(9,7))
    ax = plt.axes(projection=tiler.crs)

    ax.set_extent([-125, -113, 32, 42], crs=ccrs.PlateCarree())
    ax.add_image(tiler, 8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.5)

    # Plot pixels with no data in light gray
    ax.scatter(
        lon_1d_all, lat_1d_all,
        transform=ccrs.PlateCarree(),
        c="lightgray", s=3, alpha=0.7, label="Unused"
    )
    sc = ax.scatter(
        lon_1d_all[mask],
        lat_1d_all[mask],
        transform=ccrs.PlateCarree(),
        c=mean_bias[mask],
        s=4, cmap="bwr", norm=norm, alpha=1
    )
    cb = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.7)
    cb.set_label("Mean Bias (Pred - Obs)")
    plt.title(title)
    plt.show()

def plot_boxplot_dod_by_elev_veg(y_dod, elev, vegtyp, cat_label="c0"):
    elev_edges = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    elev_bin_idx = np.digitize(elev, elev_edges) - 1
    n_elev_bins = len(elev_edges) - 1

    unique_veg = np.unique(vegtyp)

    box_data = []
    x_labels = []
    for elev_i in range(n_elev_bins):
        for veg_v in unique_veg:
            sel = (elev_bin_idx == elev_i) & (vegtyp == veg_v)
            y_sel = y_dod[sel]
            box_data.append(y_sel)
            low  = elev_edges[elev_i]
            high = elev_edges[elev_i + 1]
            elev_label = f"E[{int(low)}-{int(high)}]"
            veg_label  = f"Veg={veg_v}"
            x_labels.append(f"{elev_label}, {veg_label}")

    plt.figure(figsize=(12,5))
    plt.boxplot(box_data, showmeans=True)
    plt.xticks(range(1, len(box_data)+1), x_labels, rotation=90)
    plt.xlabel("(ElevationBin, VegTyp)")
    plt.ylabel("Raw DoD")
    plt.title(f"{cat_label}: DoD vs. (Elev, Veg)")
    plt.tight_layout()
    plt.show()

# (C) [NEW] Simple grid-based bias map (no Cartopy)
def produce_pixel_bias_map_simple(ds, pixel_idx, y_true, y_pred,
                                  lat_var="latitude", lon_var="longitude",
                                  title="Pixel Bias: Simple Grid"):
    """
    A simpler pixel-level bias map that just does a scatter of (lon, lat) 
    colored by mean bias. Does NOT use Cartopy or any background tiles.
    """
    from matplotlib.colors import TwoSlopeNorm

    n_pixels = ds.dims["pixel"]
    sum_bias = np.zeros(n_pixels, dtype=float)
    count = np.zeros(n_pixels, dtype=float)

    # Accumulate bias per pixel
    residuals = y_pred - y_true
    for i, px in enumerate(pixel_idx):
        sum_bias[px] += residuals[i]
        count[px] += 1
    mean_bias = np.full(n_pixels, np.nan, dtype=float)
    good = (count > 0)
    mean_bias[good] = sum_bias[good] / count[good]

    # Extract lat/lon as 1D
    lat_full = ds[lat_var].values
    lon_full = ds[lon_var].values
    if lat_full.ndim == 2:
        lat_1d = lat_full[0, :]
        lon_1d = lon_full[0, :]
    else:
        lat_1d = lat_full
        lon_1d = lon_full

    # Figure out color scaling
    vmax = np.nanmax(np.abs(mean_bias)) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    plt.figure(figsize=(7,6))
    # All points in light gray
    plt.scatter(lon_1d, lat_1d, c="lightgray", s=5, alpha=0.7, label="No Data")
    # Points with actual data in bwr
    sc = plt.scatter(lon_1d[good], lat_1d[good],
                     c=mean_bias[good], cmap="bwr", norm=norm,
                     s=10, alpha=0.9, label="Bias")

    plt.colorbar(sc, shrink=0.8, label="Mean Bias (Pred - Obs)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# (D) [NEW] Plot mean predicted or observed DoD
def plot_mean_dod_map_simple(ds, mean_vals,
                             lat_var="latitude", lon_var="longitude",
                             title="Mean DoD"):
    """
    Simple scatter map of per-pixel mean DoD (predicted or observed),
    again not using Cartopy.
    """
    lat_full = ds[lat_var].values
    lon_full = ds[lon_var].values
    if lat_full.ndim == 2:
        lat_1d = lat_full[0, :]
        lon_1d = lon_full[0, :]
    else:
        lat_1d = lat_full
        lon_1d = lon_full

    plt.figure(figsize=(7,6))
    sc = plt.scatter(lon_1d, lat_1d, c=mean_vals, cmap="viridis", s=10, alpha=0.9)
    plt.colorbar(sc, shrink=0.8, label="Mean DoD (days)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()

############################################################
# 7) Feature Importance Plot (Random Forest)
############################################################
def plot_top10_features(rf_model, feat_names, title="Top 10 Feature Importances"):
    importances = rf_model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    top10_idx  = idx_sorted[:10]
    top10_vals = importances[top10_idx]
    top10_names= [feat_names[i] for i in top10_idx]

    plt.figure(figsize=(8,4))
    plt.bar(range(len(top10_vals)), top10_vals, align='center')
    plt.xticks(range(len(top10_vals)), top10_names, rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Feature Importance")
    plt.tight_layout()
    plt.show()

def plot_top5_feature_scatter(rf_model, X_valid, y_valid, cat_valid, feat_names,
                              title_prefix="(NewApproach)"):
    importances = rf_model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    top5_idx   = idx_sorted[:5]

    cat_colors = {0:'red', 1:'blue', 2:'green', 3:'purple'}

    for feat_idx in top5_idx:
        fname = feat_names[feat_idx]
        feat_vals = X_valid[:, feat_idx]

        plt.figure(figsize=(6,5))
        for cval, ccolor in cat_colors.items():
            sel_c = (cat_valid == cval)
            plt.scatter(
                y_valid[sel_c],
                feat_vals[sel_c],
                c=ccolor,
                alpha=0.4,
                s=10,
                label=f"cat={cval}"
            )

        mask_lin = np.isfinite(y_valid) & np.isfinite(feat_vals)
        if np.sum(mask_lin) > 2:
            r_val = np.corrcoef(y_valid[mask_lin], feat_vals[mask_lin])[0,1]
        else:
            r_val = np.nan

        plt.legend(title=f"r={r_val:.2f}", loc="best")
        plt.xlabel("Observed DOD")
        plt.ylabel(fname)
        plt.title(f"{title_prefix}: Feature={fname}")
        plt.tight_layout()
        plt.show()

# ------------------------------------------------------------------
# 8‑A) **NEW** helper: metrics for arbitrary burn‑fraction bins
# ------------------------------------------------------------------
def evaluate_metrics_per_bin(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              burn_vals: np.ndarray,
                              bins: list[tuple[float, Optional[float]]],
                              label: str = "BurnFrac") -> None:
    """
    Print RMSE / Bias / R² for each (lo, hi] interval in *bins*.
    A *hi* of None means 'greater than lo'.
    """
    for lo, hi in bins:
        if hi is None:                         # open upper bound
            sel = burn_vals > lo
            tag = f">{lo*100:.0f}%"
        else:
            sel = (burn_vals >= lo) & (burn_vals < hi)
            tag = f"{lo*100:.0f}-{hi*100:.0f}%"
        n = int(np.sum(sel))
        if n == 0:
            print(f"{label} {tag:>7}:  N=0 – skipped")
            continue
        rmse = np.sqrt(mean_squared_error(y_true[sel], y_pred[sel]))
        bias = np.mean(y_pred[sel] - y_true[sel])
        r2   = r2_score(y_true[sel], y_pred[sel]) if n > 1 else np.nan
        print(f"{label} {tag:>7}:  N={n:5d}  RMSE={rmse:6.2f}  "
              f"Bias={bias:7.2f}  R²={r2:6.3f}")

############################################################
# 8) The NEW random forest experiment
#    + Wilcoxon rank-sum test
#    + Additional test using the category with fewest samples
#    + Save predicted/observed DoD arrays to .txt
#    + Now we add simpler spatial maps for bias + mean predicted/observed
############################################################
def run_rf_incl_burn_categorized(
        X_all: np.ndarray,
        y_all: np.ndarray,
        cat_2d: np.ndarray,
        valid_mask: np.ndarray,
        ds: xr.Dataset,
        feat_names: list[str]
    ):
    """
    Train a Random‑Forest (burn_fraction *included* as a predictor),
    evaluate by cumulative‑burn category, and visualise results.

    NEW ➜ after the down‑sampling robustness check, plot histograms of
    the 10 mean‑bias values and 10 RMSE values obtained for every
    category that participates in the check.
    """

    # ──────────────────────────────────────────────────────────────
    # 0.  Flatten arrays & masks
    # ──────────────────────────────────────────────────────────────
    cat_flat   = cat_2d.ravel(order="C")
    cat_valid  = cat_flat[valid_mask]
    X_valid, y_valid = X_all[valid_mask], y_all[valid_mask]

    # ──────────────────────────────────────────────────────────────
    # 1.  70 / 30 split *within* each category (c0‑c3)
    # ──────────────────────────────────────────────────────────────
    train_idx, test_idx = [], []
    for c in (0, 1, 2, 3):
        rows = np.where(cat_valid == c)[0]
        if rows.size == 0:
            print(f"cat={c}: no valid rows – skipped");  continue
        tr, te = train_test_split(rows, test_size=0.30, random_state=42)
        train_idx.append(tr);  test_idx.append(te)

    if not train_idx:
        print("No training data – abort.");  return None

    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)

    X_tr, y_tr = X_valid[train_idx], y_valid[train_idx]
    X_te, y_te = X_valid[test_idx],  y_valid[test_idx]

    # ──────────────────────────────────────────────────────────────
    # 2.  Train RF
    # ──────────────────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    # basic plots
    plot_scatter(y_tr, rf.predict(X_tr), "RF: Train (70 % each cat)")
    plot_bias_hist(y_tr, rf.predict(X_tr), "RF Bias Hist: Train")

    y_pred_te = rf.predict(X_te)
    plot_scatter(y_te, y_pred_te, "RF: Test (30 % each cat)")
    plot_bias_hist(y_te, y_pred_te, "RF Bias Hist: Test (all cats)")

    # ──────────────────────────────────────────────────────────────
    # 3.  Wilcoxon rank‑sum (bias c1/2/3 vs c0)
    # ──────────────────────────────────────────────────────────────
    cat_test = cat_valid[test_idx]
    bias_all = y_pred_te - y_te
    bias_by_cat = {c: bias_all[cat_test == c] for c in range(4)
                   if (cat_test == c).any()}

    if 0 in bias_by_cat:
        print("\nWilcoxon rank‑sum: bias(cX) vs bias(c0)")
        for c in (1, 2, 3):
            if c in bias_by_cat:
                stat, p = ranksums(bias_by_cat[0], bias_by_cat[c])
                print(f"  c{c} vs c0 → stat={stat:.3f}, p={p:.3g}")
            else:
                print(f"  c{c} vs c0 → no samples – skipped")

    # ──────────────────────────────────────────────────────────────
    # 4.  Down‑sampling robustness check  (NEW histograms)
    # ──────────────────────────────────────────────────────────────
    counts   = {c: (cat_test == c).sum() for c in range(4) if (cat_test == c).any()}
    if counts:
        k_min   = min(counts.values())
        print(f"\nDown‑sampling robustness: min category size = {k_min}")

        bias_dist, rmse_dist = {}, {}

        for c, n_c in counts.items():
            idx_c = np.where(cat_test == c)[0]
            if n_c < k_min:                # should not happen, but guard
                continue
            biases, rmses = [], []
            for _ in range(10):
                sub_idx = npr.choice(idx_c, size=k_min, replace=False)
                y_sub   = y_te[sub_idx]
                y_pred  = y_pred_te[sub_idx]
                biases.append((y_pred - y_sub).mean())
                rmses.append(np.sqrt(mean_squared_error(y_sub, y_pred)))
            bias_dist[c] = biases
            rmse_dist[c] = rmses
            print(f"  cat={c}: mean(μ_bias)={np.mean(biases):.3f}, "
                  f"mean(RMSE)={np.mean(rmses):.3f}")

        # ── plot histograms
        for c in bias_dist:
            plt.figure(figsize=(10,4))

            plt.subplot(1,2,1)
            plt.hist(bias_dist[c], bins=5, color="steelblue", alpha=0.8)
            plt.axvline(np.mean(bias_dist[c]), color="k", ls="--")
            plt.title(f"cat={c} – Mean Bias (10 subsamples)")
            plt.xlabel("Mean Bias");  plt.ylabel("Count")

            plt.subplot(1,2,2)
            plt.hist(rmse_dist[c], bins=5, color="tomato", alpha=0.8)
            plt.axvline(np.mean(rmse_dist[c]), color="k", ls="--")
            plt.title(f"cat={c} – RMSE (10 subsamples)")
            plt.xlabel("RMSE");  plt.ylabel("Count")

            plt.suptitle(f"Down‑sampling distributions (k={k_min})")
            plt.tight_layout();  plt.show()

    # ──────────────────────────────────────────────────────────────
    # 5.  Save per‑cat obs / pred, per‑cat plots (unchanged)
    # ──────────────────────────────────────────────────────────────
    # …  (retain your existing code for saving .txt, per‑cat scatter,
    #     colour‑coded scatter, 10 % burn‑fraction bins, bias maps,
    #     mean‑DoD maps, feature importance, etc.) …
    # -----------------------------------------------------------------
    # return fitted model
    return rf



############################################################
# MAIN
############################################################
if __name__ == "__main__":
    path_pat = "/Users/yashnilmohanty/Desktop/data/BurnArea_Data/Merged_BurnArea_{year:04d}{month:02d}.nc"
    pre_burn = compute_pre2004_burn(coords, path_pat, 2001, 2003)

    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset3.nc")
    cumsum_2d = compute_burn_cumsum_with_initial(ds, pre_burn)
    ds["burn_cumsum"] = (("year", "pixel"), cumsum_2d)
    cat_2d = define_4cats_cumsum(cumsum_2d)

    X_all, y_all, feat_names, valid_mask = flatten_spatiotemporal(ds, target_var="DOD")
    print("Feature names:", feat_names)

    rf_model = run_rf_incl_burn_categorized(
        X_all, y_all, cat_2d, valid_mask, ds, feat_names
    )
    print("DONE.")
