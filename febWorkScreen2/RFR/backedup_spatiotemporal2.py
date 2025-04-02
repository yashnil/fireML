#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors  # for TwoSlopeNorm

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

########################
# 0) LOAD COORDINATES
########################
from obtainCoordinates import coords  # shape(n_pixels,2)

########################
# 1) Pre-2004 Burn
########################
def compute_pre2004_burn(coords, path_pattern, year_start=2001, year_end=2003):
    """
    Summation from 2001..2003 for each pixel bounding box.
    """
    n_pixels = len(coords)
    pre_burn = np.zeros((n_pixels,), dtype=np.float32)

    years = range(year_start, year_end+1)
    months= range(1,13)

    for yr in years:
        for mm in months:
            file_path = path_pattern.format(year=yr, month=mm)
            ds_nc = xr.open_dataset(file_path)
            burn_2d = ds_nc["MTBS_BurnFraction"].values
            lat_2d  = ds_nc["XLAT_M"].values
            lon_2d  = ds_nc["XLONG_M"].values

            # flatten
            flat_burn = burn_2d.ravel()
            flat_lat  = lat_2d.ravel()
            flat_lon  = lon_2d.ravel()

            # clamp fraction>1 =>1
            flat_burn = np.minimum(flat_burn, 1.0)

            # remove NaNs
            valid_mask = (
                np.isfinite(flat_burn) &
                np.isfinite(flat_lat) &
                np.isfinite(flat_lon)
            )
            fb  = flat_burn[valid_mask]
            fla = flat_lat[valid_mask]
            flo = flat_lon[valid_mask]

            # accumulate for each pixel bounding box
            for i, (coord_lat, coord_lon) in enumerate(coords):
                lat_min, lat_max = coord_lat - 0.005, coord_lat + 0.005
                lon_min, lon_max = coord_lon - 0.005, coord_lon + 0.005
                in_box = (
                    (fla >= lat_min) & (fla <= lat_max) &
                    (flo >= lon_min) & (flo <= lon_max)
                )
                box_burn = fb[in_box]
                if len(box_burn) > 0:
                    mean_frac = np.mean(box_burn)
                else:
                    mean_frac = 0.0
                pre_burn[i] += mean_frac

            ds_nc.close()

    # final clamp => <=1.0 if desired
    pre_burn = np.minimum(pre_burn, 1.0)
    return pre_burn

########################
# 2) cumsum with initial
########################
def compute_burn_cumsum_with_initial(ds, pre_burn):
    burn_2d = ds["burn_fraction"].values  # shape(15, pixel)
    n_years, n_pixels = burn_2d.shape
    cumsum_2d = np.zeros((n_years, n_pixels), dtype=np.float32)

    # year=0 => 2004
    cumsum_2d[0, :] = pre_burn + burn_2d[0, :]

    for y in range(1, n_years):
        cumsum_2d[y, :] = cumsum_2d[y - 1, :] + burn_2d[y, :]

    return cumsum_2d

########################
# 3) Gather features
########################
def gather_spatiotemporal_features(ds, target_var="DOD"):
    exclude_vars = {
        target_var.lower(),
        'lat','lon','latitude','longitude',
        'pixel','year','ncoords_vector','nyears_vector'
    }
    all_feats = {}
    ny = ds.dims['year']

    for var in ds.data_vars:
        if var.lower() in exclude_vars:
            continue
        da = ds[var]
        dims = set(da.dims)
        if dims == {'year', 'pixel'}:
            # 2D variable => shape (year, pixel)
            arr2d = da.values
            all_feats[var] = arr2d
        elif dims == {'pixel'}:
            # 1D variable => replicate across all years
            arr1d = da.values
            arr2d = np.tile(arr1d, (ny,1))
            all_feats[var] = arr2d

    return all_feats

def flatten_spatiotemporal(ds, target_var="DOD"):
    feat_dict = gather_spatiotemporal_features(ds, target_var=target_var)
    feat_names = sorted(feat_dict.keys())

    X_cols = []
    for f in feat_names:
        arr2d = feat_dict[f]  # shape(year, pixel)
        arr1d = arr2d.ravel(order='C')
        X_cols.append(arr1d)

    X_all = np.column_stack(X_cols)

    # Flatten target
    dod2d = ds[target_var].values
    y_all = dod2d.ravel(order='C')

    # valid mask
    valid_mask = (
        ~np.isnan(X_all).any(axis=1) &
        ~np.isnan(y_all)
    )

    n_years = ds.dims['year']
    n_pixels = ds.dims['pixel']
    return X_all, y_all, feat_names, valid_mask, n_years, n_pixels

########################
# 4) define c0..c3
########################
def define_4cats_cumsum(cumsum_2d):
    cat_2d = np.zeros(cumsum_2d.shape, dtype=int)
    c0 = cumsum_2d < 0.25
    c1 = (cumsum_2d >= 0.25) & (cumsum_2d < 0.5)
    c2 = (cumsum_2d >= 0.5) & (cumsum_2d < 0.75)
    c3 = cumsum_2d >= 0.75
    cat_2d[c0] = 0
    cat_2d[c1] = 1
    cat_2d[c2] = 2
    cat_2d[c3] = 3
    return cat_2d

########################
# 5) scatter
########################
def plot_scatter(y_true, y_pred, title="Scatter"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--', label='1:1 line')

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    r2   = r2_score(y_true, y_pred)

    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD")
    plt.ylabel("Observed DoD")
    plt.legend()
    plt.tight_layout()
    plt.show()

########################
# ELEV + VEG => one boxplot
########################
def plot_boxplot_dod_by_elev_veg(y_dod, elev, vegtyp, cat_label="c0"):
    elev_edges = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    elev_bin_idx = np.digitize(elev, elev_edges) - 1
    n_elev_bins = len(elev_edges) - 1

    unique_veg = np.unique(vegtyp)

    box_data = []
    x_labels = []
    for elev_i in range(n_elev_bins):
        for veg_v in unique_veg:
            sel = (
                (elev_bin_idx == elev_i) &
                (vegtyp == veg_v)
            )
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

########################
# BIAS HIST
########################
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

########################
# High-Res Satellite Map
########################
def produce_pixel_bias_map_hr_background(
        ds, pixel_idx, y_true, y_pred,
        lat_var="latitude", lon_var="longitude",
        title="Pixel Bias: High-Res Background"
    ):
    """
    Overlays the gridpoints onto a high-res Stamen Terrain map.
    (Requires internet at runtime.)
    """
    from matplotlib.colors import TwoSlopeNorm

    # 1) Prepare tile source
    #    Some versions of Cartopy have cimgt.StamenTerrain,
    #    others use cimgt.Stamen('terrain'). We'll do the latter:
    tiler = cimgt.Stamen('terrain')

    n_pixels = ds.dims["pixel"]
    sum_bias = np.zeros(n_pixels, dtype=float)
    count    = np.zeros(n_pixels, dtype=float)

    # 2) Compute mean bias for each pixel
    residuals = y_pred - y_true
    for i, px in enumerate(pixel_idx):
        sum_bias[px] += residuals[i]
        count[px]    += 1

    mean_bias = np.full(n_pixels, np.nan, dtype=float)
    mask = (count > 0)
    mean_bias[mask] = sum_bias[mask] / count[mask]

    # 3) Get lat/lon arrays
    lat_full = ds[lat_var].values
    lon_full = ds[lon_var].values

    if lat_full.ndim == 2:
        lat_1d_all = lat_full[0, :]
        lon_1d_all = lon_full[0, :]
    else:
        lat_1d_all = lat_full
        lon_1d_all = lon_full

    # 4) Range for color scale
    max_abs = np.nanmax(np.abs(mean_bias))
    if np.isnan(max_abs):
        max_abs = 1.0

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    # 5) Set up figure with Web Mercator from the tile
    fig = plt.figure(figsize=(9,7))
    ax = plt.axes(projection=tiler.crs)

    # bounding box for lat/lon => transform=PlateCarree
    ax.set_extent([-125, -113, 32, 42], crs=ccrs.PlateCarree())

    # add the tile layer => zoom=8
    ax.add_image(tiler, 8)

    # optionally add states, borders
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.5)

    # 6) Plot all pixels in light gray
    ax.scatter(
        lon_1d_all, lat_1d_all,
        transform=ccrs.PlateCarree(),
        c="lightgray", s=3, alpha=0.7,
        label="Unused"
    )

    # 7) Overplot used pixels
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

########################
# Feature Importance
########################
def plot_top10_features(rf_model, feat_names, title="Top 10 Feature Importances"):
    importances = rf_model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]  # descending
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

########################
# (C) Plot top-5 feature-value vs. DOD
########################
def plot_top5_feature_scatter(rf_model, X_valid, y_valid, cat_valid, feat_names,
                              title_prefix="(thr=0)"):
    """
    1) Identify top-5 most important features from the trained RF model.
    2) For each of those 5 features, create a scatterplot:
       - x-axis = Observed DOD
       - y-axis = Feature Value
       - Color-coded by category (0..3)
       - A dotted line of best fit for the entire data (all categories together).
    """
    importances = rf_model.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]  # descending
    top5_idx   = idx_sorted[:5]  # top-5

    cat_colors = {0:'red', 1:'blue', 2:'green', 3:'purple'}

    for feat_idx in top5_idx:
        fname = feat_names[feat_idx]
        feat_vals = X_valid[:, feat_idx]  # shape (#valid,)

        plt.figure(figsize=(6,5))
        # Scatter by category
        for cval, ccolor in cat_colors.items():
            sel_c = (cat_valid == cval)
            plt.scatter(
                y_valid[sel_c],  # x-axis => Observed DOD
                feat_vals[sel_c],# y-axis => Feature Value
                c=ccolor,
                alpha=0.4,
                label=f"cat={cval}"
            )
        # line of best fit => across ALL data
        x_all = y_valid
        y_all = feat_vals
        mask_lin = np.isfinite(x_all) & np.isfinite(y_all)
        if np.sum(mask_lin) > 2:
            x_lin = x_all[mask_lin]
            y_lin = y_all[mask_lin]
            p = np.polyfit(x_lin, y_lin, 1)  # slope, intercept
            x_min, x_max = np.min(x_lin), np.max(x_lin)
            x_vals = np.linspace(x_min, x_max, 100)
            y_fit  = np.polyval(p, x_vals)
            plt.plot(x_vals, y_fit, 'k--', label=f"Best fit (y={p[0]:.2f}x+{p[1]:.2f})")

        plt.xlabel("Observed DOD")
        plt.ylabel(f"{fname}")
        plt.title(f"{title_prefix}: Feature={fname}")
        plt.legend()
        plt.tight_layout()
        plt.show()

########################
# 8) run experiment => random forest
########################
def run_spatiotemporal_experiment(X_all, y_all, valid_mask, cat_2d,
                                  unburned_max_cat=0, ds=None,
                                  feat_names=None):
    """
    Trains on the 'unburned' subset (up to cat <= unburned_max_cat),
    then evaluates on unburned test, each cat separately,
    plots all data combined, returns fitted model.
    """
    cat_flat  = cat_2d.ravel(order='C')
    cat_valid = cat_flat[valid_mask]
    X_valid   = X_all[valid_mask]
    y_valid   = y_all[valid_mask]

    # Flatten Elevation + VegTyp
    elev_2d = ds["Elevation"].values
    veg_2d  = ds["VegTyp"].values
    pixel_idx_full = np.tile(np.arange(ds.dims["pixel"]), ds.dims["year"])

    elev_1d_full = elev_2d.ravel(order='C')
    veg_1d_full  = veg_2d.ravel(order='C')

    elev_valid = elev_1d_full[valid_mask]
    veg_valid  = veg_1d_full[valid_mask]
    pix_valid  = pixel_idx_full[valid_mask]

    is_unburned = (cat_valid <= unburned_max_cat)
    n_unburn    = np.sum(is_unburned)
    print(f" Train threshold => unburned up to cat={unburned_max_cat}, #unburned={n_unburn}")

    if n_unburn == 0:
        print("No unburned => can't train.")
        return None

    X_ub = X_valid[is_unburned]
    y_ub = y_valid[is_unburned]

    X_train, X_test, y_train, y_test = train_test_split(
        X_ub, y_ub, test_size=0.3, random_state=42
    )
    print(f"   unburned train={len(X_train)}, test={len(X_test)}")

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate on unburned TRAIN
    y_pred_train = rf.predict(X_train)
    plot_scatter(y_train, y_pred_train, f"Unburned Train (cat<={unburned_max_cat})")
    plot_bias_hist(y_train, y_pred_train,
                   title=f"Bias Hist: Unburned Train (cat<={unburned_max_cat})")

    # Evaluate on unburned TEST
    y_pred_ub = rf.predict(X_test)
    plot_scatter(y_test, y_pred_ub, f"Unburned Test (cat<={unburned_max_cat})")
    plot_bias_hist(y_test, y_pred_ub,
                   title=f"Bias Hist: Unburned Test (cat<={unburned_max_cat})")

    # Evaluate on each cat=0..3
    for cval in [0,1,2,3]:
        sel_cat = (cat_valid == cval)
        if np.sum(sel_cat) == 0:
            print(f"No samples cat={cval}")
            continue
        X_c = X_valid[sel_cat]
        y_c = y_valid[sel_cat]
        y_pred_c = rf.predict(X_c)

        plot_scatter(y_c, y_pred_c, title=f"Category={cval}")
        plot_bias_hist(y_c, y_pred_c, title=f"Bias Hist: cat={cval}")

        # pixel-level bias map
        px_c = pix_valid[sel_cat]
        produce_pixel_bias_map_hr_background(ds, px_c, y_c, y_pred_c,
            title=f"Pixel Bias: cat={cval}")

        # Combined boxplot for (ElevationBin, VegTyp)
        cat_label = f"cat={cval}, thr={unburned_max_cat}"
        y_dod_cat = y_valid[sel_cat]
        elev_cat  = elev_valid[sel_cat]
        veg_cat   = veg_valid[sel_cat]
        plot_boxplot_dod_by_elev_veg(y_dod_cat, elev_cat, veg_cat,
                                     cat_label=cat_label)

    # Evaluate on ALL valid data
    y_pred_all = rf.predict(X_valid)
    plot_scatter(y_valid, y_pred_all,
                 title=f"All Data (thr={unburned_max_cat})")
    plot_bias_hist(y_valid, y_pred_all,
                   title=f"Bias Hist: All Data (thr={unburned_max_cat})")
    produce_pixel_bias_map_hr_background(ds, pix_valid, y_valid, y_pred_all,
        title=f"Pixel Bias: All Data (thr={unburned_max_cat})")

    return rf

###############################
# MAIN
###############################
if __name__=="__main__":

    # 1) LOAD/COMPUTE PRE-BURN
    path_pat = "/Users/yashnilmohanty/Desktop/data/BurnArea_Data/Merged_BurnArea_{year:04d}{month:02d}.nc"
    pre_burn  = compute_pre2004_burn(coords, path_pat, 2001, 2003)

    # 2) LOAD DS
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset2.nc")

    cumsum_2d = compute_burn_cumsum_with_initial(ds, pre_burn)
    ds["burn_cumsum"] = (("year","pixel"), cumsum_2d)

    cat_2d = define_4cats_cumsum(cumsum_2d)

    # Flatten => X_all, y_all, feat_names, ...
    X_all, y_all, feat_names, valid_mask, n_years, n_pixels = flatten_spatiotemporal(ds,"DOD")

    # RUN #1 => thr=0 => unburned cat=0
    print("\n==== RUN #1 => unburned= cat=0 (<0.25) ====")
    rf_run1 = run_spatiotemporal_experiment(
        X_all, y_all, valid_mask, cat_2d,
        unburned_max_cat=0, ds=ds,
        feat_names=feat_names
    )

    if rf_run1 is not None:
        plot_top10_features(rf_run1, feat_names, title="Top 10 Features (thr=0)")
        # top-5 scatter => (thr=0)
        cat_flat = cat_2d.ravel(order='C')[valid_mask]
        X_valid  = X_all[valid_mask]
        y_validD = y_all[valid_mask]
        plot_top5_feature_scatter(rf_run1, X_valid, y_validD, cat_flat,
                                  feat_names, title_prefix="(thr=0)")

    # RUN #2 => thr=0.5 => unburned cat=0..1
    print("\n==== RUN #2 => unburned= cat=0,1 (<0.5) ====")
    rf_run2 = run_spatiotemporal_experiment(
        X_all, y_all, valid_mask, cat_2d,
        unburned_max_cat=1, ds=ds,
        feat_names=feat_names
    )

    if rf_run2 is not None:
        plot_top10_features(rf_run2, feat_names, title="Top 10 Features (thr=0.5)")
        # top-5 scatter => (thr=0.5)
        cat_flat = cat_2d.ravel(order='C')[valid_mask]
        X_valid  = X_all[valid_mask]
        y_validD = y_all[valid_mask]
        plot_top5_feature_scatter(rf_run2, X_valid, y_validD, cat_flat,
                                  feat_names, title_prefix="(thr=0.5)")

    print("DONE.")
