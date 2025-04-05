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

############################################################
# 0) LOAD COORDINATES
############################################################
from obtainCoordinates import coords  # shape(n_pixels,2)

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
    months= range(1, 13)

    for yr in years:
        for mm in months:
            file_path = path_pattern.format(year=yr, month=mm)
            ds_nc = xr.open_dataset(file_path)
            burn_2d = ds_nc["MTBS_BurnFraction"].values  # (ny, nx)
            lat_2d  = ds_nc["XLAT_M"].values
            lon_2d  = ds_nc["XLONG_M"].values

            # Flatten
            flat_burn = burn_2d.ravel()
            flat_lat  = lat_2d.ravel()
            flat_lon  = lon_2d.ravel()

            # Clamp fraction>1 =>1
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
                in_box = (fla >= lat_min) & (fla <= lat_max) & (flo >= lon_min) & (flo <= lon_max)
                box_burn = fb[in_box]
                mean_frac = np.mean(box_burn) if len(box_burn) > 0 else 0.0
                pre_burn[i] += mean_frac

            ds_nc.close()

    # Final clamp => <= 1.0 if desired
    pre_burn = np.minimum(pre_burn, 1.0)
    return pre_burn

############################################################
# 2) cumsum with initial
############################################################
def compute_burn_cumsum_with_initial(ds, pre_burn):
    """
    ds["burn_fraction"] => shape (year=15, pixel=...).
    We add pre_burn to year=0, and then do cumsum across years.
    """
    burn_2d = ds["burn_fraction"].values  # shape(15, pixel)
    n_years, n_pixels = burn_2d.shape
    cumsum_2d = np.zeros((n_years, n_pixels), dtype=np.float32)

    # year=0 => 2004
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
    We DO NOT exclude 'burn_fraction' => used as predictor!
    """
    exclude_vars = {
        target_var.lower(),
        'lat','lon','latitude','longitude',
        'pixel','year','ncoords_vector','nyears_vector'
    }
    all_feats = {}
    n_years = ds.dims['year']

    for var_name in ds.data_vars:
        if var_name.lower() in exclude_vars:
            continue
        
        da = ds[var_name]
        dims = set(da.dims)
        if dims == {'year', 'pixel'}:
            arr2d = da.values  # shape (year, pixel)
            all_feats[var_name] = arr2d
        elif dims == {'pixel'}:
            # replicate across years
            arr1d = da.values
            arr2d = np.tile(arr1d, (n_years,1))  # shape (year, pixel)
            all_feats[var_name] = arr2d

    return all_feats

def flatten_spatiotemporal(ds, target_var="DOD"):
    """
    Return:
      X_all => shape(N, n_features)
      y_all => shape(N,)
      feat_names => list of feature names
      valid_mask => boolean shape(N,) for rows with no NaN
    """
    feat_dict = gather_spatiotemporal_features(ds, target_var=target_var)
    feat_names = sorted(feat_dict.keys())  # sorted list

    # Stack columns
    X_cols = []
    for fname in feat_names:
        arr2d = feat_dict[fname]  # (year, pixel)
        arr1d = arr2d.ravel(order='C')  # flatten
        X_cols.append(arr1d)
    X_all = np.column_stack(X_cols)

    # Flatten target
    dod_2d = ds[target_var].values  # shape (year, pixel)
    y_all = dod_2d.ravel(order='C')

    # Valid mask => no NaN in any feature, no NaN in y
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
    """
    Color‐code each sample by its category (0..3).
    Adds a 1:1 line + legend, and shows stats (RMSE, bias, R²).
    """
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
# 6) Additional Visualization: Pixel-level bias map, boxplot
############################################################
def produce_pixel_bias_map_hr_background(ds, pixel_idx, y_true, y_pred,
        lat_var="latitude", lon_var="longitude",
        title="Pixel Bias: High-Res Background"):
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

    if lat_full.ndim == 2:
        lat_1d_all = lat_full[0, :]
        lon_1d_all = lon_full[0, :]
    else:
        lat_1d_all = lat_full
        lon_1d_all = lon_full

    max_abs = np.nanmax(np.abs(mean_bias))
    if np.isnan(max_abs):
        max_abs = 1.0

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    fig = plt.figure(figsize=(9,7))
    ax = plt.axes(projection=tiler.crs)

    ax.set_extent([-125, -113, 32, 42], crs=ccrs.PlateCarree())
    ax.add_image(tiler, 8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, alpha=0.5)

    # Plot all pixels in light gray
    ax.scatter(
        lon_1d_all, lat_1d_all,
        transform=ccrs.PlateCarree(),
        c="lightgray", s=3, alpha=0.7, label="Unused"
    )
    # Overplot used
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

############################################################
# 7) Feature Importance Plot (Random Forest)
############################################################
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

############################################################
# 8) The NEW random forest experiment:
#    70% train from each category c0..c3, 30% test
#    + Wilcoxon rank-sum test
#    + [NEW] Additional test using the category with fewest samples
############################################################
def run_rf_incl_burn_categorized(X_all, y_all, cat_2d, valid_mask, ds, feat_names):
    from sklearn.model_selection import train_test_split
    from scipy.stats import ranksums  # for Wilcoxon rank-sum test
    import numpy.random as npr  # for random sampling in the final new step

    cat_flat_all = cat_2d.ravel(order='C')  
    cat_valid    = cat_flat_all[valid_mask] 

    X_valid = X_all[valid_mask]
    y_valid = y_all[valid_mask]

    train_indices = []
    test_indices  = []

    valid_idxs = np.where(valid_mask)[0]

    # Build train/test subsets category by category
    for cval in [0,1,2,3]:
        cat_mask = (cat_valid == cval)
        cat_rows = np.where(cat_mask)[0]  
        if len(cat_rows) == 0:
            print(f"Category {cval} has no valid data. Skipping.")
            continue

        train_c, test_c = train_test_split(cat_rows, test_size=0.3, random_state=42)
        train_indices.append(train_c)
        test_indices.append(test_c)

    if not train_indices:
        print("No training data => cannot proceed.")
        return None

    train_indices = np.concatenate(train_indices)
    test_indices  = np.concatenate(test_indices)

    X_train = X_valid[train_indices]
    y_train = y_valid[train_indices]
    X_test  = X_valid[test_indices]
    y_test  = y_valid[test_indices]

    # Train random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate on TRAIN
    y_pred_train = rf.predict(X_train)
    plot_scatter(y_train, y_pred_train, title="RF: Train (70% each cat)")
    plot_bias_hist(y_train, y_pred_train, title="RF Bias Hist: Train")

    # Evaluate on TEST (all categories combined)
    y_pred_test = rf.predict(X_test)
    plot_scatter(y_test, y_pred_test, title="RF: Test (30% each cat)")
    plot_bias_hist(y_test, y_pred_test, title="RF Bias Hist: Test (all cats)")

    # -------------------------------------------------------
    # A) Wilcoxon rank-sum test of biases => c1..c3 vs c0
    # -------------------------------------------------------
    cat_test = cat_valid[test_indices]
    biases_dict = {}
    for cval in [0,1,2,3]:
        mask_c = (cat_test == cval)
        if np.any(mask_c):
            y_c     = y_test[mask_c]
            y_predc = y_pred_test[mask_c]
            bias_c  = y_predc - y_c
            biases_dict[cval] = bias_c

    if 0 in biases_dict:
        for cval in [1,2,3]:
            if cval in biases_dict:
                b0 = biases_dict[0]
                bc = biases_dict[cval]
                stat, pval = ranksums(b0, bc)
                print(f"Wilcoxon rank-sum c{cval} vs c0: stat={stat:.3f}, p={pval:.3g}")
            else:
                print(f"No test data for cat={cval}, skipping rank-sum vs c0.")

    # -----------------------------------------------
    # [NEW] 2) Additional “downsampling” test
    # -----------------------------------------------
    # Find how many test samples each category has
    test_counts = {}
    for cval in [0,1,2,3]:
        mask_c = (cat_test == cval)
        n_c = np.sum(mask_c)
        if n_c > 0:
            test_counts[cval] = n_c

    if not test_counts:
        print("[NEW] No test data for any category => cannot do downsampling test.")
    else:
        # find category with min # of test samples
        minCat   = min(test_counts, key=test_counts.get)
        minCount = test_counts[minCat]
        print(f"[NEW] The category with fewest test samples is c{minCat}, count={minCount}")

        # for each category cval, do 10 random subsets of size minCount
        # compute mean bias & rmse for each subset, then average
        n_runs = 10
        for cval in test_counts:
            c_mask = (cat_test == cval)
            idx_c  = np.where(c_mask)[0]  # indexes within test_indices
            biases_list = []
            rmse_list   = []

            # gather the subset arrays
            y_c_full  = y_test[idx_c]
            yp_c_full = y_pred_test[idx_c]

            # if cval has fewer than minCount, we skip or handle carefully
            # but by definition, cval >= minCount if cval != minCat => not necessarily; if cval == minCat we do exact
            # We'll do "if len(idx_c) < minCount => skip"
            if len(idx_c) < minCount:
                print(f"[NEW] Category {cval} has <{minCount} points => cannot sample => skipping.")
                continue

            for _ in range(n_runs):
                sub_idx = npr.choice(idx_c, size=minCount, replace=False)
                y_sub   = y_test[sub_idx]
                yp_sub  = y_pred_test[sub_idx]
                bias_sub= yp_sub - y_sub
                mean_bias = np.mean(bias_sub)
                mean_rmse = np.sqrt(mean_squared_error(y_sub, yp_sub))
                biases_list.append(mean_bias)
                rmse_list.append(mean_rmse)

            # average across the 10 runs
            avg_bias = np.mean(biases_list)
            avg_rmse = np.mean(rmse_list)
            print(f"[NEW] Category c{cval}: from 10 random subsamples of size={minCount},"
                  f" mean bias={avg_bias:.3f}, mean RMSE={avg_rmse:.3f}")

    # Evaluate on each cat=0..3 in the test set
    for cval in [0,1,2,3]:
        test_sel = (cat_test == cval)
        if not np.any(test_sel):
            print(f"No test samples cat={cval} in final test set => skip cat-level plots.")
            continue
        X_c = X_test[test_sel]
        y_c = y_test[test_sel]
        y_pred_c = rf.predict(X_c)

        plot_scatter(y_c, y_pred_c, title=f"RF: Test, cat={cval}")
        plot_bias_hist(y_c, y_pred_c, title=f"RF Bias Hist: Test cat={cval}")

    # Also do color-coded scatter for ALL test samples
    plot_scatter_by_cat(y_test, y_pred_test, cat_test,
                        title="RF: All Test Data, color by cat")

    # Pixel-level bias map
    pixel_idx_full = np.tile(np.arange(ds.dims["pixel"]), ds.dims["year"])
    pix_valid = pixel_idx_full[valid_mask]
    pix_test  = pix_valid[test_indices]
    produce_pixel_bias_map_hr_background(ds, pix_test, y_test, y_pred_test,
        title="Pixel Bias: All Test Data")

    # Boxplot by elevation/veg for entire TEST set
    elev_2d = ds["Elevation"].values
    veg_2d  = ds["VegTyp"].values
    elev_valid = elev_2d.ravel(order='C')[valid_mask]
    veg_valid  = veg_2d.ravel(order='C')[valid_mask]
    elev_test  = elev_valid[test_indices]
    veg_test   = veg_valid[test_indices]
    plot_boxplot_dod_by_elev_veg(y_test, elev_test, veg_test, cat_label="AllTest")

    # Feature importance
    plot_top10_features(rf, feat_names, title="RF Feature Importance (NewApproach)")
    plot_top5_feature_scatter(rf, X_test, y_test, cat_test,
                              feat_names, title_prefix="(NewApproach)")

    return rf


############################################################
# MAIN
############################################################
if __name__=="__main__":

    # 1) Load & compute pre-burn
    path_pat = "/Users/yashnilmohanty/Desktop/data/BurnArea_Data/Merged_BurnArea_{year:04d}{month:02d}.nc"
    pre_burn = compute_pre2004_burn(coords, path_pat, 2001, 2003)

    # 2) Load final dataset => final_dataset3.nc
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset3.nc")

    # 3) cumsum => define categories
    cumsum_2d = compute_burn_cumsum_with_initial(ds, pre_burn)
    ds["burn_cumsum"] = (("year","pixel"), cumsum_2d)
    cat_2d = define_4cats_cumsum(cumsum_2d)  # shape(15, #pixels)

    # 4) Flatten => now including burn_fraction as a predictor
    X_all, y_all, feat_names, valid_mask = flatten_spatiotemporal(ds, target_var="DOD")

    print("Feature names:", feat_names)

    # 5) Train/test with the new approach:
    rf_model = run_rf_incl_burn_categorized(
        X_all, y_all, cat_2d, valid_mask, ds, feat_names
    )

    print("DONE.")
