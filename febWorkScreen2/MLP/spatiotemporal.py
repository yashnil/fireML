#!/usr/bin/env python3
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
    import xarray as xr
    n_pixels = len(coords)
    pre_burn = np.zeros((n_pixels,), dtype=np.float32)

    years = range(year_start, year_end+1)
    months = range(1,13)

    for yr in years:
        for mm in months:
            file_path = path_pattern.format(year=yr, month=mm)
            ds_nc = xr.open_dataset(file_path)
            burn_2d = ds_nc["MTBS_BurnFraction"].values
            lat_2d  = ds_nc["XLAT_M"].values
            lon_2d  = ds_nc["XLONG_M"].values

            # Flatten
            flat_burn = burn_2d.ravel()
            flat_lat  = lat_2d.ravel()
            flat_lon  = lon_2d.ravel()

            # Clamp fraction > 1 => 1
            flat_burn = np.minimum(flat_burn, 1.0)

            # Remove NaNs
            valid_mask = (
                np.isfinite(flat_burn) &
                np.isfinite(flat_lat) &
                np.isfinite(flat_lon)
            )
            fb  = flat_burn[valid_mask]
            fla = flat_lat[valid_mask]
            flo = flat_lon[valid_mask]

            # Accumulate for each pixel bounding box
            for i, (coord_lat, coord_lon) in enumerate(coords):
                lat_min, lat_max = coord_lat - 0.005, coord_lat + 0.005
                lon_min, lon_max = coord_lon - 0.005, coord_lon + 0.005
                in_box = (
                    (fla >= lat_min) & (fla <= lat_max) &
                    (flo >= lon_min) & (flo <= lon_max)
                )
                box_burn = fb[in_box]
                mean_frac = np.mean(box_burn) if len(box_burn) > 0 else 0.0
                pre_burn[i] += mean_frac

            ds_nc.close()

    # Final clamp => <= 1.0 if desired
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
    """
    Collect 2D (year,pixel) or 1D (pixel) variables as features. 
    Excludes the target and lat/lon placeholders.
    """
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
            arr2d = da.values  # shape(year, pixel)
            all_feats[var] = arr2d
        elif dims == {'pixel'}:
            arr1d = da.values  # shape(pixel,)
            arr2d = np.tile(arr1d, (ny,1))  # replicate across years
            all_feats[var] = arr2d

    return all_feats

def flatten_spatiotemporal(ds, target_var="DOD"):
    """
    Flatten from (year,pixel) => (N, n_features).
    valid_mask indicates which rows have no NaN in X or y.
    """
    feat_dict = gather_spatiotemporal_features(ds, target_var=target_var)
    feat_names = sorted(feat_dict.keys())

    X_cols = []
    for f in feat_names:
        arr2d = feat_dict[f]  # shape(year, pixel)
        arr1d = arr2d.ravel(order='C')  # flatten
        X_cols.append(arr1d)

    X_all = np.column_stack(X_cols)

    # Flatten target
    dod2d = ds[target_var].values  # shape(year, pixel)
    y_all = dod2d.ravel(order='C')

    # valid mask => no NaN in any feature or target
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
    """
    c0 => <0.25
    c1 => [0.25, 0.5)
    c2 => [0.5, 0.75)
    c3 => >=0.75
    """
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

def plot_scatter_by_cat(y_true, y_pred, cat, title="Scatter by Category"):
    """
    Color‐code each sample by its category (0..3).
    Adds a 1:1 line + legend, and shows RMSE/bias/R².
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    plt.figure(figsize=(6,6))

    # Choose colors for each category
    cat_colors = {0:'red', 1:'green', 2:'blue', 3:'orange'}

    # Plot points category by category
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

    # 1:1 line
    mn = min(y_pred.min(), y_true.min())
    mx = max(y_pred.max(), y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--', label='1:1 line')

    # Metrics
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
# 6) Permutation Importance for MLP
########################
def compute_permutation_importance_mlp(model, X_val, y_val,
                                       n_repeats=3, random_state=42):
    """
    Simple permutation importance: for each feature in X_val, shuffle it
    and see how MSE changes relative to baseline.
    Returns an array of "importance" for each feature (bigger = more important).
    """
    import numpy as np
    rng = np.random.RandomState(random_state)

    # Baseline predictions
    y_pred = model.predict(X_val)
    baseline_mse = mean_squared_error(y_val, y_pred)

    n_features = X_val.shape[1]
    importances = np.zeros(n_features, dtype=float)

    X_val_perm = np.copy(X_val)
    for f_idx in range(n_features):
        scores = []
        for _ in range(n_repeats):
            saved_col = np.copy(X_val_perm[:, f_idx])
            perm_idx = rng.permutation(len(X_val_perm))
            # Shuffle only this column
            X_val_perm[:, f_idx] = X_val_perm[perm_idx, f_idx]

            # Predict with the shuffled feature
            y_pred_p = model.predict(X_val_perm)
            perm_mse = mean_squared_error(y_val, y_pred_p)
            scores.append(perm_mse)

            # Revert the column
            X_val_perm[:, f_idx] = saved_col

        importances[f_idx] = np.mean(scores) - baseline_mse

    return importances

########################
# 7) Feature Importance Plot
########################
def plot_top10_features(importances, feat_names, title="Top 10 Feature Importances"):
    """
    importances: array of shape (n_features,) from a permutation-based method
    feat_names: list of feature names
    """
    idx_sorted = np.argsort(importances)[::-1]  # descending
    top10_idx = idx_sorted[:10]
    top10_vals = importances[top10_idx]
    top10_names= [feat_names[i] for i in top10_idx]

    plt.figure(figsize=(8,4))
    plt.bar(range(len(top10_vals)), top10_vals, align='center')
    plt.xticks(range(len(top10_vals)), top10_names, rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Permutation Importance\n(Δ MSE from baseline)")
    plt.tight_layout()
    plt.show()

########################
# 8) Plot top-5 feature-value vs. DOD
########################
def plot_top5_feature_scatter(importances, X_valid, y_valid, cat_valid, feat_names,
                              title_prefix="(thr=0)"):
    """
    1) Identify top-5 most important features from 'importances' array.
    2) For each of those 5 features, create a scatterplot:
       - x-axis = Observed DOD
       - y-axis = Feature Value
       - Color-coded by category (0..3)
       - A dotted line of best fit for the entire data.
    """
    importances_arr = np.array(importances)
    idx_sorted = np.argsort(importances_arr)[::-1]  # descending
    top5_idx   = idx_sorted[:5]

    cat_colors = {0:'red', 1:'blue', 2:'green', 3:'purple'}

    for feat_idx in top5_idx:
        fname = feat_names[feat_idx]
        feat_vals = X_valid[:, feat_idx]

        plt.figure(figsize=(6,5))
        # Scatter by category
        for cval, ccolor in cat_colors.items():
            sel_c = (cat_valid == cval)
            plt.scatter(
                y_valid[sel_c],
                feat_vals[sel_c],
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
            plt.plot(x_vals, y_fit, 'k--', label=f"Fit y={p[0]:.2f}x+{p[1]:.2f}")

        plt.xlabel("Observed DOD")
        plt.ylabel(f"{fname}")
        plt.title(f"{title_prefix}: Feature={fname}")
        plt.legend()
        plt.tight_layout()
        plt.show()

########################
# 9) run experiment => MLP
########################
def run_spatiotemporal_experiment_mlp(X_all, y_all, valid_mask, cat_2d,
                                      unburned_max_cat=0,
                                      ds=None, feat_names=None):
    """
    Trains an MLP on the 'unburned' subset (cat <= unburned_max_cat),
    then evaluates on unburned test, each cat=0..3, and all data.
    Plots:
      - scatter
      - hist
      - top-10 feature importances (permutation)
      - top-5 feature-value scatter
    No boxplots, no spatial plots.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor

    cat_flat  = cat_2d.ravel(order='C')
    cat_valid = cat_flat[valid_mask]
    X_valid   = X_all[valid_mask]
    y_valid   = y_all[valid_mask]

    # 1) Identify "unburned" subset
    is_unburned = (cat_valid <= unburned_max_cat)
    n_unburn    = np.sum(is_unburned)
    print(f"[MLP] Train threshold => unburned up to cat={unburned_max_cat}, #unburned={n_unburn}")

    if n_unburn == 0:
        print("No unburned => can't train.")
        return None

    # 2) Train/test split on unburned
    X_ub = X_valid[is_unburned]
    y_ub = y_valid[is_unburned]

    X_train, X_test, y_train, y_test = train_test_split(
        X_ub, y_ub, test_size=0.3, random_state=42
    )
    print(f"   unburned train={len(X_train)}, test={len(X_test)}")

    # 3) Define & Train MLP
    #    Tweak hyperparams as needed
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        random_state=42,
        max_iter=1000,
        early_stopping=False
    )
    mlp.fit(X_train, y_train)

    # Evaluate on unburned TRAIN
    y_pred_train = mlp.predict(X_train)
    plot_scatter(y_train, y_pred_train, f"MLP: Unburned Train (cat<={unburned_max_cat})")
    plot_bias_hist(y_train, y_pred_train,
                   title=f"MLP Bias Hist: Unburned Train (cat<={unburned_max_cat})")

    # Evaluate on unburned TEST
    y_pred_test = mlp.predict(X_test)
    plot_scatter(y_test, y_pred_test, f"MLP: Unburned Test (cat<={unburned_max_cat})")
    plot_bias_hist(y_test, y_pred_test,
                   title=f"MLP Bias Hist: Unburned Test (cat<={unburned_max_cat})")

    # Evaluate on each cat=0..3
    for cval in [0,1,2,3]:
        sel_cat = (cat_valid == cval)
        if np.sum(sel_cat) == 0:
            print(f"No samples cat={cval}")
            continue
        X_c = X_valid[sel_cat]
        y_c = y_valid[sel_cat]
        y_pred_c = mlp.predict(X_c)

        plot_scatter(y_c, y_pred_c, title=f"MLP Category={cval}")
        plot_bias_hist(y_c, y_pred_c, title=f"MLP Bias Hist: cat={cval}")

    # Evaluate on ALL valid data
    y_pred_all = mlp.predict(X_valid)
    plot_scatter_by_cat(y_valid, y_pred_all, cat_valid,
                    title=f"MLP All Data (thr={unburned_max_cat})")
    plot_bias_hist(y_valid, y_pred_all,
                   title=f"MLP Bias Hist: All Data (thr={unburned_max_cat})")

    # ---- Compute Permutation Importances, plot top-10, top-5 scatter
    importances = compute_permutation_importance_mlp(
        mlp,
        X_valid,
        y_valid,
        n_repeats=3,
        random_state=42
    )
    plot_top10_features(importances, feat_names,
        title=f"Top 10 Feature Importances (MLP, thr={unburned_max_cat})")

    cat_flat_valid = cat_2d.ravel(order='C')[valid_mask]
    plot_top5_feature_scatter(
        importances,
        X_valid,
        y_valid,
        cat_flat_valid,
        feat_names,
        title_prefix=f"(thr={unburned_max_cat})"
    )

    return mlp

###############################
# MAIN
###############################
if __name__=="__main__":
    # 1) LOAD/COMPUTE PRE-BURN
    path_pat = "/Users/yashnilmohanty/Desktop/data/BurnArea_Data/Merged_BurnArea_{year:04d}{month:02d}.nc"
    pre_burn  = compute_pre2004_burn(coords, path_pat, 2001, 2003)

    # 2) LOAD DS
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset2.nc")

    # 3) cumsum from 2004 onward (with pre‐burn)
    cumsum_2d = compute_burn_cumsum_with_initial(ds, pre_burn)
    ds["burn_cumsum"] = (("year","pixel"), cumsum_2d)

    # 4) define categories
    cat_2d = define_4cats_cumsum(cumsum_2d)

    # 5) Flatten => X_all, y_all, ...
    X_all, y_all, feat_names, valid_mask, n_years, n_pixels = flatten_spatiotemporal(ds, "DOD")

    # RUN #1 => thr=0 => unburned cat=0
    print("\n==== MLP RUN #1 => unburned= cat=0 (<0.25) ====")
    mlp_run1 = run_spatiotemporal_experiment_mlp(
        X_all, y_all, valid_mask, cat_2d,
        unburned_max_cat=0, ds=ds,
        feat_names=feat_names
    )

    # RUN #2 => thr=0.5 => unburned cat=0..1
    print("\n==== MLP RUN #2 => unburned= cat=0,1 (<0.5) ====")
    mlp_run2 = run_spatiotemporal_experiment_mlp(
        X_all, y_all, valid_mask, cat_2d,
        unburned_max_cat=1, ds=ds,
        feat_names=feat_names
    )

    print("DONE.")
