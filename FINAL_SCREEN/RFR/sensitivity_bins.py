#!/usr/bin/env python3
# ============================================================
#  Fire-ML · Random Forest Sensitivity Analysis
#  Testing 0.15 and 0.35 burn fraction bins (instead of 0.25)
#  Outputs results to CSV summary table
# ============================================================
import time
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ─── GLOBAL SETTINGS ──────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[{time.time():.1f}] {msg}", flush=True)

# ─── FEATURE-MATRIX HELPERS ─────────────────────────────────────
def gather_features_nobf(ds, target="DSD"):
    excl = {target.lower(), 'lat', 'lon', 'latitude', 'longitude',
            'burn_fraction', 'burn_cumsum'}
    ny = ds.sizes["year"]
    feats = {}
    for v in ds.data_vars:
        if v.lower() in excl:
            continue
        da = ds[v]
        if set(da.dims) == {"year", "pixel"}:
            feats[v] = da.values
        elif set(da.dims) == {"pixel"}:
            feats[v] = np.tile(da.values, (ny, 1))
    return feats

def flatten_nobf(ds, target="DSD"):
    fd = gather_features_nobf(ds, target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order="C") for n in names])
    y = ds[target].values.ravel(order="C")
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, names, ok

# ─── RANDOM FOREST EXPERIMENT WITH CUSTOM BINS ─────────────────
def rf_sensitivity_experiment(X, y, cat2d, ok, unburned_max_cat=0, threshold=0.25):
    """
    Train Random Forest on unburned pixels and evaluate across all categories.
    
    Parameters:
    -----------
    threshold : float
        Burn fraction threshold (0.15, 0.25, or 0.35)
    """
    cat = cat2d.ravel(order="C")[ok]
    Xv, Yv = X[ok], y[ok]
    
    # 70/30 split per category
    train_idx, test_idx = [], []
    for c in range(unburned_max_cat + 1):
        rows = np.where(cat == c)[0]
        if rows.size == 0:
            continue
        tr, te = train_test_split(rows, test_size=0.30, random_state=42)
        train_idx.append(tr)
        test_idx.append(te)
    
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    
    X_tr, y_tr = Xv[train_idx], Yv[train_idx]
    X_te, y_te = Xv[test_idx], Yv[test_idx]
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    
    # Predict on all data
    y_hat_all = rf.predict(Xv)
    
    # Calculate metrics per category
    results = {}
    for c in range(4):
        m = (cat == c)
        if not m.any():
            results[f'cat{c}'] = {
                'N': 0,
                'RMSE': np.nan,
                'Bias': np.nan,
                'Bias_Std': np.nan,
                'R2': np.nan
            }
            continue
        
        y_true_cat = Yv[m]
        y_pred_cat = y_hat_all[m]
        bias = y_pred_cat - y_true_cat
        
        results[f'cat{c}'] = {
            'N': m.sum(),
            'RMSE': np.sqrt(mean_squared_error(y_true_cat, y_pred_cat)),
            'Bias': bias.mean(),
            'Bias_Std': bias.std(),
            'R2': r2_score(y_true_cat, y_pred_cat)
        }
    
    # Overall metrics
    results['overall'] = {
        'N': len(Yv),
        'RMSE': np.sqrt(mean_squared_error(Yv, y_hat_all)),
        'Bias': (y_hat_all - Yv).mean(),
        'Bias_Std': (y_hat_all - Yv).std(),
        'R2': r2_score(Yv, y_hat_all)
    }
    
    return results

# ─── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    log("Loading final_dataset5.nc ...")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset5.nc")
    
    # Build feature matrix
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DSD")
    log(f"Feature matrix ready – {ok.sum()} valid samples, {len(feat_names)} predictors")
    
    # Test different bin thresholds
    thresholds = [0.15, 0.25, 0.35]
    all_results = []
    
    for threshold in thresholds:
        log(f"\n=== Testing threshold = {threshold} ===")
        
        # Compute categories based on threshold
        bc = ds["burn_cumsum"].values
        cat2d = np.zeros_like(bc, dtype=int)
        
        if threshold == 0.15:
            # 0.15 bins: 0-0.15, 0.15-0.30, 0.30-0.45, 0.45+
            cat2d[bc < 0.15] = 0
            cat2d[(bc >= 0.15) & (bc < 0.30)] = 1
            cat2d[(bc >= 0.30) & (bc < 0.45)] = 2
            cat2d[bc >= 0.45] = 3
        elif threshold == 0.25:
            # Original 0.25 bins
            cat2d[bc < 0.25] = 0
            cat2d[(bc >= 0.25) & (bc < 0.50)] = 1
            cat2d[(bc >= 0.50) & (bc < 0.75)] = 2
            cat2d[bc >= 0.75] = 3
        elif threshold == 0.35:
            # 0.35 bins: 0-0.35, 0.35-0.70, 0.70-1.05, 1.05+
            cat2d[bc < 0.35] = 0
            cat2d[(bc >= 0.35) & (bc < 0.70)] = 1
            cat2d[(bc >= 0.70) & (bc < 1.05)] = 2
            cat2d[bc >= 1.05] = 3
        
        # Run experiment for cat 0 only
        log(f"Running experiment with unburned_max_cat=0 (threshold={threshold})")
        results = rf_sensitivity_experiment(
            X_all, y_all, cat2d, ok, 
            unburned_max_cat=0, 
            threshold=threshold
        )
        
        # Store results with threshold label
        for category, metrics in results.items():
            all_results.append({
                'Threshold': threshold,
                'Category': category,
                **metrics
            })
    
    # Convert to DataFrame and save
    df_results = pd.DataFrame(all_results)
    
    # Reorder columns for readability
    df_results = df_results[['Threshold', 'Category', 'N', 'RMSE', 'Bias', 'Bias_Std', 'R2']]
    
    # Save to CSV
    output_file = "/Users/yashnilmohanty/Desktop/rf_sensitivity_bins_results.csv"
    df_results.to_csv(output_file, index=False)
    log(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    log("ALL DONE.")

