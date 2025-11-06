#!/usr/bin/env python3
# ============================================================
#  Fire-ML · Climate Condition Skill Score Analysis
#  Evaluates skill scores across wet/cold, wet/hot, dry/cold, dry/hot conditions
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

# ─── CLIMATE CONDITION CLASSIFICATION ──────────────────────────
def classify_climate_conditions(ds):
    """
    Classify years into wet/cold, wet/hot, dry/cold, dry/hot based on
    spatially averaged precipitation and temperature.
    
    Returns:
    --------
    year_classes : dict
        Dictionary mapping year indices to condition labels
        {'wet_cold': [year_indices], 'wet_hot': [...], 'dry_cold': [...], 'dry_hot': [...]}
    """
    # Get precipitation and temperature variables
    # Assuming precipitation is 'aorcWinterPrecipitation' or similar
    # and temperature is 'aorcSpringTemperature' or similar
    # We'll use winter precipitation and spring temperature as key indicators
    
    # Try to find precipitation variables
    precip_vars = [v for v in ds.data_vars if 'precip' in v.lower() or 'rain' in v.lower()]
    temp_vars = [v for v in ds.data_vars if 'temp' in v.lower() and 'aorc' in v.lower()]
    
    if not precip_vars or not temp_vars:
        # Fallback: use winter precipitation and spring temperature if available
        if 'aorcWinterPrecipitation' in ds.data_vars:
            precip_var = 'aorcWinterPrecipitation'
        elif 'aorcWinterRain' in ds.data_vars:
            precip_var = 'aorcWinterRain'
        else:
            # Use first available precipitation variable
            precip_var = precip_vars[0] if precip_vars else None
        
        if 'aorcSpringTemperature' in ds.data_vars:
            temp_var = 'aorcSpringTemperature'
        else:
            # Use first available temperature variable
            temp_var = temp_vars[0] if temp_vars else None
    else:
        precip_var = precip_vars[0]
        temp_var = temp_vars[0]
    
    if precip_var is None or temp_var is None:
        raise ValueError("Could not find precipitation and/or temperature variables in dataset")
    
    log(f"Using precipitation variable: {precip_var}")
    log(f"Using temperature variable: {temp_var}")
    
    # Get precipitation and temperature data (shape: year, pixel)
    precip = ds[precip_var].values  # (year, pixel)
    temp = ds[temp_var].values       # (year, pixel)
    
    # Spatially average across pixels for each year
    # Handle NaN values by taking nanmean
    precip_annual = np.nanmean(precip, axis=1)  # (year,)
    temp_annual = np.nanmean(temp, axis=1)      # (year,)
    
    # Calculate thresholds (median split)
    precip_median = np.nanmedian(precip_annual)
    temp_median = np.nanmedian(temp_annual)
    
    log(f"Precipitation median: {precip_median:.4f}")
    log(f"Temperature median: {temp_median:.4f}")
    
    # Classify years
    year_classes = {
        'wet_cold': [],
        'wet_hot': [],
        'dry_cold': [],
        'dry_hot': []
    }
    
    n_years = len(precip_annual)
    for year_idx in range(n_years):
        is_wet = precip_annual[year_idx] >= precip_median
        is_hot = temp_annual[year_idx] >= temp_median
        
        if is_wet and not is_hot:
            year_classes['wet_cold'].append(year_idx)
        elif is_wet and is_hot:
            year_classes['wet_hot'].append(year_idx)
        elif not is_wet and not is_hot:
            year_classes['dry_cold'].append(year_idx)
        else:  # not is_wet and is_hot
            year_classes['dry_hot'].append(year_idx)
    
    # Print classification summary
    log("\nClimate condition classification:")
    for condition, years in year_classes.items():
        log(f"  {condition}: {len(years)} years - indices {years}")
    
    return year_classes, precip_annual, temp_annual

# ─── RANDOM FOREST EXPERIMENT WITH CLIMATE CONDITIONS ───────────
def rf_climate_skill_experiment(X, y, cat2d, ok, ds, feat_names, year_classes, unburned_max_cat=0):
    """
    Train Random Forest on unburned pixels and evaluate skill across climate conditions.
    """
    cat = cat2d.ravel(order="C")[ok]
    Xv, Yv = X[ok], y[ok]
    
    # Get year indices for each sample (needed for climate condition filtering)
    n_years = ds.sizes["year"]
    n_pixels = ds.sizes["pixel"]
    year_full = np.tile(np.arange(n_years), n_pixels)
    year_valid = year_full[ok]
    
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
    
    # Calculate metrics per climate condition
    results = {}
    
    for condition, year_indices in year_classes.items():
        if len(year_indices) == 0:
            results[condition] = {
                'N': 0,
                'RMSE': np.nan,
                'Bias': np.nan,
                'Bias_Std': np.nan,
                'R2': np.nan,
                'Years': ''
            }
            continue
        
        # Filter samples for this climate condition
        condition_mask = np.isin(year_valid, year_indices)
        
        if not condition_mask.any():
            results[condition] = {
                'N': 0,
                'RMSE': np.nan,
                'Bias': np.nan,
                'Bias_Std': np.nan,
                'R2': np.nan,
                'Years': ','.join(map(str, year_indices))
            }
            continue
        
        y_true_cond = Yv[condition_mask]
        y_pred_cond = y_hat_all[condition_mask]
        bias = y_pred_cond - y_true_cond
        
        results[condition] = {
            'N': condition_mask.sum(),
            'RMSE': np.sqrt(mean_squared_error(y_true_cond, y_pred_cond)),
            'Bias': bias.mean(),
            'Bias_Std': bias.std(),
            'R2': r2_score(y_true_cond, y_pred_cond),
            'Years': ','.join(map(str, year_indices))
        }
    
    # Overall metrics
    results['overall'] = {
        'N': len(Yv),
        'RMSE': np.sqrt(mean_squared_error(Yv, y_hat_all)),
        'Bias': (y_hat_all - Yv).mean(),
        'Bias_Std': (y_hat_all - Yv).std(),
        'R2': r2_score(Yv, y_hat_all),
        'Years': 'all'
    }
    
    return results

# ─── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    log("Loading final_dataset5.nc ...")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset5.nc")
    
    # Build feature matrix
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DSD")
    log(f"Feature matrix ready – {ok.sum()} valid samples, {len(feat_names)} predictors")
    
    # Classify climate conditions
    log("\nClassifying climate conditions...")
    year_classes, precip_annual, temp_annual = classify_climate_conditions(ds)
    
    # Compute burn categories (using 0.25 threshold)
    bc = ds["burn_cumsum"].values
    cat2d = np.zeros_like(bc, dtype=int)
    cat2d[bc < 0.25] = 0
    cat2d[(bc >= 0.25) & (bc < 0.50)] = 1
    cat2d[(bc >= 0.50) & (bc < 0.75)] = 2
    cat2d[bc >= 0.75] = 3
    log("Burn categories computed")
    
    # Run experiment for cat 0 only
    log("\nRunning Random Forest experiment with unburned_max_cat=0")
    results = rf_climate_skill_experiment(
        X_all, y_all, cat2d, ok, ds, feat_names,
        year_classes, unburned_max_cat=0
    )
    
    # Convert to DataFrame
    all_results = []
    for condition, metrics in results.items():
        all_results.append({
            'Climate_Condition': condition,
            **metrics
        })
    
    df_results = pd.DataFrame(all_results)
    
    # Reorder columns for readability
    df_results = df_results[['Climate_Condition', 'N', 'RMSE', 'Bias', 'Bias_Std', 'R2', 'Years']]
    
    # Save to CSV
    output_file = "/Users/yashnilmohanty/Desktop/climate_skill_scores.csv"
    df_results.to_csv(output_file, index=False)
    log(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("CLIMATE CONDITION SKILL SCORE ANALYSIS")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    # Also save climate condition summary
    climate_summary = pd.DataFrame({
        'Year_Index': range(len(precip_annual)),
        'Precipitation_Mean': precip_annual,
        'Temperature_Mean': temp_annual,
        'Precip_Above_Median': precip_annual >= np.nanmedian(precip_annual),
        'Temp_Above_Median': temp_annual >= np.nanmedian(temp_annual)
    })
    
    climate_summary_file = "/Users/yashnilmohanty/Desktop/climate_condition_summary.csv"
    climate_summary.to_csv(climate_summary_file, index=False)
    log(f"Climate condition summary saved to: {climate_summary_file}")
    
    log("ALL DONE.")

