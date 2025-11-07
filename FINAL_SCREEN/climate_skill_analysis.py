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
def _preferred_precip_var(ds):
    if "aorcWinterPrecipitation" in ds.data_vars:
        return "aorcWinterPrecipitation"
    if "aorcWinterRain" in ds.data_vars:
        return "aorcWinterRain"
    for candidate in ds.data_vars:
        lower = candidate.lower()
        if "precip" in lower or "rain" in lower:
            return candidate
    raise ValueError("Could not find a precipitation variable (looked for *Precipitation or *Rain).")


def _preferred_temp_var(ds):
    if "aorcSpringTemperature" in ds.data_vars:
        return "aorcSpringTemperature"
    for candidate in ds.data_vars:
        lower = candidate.lower()
        if "temperature" in lower and "aorc" in lower:
            return candidate
    raise ValueError("Could not find a temperature variable (looked for *Temperature).")


def _infer_calendar_years(ds):
    if "year" in ds.coords:
        vals = np.asarray(ds["year"].values)
        if vals.size and np.all(np.isfinite(vals)):
            # Assume these are already calendar years if they look like 4-digit numbers
            if vals.min() >= 1900:
                return vals.astype(int)
            # If zero-based, assume starting at 2004 (dataset covers 2004–2018)
            if vals.min() == 0 and vals.max() <= 50:
                return (vals + 2004).astype(int)
    # Fallback: 2004.. (for LEN years)
    n_years = ds.sizes.get("year", 0)
    return (np.arange(n_years) + 2004).astype(int)


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
    
    precip_var = _preferred_precip_var(ds)
    temp_var = _preferred_temp_var(ds)

    log(f"Using precipitation variable: {precip_var}")
    log(f"Using temperature variable: {temp_var}")
    
    # Get precipitation and temperature data (shape: year, pixel)
    precip = ds[precip_var].values  # (year, pixel)
    temp = ds[temp_var].values       # (year, pixel)
    
    # Spatially average across pixels for each year
    # Handle NaN values by taking nanmean
    precip_annual = np.nanmean(precip, axis=1)  # (year,)
    temp_annual = np.nanmean(temp, axis=1)      # (year,)

    n_years = precip_annual.size
    year_values = _infer_calendar_years(ds)

    # Rank-based split to avoid large imbalances when values equal the median
    precip_order = np.argsort(precip_annual)
    temp_order = np.argsort(temp_annual)

    half = n_years // 2
    # Build z-scores for scoring
    def _safe_std(arr):
        sd = np.nanstd(arr)
        return sd if sd > 0 else 1.0

    precip_z = (precip_annual - np.nanmean(precip_annual)) / _safe_std(precip_annual)
    temp_z = (temp_annual - np.nanmean(temp_annual)) / _safe_std(temp_annual)

    scores = {
        'wet_cold': precip_z - temp_z,      # wet and cold => high precip, low temp
        'wet_hot': precip_z + temp_z,       # wet and hot
        'dry_cold': -precip_z - temp_z,     # dry and cold
        'dry_hot': -precip_z + temp_z,      # dry and hot
    }

    base = n_years // 4
    remainder = n_years % 4
    condition_order = ['wet_cold', 'wet_hot', 'dry_cold', 'dry_hot']
    target_counts = {
        condition: base + (1 if idx < remainder else 0)
        for idx, condition in enumerate(condition_order)
    }

    available = set(range(n_years))
    year_classes_idx = {condition: [] for condition in condition_order}

    for condition in condition_order:
        count_needed = target_counts[condition]
        if count_needed <= 0:
            continue
        sorted_candidates = sorted(
            list(available),
            key=lambda idx: scores[condition][idx],
            reverse=True
        )
        selected = sorted_candidates[:count_needed]
        year_classes_idx[condition].extend(selected)
        available -= set(selected)

    if available:
        # Assign any leftovers (due to rounding) to the condition with remaining capacity
        for idx in list(available):
            # choose condition with currently smallest assignment relative to target
            deficits = {
                condition: target_counts[condition] - len(year_classes_idx[condition])
                for condition in condition_order
            }
            condition = max(deficits, key=lambda c: deficits[c])
            year_classes_idx[condition].append(idx)
            available.remove(idx)

    # Print classification summary
    log("\nClimate condition classification:")
    for condition, indices in year_classes_idx.items():
        years_actual = [int(year_values[i]) for i in sorted(indices)]
        log(f"  {condition}: {len(indices)} years - {years_actual}")
    
    return year_classes_idx, year_values, precip_annual, temp_annual

# ─── RANDOM FOREST EXPERIMENT WITH CLIMATE CONDITIONS ───────────
def rf_climate_skill_experiment(
    X,
    y,
    cat2d,
    ok,
    ds,
    feat_names,
    year_classes_idx,
    year_values,
    unburned_max_cat=0,
):
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
    
    summary_rows = []
    per_category_rows = []
    
    for condition, year_indices in year_classes_idx.items():
        if len(year_indices) == 0:
            years_str = ''
            summary_rows.append({
                'Climate_Condition': condition,
                'Years': years_str,
                'N': 0,
                'RMSE': np.nan,
                'Bias': np.nan,
                'R2': np.nan,
            })
            continue

        years_actual = [int(year_values[i]) for i in sorted(year_indices)]
        years_str = ', '.join(map(str, years_actual))

        # Filter samples for this climate condition
        condition_mask = np.isin(year_valid, year_indices)

        if not condition_mask.any():
            summary_rows.append({
                'Climate_Condition': condition,
                'Years': years_str,
                'N': 0,
                'RMSE': np.nan,
                'Bias': np.nan,
                'R2': np.nan,
            })
            continue

        y_true_cond = Yv[condition_mask]
        y_pred_cond = y_hat_all[condition_mask]
        bias = y_pred_cond - y_true_cond

        summary_rows.append({
            'Climate_Condition': condition,
            'Years': years_str,
            'N': condition_mask.sum(),
            'RMSE': np.sqrt(mean_squared_error(y_true_cond, y_pred_cond)),
            'Bias': bias.mean(),
            'R2': r2_score(y_true_cond, y_pred_cond),
        })

        for c in range(4):
            cat_mask = condition_mask & (cat == c)
            if not cat_mask.any():
                per_category_rows.append({
                    'Climate_Condition': condition,
                    'Burn_Category': f'c{c}',
                    'Years': years_str,
                    'N': 0,
                    'RMSE': np.nan,
                    'Bias': np.nan,
                    'R2': np.nan,
                })
                continue

            y_true_cat = Yv[cat_mask]
            y_pred_cat = y_hat_all[cat_mask]
            bias_cat = y_pred_cat - y_true_cat

            try:
                r2_cat = r2_score(y_true_cat, y_pred_cat)
            except ValueError:
                r2_cat = np.nan

            per_category_rows.append({
                'Climate_Condition': condition,
                'Burn_Category': f'c{c}',
                'Years': years_str,
                'N': cat_mask.sum(),
                'RMSE': np.sqrt(mean_squared_error(y_true_cat, y_pred_cat)),
                'Bias': bias_cat.mean(),
                'R2': r2_cat,
            })

    # Overall metrics across all conditions
    overall_row = {
        'N': len(Yv),
        'RMSE': np.sqrt(mean_squared_error(Yv, y_hat_all)),
        'Bias': (y_hat_all - Yv).mean(),
        'R2': r2_score(Yv, y_hat_all),
        'Years': 'all'
    }

    return summary_rows, per_category_rows, overall_row

# ─── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    log("Loading final_dataset5.nc ...")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset5.nc")
    
    # Build feature matrix
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DSD")
    log(f"Feature matrix ready – {ok.sum()} valid samples, {len(feat_names)} predictors")
    
    # Classify climate conditions
    log("\nClassifying climate conditions...")
    year_classes_idx, year_values, precip_annual, temp_annual = classify_climate_conditions(ds)
    
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
    summary_rows, per_category_rows, overall_row = rf_climate_skill_experiment(
        X_all, y_all, cat2d, ok, ds, feat_names,
        year_classes_idx, year_values, unburned_max_cat=0
    )
    
    df_summary = pd.DataFrame(summary_rows)
    df_per_category = pd.DataFrame(per_category_rows)
    df_overall = pd.DataFrame([overall_row])

    summary_cols = ['Climate_Condition', 'Years', 'N', 'RMSE', 'Bias', 'R2']
    df_summary = df_summary[summary_cols]
    per_cat_cols = ['Climate_Condition', 'Years', 'Burn_Category', 'N', 'RMSE', 'Bias', 'R2']
    df_per_category = df_per_category[per_cat_cols]
    df_overall = df_overall[['Years', 'N', 'RMSE', 'Bias', 'R2']]

    output_dir = "/Users/yashnilmohanty/Desktop"
    summary_csv = f"{output_dir}/climate_skill_scores_overall.csv"
    per_cat_csv = f"{output_dir}/climate_skill_scores_by_category.csv"
    overall_csv = f"{output_dir}/climate_skill_scores_global.csv"

    df_summary.to_csv(summary_csv, index=False)
    df_per_category.to_csv(per_cat_csv, index=False)
    df_overall.to_csv(overall_csv, index=False)

    log(f"\nOverall climate-condition table saved to: {summary_csv}")
    log(f"Per burn-category climate-condition table saved to: {per_cat_csv}")
    log(f"Global metrics saved to: {overall_csv}")

    # Pretty-print to console (similar to manuscript tables)
    print("\n" + "="*80)
    print("CLIMATE CONDITION SKILL SCORE ANALYSIS – OVERALL BY CONDITION")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80)

    for condition, group in df_per_category.groupby("Climate_Condition"):
        print(f"\n{condition.title()} years ({group['Years'].iloc[0]})")
        print(group[['Burn_Category', 'N', 'RMSE', 'Bias', 'R2']].to_string(index=False))

    # Also save climate condition summary (means, rankings, classification flags)
    year_indices = np.arange(len(precip_annual))
    climate_details = pd.DataFrame({
        'Year_Index': year_indices,
        'Year': [int(year_values[i]) for i in year_indices],
        'Precipitation_Mean': precip_annual,
        'Temperature_Mean': temp_annual,
    })
    climate_details['Precip_Rank'] = np.argsort(np.argsort(precip_annual))
    climate_details['Temp_Rank'] = np.argsort(np.argsort(temp_annual))
    climate_details['Climate_Condition'] = 'Unclassified'
    for condition, indices in year_classes_idx.items():
        climate_details.loc[indices, 'Climate_Condition'] = condition

    climate_summary = climate_details.rename(columns={'Climate_Condition': 'Assigned_Condition'})

    climate_summary_file = f"{output_dir}/climate_condition_summary.csv"
    climate_summary.to_csv(climate_summary_file, index=False)
    log(f"Climate condition summary saved to: {climate_summary_file}")
    
    log("ALL DONE.")

