#!/usr/bin/env python3
# ============================================================
#  Fire-ML · Random Forest Sensitivity Analysis
#  Threshold perturbations for burn categories (±10 %, baseline)
#  Outputs tables grouped by scenario (each scenario lists c0–c3 together)
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
def rf_sensitivity_experiment(X, y, cat2d, ok, unburned_max_cat=0):
    """
    Train Random Forest on unburned pixels and evaluate across all categories.
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
    results = []
    for c in range(4):
        m = (cat == c)
        if not m.any():
            results.append({
                'N': 0,
                'RMSE': np.nan,
                'Bias': np.nan,
                'R2': np.nan
            })
            continue
        
        y_true_cat = Yv[m]
        y_pred_cat = y_hat_all[m]
        bias = y_pred_cat - y_true_cat
        
        results.append({
            'N': m.sum(),
            'RMSE': np.sqrt(mean_squared_error(y_true_cat, y_pred_cat)),
            'Bias': bias.mean(),
            'R2': r2_score(y_true_cat, y_pred_cat)
        })
    
    # Overall metrics (last entry optional, currently unused for tables)
    overall = {
        'N': len(Yv),
        'RMSE': np.sqrt(mean_squared_error(Yv, y_hat_all)),
        'Bias': (y_hat_all - Yv).mean(),
        'R2': r2_score(Yv, y_hat_all)
    }

    return results, overall


def build_category_array(burn_cumsum, bins):
    """
    Parameters
    ----------
    burn_cumsum : np.ndarray
        Array with shape (year, pixel)
    bins : list of dicts
        Each dict requires keys: label, lower, upper, desc
        lower bound is inclusive, upper is exclusive (except np.inf)
    """
    cat2d = np.full_like(burn_cumsum, fill_value=-1, dtype=int)
    for idx, bin_def in enumerate(bins):
        lower = bin_def["lower"]
        upper = bin_def["upper"]

        mask = np.ones_like(burn_cumsum, dtype=bool)
        if np.isfinite(lower):
            mask &= (burn_cumsum >= lower)
        if np.isfinite(upper):
            mask &= (burn_cumsum < upper)
        else:
            mask &= np.isfinite(burn_cumsum)  # keep finite values

        cat2d[mask] = idx

    if (cat2d < 0).any():
        raise ValueError("Some pixels were not assigned to any category. Check bin definitions.")
    return cat2d

# ─── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    log("Loading final_dataset5.nc ...")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset5.nc")
    
    # Build feature matrix
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DSD")
    log(f"Feature matrix ready – {ok.sum()} valid samples, {len(feat_names)} predictors")
    
    bc = ds["burn_cumsum"].values

    scenarios = {
        "Baseline (paper thresholds)": [
            {"label": "C0", "lower": -np.inf, "upper": 0.25, "desc": "<25%"},
            {"label": "C1", "lower": 0.25, "upper": 0.50, "desc": "25–50%"},
            {"label": "C2", "lower": 0.50, "upper": 0.75, "desc": "50–75%"},
            {"label": "C3", "lower": 0.75, "upper": np.inf, "desc": ">75%"},
        ],
        "Threshold robustness test: -10%": [
            {"label": "C0", "lower": -np.inf, "upper": 0.15, "desc": "<15%"},
            {"label": "C1", "lower": 0.15, "upper": 0.40, "desc": "15–40%"},
            {"label": "C2", "lower": 0.40, "upper": 0.65, "desc": "40–65%"},
            {"label": "C3", "lower": 0.65, "upper": np.inf, "desc": ">65%"},
        ],
        "Threshold robustness test: +10%": [
            {"label": "C0", "lower": -np.inf, "upper": 0.35, "desc": "<35%"},
            {"label": "C1", "lower": 0.35, "upper": 0.60, "desc": "35–60%"},
            {"label": "C2", "lower": 0.60, "upper": 0.85, "desc": "60–85%"},
            {"label": "C3", "lower": 0.85, "upper": np.inf, "desc": ">85%"},
        ],
    }

    scenario_rows = []
    overall_rows = []

    for scenario_name, bins in scenarios.items():
        log(f"\n=== {scenario_name} ===")
        cat2d = build_category_array(bc, bins)
        metrics_per_cat, overall_metrics = rf_sensitivity_experiment(
            X_all, y_all, cat2d, ok, unburned_max_cat=0
        )

        for idx, bin_def in enumerate(bins):
            metrics = metrics_per_cat[idx]
            scenario_rows.append({
                "Scenario": scenario_name,
                "Category": bin_def["label"],
                "Threshold": bin_def["desc"],
                "N": metrics["N"],
                "RMSE": metrics["RMSE"],
                "Bias": metrics["Bias"],
                "R2": metrics["R2"],
            })

        overall_metrics = overall_metrics.copy()
        overall_metrics.update({
            "Scenario": scenario_name,
            "Category": "Overall",
            "Threshold": "—"
        })
        overall_rows.append(overall_metrics)

        # Print table matching manuscript style
        df_preview = pd.DataFrame([row for row in scenario_rows
                                   if row["Scenario"] == scenario_name
                                   and row["Category"] != "Overall"])
        if not df_preview.empty:
            df_preview = df_preview[["Category", "Threshold", "N", "RMSE", "Bias", "R2"]]
            print("\n" + scenario_name)
            print(df_preview.to_string(index=False))

    df_results = pd.DataFrame(scenario_rows)
    df_overall = pd.DataFrame(overall_rows)

    output_dir = "/Users/yashnilmohanty/Desktop"
    detailed_csv = f"{output_dir}/rf_threshold_sensitivity_by_scenario.csv"
    overall_csv = f"{output_dir}/rf_threshold_sensitivity_overall.csv"

    df_results.to_csv(detailed_csv, index=False)
    df_overall.to_csv(overall_csv, index=False)

    log(f"\nDetailed scenario table saved to: {detailed_csv}")
    log(f"Overall metrics saved to: {overall_csv}")
    log("ALL DONE.")

