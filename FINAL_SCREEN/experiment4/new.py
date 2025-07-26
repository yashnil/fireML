#!/usr/bin/env python3
import xarray as xr
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# We utilize final_dataset6.nc for Experiment 4 (added in FPAR analysis)

# ────────────────────────────────────────────────────────────
def gather_features_nobf(ds, target="DSD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector',
            'burn_fraction','burn_cumsum',
            'aorcsummerhumidity','aorcsummerprecipitation',
            'aorcsummerlongwave','aorcsummershortwave','aorcsummertemperature'}
    ny = ds.sizes['year']
    feats = {}
    for v in ds.data_vars:
        if v.lower() in excl: continue
        da = ds[v]
        if set(da.dims)=={'year','pixel'}:
            feats[v]=da.values
        elif set(da.dims)=={'pixel'}:
            feats[v] = np.tile(da.values, (ny,1))
    return feats

def flatten_nobf(ds, target="DOD"):
    fd = gather_features_nobf(ds,target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order='C') for n in names])
    y = ds[target].values.ravel(order='C')
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X,y,names,ok

# ────────────────────────────────────────────────────────────
def rf_eval_by_category(X, y, cat2d, ok):
    cat = cat2d.ravel(order='C')[ok]
    Xv, Yv = X[ok], y[ok]

    tr_idx, te_idx = [], []
    for c in (0,1,2,3):
        rows = np.where(cat==c)[0]
        if rows.size == 0:
            continue
        tr, te = train_test_split(rows, test_size=0.3, random_state=42)
        tr_idx.append(tr)
        te_idx.append(te)
    tr_idx = np.concatenate(tr_idx)
    te_idx = np.concatenate(te_idx)
    cat_te = cat[te_idx]

    X_tr, y_tr = Xv[tr_idx], Yv[tr_idx]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    X_te = Xv[te_idx]
    y_te = Yv[te_idx]
    yhat_te = rf.predict(X_te)

    print("Per-Category Test Metrics:")
    for c in (0, 1, 2, 3):
        sel = cat_te == c
        if not np.any(sel):
            print(f"  Category c{c}: No test samples.")
            continue
        bias = yhat_te[sel] - y_te[sel]
        mean_bias = bias.mean()
        bias_std = bias.std()
        r2 = r2_score(y_te[sel], yhat_te[sel])
        print(f"  Category c{c}:")
        print(f"    N          = {sel.sum()}")
        print(f"    Mean Bias  = {mean_bias:.2f} days")
        print(f"    Bias Std   = {bias_std:.2f} days")
        print(f"    R²         = {r2:.3f}")

# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Loading final_dataset6.nc ...")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset6.nc")

    # Burn category mask
    bc = ds["burn_cumsum"].values
    cat2d = np.zeros_like(bc, dtype=int)
    cat2d[bc < 0.25] = 0
    cat2d[(bc >= 0.25) & (bc < 0.50)] = 1
    cat2d[(bc >= 0.50) & (bc < 0.75)] = 2
    cat2d[bc >= 0.75] = 3

    X_all, y_all, _, ok = flatten_nobf(ds, "DSD")
    rf_eval_by_category(X_all, y_all, cat2d, ok)
