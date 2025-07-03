

#!/usr/bin/env python3
# ============================================================
#  Fire-ML · MLP evaluation on final_dataset5.nc  (Experiment 3)
#  70 %/30 % split inside each burn-category; burn_fraction INCLUDED
#  AORC-summer variables EXCLUDED
#  Prints: mean bias, bias SD, R² for c0–c3
# ============================================================

import time, xarray as xr, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# ─── helpers ────────────────────────────────────────────────────────────
def gather_features_incbf(ds, target="DOD"):
    """
    Collect all data vars except:
      • the target itself
      • simple coords
      • *any* variable whose name starts with 'aorcSummer'
    """
    base_skip = {target.lower(), 'lat', 'lon', 'latitude', 'longitude',
                 'pixel', 'year', 'ncoords_vector', 'nyears_vector'}
    ny = ds.sizes['year']
    feats = {}
    for v in ds.data_vars:
        if v.lower() in base_skip:
            continue
        if v.lower().startswith("aorcsummer"):   # ← NEW summer filter
            continue
        da = ds[v]
        if set(da.dims) == {'year', 'pixel'}:
            feats[v] = da.values
        elif set(da.dims) == {'pixel'}:          # broadcast to years
            feats[v] = np.tile(da.values, (ny, 1))
    return feats

def flatten_incbf(ds, target="DOD"):
    feats = gather_features_incbf(ds, target)
    names = sorted(feats)
    X = np.column_stack([feats[n].ravel(order='C') for n in names])
    y = ds[target].values.ravel(order='C')
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, names, ok

def print_metrics(tag, y_true, y_pred):
    resid = y_pred - y_true
    print(f"[{tag}]  N={y_true.size:6d}   "
          f"mean bias={resid.mean():8.3f}   "
          f"bias SD={resid.std():8.3f}   "
          f"R²={r2_score(y_true, y_pred):6.3f}")

# ─── main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time(); log = lambda s: print(f"[{time.time()-t0:6.1f}s] {s}")

    log("loading dataset …")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset5.nc")

    # burn categories
    bc = ds["burn_cumsum"].values
    cat2d = np.zeros_like(bc, int)
    cat2d[bc < 0.25]               = 0
    cat2d[(bc >= 0.25) & (bc < .5)]  = 1
    cat2d[(bc >= 0.50) & (bc < .75)] = 2
    cat2d[bc >= .75]               = 3
    log("burn categories ready")

    # feature matrix (burn_fraction kept, AORC-summer skipped)
    X_all, y_all, feat_names, ok = flatten_incbf(ds, "DOD")
    cat_flat = cat2d.ravel(order='C')[ok]
    Xv, yv   = X_all[ok], y_all[ok]
    log(f"{Xv.shape[1]} predictors, {Xv.shape[0]} rows")

    # 70/30 split per category
    tr_idx, te_idx = [], []
    for c in (0, 1, 2, 3):
        rows = np.where(cat_flat == c)[0]
        if rows.size:
            tr, te = train_test_split(rows, test_size=0.30, random_state=42)
            tr_idx.append(tr); te_idx.append(te)
    tr_idx, te_idx = np.concatenate(tr_idx), np.concatenate(te_idx)

    # scale + train
    scaler = StandardScaler().fit(Xv[tr_idx])
    X_tr, X_te = scaler.transform(Xv[tr_idx]), scaler.transform(Xv[te_idx])

    mlp = MLPRegressor(hidden_layer_sizes=(64, 64),
                       max_iter=1000, solver='adam', random_state=42)
    log("training MLP …")
    mlp.fit(X_tr, yv[tr_idx]); log("… done")

    y_hat_te = mlp.predict(X_te)

    # metrics
    print("\n=== 30 % hold-out metrics ===")
    for c in (0, 1, 2, 3):
        mask = cat_flat[te_idx] == c
        if mask.any():
            print_metrics(f"c{c}", yv[te_idx][mask], y_hat_te[mask])
        else:
            print(f"[c{c}]  N=0   — skipped")

    log("ALL DONE (Experiment 3 – MLP)")
