#!/usr/bin/env python3
# ============================================================
#  Fire-ML · LSTM evaluation on final_dataset5.nc  (Experiment 2)
#  70 %/30 % split inside each burn-category
#  burn_fraction EXCLUDED · all *aorcSummer* predictors EXCLUDED
#  Prints: mean bias, bias SD, R² for c0–c3
# ============================================================

import time, xarray as xr, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ─── helpers ────────────────────────────────────────────────────────────
def gather_features_nobf(ds, target="DSD"):
    """
    Collect every data variable except
      • the target
      • simple coordinate / bookkeeping arrays
      • ANY variable whose name begins with 'aorcSummer'
      • burn_fraction           (← excluded for Exp-2)
    """
    skip = {target.lower(), 'burn_fraction', 'burn_cumsum',
            'lat', 'lon', 'latitude', 'longitude',
            'pixel', 'year', 'ncoords_vector', 'nyears_vector'}
    ny, feats = ds.sizes['year'], {}

    for v in ds.data_vars:
        name = v.lower()
        if name in skip or name.startswith("aorcsummer"):
            continue

        da = ds[v]
        if set(da.dims) == {'year', 'pixel'}:
            feats[v] = da.values
        elif set(da.dims) == {'pixel'}:     # broadcast static vars
            feats[v] = np.tile(da.values, (ny, 1))

    return feats


def flatten_nobf(ds, target="DSD"):
    feats = gather_features_nobf(ds, target)
    names = sorted(feats)
    X = np.column_stack([feats[n].ravel(order="C") for n in names])
    y = ds[target].values.ravel(order="C")
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, names, ok


def print_metrics(tag, y_true, y_pred):
    resid = y_pred - y_true
    print(f"[{tag}]  N={y_true.size:6d}   "
          f"mean bias={resid.mean():+8.3f}   "
          f"bias SD={resid.std():8.3f}   "
          f"R²={r2_score(y_true, y_pred):6.3f}")


# ─── main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    log = lambda s: print(f"[{time.time()-t0:6.1f}s] {s}")

    log("loading dataset …")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset5.nc")

    # burn-categories
    bc = ds["burn_cumsum"].values
    cat2d = np.zeros_like(bc, int)
    cat2d[bc < 0.25]               = 0
    cat2d[(bc >= 0.25) & (bc < .5)]  = 1
    cat2d[(bc >= 0.50) & (bc < .75)] = 2
    cat2d[bc >= .75]               = 3
    log("burn categories ready")

    # feature matrix (NO burn_fraction)
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DSD")
    cat_flat = cat2d.ravel(order="C")[ok]
    Xv, yv   = X_all[ok], y_all[ok]
    log(f"{Xv.shape[1]} predictors, {Xv.shape[0]} rows")

    # 70/30 split inside each category
    tr_idx, te_idx = [], []
    for c in (0, 1, 2, 3):
        rows = np.where(cat_flat == c)[0]
        if rows.size:
            tr, te = train_test_split(rows, test_size=0.30, random_state=42)
            tr_idx.append(tr); te_idx.append(te)
    tr_idx, te_idx = np.concatenate(tr_idx), np.concatenate(te_idx)

    # ── scale features & target ──────────────────────────────────────
    xsc = StandardScaler().fit(Xv[tr_idx])
    ysc = StandardScaler().fit(yv[tr_idx].reshape(-1, 1))

    X_tr = xsc.transform(Xv[tr_idx])
    X_te = xsc.transform(Xv[te_idx])
    y_tr = ysc.transform(yv[tr_idx].reshape(-1, 1)).ravel()

    # reshape for LSTM: (samples, timesteps=1, features)
    to3d = lambda a: a.reshape(a.shape[0], 1, a.shape[1])

    # ── build & train LSTM ───────────────────────────────────────────
    model = Sequential([
        LSTM(32, input_shape=(1, X_tr.shape[1])),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse")

    log("training LSTM …")
    model.fit(to3d(X_tr), y_tr,
              epochs=100, batch_size=512,
              verbose=0)
    log("… done")

    # ── predictions (invert y-scaling) ──────────────────────────────
    y_hat_te = ysc.inverse_transform(
        model.predict(to3d(X_te), batch_size=512)).ravel()

    # ── metrics per category ────────────────────────────────────────
    print("\n=== 30 % hold-out metrics ===")
    for c in (0, 1, 2, 3):
        mask = cat_flat[te_idx] == c
        if mask.any():
            print_metrics(f"c{c}", yv[te_idx][mask], y_hat_te[mask])
        else:
            print(f"[c{c}]  N=0   — skipped")

    log("ALL DONE (Experiment 2 – LSTM)")
