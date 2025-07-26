#!/usr/bin/env python3
# ============================================================
#  Apples‑to‑apples comparison:
#   – model A = Test‑1  (unburned‑only;  NO burn_fraction)
#   – model B = Test‑3  (per‑category;  + burn_fraction)
#  Evaluate both on the *same* 30 %‑per‑category test set that
#  was produced while training model B
# ============================================================
import time, xarray as xr, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict

# ─────────────────────────────
DS_PATH = "/Users/yashnilmohanty/Desktop/final_dataset5.nc"
RSEED   = 42
N_EST   = 100
# ─────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

# ---------- helpers -----------------------------------------------------------
def gather_features(ds, exclude_bf: bool) -> Dict[str, np.ndarray]:
    excl = {'dod','lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector',
            'aorcsummerhumidity','aorcsummerprecipitation',
            'aorcsummerlongwave','aorcsummershortwave',
            'aorcsummertemperature'}
    if exclude_bf:
        excl |= {'burn_fraction', 'burn_cumsum'}

    ny   = ds.sizes["year"]
    feats = {}
    for v in ds.data_vars:
        if v.lower() in excl:
            continue
        da = ds[v]
        if set(da.dims) == {'year', 'pixel'}:
            feats[v] = da.values
        elif set(da.dims) == {'pixel'}:
            feats[v] = np.tile(da.values, (ny, 1))
    return feats


def flatten(ds, exclude_bf: bool):
    fd    = gather_features(ds, exclude_bf)
    names = sorted(fd)
    X     = np.column_stack([fd[n].ravel(order='C') for n in names])
    y     = ds['DOD'].values.ravel(order='C')
    ok    = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, np.array(names), ok


def metric_triplet(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = (y_pred - y_true).mean()
    r2   = r2_score(y_true, y_pred)
    return rmse, bias, r2

# ────────────────────────────────────────────────────────────
# 0. Load data & build feature matrices
# ────────────────────────────────────────────────────────────
T0 = time.time()
log("loading dataset …")
ds = xr.open_dataset(DS_PATH)

# burn‑cumsum categories (c0‑c3)
bc = ds["burn_cumsum"].values
cat_2d          = np.zeros_like(bc, dtype=int)
cat_2d[bc<0.25] = 0
cat_2d[(bc>=0.25)&(bc<0.5)]  = 1
cat_2d[(bc>=0.5) &(bc<0.75)] = 2
cat_2d[bc>=0.75]             = 3
cat_flat = cat_2d.ravel(order='C')

log("building feature matrices …")
XA, y_all, _, okA = flatten(ds, exclude_bf=True)   # model A:  no BF
XB, _,     _, okB = flatten(ds, exclude_bf=False)  # model B: + BF

ok   = okA & okB          # rows valid for *both* sets
cat  = cat_flat[ok]
XA   = XA[ok]
XB   = XB[ok]
y    = y_all[ok]

log(f"valid rows for both models : {ok.sum()}")

# ────────────────────────────────────────────────────────────
# 1. Train model A (unburned‑only, category 0)
# ────────────────────────────────────────────────────────────
unburned = cat == 0
XA_tr, XA_te, yA_tr, _ = train_test_split(
    XA[unburned], y[unburned], test_size=0.3, random_state=RSEED)

rfA = RandomForestRegressor(n_estimators=N_EST, random_state=RSEED)
rfA.fit(XA_tr, yA_tr)

# ────────────────────────────────────────────────────────────
# 2. Train model B (70 / 30 split *inside every category*)
# ────────────────────────────────────────────────────────────
train_idx_B, test_idx_B = [], []
for c in (0, 1, 2, 3):
    rows = np.where(cat == c)[0]
    if rows.size == 0:
        continue
    tr, te = train_test_split(rows, test_size=0.3, random_state=RSEED)
    train_idx_B.append(tr);  test_idx_B.append(te)
train_idx_B = np.concatenate(train_idx_B)
test_idx_B  = np.concatenate(test_idx_B)

XB_tr, yB_tr = XB[train_idx_B], y[train_idx_B]
XB_te, y_te  = XB[test_idx_B],  y[test_idx_B]   # common test target

rfB = RandomForestRegressor(n_estimators=N_EST, random_state=RSEED)
rfB.fit(XB_tr, yB_tr)

log(f"size of common test set: {len(test_idx_B)} "
    f"(c0={np.sum(cat[test_idx_B]==0)}, "
    f"c1={np.sum(cat[test_idx_B]==1)}, "
    f"c2={np.sum(cat[test_idx_B]==2)}, "
    f"c3={np.sum(cat[test_idx_B]==3)})")

# ────────────────────────────────────────────────────────────
# 3. Predictions on *identical* test rows
# ────────────────────────────────────────────────────────────
yhat_A = rfA.predict(XA[test_idx_B])   # model A
yhat_B = rfB.predict(XB_te)            # model B

# ────────────────────────────────────────────────────────────
# 4. Per‑category metrics (dicts)
# ────────────────────────────────────────────────────────────
statsA: Dict[int, Dict[str, float]] = {}
statsB: Dict[int, Dict[str, float]] = {}

cats_present = np.unique(cat[test_idx_B])

for c in cats_present:
    m = cat[test_idx_B] == c
    rmA, biA, r2A = metric_triplet(y_te[m], yhat_A[m])
    rmB, biB, r2B = metric_triplet(y_te[m], yhat_B[m])

    statsA[c] = {"rmse": rmA, "bias": biA, "r2": r2A}
    statsB[c] = {"rmse": rmB, "bias": biB, "r2": r2B}

# ────────────────────────────────────────────────────────────
# 5. Pretty table
# ────────────────────────────────────────────────────────────
print("\n=== Per‑category metrics on *identical* test set ===")
hdr = "{:<3s} {:>8s} {:>8s} {:>6s}   {:>8s} {:>8s} {:>6s}"
print(hdr.format("cat", "RMSE_A", "Bias_A", "R²_A",
                       "RMSE_B", "Bias_B", "R²_B"))
for c in cats_present:
    a = statsA[c];  b = statsB[c]
    print(f"{c:<3d} {a['rmse']:8.2f} {a['bias']:8.2f} {a['r2']:6.3f}   "
          f"{b['rmse']:8.2f} {b['bias']:8.2f} {b['r2']:6.3f}")

# ────────────────────────────────────────────────────────────
# 6. Side‑by‑side bar charts
# ────────────────────────────────────────────────────────────
metrics = ["bias", "rmse", "r2"]
labels  = ["Mean bias (days)", "RMSE (days)", "R²"]
x_pos   = np.arange(len(cats_present))
w       = 0.35

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for j, (met, lab) in enumerate(zip(metrics, labels)):
    ax = axes[j]
    valsA = [statsA[c][met] for c in cats_present]
    valsB = [statsB[c][met] for c in cats_present]

    ax.bar(x_pos - w/2, valsA, width=w,
           label="Model A (unburned‑only)", color="steelblue")
    ax.bar(x_pos + w/2, valsB, width=w,
           label="Model B (per‑cat + BF)", color="firebrick")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"c{c}" for c in cats_present])
    ax.set_title(lab)
    ax.set_xlabel("Category")
    if j == 0:
        ax.legend()

fig.suptitle("Apples‑to‑apples comparison on identical 30 % test set")
fig.tight_layout()
plt.show()
