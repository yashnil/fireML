
#!/usr/bin/env python3
# ============================================================
#  Absolute predictor contribution via permutation‑importance
#  • model A  = 70/30 per‑cat, burn_fraction EXCLUDED
#  • model B  = 70/30 per‑cat, burn_fraction INCLUDED
#  The test set is identical for both models
# ============================================================
import time, importlib.util, xarray as xr, numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance            # ★
from typing import Dict

# ------------------------------------------------------------------
# paths to your two “newversion.py” files
# ------------------------------------------------------------------
EXCLUDE_PY = "/Users/yashnilmohanty/Desktop/fireML/febWorkScreen2/excludeBurnFraction/newversion.py"
INCLUDE_PY = "/Users/yashnilmohanty/Desktop/fireML/febWorkScreen2/includeBurnFraction/newversion.py"
DS_NC      = "/Users/yashnilmohanty/Desktop/final_dataset5.nc"
RSEED      = 42
N_EST      = 100

def load_module(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def log(msg): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

# ------------------------------------------------------------------
# 1) load helper functions from both scripts
# ------------------------------------------------------------------
T0 = time.time()
exclude_mod = load_module("exclude_mod", EXCLUDE_PY)
include_mod = load_module("include_mod", INCLUDE_PY)

# ------------------------------------------------------------------
# 2) open dataset and build category matrix
# ------------------------------------------------------------------
log("loading dataset …")
ds = xr.open_dataset(DS_NC)

bc = ds["burn_cumsum"].values
cat_2d = np.zeros_like(bc, dtype=int)
cat_2d[bc<0.25] = 0
cat_2d[(bc>=0.25)&(bc<0.5)]  = 1
cat_2d[(bc>=0.5) &(bc<0.75)] = 2
cat_2d[bc>=0.75]             = 3
cat_flat = cat_2d.ravel(order='C')

# ------------------------------------------------------------------
# 3) build feature matrices
# ------------------------------------------------------------------
log("building feature matrices …")
XA, y, namesA, okA = exclude_mod.flatten_nobf(ds, "DSD")    # no BF
XB, _, namesB, okB = include_mod.flatten(ds, "DSD")         # + BF

ok   = okA & okB                                # rows valid for both
cat  = cat_flat[ok]
XA   = XA[ok]
XB   = XB[ok]
y    = y[ok]

# sanity check – feature names
namesA = list(namesA)
namesB = list(namesB)

# ------------------------------------------------------------------
# 4) build identical 70 %/30 % per‑category indices  (same as Test‑3)
# ------------------------------------------------------------------
tr_idx, te_idx = [], []
for c in (0,1,2,3):
    rows = np.where(cat == c)[0]
    if rows.size==0: continue
    tr, te = train_test_split(rows, test_size=0.3, random_state=RSEED)
    tr_idx.append(tr);  te_idx.append(te)
tr_idx = np.concatenate(tr_idx)
te_idx = np.concatenate(te_idx)

XA_tr, XA_te = XA[tr_idx], XA[te_idx]
XB_tr, XB_te = XB[tr_idx], XB[te_idx]
y_tr , y_te  =  y[tr_idx],  y[te_idx]

log(f"test‑set size: {len(te_idx)}   (c0={np.sum(cat[te_idx]==0)}, "
    f"c1={np.sum(cat[te_idx]==1)}, c2={np.sum(cat[te_idx]==2)}, "
    f"c3={np.sum(cat[te_idx]==3)})")

# ------------------------------------------------------------------
# 5) train the two Random‑Forest models
# ------------------------------------------------------------------
rfA = RandomForestRegressor(n_estimators=N_EST, random_state=RSEED)
rfB = RandomForestRegressor(n_estimators=N_EST, random_state=RSEED)
rfA.fit(XA_tr, y_tr)
rfB.fit(XB_tr, y_tr)

# ------------------------------------------------------------------
# 6) permutation‑importance on the identical test set
#    metric = increase in RMSE  (bigger = more important)
# ------------------------------------------------------------------
def perm_importance(rf, X, y_true, feature_names, n_repeats=10):
    base = np.sqrt(mean_squared_error(y_true, rf.predict(X)))
    res  = permutation_importance(
                rf, X, y_true, n_repeats=n_repeats,
                scoring="neg_root_mean_squared_error",
                random_state=RSEED, n_jobs=-1)
    # convert from negative RMSE to ∆RMSE
    delta = base - (-res.importances_mean)
    imp   = dict(zip(feature_names, delta))
    return imp, base

log("computing permutation importance (this may take a minute) …")
impA, base_rmse_A = perm_importance(rfA, XA_te, y_te, namesA)
impB, base_rmse_B = perm_importance(rfB, XB_te, y_te, namesB)

# ------------------------------------------------------------------
# 7) unify feature lists  (so we can compare directly)
# ------------------------------------------------------------------
all_feats = set(impA) | set(impB)
valsA = np.array([impA.get(f, 0.0) for f in all_feats])
valsB = np.array([impB.get(f, 0.0) for f in all_feats])

#  pick top‑15 by *either* model
top_idx = np.argsort(np.maximum(valsA, valsB))[::-1][:15]
top_feats = np.array(list(all_feats))[top_idx]
valsA_top = valsA[top_idx]
valsB_top = valsB[top_idx]

# ------------------------------------------------------------------
# 8) print a neat table
# ------------------------------------------------------------------
print("\n∆RMSE on identical 30 % test set  (higher = more important)")
print(f"Baseline RMSE  model‑A(noBF) = {base_rmse_A:5.2f}   "
      f"model‑B(+BF) = {base_rmse_B:5.2f}\n")
print("{:<3s} {:<30s} {:>8s} {:>8s}".format("#","Predictor","ΔRMSE_A","ΔRMSE_B"))
for j,(f,a,b) in enumerate(zip(top_feats, valsA_top, valsB_top),1):
    print(f"{j:<3d} {f:<30s} {a:8.3f} {b:8.3f}")

# ------------------------------------------------------------------
# 9) side‑by‑side bar plot  (absolute ΔRMSE)
# ------------------------------------------------------------------
x = np.arange(len(top_feats))
w = 0.35

fig, ax = plt.subplots(figsize=(12,5))
ax.bar(x-w/2, valsA_top, width=w, label="no burn_fraction", color="dodgerblue")
ax.bar(x+w/2, valsB_top, width=w, label="+ burn_fraction",  color="firebrick")

ax.set_xticks(x)
ax.set_xticklabels(top_feats, rotation=45, ha="right")
ax.set_ylabel("ΔRMSE on test set (days)")
ax.set_title("Permutation‑based absolute importance (top 15)")
ax.legend()
fig.tight_layout()
plt.show()
