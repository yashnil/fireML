#!/usr/bin/env python3
# ============================================================
# Compare top-10 feature importances: Exclude vs. Include burn_fraction
# ============================================================
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ────────────────────────────────────────────────────────────
# 1) Dynamically import both newversion.py files
# ────────────────────────────────────────────────────────────
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

exclude_path = "/Users/yashnilmohanty/Desktop/fireML/febWorkScreen2/excludeBurnFraction/newversion.py"
include_path = "/Users/yashnilmohanty/Desktop/fireML/febWorkScreen2/includeBurnFraction/newversion.py"

exclude_mod = load_module("exclude_mod", exclude_path)
include_mod = load_module("include_mod", include_path)

# ────────────────────────────────────────────────────────────
# 2) Load dataset and prepare category matrix
# ────────────────────────────────────────────────────────────
ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")
bc = ds["burn_cumsum"].values
cat_2d = np.zeros_like(bc, dtype=int)
cat_2d[bc < 0.25] = 0
cat_2d[(bc >= 0.25) & (bc < 0.5)] = 1
cat_2d[(bc >= 0.5) & (bc < 0.75)] = 2
cat_2d[bc >= 0.75] = 3

# ────────────────────────────────────────────────────────────
# 3) Flatten and train models (only to get feature importances)
# ────────────────────────────────────────────────────────────
X_ex, y_ex, names_ex, ok_ex = exclude_mod.flatten_nobf(ds, "DOD")
X_in, y_in, names_in, ok_in = include_mod.flatten(ds, "DOD")

rf_ex = exclude_mod.rf_experiment_nobf(X_ex, y_ex, cat_2d, ok_ex, ds, names_ex)
rf_in = include_mod.rf_experiment_nobf(X_in, y_in, cat_2d, ok_in, ds, names_in)

# ────────────────────────────────────────────────────────────
# 4) Extract and align top-10 feature importances
# ────────────────────────────────────────────────────────────
imp_ex = rf_ex.feature_importances_
imp_in = rf_in.feature_importances_

top10_idx = np.argsort(imp_ex)[::-1][:10]
top10_feats = [names_ex[i] for i in top10_idx]
imp_ex_top10 = imp_ex[top10_idx]
imp_in_top10 = [imp_in[names_in.index(f)] for f in top10_feats]

# ────────────────────────────────────────────────────────────
# 5) Plot side-by-side comparative histogram
# ────────────────────────────────────────────────────────────
x = np.arange(len(top10_feats))
width = 0.35

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x - width/2, imp_ex_top10, width, label="Exclude Burn Fraction", color="blue", alpha=0.8)
ax.bar(x + width/2, imp_in_top10, width, label="Include Burn Fraction", color="red", alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(top10_feats, rotation=45, ha="right")
ax.set_ylabel("Feature Importance")
ax.set_title("Top‑10 Feature Importance Comparison")
ax.legend()
plt.tight_layout()
plt.show()
