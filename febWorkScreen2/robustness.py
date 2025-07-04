
import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Path to your dataset
DATA_PATH = "/Users/yashnilmohanty/Desktop/final_dataset5.nc"

# Load dataset
ds = xr.open_dataset(DATA_PATH)

# Build cumulative-burn categories (year, pixel)
bc = ds["burn_cumsum"].values
cat_2d = np.zeros_like(bc, dtype=int)
cat_2d[bc < 0.25] = 0
cat_2d[(bc >= 0.25) & (bc < 0.50)] = 1
cat_2d[(bc >= 0.50) & (bc < 0.75)] = 2
cat_2d[bc >= 0.75] = 3

# Function to gather features excluding burn_fraction

def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(), 'lat', 'lon', 'latitude', 'longitude',
            'pixel', 'year', 'burn_fraction', 'burn_cumsum'}
    feats = {}
    ny = ds.sizes["year"]
    for v in ds.data_vars:
        if v.lower() in excl:
            continue
        da = ds[v]
        if v == "aorcWinterPrecipitation":
            da = da * 86400.0
        if set(da.dims) == {"year", "pixel"}:
            feats[v] = da.values
        elif set(da.dims) == {"pixel"}:
            feats[v] = np.tile(da.values, (ny, 1))
    return feats

# Flatten features and target
fd = gather_features_nobf(ds, "DOD")
feat_names = sorted(fd)
X = np.column_stack([fd[n].ravel(order="C") for n in feat_names])
y = ds["DOD"].values.ravel(order="C")
ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
Xv, Yv = X[ok], y[ok]

# Flatten categories
cat = cat_2d.ravel(order="C")[ok]

# Loop over each burn category for robustness test
for c in range(4):
    idx = np.where(cat == c)[0]
    if idx.size == 0:
        print(f"Category {c}: no samples available")
        continue

    # Extract data for this category
    X_c = Xv[idx]
    y_c = Yv[idx]

    # Split 70/30
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_c, y_c, test_size=0.30, random_state=42
    )

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    # Predict and compute metrics
    y_pred = rf.predict(X_te)
    residuals = y_pred - y_te
    mean_bias = residuals.mean()
    std_bias = residuals.std()
    r2 = r2_score(y_te, y_pred)

    # Print results
    print(f"Category {c}: mean bias={mean_bias:.2f}, bias std={std_bias:.2f}, R^2={r2:.2f}")
