import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --------------------------------------------------
# 1.  Load data & build category mask
# --------------------------------------------------
DATA_PATH = "/Users/yashnilmohanty/Desktop/final_dataset5.nc"
ds = xr.open_dataset(DATA_PATH)

bc = ds["burn_cumsum"].values          # (year, pixel)
cat_2d = np.zeros_like(bc, dtype=int)
cat_2d[bc < 0.25]                    = 0
cat_2d[(bc >= 0.25) & (bc < 0.50)]   = 1
cat_2d[(bc >= 0.50) & (bc < 0.75)]   = 2
cat_2d[bc >= 0.75]                   = 3

# --------------------------------------------------
# 2.  Feature matrix (burn_fraction excluded)
# --------------------------------------------------
def gather_features_nobf(ds, target="DSD"):
    excl = {target.lower(), 'lat','lon','latitude','longitude',
        'pixel','year','ncoords_vector','nyears_vector',
        'burn_fraction','burn_cumsum',
        'aorcSummerHumidity','aorcSummerPrecipitation',
        'aorcSummerLongwave','aorcSummerShortwave',
        'aorcSummerTemperature'}
    feats = {}
    ny = ds.sizes["year"]
    for v in ds.data_vars:
        if v.lower() in excl:
            continue
        da = ds[v]
        if v == "aorcWinterPrecipitation":     # mm s⁻¹ → mm day⁻¹
            da = da * 86_400.0
        if set(da.dims) == {"year", "pixel"}:
            feats[v] = da.values
        elif set(da.dims) == {"pixel"}:
            feats[v] = np.tile(da.values, (ny, 1))
    return feats

fd = gather_features_nobf(ds, "DSD")
feat_names = sorted(fd)

X = np.column_stack([fd[n].ravel(order="C") for n in feat_names])
y = ds["DSD"].values.ravel(order="C")

ok  = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
Xv, Yv           = X[ok], y[ok]
cat = cat_2d.ravel(order="C")[ok]

# --------------------------------------------------
# 3.  Category‑wise 70 / 30 split (no global shuffle)
# --------------------------------------------------
for c in range(4):
    idx = np.where(cat == c)[0]          # deterministic ordering
    if idx.size == 0:
        print(f"Category {c}: no samples available");  continue

    tr_idx, te_idx = train_test_split(
        idx, test_size=0.30, random_state=42, shuffle=True
    )

    X_tr, y_tr = Xv[tr_idx], Yv[tr_idx]
    X_te, y_te = Xv[te_idx], Yv[te_idx]

    rf = RandomForestRegressor(
        n_estimators=100,
        bootstrap=True,          # matches original script
        random_state=42
    )
    rf.fit(X_tr, y_tr)

    y_pred     = rf.predict(X_te)
    residuals  = y_pred - y_te
    print(f"Category {c}: "
          f"mean bias={residuals.mean():.2f}, "
          f"bias std={residuals.std():.2f}, "
          f"R^2={r2_score(y_te, y_pred):.2f}")
