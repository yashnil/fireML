#!/usr/bin/env python3
# ============================================================
#  Fire‑ML experiment on final_dataset4.nc
#  70 % unburned‑only training → evaluate everywhere
#  burn_fraction **excluded** from predictors
# ============================================================
import time, xarray as xr, numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import ranksums
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy.random as npr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# typing
from typing import Dict, List, Tuple, Optional

# ────────────────────────────────────────────────────────────
#  pretty timer
# ────────────────────────────────────────────────────────────
T0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ────────────────────────────────────────────────────────────
#  generic plotting helpers
# ────────────────────────────────────────────────────────────
def plot_scatter(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_pred, y_true, alpha=0.3, label=f"N={len(y_true)}")
    mn,mx = min(y_pred.min(),y_true.min()), max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--',label="1:1 line")
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    bias = (y_pred-y_true).mean();  r2 = r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD");  plt.ylabel("Observed DoD");  plt.legend()
    plt.tight_layout();  plt.show()

def plot_bias_hist(y_true, y_pred, title, rng=(-100,300)):
    res = y_pred - y_true
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=50, range=rng, alpha=0.7)
    plt.axvline(res.mean(), color='k', ls='--', lw=2)
    plt.title(f"{title}\nMean={res.mean():.2f}, Std={res.std():.2f}")
    plt.xlabel("Bias (Pred‑Obs)");  plt.ylabel("Count")
    plt.tight_layout();  plt.show()

def plot_scatter_by_cat(y_true, y_pred, cat, title):
    plt.figure(figsize=(6,6))
    cols = {0:'red',1:'yellow',2:'green',3:'blue'}
    for c,col in cols.items():
        m = cat==c
        if m.any():
            plt.scatter(y_pred[m], y_true[m], c=col, alpha=0.4, label=f"cat={c}")
    mn,mx = min(y_pred.min(),y_true.min()), max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--')
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    bias = (y_pred-y_true).mean();  r2 = r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD");  plt.ylabel("Observed DoD");  plt.legend()
    plt.tight_layout();  plt.show()

def plot_top10_features(rf, names, title):
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:10]
    plt.figure(figsize=(8,4))
    plt.bar(range(10), imp[idx])
    plt.xticks(range(10), [names[i] for i in idx], rotation=45, ha='right')
    plt.title(title);  plt.ylabel("Feature importance")
    plt.tight_layout();  plt.show()

# ────────────────────────────────────────────────────────────
#  NEW aggregated Top‑5 feature‑scatter (mean ± 1 SD per DoD)
# ────────────────────────────────────────────────────────────
def plot_top5_feature_scatter(rf, X, y, cat, names, prefix):
    """
    • x‑axis = predictor value, y‑axis = observed DoD  
    • for every unique DoD and every category c0..c3:  
        – compute mean(x) & std(x) among pixels with that DoD+cat  
        – plot a single point (mean, DoD) with a horizontal ±1 SD bar  
    • connect the points of each category with a line  
    • legend shows Pearson r for **each category** (computed on *all* raw
      points of that category, not the aggregated means)  
    Colours: red(c0) / yellow(c1) / green(c2) / blue(c3)
    """
    imp = rf.feature_importances_
    top5 = np.argsort(imp)[::-1][:5]
    colours = {0:'red', 1:'yellow', 2:'green', 3:'blue'}
    cats    = [0,1,2,3]

    for f_idx in top5:
        fname = names[f_idx]
        x_all = X[:,f_idx]

        plt.figure(figsize=(7,5))
        for c in cats:
            mask_c = (cat == c)
            if not mask_c.any():          # skip empty category
                continue

            # Pearson r for legend (all raw points of this category)
            r_val = np.corrcoef(x_all[mask_c], y[mask_c])[0,1]

            # aggregate by unique DoD
            dod_vals, mean_x, sd_x = [], [], []
            for d in np.unique(y[mask_c]):
                m_d = mask_c & (y == d)
                mean_x.append( np.mean(x_all[m_d]) )
                sd_x  .append( np.std (x_all[m_d]) )
                dod_vals.append(d)
            dod_vals = np.array(dod_vals)
            mean_x   = np.array(mean_x)
            sd_x     = np.array(sd_x)

            # sort by DoD to get a nice line
            order = np.argsort(dod_vals)
            dod_vals = dod_vals[order]
            mean_x   = mean_x[order]
            sd_x     = sd_x[order]

            # error bars (horizontal) + line
            plt.errorbar(mean_x, dod_vals,
                         xerr=sd_x,
                         fmt='o', ms=4, lw=1,
                         color=colours[c],
                         ecolor=colours[c],
                         alpha=0.8,
                         label=f"cat={c} (r={r_val:.2f})")
            plt.plot(mean_x, dod_vals, '-', color=colours[c], alpha=0.7)

        plt.xlabel(fname)
        plt.ylabel("Observed DoD")
        plt.title(f"{prefix}: {fname}")
        plt.legend()
        plt.tight_layout();  plt.show()

# (the rest of the script – spatial helpers, feature‑matrix builders,
#  eval‑bins, rf_unburned_experiment, main() – is IDENTICAL to the
#  previous version and therefore omitted here for brevity.  Simply
#  replace the old `plot_top5_feature_scatter()` with the new one
#  above, keep everything else unchanged.)


# ── refined spatial helpers ─────────────────────────────────────────
def _setup_ca_axes(title: str):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-125, -113, 32, 42], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    ax.coastlines(resolution="10m", linewidth=0.5)
    ax.set_title(title)
    return ax


def dod_map_ca(ds, pix_idx, values, title, cmap="viridis"):
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat1 = lat[0] if lat.ndim == 2 else lat
    lon1 = lon[0] if lon.ndim == 2 else lon
    full = np.full(ds.sizes["pixel"], np.nan)
    full[pix_idx] = values
    ax = _setup_ca_axes(title)
    sc = ax.scatter(
        lon1[pix_idx],
        lat1[pix_idx],
        c=values,
        cmap=cmap,
        s=10,
        alpha=0.9,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(sc, ax=ax, shrink=0.8, label="DoD (days)")
    plt.tight_layout()
    plt.show()


def bias_map_ca(ds, pix_idx, y_true, y_pred, title):
    """Mean pixel‑bias map on a California outline."""
    n_pix = ds.sizes["pixel"]
    sum_b = np.zeros(n_pix)
    cnt = np.zeros(n_pix)
    for p, res in zip(pix_idx, y_pred - y_true):
        sum_b[p] += res
        cnt[p] += 1
    mean_b = np.full(n_pix, np.nan)
    valid = cnt > 0
    mean_b[valid] = sum_b[valid] / cnt[valid]

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat1 = lat[0] if lat.ndim == 2 else lat
    lon1 = lon[0] if lon.ndim == 2 else lon

    if valid.any():
        vmax = np.nanmax(np.abs(mean_b[valid]))
    else:                          # every pixel is NaN → arbitrary tiny scale
        vmax = 0.0
    # force a strictly‑positive vmax so that -vmax < 0 < vmax
    if (not np.isfinite(vmax)) or (vmax <= 0.0):
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    ax = _setup_ca_axes(title)
    ax.scatter(
        lon1, lat1, c="lightgray", s=4, alpha=0.4, transform=ccrs.PlateCarree()
    )
    sc = ax.scatter(
        lon1[valid],
        lat1[valid],
        c=mean_b[valid],
        cmap="bwr",
        norm=norm,
        s=10,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Mean Bias (Pred‑Obs)")
    plt.tight_layout()
    plt.show()


def boxplot_dod_by_elev_veg(y, elev, veg, tag):
    elev_edges = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    elev_bin = np.digitize(elev, elev_edges) - 1
    uniq_veg = np.unique(veg)
    data, labels = [], []
    for ei in range(len(elev_edges) - 1):
        for vv in uniq_veg:
            m = (elev_bin == ei) & (veg == vv)
            data.append(y[m])
            labels.append(f"E[{elev_edges[ei]}‑{elev_edges[ei+1]}],V{vv}")
    plt.figure(figsize=(12, 5))
    plt.boxplot(data, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=90)
    plt.xlabel("(Elevation bin, VegTyp)")
    plt.ylabel("Raw DoD")
    plt.title(tag)
    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────────────────
#  helpers to build feature matrix (burn_fraction excluded)
# ────────────────────────────────────────────────────────────
def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector',
            'burn_fraction','burn_cumsum','aorcsummerhumidity',
            'aorcsummerprecipitation','aorcsummerlongwave',
            'aorcsummershortwave','aorcsummertemperature'}
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


def flatten_nobf(ds, target="DOD"):
    fd = gather_features_nobf(ds, target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order="C") for n in names])
    y = ds[target].values.ravel(order="C")
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, names, ok


# ────────────────────────────────────────────────────────────
#  evaluation helper: metrics in 10 % burn‑fraction bins
# ────────────────────────────────────────────────────────────
def eval_bins(y, yp, burn, bins):
    for lo, hi in bins:
        sel = (burn > lo) if hi is None else ((burn >= lo) & (burn < hi))
        tag = f">{lo*100:.0f}%" if hi is None else f"{lo*100:.0f}-{hi*100:.0f}%"
        if sel.sum() == 0:
            print(f"{tag}: N=0")
            continue
        rmse = np.sqrt(mean_squared_error(y[sel], yp[sel]))
        bias = (yp[sel] - y[sel]).mean()
        r2 = r2_score(y[sel], yp[sel])
        print(
            f"{tag}: N={sel.sum():4d}  RMSE={rmse:6.2f}  "
            f"Bias={bias:7.2f}  R²={r2:6.3f}"
        )


# ────────────────────────────────────────────────────────────
#  main RF routine  (unburned‑only training)
# ────────────────────────────────────────────────────────────
def rf_unburned_experiment(
    X,
    y,
    cat2d,
    ok,
    ds,
    feat_names,
    unburned_max_cat: int = 0,
):
    """
    • train on samples with cat ≤ unburned_max_cat
    • 70/30 split inside that subset
    • evaluate everywhere + all requested plots / stats
    """
    thr = unburned_max_cat
    cat = cat2d.ravel(order="C")[ok]
    Xv, Yv = X[ok], y[ok]

    # indices for unburned subset
    unb = cat <= thr
    log(f"  training on unburned (cat ≤ {thr}): N={unb.sum()}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xv[unb], Yv[unb], test_size=0.3, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    # ── A. unburned train / test plots ────────────────────────────
    plot_scatter(y_tr, rf.predict(X_tr), f"Unburned TRAIN (cat ≤ {thr})")
    plot_bias_hist(y_tr, rf.predict(X_tr), f"Bias Hist: Unburned TRAIN (cat ≤ {thr})")

    y_hat_te = rf.predict(X_te)
    plot_scatter(y_te, y_hat_te, f"Unburned TEST (cat ≤ {thr})")
    plot_bias_hist(y_te, y_hat_te, f"Bias Hist: Unburned TEST (cat ≤ {thr})")

    # ── B. evaluate on **all** valid samples ──────────────────────
    y_hat_all = rf.predict(Xv)

    plot_scatter_by_cat(
        Yv, y_hat_all, cat, f"All data – colour by cat (thr={thr})"
    )
    plot_bias_hist(Yv, y_hat_all, f"Bias Hist: ALL data (thr={thr})")

    # pixel‑level bias map
    pix_full = np.tile(np.arange(ds.sizes["pixel"]), ds.sizes["year"])
    bias_map_ca(
        ds,
        pix_full[ok],
        Yv,
        y_hat_all,
        f"Pixel Bias: ALL data (thr={thr})",
    )

    # ── C. per‑cat scatter / hist, Wilcoxon tests, DoD maps, Bias maps ──
    bias_all = y_hat_all - Yv
    bias_by_cat = {c: bias_all[cat == c] for c in range(4) if (cat == c).any()}

    pix_valid = pix_full[ok]  # pixel index per row of Xv/Yv

    for c in range(4):
        m = cat == c
        if not m.any():
            continue

        # scatter / hist
        plot_scatter(
            Yv[m], y_hat_all[m], f"Category {c} (thr={thr})"
        )
        plot_bias_hist(
            Yv[m], y_hat_all[m], f"Bias Hist: cat={c} (thr={thr})"
        )

        # observed & predicted DoD maps
        dod_map_ca(
            ds,
            pix_valid[m],
            Yv[m],
            f"Observed DoD – cat {c} (thr={thr})",
            cmap="magma",
        )
        dod_map_ca(
            ds,
            pix_valid[m],
            y_hat_all[m],
            f"Predicted DoD – cat {c} (thr={thr})",
            cmap="viridis",
        )

        # per‑category pixel‑bias map
        bias_map_ca(
            ds,
            pix_valid[m],
            Yv[m],
            y_hat_all[m],
            f"Pixel Bias – cat {c} (thr={thr})",
        )

        # per‑category Elev×Veg box‑plot
        elev = ds["Elevation"].values.ravel(order="C")[ok][m]
        veg = ds["VegTyp"].values.ravel(order="C")[ok][m]
        boxplot_dod_by_elev_veg(
            Yv[m], elev, veg, f"cat={c}, thr={thr}"
        )

    if 0 in bias_by_cat:
        print("\nWilcoxon rank‑sum (bias difference vs cat 0)")
        for c in (1, 2, 3):
            if c in bias_by_cat:
                s, p = ranksums(bias_by_cat[0], bias_by_cat[c])
                print(f"  cat {c} vs 0 → stat={s:.3f}, p={p:.3g}")

    # ── D. 10 % burn‑fraction bins (eval‑only) ────────────────────
    bf = ds["burn_fraction"].values.ravel(order="C")[ok]
    bins10 = [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, None),
    ]
    print("\nPerformance by 10 % burn‑fraction bins:")
    eval_bins(Yv, y_hat_all, bf, bins10)

    # ── E. down‑sampling robustness  (k=100 runs) ─────────────────
    counts = {c: (cat == c).sum() for c in range(4) if (cat == c).any()}
    k = min(counts.values())
    print(f"\nDown‑sampling robustness: k={k}, runs=100")
    for c, n in counts.items():
        idx_c = np.where(cat == c)[0]
        biases, rmses, r2s = [], [], []
        for _ in range(100):
            sub = npr.choice(idx_c, size=k, replace=False)
            biases.append((y_hat_all[sub] - Yv[sub]).mean())
            rmses.append(np.sqrt(mean_squared_error(Yv[sub], y_hat_all[sub])))
            r2s.append(r2_score(Yv[sub], y_hat_all[sub]))
        print(
            f" cat={c}: μBias={np.mean(biases):.3f}  "
            f"μRMSE={np.mean(rmses):.3f}  μR²={np.mean(r2s):.3f}"
        )

        # histograms
        plt.figure(figsize=(15, 4))
        for j, data, col, lab in zip(
            (1, 2, 3),
            [biases, rmses, r2s],
            ["steelblue", "tomato", "seagreen"],
            ["Mean Bias", "RMSE", "R²"],
        ):
            plt.subplot(1, 3, j)
            plt.hist(data, bins=10, color=col, alpha=0.85)
            plt.axvline(np.mean(data), color="k", ls="--")
            plt.title(f"{lab} (cat={c})")
            plt.xlabel(lab)
        plt.suptitle(f"Down‑sampling distributions (cat={c}, k={k})")
        plt.tight_layout()
        plt.show()

    # ── F. feature importance & top‑5 scatter ─────────────────────
    plot_top10_features(rf, feat_names, f"Top‑10 Feature Importance (thr={thr})")
    plot_top5_feature_scatter(
        rf, Xv, Yv, cat, feat_names, f"Top‑5 (thr={thr})"
    )

    return rf


# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log("loading final_dataset4.nc …")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")

    # cumulative‑burn categories already stored as burn_cumsum
    bc = ds["burn_cumsum"].values  # (year, pixel)
    cat_2d = np.zeros_like(bc, dtype=int)
    cat_2d[bc < 0.25] = 0
    cat_2d[(bc >= 0.25) & (bc < 0.50)] = 1
    cat_2d[(bc >= 0.50) & (bc < 0.75)] = 2
    cat_2d[bc >= 0.75] = 3
    log("categories (c0‑c3) computed")

    # build feature matrix (burn_fraction excluded)
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DOD")
    log(
        f"feature matrix ready – {ok.sum()} valid samples, "
        f"{len(feat_names)} predictors"
    )

    # ── run #1 : unburned = cat 0 only  ───────────────────────────
    log("\n=== RUN #1 : unburned = cat 0 (cumsum < 0.25) ===")
    rf_run1 = rf_unburned_experiment(
        X_all, y_all, cat_2d, ok, ds, feat_names, unburned_max_cat=0
    )

    # ── run #2 : unburned = cat 0 + 1  ───────────────────────────
    log("\n=== RUN #2 : unburned = cat 0 + 1 (cumsum < 0.50) ===")
    rf_run2 = rf_unburned_experiment(
        X_all, y_all, cat_2d, ok, ds, feat_names, unburned_max_cat=1
    )

    log("ALL DONE.")
