#!/usr/bin/env python3
# ============================================================
#  Fire‑ML  ·  MLP experiment on final_dataset4.nc
#  70 % unburned‑only training  →  evaluate everywhere
#  burn_fraction **excluded** from predictors
# ============================================================
import time, xarray as xr, numpy as np, matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ranksums
import numpy.random as npr

# ────────────────────────────────────────────────────────────
#  pretty timer
# ────────────────────────────────────────────────────────────
T0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ────────────────────────────────────────────────────────────
#  plotting helpers  (scatter / bias‑hist / importance plots)
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
    cols = {0:'red', 1:'yellow', 2:'green', 3:'blue'}
    for c,col in cols.items():
        m = cat==c
        if m.any():
            plt.scatter(y_pred[m],y_true[m],c=col,alpha=0.4,label=f"cat={c}")
    mn,mx = min(y_pred.min(),y_true.min()), max(y_pred.max(),y_true.max())
    plt.plot([mn,mx],[mn,mx],'k--')
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    bias = (y_pred-y_true).mean();  r2 = r2_score(y_true,y_pred)
    plt.title(f"{title}\nRMSE={rmse:.2f}, bias={bias:.2f}, R²={r2:.3f}")
    plt.xlabel("Predicted DoD");  plt.ylabel("Observed DoD");  plt.legend()
    plt.tight_layout();  plt.show()

def plot_top10_perm_importance(imp, names, title):
    idx = np.argsort(imp)[::-1][:10]
    plt.figure(figsize=(8,4))
    plt.bar(range(10), imp[idx])
    plt.xticks(range(10), [names[i] for i in idx], rotation=45, ha='right')
    plt.title(title);  plt.ylabel("Permutation importance (ΔMSE)")
    plt.tight_layout();  plt.show()

def plot_top5_feature_scatter(imp, X, y, cat, names, prefix):
    """
    Aggregated Top-5 feature scatter:
      – x-axis  = mean predictor value (pixels sharing DoD & category)
      – y-axis  = observed DoD
      – horizontal bar = ±1 SD
      – one coloured line per cumulative-burn category
    """
    top5    = np.argsort(imp)[::-1][:5]
    colours = {0:'red', 1:'yellow', 2:'green', 3:'blue'}
    cats    = [0, 1, 2, 3]

    for f_idx in top5:
        fname = names[f_idx]
        x_all = X[:, f_idx]

        plt.figure(figsize=(7, 5))
        for c in cats:
            mask_c = (cat == c)
            if not mask_c.any():
                continue

            r_val = np.corrcoef(x_all[mask_c], y[mask_c])[0, 1]

            # aggregate by unique DoD
            dod_vals, mean_x, sd_x = [], [], []
            for d in np.unique(y[mask_c]):
                m_d = mask_c & (y == d)
                mean_x.append(np.mean(x_all[m_d]))
                sd_x  .append(np.std (x_all[m_d]))
                dod_vals.append(d)
            dod_vals, mean_x, sd_x = map(np.asarray, (dod_vals, mean_x, sd_x))
            order = np.argsort(dod_vals)
            dod_vals, mean_x, sd_x = dod_vals[order], mean_x[order], sd_x[order]

            plt.errorbar(mean_x, dod_vals,
                         xerr=sd_x,
                         fmt='o', ms=4, lw=1,
                         color=colours[c], ecolor=colours[c],
                         alpha=0.8,
                         label=f"cat={c} (r={r_val:.2f})")
            plt.plot(mean_x, dod_vals, '-', color=colours[c], alpha=0.7)

        plt.xlabel(fname)
        plt.ylabel("Observed DoD")
        plt.title(f"{prefix}: {fname}")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ────────────────────────────────────────────────────────────
#  feature‑matrix helpers  (burn_fraction & burn_cumsum excluded)
# ────────────────────────────────────────────────────────────
def gather_features_nobf(ds, target="DOD"):
    excl = {target.lower(),'lat','lon','latitude','longitude',
            'pixel','year','ncoords_vector','nyears_vector',
            'burn_fraction','burn_cumsum','aorcsummerhumidity',
            'aorcsummerprecipitation','aorcsummerlongwave',
            'aorcsummershortwave','aorcsummertemperature'}
    ny = ds.sizes['year'];  feats = {}
    for v in ds.data_vars:
        if v.lower() in excl:  continue
        da = ds[v]
        if set(da.dims)=={'year','pixel'}:
            feats[v] = da.values
        elif set(da.dims)=={'pixel'}:
            feats[v] = np.tile(da.values, (ny,1))
    return feats

def flatten_nobf(ds, target="DOD"):
    fd = gather_features_nobf(ds, target)
    names = sorted(fd)
    X = np.column_stack([fd[n].ravel(order='C') for n in names])
    y = ds[target].values.ravel(order='C')
    ok = (~np.isnan(X).any(axis=1)) & np.isfinite(y)
    return X, y, names, ok

# ────────────────────────────────────────────────────────────
#  permutation importance for MLP
# ────────────────────────────────────────────────────────────
def perm_importance_mlp(model, X_val, y_val, n_repeats=3):
    base_pred = model.predict(X_val)
    base_mse  = mean_squared_error(y_val, base_pred)
    nfeat = X_val.shape[1];  imps = np.zeros(nfeat)
    rng = np.random.RandomState(42)
    Xp = X_val.copy()
    for f in range(nfeat):
        scores = []
        for _ in range(n_repeats):
            saved = Xp[:,f].copy()
            Xp[:,f] = rng.permutation(Xp[:,f])
            perm_mse = mean_squared_error(y_val, model.predict(Xp))
            scores.append(perm_mse)
            Xp[:,f] = saved
        imps[f] = np.mean(scores) - base_mse
    return imps

# ────────────────────────────────────────────────────────────
#  main MLP routine  (unburned‑only training)
# ────────────────────────────────────────────────────────────
def mlp_unburned_experiment(X, y, cat2d, ok, feat_names,
                            unburned_max_cat:int=0):
    thr = unburned_max_cat
    cat = cat2d.ravel(order='C')[ok]
    Xv, Yv = X[ok], y[ok]

    # unburned subset
    unb = cat <= thr
    log(f"  training on unburned (cat ≤ {thr}): N={unb.sum()}")

    # ── NEW: make a 70 % / 30 % split **inside every training category** ──
    train_idx, test_idx = [], []
    for c in range(thr + 1):                        # cats 0 … thr
        rows = np.where((cat == c) & (cat <= thr))[0]
        if rows.size == 0:
            continue
        tr, te = train_test_split(rows,
                                test_size=0.30,
                                random_state=42)
        train_idx.append(tr)
        test_idx .append(te)

    train_idx = np.concatenate(train_idx)
    test_idx  = np.concatenate(test_idx)

    X_tr_raw, y_tr = Xv[train_idx], Yv[train_idx]
    X_te_raw, y_te = Xv[test_idx ], Yv[test_idx ]


    # scale (helps MLP converge)
    xsc = StandardScaler()
    X_tr = xsc.fit_transform(X_tr_raw)
    X_te = xsc.transform(X_te_raw)
    X_all= xsc.transform(Xv)

    # model
    mlp = MLPRegressor(hidden_layer_sizes=(64,64),
                       activation='relu',
                       solver='adam',
                       random_state=42,
                       max_iter=1000)
    mlp.fit(X_tr, y_tr)

    # A) unburned TRAIN / TEST plots
    yhat_tr = mlp.predict(X_tr)
    plot_scatter(y_tr, yhat_tr, f"MLP TRAIN (cat ≤ {thr})")
    plot_bias_hist(y_tr, yhat_tr, f"Bias Hist: TRAIN (cat ≤ {thr})")

    yhat_te = mlp.predict(X_te)
    plot_scatter(y_te, yhat_te, f"MLP TEST (cat ≤ {thr})")
    plot_bias_hist(y_te, yhat_te, f"Bias Hist: TEST (cat ≤ {thr})")

    # B) all‑data evaluation
    yhat_all = mlp.predict(X_all)
    plot_scatter_by_cat(Yv, yhat_all, cat,
                        f"All data – colour by cat (thr={thr})")
    plot_bias_hist(Yv, yhat_all, f"Bias Hist: ALL data (thr={thr})")

    # C) per‑category plots  + Wilcoxon tests
    bias = yhat_all - Yv
    bias_by_cat = {c:bias[cat==c] for c in range(4) if (cat==c).any()}
    for c in range(4):
        m = cat==c
        if m.any():
            plot_scatter(Yv[m], yhat_all[m], f"Category {c} (thr={thr})")
            plot_bias_hist(Yv[m], yhat_all[m],
                           f"Bias Hist: cat={c} (thr={thr})")
    if 0 in bias_by_cat:
        print("\nWilcoxon rank‑sum (bias difference vs cat 0)")
        for c in (1,2,3):
            if c in bias_by_cat:
                s,p = ranksums(bias_by_cat[0], bias_by_cat[c])
                print(f"  cat {c} vs 0 → stat={s:.3f}, p={p:.3g}")

    # D) permutation‑based feature importance
    imp = perm_importance_mlp(mlp, X_all, Yv, n_repeats=3)
    plot_top10_perm_importance(imp, feat_names,
                               f"Top‑10 Permutation Importance (thr={thr})")
    plot_top5_feature_scatter(imp, Xv, Yv, cat, feat_names,
                           f"Top‑5 (thr={thr})")

    return mlp

# ────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log("loading final_dataset4.nc …")
    ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/final_dataset4.nc")

    # cumulative‑burn categories
    bc = ds["burn_cumsum"].values
    cat_2d = np.zeros_like(bc, dtype=int)
    cat_2d[bc < 0.25]                      = 0
    cat_2d[(bc >= 0.25) & (bc < 0.50)]     = 1
    cat_2d[(bc >= 0.50) & (bc < 0.75)]     = 2
    cat_2d[bc >= 0.75]                     = 3
    log("categories (c0‑c3) computed")

    # feature matrix
    X_all, y_all, feat_names, ok = flatten_nobf(ds, "DOD")
    log(f"feature matrix ready – {ok.sum()} valid samples, "
        f"{len(feat_names)} predictors")

    # ── RUN #1  (unburned = cat 0) ────────────────────────────────
    log("\n=== MLP RUN #1 : unburned = cat 0 (cumsum < 0.25) ===")
    mlp_run1 = mlp_unburned_experiment(
        X_all, y_all, cat_2d, ok, feat_names,
        unburned_max_cat=0)

    # ── RUN #2  (unburned = cat 0 + 1) ────────────────────────────
    log("\n=== MLP RUN #2 : unburned = cat 0 + 1 (cumsum < 0.50) ===")
    mlp_run2 = mlp_unburned_experiment(
        X_all, y_all, cat_2d, ok, feat_names,
        unburned_max_cat=1)

    log("ALL DONE.")
