# test_model.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ranksums

def compute_metrics(y_true, y_pred):
    """
    Returns dict with bias, rmse, corr, p_corr
    """
    out = {}
    if len(y_true) == 0:
        out['bias'] = np.nan
        out['rmse'] = np.nan
        out['corr'] = np.nan
        out['p_corr'] = np.nan
        return out

    errors = y_pred - y_true
    out['bias'] = np.mean(errors)
    out['rmse'] = np.sqrt(np.mean(errors**2))

    if len(y_true) > 1:
        r, p = pearsonr(y_true, y_pred)
        out['corr'] = r
        out['p_corr'] = p
    else:
        out['corr'] = np.nan
        out['p_corr'] = np.nan

    return out

def predict_unburned_test(rf_model, burn_array, dod_array, test_indices):
    """
    Predict on unburned test set, with the same "year t => DoD(t+1)" logic.
    For t in 0..13, X = burn_array[t, pix], Y_true = dod_array[t+1, pix].
    """
    y_true_list = []
    y_pred_list = []

    n_years, _ = burn_array.shape
    for t in range(n_years - 1):
        print(f"Current year = {t}")
        t_next = t + 1
        for pix in test_indices:
            X_test = np.array([[burn_array[t, pix]]])  # single feature
            y_pred = rf_model.predict(X_test)[0]

            y_true = dod_array[t_next, pix]
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)

    return np.array(y_true_list), np.array(y_pred_list)

def predict_burned_postfire(rf_model, burn_array, dod_array, burned_mask, fire_threshold=0.4):
    """
    For each burned pixel, find year t where burn_array[t, pix] > fire_threshold,
    and then predict DoD at year (t+1).
    Returns y_true_burned, y_pred_burned.
    """
    burned_indices = np.where(burned_mask)[0]
    y_true_list = []
    y_pred_list = []

    n_years, _ = burn_array.shape
    total_burned = len(burned_indices)

    for i, pix in enumerate(burned_indices):
        # Print progress every 100 pixels (adjust as needed)
        if i % 500 == 0:
            print(f"[predict_burned_postfire] Processed {i+1} of {total_burned} burned pixels...")

        b_series = burn_array[:, pix]  # shape (15,) or (n_years,)
        fire_years = np.where(b_series > fire_threshold)[0]

        for t in fire_years:
            t_next = t + 1
            if t_next < n_years:
                # burn fraction in year t
                X_test = np.array([[b_series[t]]])
                y_pred = rf_model.predict(X_test)[0]

                # DoD in year t+1
                y_true = dod_array[t_next, pix]

                y_true_list.append(y_true)
                y_pred_list.append(y_pred)

    return np.array(y_true_list), np.array(y_pred_list)

def wilcoxon_rank_sum(errors_burned, errors_unburned):
    stat, p_val = ranksums(errors_burned, errors_unburned)
    return stat, p_val

def plot_results(y_true_burned, y_pred_burned, y_true_unburn, y_pred_unburn):
    """
    Boxplot + scatter for burned vs unburned.
    """
    errors_burned = y_pred_burned - y_true_burned
    errors_unburn = y_pred_unburn - y_true_unburn

    # Boxplot
    plt.figure()
    plt.boxplot([errors_burned, errors_unburn], labels=['Burned(N=1)', 'Unburned'])
    plt.ylabel("Error (pred - true)")
    plt.title("DoD Error Distributions (N=1 postfire vs unburned)")
    plt.show()

    # Scatter burned
    plt.figure()
    plt.scatter(y_true_burned, y_pred_burned, alpha=0.3)
    mn_b = min(y_true_burned.min(), y_pred_burned.min())
    mx_b = max(y_true_burned.max(), y_pred_burned.max())
    plt.plot([mn_b, mx_b], [mn_b, mx_b], 'r--')
    plt.xlabel("True DoD (burned N=1)")
    plt.ylabel("Predicted DoD")
    plt.title("Burned Data: Observed vs. Predicted")
    plt.show()

    # Scatter unburned
    plt.figure()
    subset_size = min(5000, len(y_true_unburn))
    idx_sample = np.random.choice(len(y_true_unburn), size=subset_size, replace=False)
    plt.scatter(y_true_unburn[idx_sample], y_pred_unburn[idx_sample], alpha=0.3)
    mn_u = min(y_true_unburn.min(), y_pred_unburn.min())
    mx_u = max(y_true_unburn.max(), y_pred_unburn.max())
    plt.plot([mn_u, mx_u], [mn_u, mx_u], 'r--')
    plt.xlabel("True DoD (unburned)")
    plt.ylabel("Predicted DoD")
    plt.title("Unburned Data: Observed vs. Predicted")
    plt.show()
