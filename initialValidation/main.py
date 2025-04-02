# main.py

import numpy as np
from data_processing import (
    load_data,
    remove_pixels_if_any_invalid_dod,
    classify_burned_unburned,
    split_unburned_pixels
)
from train_model import (
    build_training_data_for_nextyear,
    train_random_forest_dod
)
from test_model import (
    compute_metrics,
    predict_unburned_test,
    predict_burned_postfire,
    wilcoxon_rank_sum,
    plot_results
)

def main():
    burn_file = "/Users/yashnilmohanty/Desktop/fire_modified.nc"
    dod_file  = "/Users/yashnilmohanty/Desktop/finalDF.nc"

    # 1) Load data
    burn_array, dod_array = load_data(burn_file, dod_file,
                                      burn_varname='burn_fraction',
                                      dod_varname='DOD')
    print("Initial shapes => burn:", burn_array.shape, " DoD:", dod_array.shape)

    # 2) Remove entire pixels with any invalid DoD
    burn_array, dod_array, valid_pixels = remove_pixels_if_any_invalid_dod(burn_array, dod_array)
    print("After removing invalid DoD => shapes:", burn_array.shape, dod_array.shape)
    print("Number of valid pixels:", len(valid_pixels))

    # 3) Classify burned vs. unburned by sum of 15-year burn fraction
    unburned_mask, burned_mask = classify_burned_unburned(burn_array, threshold=0.7)
    print("Unburned pixels:", np.sum(unburned_mask))
    print("Burned pixels:", np.sum(burned_mask))

    # 4) Split unburned into train vs test
    train_inds, test_inds = split_unburned_pixels(unburned_mask, train_ratio=0.8, seed=42)
    print("Train set size:", len(train_inds), "Test set size:", len(test_inds))

    # 5) Build training arrays => X=burn[t], y=DoD[t+1], for unburned train
    X_train, y_train = build_training_data_for_nextyear(burn_array, dod_array, train_inds)
    print("Training samples:", len(y_train))

    # 6) Train RF
    rf_model = train_random_forest_dod(
        X_train, y_train,
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    print("Random forest training complete.")

    # 7) Predict on burned (N=1 postfire)
    y_true_burned, y_pred_burned = predict_burned_postfire(
        rf_model, burn_array, dod_array, burned_mask, fire_threshold=0.4
    )
    burned_metrics = compute_metrics(y_true_burned, y_pred_burned)
    print("Burned N=1 metrics:", burned_metrics)

    print("y_true_burned min,max:", y_true_burned.min(), y_true_burned.max())
    print("y_pred_burned min,max:", y_pred_burned.min(), y_pred_burned.max())


    # 8) Predict on unburned test
    y_true_unburn, y_pred_unburn = predict_unburned_test(
        rf_model, burn_array, dod_array, test_inds
    )
    unburned_metrics = compute_metrics(y_true_unburn, y_pred_unburn)
    print("Unburned test metrics:", unburned_metrics)

    # 9) Wilcoxon rank-sum
    errors_burned = y_pred_burned - y_true_burned
    errors_unburn = y_pred_unburn - y_true_unburn
    stat, p_val = wilcoxon_rank_sum(errors_burned, errors_unburn)
    print(f"Wilcoxon => stat={stat:.3f}, p={p_val:.3e}")

    # 10) Plot
    plot_results(y_true_burned, y_pred_burned, y_true_unburn, y_pred_unburn)

if __name__ == "__main__":
    main()
