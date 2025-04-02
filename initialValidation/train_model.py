# train_model.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor

def build_training_data_for_nextyear(burn_array, dod_array, train_indices):
    """
    We want (X, y) where:
      - X includes the burn fraction in year t (and optionally other features)
      - y is DoD in year (t+1)
    We'll loop t from 0..13, because t+1 must be < 15.

    train_indices => indices of unburned pixels.

    Returns X_train, y_train as np.array
    """
    X_list = []
    y_list = []
    n_years, _ = burn_array.shape

    for t in range(n_years - 1):
        t_next = t + 1
        for pix in train_indices:
            burn_val_t = burn_array[t, pix]  # burn fraction in year t
            dod_val_tplus1 = dod_array[t_next, pix]  # DoD in year t+1
            # Now, we already know it's valid because of prior filtering
            X_list.append([burn_val_t])    # single-feature example
            y_list.append(dod_val_tplus1)

    X_train = np.array(X_list)  # shape: (N, 1)
    y_train = np.array(y_list)  # shape: (N,)
    return X_train, y_train

def train_random_forest_dod(X_train, y_train, 
                            n_estimators=100, 
                            max_depth=None, 
                            n_jobs=-1, 
                            random_state=42):
    """
    Trains an RF regressor on (X_train, y_train).
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf
