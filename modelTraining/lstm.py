import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xarray as xr

data_path = '/Users/yashnilmohanty/Desktop/processed_combined_data.nc'
ds = xr.open_dataset(data_path)

# Identify valid pixels where DOD is not NaN across all years
valid_coords_mask = ~np.isnan(ds['DOD']).any(dim='nyears')  # True for valid pixels
valid_coords = np.where(valid_coords_mask.values)[0]

predictors = [
    'Elevation', 'slope', 'VegTyp', 'aspect_ratio',
    'aorcFallTemperature', 'aorcFallRain', 'aorcFallHumidity', 'aorcFallLongwave', 'aorcFallShortwave',
    'aorcSpringTemperature', 'aorcSpringRain', 'aorcSpringHumidity', 'aorcSpringLongwave', 'aorcSpringShortwave',
    'aorcSummerTemperature', 'aorcSummerRain', 'aorcSummerHumidity', 'aorcSummerLongwave', 'aorcSummerShortwave',
    'aorcWinterTemperature', 'aorcWinterRain', 'aorcWinterHumidity', 'aorcWinterLongwave', 'aorcWinterShortwave',
    'sweSpring', 'sweWinter'
]
X_combined = np.stack([ds[var].values for var in predictors], axis=-1)  # Shape: (nyears, ncoords, features)
y = ds['DOD'].values  # Shape: (nyears, ncoords)

# Filter data for valid pixels
X_combined = X_combined[:, valid_coords, :]  # Shape: (nyears, valid_coords, features)
y = y[:, valid_coords]  # Shape: (nyears, valid_coords)

# Normalize features
scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined.reshape(-1, X_combined.shape[-1])).reshape(X_combined.shape)

# Split into training and testing sets by space
n_valid_coords = len(valid_coords)
train_coords, test_coords = train_test_split(np.arange(n_valid_coords), test_size=0.2, random_state=42)

X_train_space = X_combined[:, train_coords, :]  # Shape: (nyears, train_coords, features)
X_test_space = X_combined[:, test_coords, :]  # Shape: (nyears, test_coords, features)
y_train_space = y[:, train_coords]  # Shape: (nyears, train_coords)
y_test_space = y[:, test_coords]  # Shape: (nyears, test_coords)


# Reshape for LSTM input
n_features = X_train_space.shape[-1]
X_train_space_lstm = X_train_space.reshape(-1, 1, n_features)  # Flatten years and reshape for LSTM
X_test_space_lstm = X_test_space.reshape(-1, 1, n_features)
y_train_space_flat = y_train_space.flatten()  # Flatten years and coords
y_test_space_flat = y_test_space.flatten()

# Check for NaN values in X_combined before splitting
print("NaN count in X_combined:", np.isnan(X_combined).sum())

# Check for NaN values in each predictor variable
for i, var in enumerate(predictors):
    print(f"{var}: NaN count = {np.isnan(ds[var].values).sum()}")


'''
# Check for NaN or Inf in X_train and y_train
print("Checking X_train_space_lstm...")
print("NaN count:", np.isnan(X_train_space_lstm).sum(), "Inf count:", np.isinf(X_train_space_lstm).sum())

print("Checking y_train_space_flat...")
print("NaN count:", np.isnan(y_train_space_flat).sum(), "Inf count:", np.isinf(y_train_space_flat).sum())
'''

# Define LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(1, n_features)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_space_lstm, y_train_space_flat, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
results = model.evaluate(X_test_space_lstm, y_test_space_flat)
print(f"Test Loss: {results[0]}, Test MAE: {results[1]}")
# Test Loss: 344.3625793457031, Test MAE: 12.436144828796387