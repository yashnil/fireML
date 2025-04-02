import xarray as xr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load the processed NetCDF file
file_path = '/Users/yashnilmohanty/Desktop/preprocessed_combined_data.nc'
data = xr.open_dataset(file_path)

# List of predictor variables
predictors = [
    'aorcFallTemperature', 'aorcFallRain', 'aorcFallHumidity', 'aorcFallLongwave', 'aorcFallShortwave',
    'aorcSpringTemperature', 'aorcSpringRain', 'aorcSpringHumidity', 'aorcSpringLongwave', 'aorcSpringShortwave',
    'aorcSummerTemperature', 'aorcSummerRain', 'aorcSummerHumidity', 'aorcSummerLongwave', 'aorcSummerShortwave',
    'aorcWinterTemperature', 'aorcWinterRain', 'aorcWinterHumidity', 'aorcWinterLongwave', 'aorcWinterShortwave',
    'sweSpring', 'sweWinter', 'Elevation', 'slope', 'VegTyp', 'aspect_ratio'
]

# Target variable
target = 'DOD'

# Flatten the data for ML (convert xarray to pandas DataFrame)
def flatten_data(data, predictors, target):
    """Flatten the data and return predictors and target as DataFrames."""
    df = data.to_dataframe().reset_index()
    predictors_df = df[predictors]
    target_df = df[target]
    return predictors_df, target_df

# Flatten the dataset
predictors_df, target_df = flatten_data(data, predictors, target)

# Spatial Train-Test Split
spatial_coords = np.arange(data.dims['npixels'])  # Spatial points
train_coords, test_coords = train_test_split(spatial_coords, test_size=0.1, random_state=42)

# Subset the data based on train-test split
train_X = predictors_df.iloc[train_coords]
train_y = target_df.iloc[train_coords]
test_X = predictors_df.iloc[test_coords]
test_y = target_df.iloc[test_coords]

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_X, train_y)

# Predict on the test set
predictions = model.predict(test_X)

# Evaluate the model
mse = mean_squared_error(test_y, predictions)
r2 = r2_score(test_y, predictions)

print(f"Mean Squared Error: {mse}") # Mean Squared Error: 142.01432098307444
print(f"R^2 Score: {r2}") # R^2 Score: 0.9080116334103661

# Extract feature importances from the trained Random Forest model
importances = model.feature_importances_
feature_names = train_X.columns

# Sort feature importances in descending order
sorted_indices = np.argsort(importances)[::-1]
sorted_feature_names = feature_names[sorted_indices]
sorted_importances = importances[sorted_indices]

# Print the feature importances
print("Feature Importances:")
for feature, importance in zip(sorted_feature_names, sorted_importances):
    print(f"{feature}: {importance:.4f}")

# Plot the feature importances
plt.figure(figsize=(12, 8))
plt.barh(sorted_feature_names, sorted_importances, color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importances from Random Forest Model")
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.tight_layout()
plt.show()