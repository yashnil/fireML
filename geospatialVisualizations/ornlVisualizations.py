import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Path to your .mat file
file_path = '/Users/yashnilmohanty/Desktop/data/ORNL_DOD_north_CA_by_CY/ORNL_DOD_nCA_CY2001.mat'

# Load the .mat file
data = sio.loadmat(file_path)

# Extract the specific variable
variable = data['DOD_nCA']
# predict: DOD (date of snow disappearance)

# Separate the data into latitude, longitude, and the third value
latitude = variable[:, 0]
longitude = variable[:, 1]
value = variable[:, 2]

# Create a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(longitude, latitude, c=value, cmap='viridis', s=1)
plt.colorbar(label='Value')  # Adjust the label to match the variable being visualized
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Visualization of DOD_nCA Data')
plt.show()
