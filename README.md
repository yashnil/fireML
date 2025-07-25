# Machine Learning Insights for Snow Disappearance Predictability in Northern California Burned Areas

## 🌲 Overview

This project investigates how wildfire disturbances affect the predictability of snow disappearance dates (DSD) across Northern California using machine learning models. Leveraging satellite-derived DSD data and high-resolution hydrometeorological predictors, we assess whether and how wildfire burn history degrades or alters the statistical relationships used in predictive models.
Our core objective is to evaluate the spatiotemporal skill and limitations of ML models in capturing post-fire snow dynamics—an essential factor for understanding water availability and forecasting hydrologic resources in fire-prone mountain regions.


## 📘 Abstract

Wildfires are reshaping snow dynamics across the Western United States, disrupting water resources essential for agriculture, drinking water, and hydropower. In this study, we use Random Forests, Multilayer Perceptrons (MLP), and Long Short-Term Memory (LSTM) networks to predict the **Day of Snow Disappearance (DSD)** using geographic and hydroclimatic features. We stratify the analysis by cumulative burn area and assess the impact of both implicit (sample inclusion) and explicit (predictor inclusion) representations of fire history.

**Key Findings:**

* Models trained only on unburned pixels underperform significantly in burned zones.
* Including burned pixels in training improves model generalization, especially in high-severity fire zones.
* Including burn fraction and vegetation dynamics (e.g., FPAR) as predictors offers only marginal improvements.
* Spatial and elevation-based heterogeneity in snow predictability persists post-fire.

## 📁 Repository Structure

```
FIREML/FINAL_SCREEN/
│
├── RFR/                        # Experiment 1 – Random Forest on unburned training
│   └── newversion.py
│
├── LSTM/                       # Experiment 1 – LSTM on unburned training
│   └── newversion.py
│
├── MLP/                        # Experiment 1 – MLP on unburned training
│   └── newversion.py
│
├── excludeBurnFraction/       # Experiment 2 – Stratified training without burn fraction
│   └── newversion.py
│
├── includeBurnFraction/       # Experiment 3 – Stratified training with burn fraction
│   └── newversion.py
│
├── experiment4/               # Experiment 4 – Stratified training with FPAR predictors
│   └── new.py
│
├── IDENTICAL/                 # "Apples to apples" identical-setup test
│   └── main.py
│
├── comparativeHistograms/    # Plots comparing metrics, feature importances
│   ├── main.py
│   ├── featureImportance.py
│   └── permutationImportance.py
│
├── robustness.py             # Per-category training/eval for robustness tests
├── rf_burned_stratified.py   # Train and test only on burned pixels

```

## 🧪 Experiments Overview

| Experiment  | Training Data | Key Purpose |
|:-------------:|:-------------:|:-------------:|
| **1**      | 70% of unburned (c0) pixels     | Benchmark model on unburned regions and test on burned ones     |
| **2**      | 70% of each burn category (c0–c3)     | Evaluate if stratified inclusion improves burned predictions    |
| **3**      | Same as 2 + Burn Fraction predictor     | Test whether explicitly adding fire info improves performance     |
| **4**      | Same as 2 + FPAR predictors     | Evaluate value of dynamic vegetation metrics (e.g., regrowth)     |

## 📊 Evaluation Metrics

Each model is evaluated using:
* **R² (Coefficient of Determination)**
* **Mean Bias**
* **Bias Standard Deviation**
* **Root Mean Squared Error (RMSE)**
* **Spearman Correlation Coefficient (for predictor-target relationships)**

Statistical significance of biases is tested using **Wilcoxon Rank Sum Tests**. Feature importance is assessed using:
* **Mean Decrease in Impurity (MDI)**
* **Permutation Importance**

## 📡 Data Sources

| Dataset  | Description |
|:-------------:|:-------------:|
| **MODIS MOD10A2**       | Snow cover → ORNL Snow Disappearance (DSD) product     |
| **SNODAS**       | Peak Snow Water Equivalent (SWE), winter average, date of peak     |
| **AORC**       | Hourly temperature, precipitation, shortwave radiation, longwave radiation, humidity *(Fall, Winter, Spring)*     |
| **MTBS**       | Annual fire history (burn fraction)     |
| **SRTM**       | Topography: elevation, slope, aspect ratio     |
| **GLCC**       | USGS Land cover classification     |
| **MODIS FPAR**       | Fraction of Absorbed Photosynthetically Active Radiation     |

## 🔍 Key Insights

* **Fire-induced errors:** DSD predictions systematically overestimate snow duration in heavily burned pixels.
* **Implicit fire info works better:** Including burned samples in training improves skill more than adding fire variables explicitly.
* **Spatial heterogeneity:** Prediction biases vary with elevation, land cover, and burn severity.
* **Predictor importance:** Peak SWE, spring temperature, and elevation dominate prediction; burn fraction is low-ranked in importance.

## 💻 Getting Started

### 📦 Dependencies

Install the required Python libraries:
```
pip install -r requirements.txt
```
Required packages include:
* ```scikit-learn```
* ```xgboost```
* ```keras```
* ```tensorflow```
* ```matplotlib```
* ```numpy```
* ```pandas```
* ```scipy```
* ```seaborn```
* ```xarray```
* ```rasterio```
* ```geopandas```
* ```cartopy```
* ```netCDF4```

### ▶️ Running Models
To run any experiment:

```
python FIREML/FINAL_SCREEN/RFR/newversion.py
```

Replace ```RFR``` with ```MLP```, ```LSTM```, ```includeBurnFraction```, etc., depending on your experiment.

### 📈 Generating Evaluation Plots
For comparative plots:
```
python FIREML/FINAL_SCREEN/comparativeHistograms/main.py
python FIREML/FINAL_SCREEN/comparativeHistograms/featureImportance.py
```

## 📑 Paper and Citation
If you use this work, please cite our paper:
> Mohanty, Y., Abolafia-Rosenzweig, R., He, C., & McGrath, D. (2025). *Machine Learning Insights for Snow Disappearance Predictability in Northern California Burned Areas*. Environmental Research Letters. (In Review)

## 🧠 Future Work
* Integrating hybrid ML-physics models
* Improving burn-sensitive predictors (e.g., soil burn severity, surface albedo)
* Extending the approach to other snow-dominated watersheds

## 📬 Contact
Questions or feedback? Reach out to:

- 📧 yashnilmohanty@gmail.com
- 📧 abolafia@ucar.edu
- 🔗 GitHub: yashnil

## 🏷 License

This project is licensed under the MIT License.
