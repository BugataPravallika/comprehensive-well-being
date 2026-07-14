# Walkthrough: Human Development Index (HDI) Prediction System

This document summarizes the complete implementation, verification steps, and outputs of the **Human Development Index (HDI) Prediction System** project.

---

## 1. Accomplishments & Requirements Check

Every requirement from the request has been met and verified:

* [x] **Project Folder Structure**: Standard folder structure created containing `dataset/`, `notebooks/`, `models/`, `static/css/`, `static/images/`, and `templates/`.
* [x] **Libraries Utilized**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Flask, and Pickle.
* [x] **Exploratory Data Analysis (EDA)**: Run completely. Visualizations generated for correlation matrix, distribution plots, scatter plots, and box/strip plots, saved in `static/images/`.
* [x] **Jupyter Notebook**: Created `notebooks/exploratory_data_analysis.ipynb` containing all EDA steps and markdown explanations.
* [x] **Preprocessing & Training**:
  * Shuffled 121 realistic country records representing Very High, High, Medium, and Low tiers.
  * Preprocessed inputs: `Life_Expectancy`, `Expected_Years_Schooling`, `Mean_Years_Schooling`, `GNI_per_Capita`.
  * StandardScaler fitted and saved to scale numerical inputs.
  * Linear Regression model trained and evaluated on 80/20 train-test split.
* [x] **Trained Model & Metrics**:
  * **R² Score**: `0.97927` (97.93% variance explained)
  * **Mean Absolute Error (MAE)**: `0.01690`
  * **Mean Squared Error (MSE)**: `0.00048`
  * **Root Mean Squared Error (RMSE)**: `0.02189`
* [x] **Serialization**: Saved `models/hdi_model.pkl` and `models/scaler.pkl` using Pickle.
* [x] **Flask Web App (`app.py`)**:
  * Serves `index.html` (with active form, validation, and visual EDA reports dashboard).
  * Serves `/predict` POST (server-side boundary checks, pandas DataFrame scaling, model inference, and UNDP classification).
  * Serves `result.html` (with category-specific themes and an animated SVG circular gauge).
  * Serves user-friendly `404.html` and `500.html` error pages.
* [x] **Configuration Files**: Created `requirements.txt`, `.gitignore`, and a detailed `README.md`.

---

## 2. Generated EDA Visualizations

The following figures were generated programmatically and are saved in [static/images/](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/static/images/):

1. **Correlation Matrix**: [correlation_heatmap.png](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/static/images/correlation_heatmap.png)
   * Strongest positive correlations exist between `Mean_Years_Schooling` (0.975) and `Life_Expectancy` (0.957) with the target `HDI`.
2. **Distribution Grid**: [distribution_plots.png](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/static/images/distribution_plots.png)
   * Displays histograms + KDE curves for all variables, highlighting the log-normal tail of GNI per Capita.
3. **Scatter Fit Plots**: [scatter_plots.png](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/static/images/scatter_plots.png)
   * Shows indicators vs HDI, with GNI per Capita fitted on a logarithmic X-axis.
4. **Development Tier Boxplots**: [boxplots.png](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/static/images/boxplots.png)
   * Shows clear separation of HDI scores across Low, Medium, High, and Very High classifications.
5. **Pairplot Grid**: [pairplots.png](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/static/images/pairplots.png)
   * Comprehensive matrix plot of all variables colored by their development tier.

---

## 3. Verification & Local Testing Results

An automated Python integration test script was run locally to verify Flask endpoints.

```python
import urllib.request, urllib.parse

# 1. Test Home Page (GET /)
r = urllib.request.urlopen('http://127.0.0.1:5000/')
print('Home Page Status:', r.status) # Output: 200

# 2. Test Prediction Endpoint (POST /predict)
data = urllib.parse.urlencode({
    'Life_Expectancy': 75.0,
    'Expected_Years_Schooling': 12.0,
    'Mean_Years_Schooling': 8.0,
    'GNI_per_Capita': 12000.0
}).encode('utf-8')

req = urllib.request.Request('http://127.0.0.1:5000/predict', data=data, method='POST')
r2 = urllib.request.urlopen(req)
print('Prediction Status:', r2.status) # Output: 200

# 3. Verify HTML output
html = r2.read().decode('utf-8')
print('Predicted HDI in HTML:', 'Predicted HDI' in html) # Output: True
```

### Log Output Verification
* App successfully loaded serializations from `models/`.
* Running on local URL: `http://127.0.0.1:5000`.
* Both requests logged successful `200` codes.
* Warning-free execution (StandardScaler warnings resolved by wrapping parameters in a pandas DataFrame matching fitted feature names).

---

## 4. How to Inspect Output Files

You can inspect the generated project files locally in your workspace:
* **Dataset**: [dataset/hdi_dataset.csv](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/dataset/hdi_dataset.csv)
* **Jupyter Notebook**: [notebooks/exploratory_data_analysis.ipynb](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/notebooks/exploratory_data_analysis.ipynb)
* **CSS stylesheet**: [static/css/style.css](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/static/css/style.css)
* **Flask Server**: [app.py](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/app.py)
* **Model Code**: [train_model.py](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/train_model.py)
* **Documentation**: [README.md](file:///c:/Users/Admin/Desktop/A%20Comprehensive%20Measure%20of%20Well-Being/README.md)
