import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent

# Set plot style for premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 150
})

def generate_hdi_dataset(save_path=None):
    """
    Generates a realistic, statistically coherent dataset of countries with socioeconomic
    indicators and calculates their Human Development Index (HDI) based on the official UNDP formula.
    """
    if save_path is None:
        save_path = BASE_DIR / "dataset" / "hdi_dataset.csv"
    else:
        save_path = Path(save_path)

    np.random.seed(42)
    
    # Define countries across four development tiers to simulate authentic global diversity
    country_profiles = {
        "Very High": [
            "Norway", "Switzerland", "Ireland", "Germany", "Sweden", "Singapore", "Australia", 
            "Denmark", "Netherlands", "Finland", "USA", "UK", "Canada", "Japan", "South Korea", 
            "Israel", "Slovenia", "Spain", "France", "Italy", "New Zealand", "Austria", "Belgium", 
            "UAE", "Saudi Arabia", "Estonia", "Poland", "Greece", "Chile", "Croatia", "Portugal", "Latvia"
        ],
        "High": [
            "Seychelles", "Bulgaria", "Romania", "Malaysia", "Costa Rica", "Mauritius", "Georgia", 
            "Sri Lanka", "Albania", "Cuba", "Ukraine", "Mexico", "Brazil", "Colombia", "Peru", 
            "Ecuador", "Thailand", "Armenia", "Algeria", "Tunisia", "Turkey", "Uruguay", "Panama", 
            "Serbia", "Iran", "China", "South Africa", "Dominican Republic", "Azerbaijan", "Belarus", "Jordan"
        ],
        "Medium": [
            "Philippines", "Egypt", "Vietnam", "Gabon", "Indonesia", "India", "Iraq", "Morocco", 
            "Kyrgyzstan", "Tajikistan", "Bangladesh", "Nepal", "Bhutan", "Congo", "Guatemala", 
            "Honduras", "Vanuatu", "Nicaragua", "Bolivia", "El Salvador", "Cape Verde", "Timor-Leste", 
            "Ghana", "India", "Micronesia", "Myanmar", "Kenya", "Pakistan", "Angola", "Equatorial Guinea"
        ],
        "Low": [
            "Syria", "Madagascar", "Tanzania", "Yemen", "Senegal", "Afghanistan", "Sudan", 
            "Malawi", "Ethiopia", "Gambia", "Guinea", "Mozambique", "Mali", "Burundi", 
            "Central African Republic", "Niger", "Chad", "South Sudan", "Somalia", "Eritrea", 
            "Sierra Leone", "Burkina Faso", "Liberia", "Rwanda", "Uganda", "Benin", "Togo", "Lesotho"
        ]
    }
    
    records = []
    
    for tier, countries in country_profiles.items():
        for country in countries:
            # Set statistical distributions based on developmental tier
            if tier == "Very High":
                le = np.random.normal(81.5, 2.2)      # Life Expectancy (years)
                eys = np.random.normal(16.5, 1.2)     # Expected Years of Schooling
                mys = np.random.normal(12.2, 0.8)     # Mean Years of Schooling
                gni = np.exp(np.random.normal(np.log(45000), 0.3)) # GNI per capita (PPP)
            elif tier == "High":
                le = np.random.normal(75.2, 2.8)
                eys = np.random.normal(13.8, 1.2)
                mys = np.random.normal(9.6, 1.0)
                gni = np.exp(np.random.normal(np.log(15500), 0.35))
            elif tier == "Medium":
                le = np.random.normal(69.8, 3.5)
                eys = np.random.normal(11.8, 1.5)
                mys = np.random.normal(6.8, 1.2)
                gni = np.exp(np.random.normal(np.log(5800), 0.4))
            else:  # Low development
                le = np.random.normal(61.2, 4.5)
                eys = np.random.normal(9.2, 1.8)
                mys = np.random.normal(4.2, 1.5)
                gni = np.exp(np.random.normal(np.log(1800), 0.45))
                
            # Clamp value ranges to remain biologically and logically valid
            le = np.clip(le, 45.0, 86.0)
            eys = np.clip(eys, 4.0, 21.0)
            mys = np.clip(mys, 1.0, 15.0)
            gni = np.clip(gni, 400.0, 110000.0)
            
            # Calculate indices based on official UNDP formulas
            lei = (le - 20) / (85 - 20)
            eysi = eys / 18
            mysi = mys / 15
            ei = (eysi + mysi) / 2
            
            # GNI Index (logarithmic transformation)
            ii = (np.log(gni) - np.log(100)) / (np.log(75000) - np.log(100))
            
            # Geometric mean of the three dimensions
            hdi_calc = (lei * ei * ii) ** (1/3)
            
            # Add a slight amount of Gaussian noise to simulate real-world data collection issues
            noise = np.random.normal(0, 0.008)
            hdi = np.clip(hdi_calc + noise, 0.25, 0.99)
            
            records.append({
                "Country": country,
                "Development_Tier": tier,
                "Life_Expectancy": round(le, 2),
                "Expected_Years_Schooling": round(eys, 2),
                "Mean_Years_Schooling": round(mys, 2),
                "GNI_per_Capita": round(gni, 2),
                "HDI": round(hdi, 3)
            })
            
    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create dataset directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[+] Dataset successfully generated and saved to: {save_path}")
    return df

def run_eda(df, output_dir=None):
    """
    Performs complete Exploratory Data Analysis (EDA) on the dataset and
    saves high-quality plots to the specified output directory.
    """
    if output_dir is None:
        output_dir = BASE_DIR / "static" / "images"
    else:
        output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Exploratory Data Analysis Summary ---")
    print(f"Dataset Shape: {df.shape}")
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nStatistical Summary:")
    print(df.describe())
    
    # 1. Correlation Matrix Heatmap
    plt.figure(figsize=(8, 6))
    corr_cols = ["Life_Expectancy", "Expected_Years_Schooling", "Mean_Years_Schooling", "GNI_per_Capita", "HDI"]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation Matrix of Socioeconomic Indicators & HDI")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=150)
    plt.close()
    
    # 2. Distribution Plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    features = ["Life_Expectancy", "Expected_Years_Schooling", "Mean_Years_Schooling", "GNI_per_Capita", "HDI"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for i, (col, color) in enumerate(zip(features, colors)):
        sns.histplot(df[col], kde=True, ax=axes[i], color=color, edgecolor="black", alpha=0.7)
        axes[i].set_title(f"Distribution of {col.replace('_', ' ')}")
        axes[i].set_xlabel("")
        
    # Hide the 6th empty subplot
    axes[5].set_visible(False)
    plt.suptitle("Feature Distributions (Histograms & KDEs)", fontsize=20, y=0.98)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distribution_plots.png", dpi=150)
    plt.close()
    
    # 3. Scatter Plots vs HDI
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    scatter_features = ["Life_Expectancy", "Expected_Years_Schooling", "Mean_Years_Schooling", "GNI_per_Capita"]
    
    for i, col in enumerate(scatter_features):
        if col == "GNI_per_Capita":
            # Apply log-scale on X for GNI to visualize its logarithmic nature vs HDI
            sns.regplot(data=df, x=col, y="HDI", ax=axes[i], scatter_kws={'alpha':0.6, 'color':'#2b5c8f'}, line_kws={'color':'#d9534f', 'linewidth': 2})
            axes[i].set_xscale('log')
            axes[i].set_xlabel(f"GNI per Capita (Log Scale, PPP $)")
        else:
            sns.regplot(data=df, x=col, y="HDI", ax=axes[i], scatter_kws={'alpha':0.6, 'color':'#2b5c8f'}, line_kws={'color':'#d9534f', 'linewidth': 2})
            axes[i].set_xlabel(col.replace('_', ' '))
            
        axes[i].set_title(f"{col.replace('_', ' ')} vs HDI")
        axes[i].set_ylabel("Human Development Index (HDI)")
        
    plt.suptitle("Socioeconomic Indicators vs. Human Development Index (HDI)", fontsize=20, y=0.98)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_plots.png", dpi=150)
    plt.close()
    
    # 4. Boxplots & Strip plots (HDI across Development Tiers)
    plt.figure(figsize=(10, 6))
    tier_order = ["Low", "Medium", "High", "Very High"]
    sns.boxplot(data=df, x="Development_Tier", y="HDI", order=tier_order, palette="Set2", width=0.5, showfliers=False)
    sns.stripplot(data=df, x="Development_Tier", y="HDI", order=tier_order, color="black", alpha=0.3, jitter=0.2, size=5)
    plt.title("Distribution of HDI across Development Tiers")
    plt.xlabel("Development Tier")
    plt.ylabel("Human Development Index (HDI)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplots.png", dpi=150)
    plt.close()

    # 5. Pairplots
    plt.figure(figsize=(12, 10))
    pairplot_cols = ["Life_Expectancy", "Expected_Years_Schooling", "Mean_Years_Schooling", "GNI_per_Capita", "HDI", "Development_Tier"]
    g = sns.pairplot(df[pairplot_cols], hue="Development_Tier", hue_order=tier_order, palette="magma", diag_kind="kde", height=2.5)
    g.fig.suptitle("Pairwise Relationships and Distributions by Development Tier", y=1.02, fontsize=18)
    g.savefig(f"{output_dir}/pairplots.png", dpi=150)
    plt.close()
    
    print(f"[+] All EDA plots saved successfully to: {output_dir}")

def preprocess_and_train(df, model_dir=None):
    """
    Processes the data, trains a Linear Regression model, prints metrics,
    and serializes the trained model and scaler to the models directory.
    """
    if model_dir is None:
        model_dir = BASE_DIR / "models"
    else:
        model_dir = Path(model_dir)

    os.makedirs(model_dir, exist_ok=True)
    
    # Define features and target
    X = df[["Life_Expectancy", "Expected_Years_Schooling", "Mean_Years_Schooling", "GNI_per_Capita"]]
    y = df["HDI"]
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Data Preprocessing & Training ---")
    print(f"Training Set Size: {X_train.shape[0]} samples")
    print(f"Testing Set Size: {X_test.shape[0]} samples")
    
    # Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n--- Model Evaluation Metrics ---")
    print(f"R² Score (Coefficient of Determination): {r2:.5f}")
    print(f"Mean Absolute Error (MAE):               {mae:.5f}")
    print(f"Mean Squared Error (MSE):                {mse:.5f}")
    print(f"Root Mean Squared Error (RMSE):          {rmse:.5f}")
    
    # Coefficients analysis
    print("\n--- Model Coefficients (Feature Importance) ---")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature:<25} : {coef:.5f}")
    print(f"Intercept                 : {model.intercept_:.5f}")
    
    # Save the model and scaler
    model_path = os.path.join(model_dir, "hdi_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"\n[+] Trained Model successfully saved to: {model_path}")
    print(f"[+] Fitted Scaler successfully saved to: {scaler_path}")
    
    # Return metrics for report
    return r2, mae, mse, rmse

if __name__ == "__main__":
    print("=================================================================")
    print(" HUMAN DEVELOPMENT INDEX (HDI) PREDICTION SYSTEM - TRAINING")
    print("=================================================================")
    
    # Generate the dataset
    df = generate_hdi_dataset()
    
    # Run Exploratory Data Analysis
    run_eda(df)
    
    # Train and serialize the model
    preprocess_and_train(df)
    
    print("\n=================================================================")
    print(" TRAINING PROCESS COMPLETE! READY TO RUN FLASK APPLICATION")
    print("=================================================================")
