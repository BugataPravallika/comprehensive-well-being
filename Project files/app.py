import os
import pickle
import math
import json
import sqlite3
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from fpdf import FPDF, XPos, YPos

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Premium PDF generation class
class HDIPDF(FPDF):
    def header(self):
        # Header banner styling (deep slate-900 color)
        self.set_fill_color(15, 23, 42)
        self.rect(0, 0, 210, 40, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, "HUMAN DEVELOPMENT INDEX (HDI) REPORT", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("helvetica", "I", 10)
        self.cell(0, 8, "Socioeconomic Predictive Well-Being Assessment", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(15)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | HDI Prediction System", align="C")

# Paths for serialized models and databases
MODEL_PATH = BASE_DIR / "models" / "hdi_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "hdi_dataset.csv"
DB_PATH = BASE_DIR / "dataset" / "predictions.db"

# Global variables
model = None
scaler = None
df_data = None
countries_json = "{}"
dataset_stats = {}

RECOMMENDATIONS = {
    "very-high": [
        {"icon": "fa-shield-halved", "title": "Maintain Healthcare Quality", "desc": "Keep investing in preventative healthcare, geriatric medicine, and universal wellness access to sustain high life expectancy."},
        {"icon": "fa-microscope", "title": "Encourage Research & Innovation", "desc": "Provide public funding and tax incentives for tech R&D, biomedical engineering, and clean green initiatives."},
        {"icon": "fa-leaf", "title": "Promote Environmental Sustainability", "desc": "Implement carbon pricing and circular economy frameworks to offset industrial footprint and combat climate change."},
        {"icon": "fa-scale-balanced", "title": "Reduce Income Inequality", "desc": "Use progressive tax structures and social safety nets to bridge any income gaps and improve the Gini index."}
    ],
    "high": [
        {"icon": "fa-seedling", "title": "Focus on Innovation & High Tech", "desc": "Transition from manufacturing-driven industries to knowledge-based services and high-value innovations."},
        {"icon": "fa-scale-balanced", "title": "Reduce Economic Inequality", "desc": "Ensure that wealth gains are shared equitably across regional divisions and socio-economic demographics."},
        {"icon": "fa-wind", "title": "Promote Sustainable Development", "desc": "Adopt eco-friendly infrastructure and clean energy (wind, solar) to transition away from carbon-heavy production."},
        {"icon": "fa-graduation-cap", "title": "Expand Higher Education Access", "desc": "Increase subsidies and scholarship options for tertiary education and advanced vocational programs."}
    ],
    "medium": [
        {"icon": "fa-graduation-cap", "title": "Improve Access to Higher Education", "desc": "Increase secondary-to-tertiary transition rates. Subsidize university fees and expand technological institutes."},
        {"icon": "fa-heart-pulse", "title": "Strengthen Healthcare Infrastructure", "desc": "Build more rural clinics, train medical staff, and implement national immunization/disease prevention drives."},
        {"icon": "fa-industry", "title": "Promote Diverse Economic Development", "desc": "Support local small-to-medium enterprises (SMEs) and reduce dependency on raw agricultural or commodity exports."},
        {"icon": "fa-arrows-split-up-and-left", "title": "Bridge Regional Gaps", "desc": "Invest in infrastructure (roads, clean water, internet) to connect marginalized rural areas with urban hubs."}
    ],
    "low": [
        {"icon": "fa-hospital", "title": "Invest Heavily in Basic Healthcare", "desc": "Prioritize infant/maternal care, expand access to clean drinking water, and combat infectious diseases."},
        {"icon": "fa-school", "title": "Increase Basic Education Funding", "desc": "Focus on primary and secondary school enrollment. Provide free school meals, textbooks, and train qualified teachers."},
        {"icon": "fa-briefcase", "title": "Enhance Employment Opportunities", "desc": "Create jobs through labor-intensive infrastructure projects, vocational training, and micro-finance lending."},
        {"icon": "fa-hand-holding-dollar", "title": "Increase Per-Capita Income", "desc": "Promote basic financial inclusion, agricultural productivity, and support wage growth for minimum wage earners."}
    ]
}

def load_ml_assets():
    """Loads the serialized model and scaler."""
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("[+] Machine Learning assets loaded successfully.")
        return True
    else:
        print("[!] Warning: Machine Learning assets not found. Run train_model.py first.")
        return False

def load_dataset_data():
    """Loads the country indicators dataset and computes statistics for auto-fill & dashboard."""
    global df_data, countries_json, dataset_stats
    if os.path.exists(DATASET_PATH):
        try:
            df_data = pd.read_csv(DATASET_PATH)
            # Create country profiles dictionary
            country_dict = {}
            for _, row in df_data.iterrows():
                country_dict[row["Country"]] = {
                    "Life_Expectancy": float(row["Life_Expectancy"]),
                    "Expected_Years_Schooling": float(row["Expected_Years_Schooling"]),
                    "Mean_Years_Schooling": float(row["Mean_Years_Schooling"]),
                    "GNI_per_Capita": float(row["GNI_per_Capita"]),
                    "HDI": float(row["HDI"])
                }
            countries_json = json.dumps(country_dict)
            
            # Dataset stats for descriptive statistics tab
            num_cols = ["Life_Expectancy", "Expected_Years_Schooling", "Mean_Years_Schooling", "GNI_per_Capita", "HDI"]
            describe_df = df_data[num_cols].describe().round(2).reset_index()
            describe_df.rename(columns={"index": "Metric"}, inplace=True)

            # Calculate Correlation Matrix
            corr_df = df_data[num_cols].corr().round(4).reset_index()
            corr_df.rename(columns={"index": "Feature"}, inplace=True)
            corr_records = corr_df.to_dict(orient="records")

            # Calculate Missing Values Summary
            missing_series = df_data.isnull().sum()
            missing_records = [{"Column": col, "MissingCount": int(val), "Percentage": float(round(val / len(df_data) * 100, 2))} for col, val in missing_series.items()]

            dataset_stats = {
                "shape": df_data.shape,
                "missing_values": df_data.isnull().sum().to_dict(),
                "missing_records": missing_records,
                "stats_records": describe_df.to_dict(orient="records"),
                "corr_matrix": corr_records,
                "countries_list": sorted(df_data["Country"].tolist())
            }
            print("[+] Dataset indicators loaded successfully.")
        except Exception as e:
            print(f"[!] Error loading dataset stats: {e}")

def init_db():
    """Initializes predictions SQLite database."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                life_expectancy REAL,
                expected_schooling REAL,
                mean_schooling REAL,
                gni REAL,
                predicted_hdi REAL,
                category TEXT
            )
        """)
        conn.commit()
        conn.close()
        print("[+] SQLite prediction database initialized.")
    except Exception as e:
        print(f"[!] SQLite DB initialization error: {e}")

def save_prediction_to_db(life_exp, exp_school, mean_school, gni, hdi, category):
    """Saves a prediction entry to SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (life_expectancy, expected_schooling, mean_schooling, gni, predicted_hdi, category)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (life_exp, exp_school, mean_school, gni, hdi, category))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] Error saving prediction to SQLite: {e}")

def get_feature_importance():
    """Dynamically extracts features coefficients and computes relative importance percentages."""
    global model
    if model is None:
        return []
    features = ["Life_Expectancy", "Expected_Years_Schooling", "Mean_Years_Schooling", "GNI_per_Capita"]
    explanations = {
        "Life_Expectancy": "Reflects the longevity and health dimension. A higher life expectancy indicates better public health, sanitation, and healthcare access.",
        "Expected_Years_Schooling": "Measures early educational access. It captures the educational horizon of the youth and scales with years expected in primary and secondary schools.",
        "Mean_Years_Schooling": "Reflects adult educational attainment. Historically, this feature shows a major weight since a well-educated adult population drives sustainable development.",
        "GNI_per_Capita": "Represents the standard of living. It operates on a logarithmic scale in UNDP calculations because income has diminishing marginal returns to human well-being."
    }
    coefs = model.coef_
    abs_coefs = np.abs(coefs)
    total_abs = np.sum(abs_coefs)
    
    importance = []
    for f, c, ac in zip(features, coefs, abs_coefs):
        pct = (ac / total_abs * 100) if total_abs > 0 else 0
        importance.append({
            "feature": f,
            "feature_pretty": f.replace("_", " "),
            "coefficient": float(c),
            "percentage": float(round(pct, 2)),
            "explanation": explanations.get(f, "")
        })
    # Sort by relative contribution
    return sorted(importance, key=lambda x: x["percentage"], reverse=True)

# Load all assets and databases on startup
load_ml_assets()
load_dataset_data()
init_db()

@app.template_filter('log_10')
def log_10_filter(value):
    """Custom template filter to compute base-10 log for GNI scaling in UI progress bar."""
    try:
        val = float(value)
        if val <= 0:
            return 2.0  # fallback to log10(100)
        return math.log10(val)
    except (ValueError, TypeError):
        return 2.0

def render_index_page(error_msg=None, form_data=None):
    """Clean helper to render index page with appropriate contextual attributes."""
    return render_template("index.html",
                           countries_json=countries_json,
                           dataset_stats=dataset_stats,
                           error_msg=error_msg,
                           form_data=form_data)

@app.route("/", methods=["GET"])
def home():
    """Renders the Home and Dashboard Page."""
    return render_index_page()

@app.route("/predict", methods=["POST"])
def predict():
    """Handles the prediction POST request, performs server-side validation, and outputs results."""
    global model, scaler
    
    # Reload assets if they were not loaded previously
    if model is None or scaler is None:
        if not load_ml_assets():
            return render_index_page(error_msg="Machine Learning model is not trained yet. Please run train_model.py first.")
            
    # Retrieve form data
    try:
        life_exp_raw = request.form.get("Life_Expectancy")
        exp_school_raw = request.form.get("Expected_Years_Schooling")
        mean_school_raw = request.form.get("Mean_Years_Schooling")
        gni_raw = request.form.get("GNI_per_Capita")
        
        # Check for missing fields
        if not all([life_exp_raw, exp_school_raw, mean_school_raw, gni_raw]):
            raise ValueError("All fields are required.")
            
        life_exp = float(life_exp_raw)
        exp_school = float(exp_school_raw)
        mean_school = float(mean_school_raw)
        gni = float(gni_raw)
        
        form_data = {
            "Life_Expectancy": life_exp_raw,
            "Expected_Years_Schooling": exp_school_raw,
            "Mean_Years_Schooling": mean_school_raw,
            "GNI_per_Capita": gni_raw
        }
        
    except (ValueError, TypeError):
        return render_index_page(error_msg="Invalid input values. Please enter valid numeric values.",
                                 form_data=request.form)

    # Server-side validation boundary checks
    if not (45.0 <= life_exp <= 86.0):
        return render_index_page(error_msg="Life Expectancy must be between 45.0 and 86.0 years.", form_data=form_data)
        
    if not (4.0 <= exp_school <= 21.0):
        return render_index_page(error_msg="Expected Years of Schooling must be between 4.0 and 21.0 years.", form_data=form_data)
        
    if not (1.0 <= mean_school <= 15.0):
        return render_index_page(error_msg="Mean Years of Schooling must be between 1.0 and 15.0 years.", form_data=form_data)
        
    if not (400.0 <= gni <= 110000.0):
        return render_index_page(error_msg="GNI per Capita must be between $400.0 and $110,000.0.", form_data=form_data)

    if mean_school > exp_school:
        return render_index_page(error_msg="Mean Years of Schooling cannot exceed Expected Years of Schooling.", form_data=form_data)

    # Process inputs for model prediction using DataFrame to preserve feature names
    input_features = pd.DataFrame([[life_exp, exp_school, mean_school, gni]], 
                                   columns=["Life_Expectancy", "Expected_Years_Schooling", "Mean_Years_Schooling", "GNI_per_Capita"])
    
    # Scale inputs
    input_scaled = scaler.transform(input_features)
    
    # Run prediction
    predicted_hdi = model.predict(input_scaled)[0]
    predicted_hdi = float(np.clip(predicted_hdi, 0.0, 1.0))
    
    # Classify the predicted score into UNDP development tiers
    if predicted_hdi >= 0.800:
        tier_class = "very-high"
        tier_name = "Very High Human Development"
        tier_icon = "fa-solid fa-crown"
    elif predicted_hdi >= 0.700:
        tier_class = "high"
        tier_name = "High Human Development"
        tier_icon = "fa-solid fa-circle-arrow-up"
    elif predicted_hdi >= 0.550:
        tier_class = "medium"
        tier_name = "Medium Human Development"
        tier_icon = "fa-solid fa-scale-unbalanced"
    else:
        tier_class = "low"
        tier_name = "Low Human Development"
        tier_icon = "fa-solid fa-triangle-exclamation"
        
    # Save results to SQLite history
    save_prediction_to_db(life_exp, exp_school, mean_school, gni, predicted_hdi, tier_name)
    
    # Model parameters for client-side What-If Simulator
    scaler_means = list(scaler.mean_)
    scaler_scales = list(scaler.scale_)
    model_coefs = list(model.coef_)
    model_intercept = float(model.intercept_)

    return render_template("result.html",
                           hdi_value=predicted_hdi,
                           tier_class=tier_class,
                           tier_name=tier_name,
                           tier_icon=tier_icon,
                           inputs={
                               "Life_Expectancy": life_exp,
                               "Expected_Years_Schooling": exp_school,
                               "Mean_Years_Schooling": mean_school,
                               "GNI_per_Capita": gni
                           },
                           recommendations=RECOMMENDATIONS.get(tier_class, []),
                           feature_importance=get_feature_importance(),
                           simulator_meta={
                               "means": scaler_means,
                               "scales": scaler_scales,
                               "coefs": model_coefs,
                               "intercept": model_intercept
                           })

@app.route("/predict/report", methods=["GET"])
def download_report():
    """Generates a downloadable PDF summary report of the prediction and policy recommendations."""
    global model, scaler
    if model is None or scaler is None:
        return "Model assets are not loaded.", 500
        
    try:
        le = float(request.args.get("le", 0))
        eys = float(request.args.get("eys", 0))
        mys = float(request.args.get("mys", 0))
        gni = float(request.args.get("gni", 0))
    except (ValueError, TypeError):
        return "Invalid parameters.", 400

    # Prediction logic
    input_features = pd.DataFrame([[le, eys, mys, gni]], 
                                   columns=["Life_Expectancy", "Expected_Years_Schooling", "Mean_Years_Schooling", "GNI_per_Capita"])
    input_scaled = scaler.transform(input_features)
    predicted_hdi = model.predict(input_scaled)[0]
    predicted_hdi = float(np.clip(predicted_hdi, 0.0, 1.0))
    
    # Classification
    if predicted_hdi >= 0.800:
        tier_class = "very-high"
        tier_name = "Very High Human Development"
    elif predicted_hdi >= 0.700:
        tier_class = "high"
        tier_name = "High Human Development"
    elif predicted_hdi >= 0.550:
        tier_class = "medium"
        tier_name = "Medium Human Development"
    else:
        tier_class = "low"
        tier_name = "Low Human Development"
        
    recs = RECOMMENDATIONS.get(tier_class, [])

    # Instantiate PDF
    pdf = HDIPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Section 1: Summary
    pdf.set_text_color(30, 41, 59) # Slate-800
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 10, "1. Assessment Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("helvetica", "", 10)
    intro_txt = ("This official evaluation report details the socioeconomic metrics and predicted Human Development "
                 "Index (HDI) rating. The score is computed using standard StandardScaler normalization and a Multiple "
                 "Linear Regression model trained on global country records.")
    pdf.multi_cell(0, 5, intro_txt)
    pdf.ln(6)
    
    # Results Card
    pdf.set_fill_color(248, 250, 252) # Slate-50 background
    pdf.rect(10, pdf.get_y(), 190, 24, "F")
    
    pdf.set_xy(15, pdf.get_y() + 4)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(90, 6, "Predicted HDI Value:")
    pdf.cell(0, 6, "Development Classification Tier:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_xy(15, pdf.get_y())
    pdf.set_font("helvetica", "B", 14)
    if tier_class == "very-high":
        pdf.set_text_color(16, 185, 129)
    elif tier_class == "high":
        pdf.set_text_color(59, 130, 246)
    elif tier_class == "medium":
        pdf.set_text_color(245, 158, 11)
    else:
        pdf.set_text_color(239, 68, 68)
        
    pdf.cell(90, 8, f"{predicted_hdi:.3f}")
    pdf.cell(0, 8, f"{tier_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(30, 41, 59) # Reset color
    pdf.ln(10)
    
    # Section 2: Table of Inputs
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 10, "2. Input Indicators Profile", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Table Header
    pdf.set_fill_color(79, 70, 229) # Indigo-600
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(100, 8, " Socioeconomic Dimension / Indicator", border=1, fill=True)
    pdf.cell(90, 8, " Assessed Input Value", border=1, fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Table Body
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("helvetica", "", 9.5)
    
    table_rows = [
        ("Life Expectancy at Birth", f"{le:.2f} years"),
        ("Expected Years of Schooling", f"{eys:.2f} years"),
        ("Mean Years of Schooling", f"{mys:.2f} years"),
        ("Gross National Income per Capita (PPP)", f"${gni:,.2f}")
    ]
    
    for label, val in table_rows:
        pdf.cell(100, 7.5, f"  {label}", border=1)
        pdf.cell(90, 7.5, val, border=1, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
    pdf.ln(8)
    
    # Section 3: Recommendations
    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 10, "3. Targeted Development Recommendations", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    for r in recs:
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 5, f"* {r['title']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 9)
        pdf.multi_cell(0, 4.5, r['desc'])
        pdf.ln(2.5)
        
    response = make_response(bytes(pdf.output()))
    response.headers["Content-Disposition"] = "attachment; filename=hdi_assessment_report.pdf"
    response.headers["Content-type"] = "application/pdf"
    return response

@app.route("/history", methods=["GET"])
def history():
    """Renders predictions log dashboard with sorting, searching, and clearing tools."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    search_q = request.args.get("search", "").strip()
    sort_by = request.args.get("sort", "timestamp")
    order = request.args.get("order", "desc")
    
    # Secure sorting fields
    valid_sorts = ["timestamp", "predicted_hdi", "life_expectancy", "gni"]
    if sort_by not in valid_sorts:
        sort_by = "timestamp"
    if order not in ["asc", "desc"]:
        order = "desc"
        
    query = "SELECT timestamp, life_expectancy, expected_schooling, mean_schooling, gni, predicted_hdi, category FROM predictions"
    params = []
    
    if search_q:
        query += " WHERE category LIKE ?"
        params.append(f"%{search_q}%")
        
    query += f" ORDER BY {sort_by} {order.upper()}"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    history_list = []
    for r in rows:
        history_list.append({
            "timestamp": r[0],
            "life_expectancy": r[1],
            "expected_schooling": r[2],
            "mean_schooling": r[3],
            "gni": r[4],
            "predicted_hdi": r[5],
            "category": r[6]
        })
        
    return render_template("history.html", history=history_list, search=search_q, sort=sort_by, order=order)

@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Clears SQLite predictions logs."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] Error clearing prediction database: {e}")
    return redirect(url_for("history"))

@app.route("/history/download", methods=["GET"])
def download_history():
    """Generates a downloadable CSV of all prediction history logs."""
    import io
    import csv
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, life_expectancy, expected_schooling, mean_schooling, gni, predicted_hdi, category FROM predictions ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(["Timestamp", "Life Expectancy (Years)", "Expected Schooling (Years)", "Mean Schooling (Years)", "GNI per Capita ($)", "Predicted HDI", "Development Category"])
        cw.writerows(rows)
        
        response = make_response(si.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=hdi_prediction_history.csv"
        response.headers["Content-type"] = "text/csv"
        return response
    except Exception as e:
        print(f"[!] CSV history generation error: {e}")
        return redirect(url_for("history"))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint returning predicted HDI, tier, and recommendations for a JSON payload."""
    global model, scaler
    if model is None or scaler is None:
        return jsonify({"error": "Model assets not loaded"}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        life_exp = float(data.get("Life_Expectancy", 0))
        exp_school = float(data.get("Expected_Years_Schooling", 0))
        mean_school = float(data.get("Mean_Years_Schooling", 0))
        gni = float(data.get("GNI_per_Capita", 0))
        
        # Validation checks
        if not (45.0 <= life_exp <= 86.0) or not (4.0 <= exp_school <= 21.0) or \
           not (1.0 <= mean_school <= 15.0) or not (400.0 <= gni <= 110000.0):
            return jsonify({"error": "Input parameters out of bounds"}), 400
            
        if mean_school > exp_school:
            return jsonify({"error": "Mean schooling cannot exceed expected schooling"}), 400
            
        input_features = pd.DataFrame([[life_exp, exp_school, mean_school, gni]], 
                                       columns=["Life_Expectancy", "Expected_Years_Schooling", "Mean_Years_Schooling", "GNI_per_Capita"])
        input_scaled = scaler.transform(input_features)
        predicted_hdi = float(np.clip(model.predict(input_scaled)[0], 0.0, 1.0))
        
        if predicted_hdi >= 0.800:
            tier_class = "very-high"
            tier_name = "Very High Human Development"
        elif predicted_hdi >= 0.700:
            tier_class = "high"
            tier_name = "High Human Development"
        elif predicted_hdi >= 0.550:
            tier_class = "medium"
            tier_name = "Medium Human Development"
        else:
            tier_class = "low"
            tier_name = "Low Human Development"
            
        return jsonify({
            "predicted_hdi": predicted_hdi,
            "tier_class": tier_class,
            "tier_name": tier_name,
            "recommendations": RECOMMENDATIONS.get(tier_class, [])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("500.html"), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
