# ============================================================
#   🔋 Battery Carbon Footprint — Flask Backend (app.py)
#
#   HOW TO RUN:
#   1. pip install flask flask-cors pandas numpy scikit-learn
#   2. python app.py
#   3. Server starts at http://localhost:5000
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)   # Allow frontend (HTML file) to call this API

# ── GLOBAL MODEL VARIABLES ───────────────────────────────────
model       = None
le_chem     = None
le_country  = None
model_stats = {}

# ============================================================
#   STEP 1: GENERATE SYNTHETIC DATASET
# ============================================================
def generate_dataset(n=2000):
    np.random.seed(42)

    capacity_kwh           = np.random.uniform(1, 100, n)
    chemistry              = np.random.choice(["NMC", "LFP", "NCA"], n, p=[0.5, 0.35, 0.15])
    manufacturing_country  = np.random.choice(["China", "USA", "Germany", "Norway"], n, p=[0.5, 0.2, 0.2, 0.1])
    cycle_life             = np.random.randint(500, 3000, n)
    depth_of_discharge     = np.random.uniform(0.5, 1.0, n)
    grid_carbon_intensity  = np.random.uniform(20, 820, n)
    annual_energy_kwh      = np.random.uniform(100, 5000, n)
    recycling_rate         = np.random.uniform(0.1, 0.95, n)

    chem_factor   = {"NMC": 1.0, "NCA": 1.15, "LFP": 0.70}
    country_grid  = {"China": 0.68, "USA": 0.40, "Germany": 0.35, "Norway": 0.03}

    chem_f    = np.array([chem_factor[c]  for c in chemistry])
    country_g = np.array([country_grid[c] for c in manufacturing_country])

    manufacturing_co2 = (
        capacity_kwh * 75 * chem_f
        + capacity_kwh * 75 * chem_f * country_g * 0.4
        + np.random.normal(0, 150, n)
    )
    years   = cycle_life * depth_of_discharge / 365
    use_co2 = annual_energy_kwh * (grid_carbon_intensity / 1000) * years + np.random.normal(0, 80, n)
    eol_co2 = capacity_kwh * 8 * (1 - recycling_rate) + np.random.normal(0, 20, n)
    total_co2 = np.clip(manufacturing_co2 + use_co2 + eol_co2, 100, None)

    df = pd.DataFrame({
        "capacity_kwh":          capacity_kwh,
        "chemistry":             chemistry,
        "manufacturing_country": manufacturing_country,
        "cycle_life":            cycle_life,
        "depth_of_discharge":    depth_of_discharge,
        "grid_carbon_intensity": grid_carbon_intensity,
        "annual_energy_kwh":     annual_energy_kwh,
        "recycling_rate":        recycling_rate,
        "manufacturing_co2":     np.clip(manufacturing_co2, 0, None),
        "use_co2":               np.clip(use_co2, 0, None),
        "eol_co2":               np.clip(eol_co2, 0, None),
        "total_co2_kg":          total_co2,
    })
    return df


# ============================================================
#   STEP 2: TRAIN THE ML MODEL (runs once on startup)
# ============================================================
def train_model():
    global model, le_chem, le_country, model_stats

    print("🔧 Generating dataset and training model...")
    df = generate_dataset(2000)

    le_chem    = LabelEncoder()
    le_country = LabelEncoder()
    df["chemistry_enc"] = le_chem.fit_transform(df["chemistry"])
    df["country_enc"]   = le_country.fit_transform(df["manufacturing_country"])

    FEATURES = [
        "capacity_kwh", "chemistry_enc", "country_enc",
        "cycle_life", "depth_of_discharge",
        "grid_carbon_intensity", "annual_energy_kwh", "recycling_rate"
    ]

    X = df[FEATURES]
    y = df["total_co2_kg"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5,
        learning_rate=0.05, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    model_stats = {
        "mae":      round(mean_absolute_error(y_test, y_pred), 2),
        "rmse":     round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "r2":       round(r2_score(y_test, y_pred), 4),
        "samples":  len(df),
        "model":    "Gradient Boosting Regressor"
    }
    print(f"✅ Model trained! R² = {model_stats['r2']}  MAE = {model_stats['mae']} kg")


# ============================================================
#   API ROUTES
# ============================================================

# Route 1: Health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":      "running",
        "message":     "Battery Carbon Footprint API is live 🔋",
        "model_stats": model_stats
    })


# Route 2: Predict CO2
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate required fields
        required = [
            "capacity_kwh", "chemistry", "manufacturing_country",
            "cycle_life", "depth_of_discharge",
            "grid_carbon_intensity", "annual_energy_kwh", "recycling_rate"
        ]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        chemistry = data["chemistry"]
        country   = data["manufacturing_country"]

        if chemistry not in le_chem.classes_:
            return jsonify({"error": f"Invalid chemistry. Use: {list(le_chem.classes_)}"}), 400
        if country not in le_country.classes_:
            return jsonify({"error": f"Invalid country. Use: {list(le_country.classes_)}"}), 400

        # Build feature row for model
        cap    = float(data["capacity_kwh"])
        dod    = float(data["depth_of_discharge"])
        cycles = int(data["cycle_life"])
        gi     = float(data["grid_carbon_intensity"])
        ae     = float(data["annual_energy_kwh"])
        rec    = float(data["recycling_rate"])

        row = pd.DataFrame([{
            "capacity_kwh":          cap,
            "chemistry_enc":         le_chem.transform([chemistry])[0],
            "country_enc":           le_country.transform([country])[0],
            "cycle_life":            cycles,
            "depth_of_discharge":    dod,
            "grid_carbon_intensity": gi,
            "annual_energy_kwh":     ae,
            "recycling_rate":        rec,
        }])

        # ML Model prediction
        total_co2 = float(model.predict(row)[0])

        # Phase breakdown (formula-based, scaled to match model output)
        chem_factor  = {"NMC": 1.0, "NCA": 1.15, "LFP": 0.70}
        country_grid = {"China": 0.68, "USA": 0.40, "Germany": 0.35, "Norway": 0.03}
        cf = chem_factor[chemistry]
        cg = country_grid[country]

        mfg_raw = max(cap * 75 * cf + cap * 75 * cf * cg * 0.4, 0)
        years   = (cycles * dod) / 365
        use_raw = max(ae * (gi / 1000) * years, 0)
        eol_raw = max(cap * 8 * (1 - rec), 0)

        raw_sum = mfg_raw + use_raw + eol_raw
        scale   = total_co2 / raw_sum if raw_sum > 0 else 1

        return jsonify({
            "total_co2_kg":      round(total_co2),
            "manufacturing_co2": round(mfg_raw * scale),
            "use_co2":           round(use_raw * scale),
            "eol_co2":           round(eol_raw * scale),
            "lifecycle_years":   round(years, 1),
            "co2_per_kwh":       round(total_co2 / cap, 1),
            "model_r2":          model_stats["r2"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Route 3: Model info
@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model":             model_stats.get("model"),
        "r2_score":          model_stats.get("r2"),
        "mae_kg":            model_stats.get("mae"),
        "rmse_kg":           model_stats.get("rmse"),
        "trained_on":        model_stats.get("samples"),
        "valid_chemistry":   list(le_chem.classes_),
        "valid_countries":   list(le_country.classes_),
    })


# ============================================================
#   START SERVER
# ============================================================
if __name__ == "__main__":
    train_model()
    print("\n🚀 Server running at http://localhost:5000")
    print("📡 Frontend can call: POST http://localhost:5000/predict\n")
    app.run(debug=True, port=5000)
