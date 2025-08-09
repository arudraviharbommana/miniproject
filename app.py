from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import CORS
import joblib
import pandas as pd
import pickle
import os
import math

app = Flask(__name__)
CORS(app)
CORS(app)

# Load model and columns
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.pkl')
RESULTS_PATH = os.path.join(MODEL_DIR, 'model_results.pkl')

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

def convert_nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Encoding maps (must match notebook LabelEncoder order)
        continent_map = {"Asia": 0, "Europe": 1, "Africa": 2, "North America": 3, "South America": 4, "Oceania": 5}
        education_map = {"Bachelor's": 0, "Master's": 2, "Doctorate": 3, "High School": 1}
        yesno_map = {"Yes": 1, "No": 0}
        region_map = {"Midwest": 0, "Northeast": 1, "South": 3, "West": 2}
        unit_map = {"Hour": 0, "Week": 2, "Month": 3, "Year": 1}

        # Encode categorical
        encoded = {
            'continent': continent_map.get(data['continent'], 0),
            'education_of_employee': education_map.get(data['education_of_employee'], 0),
            'has_job_experience': yesno_map.get(data['has_job_experience'], 0),
            'requires_job_training': yesno_map.get(data['requires_job_training'], 0),
            'region_of_employment': region_map.get(data['region_of_employment'], 0),
            'unit_of_wage': unit_map.get(data['unit_of_wage'], 0),
            'full_time_position': yesno_map.get(data['full_time_position'], 0)
        }

        # Standardize numerical (use same transformation as notebook)
        import numpy as np
        # Example means/stds (replace with actual values from notebook if needed)
        # For now, just apply log/sqrt as in notebook
        no_of_employees_log = np.log(data['no_of_employees'])
        yr_of_estab_log = np.log(data['yr_of_estab'])
        prevailing_wage_sqrt = np.sqrt(data['prevailing_wage'])

        # StandardScaler values (replace with actual means/stds from notebook for true standardization)
        # For demo, use dummy mean/std
        no_of_employees_log_stand = (no_of_employees_log - 5.5) / 1.2
        yr_of_estab_log_stand = (yr_of_estab_log - 7.5) / 0.2
        prevailing_wage_sqrt_stand = (prevailing_wage_sqrt - 500) / 100

        # Build final input
        model_input = {
            **encoded,
            'no_of_employees_log_stand': no_of_employees_log_stand,
            'yr_of_estab_log_stand': yr_of_estab_log_stand,
            'prevailing_wage_sqrt_stand': prevailing_wage_sqrt_stand
        }
        df = pd.DataFrame([model_input])
        df = df.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(df)[0]
        label_map = {0: "Denied", 1: "Certified"}
        readable_pred = label_map.get(int(prediction), str(prediction))
        return jsonify({"prediction": readable_pred, "raw": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/model/results", methods=["GET"])
def model_results():
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        results = convert_nan_to_none(results)
        results["accuracy"] = float(results["accuracy"])
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/model/info", methods=["GET"])
def model_info():
    try:
        info = {
            "model_file": os.path.basename(MODEL_PATH),
            "columns_file": os.path.basename(COLUMNS_PATH),
            "results_file": os.path.basename(RESULTS_PATH),
            "model_columns": model_columns
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
