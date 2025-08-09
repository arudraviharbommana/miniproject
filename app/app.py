from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model and columns
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')
COLUMNS_PATH = os.path.join(MODEL_DIR, 'model_columns.pkl')
RESULTS_PATH = os.path.join(MODEL_DIR, 'model_results.pkl')

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(df)[0]
        return jsonify({"prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/model/results", methods=["GET"])
def model_results():
    try:
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        # Convert numpy types to native Python types for JSON serialization
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
    app.run(debug=True)
