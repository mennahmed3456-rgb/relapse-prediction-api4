from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# تحميل الموديل
try:
    model = joblib.load("final_relapse_model.pkl")
    print("✅ Model loaded successfully")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def score_to_class(score):
    if score < 0.25:
        return "Stable"
    elif score < 0.55:
        return "At_Risk"
    else:
        return "Relapsed"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded",
                "status": "error"
            }), 500
        
        # Get JSON data
        data = request.json
        
        if not data:
            return jsonify({
                "error": "No data provided",
                "status": "error"
            }), 400
        
        # Validate all required fields exist
        required_fields = [
            "Academic_Performance_Decline",
            "Social_Isolation",
            "Financial_Issues",
            "Physical_Mental_Health_Problems",
            "Legal_Consequences",
            "Relationship_Strain",
            "Risk_Taking_Behavior",
            "Withdrawal_Symptoms",
            "Denial_and_Resistance_to_Treatment"
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "error": f"Missing field: {field}",
                    "required_fields": required_fields,
                    "status": "error"
                }), 400
        
        # Extract features
        features = [
            float(data["Academic_Performance_Decline"]),
            float(data["Social_Isolation"]),
            float(data["Financial_Issues"]),
            float(data["Physical_Mental_Health_Problems"]),
            float(data["Legal_Consequences"]),
            float(data["Relationship_Strain"]),
            float(data["Risk_Taking_Behavior"]),
            float(data["Withdrawal_Symptoms"]),
            float(data["Denial_and_Resistance_to_Treatment"])
        ]

        # Reshape and predict
        features_array = np.array(features).reshape(1, -1)
        score = float(model.predict(features_array)[0])
        category = score_to_class(score)

        return jsonify({
            "risk_score": round(score, 4),
            "risk_category": category,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "endpoint": "/predict"
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Relapse Prediction API",
        "version": "1.0",
        "endpoints": {
            "GET /health": "Check API health",
            "POST /predict": "Make prediction",
            "GET /": "This info page"
        },
        "required_fields": [
            "Academic_Performance_Decline",
            "Social_Isolation",
            "Financial_Issues", 
            "Physical_Mental_Health_Problems",
            "Legal_Consequences",
            "Relationship_Strain",
            "Risk_Taking_Behavior",
            "Withdrawal_Symptoms",
            "Denial_and_Resistance_to_Treatment"
        ]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)