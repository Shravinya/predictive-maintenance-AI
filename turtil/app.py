from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Load trained model & scaler
model_path = "predictive_maintenance_model.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("❌ Model or Scaler file not found. Please train the model first.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json  

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Define required features (same as training)
        required_features = ['metric1', 'metric2', 'metric3', 'metric4', 'metric5', 
                             'metric6', 'metric7', 'metric8', 'metric9']

        # Check if all required features are present
        if not all(feature in df.columns for feature in required_features):
            missing = [feature for feature in required_features if feature not in df.columns]
            return jsonify({'error': f'Missing required features: {missing}'}), 400

        # Normalize input data
        df_scaled = scaler.transform(df[required_features])

        # Make predictions
        predictions = model.predict(df_scaled)

        return jsonify({'prediction': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
