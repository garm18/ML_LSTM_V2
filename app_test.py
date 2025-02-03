from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# File paths
MODEL_PATH = "lstm-speed_model.pkl"
DATA_PATH = "sf07.csv"

# Load model if it exists
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = LinearRegression()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = pd.DataFrame([data['features']])
    prediction = model.predict(input_features)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/retrain', methods=['POST'])
def retrain():
    # Load training data
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': 'Training data not found.'}), 404

    training_data = pd.read_csv(DATA_PATH)
    X = training_data.drop('target', axis=1)
    y = training_data['target']

    # Retrain the model
    model.fit(X, y)

    # Save the model
    joblib.dump(model, MODEL_PATH)
    return jsonify({'message': 'Model retrained successfully'})

if __name__ == '__main__':
    app.run(debug=True)
