import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Nonaktifkan oneDNN (opsional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Sembunyikan informasi & warning dari TensorFlow

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Configuration
MODEL_PATH = 'model/lstm_speed-model_dummy-dataset.h5'  # Path untuk model LSTM
SCALER_PATH = 'model/scaler_dummy-dataset.pkl'    # Path untuk scaler
RAW_DATA_PATH = 'data/RSSI_dummy_dataset.csv' # Path untuk data mentah
SEQUENCE_LENGTH = 10

# Load model dan scaler
model = load_model(MODEL_PATH)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

def preprocess_data(data):
    """
    Preprocess data menggunakan scaler yang sudah di-train
    """
    scaled_data = scaler.transform(data)
    return scaled_data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400

        # Preprocessing input data
        input_data = np.array(data['features']).reshape(-1, 1)
        scaled_input = preprocess_data(input_data)
        
        # Reshape untuk LSTM (batch_size, sequence_length, features)
        model_input = scaled_input.reshape(1, SEQUENCE_LENGTH, 1)
        
        # Make prediction
        prediction = model.predict(model_input)
        
        # Inverse transform prediction
        prediction_original = scaler.inverse_transform(prediction.reshape(-1, 1))
        
        return jsonify({
            'prediction': prediction_original.flatten().tolist(),
            'scaled_prediction': prediction.flatten().tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/raw-data', methods=['GET']) #menampilkan data mentah
def get_raw_data():
    """
    Endpoint untuk menampilkan data mentah (opsional)
    """
    try:
        if not os.path.exists(RAW_DATA_PATH):
            return jsonify({'error': 'Raw data file not found'}), 404
            
        # Baca beberapa baris pertama saja (misalnya 100 baris)
        df = pd.read_csv(RAW_DATA_PATH, nrows=100)
        
        return jsonify({
            'data': df.to_dict(orient='records'),
            'total_rows': len(df),
            'columns': df.columns.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/data-summary', methods=['GET']) #menampilkan ringkasan statistik data
def get_data_summary():
    """
    Endpoint untuk menampilkan ringkasan statistik data
    """
    try:
        if not os.path.exists(RAW_DATA_PATH):
            return jsonify({'error': 'Raw data file not found'}), 404
            
        df = pd.read_csv(RAW_DATA_PATH)
        
        summary = {
            'total_rows': len(df),
            'columns': df.columns.tolist(),
            'numerical_summary': df.describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'first_timestamp': df['timestamp'].min() if 'timestamp' in df.columns else None,
            'last_timestamp': df['timestamp'].max() if 'timestamp' in df.columns else None
        }
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)