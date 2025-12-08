import numpy as np
import json
import joblib
import os
import glob
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = "models/driver_model.h5"
SCALER_PATH = "models/scaler.pkl"
HUMAN_DATA_DIR = "data/raw_human"
TIMESTEPS = 360  # Must match training
FEATURES = ['speed', 'acceleration', 'speed_limit', 'is_speeding', 'throttle', 'brake']

def get_latest_human_file():
    """Finds the most recently created file in the human data folder."""
    list_of_files = glob.glob(os.path.join(HUMAN_DATA_DIR, '*.json'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def predict_trip(file_path):
    # 1. Load Model and Scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or Scaler not found. Train the model first!")
        return

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    print(f"\n--- Analyzing Trip: {os.path.basename(file_path)} ---")

    # 2. Load and Parse JSON
    with open(file_path, 'r') as f:
        trip = json.load(f)
    
    # Extract features
    raw_sequence = []
    for point in trip['sequence']:
        features = [point[f] for f in FEATURES]
        raw_sequence.append(features)
    
    # 3. Fix Sequence Length (Padding or Truncating)
    # The model DEMANDS exactly 360 timesteps. Human trips might be 359 or 361.
    raw_sequence = np.array(raw_sequence)
    
    if len(raw_sequence) > TIMESTEPS:
        # Truncate (cut off extra seconds)
        raw_sequence = raw_sequence[:TIMESTEPS]
    elif len(raw_sequence) < TIMESTEPS:
        # Pad (repeat the last frame)
        padding = np.tile(raw_sequence[-1], (TIMESTEPS - len(raw_sequence), 1))
        raw_sequence = np.vstack((raw_sequence, padding))
        
    # 4. Scale Data
    # Reshape to (360, 6) -> Scale -> Reshape back to (1, 360, 6)
    scaled_sequence = scaler.transform(raw_sequence)
    input_data = scaled_sequence.reshape(1, TIMESTEPS, len(FEATURES))
    
    # 5. Predict
    prediction = model.predict(input_data, verbose=0)[0][0]
    
    # 6. Output Results
    print(f"\n>>> CALCULATED RISK SCORE: {prediction:.4f}")
    
    # Interpret score
    if prediction < 0.3:
        print(">>> VERDICT: SAFE DRIVER 🟢")
        print("    Analysis: Consistent speed, smooth controls.")
    elif prediction < 0.7:
        print(">>> VERDICT: MODERATE RISK 🟡")
        print("    Analysis: Occasional hard braking or minor speeding.")
    else:
        print(">>> VERDICT: HIGH RISK / AGGRESSIVE 🔴")
        print("    Analysis: Dangerous speeding, erratic acceleration, harsh braking.")

if __name__ == "__main__":
    # You can specify a file, or just grab the latest one
    latest_file = get_latest_human_file()
    
    if latest_file:
        predict_trip(latest_file)
    else:
        print("No human trips found. Run play_trip.py first!")