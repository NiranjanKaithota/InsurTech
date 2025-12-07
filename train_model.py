import numpy as np
import pandas as pd
import json
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam  # Import Adam
import matplotlib.pyplot as plt
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Configuration ---
DATA_DIR = "data/raw"
MODEL_SAVE_PATH = "models/driver_model.h5"
TIMESTEPS = 360  # Updated to match your 3-minute (360 sec) duration
FEATURES = ['speed', 'acceleration', 'speed_limit', 'is_speeding', 'throttle', 'brake']

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# --- 2. Data Loading Function ---
def load_data():
    """Loads all trip files and extracts sequences and labels."""
    sequences = []
    labels = []
    
    # Use glob to find all json files in the directory
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    
    if not json_files:
        print(f"Error: No .json files found in {DATA_DIR}.")
        print("Please run generate_data.py first.")
        return None, None

    print(f"Found {len(json_files)} trip files. Loading...")

    for file_path in json_files:
        with open(file_path, 'r') as f:
            trip = json.load(f)
            
            # Extract the sequence data for our chosen features
            sequence_data = []
            for point in trip['sequence']:
                
                # Dynamically build the feature list for this time step
                features_for_this_point = [point[feature_name] for feature_name in FEATURES]
                sequence_data.append(features_for_this_point)
            
            # Ensure sequence is the correct length
            if len(sequence_data) == TIMESTEPS:
                sequences.append(sequence_data)
                labels.append(trip['risk_label'])
            else:
                print(f"Skipping file {file_path}: incorrect length. Expected {TIMESTEPS}, got {len(sequence_data)}")

    print(f"Successfully loaded {len(sequences)} sequences.")
    return np.array(sequences), np.array(labels)

# --- 3. Preprocessing Function ---
def preprocess_data(X, y):
    """Scales the data and splits into train/test sets."""
    
    # We must scale the features. We use MinMaxScaler.
    # We must reshape to 2D for the scaler, then reshape back to 3D.
    
    # Create a new scaler object for this dataset
    scaler = MinMaxScaler()
    
    # Reshape to (samples * timesteps, features)
    X_reshaped = X.reshape(-1, len(FEATURES))
    
    # Fit and transform the data
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    
    # Reshape back to (samples, timesteps, features)
    X_scaled = X_scaled_reshaped.reshape(X.shape)
    
    # Split the data (using 20% for validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Data shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

# --- 4. Model Building Function ---
def build_model():
    """Builds the LSTM model architecture."""
    model = Sequential()
    
    # Input shape is (TIMESTEPS, FEATURES)
    # Using 64 units and a Dropout of 0.5 as we established
    model.add(LSTM(64, input_shape=(TIMESTEPS, len(FEATURES))))
    
    # Dropout layer to prevent overfitting
    model.add(Dropout(0.5))
    
    # A hidden Dense layer
    model.add(Dense(32, activation='leaky-relu'))
    
    # Final output layer (1 neuron, sigmoid for 0-1 score)
    model.add(Dense(1, activation='sigmoid'))
    
    # --- Optimizer with stable learning rate and gradient clipping ---
    optimizer = Adam(learning_rate=0.0001, clipvalue=1.0) 
    
    model.compile(loss='mean_squared_error', 
                  optimizer=optimizer, 
                  metrics=['mean_absolute_error'])
    
    model.summary()
    return model

# --- 5. Training and Plotting Function ---
def train_and_plot(model, X_train, y_train, X_test, y_test):
    """Trains the model and plots the history."""
    print("\n--- Starting Model Training ---")
    
    # Train for 50 epochs with a batch size of 16
    history = model.fit(
        X_train, y_train,
        epochs=35,  # Changed from 50 to 35
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1 
        # ,callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )
    
    print("--- Training Complete ---")

    # Plot training & validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_loss_plot.png')
    print("Saved training loss plot to 'training_loss_plot.png'")
    
    return model

# --- 6. Main Execution ---
def main():
    # Step 1: Load Data
    X, y = load_data()
    if X is None or len(X) == 0:
        print("Data loading failed or no data found. Exiting.")
        return
        
    # Step 2: Preprocess Data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Step 3: Build Model
    model = build_model()
    
    # Step 4: Train Model
    model = train_and_plot(model, X_train, y_train, X_test, y_test)
    
    # Step 5: Evaluate Final Model
    print("\n--- Evaluating Model on Test Set ---")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Loss (MSE): {loss:.4f}")
    print(f"Final Test Mean Absolute Error: {mae:.4f}")
    
    # Step 6: Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel successfully saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()