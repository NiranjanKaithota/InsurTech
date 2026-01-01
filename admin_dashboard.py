import streamlit as st
import pandas as pd
import pymongo
import joblib
import numpy as np
import datetime
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
MONGO_URI = "mongodb+srv://niranjanskai23:asdfghjkl@tripdata.vmyrcn3.mongodb.net/?appName=TripData"

MODEL_PATH = "models/driver_model.h5"
SCALER_PATH = "models/scaler.pkl"
TIMESTEPS = 360
FEATURES = ['speed', 'acceleration', 'speed_limit', 'is_speeding', 'throttle', 'brake']

st.set_page_config(page_title="UBI Cloud Admin", page_icon="☁️", layout="wide")

# --- DATABASE CONNECTION ---
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(MONGO_URI)

client = init_connection()
db = client["ubi_database"]

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_ai_model():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def generate_trip_explanation(trip_data):
    """
    Scans the trip sequence to find risky events with GRACE PERIOD logic.
    """
    events = []
    sequence = trip_data['sequence']
    
    # Thresholds
    HARD_BRAKE_THRESH = -3.0
    RAPID_ACCEL_THRESH = 3.5
    SPEEDING_BUFFER = 5.0 
    
    # Grace Period Logic
    grace_duration = 5.0  # seconds to adjust to new limit
    grace_end_time = -1.0
    prev_limit = sequence[0]['speed_limit'] if sequence else 0

    for i, point in enumerate(sequence):
        timestamp = point['time']
        speed = point['speed']
        limit = point['speed_limit']
        accel = point['acceleration']
        
        # 1. Check for Limit Drop (Start Grace Period)
        if limit < prev_limit:
            # We entered a slower zone. Give them time to slow down.
            grace_end_time = timestamp + grace_duration
            
        prev_limit = limit
        
        # 2. Speeding Check with Grace Logic
        is_in_grace_period = (timestamp < grace_end_time)
        
        # If in grace period, ONLY flag if they are NOT slowing down (accel >= 0)
        # If not in grace period, flag normally
        if speed > (limit + SPEEDING_BUFFER):
            should_flag = True
            
            if is_in_grace_period:
                if accel < -0.5: 
                    # They are braking actively, so forgive the speeding
                    should_flag = False
                else:
                    # They are speeding AND not slowing down -> Flag it
                    should_flag = True
            
            if should_flag:
                # Reduce frequency: Only log every ~1 second (every 10th frame)
                # This prevents the "flood" of logs you saw in your screenshot
                if i % 10 == 0: 
                    events.append({
                        "time": timestamp,
                        "type": "Speeding",
                        "value": f"{int(speed)} km/h (Limit: {limit})",
                        "severity": "High" if speed > limit + 15 else "Moderate"
                    })
            
        # 3. Hard Braking / Rapid Accel (Always Check)
        elif accel < HARD_BRAKE_THRESH:
             if i % 10 == 0: # Debounce
                events.append({
                    "time": timestamp,
                    "type": "Hard Brake",
                    "value": f"{accel} m/s²",
                    "severity": "High"
                })
            
        elif accel > RAPID_ACCEL_THRESH:
             if i % 10 == 0: # Debounce
                events.append({
                    "time": timestamp,
                    "type": "Rapid Accel",
                    "value": f"{accel} m/s²",
                    "severity": "Moderate"
                })

    return pd.DataFrame(events)


def get_risk_verdict(score):
    if score < 0.3: return "SAFE", "🟢", "Discount Applied: -15%"
    if score < 0.7: return "MODERATE", "🟡", "Standard Premium"
    return "HIGH RISK", "🔴", "Premium Hike: +20%"

def analyze_trip_ai(trip_data, model, scaler):
    """Runs prediction on trip data fetched from cloud."""
    # Preprocess
    raw_sequence = []
    for point in trip_data['sequence']:
        features = [point[f] for f in FEATURES]
        raw_sequence.append(features)
        
    raw_sequence = np.array(raw_sequence)
    
    # Fix Length
    if len(raw_sequence) > TIMESTEPS:
        raw_sequence = raw_sequence[:TIMESTEPS]
    elif len(raw_sequence) < TIMESTEPS:
        padding = np.tile(raw_sequence[-1], (TIMESTEPS - len(raw_sequence), 1))
        raw_sequence = np.vstack((raw_sequence, padding))
        
    # Predict
    scaled_sequence = scaler.transform(raw_sequence)
    input_data = scaled_sequence.reshape(1, TIMESTEPS, len(FEATURES))
    prediction = model.predict(input_data, verbose=0)[0][0]
    
    return float(prediction)

# --- MAIN APP ---
def main():
    st.title("☁️ UBI Cloud Command Center")
    st.markdown("### Real-Time Telematics & Premium Adjustment")
    st.divider()

    try:
        model, scaler = load_ai_model()
    except:
        st.error("Model not found. Ensure models/driver_model.h5 exists.")
        return

    # Sidebar: User Selection from Cloud
    st.sidebar.header("📁 Policy Holders")
    users = list(db.users.find())
    
    if not users:
        st.warning("No users in Cloud DB. Run db_setup.py")
        return

    user_names = [u['name'] for u in users]
    selected_name = st.sidebar.selectbox("Select User", user_names)
    selected_user = next(u for u in users if u['name'] == selected_name)
    user_id = selected_user['user_id']

    # Fetch Trips for this User from Cloud
    trips = list(db.trips.find({"user_id": user_id}).sort("timestamp", -1))

    # --- Metrics ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Policy No", selected_user.get("policy_no", "N/A"))
    c2.metric("Vehicle", selected_user.get("vehicle", "N/A"))
    c3.metric("Cloud Trips Logged", len(trips))

    st.divider()

    # --- Trip History Table ---
    st.subheader(f"📡 Trip Feed: {selected_name}")
    
    trip_rows = []
    pending_trips = []

    for t in trips:
        risk_val = t.get('risk_label')
        
        # Format Date
        ts = t.get('timestamp', 0)
        date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')

        if risk_val is None:
            status = "PENDING ⏳"
            premium = "---"
            pending_trips.append(t)
        else:
            verdict, icon, premium_adj = get_risk_verdict(risk_val)
            status = f"{verdict} {icon}"
            premium = premium_adj

        trip_rows.append({
            "Trip ID": t.get('trip_id'),
            "Date": date_str,
            "Duration": f"{len(t['sequence'])/10:.1f}s",
            "Status": status,
            "Premium Action": premium
        })

    if trip_rows:
        st.dataframe(pd.DataFrame(trip_rows), use_container_width=True)
    else:
        st.info("Waiting for data from vehicle...")

    # --- Validation Section ---
    st.divider()
    
    if pending_trips:
        st.subheader("⚡ Action Required: New Trip Data Received")
        
        # Select trip
        trip_options = {t['trip_id']: t for t in pending_trips}
        selected_trip_id = st.selectbox("Select Trip to Analyze", list(trip_options.keys()))
        
        if st.button("Run AI Risk Assessment", type="primary"):
            target_trip = trip_options[selected_trip_id]
            
            with st.spinner("Fetching data from cloud & processing..."):
                # 1. Run Model
                score = analyze_trip_ai(target_trip, model, scaler)
                
                # 2. Run Explainability Engine
                explanation_df = generate_trip_explanation(target_trip)
                
                # 3. Update Cloud DB
                db.trips.update_one(
                    {"_id": target_trip["_id"]},
                    {"$set": {"risk_label": score}}
                )
                
            # --- DISPLAY RESULTS ---
            verdict, icon, adj = get_risk_verdict(score)
            
            st.success(f"Risk Score: {score:.4f}")
            st.info(f"Verdict: {verdict} {icon}")
            st.warning(f"Recommended Action: **{adj}**")
            
            st.divider()
            st.subheader("🔍 Risk Explainability Report")
            
            if not explanation_df.empty:
                # 1. Summary Metrics
                c1, c2, c3 = st.columns(3)
                n_speeding = len(explanation_df[explanation_df['type'] == 'Speeding'])
                n_brake = len(explanation_df[explanation_df['type'] == 'Hard Brake'])
                n_accel = len(explanation_df[explanation_df['type'] == 'Rapid Accel'])
                
                c1.metric("Speeding Incidents", n_speeding, delta_color="inverse")
                c2.metric("Hard Brakes", n_brake, delta_color="inverse")
                c3.metric("Rapid Accelerations", n_accel, delta_color="inverse")
                
                # 2. Interactive Chart
                # Create a chart showing Speed vs Time, highlighting speeding zones
                st.markdown("#### Trip Velocity Profile")
                
                # Convert sequence to DF for plotting
                seq_df = pd.DataFrame(target_trip['sequence'])
                
                # We use a line chart for speed and limit
                chart_data = seq_df[['time', 'speed', 'speed_limit']].set_index('time')
                st.line_chart(chart_data, color=["#00CCFF", "#FF4B4B"]) 
                # Note: Streamlit colors: Speed (Blue), Limit (Red)
                
                # 3. Detailed Event Log
                st.markdown("#### Detected Risk Events")
                st.dataframe(
                    explanation_df, 
                    column_config={
                        "time": "Time (s)",
                        "type": "Violation Type",
                        "value": "Recorded Value",
                        "severity": st.column_config.TextColumn(
                            "Severity",
                            help="High severity impacts premium more",
                            validate="^(High|Moderate)$"
                        ),
                    },
                    use_container_width=True
                )
            else:
                st.success("✅ No specific risk events detected. The driver demonstrated smooth, compliant behavior.")

            if st.button("Refresh Data"):
                st.rerun()

    else:
        st.success("All cloud data is up to date.")

if __name__ == "__main__":
    main()