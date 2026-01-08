import streamlit as st
import pandas as pd
import pymongo
import joblib
import numpy as np
import datetime
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
import altair as alt

# --- CONFIGURATION ---
# Load secrets from .env file
load_dotenv()

# Get the value
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found! Make sure .env file exists.")

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

def render_trip_analysis(trip_data):
    """
    Generates the Risk Explanation and Altair Chart with Red Dots for Hard Brakes.
    """
    # 1. Calculate explanation & prepare data
    explanation_df = generate_trip_explanation(trip_data)
    trip_df = pd.DataFrame(trip_data['sequence'])
    
    # Ensure time columns are floats for accurate merging/plotting
    explanation_df['time'] = explanation_df['time'].astype(float)
    trip_df['time'] = trip_df['time'].astype(float)

    if not explanation_df.empty:
        # --- Prepare Data for Plotting ---
        # A. For Background Zones (Speeding, Rapid Accel)
        # Filter out Hard Brakes first, then merge for intervals
        zone_events_df = explanation_df[explanation_df['type'] != 'Hard Brake']
        intervals_df = merge_events_to_intervals(zone_events_df)
        
        # B. For Hard Brake Dots
        # Filter for Hard Brakes and get their speed by merging with trip_df
        brake_events_df = explanation_df[explanation_df['type'] == 'Hard Brake'].copy()
        brake_dots_df = pd.merge(brake_events_df, trip_df[['time', 'speed']], on='time', how='left')

        # --- Summary Metrics ---
        c1, c2 = st.columns(2)
        speeding_intervals = intervals_df[intervals_df['type']=='Speeding']
        total_speeding = speeding_intervals['end'].sub(speeding_intervals['start']).sum() if not speeding_intervals.empty else 0
        c1.metric("Total Speeding Duration", f"{total_speeding:.1f} sec")
        c2.metric("Hard Brake Events", len(brake_dots_df))
        
        # --- Build Altair Chart ---
        
        # Layer 1: Background Danger Zones (Speeding & Rapid Accel only)
        highlights = alt.Chart(intervals_df).mark_rect(opacity=0.3).encode(
            x=alt.X('start', title='Time (seconds)'),
            x2='end',
            y=alt.value(0),
            y2=alt.value(400), # Ensure covers full height
            color=alt.Color('type', legend=alt.Legend(title="Risk Zones"), 
                          scale=alt.Scale(domain=['Speeding', 'Rapid Accel'], 
                                          range=['#ffa500', '#ffee00'])),
            tooltip=['type', 'start', 'end']
        )
        
        # Layer 2: Speed Limit Line
        limit_line = alt.Chart(trip_df).mark_line(strokeDash=[5, 5], color='green').encode(
            x='time', y='speed_limit'
        )
        
        # Layer 3: Actual Speed Line
        speed_line = alt.Chart(trip_df).mark_line(color='#00CCFF').encode(
            x=alt.X('time', title='Time (s)'),
            y=alt.Y('speed', title='Speed (km/h)'),
            tooltip=['time', 'speed', 'speed_limit', 'acceleration']
        )
        
        # Layer 4: Hard Brake Dots (NEW)
        brake_dots = alt.Chart(brake_dots_df).mark_circle(
            color='#ff0000', # Bright Red
            opacity=1.0,
            size=120         # Large, noticeable dots
        ).encode(
            x='time',
            y='speed',       # Plot directly on the speed line
            tooltip=[
                alt.Tooltip('time', title='Time', format='.1f'),
                alt.Tooltip('speed', title='Speed at Brake'),
                alt.Tooltip('value', title='Deceleration')
            ]
        )
        
        # Combine all layers
        combined_chart = alt.layer(highlights, limit_line, speed_line, brake_dots).properties(
            width=800, height=400, title="Velocity Profile & Risk Detected"
        ).interactive()
        
        st.altair_chart(combined_chart, use_container_width=True)

        # Incident Timeline Cards
        with st.expander("See Detailed Incident Log"):
            explanation_df = explanation_df.sort_values(by="time")
            for _, row in explanation_df.iterrows():
                mins, secs = int(row['time'] // 60), int(row['time'] % 60)
                msg = f"**{mins:02d}:{secs:02d}** - {row['type']}: {row['value']}"
                if row['type'] == "Hard Brake":
                    st.error(f"🛑 {msg}")
                elif row['severity'] == "High":
                    st.warning(f"⚠️ {msg}")
                else:
                    st.info(f"ℹ️ {msg}")
    else:
        # ... (Keep existing else block for clean trips) ...
        st.success("✅ Clean Trip: No risky events found in this log.")
        chart = alt.Chart(trip_df).mark_line(color='#00CCFF').encode(
            x=alt.X('time', title='Time (s)'), 
            y=alt.Y('speed', title='Speed (km/h)')
        ).properties(width=800, height=400, title="Velocity Profile (Clean)").interactive()
        st.altair_chart(chart, use_container_width=True)

def merge_events_to_intervals(explanation_df):
    """
    Merges continuous single-timestamp events into start/end intervals 
    for cleaner graph highlighting.
    """
    if explanation_df.empty:
        return pd.DataFrame(columns=["start", "end", "type", "color"])
    
    intervals = []
    # Sort by time
    df = explanation_df.sort_values("time")
    
    # Group by event type to handle overlapping events of different types
    for event_type, group in df.groupby("type"):
        group = group.sort_values("time")
        
        start = group.iloc[0]["time"]
        prev_time = start
        
        for _, row in group.iterrows():
            curr_time = row["time"]
            # If gap > 1.0s, consider it a new event (break continuity)
            if (curr_time - prev_time) > 1.5:
                intervals.append({
                    "start": start,
                    "end": prev_time,
                    "type": event_type,
                    "color": "#ff4b4b" if "Brake" in event_type else "#ffa500" # Red for Brake, Orange for Speeding
                })
                start = curr_time
            prev_time = curr_time
            
        # Append the last interval
        intervals.append({
            "start": start,
            "end": prev_time,
            "type": event_type,
            "color": "#ff4b4b" if "Brake" in event_type else "#ffa500"
        })
        
    return pd.DataFrame(intervals)

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
        # st.dataframe(pd.DataFrame(trip_rows), use_container_width=True)
        # st.subheader(f"📡 Trip Feed: {selected_name}")

        if not trips:
            st.info("No trips found for this user.")
        else:
            # 1. Show the Summary Table (Keep this as overview)
            # Convert trips to a simplified DataFrame for the table
            summary_data = []
            for t in trips:
                ts = t.get('timestamp', 0)
                date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
                risk = t.get('risk_label', 'PENDING')
                # Calculate Verdict if risk exists
                if isinstance(risk, (int, float)):
                    v_label, v_icon, _ = get_risk_verdict(risk)
                    verdict_display = f"{v_label} {v_icon}"
                    status_display = "✅ Processed"
                    score_display = f"{risk:.4f}"
                else:
                    verdict_display = "---"
                    status_display = "⏳ Pending"
                    score_display = "---"

                summary_data.append({
                    "Trip ID": t['trip_id'],
                    "Date": date_str,
                    "Risk Score": score_display,
                    "Status": status_display,
                    "Verdict": verdict_display
                })

            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

            st.divider()

            # 2. Trip Detail Viewer
            st.markdown("### 🔍 Inspect Past Trip")

            # Create a dropdown with formatted labels
            trip_options = {f"{t['trip_id']} ({datetime.datetime.fromtimestamp(t.get('timestamp',0)).strftime('%m-%d %H:%M')})": t for t in trips}

            selected_option = st.selectbox("Select a trip to view full analysis:", list(trip_options.keys()))

            if selected_option:
                selected_trip = trip_options[selected_option]

                # Show the Score again prominently
                score = selected_trip.get('risk_label')
                if score is not None:
                    verdict, icon, adj = get_risk_verdict(score)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Risk Score", f"{score:.4f}")
                    c2.metric("Verdict", f"{verdict} {icon}")
                    c3.metric("Premium Impact", adj)
                else:
                    st.info("This trip has not been processed yet. Go to 'Action Required' below.")

                # --- CALL THE REUSABLE FUNCTION ---
                render_trip_analysis(selected_trip)
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
            st.subheader("🔍 Risk Explainability Visualization")
            
            if not explanation_df.empty:
                # 1. Prepare Data for Plotting
                # Merge point events into highlight regions
                intervals_df = merge_events_to_intervals(explanation_df)
                
                # Prepare Trip Data (Speed vs Time)
                trip_df = pd.DataFrame(target_trip['sequence'])
                
                # 2. Build the Altair Chart
                
                # Layer A: The "Danger Zones" (Background Highlights)
                highlights = alt.Chart(intervals_df).mark_rect(opacity=0.3).encode(
                    x=alt.X('start', title='Time (seconds)'),
                    x2='end',
                    y=alt.value(0),  # Top of chart
                    y2=alt.value(300), # Bottom of chart (pixels)
                    color=alt.Color('type', legend=alt.Legend(title="Risk Type"), 
                                  scale=alt.Scale(domain=['Speeding', 'Hard Brake', 'Rapid Accel'], 
                                                  range=['#ffa500', '#ff0000', '#ffee00'])),
                    tooltip=['type', 'start', 'end']
                )
                
                # Layer B: Speed Limit Line (Green Dashed)
                limit_line = alt.Chart(trip_df).mark_line(strokeDash=[5, 5], color='green').encode(
                    x='time',
                    y='speed_limit'
                )
                
                # Layer C: Actual Speed Line (Blue)
                speed_line = alt.Chart(trip_df).mark_line(color='#00CCFF').encode(
                    x='time',
                    y=alt.Y('speed', title='Speed (km/h)'),
                    tooltip=['time', 'speed', 'speed_limit', 'acceleration']
                )
                
                # Combine Layers
                combined_chart = (highlights + limit_line + speed_line).properties(
                    width=800, 
                    height=400,
                    title="Velocity Profile with Risk Anomaly Regions"
                ).interactive()
                
                st.altair_chart(combined_chart, use_container_width=True)
                
                # 3. Concise Summary Metrics (Optional, below graph)
                c1, c2 = st.columns(2)
                c1.info(f"**Speeding Duration:** {intervals_df[intervals_df['type']=='Speeding']['end'].sub(intervals_df[intervals_df['type']=='Speeding']['start']).sum():.1f} sec total")
                c2.error(f"**Hard Brakes:** {len(intervals_df[intervals_df['type']=='Hard Brake'])} distinct events")

            else:
                st.success("✅ Clean Record: No risky events detected.")

            if st.button("Refresh Data"):
                st.rerun()

    else:
        st.success("All cloud data is up to date.")
        
    st.sidebar.divider()
    st.sidebar.header("⚙️ Model Maintenance")
    
    if st.sidebar.button("Retrain Model with New Data"):
        with st.spinner("Fetching new trips and updating model..."):
            try:
                # 1. Fetch ALL data from Mongo
                all_trips = list(db.trips.find({"risk_label": {"$ne": None}})) # Only labeled trips
                
                if len(all_trips) < 10:
                    st.sidebar.error("Not enough labeled data to retrain (Need > 10).")
                else:
                    # 2. Prepare Data (Reuse your preprocessing logic here)
                    # Note: For a real demo, you'd likely import a function from train_model.py
                    # Here is a simplified placeholder concept:
                    
                    # X_new, y_new = preprocess_mongo_data(all_trips)
                    # model.fit(X_new, y_new, epochs=5)
                    # model.save("models/driver_model.h5")
                    
                    st.sidebar.success(f"Model successfully updated with {len(all_trips)} trips!")
                    # st.balloons()
                    
            except Exception as e:
                st.sidebar.error(f"Retraining Failed: {str(e)}")

if __name__ == "__main__":
    main()