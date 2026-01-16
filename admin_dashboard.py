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
from streamlit_option_menu import option_menu

load_dotenv()

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

def calculate_personalized_premium(user_policy, risk_score, current_monthly_distance):
    """
    Calculates premium based on specific user add-ons and limits.
    """
    # 1. Extract Policy Details
    config = user_policy.get('policy_config', {})
    
    base_fee = config.get('base_premium', 500)
    dist_cap = config.get('distance_cap_km', 1000)
    rate_normal = config.get('rate_within_cap', 2.0)
    rate_overage = config.get('rate_overage', 10.0)
    low_usage_thresh = config.get('low_usage_threshold', 100)
    low_usage_disc = config.get('low_usage_discount', 100)

    # 2. Calculate Risk Multiplier (Exponential Penalty for bad drivers)
    # Score 0.1 -> 1.01x (Tiny penalty)
    # Score 0.9 -> 1.81x (Huge penalty)
    risk_multiplier = 1.0 + (risk_score ** 2)

    # 3. Distance Logic (The "Cap" Feature)
    dist_cost = 0.0
    note = "Standard Usage"
    
    if current_monthly_distance <= low_usage_thresh:
        # Scenario A: Low Usage Discount
        dist_cost = current_monthly_distance * rate_normal
        base_fee -= low_usage_disc  # Apply discount
        note = "Low Usage Discount Applied ✅"
        
    elif current_monthly_distance <= dist_cap:
        # Scenario B: Within Limit
        dist_cost = current_monthly_distance * rate_normal
        note = "Within Distance Cap"
        
    else:
        # Scenario C: Overage Penalty
        # First X kms are normal rate
        normal_dist_cost = dist_cap * rate_normal
        # Extra kms are at high rate
        extra_km = current_monthly_distance - dist_cap
        overage_cost = extra_km * rate_overage
        
        dist_cost = normal_dist_cost + overage_cost
        note = f"⚠️ Overage Alert: Exceeded Cap by {extra_km:.1f} km"

    # 4. Apply Risk Factor to the Distance Cost (Not the base fee)
    # Bad drivers pay more to drive, but fixed fee remains fixed.
    final_variable_cost = dist_cost * risk_multiplier
    
    total_premium = base_fee + final_variable_cost

    return {
        "Total Premium": round(total_premium, 2),
        "Base Fee": base_fee,
        "Variable Cost": round(final_variable_cost, 2),
        "Risk Multiplier": round(risk_multiplier, 2),
        "Status Note": note
    }

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

# MAIN APP
def main():
    # --- Sidebar Global Selection ---
    with st.sidebar:
        st.header("🚘 User Profile")
        
        # Fetch Users
        users = list(db.users.find())
        if not users:
            st.warning("No users found in Cloud DB.")
            return

        user_names = [u['name'] for u in users]
        selected_name = st.selectbox("Select Policy Holder", user_names)
        selected_user = next(u for u in users if u['name'] == selected_name)
        user_id = selected_user['user_id']
        
        # Display Mini Profile
        st.caption(f"**ID:** {user_id}")
        st.caption(f"**Plan:** {selected_user.get('policy_type', 'Standard')}")
        st.caption(f"**Vehicle:** {selected_user.get('vehicle', 'Unknown')}")
        
        st.divider()
        
    # Navigation Menu
    selected_page = option_menu(
            menu_title=None,
            options=["Dashboard", "New Trip Analysis", "Past Trip Analytics", "Premium & Policy"],
            icons=["speedometer", "cpu", "graph-up-arrow", "currency-dollar"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
    # Fetch Trips for this User
    trips = list(db.trips.find({"user_id": user_id}).sort("timestamp", -1))
    
    # --- PAGE 1: DASHBOARD (Trip Logs) ---
    if selected_page == "Dashboard":
        st.title(f"👋 Welcome, {selected_name}")
        st.markdown("### 📡 Real-Time Trip Activity Log")
        st.divider()

        # Metrics Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Trips", len(trips))
        total_dist = sum([t.get('total_distance_m', 0) for t in trips]) / 1000.0
        c2.metric("Total Distance", f"{total_dist:.1f} km")
        avg_score = np.mean([t.get('risk_label', 0.5) for t in trips if t.get('risk_label') is not None]) if trips else 0.5
        c3.metric("Avg Risk Score", f"{avg_score:.2f}")

        st.subheader("Recent Activity")
        if not trips:
            st.info("No activity logged yet.")
        else:
            summary_data = []
            for t in trips:
                ts = t.get('timestamp', 0)
                date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
                risk = t.get('risk_label')
                
                if risk is not None:
                    verdict, icon, _ = get_risk_verdict(risk)
                    display_status = f"{verdict} {icon}"
                    score_disp = f"{risk:.4f}"
                else:
                    display_status = "⏳ Pending Analysis"
                    score_disp = "---"

                summary_data.append({
                    "Date": date_str,
                    "Trip ID": t['trip_id'],
                    "Distance (km)": f"{t.get('total_distance_m',0)/1000:.1f}",
                    "Risk Score": score_disp,
                    "Status": display_status
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    # --- PAGE 2: NEW TRIP ANALYSIS (Pending Actions) ---
    elif selected_page == "New Trip Analysis":
        st.title("⚡ AI Risk Assessment Center")
        st.caption("Process raw telemetry data from vehicle to generate risk insights.")
        st.divider()

        pending_trips = [t for t in trips if t.get('risk_label') is None]

        if not pending_trips:
            st.success("✅ All trips have been processed! No pending actions.")
        else:
            st.info(f"You have {len(pending_trips)} pending trips to analyze.")
            
            # Selector
            trip_options = {f"{t['trip_id']} ({datetime.datetime.fromtimestamp(t.get('timestamp',0)).strftime('%H:%M')})": t for t in pending_trips}
            selected_option_id = st.selectbox("Select Pending Trip", list(trip_options.keys()))
            target_trip = trip_options[selected_option_id]

            if st.button("🚀 Run AI Analysis", type="primary"):
                 with st.spinner("Running LSTM Model & Explainability Engine..."):
                    try:
                        model, scaler = load_ai_model()
                        # 1. Prediction
                        score = analyze_trip_ai(target_trip, model, scaler)
                        # 2. Explanation
                        explanation_df = generate_trip_explanation(target_trip)
                        # 3. Update DB
                        db.trips.update_one({"_id": target_trip["_id"]}, {"$set": {"risk_label": score}})
                        
                        st.success(f"Analysis Complete! Risk Score: {score:.4f}")
                        verdict, icon, adj = get_risk_verdict(score)
                        st.metric("Verdict", f"{verdict} {icon}", delta=adj)

                        # Show Graph Immediately
                        render_trip_analysis(target_trip)
                        
                    except Exception as e:
                        st.error(f"Analysis Failed: {str(e)}")

    # --- PAGE 3: PAST TRIP ANALYTICS (Detailed Graph View) ---
    elif selected_page == "Past Trip Analytics":
        st.title("🔍 Historical Forensics")
        st.caption("Deep dive into past driving behavior and anomalies.")
        st.divider()

        processed_trips = [t for t in trips if t.get('risk_label') is not None]
        
        if not processed_trips:
            st.warning("No processed trips available. Please analyze a trip first.")
        else:
            trip_options = {f"{t['trip_id']} | Risk: {t['risk_label']:.2f} | {datetime.datetime.fromtimestamp(t.get('timestamp',0)).strftime('%Y-%m-%d %H:%M')}": t for t in processed_trips}
            selected_option = st.selectbox("Select Historical Trip", list(trip_options.keys()))
            
            if selected_option:
                selected_trip = trip_options[selected_option]
                render_trip_analysis(selected_trip)

    # --- PAGE 4: PREMIUM & POLICY (Financials) ---
    elif selected_page == "Premium & Policy":
        st.title("💰 Smart Premium Calculator")
        st.markdown("### Personalized Billing based on AI Risk & Usage")
        st.divider()

        # 1. Policy Details Section
        st.subheader("📜 Policy Configuration")
        config = selected_user.get('policy_config', {})
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Base Plan Fee", f"₹{config.get('base_premium', 500)}")
        c2.metric("Distance Cap", f"{config.get('distance_cap_km', 1000)} km")
        c3.metric("Overage Rate", f"₹{config.get('rate_overage', 10)}/km")
        c4.metric("Low Usage Limit", f"{config.get('low_usage_threshold', 100)} km")

        with st.expander("ℹ️ View Calculation Formula"):
            st.latex(r"P_{total} = P_{base} + (Dist \times Rate_{plan} \times (1 + Score^2))")
            st.markdown("""
            * **Logic:**
            * If `Distance < Low_Usage_Limit`: Apply Discount.
            * If `Distance > Cap`: Apply Overage Penalty Rate.
            * **Risk Multiplier:** $(1 + RiskScore^2)$ scales the usage cost.
            """)

        st.divider()

        # 2. Simulation Section
        st.subheader("💳 Current Month Bill Simulation")
        
        # In a real app, this would be sum of all trips this month.
        # For demo, let's select a trip to simulate adding it to the bill.
        latest_trip = trips[0] if trips else None
        
        if latest_trip:
            # We use the latest trip's distance + risk for the simulation
            sim_dist = latest_trip.get('total_distance_m', 0) / 1000.0
            sim_score = latest_trip.get('risk_label', 0.5)
            
            if sim_score is None: sim_score = 0.5 # Handle pending

            # Calculate
            financials = calculate_personalized_premium(selected_user, sim_score, sim_dist)

            # Display Bill
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.info(f"**Simulation Context**\n\nTrip Dist: {sim_dist:.2f} km\n\nRisk Score: {sim_score:.4f}")
            
            with col2:
                b1, b2, b3 = st.columns(3)
                b1.metric("Base Fee", f"₹{financials['Base Fee']}")
                b2.metric("Usage Cost", f"₹{financials['Variable Cost']}")
                b3.metric("Total Estimate", f"₹{financials['Total Premium']}", delta=financials['Status Note'])
                
                st.caption(f"Applied Risk Multiplier: **{financials['Risk Multiplier']}x**")
        else:
            st.warning("No trip data available to simulate bill.")

if __name__ == "__main__":
    main()
