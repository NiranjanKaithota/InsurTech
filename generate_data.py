import numpy as np
import pandas as pd
import json
import os
import random

# --- 1. Configuration ---
DATA_DIR = "data/raw"
NUM_TRIPS_PER_STYLE = 500   # Increased for a more stable model
TRIP_DURATION = 360         # seconds per trip
TIMESTEP = 1.0              # 1 second calculation interval

# --- 2. Physics & Vehicle Constants ---
CAR_MASS = 1500.0             # kg
MAX_ENGINE_FORCE = 3000.0     # Newtons
MAX_BRAKE_FORCE = 8000.0      # Newtons (brakes are stronger)
DRAG_COEFFICIENT = 0.3        # Aerodynamic drag coeff.
AIR_DENSITY = 1.225           # kg/m^3
CAR_FRONTAL_AREA = 2.2        # m^2

# --- 3. Speed Zone Definitions (km/h) ---
SPEED_ZONES = {
    "residential": 30,
    "main_road": 60,
    "highway": 80
}

# --- 4. Driver Profiles (THIS is where you control behavior) ---
DRIVER_PROFILES = {
    "safe": {
        "speed_bias_kmh": -2.0,   # tends to stay slightly under the limit
        "kp": 0.25,               # how strongly driver reacts to speed error
        "accel_comfort": 0.8,     # m/s^2 typical gentle accel
        "decel_comfort": 1.0,     # m/s^2 gentle braking
        "decel_harsh": 2.5,       # m/s^2 harsh braking cap
        "throttle_noise": 0.03,   # random variation
        "brake_noise": 0.03,
        "risk_label": 0.1,
    },
    "aggressive": {
        "speed_bias_kmh": 8.0,    # likes to drive above the limit
        "kp": 0.5,                # reacts more strongly to error
        "accel_comfort": 2.0,     # stronger accel
        "decel_comfort": 2.5,
        "decel_harsh": 4.5,
        "throttle_noise": 0.08,
        "brake_noise": 0.06,
        "risk_label": 0.9,
    }
}

# Ensure directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# --- 5. Trip Plan Functions ---
def create_trip_plan(duration):
    """Creates a random sequence of speed zones for a trip."""
    plan = []
    current_time = 0
    while current_time < duration:
        zone_name = random.choice(list(SPEED_ZONES.keys()))
        speed_limit = SPEED_ZONES[zone_name]
        zone_duration = random.randint(20, 40)
        end_time = min(current_time + zone_duration, duration)
        
        plan.append({
            "start": current_time,
            "end": end_time,
            "limit": speed_limit
        })
        current_time = end_time
    return plan

def get_current_limit(time, plan):
    """Finds the speed limit (in km/h) at a specific time."""
    for zone in plan:
        if zone["start"] <= time < zone["end"]:
            return zone["limit"]
    return plan[-1]["limit"]

# --- 6. Helper: Driver Controller ---
def compute_driver_inputs(
    current_speed_ms,
    speed_limit_kmh,
    profile
):
    """
    Compute throttle and brake based on a simple feedback controller,
    using tunable parameters in 'profile'.
    """
    # Target speed is speed limit plus driver bias
    target_speed_kmh = speed_limit_kmh + profile["speed_bias_kmh"]
    # Don't allow negative target speeds
    target_speed_kmh = max(0.0, target_speed_kmh)
    target_speed_ms = target_speed_kmh / 3.6

    # Speed error (m/s)
    error = target_speed_ms - current_speed_ms

    # Proportional controller: desired accel in m/s^2
    kp = profile["kp"]
    desired_accel = kp * error

    # Clamp desired accel to comfort / harsh limits
    max_accel = profile["accel_comfort"]   # positive
    max_decel = profile["decel_harsh"]     # positive magnitude

    if desired_accel > 0:
        # Accelerating, cap by comfort accel
        desired_accel = min(desired_accel, max_accel)
    else:
        # Braking, cap by harsh decel
        desired_accel = max(desired_accel, -max_decel)

    # Convert desired_accel to throttle / brake.
    # We ignore drag in this inverse mapping for simplicity.
    throttle_input = 0.0
    brake_input = 0.0

    if desired_accel >= 0:
        # a = F / m -> F = a * m
        required_force = desired_accel * CAR_MASS
        throttle_input = required_force / MAX_ENGINE_FORCE if MAX_ENGINE_FORCE > 0 else 0.0
    else:
        required_force = abs(desired_accel) * CAR_MASS
        brake_input = required_force / MAX_BRAKE_FORCE if MAX_BRAKE_FORCE > 0 else 0.0

    # Add some random noise so driving is not perfectly smooth
    throttle_input += np.random.normal(0.0, profile["throttle_noise"])
    brake_input += np.random.normal(0.0, profile["brake_noise"])

    # Clamp to [0, 1]
    throttle_input = float(np.clip(throttle_input, 0.0, 1.0))
    brake_input = float(np.clip(brake_input, 0.0, 1.0))

    return throttle_input, brake_input, desired_accel

# --- 7. Main Simulation Function ---
def generate_trip(style, trip_id):
    """
    Generates a trip using a physics and driver-agent simulation,
    now including input "jerkiness" for robust training.
    """
    time_steps = np.arange(TRIP_DURATION)
    data_points = []
    
    # Physics variables (use m/s for calculations)
    current_speed_ms = 0.0
    
    trip_plan = create_trip_plan(TRIP_DURATION)
    
    if style == 'aggressive':
        target_speed_factor = 1.25 
        throttle_gain = 1.0       
        brake_gain = 0.8          
        risk_score = 0.9
        input_noise_factor = 0.25 # Aggressive inputs are noisier/jerkier
    else: # 'safe'
        target_speed_factor = 0.95 
        throttle_gain = 0.4       
        brake_gain = 0.3          
        risk_score = 0.1
        input_noise_factor = 0.15 # Safe inputs have some keyboard noise, but less extreme

    for t in time_steps:
        current_speed_limit_kmh = get_current_limit(t, trip_plan)
        target_speed_ms = (current_speed_limit_kmh * target_speed_factor) / 3.6
        
        # --- 5a. Driver Agent Logic (Base Input) ---
        throttle_input = 0.0
        brake_input = 0.0
        error = target_speed_ms - current_speed_ms
        
        if error > 1.0: 
            throttle_input = min(1.0, error * throttle_gain) 
        elif error < -1.0: 
            brake_input = min(1.0, abs(error) * brake_gain) 

        # --- NEW: Introduce Keyboard Jerkiness/Noise ---
        # This simulates the high-frequency up/down of human tapping input.
        if np.random.random() < 0.3: # 30% chance of a random input "tap" event
            
            # Apply a random delta scaled by the factor
            noise_delta = np.random.uniform(-input_noise_factor, input_noise_factor)
            
            # Apply to throttle and clip
            if throttle_input > 0:
                throttle_input = np.clip(throttle_input + noise_delta, 0.0, 1.0)
            
            # Apply to brake and clip
            if brake_input > 0:
                brake_input = np.clip(brake_input + noise_delta, 0.0, 1.0)
        
        # --- 5b. Physics Engine Logic ---
        force_engine = throttle_input * MAX_ENGINE_FORCE
        force_brake = brake_input * MAX_BRAKE_FORCE
        force_drag = -0.75 * AIR_DENSITY * DRAG_COEFFICIENT * CAR_FRONTAL_AREA * (current_speed_ms ** 2)
        
        net_force = force_engine - force_brake + force_drag
        acceleration_ms2 = net_force / CAR_MASS
        
        current_speed_ms += acceleration_ms2 * TIMESTEP
        
        if current_speed_ms < 0: current_speed_ms = 0.0
        if current_speed_ms > 55.0: current_speed_ms = 55.0 

        # --- 5c. Store Data Point ---
        current_speed_kmh = current_speed_ms * 3.6
        current_speed_limit_kmh = get_current_limit(t, trip_plan) # Re-fetch limit for safety
        
        point = {
            "time": int(t),
            "speed": round(current_speed_kmh, 2),
            "acceleration": round(acceleration_ms2, 2), 
            "speed_limit": current_speed_limit_kmh,
            "is_speeding": 1 if current_speed_kmh > (current_speed_limit_kmh + 2) else 0,
            "throttle": round(throttle_input, 2),
            "brake": round(brake_input, 2)
        }
        data_points.append(point)

    trip_data = {
        "trip_id": trip_id,
        "style": style,
        "risk_label": risk_score,
        "trip_plan": trip_plan,
        "sequence": data_points
    }
    
    return trip_data

# --- 8. Main Execution ---
def main():
    print(f"Generating {NUM_TRIPS_PER_STYLE * 2} physics-simulated trips...")
    
    trip_count = 0
    
    for i in range(NUM_TRIPS_PER_STYLE):
        trip = generate_trip("safe", f"safe_{i}")
        filename = f"{DATA_DIR}/safe_{i}.json"
        with open(filename, "w") as f:
            json.dump(trip, f, indent=4)
        trip_count += 1
        
    for i in range(NUM_TRIPS_PER_STYLE):
        trip = generate_trip("aggressive", f"aggressive_{i}")
        filename = f"{DATA_DIR}/aggressive_{i}.json"
        with open(filename, "w") as f:
            json.dump(trip, f, indent=4)
        trip_count += 1

    print(f"Success! Generated {trip_count} trips in '{DATA_DIR}'")

if __name__ == "__main__":
    main()