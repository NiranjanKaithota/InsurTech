import numpy as np
import pandas as pd
import json
import os
import random

# --- 1. Configuration ---
DATA_DIR = "data/raw"
NUM_TRIPS_PER_STYLE = 500   # Increased for a more stable model
TRIP_DURATION = 360         # 2 minutes per trip

# --- 2. Physics & Vehicle Constants ---
CAR_MASS = 1500.0             # kg
MAX_ENGINE_FORCE = 3000.0     # Newtons
MAX_BRAKE_FORCE = 8000.0      # Newtons (brakes are stronger)
DRAG_COEFFICIENT = 0.3        # Aerodynamic drag coeff.
AIR_DENSITY = 1.225           # kg/m^3
CAR_FRONTAL_AREA = 2.2        # m^2
TIMESTEP = 1.0                # 1 second calculation interval

# --- 3. Speed Zone Definitions ---
SPEED_ZONES = {
    "residential": 30,
    "main_road": 60,
    "highway": 80
}

# Ensure directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# --- 4. Trip Plan Functions ---
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
        # current_time += 1
    return plan

def get_current_limit(time, plan):
    """Finds the speed limit (in km/h) at a specific time."""
    for zone in plan:
        if zone["start"] <= time < zone["end"]:
            return zone["limit"]
    return plan[-1]["limit"]

# --- 5. Main Simulation Function ---
def generate_trip(style, trip_id):
    """
    Generates a trip using a physics and driver-agent simulation.
    """
    time_steps = np.arange(TRIP_DURATION)
    data_points = []
    
    # Physics variables (use m/s for calculations)
    current_speed_ms = 0.0
    
    trip_plan = create_trip_plan(TRIP_DURATION)
    
    if style == 'aggressive':
        # Aggressive agent: high gain, low smoothness, high target speed
        target_speed_factor = 1.25 # 25% over limit
        throttle_gain = 1.0       # Slams the throttle
        brake_gain = 0.8          # Brakes hard
        risk_score = 0.9
    else: # 'safe'
        # Safe agent: low gain, high smoothness, compliant target speed
        target_speed_factor = 0.95 # 5% under limit
        throttle_gain = 0.4       # Smoothly applies throttle
        brake_gain = 0.3          # Smoothly applies brake
        risk_score = 0.1

    for t in time_steps:
        # Get current speed limit (km/h) and convert target to m/s
        current_speed_limit_kmh = get_current_limit(t, trip_plan)
        target_speed_ms = (current_speed_limit_kmh * target_speed_factor) / 3.6
        
        # --- 5a. Driver Agent Logic ---
        # Decides throttle and brake input (0.0 to 1.0)
        
        throttle_input = 0.0
        brake_input = 0.0
        
        # Calculate speed error
        error = target_speed_ms - current_speed_ms
        
        if error > 1.0: # Need to speed up
            throttle_input = min(1.0, error * throttle_gain) # Apply throttle
        elif error < -1.0: # Need to slow down
            brake_input = min(1.0, abs(error) * brake_gain) # Apply brake
        
        # --- 5b. Physics Engine Logic ---
        
        # Calculate forces
        force_engine = throttle_input * MAX_ENGINE_FORCE
        force_brake = brake_input * MAX_BRAKE_FORCE
        
        # F_drag = 0.5 * rho * C_d * A * v^2
        # Force is negative (opposes motion)
        force_drag = -0.5 * AIR_DENSITY * DRAG_COEFFICIENT * CAR_FRONTAL_AREA * (current_speed_ms ** 2)
        
        # F_net = F_engine - F_brake + F_drag
        net_force = force_engine - force_brake + force_drag
        
        # a = F / m
        acceleration_ms2 = net_force / CAR_MASS
        
        # v = u + at
        current_speed_ms += acceleration_ms2 * TIMESTEP
        
        # Constraints
        if current_speed_ms < 0: current_speed_ms = 0.0
        if current_speed_ms > 55.0: current_speed_ms = 55.0 # ~200 km/h cap

        # --- 5c. Store Data Point ---
        
        # Convert speed back to km/h for storage
        current_speed_kmh = current_speed_ms * 3.6
        
        point = {
            "time": int(t),
            "speed": round(current_speed_kmh, 2),
            "acceleration": round(acceleration_ms2, 2), # Store real accel (m/s^2)
            "speed_limit": current_speed_limit_kmh,
            "is_speeding": 1 if current_speed_kmh > (current_speed_limit_kmh + 2) else 0,
            # Store the driver inputs as features too
            "throttle": round(throttle_input, 2),
            "brake": round(brake_input, 2)
        }
        data_points.append(point)

    # --- 5d. Final Trip Object ---
    trip_data = {
        "trip_id": trip_id,
        "style": style,
        "risk_label": risk_score,
        "trip_plan": trip_plan,
        "sequence": data_points
    }
    
    return trip_data

# --- 6. Main Execution ---
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