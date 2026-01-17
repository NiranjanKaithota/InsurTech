from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import pygame
import os
import random
import time
import math
import threading
import json
import pymongo
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

DATA_DIR = "data/web_trips"
os.makedirs(DATA_DIR, exist_ok=True)

# Simulation Constants
TRIP_DURATION = 120
TIMESTEP = 0.05      
CAR_MASS = 1500.0
MAX_ENGINE_FORCE = 8500.0     
MAX_BRAKE_FORCE = 9000.0
DRAG_COEFFICIENT = 0.5
ROLLING_RESISTANCE = 250.0
AIR_DENSITY = 1.325
CAR_FRONTAL_AREA = 2.2

SPEED_ZONES = {"residential": 30, "main_road": 60, "highway": 80}

# --- GLOBAL STATE ---
sim_state = {
    "active_user_id": None,
    "speed_kmh": 0.0,
    "speed_limit": 60,
    "throttle_input": 0.0,
    "brake_input": 0.0,
    "risk_status": "SAFE",
    "distance_m": 0.0,
    "time_remaining": TRIP_DURATION,
    "running": False,
    "paused": False,         # <--- NEW: Freezes physics
    "stop_command": None     # <--- NEW: 'save' or 'force'
}

web_controls = {"up": False, "down": False}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ubi_secret'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# --- HELPER FUNCTIONS ---
def create_trip_plan(duration):
    plan = []
    current_time = 0
    while current_time < duration:
        zone = random.choice(list(SPEED_ZONES.keys()))
        limit = SPEED_ZONES[zone]
        duration_zone = random.randint(20, 40)
        end_time = min(current_time + duration_zone, duration)
        plan.append({"start": current_time, "end": end_time, "limit": limit})
        current_time = end_time
    return plan

def get_current_limit(time_s, plan):
    for zone in plan:
        if zone["start"] <= time_s < zone["end"]:
            return zone["limit"]
    return plan[-1]["limit"]

def save_trip_data(user_id, data_log, trip_plan, total_distance):
    """Uploads data to MongoDB and saves locally."""
    trip_id = f"web_{int(time.time())}"
    
    trip_data = {
        "trip_id": trip_id,
        "user_id": user_id,
        "style": "web_simulator",
        "risk_label": None,
        "trip_plan": trip_plan,
        "sequence": data_log,
        "timestamp": time.time(),
        "total_distance_m": total_distance
    }

    print(f"💾 Saving trip {trip_id}...")

    if MONGO_URI:
        try:
            client = pymongo.MongoClient(MONGO_URI)
            db = client["ubi_database"]
            db.trips.insert_one(trip_data)
            print("✅ SUCCESS: Uploaded to MongoDB!")
            socketio.emit('upload_status', {'status': 'success', 'msg': 'Trip Log Saved & Uploaded!'})
        except Exception as e:
            print(f"❌ Cloud Upload Failed: {e}")
            socketio.emit('upload_status', {'status': 'error', 'msg': 'Saved locally (Cloud Error).'})
    else:
        socketio.emit('upload_status', {'status': 'warning', 'msg': 'Saved locally (No Cloud URI).'})

    try:
        local_path = os.path.join(DATA_DIR, f"{trip_id}.json")
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(trip_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Local Save Failed: {e}")

def reset_simulation():
    sim_state["speed_kmh"] = 0.0
    sim_state["throttle_input"] = 0.0
    sim_state["brake_input"] = 0.0
    sim_state["risk_status"] = "SAFE"
    sim_state["distance_m"] = 0.0
    sim_state["time_remaining"] = TRIP_DURATION
    sim_state["running"] = False
    sim_state["paused"] = False
    sim_state["stop_command"] = None

def run_physics_loop():
    print("--- Physics Engine Ready ---")
    pygame.init()
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    clock = pygame.time.Clock()

    current_speed_ms = 0.0
    current_time_s = 0.0
    throttle_hold_time = 0.0
    brake_hold_time = 0.0
    data_log = []
    trip_plan = []

    while True:
        if sim_state["running"] and sim_state["active_user_id"]:
            
            # --- PAUSE LOGIC ---
            if sim_state["paused"]:
                # If paused, we check if the user sent a stop command, otherwise just wait
                if sim_state["stop_command"] == 'force':
                     print("--- Force Quit Triggered ---")
                     reset_simulation()
                     sim_state["active_user_id"] = None
                     current_time_s = 0.0
                     socketio.emit('trip_completed', {'aborted': True})
                
                elif sim_state["stop_command"] == 'save':
                     print("--- Save & Quit Triggered ---")
                     save_trip_data(sim_state["active_user_id"], data_log, trip_plan, sim_state["distance_m"])
                     reset_simulation()
                     sim_state["active_user_id"] = None
                     current_time_s = 0.0
                     # Frontend handles the alert via upload_status event

                time.sleep(0.1) # Idle while paused
                continue 
            # -------------------

            if current_time_s == 0.0:
                 trip_plan = create_trip_plan(TRIP_DURATION)
                 data_log = []

            # 1. Input Handling (Smoothed)
            is_accel = web_controls["up"]
            is_brake = web_controls["down"]

            if is_accel:
                sim_state["throttle_input"] = min(1.0, sim_state["throttle_input"] + 0.02)
            else:
                sim_state["throttle_input"] = max(0.0, sim_state["throttle_input"] - 0.01)

            if is_brake:
                sim_state["brake_input"] = min(1.0, sim_state["brake_input"] + 0.04)
            else:
                sim_state["brake_input"] = max(0.0, sim_state["brake_input"] - 0.05)

            # 2. Physics
            limit = get_current_limit(current_time_s, trip_plan)
            sim_state["speed_limit"] = limit

            force_engine = sim_state["throttle_input"] * MAX_ENGINE_FORCE
            force_brake = sim_state["brake_input"] * MAX_BRAKE_FORCE
            resistance = ROLLING_RESISTANCE if current_speed_ms > 0 else 0
            force_drag = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * CAR_FRONTAL_AREA * (current_speed_ms ** 2)
            
            net_force = force_engine - force_brake - force_drag - resistance
            accel = net_force / CAR_MASS

            current_speed_ms += accel * TIMESTEP
            
            MAX_SPEED_KMH = 180.0
            MAX_SPEED_MS = MAX_SPEED_KMH / 3.6
            current_speed_ms = max(0, min(current_speed_ms, MAX_SPEED_MS))
            
            sim_state["speed_kmh"] = current_speed_ms * 3.6
            sim_state["distance_m"] += current_speed_ms * TIMESTEP
            
            # 3. Logging
            is_speeding = 1 if sim_state["speed_kmh"] > (limit + 2) else 0
            
            log_entry = {
                "time": round(current_time_s, 2),
                "speed": round(sim_state["speed_kmh"], 2),
                "acceleration": round(accel, 2),
                "speed_limit": limit,
                "is_speeding": is_speeding,
                "throttle": round(sim_state["throttle_input"], 2),
                "brake": round(sim_state["brake_input"], 2),
                "distance": round(sim_state["distance_m"], 2)
            }
            data_log.append(log_entry)

            # 4. Risk Status
            if sim_state["speed_kmh"] <= limit:
                 sim_state["risk_status"] = "SAFE"
            elif sim_state["speed_kmh"] <= limit + 10:
                 sim_state["risk_status"] = "MODERATE"
            else:
                 sim_state["risk_status"] = "DANGEROUS"

            sim_state["time_remaining"] = max(0, int(TRIP_DURATION - current_time_s))

            socketio.emit('update_state', sim_state)

            current_time_s += TIMESTEP

            # Check Finish
            if current_time_s >= TRIP_DURATION:
                save_trip_data(sim_state["active_user_id"], data_log, trip_plan, sim_state["distance_m"])
                reset_simulation()
                sim_state["active_user_id"] = None
                current_time_s = 0.0
                socketio.emit('trip_completed', {'aborted': False})

        else:
            time.sleep(0.1)

        clock.tick(int(1 / TIMESTEP))

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_trip')
def handle_start_trip(data):
    reset_simulation()
    sim_state["active_user_id"] = data.get('user_id', 'u_guest')
    sim_state["running"] = True

@socketio.on('control_input')
def handle_control(data):
    key = data.get('key')
    pressed = data.get('pressed')
    if key in web_controls:
        web_controls[key] = pressed

# --- NEW EVENT HANDLERS FOR PAUSE MENU ---
@socketio.on('pause_toggle')
def handle_pause(data):
    # Toggle pause state based on frontend request
    should_pause = data.get('paused', False)
    sim_state["paused"] = should_pause
    print(f"Simulation Paused: {should_pause}")

@socketio.on('driver_choice')
def handle_choice(data):
    choice = data.get('choice') # 'save' or 'force'
    sim_state["stop_command"] = choice

if __name__ == '__main__':
    sim_thread = threading.Thread(target=run_physics_loop)
    sim_thread.daemon = True 
    sim_thread.start()
    print("🚀 Web Simulator running on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)