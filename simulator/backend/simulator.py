import pygame
import json
import os
import random
import time
import math
import pymongo
from dotenv import load_dotenv

# ===================== LIVE STATE (FOR FRONTEND API) =====================
latest_state = {
    "speed": 0.0,
    "speed_limit": 0,
    "throttle": 0.0,
    "brake": 0.0,
    "risk": "SAFE"
}

# Shared control state for API interactions
remote_controls = {
    "up": False,
    "down": False
}

# ===================== CONFIGURATION =====================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    print("WARNING: MONGO_URI not found. Data upload will be skipped.")
    MONGO_URI = None

DATA_DIR = "data/raw_human"
TRIP_DURATION = 120
TIMESTEP = 0.1
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

# ===================== PHYSICS CONSTANTS =====================
CAR_MASS = 1500.0
MAX_ENGINE_FORCE = 6500.0
MAX_BRAKE_FORCE = 9000.0
DRAG_COEFFICIENT = 0.5
ROLLING_RESISTANCE = 200.0
AIR_DENSITY = 1.225
CAR_FRONTAL_AREA = 2.2

# ===================== SPEED ZONES =====================
SPEED_ZONES = {
    "residential": 30,
    "main_road": 60,
    "highway": 80
}

os.makedirs(DATA_DIR, exist_ok=True)

# ===================== HELPER FUNCTIONS =====================
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

# ===================== UI FUNCTIONS (ORIGINAL VISUALS) =====================
def draw_dashboard(screen, speed, speed_limit, throttle, brake):
    c_bg = (20, 20, 30)
    c_gauge = (40, 40, 50)
    c_accent = (0, 200, 255)
    c_danger = (255, 50, 50)

    center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150)
    radius = 120

    # Draw Gauge Background
    pygame.draw.circle(screen, c_gauge, center, radius)
    pygame.draw.circle(screen, c_bg, center, radius - 20)

    # Draw Needle
    max_disp_speed = 140
    angle = 225 - (min(speed, max_disp_speed) / max_disp_speed) * 270
    rad = math.radians(angle)

    end = (
        center[0] + (radius - 10) * math.cos(-rad),
        center[1] + (radius - 10) * math.sin(-rad)
    )
    pygame.draw.line(screen, c_accent, center, end, 4)

    # Text Elements
    font_big = pygame.font.SysFont("consolas", 60, bold=True)
    screen.blit(font_big.render(str(int(speed)), True, (255, 255, 255)),
                (center[0] - 40, center[1] - 40))

    # Speed Limit Sign
    sign_pos = (center[0] + 180, center[1] - 50)
    pygame.draw.circle(screen, (200, 200, 200), sign_pos, 45)
    pygame.draw.circle(screen, (200, 0, 0), sign_pos, 45, 8)
    screen.blit(font_big.render(str(speed_limit), True, (0, 0, 0)),
                (sign_pos[0] - 30, sign_pos[1] - 35))

    # Pedals
    bar_h = 150
    # Throttle Bar
    pygame.draw.rect(screen, c_gauge, (SCREEN_WIDTH - 80, SCREEN_HEIGHT - 200, 20, bar_h))
    pygame.draw.rect(screen, (0, 255, 100),
                     (SCREEN_WIDTH - 80, SCREEN_HEIGHT - 200 + bar_h * (1 - throttle), 20, bar_h * throttle))

    # Brake Bar
    pygame.draw.rect(screen, c_gauge, (SCREEN_WIDTH - 120, SCREEN_HEIGHT - 200, 20, bar_h))
    pygame.draw.rect(screen, c_danger,
                     (SCREEN_WIDTH - 120, SCREEN_HEIGHT - 200 + bar_h * (1 - brake), 20, bar_h * brake))

def draw_scrolling_road(screen, speed, frame):
    horizon = SCREEN_HEIGHT // 2 - 50
    pygame.draw.rect(screen, (10, 10, 15), (0, 0, SCREEN_WIDTH, horizon))
    pygame.draw.rect(screen, (40, 40, 40), (0, horizon, SCREEN_WIDTH, SCREEN_HEIGHT))

    offset = (frame * speed * 0.4) % 40
    center_x = SCREEN_WIDTH // 2

    for i in range(15):
        y = horizon + i * 40 + offset
        if y < SCREEN_HEIGHT:
            width = 50 + (y - horizon) * 2
            pygame.draw.line(screen, (80, 80, 80),
                             (center_x - width, y),
                             (center_x + width, y), 2)

# ===================== MAIN SIMULATION =====================
def play_trip(trip_id="human_0", style="human", user_id="u_001"):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Driving DNA - Simulator Mode - {user_id}")
    clock = pygame.time.Clock()

    trip_plan = create_trip_plan(TRIP_DURATION)
    data_points = []

    # Physics State
    current_speed_ms = 0.0
    throttle_input = 0.0
    brake_input = 0.0
    distance_m = 0.0
    
    # Input Smoothing State
    throttle_hold_time = 0.0
    brake_hold_time = 0.0

    current_time_s = 0.0
    frame = 0
    running = True

    while running and current_time_s < TRIP_DURATION:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Optional: Allow mouse click start if needed, but not strictly required here

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
        
        # --- INPUT HANDLING (Exponential Logic from play_trip_new.py) ---
        # Checks both Keyboard AND Remote API controls
        is_accel = keys[pygame.K_UP] or remote_controls["up"]
        is_brake = keys[pygame.K_DOWN] or remote_controls["down"]

        # Throttle Logic
        if is_accel:
            throttle_hold_time += TIMESTEP
            step = 0.05 + (0.15 * min(1.0, throttle_hold_time / 2.0))
            throttle_input = min(1.0, throttle_input + step)
        else:
            throttle_hold_time = 0.0
            throttle_input = max(0.0, throttle_input - 0.15)

        # Brake Logic
        if is_brake:
            brake_hold_time += TIMESTEP
            step = 0.1 + (0.25 * min(1.0, brake_hold_time / 1.0))
            brake_input = min(1.0, brake_input + step)
        else:
            brake_hold_time = 0.0
            brake_input = max(0.0, brake_input - 0.15)

        # --- PHYSICS CALCULATION ---
        current_limit_kmh = get_current_limit(current_time_s, trip_plan)

        force_engine = throttle_input * MAX_ENGINE_FORCE
        force_brake = brake_input * MAX_BRAKE_FORCE
        
        resistance = ROLLING_RESISTANCE if current_speed_ms > 0 else 0
        force_drag = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * CAR_FRONTAL_AREA * (current_speed_ms ** 2)
        
        net_force = force_engine - force_brake - force_drag - resistance
        accel = net_force / CAR_MASS

        current_speed_ms += accel * TIMESTEP
        
        # Speed Clamping
        MAX_SPEED_KMH = 165.0
        MAX_SPEED_MS = MAX_SPEED_KMH / 3.6
        
        if current_speed_ms < 0: 
            current_speed_ms = 0
        elif current_speed_ms > MAX_SPEED_MS: 
            current_speed_ms = MAX_SPEED_MS

        current_speed_kmh = current_speed_ms * 3.6
        
        # Distance Calculation
        distance_step = current_speed_ms * TIMESTEP
        distance_m += distance_step

        # --- LIVE STATE UPDATE (For API) ---
        latest_state["speed"] = round(current_speed_kmh, 1)
        latest_state["speed_limit"] = current_limit_kmh
        latest_state["throttle"] = round(throttle_input * 100, 1)
        latest_state["brake"] = round(brake_input * 100, 1)

        if current_speed_kmh <= current_limit_kmh:
            latest_state["risk"] = "SAFE"
        elif current_speed_kmh <= current_limit_kmh + 10:
            latest_state["risk"] = "RISKY"
        else:
            latest_state["risk"] = "DANGEROUS"

        # --- DATA RECORDING ---
        is_speeding = 1 if current_speed_kmh > (current_limit_kmh + 2) else 0
        
        data_points.append({
            "time": round(current_time_s, 2),
            "speed": round(current_speed_kmh, 2),
            "acceleration": round(accel, 2), # Added acceleration
            "speed_limit": current_limit_kmh,
            "is_speeding": is_speeding,      # Added speeding flag
            "throttle": round(throttle_input, 2),
            "brake": round(brake_input, 2),
            "distance": round(distance_m, 2) # Added distance
        })

        # --- DRAWING ---
        screen.fill((20, 20, 30))
        draw_scrolling_road(screen, current_speed_kmh, frame)
        draw_dashboard(screen, current_speed_kmh, current_limit_kmh, throttle_input, brake_input)

        pygame.display.flip()
        current_time_s += TIMESTEP
        frame += 1
        clock.tick(int(1 / TIMESTEP))

    pygame.quit()

    # ===================== CLOUD & LOCAL SAVE =====================
    print(f"Connecting to Cloud for user {user_id}...")
    if MONGO_URI:
        try:
            client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db = client["ubi_database"]
            trips_col = db["trips"]
            
            trip_data = {
                "trip_id": trip_id,
                "user_id": user_id,
                "style": style, # simulator vs hill_climb
                "risk_label": None,
                "trip_plan": trip_plan,
                "sequence": data_points,
                "timestamp": time.time(),
                "total_distance_m": distance_m
            }
            
            # 1. Cloud Upload
            trips_col.insert_one(trip_data)
            print(f"✅ SUCCESS: Trip uploaded for {user_id}!")
            
            # 2. Local Save (Robust)
            try:
                # Retreive the inserted ID to keep local/cloud sync
                inserted = trips_col.find_one({"trip_id": trip_id, "user_id": user_id}, sort=[("timestamp", -1)])
                if inserted and "_id" in inserted:
                    trip_data["_id"] = str(inserted["_id"])
                
                local_path = os.path.join(DATA_DIR, f"{trip_id}.json")
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(trip_data, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved local copy to {local_path}")
            except Exception as e_local:
                print(f"⚠️ Local Save Warning: {e_local}")
                
        except Exception as e:
            print(f"❌ ERROR: Could not upload. {e}")
    else:
        print("Skipped Cloud Upload (No URI)")


if __name__ == "__main__":
    user = input("Enter User ID (default u_001): ").strip() or "u_001"
    play_trip(trip_id=f"sim_{int(time.time())}", style="simulator", user_id=user)