import pygame
import numpy as np
import json
import os
import random
import time
import math
import pymongo 
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    # Fallback for testing if no .env exists
    print("WARNING: MONGO_URI not found. Data upload will be skipped.")
    MONGO_URI = None

DATA_DIR = "data/raw_human"
TRIP_DURATION = 120          
TIMESTEP = 0.1               
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

# --- PHYSICS CONSTANTS ---
CAR_MASS = 1500.0             
MAX_ENGINE_FORCE = 6500.0     
MAX_BRAKE_FORCE = 9000.0      
DRAG_COEFFICIENT = 0.5     
ROLLING_RESISTANCE = 200.0    
AIR_DENSITY = 1.225
CAR_FRONTAL_AREA = 2.2

# --- SPEED ZONES ---
SPEED_ZONES = {
    "residential": 30,
    "main_road": 60,
    "highway": 80
}

os.makedirs(DATA_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---
def create_trip_plan(duration):
    plan = []
    current_time = 0
    while current_time < duration:
        zone_name = random.choice(list(SPEED_ZONES.keys()))
        speed_limit = SPEED_ZONES[zone_name]
        zone_duration = random.randint(20, 40)
        end_time = min(current_time + zone_duration, duration)
        plan.append({"start": current_time, "end": end_time, "limit": speed_limit})
        current_time = end_time
    return plan

def get_current_limit(time_s, plan):
    for zone in plan:
        if zone["start"] <= time_s < zone["end"]:
            return zone["limit"]
    return plan[-1]["limit"]

# --- HILL CLIMB UI FUNCTIONS ---

def draw_car(screen, x, y, throttle, brake, time_elapsed):
    """
    Draws a 2D side-view car (Jeep style).
    """
    # Bobbing effect based on throttle (simulates engine vibration/suspension)
    bob_offset = math.sin(time_elapsed * 10) * 2 if throttle > 0 else 0
    
    # Body coordinates
    car_body_color = (200, 50, 50) # Red Jeep
    roll_cage_color = (50, 50, 50)
    
    # Main Chassis
    chassis_rect = pygame.Rect(x, y + bob_offset, 140, 50)
    pygame.draw.rect(screen, car_body_color, chassis_rect, border_radius=10)
    
    # Top / Cabin
    cabin_points = [
        (x + 20, y + bob_offset), 
        (x + 40, y - 40 + bob_offset), 
        (x + 110, y - 40 + bob_offset), 
        (x + 120, y + bob_offset)
    ]
    pygame.draw.polygon(screen, roll_cage_color, cabin_points, 4)
    
    # Driver Head (Abstract)
    pygame.draw.circle(screen, (255, 200, 150), (x + 70, y - 10 + bob_offset), 12)
    
    # Wheels (rotating effect handled by simple spokes)
    wheel_radius = 28
    wheel_centers = [(x + 30, y + 50 + bob_offset), (x + 110, y + 50 + bob_offset)]
    
    wheel_rotation = -(time_elapsed * 1000) % 360 # Rotate based on time/speed roughly
    
    for cx, cy in wheel_centers:
        # Tire
        pygame.draw.circle(screen, (30, 30, 30), (cx, cy), wheel_radius)
        # Rim
        pygame.draw.circle(screen, (150, 150, 150), (cx, cy), wheel_radius - 8)
        
        # Spokes (visualize rotation)
        # We cheat rotation by just drawing lines that don't actually rotate in this simple func
        # unless we pass rotation angle. For simplicity, we just draw static spokes
        # To make them look like they are moving, we can offset them or just leave as solid.
        # Let's add a "brake glow" to wheels if braking
        if brake > 0:
            pygame.draw.circle(screen, (255, 100, 50), (cx, cy), 10) 

def draw_background(screen, scroll_x):
    """
    Draws parallax background layers.
    """
    # Sky
    screen.fill((135, 206, 235)) # Light Blue
    
    # Sun
    pygame.draw.circle(screen, (255, 255, 0), (SCREEN_WIDTH - 100, 80), 40)
    
    # Distant Hills (Slow scroll: 0.2x speed)
    hill_color = (34, 139, 34)
    hill_offset = int(scroll_x * 0.2) % SCREEN_WIDTH
    
    # Draw two sine waves for hills
    points = []
    for x in range(0, SCREEN_WIDTH + 100, 20):
        # Determine world X to calculate consistent height
        world_x = x + (scroll_x * 0.2)
        h = 200 + 50 * math.sin(world_x * 0.005) + 20 * math.sin(world_x * 0.02)
        points.append((x, SCREEN_HEIGHT - h))
    
    # Close polygon at bottom
    points.append((SCREEN_WIDTH, SCREEN_HEIGHT))
    points.append((0, SCREEN_HEIGHT))
    
    if len(points) > 2:
        pygame.draw.polygon(screen, (100, 160, 100), points)

def draw_terrain(screen, scroll_x):
    """
    Draws the foreground road/ground (Fast scroll: 1.0x speed)
    """
    ground_y = SCREEN_HEIGHT - 100
    
    # Draw Ground Block
    pygame.draw.rect(screen, (101, 67, 33), (0, ground_y, SCREEN_WIDTH, 100)) # Dirt brown
    pygame.draw.rect(screen, (50, 200, 50), (0, ground_y, SCREEN_WIDTH, 15)) # Grass top
    
    # Draw road markers or texture to show speed
    marker_spacing = 200
    offset = int(scroll_x) % marker_spacing
    
    for x in range(-offset, SCREEN_WIDTH, marker_spacing):
        # Little stones or grass tufts
        pygame.draw.circle(screen, (80, 50, 20), (x + 50, ground_y + 40), 5)
        pygame.draw.circle(screen, (90, 60, 30), (x + 120, ground_y + 70), 8)

def draw_hud(screen, speed, speed_limit, throttle, brake, distance, time_s):
    """
    Overlay HUD similar to mobile games.
    """
    font_large = pygame.font.SysFont("consolas", 40, bold=True)
    font_small = pygame.font.SysFont("consolas", 20, bold=True)
    
    # --- Top Left: Gauges ---
    # RPM/Speed container
    pygame.draw.rect(screen, (0, 0, 0, 150), (20, 20, 220, 100), border_radius=10)
    
    # Speed Text
    col_speed = (255, 255, 255)
    if speed > speed_limit + 5: col_speed = (255, 50, 50) # Red if speeding
    
    lbl_speed = font_large.render(f"{int(speed)}", True, col_speed)
    lbl_unit = font_small.render("km/h", True, (200, 200, 200))
    screen.blit(lbl_speed, (40, 30))
    screen.blit(lbl_unit, (40 + lbl_speed.get_width() + 10, 50))
    
    # Distance
    lbl_dist = font_small.render(f"Dist: {int(distance)} m", True, (255, 255, 0))
    screen.blit(lbl_dist, (40, 80))

    # --- Top Center: Speed Limit ---
    # Draw a road sign
    sign_x = SCREEN_WIDTH // 2
    sign_y = 60
    pygame.draw.circle(screen, (240, 240, 240), (sign_x, sign_y), 40)
    pygame.draw.circle(screen, (200, 0, 0), (sign_x, sign_y), 40, 8)
    lbl_limit = font_large.render(f"{speed_limit}", True, (0, 0, 0))
    screen.blit(lbl_limit, (sign_x - lbl_limit.get_width()//2, sign_y - lbl_limit.get_height()//2))
    
    # --- Bottom Right: Pedals ---
    # Gas
    gas_rect = pygame.Rect(SCREEN_WIDTH - 100, SCREEN_HEIGHT - 120, 60, 100)
    gas_pressed_h = int(100 * throttle)
    pygame.draw.rect(screen, (50, 50, 50), gas_rect, border_radius=5) # Background
    pygame.draw.rect(screen, (0, 255, 0), (gas_rect.x, gas_rect.y + (100-gas_pressed_h), 60, gas_pressed_h), border_radius=5)
    
    lbl_gas = font_small.render("GAS", True, (255, 255, 255))
    screen.blit(lbl_gas, (gas_rect.centerx - lbl_gas.get_width()//2, gas_rect.bottom + 5))

    # Brake
    brake_rect = pygame.Rect(SCREEN_WIDTH - 180, SCREEN_HEIGHT - 120, 60, 100)
    brake_pressed_h = int(100 * brake)
    pygame.draw.rect(screen, (50, 50, 50), brake_rect, border_radius=5)
    pygame.draw.rect(screen, (255, 0, 0), (brake_rect.x, brake_rect.y + (100-brake_pressed_h), 60, brake_pressed_h), border_radius=5)

    lbl_brake = font_small.render("BRK", True, (255, 255, 255))
    screen.blit(lbl_brake, (brake_rect.centerx - lbl_brake.get_width()//2, brake_rect.bottom + 5))
    
    # --- Live Graph (Miniaturized at bottom left) ---
    # (Optional, kept minimal)
    # pygame.draw.rect(screen, (0,0,0), (20, SCREEN_HEIGHT - 80, 200, 60))

# --- MAIN LOOP ---
def play_trip(trip_id="human_0", style="human", user_id="u_001"):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Driving DNA - Hill Climb Mode - User: {user_id}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24)

    # --- START SCREEN ---
    waiting_for_start = True
    start_btn_rect = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 30, 200, 60)
    
    while waiting_for_start:
        screen.fill((50, 150, 200)) # Nice blue
        
        title_surf = pygame.font.SysFont("consolas", 50, bold=True).render("UBI HILL CLIMBER", True, (255, 255, 255))
        screen.blit(title_surf, (SCREEN_WIDTH//2 - title_surf.get_width()//2, 150))
        
        user_surf = font.render(f"Driver ID: {user_id}", True, (255, 255, 0))
        screen.blit(user_surf, (SCREEN_WIDTH//2 - user_surf.get_width()//2, 220))
        
        pygame.draw.rect(screen, (0, 180, 0), start_btn_rect, border_radius=10)
        pygame.draw.rect(screen, (255, 255, 255), start_btn_rect, 3, border_radius=10)
        btn_text = font.render("START TRIP", True, (255, 255, 255))
        screen.blit(btn_text, (start_btn_rect.centerx - btn_text.get_width()//2, start_btn_rect.centery - btn_text.get_height()//2))
        
        ins_surf = font.render("Controls: UP (Gas) | DOWN (Brake)", True, (240, 240, 240))
        screen.blit(ins_surf, (SCREEN_WIDTH//2 - ins_surf.get_width()//2, 400))

        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_btn_rect.collidepoint(event.pos): waiting_for_start = False
    
    # --- SIMULATION VARS ---
    current_speed_ms = 0.0
    current_time_s = 0.0
    distance_m = 0.0
    
    trip_plan = create_trip_plan(TRIP_DURATION)
    data_points = []
    
    throttle_input = 0.0
    brake_input = 0.0
    throttle_hold_time = 0.0
    brake_hold_time = 0.0
    
    # View Scroll Offset (pixels)
    scroll_x_pixels = 0.0
    
    running = True

    while running and current_time_s < TRIP_DURATION:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]: running = False
        
        # --- INPUT HANDLING ---
        if keys[pygame.K_UP]:
            throttle_hold_time += TIMESTEP
            step = 0.05 + (0.15 * min(1.0, throttle_hold_time / 2.0))
            throttle_input = min(1.0, throttle_input + step)
        else:
            throttle_hold_time = 0.0 
            throttle_input = max(0.0, throttle_input - 0.15) 
            
        if keys[pygame.K_DOWN]:
            brake_hold_time += TIMESTEP
            step = 0.1 + (0.25 * min(1.0, brake_hold_time / 1.0))
            brake_input = min(1.0, brake_input + step)
        else:
            brake_hold_time = 0.0
            brake_input = max(0.0, brake_input - 0.15)
        
        if keys[pygame.K_SPACE]: 
            brake_input = 1.0; throttle_input = 0.0
            
        # --- PHYSICS ---
        current_limit_kmh = get_current_limit(current_time_s, trip_plan)
        
        force_engine = throttle_input * MAX_ENGINE_FORCE
        force_brake = brake_input * MAX_BRAKE_FORCE
        resistance = ROLLING_RESISTANCE if current_speed_ms > 0 else 0
        force_drag = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * CAR_FRONTAL_AREA * (current_speed_ms ** 2)
        net_force = force_engine - force_brake - force_drag - resistance
        
        accel = net_force / CAR_MASS
        current_speed_ms += accel * TIMESTEP
        
        MAX_SPEED_KMH = 165.0
        MAX_SPEED_MS = MAX_SPEED_KMH / 3.6
        if current_speed_ms < 0: current_speed_ms = 0
        elif current_speed_ms > MAX_SPEED_MS: current_speed_ms = MAX_SPEED_MS
            
        current_speed_kmh = current_speed_ms * 3.6
        
        # --- DISTANCE & SCROLL CALC ---
        # Distance calculation (meters)
        distance_step = current_speed_ms * TIMESTEP
        distance_m += distance_step
        
        # Scroll calculation (pixels for visual)
        # 1 meter = 10 pixels roughly for visual speed effect
        scroll_x_pixels += distance_step * 10 
        
        # Data Recording
        is_speeding = 1 if current_speed_kmh > (current_limit_kmh + 2) else 0
        data_points.append({
            "time": round(current_time_s, 2),
            "speed": round(current_speed_kmh, 2),
            "acceleration": round(accel, 2),
            "speed_limit": current_limit_kmh,
            "is_speeding": is_speeding,
            "throttle": round(throttle_input, 2),
            "brake": round(brake_input, 2),
            "distance": round(distance_m, 2)
        })
        
        # --- DRAWING ---
        draw_background(screen, scroll_x_pixels)
        draw_terrain(screen, scroll_x_pixels)
        
        # Draw Car (Fixed X position: 200px from left)
        draw_car(screen, 200, SCREEN_HEIGHT - 135, throttle_input, brake_input, current_time_s)
        
        draw_hud(screen, current_speed_kmh, current_limit_kmh, throttle_input, brake_input, distance_m, current_time_s)
        
        # Progress Bar at very bottom
        prog = current_time_s / TRIP_DURATION
        pygame.draw.rect(screen, (0, 0, 0), (0, SCREEN_HEIGHT-10, SCREEN_WIDTH, 10))
        pygame.draw.rect(screen, (0, 255, 0), (0, SCREEN_HEIGHT-10, int(SCREEN_WIDTH * prog), 10))
        
        pygame.display.flip()
        current_time_s += TIMESTEP
        clock.tick(int(1.0/TIMESTEP))

    pygame.quit()
    
    # --- CLOUD UPLOAD ---
    print(f"Connecting to Cloud for user {user_id}...")
    if MONGO_URI:
        try:
            client = pymongo.MongoClient(MONGO_URI)
            db = client["ubi_database"]
            trips_col = db["trips"]
            
            trip_data = {
                "trip_id": trip_id,
                "user_id": user_id, 
                "style": style,
                "risk_label": None, 
                "trip_plan": trip_plan,
                "sequence": data_points,
                "timestamp": time.time(),
                "total_distance_m": distance_m
            }

            trips_col.insert_one(trip_data)
            print(f"✅ SUCCESS: Trip uploaded for {user_id}!")
            
            # Save Local
            try:
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
    print("\n--- SELECT DRIVER (Hill Climb Mode) ---")
    print("Available Users: u_001 (Niranjan), u_002 (Iranna), u_003 (Rushil)")
    target_user = input("Enter User ID to drive as (default u_001): ").strip()
    if target_user == "": target_user = "u_001"
        
    t_id = f"human_{int(time.time())}"
    play_trip(trip_id=t_id, user_id=target_user)