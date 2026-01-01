import pygame
import numpy as np
import json
import os
import random
import time
import math
import pymongo 
import os
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Load secrets from .env file
load_dotenv()

# Get the value
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found! Make sure .env file exists.")

DATA_DIR = "data/raw_human"
TRIP_DURATION = 120          
TIMESTEP = 0.1               
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

# --- 2. Physics Constants (TUNED) ---
CAR_MASS = 1500.0             
MAX_ENGINE_FORCE = 6500.0     
MAX_BRAKE_FORCE = 9000.0      
DRAG_COEFFICIENT = 0.5     
ROLLING_RESISTANCE = 200.0    
AIR_DENSITY = 1.225
CAR_FRONTAL_AREA = 2.2

# --- 3. Speed Zones ---
SPEED_ZONES = {
    "residential": 30,
    "main_road": 60,
    "highway": 80
}

os.makedirs(DATA_DIR, exist_ok=True)

# --- 4. Helper Functions ---
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

# --- 5. UI Drawing Functions ---
def draw_dashboard(screen, speed, speed_limit, throttle, brake):
    # Colors
    c_bg = (20, 20, 30)
    c_gauge_bg = (40, 40, 50)
    c_accent = (0, 200, 255)
    c_danger = (255, 50, 50)
    c_text = (255, 255, 255)
    
    # Speedometer
    center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150)
    radius = 120
    
    # Background
    pygame.draw.circle(screen, c_gauge_bg, center, radius, 0)
    pygame.draw.circle(screen, c_bg, center, radius - 20, 0)
    
    # Needle Logic
    max_disp_speed = 140
    angle_start = 225
    angle_end = -45
    angle_range = angle_start - angle_end
    speed_angle = angle_start - (min(speed, max_disp_speed) / max_disp_speed) * angle_range
    rad_angle = math.radians(speed_angle)
    end_pos = (center[0] + (radius - 10) * math.cos(-rad_angle), 
               center[1] + (radius - 10) * math.sin(-rad_angle))
    pygame.draw.line(screen, c_accent, center, end_pos, 4)
    
    # Text
    font_large = pygame.font.SysFont("consolas", 60, bold=True)
    txt_speed = font_large.render(f"{int(speed)}", True, c_text)
    screen.blit(txt_speed, (center[0] - txt_speed.get_width()//2, center[1] - 40))
    font_small = pygame.font.SysFont("consolas", 20)
    txt_unit = font_small.render("km/h", True, (150, 150, 150))
    screen.blit(txt_unit, (center[0] - txt_unit.get_width()//2, center[1] + 20))

    # Speed Limit Sign
    sign_pos = (center[0] + 180, center[1] - 50)
    pygame.draw.circle(screen, (200, 200, 200), sign_pos, 45)
    pygame.draw.circle(screen, (200, 0, 0), sign_pos, 45, 8)
    txt_limit = font_large.render(f"{speed_limit}", True, (0, 0, 0))
    screen.blit(txt_limit, (sign_pos[0] - txt_limit.get_width()//2, sign_pos[1] - txt_limit.get_height()//2))
    
    # Pedals
    bar_w, bar_h = 20, 150
    # Throttle
    t_rect_bg = pygame.Rect(SCREEN_WIDTH - 80, SCREEN_HEIGHT - 200, bar_w, bar_h)
    t_height = int(bar_h * throttle)
    t_rect_fg = pygame.Rect(SCREEN_WIDTH - 80, SCREEN_HEIGHT - 200 + (bar_h - t_height), bar_w, t_height)
    pygame.draw.rect(screen, c_gauge_bg, t_rect_bg)
    pygame.draw.rect(screen, (0, 255, 100), t_rect_fg)
    
    # Brake
    b_rect_bg = pygame.Rect(SCREEN_WIDTH - 120, SCREEN_HEIGHT - 200, bar_w, bar_h)
    b_height = int(bar_h * brake)
    b_rect_fg = pygame.Rect(SCREEN_WIDTH - 120, SCREEN_HEIGHT - 200 + (bar_h - b_height), bar_w, b_height)
    pygame.draw.rect(screen, c_gauge_bg, b_rect_bg)
    pygame.draw.rect(screen, c_danger, b_rect_fg)

def draw_scrolling_road(screen, speed, frame_count):
    c_road = (40, 40, 40)
    horizon_y = SCREEN_HEIGHT // 2 - 50
    
    pygame.draw.rect(screen, (10, 10, 15), (0, 0, SCREEN_WIDTH, horizon_y))
    pygame.draw.rect(screen, c_road, (0, horizon_y, SCREEN_WIDTH, SCREEN_HEIGHT))
    
    offset = (frame_count * (speed * 0.5)) % 100 
    center_x = SCREEN_WIDTH // 2
    
    pygame.draw.line(screen, (100, 100, 100), (center_x, horizon_y), (0, SCREEN_HEIGHT), 2)
    pygame.draw.line(screen, (100, 100, 100), (center_x, horizon_y), (SCREEN_WIDTH, SCREEN_HEIGHT), 2)
    
    for i in range(10):
        y = horizon_y + (i * 40 + offset)
        if y > SCREEN_HEIGHT: y -= 400
        if y < horizon_y: continue
        
        dist = (y - horizon_y) / (SCREEN_HEIGHT - horizon_y)
        width = 20 + dist * 800
        x_start = center_x - width // 2
        
        if y > horizon_y:
            pygame.draw.line(screen, (80, 80, 80), (x_start, y), (x_start + width, y), 2)

def draw_live_graph(screen, data_points):
    rect = pygame.Rect(50, SCREEN_HEIGHT - 150, 300, 100)
    pygame.draw.rect(screen, (30, 30, 40), rect)
    pygame.draw.rect(screen, (100, 100, 100), rect, 1)
    
    if len(data_points) < 2: return
    view_data = data_points[-100:]
    max_speed = 120
    
    for i in range(1, len(view_data)):
        x1 = rect.left + ((i-1) / len(view_data)) * rect.width
        y1_spd = rect.bottom - (view_data[i-1]['speed'] / max_speed) * rect.height
        y1_lim = rect.bottom - (view_data[i-1]['speed_limit'] / max_speed) * rect.height
        
        x2 = rect.left + (i / len(view_data)) * rect.width
        y2_spd = rect.bottom - (view_data[i]['speed'] / max_speed) * rect.height
        y2_lim = rect.bottom - (view_data[i]['speed_limit'] / max_speed) * rect.height
        
        pygame.draw.line(screen, (150, 50, 50), (x1, y1_lim), (x2, y2_lim), 2)
        pygame.draw.line(screen, (0, 200, 255), (x1, y1_spd), (x2, y2_spd), 2)

# --- 6. Main Loop ---
def play_trip(trip_id="human_0", style="human", user_id="u_001"):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Driving DNA - User: {user_id}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24)

    # --- START SCREEN LOOP ---
    waiting_for_start = True
    start_btn_rect = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 30, 200, 60)
    
    while waiting_for_start:
        screen.fill((20, 20, 30))
        
        # Draw Title
        title_surf = pygame.font.SysFont("consolas", 50, bold=True).render("UBI DATA COLLECTOR", True, (255, 255, 255))
        screen.blit(title_surf, (SCREEN_WIDTH//2 - title_surf.get_width()//2, 150))
        
        user_surf = font.render(f"Driver ID: {user_id}", True, (0, 200, 255))
        screen.blit(user_surf, (SCREEN_WIDTH//2 - user_surf.get_width()//2, 220))
        
        # Draw Button
        pygame.draw.rect(screen, (0, 150, 0), start_btn_rect, border_radius=10)
        pygame.draw.rect(screen, (0, 255, 0), start_btn_rect, 3, border_radius=10)
        
        btn_text = font.render("START ENGINE", True, (255, 255, 255))
        screen.blit(btn_text, (start_btn_rect.centerx - btn_text.get_width()//2, start_btn_rect.centery - btn_text.get_height()//2))
        
        ins_surf = font.render("Controls: UP (Gas) | DOWN (Brake) | SPACE (Panic)", True, (150, 150, 150))
        screen.blit(ins_surf, (SCREEN_WIDTH//2 - ins_surf.get_width()//2, 400))

        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_btn_rect.collidepoint(event.pos):
                    waiting_for_start = False
    
    # --- SIMULATION LOOP ---
    current_speed_ms = 0.0
    current_time_s = 0.0
    trip_plan = create_trip_plan(TRIP_DURATION)
    data_points = []
    
    throttle_input = 0.0
    brake_input = 0.0
    
    throttle_hold_time = 0.0
    brake_hold_time = 0.0
    
    running = True
    frame_count = 0

    while running and current_time_s < TRIP_DURATION:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]: running = False
        
        # Exponential Inputs
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
            
        # Physics
        current_limit_kmh = get_current_limit(current_time_s, trip_plan)
        force_engine = throttle_input * MAX_ENGINE_FORCE
        force_brake = brake_input * MAX_BRAKE_FORCE
        resistance = ROLLING_RESISTANCE if current_speed_ms > 0 else 0
        force_drag = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * CAR_FRONTAL_AREA * (current_speed_ms ** 2)
        net_force = force_engine - force_brake - force_drag - resistance
        
        accel = net_force / CAR_MASS
        current_speed_ms += accel * TIMESTEP
        if current_speed_ms < 0: current_speed_ms = 0
        if current_speed_ms > 60: current_speed_ms = 60 
        current_speed_kmh = current_speed_ms * 3.6
        
        is_speeding = 1 if current_speed_kmh > (current_limit_kmh + 2) else 0
        
        data_points.append({
            "time": round(current_time_s, 2),
            "speed": round(current_speed_kmh, 2),
            "acceleration": round(accel, 2),
            "speed_limit": current_limit_kmh,
            "is_speeding": is_speeding,
            "throttle": round(throttle_input, 2),
            "brake": round(brake_input, 2),
        })
        
        # Drawing
        screen.fill((20, 20, 30))
        draw_scrolling_road(screen, current_speed_kmh, frame_count)
        draw_dashboard(screen, current_speed_kmh, current_limit_kmh, throttle_input, brake_input)
        draw_live_graph(screen, data_points)
        
        # Progress Bar
        prog = current_time_s / TRIP_DURATION
        pygame.draw.rect(screen, (0, 200, 255), (0, 0, int(SCREEN_WIDTH * prog), 5))
        
        pygame.display.flip()
        current_time_s += TIMESTEP
        frame_count += 1
        clock.tick(int(1.0/TIMESTEP))

    pygame.quit()
    
    # --- CLOUD UPLOAD ---
    print(f"Connecting to Cloud for user {user_id}...")
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
            "timestamp": time.time()
        }

        trips_col.insert_one(trip_data)
        print(f"✅ SUCCESS: Trip uploaded for {user_id}!")
        # Also save a local copy in data/raw_human
        try:
            # Ensure directory exists (already created at module load, but be safe)
            os.makedirs(DATA_DIR, exist_ok=True)
            # Include Mongo _id if available by re-querying the inserted document's id
            # Mongo returns an ObjectId on insert; we'll attach its string form
            # Note: insert_one doesn't assign _id to our local dict, so we can fetch last inserted
            # But safer is to use the result of insert_one; however we didn't capture it above.
            # To keep the change minimal, we will insert with result and update filename accordingly.
            # Re-insert with capturing result would duplicate data in DB; instead, attempt to get the
            # most recent trip with same trip_id and user_id and timestamp.
            inserted = trips_col.find_one({"trip_id": trip_id, "user_id": user_id}, sort=[("timestamp", -1)])
            if inserted and "_id" in inserted:
                trip_data_local = dict(trip_data)
                trip_data_local["_id"] = str(inserted["_id"]) 
            else:
                trip_data_local = trip_data

            local_path = os.path.join(DATA_DIR, f"{trip_id}.json")
            with open(local_path, "w", encoding="utf-8") as f:
                json.dump(trip_data_local, f, ensure_ascii=False, indent=2)

            print(f"💾 Saved local copy to {local_path}")
        except Exception as e_local:
            print(f"⚠️ WARNING: Could not save local copy. {e_local}")
    except Exception as e:
        print(f"❌ ERROR: Could not upload. {e}")

if __name__ == "__main__":
    print("\n--- SELECT DRIVER ---")
    print("Available Users: u_001 (Niranjan), u_002 (Iranna), u_003 (Rushil)")
    target_user = input("Enter User ID to drive as (default u_001): ").strip()
    if target_user == "": target_user = "u_001"
        
    t_id = f"human_{int(time.time())}"
    play_trip(trip_id=t_id, user_id=target_user)