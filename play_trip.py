import pygame
import numpy as np
import json
import os
import random
import time
import math

# --- 1. Configuration ---
DATA_DIR = "data/raw_human"
TRIP_DURATION = 120          # Reduced to 2 mins for better exhibition flow
TIMESTEP = 0.1               # 10 Hz simulation
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600

# --- 2. Physics & Vehicle Constants ---
CAR_MASS = 1500.0             
MAX_ENGINE_FORCE = 3000.0     
MAX_BRAKE_FORCE = 8000.0      
DRAG_COEFFICIENT = 0.5
AIR_DENSITY = 1.225
CAR_FRONTAL_AREA = 2.2

# --- 3. Speed Zones ---
SPEED_ZONES = {
    "residential": 30,
    "main_road": 60,
    "highway": 80
}

os.makedirs(DATA_DIR, exist_ok=True)

# --- 4. Logic Functions ---
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

# --- 5. UI Drawing Helpers ---
def draw_dashboard(screen, speed, speed_limit, throttle, brake):
    # Colors
    c_bg = (20, 20, 30)
    c_gauge_bg = (40, 40, 50)
    c_accent = (0, 200, 255)
    c_danger = (255, 50, 50)
    c_text = (255, 255, 255)
    
    # 1. Draw Speedometer (Center Arc)
    center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150)
    radius = 120
    
    # Background Arc
    pygame.draw.circle(screen, c_gauge_bg, center, radius, 0)
    pygame.draw.circle(screen, c_bg, center, radius - 20, 0)
    
    # Speed Indicator
    max_disp_speed = 140
    angle_start = 225
    angle_end = -45
    angle_range = angle_start - angle_end
    
    speed_angle = angle_start - (min(speed, max_disp_speed) / max_disp_speed) * angle_range
    rad_angle = math.radians(speed_angle)
    
    # Needle
    end_pos = (center[0] + (radius - 10) * math.cos(-rad_angle), 
               center[1] + (radius - 10) * math.sin(-rad_angle))
    pygame.draw.line(screen, c_accent, center, end_pos, 4)
    
    # Text Speed
    font_large = pygame.font.SysFont("consolas", 60, bold=True)
    txt_speed = font_large.render(f"{int(speed)}", True, c_text)
    screen.blit(txt_speed, (center[0] - txt_speed.get_width()//2, center[1] - 40))
    
    font_small = pygame.font.SysFont("consolas", 20)
    txt_unit = font_small.render("km/h", True, (150, 150, 150))
    screen.blit(txt_unit, (center[0] - txt_unit.get_width()//2, center[1] + 20))

    # 2. Speed Limit Sign
    sign_pos = (center[0] + 180, center[1] - 50)
    pygame.draw.circle(screen, (200, 200, 200), sign_pos, 45) # White bg
    pygame.draw.circle(screen, (200, 0, 0), sign_pos, 45, 8)  # Red border
    txt_limit = font_large.render(f"{speed_limit}", True, (0, 0, 0))
    screen.blit(txt_limit, (sign_pos[0] - txt_limit.get_width()//2, sign_pos[1] - txt_limit.get_height()//2))
    
    # 3. Pedals (HUD style bars)
    bar_w = 20
    bar_h = 150
    
    # Throttle (Right side)
    t_rect_bg = pygame.Rect(SCREEN_WIDTH - 80, SCREEN_HEIGHT - 200, bar_w, bar_h)
    t_height = int(bar_h * throttle)
    t_rect_fg = pygame.Rect(SCREEN_WIDTH - 80, SCREEN_HEIGHT - 200 + (bar_h - t_height), bar_w, t_height)
    pygame.draw.rect(screen, c_gauge_bg, t_rect_bg)
    pygame.draw.rect(screen, (0, 255, 100), t_rect_fg)
    lbl_t = font_small.render("GAS", True, (150, 150, 150))
    screen.blit(lbl_t, (SCREEN_WIDTH - 85, SCREEN_HEIGHT - 40))

    # Brake (Left side of throttle)
    b_rect_bg = pygame.Rect(SCREEN_WIDTH - 120, SCREEN_HEIGHT - 200, bar_w, bar_h)
    b_height = int(bar_h * brake)
    b_rect_fg = pygame.Rect(SCREEN_WIDTH - 120, SCREEN_HEIGHT - 200 + (bar_h - b_height), bar_w, b_height)
    pygame.draw.rect(screen, c_gauge_bg, b_rect_bg)
    pygame.draw.rect(screen, c_danger, b_rect_fg)
    lbl_b = font_small.render("BRK", True, (150, 150, 150))
    screen.blit(lbl_b, (SCREEN_WIDTH - 125, SCREEN_HEIGHT - 40))

def draw_scrolling_road(screen, speed, frame_count):
    # Simple perspective grid
    c_road = (40, 40, 40)
    c_line = (255, 255, 255)
    
    horizon_y = SCREEN_HEIGHT // 2 - 50
    pygame.draw.rect(screen, (10, 10, 15), (0, 0, SCREEN_WIDTH, horizon_y)) # Sky
    pygame.draw.rect(screen, c_road, (0, horizon_y, SCREEN_WIDTH, SCREEN_HEIGHT)) # Ground
    
    # Moving stripes logic
    # Speed factor determines how fast lines move down
    offset = (frame_count * (speed * 0.5)) % 100 
    
    # Draw perspective lines
    center_x = SCREEN_WIDTH // 2
    pygame.draw.line(screen, (100, 100, 100), (center_x, horizon_y), (0, SCREEN_HEIGHT), 2)
    pygame.draw.line(screen, (100, 100, 100), (center_x, horizon_y), (SCREEN_WIDTH, SCREEN_HEIGHT), 2)
    
    # Draw horizontal moving lines
    for i in range(10):
        y = horizon_y + (i * 40 + offset)
        if y > SCREEN_HEIGHT: y -= 400
        if y < horizon_y: continue
        
        # Perspective width calculation
        dist = (y - horizon_y) / (SCREEN_HEIGHT - horizon_y)
        width = 20 + dist * 800
        x_start = center_x - width // 2
        
        # Only draw if below horizon
        if y > horizon_y:
            pygame.draw.line(screen, (80, 80, 80), (x_start, y), (x_start + width, y), 2)

def draw_live_graph(screen, data_points):
    # Draw a small rolling graph of speed vs limit at bottom left
    rect = pygame.Rect(50, SCREEN_HEIGHT - 150, 300, 100)
    pygame.draw.rect(screen, (30, 30, 40), rect)
    pygame.draw.rect(screen, (100, 100, 100), rect, 1)
    
    if len(data_points) < 2: return
    
    # Show last 100 points
    view_data = data_points[-100:]
    max_speed = 120
    
    for i in range(1, len(view_data)):
        x1 = rect.left + ((i-1) / len(view_data)) * rect.width
        y1_spd = rect.bottom - (view_data[i-1]['speed'] / max_speed) * rect.height
        y1_lim = rect.bottom - (view_data[i-1]['speed_limit'] / max_speed) * rect.height
        
        x2 = rect.left + (i / len(view_data)) * rect.width
        y2_spd = rect.bottom - (view_data[i]['speed'] / max_speed) * rect.height
        y2_lim = rect.bottom - (view_data[i]['speed_limit'] / max_speed) * rect.height
        
        # Draw Limit (Red line)
        pygame.draw.line(screen, (150, 50, 50), (x1, y1_lim), (x2, y2_lim), 2)
        # Draw Speed (Blue line)
        pygame.draw.line(screen, (0, 200, 255), (x1, y1_spd), (x2, y2_spd), 2)

# --- 6. Main Loop ---
def play_trip(trip_id="human_0", style="human"):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Driving DNA - Simulator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24)

    # --- Start Screen ---
    waiting = True
    while waiting:
        screen.fill((20, 20, 30))
        title = pygame.font.SysFont("consolas", 60, bold=True).render("DRIVER PROFILING AI", True, (0, 200, 255))
        sub = font.render("Physics-Based Data Collector", True, (150, 150, 150))
        ins = font.render("CONTROLS: Up (Gas) | Down (Brake) | Space (Panic)", True, (200, 200, 200))
        cta = font.render("PRESS ANY KEY TO START", True, (0, 255, 100))
        
        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 150))
        screen.blit(sub, (SCREEN_WIDTH//2 - sub.get_width()//2, 220))
        screen.blit(ins, (SCREEN_WIDTH//2 - ins.get_width()//2, 350))
        
        # Blink effect
        if time.time() % 1 > 0.5:
            screen.blit(cta, (SCREEN_WIDTH//2 - cta.get_width()//2, 450))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                waiting = False

    # --- Sim Setup ---
    current_speed_ms = 0.0
    current_time_s = 0.0
    trip_plan = create_trip_plan(TRIP_DURATION)
    data_points = []
    
    throttle_input = 0.0
    brake_input = 0.0
    
    # Physics State
    running = True
    frame_count = 0

    while running and current_time_s < TRIP_DURATION:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]: running = False
        
        # 2. Input Logic (Smoothness)
        if keys[pygame.K_UP]: throttle_input = min(1.0, throttle_input + 0.05)
        else: throttle_input = max(0.0, throttle_input - 0.05)
            
        if keys[pygame.K_DOWN]: brake_input = min(1.0, brake_input + 0.1)
        else: brake_input = max(0.0, brake_input - 0.05)
        
        if keys[pygame.K_SPACE]: 
            brake_input = 1.0
            throttle_input = 0.0
            
        # 3. Physics Calculation
        current_limit_kmh = get_current_limit(current_time_s, trip_plan)
        
        force_engine = throttle_input * MAX_ENGINE_FORCE
        force_brake = brake_input * MAX_BRAKE_FORCE
        force_drag = -0.75 * AIR_DENSITY * DRAG_COEFFICIENT * CAR_FRONTAL_AREA * (current_speed_ms ** 2)
        
        net_force = force_engine - force_brake + force_drag
        accel = net_force / CAR_MASS
        
        current_speed_ms += accel * TIMESTEP
        if current_speed_ms < 0: current_speed_ms = 0
        if current_speed_ms > 60: current_speed_ms = 60 # Cap speed
        
        current_speed_kmh = current_speed_ms * 3.6
        is_speeding = 1 if current_speed_kmh > (current_limit_kmh + 2) else 0
        
        # 4. Data Logging
        data_points.append({
            "time": round(current_time_s, 2),
            "speed": round(current_speed_kmh, 2),
            "acceleration": round(accel, 2),
            "speed_limit": current_limit_kmh,
            "is_speeding": is_speeding,
            "throttle": round(throttle_input, 2),
            "brake": round(brake_input, 2),
        })
        
        # 5. Drawing
        screen.fill((20, 20, 30))
        draw_scrolling_road(screen, current_speed_kmh, frame_count)
        draw_dashboard(screen, current_speed_kmh, current_limit_kmh, throttle_input, brake_input)
        draw_live_graph(screen, data_points)
        
        # Time Progress Bar
        pygame.draw.rect(screen, (50, 50, 50), (0, 0, SCREEN_WIDTH, 10))
        progress = current_time_s / TRIP_DURATION
        pygame.draw.rect(screen, (0, 200, 255), (0, 0, int(SCREEN_WIDTH * progress), 10))
        
        pygame.display.flip()
        
        current_time_s += TIMESTEP
        frame_count += 1
        clock.tick(int(1.0/TIMESTEP))

    pygame.quit()
    
    # --- Save Data ---
    trip_data = {
        "trip_id": trip_id,
        "style": style,
        "risk_label": None, 
        "trip_plan": trip_plan,
        "sequence": data_points
    }

    filename = os.path.join(DATA_DIR, f"{trip_id}.json")
    with open(filename, "w") as f:
        json.dump(trip_data, f, indent=4)
    print(f"Trip saved to {filename}")

if __name__ == "__main__":
    # Generate a unique ID based on timestamp
    t_id = f"human_{int(time.time())}"
    play_trip(trip_id=t_id)