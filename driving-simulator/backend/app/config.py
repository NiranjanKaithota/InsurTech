import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

TRIP_DURATION = 120
TIMESTEP = 0.1

CAR_MASS = 1500.0
MAX_ENGINE_FORCE = 6500.0
MAX_BRAKE_FORCE = 9000.0
DRAG_COEFFICIENT = 0.5
ROLLING_RESISTANCE = 200.0
AIR_DENSITY = 1.225
CAR_FRONTAL_AREA = 2.2

MAX_SPEED_KMH = 165.0
MAX_SPEED_MS = MAX_SPEED_KMH / 3.6

SPEED_ZONES = {
    "residential": 30,
    "main_road": 60,
    "highway": 80
}
