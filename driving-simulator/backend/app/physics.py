from .config import *

def update_physics(speed_ms, throttle, brake):
    force_engine = throttle * MAX_ENGINE_FORCE
    force_brake = brake * MAX_BRAKE_FORCE

    drag = 0.5 * AIR_DENSITY * DRAG_COEFFICIENT * CAR_FRONTAL_AREA * (speed_ms ** 2)
    resistance = ROLLING_RESISTANCE if speed_ms > 0 else 0

    net_force = force_engine - force_brake - drag - resistance
    accel = net_force / CAR_MASS

    speed_ms += accel * TIMESTEP
    speed_ms = max(0.0, min(speed_ms, MAX_SPEED_MS))

    return speed_ms, accel
