from fastapi import FastAPI, WebSocket
import asyncio
import time

from .physics import update_physics
from .trip_plan import create_trip_plan, get_current_limit
from .mongo import upload_trip
from .config import *

app = FastAPI()

@app.websocket("/drive")
async def drive(ws: WebSocket):
    await ws.accept()

    speed_ms = 0.0
    time_s = 0.0

    throttle = 0.0
    brake = 0.0

    trip_plan = create_trip_plan(TRIP_DURATION)
    data_log = []

    while time_s < TRIP_DURATION:
        inputs = await ws.receive_json()

        throttle = inputs.get("throttle", throttle)
        brake = inputs.get("brake", brake)

        speed_ms, accel = update_physics(speed_ms, throttle, brake)
        speed_kmh = speed_ms * 3.6

        limit = get_current_limit(time_s, trip_plan)
        is_speeding = int(speed_kmh > limit + 2)

        payload = {
            "time": round(time_s, 2),
            "speed": round(speed_kmh, 2),
            "acceleration": round(accel, 2),
            "speed_limit": limit,
            "throttle": throttle,
            "brake": brake,
            "is_speeding": is_speeding
        }

        data_log.append(payload)
        await ws.send_json(payload)

        time_s += TIMESTEP
        await asyncio.sleep(TIMESTEP)

    upload_trip(f"human_{int(time.time())}", "u_001", trip_plan, data_log)
    await ws.close()
