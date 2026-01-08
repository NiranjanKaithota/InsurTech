import random
from .config import SPEED_ZONES

def create_trip_plan(duration):
    plan = []
    current_time = 0

    while current_time < duration:
        zone = random.choice(list(SPEED_ZONES.keys()))
        limit = SPEED_ZONES[zone]
        length = random.randint(20, 40)

        end = min(current_time + length, duration)
        plan.append({
            "start": current_time,
            "end": end,
            "limit": limit
        })
        current_time = end

    return plan


def get_current_limit(time_s, plan):
    for zone in plan:
        if zone["start"] <= time_s < zone["end"]:
            return zone["limit"]
    return plan[-1]["limit"]
