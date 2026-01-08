import time
import pymongo
from .config import MONGO_URI

def upload_trip(trip_id, user_id, trip_plan, data):
    client = pymongo.MongoClient(MONGO_URI)
    db = client["ubi_database"]

    db.trips.insert_one({
        "trip_id": trip_id,
        "user_id": user_id,
        "trip_plan": trip_plan,
        "sequence": data,
        "timestamp": time.time()
    })
