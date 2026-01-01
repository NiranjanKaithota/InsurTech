import pymongo
import datetime
import os
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()

# Get the value
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found! Make sure .env file exists.")

client = pymongo.MongoClient(MONGO_URI)
db = client["ubi_database"]
users_col = db["users"]

# List of dummy users to create
dummy_users = [
    {
        "user_id": "u_001",
        "name": "Niranjan",
        "policy_no": "POL-8842-X",
        "vehicle": "Honda City",
        "joined_date": datetime.datetime.now(),
        "risk_profile": "Unknown"
    },
    {
        "user_id": "u_002",
        "name": "Iranna GG",
        "policy_no": "POL-1129-A",
        "vehicle": "Hyundai Creta",
        "joined_date": datetime.datetime.now(),
        "risk_profile": "Unknown"
    },
    {
        "user_id": "u_003",
        "name": "Rushil Shod",
        "policy_no": "POL-9933-B",
        "vehicle": "Tata Nexon",
        "joined_date": datetime.datetime.now(),
        "risk_profile": "Unknown"
    }
]

print("--- Initializing User Database ---")

for user in dummy_users:
    # Check if user already exists to avoid duplicates
    if not users_col.find_one({"user_id": user["user_id"]}):
        users_col.insert_one(user)
        print(f"✅ Created User: {user['name']} ({user['user_id']})")
    else:
        print(f"ℹ️  User already exists: {user['name']}")

print("--- Database Setup Complete ---")