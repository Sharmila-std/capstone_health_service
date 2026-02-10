from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

# Try to get URI from env, else use the one from test.py
uri = os.getenv("MONGO_URI")
if not uri:
    print("MONGO_URI not found in env, using hardcoded fallback...")
    uri = "mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone"

print(f"Connecting to: {uri.split('@')[1] if '@' in uri else uri}...")

try:
    client = MongoClient(uri)
    db = client["mydb"] # Using 'mydb' as seen in test.py
    
    # Check collections
    print("Collections:", db.list_collection_names())

    # Test Insert
    patients_col = db["patients"]
    res = patients_col.insert_one({"name": "Direct DB Test", "created_at": datetime.utcnow()})
    print(f"Inserted ID: {res.inserted_id}")
    
    # Cleanup
    patients_col.delete_one({"_id": res.inserted_id})
    print("Cleanup done.")

except Exception as e:
    print(f"ERROR: {e}")
