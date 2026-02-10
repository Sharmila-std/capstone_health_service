
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import datetime
import bcrypt

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "mydb")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
doctors_col = db["doctors"]

# Create a test doctor
test_doctor = {
    "name": "Dr. Strange",
    "username": "strange",
    "password_sha256": "test", # Dummy, normally hashed
    "role": "doctor",
    "specialisation": "Cardiology",
    "years_experience": 10,
    "hospital_name": "Metro General",
    "hospital_address": "177A Bleecker St",
    "doctor_phone": "555-0199",
    "approved": True,
    "created_at": datetime.datetime.utcnow()
}

# Check if exists
if not doctors_col.find_one({"username": "strange"}):
    doctors_col.insert_one(test_doctor)
    print("Test doctor 'Dr. Strange' created and approved.")
else:
    doctors_col.update_one({"username": "strange"}, {"$set": {"approved": True}})
    print("Test doctor 'Dr. Strange' already exists. Ensured approved=True.")
