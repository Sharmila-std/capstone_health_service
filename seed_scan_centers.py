
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "mydb")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
scan_col = db["scan_centers"]

centers = [
    {
        "name": "City MRI & CT Scan Center",
        "address": "123 Main St, Downtown",
        "phone": "555-0101",
        "services": ["MRI", "CT Scan", "X-Ray", "Ultrasound"]
    },
    {
        "name": "Quick Diagnostics Lab",
        "address": "456 Oak Ave, Westside",
        "phone": "555-0102",
        "services": ["Blood Test", "Urine Test", "X-Ray"]
    },
    {
        "name": "Advanced Imaging Hub",
        "address": "789 Pine Rd, North Hills",
        "phone": "555-0103",
        "services": ["MRI", "PET Scan", "Mammography"]
    }
]

# Simple upsert based on name
for c in centers:
    scan_col.update_one({"name": c["name"]}, {"$set": c}, upsert=True)

print("Scan centers seeded successfully.")
