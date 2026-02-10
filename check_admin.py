
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone")
DB_NAME = os.getenv("DB_NAME", "mydb")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
admins_col = db["admins"]

user = admins_col.find_one({"username": "admin1"})
print(f"Admin 'admin1' exists: {bool(user)}")
if user:
    print(f"Admin details: {user}")
