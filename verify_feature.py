
import os
import requests
from bson import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone")
client = MongoClient(MONGO_URI)
db = client[os.getenv("DB_NAME", "mydb")]
reports_col = db["reports"]

# Try to find a report to test with
report = reports_col.find_one()
if not report:
    print("No reports found in DB to test with.")
    exit()

report_id = str(report["_id"])
print(f"Testing with Report ID: {report_id}")

url = "http://127.0.0.1:5000/api/qa_report"

# Test data
test_cases = [
    {"report_id": report_id, "question": "What is the diagnosis?", "language": "English", "empathetic": False},
    {"report_id": report_id, "question": "Kya koi gambheer samasya hai?", "language": "Hindi", "empathetic": True}
]

# Note: This requires a session to be authenticated if running against the real server.
# For simplicity, I'll just check if the logic in rag_utils.py works.
import rag_utils

print("\n--- Testing rag_utils directly ---")
try:
    print("Test 1: Normal English")
    answer, docs = rag_utils.ask_report(report_id, "What is the diagnosis?")
    print(f"Answer: {answer[:100]}...")

    print("\nTest 2: Empathetic Hindi")
    answer, docs = rag_utils.ask_report(report_id, "Kya koi gambheer samasya hai?", language="Hindi", empathetic=True)
    print(f"Answer: {answer[:100]}...")
except Exception as e:
    print(f"FAILED: {e}")
