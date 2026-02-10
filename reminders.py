# reminders.py
from datetime import datetime, timedelta
from extensions import scheduler
from flask_mail import Message
from extensions import mail
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def get_db():
    mongo_uri = os.getenv("MONGODB_URI", "mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone")
    client = MongoClient(mongo_uri)
    return client.get_database()

def send_appointment_reminders():
    db = get_db()
    now = datetime.utcnow()
    window_start = now + timedelta(hours=23)   # send reminder for next 24 hours
    window_end = now + timedelta(hours=25)
    appointments = db.appointments.find({
        "datetime": {"$gte": window_start, "$lte": window_end},
        "reminder_sent": {"$ne": True}
    })
    for appt in appointments:
        patient = db.users.find_one({"_id": appt["patient_id"]}) if appt.get("patient_id") else None
        if patient and patient.get("email"):
            msg = Message("Appointment reminder", recipients=[patient["email"]])
            msg.body = f"Hi {patient.get('name')},\n\nYou have an appointment at {appt['datetime']} with doctor id {appt['doctor_id']}.\n\nThanks."
            try:
                mail.send(msg)
                db.appointments.update_one({"_id": appt["_id"]}, {"$set": {"reminder_sent": True}})
            except Exception as e:
                print("Failed to send reminder:", e)

def start_scheduler():
    # schedule job every hour
    scheduler.add_job(func=send_appointment_reminders, trigger="interval", minutes=60, id="appointment_reminders", replace_existing=True)
    scheduler.start()
