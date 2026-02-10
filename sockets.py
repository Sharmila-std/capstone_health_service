# sockets.py
from extensions import socketio
from flask_socketio import emit, join_room, leave_room
from datetime import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def get_db():
    mongo_uri = os.getenv("MONGODB_URI", "mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone")
    client = MongoClient(mongo_uri)
    return client.get_database()

@socketio.on("join")
def handle_join(data):
    # data: {"room": "<doctor_id>_<patient_id>", "username": "patient", "role": "patient"}
    room = data.get("room")
    join_room(room)
    emit("status", {"msg": f"{data.get('username')} has joined the chat."}, room=room)

@socketio.on("leave")
def handle_leave(data):
    room = data.get("room")
    leave_room(room)
    emit("status", {"msg": f"{data.get('username')} has left the chat."}, room=room)

@socketio.on("send_message")
def handle_message(data):
    """
    data = {
      "room": "doctorid_patientid",
      "sender": "patient" or "doctor",
      "text": "message text",
      "doctor_id": "...",
      "patient_id": "..."
    }
    """
    room = data.get("room")
    sender = data.get("sender")
    text = data.get("text")
    db = get_db()
    doc = {
        "sender": sender,
        "text": text,
        "doctor_id": data.get("doctor_id"),
        "patient_id": data.get("patient_id"),
        "timestamp": datetime.utcnow()
    }
    db.messages.insert_one(doc)
    emit("new_message", {
        "sender": sender,
        "text": text,
        "timestamp": doc["timestamp"].isoformat()
    }, room=room)
