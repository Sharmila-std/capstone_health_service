# admin_panel.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

admin = Blueprint("admin", __name__, template_folder="templates")

def get_db():
    mongo_uri = os.getenv("MONGODB_URI", "mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone")
    client = MongoClient(mongo_uri)
    return client.get_database()

def require_admin():
    # Simple example: store 'is_admin' flag in session, or check username
    return session.get("role") == "admin" or session.get("is_admin") == True

@admin.route("/admin")
def admin_dashboard():
    if not require_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("login"))
    db = get_db()
    pending_doctors = list(db.users.find({"role": "doctor", "approved": {"$ne": True}}))
    approved_doctors = list(db.users.find({"role": "doctor", "approved": True}))
    return render_template("admin_dashboard.html", pending_doctors=pending_doctors, approved_doctors=approved_doctors)

@admin.route("/admin/approve_doctor/<username>", methods=["POST"])
def approve_doctor(username):
    if not require_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("login"))
    db = get_db()
    db.users.update_one({"username": username, "role": "doctor"}, {"$set": {"approved": True}})
    flash(f"Doctor {username} approved.", "success")
    return redirect(url_for("admin.admin_dashboard"))

@admin.route("/admin/reject_doctor/<username>", methods=["POST"])
def reject_doctor(username):
    if not require_admin():
        flash("Admin access required.", "danger")
        return redirect(url_for("login"))
    db = get_db()
    db.users.delete_one({"username": username, "role": "doctor"})
    flash(f"Doctor {username} rejected and removed.", "warning")
    return redirect(url_for("admin.admin_dashboard"))
