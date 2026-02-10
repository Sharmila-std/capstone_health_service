# profiles.py
import os
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

profiles = Blueprint("profiles", __name__, template_folder="templates")

UPLOAD_FOLDER = os.path.join(os.getcwd(), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db():
    mongo_uri = os.getenv("MONGODB_URI", "mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone")
    client = MongoClient(mongo_uri)
    return client.get_database()

@profiles.route("/profile")
def profile():
    if not session.get("username"):
        flash("Please login to view profile.", "warning")
        return redirect(url_for("login"))

    db = get_db()
    user = db.users.find_one({"username": session["username"]})
    if user:
        user["_id"] = str(user["_id"])
    return render_template("profile.html", user=user)

@profiles.route("/profile/edit", methods=["GET", "POST"])
def edit_profile():
    if not session.get("username"):
        flash("Please login to edit profile.", "warning")
        return redirect(url_for("login"))
    db = get_db()
    user = db.users.find_one({"username": session["username"]})
    if request.method == "POST":
        name = request.form.get("name")
        phone = request.form.get("phone")
        bio = request.form.get("bio")
        updates = {"name": name, "phone": phone, "bio": bio}
        # handle file
        file = request.files.get("profile_picture")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dest = os.path.join(UPLOAD_FOLDER, filename)
            file.save(dest)
            updates["profile_picture"] = f"/static/uploads/{filename}"
        db.users.update_one({"username": session["username"]}, {"$set": updates})
        flash("Profile updated.", "success")
        return redirect(url_for("profiles.profile"))
    return render_template("profile_edit.html", user=user)
