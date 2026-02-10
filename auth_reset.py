# auth_reset.py
import os
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from extensions import mail
from flask_mail import Message
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

auth_reset = Blueprint("auth_reset", __name__)

# Serializer
def get_serializer():
    secret = os.getenv("SECRET_KEY") or current_app.secret_key
    return URLSafeTimedSerializer(secret)

# You must adapt this function to use your app's Mongo connection object.
def get_db():
    # If your app uses a global `mongo` or `db`, import it instead.
    mongo_uri = os.getenv("MONGODB_URI", "mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone")
    client = MongoClient(mongo_uri)
    return client.get_database()  # default DB

@auth_reset.route("/password_reset_request", methods=["GET", "POST"])
def password_reset_request():
    if request.method == "POST":
        email_or_username = (request.form.get("email_or_username") or "").strip()
        db = get_db()
        user = db.users.find_one({"$or": [{"username": email_or_username}, {"email": email_or_username}]})
        if not user:
            flash("No user found with that username or email.", "warning")
            return redirect(url_for("auth_reset.password_reset_request"))

        s = get_serializer()
        token = s.dumps(user["username"], salt="password-reset-salt")

        reset_url = url_for("auth_reset.password_reset", token=token, _external=True)
        # send email
        msg = Message("Password Reset Request", recipients=[user.get("email") or user.get("username")])
        msg.body = render_template("password_reset_email.txt", reset_url=reset_url, name=user.get("name"))
        mail.send(msg)

        flash("Password reset link sent to the registered email (check spam).", "success")
        return redirect(url_for("login"))  # adjust your login route name

    return render_template("password_reset_request.html")

@auth_reset.route("/password_reset/<token>", methods=["GET", "POST"])
def password_reset(token):
    s = get_serializer()
    try:
        username = s.loads(token, salt="password-reset-salt", max_age=3600)  # 1 hour
    except SignatureExpired:
        flash("The password reset link has expired. Please request a new one.", "danger")
        return redirect(url_for("auth_reset.password_reset_request"))
    except BadSignature:
        flash("Invalid password reset token.", "danger")
        return redirect(url_for("auth_reset.password_reset_request"))

    if request.method == "POST":
        password_sha256 = (request.form.get("password_sha256") or "").strip()
        if not password_sha256:
            flash("Provide a new password.", "warning")
            return redirect(url_for("auth_reset.password_reset", token=token))

        db = get_db()
        db.users.update_one({"username": username}, {"$set": {"password_sha256": password_sha256}})
        flash("Password updated. Please login with the new password.", "success")
        return redirect(url_for("login"))

    return render_template("password_reset.html", token=token)
