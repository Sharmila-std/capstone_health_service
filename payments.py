# payments.py
import stripe
import os
from flask import Blueprint, request, jsonify, render_template, url_for
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

payments = Blueprint("payments", __name__, template_folder="templates")

def get_db():
    mongo_uri = os.getenv("MONGODB_URI", "mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone")
    client = MongoClient(mongo_uri)
    return client.get_database()

@payments.route("/pay", methods=["GET"])
def pay_page():
    # Example page to initiate a payment
    amount = request.args.get("amount", 100)  # rupees or cents depending on assumptions
    publish_key = os.getenv("STRIPE_PUBLISHABLE_KEY")
    return render_template("pay.html", amount=amount, publish_key=publish_key)

@payments.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    data = request.json
    # expected: {"amount": 1000, "currency": "inr", "success_url": "...", "cancel_url": "..."}
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": data.get("currency", "inr"),
                    "product_data": {"name": data.get("description", "Payment")},
                    "unit_amount": int(data["amount"])  # amount in paise if INR
                },
                "quantity": 1
            }],
            mode="payment",
            success_url=data.get("success_url"),
            cancel_url=data.get("cancel_url"),
        )
        return jsonify({"id": session.id})
    except Exception as e:
        return jsonify(error=str(e)), 400
