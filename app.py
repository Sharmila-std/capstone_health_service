# app.py
import os
import hashlib
import traceback
import secrets
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from functools import wraps
from bson import ObjectId
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, abort
from pymongo import MongoClient
import bcrypt
from dotenv import load_dotenv

# SendGrid
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To

# SocketIO + Scheduler
from flask_socketio import SocketIO, join_room, leave_room, emit
from apscheduler.schedulers.background import BackgroundScheduler

# Gemini (optional)
import google.generativeai as genai
import easyocr
import cv2
try:
    import rag_utils  # RAG Logic
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: RAG features unavailable due to import error: {e}")
    RAG_AVAILABLE = False
    rag_utils = None
from werkzeug.utils import secure_filename

# ML Models
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image

import sys
print(f"DEBUG: Python executable: {sys.executable}")
print(f"DEBUG: Numpy version: {np.__version__}")

# load .env
load_dotenv()
print(f"DEBUG: SENDGRID_API_KEY loaded: {bool(os.getenv('SENDGRID_API_KEY'))}")
print(f"DEBUG: SENDER_EMAIL loaded: {os.getenv('SENDER_EMAIL')}")

# -----------------------
# Config from .env
# -----------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone")
DB_NAME = os.getenv("DB_NAME", "mydb")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(16))

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")  # verified sender in SendGrid

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

# -----------------------
# Flask + SocketIO
# -----------------------
app = Flask(__name__, template_folder="templates")
app.secret_key = SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------
# Mongo client & collections
# ---------------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
print(f"DEBUG: Connected to DB: {DB_NAME}")

patients_col = db["patients"]
doctors_col = db["doctors"]
pat_msg_col  = db["pat_msg"]
doc_msg_col  = db["doc_msg"]
bookings_col = db["bookings"]
foods_col    = db["foods"]
bloodtests_col = db["bloodtests"]   # <-- FIXED LINE
password_tokens_col = db["password_tokens"]
# inside your existing DB section where you define other collections
medicines_col = db["medicines"]
carts_col = db["carts"]
orders_col = db["orders"]
admins_col = db["admin"]
reports_col = db["reports"] # New collection for reports


# Email for receiving admin notification (optional)
ADMIN_ORDER_EMAIL = os.getenv("ADMIN_ORDER_EMAIL", SENDER_EMAIL)

ocr_reader = easyocr.Reader(['en'])
UPLOAD_FOLDER = "uploads/prescriptions"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Skin disease image upload folder
SKIN_UPLOAD_FOLDER = "uploads/skin_images"
os.makedirs(SKIN_UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

# Report upload folder
REPORT_UPLOAD_FOLDER = "uploads/reports"
os.makedirs(REPORT_UPLOAD_FOLDER, exist_ok=True)


# -----------------------
# Load ML Models
# -----------------------
import warnings
try:
    # Suppress sklearn version warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Diabetes Model
    with open('diabetes_gb_model.pkl', 'rb') as f:
        diabetes_model = pickle.load(f)
    with open('diabetes_scaler.pkl', 'rb') as f:
        diabetes_scaler = pickle.load(f)
    
    # Heart Disease Model
    with open('heart_nb_model.pkl', 'rb') as f:
        heart_model = pickle.load(f)
    with open('heart_scaler.pkl', 'rb') as f:
        heart_scaler = pickle.load(f)
    
    # Stroke Model
    with open('stroke_gb_model.pkl', 'rb') as f:
        stroke_model = pickle.load(f)
    with open('stroke_scaler.pkl', 'rb') as f:
        stroke_scaler = pickle.load(f)
    
    print("DEBUG: ML models loaded successfully")
except Exception as e:
    print(f"WARNING: Failed to load ML models: {e}")
    print("HINT: Models may need to be retrained with current scikit-learn version")
    diabetes_model = diabetes_scaler = None
    heart_model = heart_scaler = None
    stroke_model = stroke_scaler = None

# -----------------------
# Load Skin Disease DL Model
# -----------------------
try:
    # IMPORTANT: The model 'skin_model_final.keras' appears to be a Keras 3 model.
    # Standard 'import keras' in this environment is Keras 3.
    import keras as keras_v3
    
    print(f"DEBUG: Attempting to load skin model with Keras {keras_v3.__version__}")
    skin_model = keras_v3.models.load_model('skin_model_final.keras')
    SKIN_CLASS_NAMES = ['acne', 'bags', 'redness']
    print("DEBUG: Skin disease model loaded successfully with Keras 3")
except Exception as e:
    print(f"WARNING: Failed to load skin disease model with Keras 3: {e}")
    # Fallback to tf_keras (Keras 2) just in case
    try:
        import tf_keras as keras_v2
        skin_model = keras_v2.models.load_model('skin_model_final.keras')
        SKIN_CLASS_NAMES = ['acne', 'bags', 'redness']
        print("DEBUG: Skin disease model loaded successfully with Keras 2 (tf_keras)")
    except Exception as e2:
        print(f"WARNING: Failed to load skin disease model with Keras 2: {e2}")
        skin_model = None
        SKIN_CLASS_NAMES = []


def get_or_create_cart(patient_id):
    """Return a cart doc for patient_id (create if missing)"""
    cart = carts_col.find_one({"patient_id": str(patient_id)})
    if not cart:
        cart = {
            "patient_id": str(patient_id),
            "items": [],  # each item: {medicine_id, name, qty, price}
            "updated_at": datetime.utcnow()
        }
        carts_col.insert_one(cart)
        # reload with _id
        cart = carts_col.find_one({"patient_id": str(patient_id)})
    return cart

# Pharmacy routes and helpers are consolidated below

# -----------------------
# Utility and auth
# -----------------------
def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                flash("Unauthorized access", "error")
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return wrapped
    return decorator

def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    try:
        user = patients_col.find_one({"_id": ObjectId(uid)}) or doctors_col.find_one({"_id": ObjectId(uid)})
    except Exception:
        user = patients_col.find_one({"_id": uid}) or doctors_col.find_one({"_id": uid})
    return user

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_skin_image(image_path):
    """Preprocess image for skin disease model prediction"""
    try:
        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# -----------------------
# Skin Disease Prediction Route
# -----------------------
@app.route("/predict_skin_disease", methods=["POST"])
@login_required("patient")
def predict_skin_disease():
    """Predict skin disease from uploaded face images"""
    if not skin_model:
        return jsonify({"ok": False, "msg": "Skin disease model not available"})
    
    # Check if all three images are provided
    if 'left_image' not in request.files or 'right_image' not in request.files or 'front_image' not in request.files:
        return jsonify({"ok": False, "msg": "Please upload all three images (left, right, and front)"})
    
    left_file = request.files['left_image']
    right_file = request.files['right_image']
    front_file = request.files['front_image']
    
    # Validate files
    if left_file.filename == '' or right_file.filename == '' or front_file.filename == '':
        return jsonify({"ok": False, "msg": "Please select all three images"})
    
    if not (allowed_file(left_file.filename) and allowed_file(right_file.filename) and allowed_file(front_file.filename)):
        return jsonify({"ok": False, "msg": "Invalid file format. Please upload JPG, JPEG, or PNG images"})
    
    try:
        # Save uploaded images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = session.get("user_id", "unknown")
        
        left_filename = secure_filename(f"{user_id}_{timestamp}_left.jpg")
        right_filename = secure_filename(f"{user_id}_{timestamp}_right.jpg")
        front_filename = secure_filename(f"{user_id}_{timestamp}_front.jpg")
        
        left_path = os.path.join(SKIN_UPLOAD_FOLDER, left_filename)
        right_path = os.path.join(SKIN_UPLOAD_FOLDER, right_filename)
        front_path = os.path.join(SKIN_UPLOAD_FOLDER, front_filename)
        
        left_file.save(left_path)
        right_file.save(right_path)
        front_file.save(front_path)
        
        # Preprocess images
        left_img = preprocess_skin_image(left_path)
        right_img = preprocess_skin_image(right_path)
        front_img = preprocess_skin_image(front_path)
        
        if left_img is None or right_img is None or front_img is None:
            return jsonify({"ok": False, "msg": "Error processing images. Please try again with valid images"})
        
        # Make predictions
        left_pred = skin_model.predict(left_img, verbose=0)
        right_pred = skin_model.predict(right_img, verbose=0)
        front_pred = skin_model.predict(front_img, verbose=0)
        
        # Average predictions across all three images
        avg_pred = (left_pred + right_pred + front_pred) / 3.0
        
        # Get predicted class and probability
        predicted_class_idx = np.argmax(avg_pred[0])
        probability = float(avg_pred[0][predicted_class_idx] * 100)
        
        # Determine if disease is present (threshold: 60%)
        if probability < 60:
            return jsonify({
                "ok": True,
                "prediction": -1,
                "disease_name": "No disease present",
                "probability": round(probability, 2),
                "message": "No Skin Disease Detected",
                "details": f"The model analyzed your images with {round(probability, 2)}% confidence. No significant skin condition was detected."
            })
        
        disease_name = SKIN_CLASS_NAMES[predicted_class_idx].capitalize()
        
        return jsonify({
            "ok": True,
            "prediction": int(predicted_class_idx),
            "disease_name": disease_name,
            "probability": round(probability, 2),
            "message": f"{disease_name} Detected",
            "details": f"The model detected {disease_name.lower()} with {round(probability, 2)}% confidence based on the analysis of your face images."
        })
        
    except Exception as e:
        print(f"Error in skin disease prediction: {e}")
        traceback.print_exc()
        return jsonify({"ok": False, "msg": f"Prediction failed: {str(e)}"})




# -----------------------------
# SHOW ALL MEDICINES
# -----------------------------
# Duplicate /pharmacy route removed. Using the one defined later.




# -----------------------------
# ADD TO CART 
# -----------------------------
@app.route("/add_to_cart", methods=["POST"])
@login_required("patient")
def add_to_cart():

    data = request.get_json()
    med_id = data.get("med_id")
    qty = int(data.get("qty", 1))
    user_id = session["user_id"]

    med = medicines_col.find_one({"_id": ObjectId(med_id)})
    if not med:
        return jsonify({"ok": False, "msg": "Medicine not found"})

    # Allow adding to cart even if stock is low (for future purchase)
    msg = "Added to cart"
    if med["stock"] < qty:
        msg = "Added to cart (Out of Stock - for future purchase)"

    # Find user's cart
    cart = carts_col.find_one({"patient_id": user_id})

    if not cart:
        # create new
        carts_col.insert_one({
            "patient_id": user_id,
            "items": [
                {
                    "med_id": med_id,
                    "name": med["name"],
                    "price": med["price"],
                    "qty": qty
                }
            ]
        })
    else:
        # update existing cart
        items = cart["items"]

        # If same medicine already exists → increase qty
        found = False
        for item in items:
            # Handle both med_id and medicine_id keys
            existing_id = item.get("med_id") or item.get("medicine_id")
            if str(existing_id) == str(med_id):
                item["qty"] += qty
                found = True
                break

        if not found:
            items.append({
                "med_id": med_id,
                "name": med["name"],
                "price": med["price"],
                "qty": qty
            })

        carts_col.update_one(
            {"patient_id": user_id},
            {"$set": {"items": items}}
        )

    return jsonify({"ok": True, "msg": msg})


# -----------------------------
# VIEW CART
# -----------------------------
@app.route("/cart")
@login_required("patient")
def cart_page():
    user = current_user()
    cart = carts_col.find_one({"patient_id": session["user_id"]})

    if not cart:
        return render_template("cart.html", items=[], user=user, total=0)

    total_amount = sum(item["price"] * item["qty"] for item in cart["items"])

    return render_template("cart.html",
                           items=cart["items"],
                           user=user,
                           total=total_amount)



# -----------------------------
# REMOVE ITEM FROM CART
# -----------------------------
@app.route("/remove_from_cart", methods=["POST"])
@login_required("patient")
def remove_from_cart():
    med_id = request.get_json().get("med_id")
    user_id = session["user_id"]

    cart = carts_col.find_one({"patient_id": user_id})
    if not cart:
        return jsonify({"ok": False})

    new_items = []
    for i in cart["items"]:
        existing_id = i.get("med_id") or i.get("medicine_id")
        if str(existing_id) != str(med_id):
            new_items.append(i)

    carts_col.update_one(
        {"patient_id": user_id},
        {"$set": {"items": new_items}}
    )

    return jsonify({"ok": True})

@app.route("/prescription_checker")
@login_required("patient")
def prescription_checker():
    return render_template("prescription_checker.html")

@app.route("/process_prescription", methods=["POST"])
@login_required("patient")
def process_prescription():
    if "file" not in request.files:
        flash("Please upload an image", "error")
        return redirect(url_for("prescription_checker"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("prescription_checker"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # ----- OCR EXTRACT -----
    try:
        result = ocr_reader.readtext(filepath, detail=0)
        raw_text = " ".join(result)
    except Exception as e:
        flash("OCR failed. Try another image.", "error")
        return redirect(url_for("prescription_checker"))

    # ----- AI CLEANING -----
    ai_prompt = f"""
    Extract structured information from this prescription:

    {raw_text}

    Return JSON with keys:
    medicines: list of medicine names
    tests: list of tests or scans mentioned
    explanation: simplified explanation for patient
    """

    ok, ai_output = query_gemini(ai_prompt)

    if not ok:
        ai_output = "{}"

    # Parse AI JSON fallback
    import re, json
    json_text = re.findall(r"\{.*\}", ai_output, re.DOTALL)
    try:
        parsed = json.loads(json_text[0])
    except:
        parsed = {
            "medicines": [],
            "tests": [],
            "explanation": ai_output
        }

    # Save to DB
    pres_doc = {
        "user_id": session["user_id"],
        "image_path": filepath,
        "raw_text": raw_text,
        "ai_clean_text": parsed.get("explanation", ""),
        "medicines": parsed.get("medicines", []),
        "tests": parsed.get("tests", []),
        "created_at": datetime.utcnow()
    }
    prescriptions_col = db["prescriptions"]
    pres_id = prescriptions_col.insert_one(pres_doc).inserted_id

    return redirect(url_for("show_prescription_result", pres_id=pres_id))

@app.route("/prescription_result/<pres_id>")
@login_required("patient")
def show_prescription_result(pres_id):
    pres = db["prescriptions"].find_one({"_id": ObjectId(pres_id)})
    if not pres:
        flash("Prescription not found", "error")
        return redirect(url_for("prescription_checker"))
    
    return render_template(
        "prescription_result.html",
        raw_text=pres.get("raw_text", ""),
        ai_clean_text=pres.get("ai_clean_text", ""),
        medicines=pres.get("medicines", []),
        tests=pres.get("tests", []),
        pres_id=str(pres_id)
    )

@app.route("/add_prescription_meds_to_cart/<pres_id>")
@login_required("patient")
def add_prescription_meds_to_cart(pres_id):
    pres = db["prescriptions"].find_one({"_id": ObjectId(pres_id)})
    if not pres:
        flash("Prescription not found", "error")
        return redirect(url_for("prescription_checker"))

    meds = pres.get("medicines", [])
    user_id = session["user_id"]
    user_email = (current_user() or {}).get("email")

    # Load patient’s cart
    cart = carts_col.find_one({"patient_id": str(user_id)})
    if not cart:
        cart = {
            "patient_id": str(user_id),
            "items": [],
            "updated_at": datetime.utcnow()
        }
        carts_col.insert_one(cart)
        cart = carts_col.find_one({"patient_id": str(user_id)})

    added_count = 0
    waitlist_count = 0
    items = cart.get("items", [])

    for m in meds:
        # Flexible search
        med_obj = medicines_col.find_one({"name": {"$regex": m, "$options": "i"}})
        
        if med_obj and med_obj.get("stock", 0) > 0:
            # Add to cart
            med_id = str(med_obj["_id"])
            found = False
            for item in items:
                existing_id = item.get("med_id") or item.get("medicine_id")
                if str(existing_id) == med_id:
                    # Don't double add quantity for prescription import, just ensure it's there? 
                    # Or maybe just add 1. Let's add 1.
                    item["qty"] += 1
                    found = True
                    break
            
            if not found:
                items.append({
                    "med_id": med_id,
                    "name": med_obj["name"],
                    "qty": 1,
                    "price": med_obj["price"]
                })
            added_count += 1
        else:
            # Add to waitlist
            waitlist_count += 1
            db["medicine_waitlist"].update_one(
                {"user_id": user_id, "med_name": m},
                {
                    "$set": {
                        "email": user_email,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )

    carts_col.update_one({"patient_id": str(user_id)}, {"$set": {"items": items, "updated_at": datetime.utcnow()}})

    msg = []
    if added_count > 0:
        msg.append(f"{added_count} medicines added to cart.")
    if waitlist_count > 0:
        msg.append(f"{waitlist_count} medicines added to waitlist (out of stock). You will be notified via email.")
    
    if not msg:
        flash("No medicines found to process.", "info")
    else:
        flash(" ".join(msg), "success")

    return redirect(url_for("cart_page"))

def check_waitlist_and_notify(med_name):
    """
    Checks if any user is waiting for 'med_name' and notifies them.
    Call this whenever stock is updated/added.
    """
    # Find waiters for this medicine (case-insensitive match)
    waiters = list(db["medicine_waitlist"].find({"med_name": {"$regex": f"^{med_name}$", "$options": "i"}}))
    
    for w in waiters:
        email = w.get("email")
        if email:
            subject = f"Good news! {med_name} is now available"
            html = f"""
            <p>Hello,</p>
            <p>The medicine <strong>{med_name}</strong> you were looking for is now available in our pharmacy.</p>
            <p><a href="{url_for('pharmacy', _external=True)}">Click here to buy now</a></p>
            """
            try:
                send_sendgrid_email(email, subject, html)
                # Remove from waitlist after notifying
                db["medicine_waitlist"].delete_one({"_id": w["_id"]})
                print(f"DEBUG: Notified {email} about {med_name}")
            except Exception as e:
                print(f"ERROR notifying {email}: {e}")

@app.route("/test_waitlist_trigger/<med_name>")
def test_waitlist_trigger(med_name):
    """Helper to manually trigger notification for testing"""
    check_waitlist_and_notify(med_name)
    return f"Triggered check for {med_name}"

@app.route("/waitlist")
@login_required("patient")
def waitlist_page():
    user_id = session["user_id"]
    waitlist_items = list(db["medicine_waitlist"].find({"user_id": user_id}).sort("created_at", -1))
    return render_template("waitlist.html", items=waitlist_items)

@app.route("/remove_from_waitlist/<item_id>")
@login_required("patient")
def remove_from_waitlist(item_id):
    db["medicine_waitlist"].delete_one({"_id": ObjectId(item_id), "user_id": session["user_id"]})
    flash("Removed from waitlist", "success")
    return redirect(url_for("waitlist_page"))


# -----------------------------
# PLACE ORDER (CHECKOUT)
# -----------------------------
@app.route("/checkout", methods=["POST"])
@login_required("patient")
def checkout():
    user_id = session["user_id"]
    cart = carts_col.find_one({"patient_id": user_id})

    if not cart or len(cart["items"]) == 0:
        return jsonify({"ok": False, "msg": "Cart is empty"})

    # Check stock before confirming
    for item in cart["items"]:
        med_id = item.get("med_id") or item.get("medicine_id")
        med = medicines_col.find_one({"_id": ObjectId(med_id)})
        if not med or med["stock"] < item["qty"]:
            return jsonify({"ok": False, "msg": f"Insufficient stock for {item['name']}"})

    # Deduct stock
    for item in cart["items"]:
        med_id = item.get("med_id") or item.get("medicine_id")
        medicines_col.update_one(
            {"_id": ObjectId(med_id)},
            {"$inc": {"stock": -item["qty"]}}
        )

    # Save order
    order_id = ObjectId()
    data = request.get_json()
    payment_mode = data.get("payment_mode", "Cash on Delivery")
    
    order_doc = {
        "_id": order_id,
        "patient_id": user_id,
        "items": cart["items"],
        "total_amount": sum(i["price"] * i["qty"] for i in cart["items"]),
        "order_time": datetime.utcnow(),
        "status": "Placed",
        "payment_mode": payment_mode,
        "estimated_delivery": (datetime.utcnow() + timedelta(days=3)).strftime("%Y-%m-%d")
    }

    orders_col.insert_one(order_doc)

    # Clear the cart
    carts_col.update_one(
        {"patient_id": user_id},
        {"$set": {"items": []}}
    )

    # Send Order Confirmation Email
    user = current_user()
    if user and (user.get("email") or user.get("username")):
        recipient = user.get("email") or user.get("username")
        subject = f"Order Confirmation - #{str(order_id)[-6:]}"
        
        items_html = "".join([f"<li>{i['name']} (x{i['qty']}) - ${i['price']*i['qty']:.2f}</li>" for i in cart['items']])
        
        html_content = f"""
        <p>Dear {user.get('name')},</p>
        <p>Thank you for your order!</p>
        <p><strong>Order ID:</strong> {str(order_id)}</p>
        <p><strong>Status:</strong> Placed</p>
        <p><strong>Payment Mode:</strong> {payment_mode}</p>
        <p><strong>Estimated Delivery:</strong> {order_doc['estimated_delivery']}</p>
        <h3>Items:</h3>
        <ul>{items_html}</ul>
        <p><strong>Total:</strong> ${order_doc['total_amount']:.2f}</p>
        <p>You can track your order in the 'My Orders' section.</p>
        <p>Regards,<br>HealthCare Platform</p>
        """
        send_sendgrid_email(recipient, subject, html_content)

    return jsonify({"ok": True, "msg": "Order placed successfully! Confirmation email sent."})

@app.route("/my_orders")
@login_required("patient")
def my_orders():
    user_id = session["user_id"]
    orders = list(orders_col.find({"patient_id": user_id}).sort("order_time", -1))
    return render_template("my_orders.html", orders=orders)

@app.route("/cancel_order/<order_id>")
@login_required("patient")
def cancel_order(order_id):
    user_id = session["user_id"]
    order = orders_col.find_one({"_id": ObjectId(order_id), "patient_id": user_id})
    
    if not order:
        flash("Order not found", "error")
        return redirect(url_for("my_orders"))
        
    if order.get("status") == "Delivered":
        flash("Cannot cancel delivered order", "error")
        return redirect(url_for("my_orders"))
        
    if order.get("status") == "Cancelled":
        flash("Order is already cancelled", "info")
        return redirect(url_for("my_orders"))

    # Update status
    orders_col.update_one(
        {"_id": ObjectId(order_id)},
        {"$set": {"status": "Cancelled"}}
    )
    
    # Restore stock
    for item in order.get("items", []):
        med_id = item.get("med_id") or item.get("medicine_id")
        if med_id:
            medicines_col.update_one(
                {"_id": ObjectId(med_id)},
                {"$inc": {"stock": item["qty"]}}
            )
            
    flash("Order cancelled successfully", "success")
    return redirect(url_for("my_orders"))

# -----------------------
# SendGrid helper
# -----------------------
def send_sendgrid_email(to_email: str, subject: str, html_content: str, plain_text: str = None):
    """
    Send email via SendGrid API.
    Returns True on success, False on error.
    """
    if not SENDGRID_API_KEY or not SENDER_EMAIL:
        app.logger.error("SendGrid API key or sender email not configured.")
        return False
    try:
        message = Mail(
            from_email=SENDER_EMAIL,
            to_emails=To(to_email),
            subject=subject,
            html_content=html_content
        )
        # optional: include plain_text if provided (not necessary)
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        app.logger.info("SendGrid sent: status=%s to=%s", response.status_code, to_email)
        return True
    except Exception as e:
        app.logger.exception("SendGrid send failed: %s", e)
        return False

# -----------------------
# Gemini / LLM initialisation
# -----------------------
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        app.logger.exception("Gemini configure failed")

def send_chat_notification(to_email, sender_name, message_text):
    """
    Sends email when a new chat message is received.
    """
    if not to_email:
        print("DEBUG: No email available for notification.")
        return False

    subject = f"New Message from {sender_name}"
    html_content = f"""
        <p>Hello,</p>
        <p>You received a new message from <strong>{sender_name}</strong>:</p>
        <blockquote style="background:#f6f6f6;padding:10px;border-left:4px solid #0f766e;">
            {message_text}
        </blockquote>
        <p>Please login to your account to reply.</p>
        <p>Regards,<br>HealthCare Platform</p>
    """

    try:
        message = Mail(
            from_email=SENDER_EMAIL,
            to_emails=to_email,
            subject=subject,
            html_content=html_content,
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
        print("DEBUG: Chat notification sent to:", to_email)
        return True
    except Exception as e:
        print("SendGrid Error:", e)
        return False


def query_gemini(prompt: str):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        text = getattr(response, "text", None) or str(response)
        return True, text
    except Exception as e:
        app.logger.exception("Gemini request failed")
        return False, str(e)

def query_flan(prompt: str):
    return query_gemini(prompt)

# -----------------------
# Scheduler (appointment reminders)
# -----------------------
scheduler = BackgroundScheduler()

def send_appointment_reminders():
    """
    Example: find bookings with date/time in the next 24 hours and send reminders.
    Adjust this function to match how you store booking datetimes.
    """
    try:
        now = datetime.utcnow()
        window_start = now + timedelta(hours=23)
        window_end = now + timedelta(hours=25)
        # bookings assumed to have 'date' (YYYY-MM-DD) and 'time' (HH:MM)
        for appt in bookings_col.find({"reminder_sent": {"$ne": True}}):
            appt_date = appt.get("date")
            appt_time = appt.get("time")
            if not appt_date or not appt_time:
                continue
            try:
                appt_dt = datetime.strptime(f"{appt_date} {appt_time}", "%Y-%m-%d %H:%M")
            except Exception:
                # If stored differently, skip
                continue
            if window_start <= appt_dt <= window_end:
                patient_id = appt.get("patient_id")
                patient = None
                try:
                    patient = patients_col.find_one({"_id": ObjectId(patient_id)})
                except Exception:
                    patient = patients_col.find_one({"_id": patient_id})
                if patient:
                    recipient = patient.get("email") or patient.get("username")
                    if recipient:
                        subject = "Appointment Reminder"
                        html = f"<p>Hi {patient.get('name','Patient')},</p><p>This is a reminder for your appointment on <strong>{appt_date} {appt_time}</strong>.</p>"
                        ok = send_sendgrid_email(recipient, subject, html)
                        if ok:
                            bookings_col.update_one({"_id": appt["_id"]}, {"$set": {"reminder_sent": True}})
    except Exception:
        app.logger.exception("Error in send_appointment_reminders")

def start_scheduler():
    try:
        scheduler.add_job(send_appointment_reminders, 'interval', minutes=60, id="appointment_reminder", replace_existing=True)
        scheduler.start()
        app.logger.info("Scheduler started")
    except Exception:
        app.logger.exception("Failed to start scheduler")

start_scheduler()

# -----------------------
# Utility and auth
# -----------------------
# Auth helpers moved to top


# -----------------------
# Socket.IO events: chat & typing
# -----------------------
@socketio.on("join")
def on_join(data):
    room = data.get("room")
    username = data.get("username", "Unknown")
    if not room:
        return
    join_room(room)
    app.logger.info("%s joined %s", username, room)
    emit("status", {"msg": f"{username} joined the chat."}, room=room)

@socketio.on("leave")
def on_leave(data):
    room = data.get("room")
    username = data.get("username", "Unknown")
    if room:
        leave_room(room)
        emit("status", {"msg": f"{username} left the chat."}, room=room)

@socketio.on("typing")
def on_typing(data):
    # payload: { room, username, typing: True/False }
    room = data.get("room")
    username = data.get("username", "Unknown")
    typing = bool(data.get("typing", False))
    emit("typing", {"username": username, "typing": typing}, room=room, include_self=False)

@socketio.on("send_message")
def on_send_message(data):
    try:
        room = data.get("room")
        sender = data.get("sender")
        text = (data.get("text") or "").strip()
        doctor_id = data.get("doctor_id")
        patient_id = data.get("patient_id")
        timestamp = datetime.utcnow()

        if not (room and sender and text and doctor_id and patient_id):
            print("DEBUG: Missing fields", data)
            return

        # Save to DB
        msg_doc = {
            "doctor_id": str(doctor_id),
            "patient_id": str(patient_id),
            "msg": text,
            "timestamp": timestamp
        }

        if sender == "patient":
            msg_doc["seen"] = False
            pat_msg_col.insert_one(msg_doc)
        else:
            doc_msg_col.insert_one(msg_doc)

        # Realtime emit
        emit("new_message", {
            "sender": sender,
            "text": text,
            "timestamp": timestamp.isoformat(),
            "doctor_id": doctor_id,
            "patient_id": patient_id
        }, room=room)

        # --------------------------
        # EMAIL ALERT (Important!)
        # --------------------------
        if sender == "patient":
            doctor = doctors_col.find_one({"_id": ObjectId(doctor_id)})
            patient = patients_col.find_one({"_id": ObjectId(patient_id)})

            if doctor and doctor.get("email"):
                print("DEBUG sending email to doctor:", doctor["email"])
                send_chat_notification(
                    to_email=doctor["email"],
                    sender_name=patient["name"],
                    message_text=text
                )

        if sender == "doctor":
            doctor = doctors_col.find_one({"_id": ObjectId(doctor_id)})
            patient = patients_col.find_one({"_id": ObjectId(patient_id)})

            if patient and patient.get("email"):
                print("DEBUG sending email to patient:", patient["email"])
                send_chat_notification(
                    to_email=patient["email"],
                    sender_name=doctor["name"],
                    message_text=text
                )

    except Exception as e:
        print("ERROR in send_message:", e)

# -----------------------
# Routes: home/login/register/logout
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    username = (request.form.get("username") or "").strip().lower()
    role = (request.form.get("role") or "").strip().lower()
    raw_password = request.form.get("password", "") or ""
    if not (username and role and raw_password):
        flash("Please fill username, role and password", "error")
        return redirect(url_for("login"))
    if role == "patient":
        collection = patients_col
    elif role == "doctor":
        collection = doctors_col
    elif role == "admin":
        collection = admins_col
    else:
        flash("Invalid role", "error")
        return redirect(url_for("login"))
    user = collection.find_one({"username": username})
    if not user:
        flash("Invalid username or role", "error")
        return redirect(url_for("login"))
    # Password verification: stored is bcrypt of client-side SHA256 hex string
    h = hashlib.sha256()
    h.update(raw_password.encode("utf-8"))
    password_sha256_hex = h.hexdigest().encode("utf-8")
    stored_bcrypt = user.get("password_bcrypt")
    if isinstance(stored_bcrypt, str):
        stored_bcrypt = stored_bcrypt.encode("utf-8")
    if not stored_bcrypt or not bcrypt.checkpw(password_sha256_hex, stored_bcrypt):
        flash("Incorrect username/password", "error")
        return redirect(url_for("login"))
    session["user_id"] = str(user["_id"])
    session["role"] = role
    session["name"] = user.get("name")
    if role == "patient":
        return redirect(url_for("patient_dashboard"))
    elif role == "doctor":
        return redirect(url_for("doctor_dashboard"))
    elif role == "admin":
        return redirect(url_for("admin_dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("index"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    # Common fields
    name = request.form.get("name", "").strip()
    username = request.form.get("username", "").strip().lower()
    email = request.form.get("email", "").strip().lower()
    role = request.form.get("role", "").strip().lower()
    raw_password = request.form.get("password", "").strip()

    print(f"DEBUG: Register attempt - Name: {name}, User: {username}, Email: {email}, Role: {role}")
    print("STEP CHECK: name=", bool(name))
    print("STEP CHECK: username=", bool(username))
    print("STEP CHECK: email=", bool(email))
    print("STEP CHECK: raw_password=", bool(raw_password))
    print("STEP CHECK: role=", bool(role))

    print("EXISTS username?", patients_col.find_one({"username": username}))
    print("EXISTS email?", patients_col.find_one({"email": email}))

    if not (name and username and email and raw_password and role):
        flash("All required fields must be filled.", "error")
        return redirect(url_for("register"))

    # Ensure unique username/email
    if patients_col.find_one({"username": username}) or doctors_col.find_one({"username": username}) or admins_col.find_one({"username": username}):
        flash("Username already exists.", "error")
        return redirect(url_for("register"))

    if patients_col.find_one({"email": email}) or doctors_col.find_one({"email": email}) or admins_col.find_one({"email": email}):
        flash("Email already registered.", "error")
        return redirect(url_for("register"))

    # Hash password (SHA256 → bcrypt)
    h = hashlib.sha256()
    h.update(raw_password.encode("utf-8"))
    sha256_hex = h.hexdigest()

    server_bcrypt = bcrypt.hashpw(sha256_hex.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    # Prepare document
    base_doc = {
        "name": name,
        "username": username,
        "email": email,
        "password_sha256": sha256_hex,
        "password_bcrypt": server_bcrypt,
        "role": role,
        "created_at": datetime.utcnow()
    }

    if role == "patient":
        base_doc.update({
            "age": request.form.get("age"),
            "gender": request.form.get("gender"),
            "address": request.form.get("address"),
            "phone": request.form.get("phone")
        })
        print("DEBUG: Inserting patient...")
        try:
            result = patients_col.insert_one(base_doc)
            inserted_id = result.inserted_id
            print(f"DEBUG: Patient inserted with ID: {inserted_id}")
        except Exception as e:
            print(f"ERROR inserting patient: {e}")
            flash(f"Error registering patient: {e}", "error")
            return redirect(url_for("register"))

    elif role == "doctor":
        base_doc.update({
            "specialisation": request.form.get("specialisation"),
            "years_experience": request.form.get("years_experience"),
            "hospital_name": request.form.get("hospital_name"),
            "hospital_address": request.form.get("hospital_address"),
            "doctor_phone": request.form.get("doctor_phone"),
            "approved": True
        })
        print("DEBUG: Inserting doctor...")
        try:
            result = doctors_col.insert_one(base_doc)
            inserted_id = result.inserted_id
            print(f"DEBUG: Doctor inserted with ID: {inserted_id}")
        except Exception as e:
            print(f"ERROR inserting doctor: {e}")
            flash(f"Error registering doctor: {e}", "error")
            return redirect(url_for("register"))
    elif role == "admin":
        print("DEBUG: Inserting admin...")
        try:
            result = admins_col.insert_one(base_doc)
            inserted_id = result.inserted_id
            print(f"DEBUG: Admin inserted with ID: {inserted_id}")
        except Exception as e:
            print(f"ERROR inserting admin: {e}")
            flash(f"Error registering admin: {e}", "error")
            return redirect(url_for("register"))
    else:
        flash("Invalid role selected.", "error")
        return redirect(url_for("register"))

    # Auto login after registration
    session["user_id"] = str(inserted_id)
    session["role"] = role
    session["name"] = name

    flash("Registration successful!", "success")

    if role == "patient":
        return redirect(url_for("patient_dashboard"))
    elif role == "doctor":
        return redirect(url_for("doctor_dashboard"))
    elif role == "admin":
        return redirect(url_for("admin_dashboard"))

# -----------------------
# Admin: approve doctors
# -----------------------
@app.route("/admin/doctors")
def admin_list_doctors():
    # simple admin check: you can adapt to a real admin user
    if session.get("role") != "doctor" and session.get("username") != "admin":
        # For simplicity, require explicit session 'is_admin' flag; otherwise block
        if not session.get("is_admin"):
            return abort(403)
    docs = list(doctors_col.find().sort("created_at", -1))
    return render_template("admin_doctors.html", doctors=docs)

@app.route("/admin/approve_doctor/<doc_id>", methods=["POST"])
@login_required("admin")
def admin_approve_doctor(doc_id):
    doctors_col.update_one({"_id": ObjectId(doc_id)}, {"$set": {"approved": True}})
    flash("Doctor approved", "success")
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/reject_doctor/<doc_id>", methods=["POST"])
@login_required("admin")
def admin_reject_doctor(doc_id):
    doctors_col.delete_one({"_id": ObjectId(doc_id)})
    flash("Doctor rejected and removed", "success")
    return redirect(url_for("admin_dashboard"))

# -----------------------
# E-PHARMACY ROUTES
# -----------------------


@app.route("/pharmacy")
@login_required("patient")
def pharmacy():
    """Show pharmacy product listing page"""
    # optionally accept search query
    q = (request.args.get("q") or "").strip()
    filter_q = {}
    if q:
        filter_q = {"$or": [
            {"name": {"$regex": q, "$options": "i"}},
            {"description": {"$regex": q, "$options": "i"}},
            {"tags": {"$regex": q, "$options": "i"}}
        ]}
    meds = list(medicines_col.find(filter_q).sort("name", 1))
    # convert ObjectId to string and expose stock & price
    for m in meds:
        m["_id"] = str(m["_id"])
        m["stock"] = int(m.get("stock", 0))
        # price ensure numeric
        try:
            m["price"] = float(m.get("price", 0))
        except:
            m["price"] = 0.0
    return render_template("pharmacy.html", medicines=meds, query=q)


@app.route("/api/add_to_cart", methods=["POST"])
@login_required("patient")
def api_add_to_cart():
    data = request.get_json(force=True)
    med_id = data.get("medicine_id")
    qty = int(data.get("qty", 1))
    if qty < 1: qty = 1

    # fetch medicine
    med = medicines_col.find_one({"_id": ObjectId(med_id)}) if ObjectId.is_valid(med_id) else medicines_col.find_one({"_id": med_id})
    if not med:
        return jsonify({"ok": False, "msg": "Medicine not found."}), 404

    if int(med.get("stock", 0)) < qty:
        return jsonify({"ok": False, "msg": "Not enough stock."})

    patient_id = session.get("user_id")
    cart = get_or_create_cart(patient_id)

    # check if already in cart
    found = False
    for it in cart.get("items", []):
        if str(it.get("medicine_id")) == str(med["_id"]):
            it["qty"] = int(it.get("qty", 0)) + qty
            found = True
            break

    if not found:
        cart["items"].append({
            "medicine_id": str(med["_id"]),
            "name": med.get("name"),
            "qty": qty,
            "price": float(med.get("price", 0.0))
        })

    cart["updated_at"] = datetime.utcnow()
    carts_col.update_one({"patient_id": str(patient_id)}, {"$set": cart}, upsert=True)

    return jsonify({"ok": True, "cart": cart})

@app.route("/get_cart")
@login_required("patient")
def get_cart():
    user_id = session.get("user_id")
    cart = carts_col.find_one({"patient_id": str(user_id)})
    
    if not cart:
        return jsonify({"ok": True, "items": [], "total": 0})

    items = []
    total = 0
    
    for item in cart.get("items", []):
        # Ensure price is float
        price = float(item.get("price", 0))
        qty = int(item.get("qty", 1))
        subtotal = price * qty
        total += subtotal
        
        items.append({
            "med_id": item.get("med_id") or item.get("medicine_id"), # Ensure this matches what add_to_cart saves
            "name": item.get("name"),
            "price": price,
            "qty": qty,
            "subtotal": subtotal
        })
        
    return jsonify({"ok": True, "items": items, "total": total})

@app.route("/update_cart", methods=["POST"])
@login_required("patient")
def update_cart():
    data = request.get_json()
    med_id = data.get("med_id")
    qty = int(data.get("qty", 1))
    user_id = session.get("user_id")
    
    cart = carts_col.find_one({"patient_id": str(user_id)})
    if not cart:
        return jsonify({"ok": False, "msg": "Cart not found"})
        
    items = cart.get("items", [])
    found = False
    new_items = []
    
    for item in items:
        existing_id = item.get("med_id") or item.get("medicine_id")
        if str(existing_id) == str(med_id):
            if qty > 0:
                item["qty"] = qty
                new_items.append(item)
            found = True
        else:
            new_items.append(item)
            
    if not found and qty > 0:
        # If updating an item not in cart (unlikely but possible), fetch and add?
        # For now, just ignore or return error. 
        # But let's assume it's an update to existing item.
        pass

    carts_col.update_one(
        {"patient_id": str(user_id)},
        {"$set": {"items": new_items, "updated_at": datetime.utcnow()}}
    )
    
    return jsonify({"ok": True, "msg": "Cart updated"})


@app.route("/api/update_cart_item", methods=["POST"])
@login_required("patient")
def api_update_cart_item():
    data = request.get_json(force=True)
    med_id = data.get("medicine_id")
    qty = int(data.get("qty", 0))
    patient_id = session.get("user_id")
    cart = carts_col.find_one({"patient_id": str(patient_id)})
    if not cart:
        return jsonify({"ok": False, "msg": "No cart found."})

    updated = False
    new_items = []
    for it in cart.get("items", []):
        if str(it.get("medicine_id")) == str(med_id):
            if qty > 0:
                it["qty"] = qty
                new_items.append(it)
            # if qty==0 -> remove
            updated = True
        else:
            new_items.append(it)
    cart["items"] = new_items
    cart["updated_at"] = datetime.utcnow()
    carts_col.update_one({"patient_id": str(patient_id)}, {"$set": cart})

    return jsonify({"ok": True, "cart": cart})

@app.route("/api/remove_cart_item", methods=["POST"])
@login_required("patient")
def api_remove_cart_item():
    data = request.get_json(force=True)
    med_id = data.get("medicine_id")
    patient_id = session.get("user_id")
    carts_col.update_one({"patient_id": str(patient_id)}, {"$pull": {"items": {"medicine_id": str(med_id)}}})
    cart = carts_col.find_one({"patient_id": str(patient_id)}) or {"patient_id": patient_id, "items": []}
    return jsonify({"ok": True, "cart": cart})

@app.route("/place_order", methods=["POST"])
@login_required("patient")
def place_order():
    data = request.get_json(force=True)
    payment_method = data.get("payment_method", "cash")
    address = data.get("address", None) or (current_user() or {}).get("address")

    patient_id = session.get("user_id")
    patient = current_user()
    if not patient:
        return jsonify({"ok": False, "msg": "User not found."}), 400

    cart = carts_col.find_one({"patient_id": str(patient_id)}) or {"items": []}
    if not cart.get("items"):
        return jsonify({"ok": False, "msg": "Cart empty."})

    # Validate stock and compute totals
    total = 0.0
    order_items = []
    for it in cart["items"]:
        med = medicines_col.find_one({"_id": ObjectId(it["medicine_id"])}) if ObjectId.is_valid(it["medicine_id"]) else medicines_col.find_one({"_id": it["medicine_id"]})
        if not med:
            return jsonify({"ok": False, "msg": f"Medicine {it.get('name')} not found."})
        stock = int(med.get("stock", 0))
        if stock < int(it["qty"]):
            return jsonify({"ok": False, "msg": f"Insufficient stock for {med.get('name')}. Available: {stock}"})
        item_total = float(it.get("price", med.get("price", 0.0))) * int(it["qty"])
        total += item_total
        order_items.append({
            "medicine_id": str(med["_id"]),
            "name": med.get("name"),
            "qty": int(it["qty"]),
            "unit_price": float(it.get("price", med.get("price", 0.0))),
            "total_price": item_total
        })

    # Deduct stock (careful: not using transactions; keep low concurrency assumption)
    for it in order_items:
        medicines_col.update_one({"_id": ObjectId(it["medicine_id"])}, {"$inc": {"stock": -int(it["qty"])}})

    # Create order doc
    order = {
        "patient_id": str(patient_id),
        "patient_name": patient.get("name"),
        "patient_email": patient.get("email") or patient.get("username"),
        "items": order_items,
        "total": total,
        "address": address,
        "payment_method": payment_method,
        "status": "placed",
        "created_at": datetime.utcnow()
    }
    res = orders_col.insert_one(order)
    order_id = str(res.inserted_id)

    # Clear cart
    carts_col.delete_one({"patient_id": str(patient_id)})

    # Send email to patient and admin
    html = f"<p>Hi {patient.get('name')},</p><p>Your order <b>{order_id}</b> has been placed successfully. Total: ₹{total:.2f}.</p>"
    html += "<p>Items:</p><ul>"
    for it in order_items:
        html += f"<li>{it['name']} x {it['qty']} — ₹{it['total_price']:.2f}</li>"
    html += "</ul>"
    html += "<p>We will dispatch soon. Thank you.</p>"

    # patient mail
    try:
        send_sendgrid_email(patient.get("email") or patient.get("username"), f"Order placed: {order_id}", html)
    except Exception:
        app.logger.exception("Failed to send order mail to patient")

    # admin mail
    try:
        send_sendgrid_email(ADMIN_ORDER_EMAIL, f"New ePharmacy Order: {order_id}", f"Order {order_id} details:\nTotal: {total}\nPatient: {patient.get('name')} ({patient.get('email')})")
    except Exception:
        app.logger.exception("Failed to send order mail to admin")

    return jsonify({"ok": True, "order_id": order_id})

@app.route("/orders")
@login_required("patient")
def orders_page():
    patient_id = session.get("user_id")
    ords = list(orders_col.find({"patient_id": str(patient_id)}).sort("created_at", -1))
    for o in ords:
        o["_id"] = str(o["_id"])
    return render_template("orders.html", orders=ords)

# -----------------------
# Admin add/edit medicines (simple)
# -----------------------
@app.route("/admin_dashboard")
@login_required("admin")
def admin_dashboard():
    pending_doctors = list(doctors_col.find({"approved": {"$ne": True}}))
    approved_doctors = list(doctors_col.find({"approved": True}))
    patients = list(patients_col.find({}))
    
    # Fetch all orders from all patients
    all_orders = list(orders_col.find({}).sort("order_time", -1))
    
    return render_template("admin_dashboard.html", 
                           pending_doctors=pending_doctors, 
                           approved_doctors=approved_doctors,
                           patients=patients,
                           all_orders=all_orders)

@app.route("/admin/add_medicine", methods=["GET", "POST"])
@login_required("admin")
def admin_add_medicine():
    if request.method == "GET":
        return render_template("admin_add_medicine.html")
    # POST handling
    name = request.form.get("name")
    description = request.form.get("description")
    price = float(request.form.get("price") or 0.0)
    stock = int(request.form.get("stock") or 0)
    tags = (request.form.get("tags") or "").split(",")
    medicines_col.insert_one({
        "name": name,
        "description": description,
        "price": price,
        "stock": stock,
        "tags": [t.strip() for t in tags if t.strip()],
        "created_at": datetime.utcnow()
    })
    
    # Check waitlist and notify users
    check_waitlist_and_notify(name)
    
    flash("Medicine added", "success")
    return redirect(url_for("admin_dashboard"))



@app.route("/order_success/<order_id>")
@login_required("patient")
def order_success(order_id):
    return render_template("order_success.html", order_id=order_id)


# -----------------------
# Password reset (token-based)
# -----------------------
def generate_reset_token():
    return secrets.token_urlsafe(32)

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html")
    email = (request.form.get("email") or "").strip().lower()
    if not email:
        flash("Enter email", "error")
        return redirect(url_for("forgot_password"))
    user = patients_col.find_one({"email": email}) or doctors_col.find_one({"email": email})
    if not user:
        # Try username as fallback if email not found (legacy support)
        user = patients_col.find_one({"username": email}) or doctors_col.find_one({"username": email})
    
    if not user:
        flash("If the email exists, a reset link will be sent", "info")
        return redirect(url_for("login"))
    
    # Ensure user has an email address to send to
    user_email = user.get("email")
    if not user_email:
        # If user record doesn't have email, we can't send. 
        # But if they entered an email in the form, maybe we use that? 
        # No, that's insecure. We must use the registered email.
        # If they logged in with username and we found them, but they have no email field...
        flash("No email address associated with this account. Contact admin.", "error")
        return redirect(url_for("login"))

    token = generate_reset_token()
    expires_at = datetime.utcnow() + timedelta(hours=2)
    password_tokens_col.insert_one({
        "user_id": str(user["_id"]),
        "token": token,
        "expires_at": expires_at,
        "used": False
    })
    reset_link = url_for("reset_password", token=token, _external=True)
    subject = "Password reset for your account"
    html = f"<p>Hi {user.get('name','User')},</p><p>Click the link to reset your password: <a href='{reset_link}'>{reset_link}</a></p><p>This link expires in 2 hours.</p>"
    send_sendgrid_email(user_email, subject, html)
    flash("If the email exists, a reset link will be sent", "info")
    return redirect(url_for("login"))

@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):
    rec = password_tokens_col.find_one({"token": token, "used": False})
    if not rec or rec.get("expires_at") < datetime.utcnow():
        flash("Invalid or expired token", "error")
        return redirect(url_for("login"))

    if request.method == "GET":
        return render_template("reset_password.html", token=token)

    # Get NEW RAW password
    new_raw_password = (request.form.get("password") or "").strip()

    if not new_raw_password:
        flash("Enter new password", "error")
        return redirect(url_for("reset_password", token=token))

    # FIRST → SHA256 the raw password (same as login hashing)
    h = hashlib.sha256()
    h.update(new_raw_password.encode("utf-8"))
    new_sha256_hex = h.hexdigest()

    # THEN → bcrypt the sha256 hex
    new_bcrypt = bcrypt.hashpw(new_sha256_hex.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    # Update in DB (patient or doctor)
    user_id = rec.get("user_id")
    patients_col.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password_sha256": new_sha256_hex, "password_bcrypt": new_bcrypt}}
    )
    doctors_col.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password_sha256": new_sha256_hex, "password_bcrypt": new_bcrypt}}
    )

    # Mark token used
    password_tokens_col.update_one({"_id": rec["_id"]}, {"$set": {"used": True}})

    flash("Password updated. Please login with your new password.", "success")
    return redirect(url_for("login"))

# -----------------------
# Chat UI endpoints
# -----------------------
@app.route("/chat_with_doctor/<doctor_id>")
@login_required("patient")
def chat_with_doctor(doctor_id):
    doctor = doctors_col.find_one({"_id": ObjectId(doctor_id)})
    doctor_name = doctor["name"] if doctor else "Unknown Doctor"

    patient_id = session.get("user_id")
    patient_name = session.get("name")

    return render_template(
        "patient_chat.html",
        doctor_id=str(doctor_id),
        doctor_name=doctor_name,
        patient_id=str(patient_id),
        patient_name=patient_name
    )

@app.route("/book_test", methods=["POST"])
@login_required("patient")
def book_test():
    data = request.get_json()

    center_id = data.get("center_id")
    center_name = data.get("center_name")
    test_name = data.get("test_name")
    date = data.get("date")
    time = data.get("time")
    patient_id = session["user_id"]

    if not (center_id and test_name and date and time):
        return jsonify({"ok": False, "msg": "Missing fields"})

    # Prevent double booking
    exists = db["diagnostic_bookings"].find_one({
        "center_id": center_id,
        "test_name": test_name,
        "date": date,
        "time": time
    })

    if exists:
        return jsonify({"ok": False, "msg": "Slot already taken"})

    # Generate Token Number (incremental)
    count = db["diagnostic_bookings"].count_documents({
        "center_id": center_id,
        "date": date
    })
    token_number = count + 1

    booking = {
        "patient_id": patient_id,
        "center_id": center_id,
        "center_name": center_name,
        "test_name": test_name,
        "date": date,
        "time": time,
        "token": token_number,
        "created_at": datetime.utcnow()
    }

    db["diagnostic_bookings"].insert_one(booking)

    return jsonify({"ok": True, "token": token_number})



@app.route("/anonymous_chat_with_doctor/<doctor_id>")
def anonymous_chat_with_doctor(doctor_id):
    # If logged-in patient, redirect to normal chat handler
    if session.get("user_id") and session.get("role") == "patient":
        return redirect(url_for("chat_with_doctor", doctor_id=doctor_id))

    # Ensure a persistent anonymous id in session so messages from same browser stay grouped
    if not session.get("anon_user_id"):
        session["anon_user_id"] = "anon_" + secrets.token_hex(8)
        session["anon_name"] = "Anonymous"

    patient_id = session.get("anon_user_id")
    patient_name = session.get("anon_name", "Anonymous")

    # Fetch doctor name for display
    try:
        doctor = doctors_col.find_one({"_id": ObjectId(doctor_id)})
    except Exception:
        doctor = doctors_col.find_one({"_id": doctor_id})
    doctor_name = doctor.get("name") if doctor else None

    return render_template("patient_chat.html", doctor_id=doctor_id, doctor_name=doctor_name, patient_id=patient_id, patient_name=patient_name)

@app.route("/doctor_chat")
@login_required("doctor")
def doctor_chat():
    doctor_id = session.get("user_id")
    # get doctor object
    try:
        doctor = doctors_col.find_one({"_id": ObjectId(doctor_id)})
    except Exception:
        doctor = doctors_col.find_one({"_id": doctor_id})
    # patients who messaged
    patient_ids = pat_msg_col.distinct("patient_id", {"doctor_id": doctor_id})
    patients = []
    for pid in patient_ids:
        try:
            user = patients_col.find_one({"_id": ObjectId(pid)})
        except Exception:
            user = patients_col.find_one({"_id": pid})
        if user:
            new_count = pat_msg_col.count_documents({"doctor_id": doctor_id, "patient_id": str(user["_id"]), "seen": False})
            patients.append({"_id": str(user["_id"]), "name": user.get("name", "Anonymous"), "new_msg_count": new_count})
    return render_template("doctor_chat.html", patients=patients, user=doctor)

# -----------------------
# HTTP message endpoints (fallback)
# -----------------------
@app.route("/send_patient_msg", methods=["POST"])
@login_required("patient")
def send_patient_msg():
    data = request.json or request.get_json(force=True)
    msg = (data.get("msg") or "").strip()
    doctor_id = data.get("doctor_id")
    patient_id = session.get("user_id")

    if not (msg and doctor_id and patient_id):
        return jsonify({"ok": False, "msg": "Missing fields"})

    # Save message in DB
    pat_msg_col.insert_one({
        "patient_id": str(patient_id),
        "doctor_id": str(doctor_id),
        "msg": msg,
        "timestamp": datetime.utcnow(),
        "seen": False
    })

    # Real-time emit
    room = f"{doctor_id}_{patient_id}"
    socketio.emit("new_message", {
        "sender": "patient",
        "text": msg,
        "timestamp": datetime.utcnow().isoformat(),
        "doctor_id": doctor_id,
        "patient_id": patient_id
    }, room=room)

    # -------------------------
    # EMAIL NOTIFICATION HERE
    # -------------------------
    doctor = doctors_col.find_one({"_id": ObjectId(doctor_id)})
    patient = patients_col.find_one({"_id": ObjectId(patient_id)})
    if doctor and doctor.get("email"):
        send_chat_notification(
            to_email=doctor["email"],
            sender_name=patient["name"],
            message_text=msg
        )

    return jsonify({"ok": True})


@app.route("/send_doctor_msg", methods=["POST"])
@login_required("doctor")
def send_doctor_msg():
    data = request.json or request.get_json(force=True)
    doctor_id = data.get("doctor_id")
    patient_id = data.get("patient_id")
    msg = (data.get("msg") or "").strip()

    if not (doctor_id and patient_id and msg):
        return jsonify({"ok": False, "msg": "Missing fields"})

    # Save message
    doc_msg_col.insert_one({
        "doctor_id": str(doctor_id),
        "patient_id": str(patient_id),
        "msg": msg,
        "timestamp": datetime.utcnow()
    })

    room = f"{doctor_id}_{patient_id}"
    socketio.emit("new_message", {
        "sender": "doctor",
        "text": msg,
        "timestamp": datetime.utcnow().isoformat(),
        "doctor_id": doctor_id,
        "patient_id": patient_id
    }, room=room)

    # -------------------------
    # EMAIL NOTIFICATION HERE
    # -------------------------
    doctor = doctors_col.find_one({"_id": ObjectId(doctor_id)})
    patient = patients_col.find_one({"_id": ObjectId(patient_id)})

    if patient and patient.get("email"):
        send_chat_notification(
            to_email=patient["email"],
            sender_name=doctor["name"],
            message_text=msg
        )

    return jsonify({"ok": True})


@app.route("/get_patients_for_doctor")
@login_required("doctor")
def get_patients_for_doctor():
    doctor_id = request.args.get("doctor_id") or session.get("user_id")
    patient_ids = pat_msg_col.distinct("patient_id", {"doctor_id": doctor_id})
    patients = []
    for pid in patient_ids:
        try:
            patient = patients_col.find_one({"_id": ObjectId(pid)})
        except Exception:
            patient = patients_col.find_one({"_id": pid})
        if patient:
            new_count = pat_msg_col.count_documents({"doctor_id": doctor_id, "patient_id": str(patient["_id"]), "seen": False})
            patients.append({"_id": str(patient["_id"]), "name": patient.get("name", "Anonymous"), "new_msg_count": new_count})
    return jsonify({"patients": patients})

@app.route("/get_messages_for_doctor")
def get_messages_for_doctor():
    # Allow fetching for logged-in patient/doctor or anonymous session owner (anon_user_id)
    doctor_id = request.args.get("doctor_id")
    patient_id = request.args.get("patient_id")
    if not (doctor_id and patient_id):
        return jsonify({"messages": []})

    # Access control: allow if requester is the patient (session user) or anon owner, or the doctor
    allowed = False
    if session.get("user_id") and session.get("user_id") == patient_id:
        allowed = True
    if session.get("anon_user_id") and session.get("anon_user_id") == patient_id:
        allowed = True
    if session.get("role") == "doctor" and session.get("user_id") == doctor_id:
        allowed = True
    if not allowed:
        return jsonify({"messages": []})

    messages = []
    pat_msgs = pat_msg_col.find({"doctor_id": doctor_id, "patient_id": patient_id})
    for m in pat_msgs:
        messages.append({"sender": "patient", "msg": m["msg"], "timestamp": m["timestamp"].isoformat(), "seen": m.get("seen", False)})
    doc_msgs = doc_msg_col.find({"doctor_id": doctor_id, "patient_id": patient_id})
    for m in doc_msgs:
        messages.append({"sender": "doctor", "msg": m["msg"], "timestamp": m["timestamp"].isoformat()})
    messages.sort(key=lambda x: x["timestamp"])
    return jsonify({"messages": messages})

@app.route("/mark_messages_seen", methods=["POST"])
@login_required("doctor")
def mark_messages_seen():
    data = request.json or request.get_json(force=True)
    doctor_id = data.get("doctor_id")
    patient_id = data.get("patient_id")
    if not (doctor_id and patient_id):
        return jsonify({"ok": False, "msg": "Missing fields"})
    pat_msg_col.update_many({"doctor_id": doctor_id, "patient_id": patient_id, "seen": False}, {"$set": {"seen": True}})
    return jsonify({"ok": True})

# -----------------------
# Marketplace / Bloodtest booking
# -----------------------

@app.route("/marketplace")
@login_required("patient")
def marketplace():
    try:
        # Fetch from correct collection
        centers = list(db["test_centers"].find({}))
        for c in centers:
            c["_id"] = str(c["_id"])
        print("DEBUG Centers Loaded:", centers)  # debug
        
        return render_template("marketplace.html", centers=centers)

    except Exception as e:
        print("Marketplace Load Error:", e)
        return "Error loading marketplace"

@app.route("/book_diagnostic", methods=["POST"])
@login_required("patient")
def book_diagnostic():
    data = request.get_json()
    center_id = data.get("center_id")
    hour = data.get("hour")
    date_str = data.get("date") # Expect YYYY-MM-DD
    patient_id = session["user_id"]

    if not (center_id and hour and date_str):
        return jsonify({"ok": False, "msg": "Missing fields"})

    # Fetch center
    center = db["test_centers"].find_one({"_id": ObjectId(center_id)})
    if not center:
        return jsonify({"ok": False, "msg": "Center not found"})

    # Fetch patient
    patient = patients_col.find_one({"_id": ObjectId(patient_id)})
    if not patient:
        return jsonify({"ok": False, "msg": "Patient not found"})

    # Check availability in diagnostic_bookings collection
    existing_booking = db["diagnostic_bookings"].find_one({
        "center_id": center_id,
        "date": date_str,
        "hour": hour
    })

    if existing_booking:
        return jsonify({"ok": False, "msg": "Slot already booked for this date"})

    # Generate Token
    # Count bookings for this center on this date to generate a simple token number
    count = db["diagnostic_bookings"].count_documents({
        "center_id": center_id,
        "date": date_str
    })
    token = count + 1

    # Save FULL booking record
    booking_doc = {
        "center_id": center_id,
        "center_name": center.get("center_name"),
        "type": center.get("type"),
        "location": center.get("location"),

        "patient_id": patient_id,
        "patient_name": patient.get("name"),
        "patient_email": patient.get("email"),

        "date": date_str,
        "hour": hour,
        "token": token,
        "booked_at": datetime.utcnow()
    }

    res = db["diagnostic_bookings"].insert_one(booking_doc)
    booking_id = str(res.inserted_id)

    # Send Confirmation Email
    recipient = patient.get("email") or patient.get("username")
    if recipient:
        subject = f"Diagnostic Booking Confirmed - {center.get('center_name')}"
        html_content = f"""
        <p>Dear {patient.get('name')},</p>
        <p>Your booking for <strong>{center.get('center_name')}</strong> has been confirmed.</p>
        <p><strong>Date:</strong> {date_str}</p>
        <p><strong>Time:</strong> {hour}</p>
        <p><strong>Token Number:</strong> {token}</p>
        <p><strong>Booking ID:</strong> {booking_id}</p>
        <p>Please reach the center 15 minutes prior to your slot.</p>
        <p>Regards,<br>HealthCare Platform</p>
        """
        send_sendgrid_email(recipient, subject, html_content)

    return jsonify({
        "ok": True,
        "msg": "Booking successful",
        "token": token,
        "booking_id": booking_id
    })

@app.route("/api/get_booked_slots")
def get_booked_slots():
    center_id = request.args.get("center_id")
    date_str = request.args.get("date")
    
    if not (center_id and date_str):
        return jsonify({"booked_hours": []})
        
    bookings = list(db["diagnostic_bookings"].find({
        "center_id": center_id,
        "date": date_str
    }))
    
    booked_hours = [b["hour"] for b in bookings]
    return jsonify({"booked_hours": booked_hours})

@app.route("/my_diagnostic_bookings")
@login_required("patient")
def my_diagnostic_bookings():
    patient_id = session["user_id"]

    bookings = list(db["diagnostic_bookings"].find({"patient_id": patient_id}))
    for b in bookings:
        b["_id"] = str(b["_id"])
        b["booked_at"] = b["booked_at"].strftime("%Y-%m-%d %H:%M:%S")

    return render_template("my_diagnostic_bookings.html", bookings=bookings)

@app.route("/cancel_diagnostic_booking", methods=["POST"])
@login_required("patient")
def cancel_diagnostic_booking():
    data = request.get_json()
    booking_id = data.get("booking_id")

    booking = db["diagnostic_bookings"].find_one({"_id": ObjectId(booking_id)})
    if not booking:
        return jsonify({"ok": False, "msg": "Booking not found"})

    center_id = booking["center_id"]
    hour = booking["hour"]

    # 1. Free the slot inside test_centers
    center = db["test_centers"].find_one({"_id": ObjectId(center_id)})
    if center:
        slots = center["slots"]
        for s in slots:
            if s.get("hour") == hour:
                s["booked_by"] = None
                s["token"] = None

        db["test_centers"].update_one(
            {"_id": ObjectId(center_id)},
            {"$set": {"slots": slots}}
        )

    # 2. Remove booking document
    db["diagnostic_bookings"].delete_one({"_id": ObjectId(booking_id)})

    return jsonify({"ok": True, "msg": "Booking cancelled successfully"})

@app.route("/statistics")
@login_required("patient")
def statistics():
    return render_template("statistics.html")

@app.route("/add_stats", methods=["GET", "POST"])
@login_required("patient")
def add_stats():
    if request.method == "GET":
        return render_template("add_stats.html")

    data = request.form
    pid = session["user_id"]

    doc = {
        "patient_id": pid,
        "date": data["date"],
        "bp_systolic": int(data["bp_systolic"]),
        "bp_diastolic": int(data["bp_diastolic"]),
        "sugar_level": int(data["sugar_level"]),
        "heart_rate": int(data["heart_rate"]),
        "weight": float(data["weight"]),
        "height": float(data.get("height", 0))
    }

    db.health_stats.insert_one(doc)
    flash("Daily stats added!", "success")
    return redirect("/statistics")

@app.route("/health_risks")
@login_required("patient")
def health_risks():
    user_id = session["user_id"]
    
    # Get all available dates for dropdown
    all_stats = list(db.health_stats.find({"patient_id": user_id}, {"date": 1}).sort("date", -1))
    available_dates = [s["date"] for s in all_stats]
    
    # Determine selected date
    selected_date = request.args.get("date")
    if not selected_date and available_dates:
        selected_date = available_dates[0]
        
    # Get stats for selected date
    stats = None
    if selected_date:
        stats = db.health_stats.find_one({"patient_id": user_id, "date": selected_date})
    
    if not stats:
        return render_template("health_risks.html", stats=None, available_dates=available_dates, selected_date=None, symptoms=[])

    # Get user details (age, gender)
    user = patients_col.find_one({"_id": ObjectId(user_id)})
    age = int(user.get("age", 30))
    gender = user.get("gender", "Male")
    
    # Stats
    systolic = stats.get("bp_systolic", 120)
    diastolic = stats.get("bp_diastolic", 80)
    sugar = stats.get("sugar_level", 90)
    weight = stats.get("weight", 70)
    height = stats.get("height", 170) # cm
    
    # BMI Calculation
    bmi = 0
    if height > 0:
        bmi = weight / ((height/100) ** 2)
    bmi = round(bmi, 1)
    
    bmi_category = "Normal"
    if bmi < 18.5: bmi_category = "Underweight"
    elif bmi >= 25 and bmi < 30: bmi_category = "Overweight"
    elif bmi >= 30: bmi_category = "Obese"

    # --- Risk Logic (Simplified Models) ---
    
    # 1. Heart Disease Risk (Framingham-ish simplified)
    heart_score = 0
    heart_factors = []
    
    if age > 45: 
        heart_score += 2
        heart_factors.append("Age > 45")
    if gender.lower() == "male": 
        heart_score += 1
    if systolic > 140 or diastolic > 90: 
        heart_score += 3
        heart_factors.append("High Blood Pressure")
    if sugar > 126: # Diabetes
        heart_score += 2
        heart_factors.append("High Blood Sugar")
    if bmi >= 30:
        heart_score += 1
        heart_factors.append("Obesity")
        
    # Normalize score to 0-100 roughly
    heart_risk_percent = min(heart_score * 10, 95)
    heart_level = "Low"
    if heart_risk_percent > 30: heart_level = "Moderate"
    if heart_risk_percent > 60: heart_level = "High"

    # 2. Diabetes Risk
    diabetes_score = 0
    diabetes_factors = []
    
    if age > 45: 
        diabetes_score += 2
        diabetes_factors.append("Age > 45")
    if bmi >= 25: 
        diabetes_score += 3
        diabetes_factors.append("Overweight/Obese")
    if systolic > 140:
        diabetes_score += 1
        diabetes_factors.append("High BP")
    
    diabetes_risk_percent = min(diabetes_score * 12, 95)
    diabetes_level = "Low"
    if diabetes_risk_percent > 30: diabetes_level = "Moderate"
    if diabetes_risk_percent > 60: diabetes_level = "High"
    
    # 3. Hypertension Risk
    ht_score = 0
    ht_factors = []
    
    if systolic >= 130 or diastolic >= 80:
        ht_score += 5
        ht_factors.append("Elevated BP Reading")
    if bmi >= 25:
        ht_score += 2
        ht_factors.append("Overweight")
    if age > 60:
        ht_score += 1
        ht_factors.append("Age > 60")
        
    ht_risk_percent = min(ht_score * 10, 95)
    ht_level = "Low"
    if ht_risk_percent > 30: ht_level = "Moderate"
    if ht_risk_percent > 60: ht_level = "High"

    risks = {
        "heart": {"score": heart_risk_percent, "level": heart_level, "factors": heart_factors},
        "diabetes": {"score": diabetes_risk_percent, "level": diabetes_level, "factors": diabetes_factors},
        "hypertension": {"score": ht_risk_percent, "level": ht_level, "factors": ht_factors}
    }

    # Get symptom history
    symptoms = list(db.symptoms.find({"patient_id": user_id}).sort("timestamp", -1).limit(5))

    return render_template("health_risks.html", stats=stats, risks=risks, bmi=bmi, bmi_category=bmi_category, available_dates=available_dates, selected_date=selected_date, symptoms=symptoms)

# -----------------------
# ML Prediction Routes
# -----------------------

@app.route("/predict_diabetes", methods=["POST"])
@login_required("patient")
def predict_diabetes():
    """
    Predict diabetes probability using the trained ML model.
    Expected features may vary - adjust based on your model training.
    """
    try:
        if not diabetes_model or not diabetes_scaler:
            return jsonify({"ok": False, "msg": "Diabetes model not loaded"})
        
        data = request.get_json()
        
        # Features: gender, age, hypertension, heart_disease_history, smoking_history, bmi, HbA1c_level, blood_glucose
        features = [
            float(data.get('gender', 0)),  # 0=Female, 1=Male
            float(data.get('age', 0)),
            float(data.get('hypertension', 0)),  # 0=No, 1=Yes
            float(data.get('heart_disease_history', 0)),  # 0=No, 1=Yes
            float(data.get('smoking_history', 0)),  # Encoded value
            float(data.get('bmi', 0)),
            float(data.get('HbA1c_level', 0)),
            float(data.get('blood_glucose', 0))
        ]
        
        # Scale features
        features_scaled = diabetes_scaler.transform([features])
        
        # Predict
        prediction = diabetes_model.predict(features_scaled)[0]
        probability = diabetes_model.predict_proba(features_scaled)[0]
        
        # Get probability of having diabetes (class 1)
        diabetes_prob = float(probability[1] * 100)
        
        return jsonify({
            "ok": True,
            "prediction": int(prediction),
            "probability": round(diabetes_prob, 2),
            "message": f"{'High' if prediction == 1 else 'Low'} risk of diabetes",
            "details": f"{diabetes_prob:.1f}% chance of having diabetes"
        })
        
    except Exception as e:
        print(f"Error in diabetes prediction: {e}")
        traceback.print_exc()
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/predict_heart", methods=["POST"])
@login_required("patient")
def predict_heart():
    """
    Predict heart disease probability using the trained ML model.
    """
    try:
        if not heart_model or not heart_scaler:
            return jsonify({"ok": False, "msg": "Heart disease model not loaded"})
        
        data = request.get_json()
        
        # Features: age, gender, cholesterol, blood_pressure, heart_rate, smoking_history, alcohol_intake, exercise_hours, diabetes, obesity, stress_level, blood_sugar, exercise, family_history
        features = [
            float(data.get('age', 0)),
            float(data.get('gender', 0)),  # 0=Female, 1=Male
            float(data.get('cholesterol', 0)),
            float(data.get('blood_pressure', 0)),
            float(data.get('heart_rate', 0)),
            float(data.get('smoking_history', 0)),  # Encoded value
            float(data.get('alcohol_intake', 0)),
            float(data.get('exercise_hours', 0)),
            float(data.get('diabetes', 0)),  # 0=No, 1=Yes
            float(data.get('obesity', 0)),  # 0=No, 1=Yes
            float(data.get('stress_level', 0)),
            float(data.get('blood_sugar', 0)),
            float(data.get('exercise', 0)),  # Encoded value
            float(data.get('family_history', 0))  # 0=No, 1=Yes
        ]
        
        # Scale features
        features_scaled = heart_scaler.transform([features])
        
        # Predict
        prediction = heart_model.predict(features_scaled)[0]
        probability = heart_model.predict_proba(features_scaled)[0]
        
        # Get probability of having heart disease (class 1)
        heart_prob = float(probability[1] * 100)
        
        return jsonify({
            "ok": True,
            "prediction": int(prediction),
            "probability": round(heart_prob, 2),
            "message": f"{'High' if prediction == 1 else 'Low'} risk of heart disease",
            "details": f"{heart_prob:.1f}% chance of having heart disease"
        })
        
    except Exception as e:
        print(f"Error in heart disease prediction: {e}")
        traceback.print_exc()
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/predict_stroke", methods=["POST"])
@login_required("patient")
def predict_stroke():
    """
    Predict stroke/hypertension probability using the trained ML model.
    """
    try:
        if not stroke_model or not stroke_scaler:
            return jsonify({"ok": False, "msg": "Stroke model not loaded"})
        
        data = request.get_json()
        
        # Features: gender, age, hypertension, heart_disease, ever_married, work_type, residence, avg_glucose, bmi, smoking_history
        features = [
            float(data.get('gender', 0)),  # 0=Female, 1=Male
            float(data.get('age', 0)),
            float(data.get('hypertension', 0)),  # 0=No, 1=Yes
            float(data.get('heart_disease', 0)),  # 0=No, 1=Yes
            float(data.get('ever_married', 0)),  # 0=No, 1=Yes
            float(data.get('work_type', 0)),  # Encoded value
            float(data.get('residence', 0)),  # 0=Rural, 1=Urban
            float(data.get('avg_glucose', 0)),
            float(data.get('bmi', 0)),
            float(data.get('smoking_history', 0))  # Encoded value
        ]
        
        # Scale features
        features_scaled = stroke_scaler.transform([features])
        
        # Predict
        prediction = stroke_model.predict(features_scaled)[0]
        probability = stroke_model.predict_proba(features_scaled)[0]
        
        # Get probability of having stroke (class 1)
        stroke_prob = float(probability[1] * 100)
        
        return jsonify({
            "ok": True,
            "prediction": int(prediction),
            "probability": round(stroke_prob, 2),
            "message": f"{'High' if prediction == 1 else 'Low'} risk of stroke/hypertension",
            "details": f"{stroke_prob:.1f}% chance of stroke/hypertension"
        })
        
    except Exception as e:
        print(f"Error in stroke prediction: {e}")
        traceback.print_exc()
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/timeline")
@login_required("patient")
def timeline():
    user_id = session["user_id"]
    events = []

    # 1. Appointments
    appts = list(bookings_col.find({"patient_id": user_id}))
    for a in appts:
        doc_name = "Doctor"
        doc = doctors_col.find_one({"_id": ObjectId(a["doctor_id"])})
        if doc: doc_name = doc.get("name", "Doctor")
        
        events.append({
            "type": "Appointment",
            "date": datetime.strptime(f"{a['date']} {a['time']}", "%Y-%m-%d %H:%M"),
            "title": f"Appointment with {doc_name}",
            "details": f"Scheduled for {a['time']}",
            "link": None,
            "date_str": f"{a['date']} {a['time']}"
        })

    # 2. Prescriptions
    pres = list(db["prescriptions"].find({"patient_id": user_id}))
    for p in pres:
        events.append({
            "type": "Prescription",
            "date": p.get("created_at", datetime.utcnow()),
            "title": "Prescription Uploaded",
            "details": f"{len(p.get('medicines', []))} medicines identified",
            "link": url_for('show_prescription_result', pres_id=p['_id']),
            "date_str": p.get("created_at", datetime.utcnow()).strftime("%Y-%m-%d %H:%M")
        })

    # 3. Tests (Diagnostic Bookings)
    tests = list(db["diagnostic_bookings"].find({"patient_id": user_id}))
    for t in tests:
        # Handle date/time parsing carefully as format might vary
        try:
            dt_str = f"{t['date']} {t['time']}"
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        except:
            dt = t.get("created_at", datetime.utcnow())

        events.append({
            "type": "Test",
            "date": dt,
            "title": f"Test: {t.get('test_name')}",
            "details": f"Center: {t.get('center_name')}",
            "link": "/my_diagnostic_bookings",
            "date_str": dt.strftime("%Y-%m-%d %H:%M")
        })

    # 4. Orders
    orders = list(orders_col.find({"patient_id": user_id}))
    for o in orders:
        events.append({
            "type": "Order",
            "date": o.get("order_time", datetime.utcnow()),
            "title": f"Order #{str(o['_id'])[-6:]}",
            "details": f"Status: {o.get('status')} - ${o.get('total_amount', 0):.2f}",
            "link": "/my_orders",
            "date_str": o.get("order_time", datetime.utcnow()).strftime("%Y-%m-%d %H:%M")
        })

    # 5. Chats (Last message per doctor?) - Let's just show recent messages or grouped?
    # Showing every message might clutter. Let's show "Chat Session" if we can group.
    # For simplicity, let's show individual messages but maybe limit? 
    # Or better: "Chat with Dr. X" for each message.
    msgs = list(pat_msg_col.find({"patient_id": user_id}))
    for m in msgs:
        doc = doctors_col.find_one({"_id": ObjectId(m["doctor_id"])})
        doc_name = doc.get("name", "Doctor") if doc else "Doctor"
        events.append({
            "type": "Chat",
            "date": m.get("timestamp", datetime.utcnow()),
            "title": f"Sent message to {doc_name}",
            "details": m.get("msg", "")[:50] + "...",
            "link": None,
            "date_str": m.get("timestamp", datetime.utcnow()).strftime("%Y-%m-%d %H:%M")
        })

    # Sort by date descending
    events.sort(key=lambda x: x["date"], reverse=True)

    return render_template("timeline.html", events=events)

@app.route("/ai_suggestions")
@login_required("patient")
def ai_suggestions():
    pid = session["user_id"]

    # Get last 7 stats
    docs = list(db.health_stats.find({"patient_id": pid}).sort("date", -1).limit(7))

    if not docs:
        return jsonify({"reply": "No health stats available to analyze."})

    # Prepare text summary for AI
    summary = "Here are my last recorded health stats:\n\n"
    for d in docs:
        summary += (
            f"Date: {d['date']}, "
            f"Systolic: {d['bp_systolic']}, "
            f"Diastolic: {d['bp_diastolic']}, "
            f"Sugar: {d['sugar_level']}, "
            f"Heart Rate: {d['heart_rate']}, "
            f"Weight: {d['weight']} kg\n"
        )

    prompt = f"""
You are a medical expert AI.

Analyze the following patient health records and provide:

1. Health analysis summary  
2. Abnormal or dangerous patterns  
3. Suggestions for improvement  
4. Lifestyle improvements  
5. Diet recommendations  
6. Whether any doctor visit is advised  

Records:
{summary}

Give the response in simple, clear bullet points.
"""

    ok, reply = query_gemini(prompt)

    if not ok:
        reply = "AI service is currently unavailable."

    return jsonify({"reply": reply})


@app.route("/get_stats_data")
@login_required("patient")
def get_stats_data():
    pid = session["user_id"]
    
    start_date = request.args.get("start")
    end_date = request.args.get("end")
    
    query = {"patient_id": pid}
    
    if start_date and end_date:
        query["date"] = {"$gte": start_date, "$lte": end_date}
    elif start_date:
        query["date"] = {"$gte": start_date}
    elif end_date:
        query["date"] = {"$lte": end_date}

    docs = list(db.health_stats.find(query).sort("date", 1))

    cleaned = [
        {
            "date": d["date"],
            "bp_systolic": d["bp_systolic"],
            "bp_diastolic": d["bp_diastolic"],
            "sugar_level": d["sugar_level"],
            "heart_rate": d["heart_rate"],
            "weight": d["weight"]
        }
        for d in docs
    ]

    return jsonify({"ok": True, "data": cleaned})

@app.route("/early_records")
@login_required("patient")
def early_records():
    pid = session["user_id"]
    records = list(db.health_stats.find({"patient_id": pid}).sort("date", -1))
    return render_template("early_records.html", records=records)


@app.route("/stats_ai_insights", methods=["POST"])
@login_required("patient")
def stats_ai_insights():
    data = request.get_json()
    stats_json = data.get("stats_json", "")

    prompt = f"""
    You are a medical analytics AI.

    The following is the user's recent health statistics in JSON:
    {stats_json}

    Analyse trends in:
    - Blood pressure
    - Sugar levels
    - Heart rate
    - Weight

    Provide:
    - Summary of their health pattern
    - Whether their readings indicate risk
    - What lifestyle or medical advice you recommend
    - 3 action points to improve health

    Keep the tone encouraging and medical-grade.
    """

    ok, reply = query_gemini(prompt)

    if ok:
        return jsonify({"ok": True, "insights": reply})
    else:
        return jsonify({"ok": False, "insights": reply})


# -----------------------
# Symptom analysis (AI)
# -----------------------
@app.route("/symptom")
@login_required("patient")
def symptom_page():
    return render_template("symptom.html")

@app.route("/analyze_symptoms", methods=["POST"])
@login_required("patient")
def analyze_symptoms():
    try:
        data = request.get_json(force=True)
        user_text = data.get("symptoms", "").strip()
        if not user_text:
            return jsonify({"ok": False, "message": "Empty symptoms"})
        prompt = f"""
A patient describes the following symptoms: {user_text}
Please produce:
- Diseases: list up to top 3 comma-separated (short names).
- Severity: one of RED, YELLOW, GREEN only (caps).
- Explanation: a 1-2 sentence summary.

Format exactly:
Diseases: ...
Severity: RED/YELLOW/GREEN
Explanation: ...
"""
        ok, payload = query_flan(prompt)
        if ok:
            model_text = payload.strip()
            diseases = ""
            severity = ""
            explanation = ""
            for line in model_text.splitlines():
                line = line.strip()
                if line.lower().startswith("diseases:"):
                    diseases = line.split(":", 1)[1].strip()
                elif line.lower().startswith("severity:"):
                    severity = line.split(":", 1)[1].strip().upper()
                elif line.lower().startswith("explanation:"):
                    explanation = line.split(":", 1)[1].strip()
            if severity not in {"RED", "YELLOW", "GREEN"}:
                lt = model_text.lower()
                if any(w in lt for w in ["chest pain","difficulty breathing","unconscious","severe bleeding"]):
                    severity = "RED"
                elif any(w in lt for w in ["fever","cough","dizziness","moderate pain"]):
                    severity = "YELLOW"
                else:
                    severity = "GREEN"
            if not explanation:
                explanation = model_text
            
            # Save to DB
            symptom_doc = {
                "patient_id": session["user_id"],
                "symptoms": user_text,
                "diseases": diseases,
                "severity": severity,
                "explanation": explanation,
                "timestamp": datetime.utcnow()
            }
            db.symptoms.insert_one(symptom_doc)

            return jsonify({"ok": True, "diseases": diseases, "severity": severity, "explanation": explanation, "raw": model_text})
        else:
            # fallback simple heuristic
            text = user_text.lower()
            heur_diseases = []
            if any(w in text for w in ["fever","cough","sore throat","body ache"]):
                heur_diseases.append("Viral infection / Influenza")
            if any(w in text for w in ["headache","migraine"]):
                heur_diseases.append("Tension headache / Migraine")
            heur_diseases = ", ".join(heur_diseases) if heur_diseases else "General viral/unknown"
            heur_sev = "RED" if any(w in text for w in ["chest pain","difficulty breathing"]) else ("YELLOW" if any(w in text for w in ["fever","dizziness"]) else "GREEN")
            heur_expl = f"Model unavailable: {payload}. Heuristic suggestion: {heur_diseases} — severity {heur_sev}."
            
            # Save fallback to DB
            symptom_doc = {
                "patient_id": session["user_id"],
                "symptoms": user_text,
                "diseases": heur_diseases,
                "severity": heur_sev,
                "explanation": heur_expl,
                "timestamp": datetime.utcnow()
            }
            db.symptoms.insert_one(symptom_doc)

            return jsonify({"ok": True, "diseases": heur_diseases, "severity": heur_sev, "explanation": heur_expl, "raw": payload})
    except Exception:
        app.logger.exception("analyze_symptoms failed")
        return jsonify({"ok": False, "message": "Server error"})

@app.route("/early_symptoms")
@login_required("patient")
def early_symptoms():
    user_id = session["user_id"]
    history = list(db.symptoms.find({"patient_id": user_id}).sort("timestamp", -1))
    return render_template("early_symptoms.html", history=history)

# -----------------------
# Diet planner
# -----------------------
@app.route("/diet_planner")
@login_required("patient")
def diet_planner():
    user_id = session.get("user_id")
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    today_log = foods_col.find_one({"user_id": user_id, "date": today_str})
    history = list(foods_col.find({"user_id": user_id}).sort("date", -1).limit(7))
    return render_template("diet_planner.html", today=today_log, history=history, today_date=today_str)

@app.route("/add_food", methods=["POST"])
@login_required("patient")
def add_food():
    data = request.get_json()
    food_name = data.get("food_name")
    calories = int(data.get("calories", 0))
    meal_type = data.get("meal_type")
    user_id = session.get("user_id")
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    if not food_name:
        return jsonify({"ok": False, "msg": "Food name required"})
    foods_col.update_one(
        {"user_id": user_id, "date": today_str},
        {
            "$push": {"items": {"food": food_name, "calories": calories, "meal": meal_type, "time": datetime.utcnow().strftime("%H:%M")}},
            "$inc": {"total_calories": calories},
            "$setOnInsert": {"created_at": datetime.utcnow()}
        },
        upsert=True
    )
    return jsonify({"ok": True})

@app.route("/book_appointment", methods=["GET", "POST"])
@login_required("patient")
def book_appointment():

    # ---------------- GET: Load Page ----------------
    if request.method == "GET":
        doctors = list(doctors_col.find({}))
        for d in doctors:
            d["_id"] = str(d["_id"])
        return render_template("book_appointment.html", doctors=doctors)

    # ---------------- POST: Save Booking ----------------
    data = request.get_json()
    doctor_id = str(data.get("doctor_id"))
    date_str = data.get("date")
    time_slot = data.get("time")
    patient_id = session["user_id"]

    # Validation
    if not (doctor_id and date_str and time_slot):
        return jsonify({"ok": False, "msg": "Missing fields"})

    # Allowed slots
    allowed_slots = [
        "09:00","10:00","11:00","12:00","13:00",
        "14:00","15:00","16:00","17:00","18:00"
    ]
    if time_slot not in allowed_slots:
        return jsonify({"ok": False, "msg": "Invalid time slot"})

    # DEBUG PRINTS
    print("DEBUG doctor_id=", doctor_id)
    print("DEBUG date_str=", date_str)
    print("DEBUG time_slot=", time_slot)
    print("DEBUG existing bookings =", list(bookings_col.find({})))

    # Check for existing booking
    exists = bookings_col.find_one({
        "doctor_id": doctor_id,
        "date": date_str,
        "time": time_slot
    })

    if exists:
        # Suggest alternative doctors
        other_docs = list(doctors_col.find({"_id": {"$ne": ObjectId(doctor_id)}}))
        available_others = []
        for doc in other_docs:
            doc_id = str(doc["_id"])
            taken = bookings_col.find_one({
                "doctor_id": doc_id,
                "date": date_str,
                "time": time_slot
            })
            if not taken:
                available_others.append({"id": doc_id, "name": doc["name"]})

        # Next day suggestion
        try:
            curr_date = datetime.strptime(date_str, "%Y-%m-%d")
            next_date = curr_date + timedelta(days=1)
            next_day_str = next_date.strftime("%Y-%m-%d")

            next_taken = bookings_col.find_one({
                "doctor_id": doctor_id,
                "date": next_day_str,
                "time": time_slot
            })
            next_day_slot = "" if next_taken else time_slot
            if next_taken:
                next_day_str = "Fully Booked"

        except:
            next_day_str = "Next Day"
            next_day_slot = time_slot

        return jsonify({
            "ok": False,
            "msg": "Slot already booked",
            "alternatives": {
                "others": available_others,
                "next_day": next_day_str,
                "next_day_slot": next_day_slot
            }
        })

    # -------- Insert Booking --------
    booking_id = str(ObjectId())
    bookings_col.insert_one({
        "_id": ObjectId(booking_id),
        "doctor_id": doctor_id,
        "patient_id": patient_id,
        "date": date_str,
        "time": time_slot,
        "created_at": datetime.utcnow()
    })

    return jsonify({
        "ok": True,
        "msg": "Appointment booked successfully!",
        "booking_id": booking_id
    })


@app.route("/get_doctors")
def get_doctors():
    spec = request.args.get("spec", "")
    docs = list(doctors_col.find({
        "specialisation": {"$regex": spec, "$options": "i"}
    }))
    for d in docs:
        d['_id'] = str(d['_id'])
    return jsonify({"doctors": docs})



@app.route("/home")
def home():
    user = current_user()
    if user:
        if session.get("role") == "patient":
            return redirect(url_for("patient_dashboard"))
        elif session.get("role") == "doctor":
            return redirect(url_for("doctor_dashboard"))
    return render_template("home.html", user=user)

@app.route("/patient_dashboard")
@login_required("patient")
def patient_dashboard():
    user = current_user()
    return render_template("patient.html", user=user)

@app.route("/doctor_dashboard")
@login_required("doctor")
def doctor_dashboard():
    user = current_user()
    return render_template("doctor.html", user=user)

@app.route("/virtual_assistant", methods=["GET", "POST"])
@login_required("patient")
def virtual_assistant():
    if request.method == "GET":
        return render_template("virtual_assistant.html")
    
    data = request.get_json()
    msg = data.get("msg", "").strip()
    if not msg:
        return jsonify({"ok": False, "msg": "Empty message"})
    
    # Use Gemini
    prompt = f"You are a helpful virtual health assistant. The user asks: {msg}"
    ok, reply = query_gemini(prompt)
    if ok:
        return jsonify({"ok": True, "reply": reply})
    else:
        return jsonify({"ok": False, "msg": "AI service unavailable"})

@app.route("/anonymous_chat")
def anonymous_chat():
    return render_template("chat.html")

@app.route("/get_specialisations")
def get_specialisations():
    specs = doctors_col.distinct("specialisation")  # no filtering
    return jsonify({"specialisations": specs})

# -----------------------------
# RAG REPORTS FEATURE
# -----------------------------

@app.route("/report_analysis")
@login_required("patient")
def report_analysis():
    if not RAG_AVAILABLE:
        flash("Report analysis feature is currently unavailable. Please contact support.", "error")
        return redirect(url_for("patient_dashboard"))
    user_id = session["user_id"]
    # Get user reports
    user_reports = list(reports_col.find({"user_id": user_id}).sort("created_at", -1))
    return render_template("report_analysis.html", reports=user_reports, role="patient") 

@app.route("/upload_report", methods=["POST"])
@login_required("patient")
def upload_report():
    # Check for file in either 'file' or 'report_file' field
    if "file" not in request.files and "report_file" not in request.files:
        flash("No file part", "error")
        return redirect(url_for("report_analysis"))
    
    file = request.files.get("file") or request.files.get("report_file")
    
    if not file or file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for("report_analysis"))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{session['user_id']}_{timestamp}_{filename}"
        filepath = os.path.join(REPORT_UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Create DB Entry
        report_data = {
            "user_id": session["user_id"],
            "original_name": filename,
            "filepath": filepath,
            "created_at": datetime.utcnow(),
            "processed": False,
            "processing_status": "Processing...",
            "error_message": None
        }
        report_id = reports_col.insert_one(report_data).inserted_id
        
        try:
            if not RAG_AVAILABLE:
                reports_col.update_one({"_id": report_id}, {"$set": {"processing_status": "Failed", "error_message": "RAG system unavailable"}})
                flash("Report analysis system is currently unavailable.", "error")
                return redirect(url_for("report_analysis"))
            
            success, msg = rag_utils.process_report(filepath, str(report_id))
            if success:
                reports_col.update_one({"_id": report_id}, {"$set": {"processed": True, "processing_status": "Completed"}})
                flash("Report uploaded and processed successfully!", "success")
                return redirect(url_for("report_chat", report_id=report_id))
            else:
                reports_col.update_one({"_id": report_id}, {"$set": {"processing_status": "Failed", "error_message": msg}})
                flash(f"Error processing report: {msg}", "error")
                return redirect(url_for("report_analysis"))
                
        except Exception as e:
            reports_col.update_one({"_id": report_id}, {"$set": {"processing_status": "Failed", "error_message": str(e)}})
            flash(f"System error: {str(e)}", "error")
            return redirect(url_for("report_analysis"))
            
    flash("Invalid file type (PDF only)", "error")
    return redirect(url_for("report_analysis"))

@app.route("/report_chat/<report_id>")
@login_required("patient")
def report_chat(report_id):
    if not RAG_AVAILABLE:
        flash("Report chat feature is currently unavailable.", "error")
        return redirect(url_for("report_analysis"))
    report = reports_col.find_one({"_id": ObjectId(report_id), "user_id": session["user_id"]})
    if not report:
        flash("Report not found", "error")
        return redirect(url_for("report_analysis"))
        
    return render_template("report_chat.html", report_id=str(report_id), report=report)

@app.route("/api/qa_report", methods=["POST"])
@login_required("patient")
def qa_report():
    if not RAG_AVAILABLE:
        return jsonify({"ok": False, "msg": "RAG system unavailable"})
    
    data = request.get_json()
    report_id = data.get("report_id")
    question = data.get("question")
    language = data.get("language", "English")
    empathetic = data.get("empathetic", False)
    
    if not report_id or not question:
        return jsonify({"ok": False, "msg": "Missing data"})
        
    # Verify ownership
    report = reports_col.find_one({"_id": ObjectId(report_id), "user_id": session["user_id"]})
    if not report:
        return jsonify({"ok": False, "msg": "Report not found"})
    
    question_lower = question.lower()
    
    # Determine if this is a targeted query (abnormal values, summary) or general question
    is_targeted_query = any(word in question_lower for word in [
        'abnormal', 'problem', 'issue', 'concern', 'high', 'low', 'outside', 'range',
        'summarize', 'summary', 'overview'
    ])
    
    if is_targeted_query:
        # Use enhanced RAG for targeted queries
        answer, docs = rag_utils.ask_report(report_id, question, language=language, empathetic=empathetic)
        # Get Recommendations based on the answer
        recommendations = rag_utils.recommend_next_steps(answer)
    else:
        # Use general Q&A for comprehensive answers
        answer = rag_utils.ask_general_question(report_id, question, language=language, empathetic=empathetic)
        # For general questions, don't force recommendations unless keywords are detected
        recommendations = rag_utils.recommend_next_steps(answer)
    
    return jsonify({
        "ok": True, 
        "answer": answer, 
        "recommendations": recommendations,
    })

@app.route("/delete_report/<report_id>")
@login_required("patient")
def delete_report(report_id):
    reports_col.delete_one({"_id": ObjectId(report_id), "user_id": session["user_id"]})
    flash("Report deleted", "success")
    return redirect(url_for("report_analysis"))
    
@app.route("/download_report/<report_id>")
@login_required("patient")
def download_report(report_id):
    from flask import send_file
    report = reports_col.find_one({"_id": ObjectId(report_id), "user_id": session["user_id"]})
    if report and os.path.exists(report["filepath"]):
        return send_file(report["filepath"], as_attachment=True)
    flash("File not found", "error")
    return redirect(url_for("report_analysis"))

# -----------------------
# Run server
# -----------------------
