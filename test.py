from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Load environment variables
load_dotenv()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")

client = MongoClient("mongodb+srv://sharmila:123456_sharmila@capstone.3xycmpu.mongodb.net/?appName=capstone")
db = client["mydb"]
medicines = db["medicines"]

def send_sendgrid_email(to_email, subject, html_content):
    """Send email via SendGrid"""
    try:
        message = Mail(
            from_email=SENDER_EMAIL,
            to_emails=to_email,
            subject=subject,
            html_content=html_content
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(f"Email sent to {to_email}: {response.status_code}")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def check_waitlist_and_notify(med_name):
    """Check if anyone is waiting for this medicine and notify them"""
    # Find waiters for this medicine (case-insensitive match)
    waiters = list(db["medicine_waitlist"].find({"med_name": {"$regex": f"^{med_name}$", "$options": "i"}}))
    
    print(f"Found {len(waiters)} users waiting for '{med_name}'")
    
    for w in waiters:
        email = w.get("email")
        if email:
            subject = f"Good news! {med_name} is now available"
            html = f"""
            <p>Hello,</p>
            <p>The medicine <strong>{med_name}</strong> you were looking for is now available in our pharmacy.</p>
            <p><a href="http://127.0.0.1:5000/pharmacy">Click here to buy now</a></p>
            """
            if send_sendgrid_email(email, subject, html):
                # Remove from waitlist after notifying
                db["medicine_waitlist"].delete_one({"_id": w["_id"]})
                print(f"✓ Notified {email} about {med_name}")
            else:
                print(f"✗ Failed to notify {email}")

# Add medicine
med = {
    "name": "Oxprelol",
    "description": "Used for high blood pressure and heart-related issues.",
    "price": 85.00,
    "stock": 50,
    "created_at": datetime.utcnow()
}

inserted = medicines.insert_one(med)
print("Inserted ID:", inserted.inserted_id)

# Trigger notification check
print("\nChecking waitlist...")
check_waitlist_and_notify(med["name"])