from app import patients_col, doctors_col, db
from datetime import datetime

print("DEBUG: Testing database connection from app context...")

try:
    # Test Patient Insertion
    patient_doc = {
        "name": "Debug Patient",
        "username": "debug_patient",
        "email": "debug_patient@example.com",
        "role": "patient",
        "created_at": datetime.utcnow()
    }
    print("Attempting to insert patient...")
    res_p = patients_col.insert_one(patient_doc)
    print(f"SUCCESS: Patient inserted with ID: {res_p.inserted_id}")

    # Verify retrieval
    found_p = patients_col.find_one({"_id": res_p.inserted_id})
    if found_p:
        print("SUCCESS: Patient retrieved from DB.")
    else:
        print("FAILURE: Patient NOT found after insertion.")

    # Test Doctor Insertion
    doctor_doc = {
        "name": "Debug Doctor",
        "username": "debug_doctor",
        "email": "debug_doctor@example.com",
        "role": "doctor",
        "created_at": datetime.utcnow()
    }
    print("Attempting to insert doctor...")
    res_d = doctors_col.insert_one(doctor_doc)
    print(f"SUCCESS: Doctor inserted with ID: {res_d.inserted_id}")

    # Verify retrieval
    found_d = doctors_col.find_one({"_id": res_d.inserted_id})
    if found_d:
        print("SUCCESS: Doctor retrieved from DB.")
    else:
        print("FAILURE: Doctor NOT found after insertion.")

    # Clean up
    patients_col.delete_one({"_id": res_p.inserted_id})
    doctors_col.delete_one({"_id": res_d.inserted_id})
    print("Cleaned up debug documents.")

except Exception as e:
    print(f"ERROR: {e}")
