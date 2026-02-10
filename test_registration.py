import requests
import random
import string

BASE_URL = "http://127.0.0.1:5000"

def random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

username = f"testuser_{random_string()}"
email = f"{username}@example.com"
password = "testpassword123"

print(f"Attempting to register user: {username}")

payload = {
    "name": "Test User",
    "username": username,
    "email": email,
    "password": password,
    "role": "patient",
    "age": "25",
    "gender": "Male",
    "address": "123 Test St",
    "phone": "1234567890"
}

try:
    # Use a session to persist cookies
    s = requests.Session()
    
    # GET register page first (to check if it loads)
    r = s.get(f"{BASE_URL}/register")
    print(f"GET /register status: {r.status_code}")
    
    # POST registration
    r = s.post(f"{BASE_URL}/register", data=payload, allow_redirects=False)
    print(f"POST /register status: {r.status_code}")
    
    if r.status_code == 302:
        print(f"Redirect location: {r.headers.get('Location')}")
        if "patient_dashboard" in r.headers.get('Location', ''):
            print("SUCCESS: Redirected to patient_dashboard")
        else:
            print("WARNING: Redirected to unexpected location")
    else:
        print("FAILURE: Did not redirect (302).")
        print("Response text snippet:", r.text[:500])

except Exception as e:
    print(f"ERROR: {e}")
