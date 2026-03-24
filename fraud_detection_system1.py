import cv2
from pyzbar.pyzbar import decode
import urllib.parse
import sqlite3
from datetime import datetime
import random
import time
import pickle
import numpy as np

# ---------------- LOAD MODELS ----------------
model = pickle.load(open("fraud_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
iso_model = pickle.load(open("iso_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- DATABASE ----------------
conn = sqlite3.connect("fraud.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions(
id INTEGER PRIMARY KEY,
merchant TEXT,
amount REAL,
risk_score REAL,
decision TEXT,
drift INTEGER,
timestamp TEXT
)
""")
conn.commit()

# ---------------- DRIFT STORAGE ----------------
recent_risks = []
WINDOW_SIZE = 5

def detect_drift(current_risk):
    global recent_risks

    drift_flag = 0

    if len(recent_risks) >= WINDOW_SIZE:
        avg_risk = sum(recent_risks) / len(recent_risks)

        if abs(current_risk - avg_risk) > 0.3:
            print("⚠ Concept Drift Detected!")
            drift_flag = 1

    recent_risks.append(current_risk)

    if len(recent_risks) > WINDOW_SIZE:
        recent_risks.pop(0)

    return drift_flag

# ---------------- OTP SYSTEM ----------------
def otp_verification():
    while True:
        otp = random.randint(100000, 999999)
        print("\n📩 OTP SENT:", otp)

        start = time.time()
        attempts = 0

        while attempts < 3:
            remaining = int(180 - (time.time() - start))

            if remaining <= 0:
                print("⏰ OTP expired")
                break

            user = input("Enter OTP: ")
            if user == str(otp):
                return True
            else:
                attempts += 1

        if input("Resend OTP? (yes/no): ") != "yes":
            return False

# ---------------- PIN ----------------
def verify_pin():
    return input("Enter PIN: ") == "1234"

# ---------------- SECURITY ----------------
def security_questions():
    questions = [
        ("Pet name? ", "tommy"),
        ("Birth city? ", "hyderabad"),
        ("Favorite food? ", "biryani")
    ]

    selected = random.sample(questions, 2)

    for q, ans in selected:
        if input(q).lower() != ans:
            return False

    return True

# ---------------- MASKED OTP ----------------
def mask_number(num):
    return num[0] + "XXXXXX" + num[-2:]

def backup_otp():
    number = "9123456789"
    masked = mask_number(number)

    otp = random.randint(100000, 999999)
    print(f"OTP to {masked}:", otp)

    return input("Enter Backup OTP: ") == str(otp)

# ---------------- EMAIL OTP ----------------
def email_otp():
    otp = random.randint(100000, 999999)
    print("Email OTP:", otp)
    return input("Enter Email OTP: ") == str(otp)

# ---------------- BEHAVIOR ----------------
def behavioral_risk(amount):
    if amount > 5000:
        return 0.8
    elif amount > 2000:
        return 0.5
    else:
        return 0.2

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    for obj in decode(frame):
        qr = obj.data.decode("utf-8")

        if "upi://" in qr:
            params = urllib.parse.parse_qs(qr.split("?")[1])
            merchant = params.get("pa", ["unknown"])[0]

            amount = float(input("Enter amount: "))

            features = np.zeros((1, 30))
            features[0][-1] = amount
            scaled = scaler.transform(features)

            prob = (0.6 * model.predict_proba(scaled)[0][1] +
                    0.4 * rf_model.predict_proba(scaled)[0][1])

            anomaly = abs(iso_model.decision_function(scaled)[0])
            behavior = behavioral_risk(amount)
            amount_factor = min(amount / 10000, 1)

            risk_score = min((0.2*prob + 0.2*anomaly + 0.4*behavior + 0.4*amount_factor),1)

            # ---------------- DRIFT ----------------
            drift_flag = detect_drift(risk_score)

            # ---------------- DECISION ----------------
            if risk_score < 0.4:
                decision = "APPROVE"

            elif risk_score < 0.7:
                if otp_verification() and verify_pin():
                    decision = "APPROVE"
                else:
                    decision = "BLOCK"

            else:
                if (otp_verification() and verify_pin() and
                    security_questions() and backup_otp() and email_otp()):
                    decision = "APPROVE"
                else:
                    decision = "BLOCK"

            print("Decision:", decision)

            # SAVE
            cursor.execute("""
            INSERT INTO transactions VALUES(NULL,?,?,?,?,?,?)
            """,(merchant,amount,risk_score,decision,drift_flag,str(datetime.now())))
            conn.commit()

    cv2.imshow("Scanner", frame)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()