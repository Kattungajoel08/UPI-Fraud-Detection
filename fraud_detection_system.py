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
timestamp TEXT
)
""")
conn.commit()

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

            print(f"⏳ Time left: {remaining//60}:{remaining%60:02d}")
            user = input("Enter OTP: ")

            if user == str(otp):
                print("✅ OTP Verified")
                return True
            else:
                attempts += 1
                print(f"❌ Wrong OTP. Attempts left: {3-attempts}")

        if input("Resend OTP? (yes/no): ").lower() != "yes":
            return False

# ---------------- PIN ----------------
def verify_pin():
    return input("Enter PIN: ") == "1234"

# ---------------- SECURITY QUESTIONS ----------------
def security_questions():
    questions = [
        ("Pet name? ", "tommy"),
        ("Birth city? ", "hyderabad"),
        ("Favorite food? ", "biryani")
    ]

    selected = random.sample(questions, 2)

    for q, ans in selected:
        if input(q).lower() != ans:
            print("❌ Wrong answer")
            return False

    print("✅ Security Verified")
    return True

# ---------------- MASKED OTP ----------------
def mask_number(num):
    return num[0] + "XXXXXX" + num[-2:]

def backup_otp():
    number = "9123456789"
    masked = mask_number(number)

    otp = random.randint(100000, 999999)
    print(f"\n📱 OTP sent to backup number: {masked}")
    print("OTP:", otp)

    user = input("Enter Backup OTP: ")
    return user == str(otp)

# ---------------- EMAIL OTP ----------------
def email_otp():
    email = "user@gmail.com"

    otp = random.randint(100000, 999999)
    print(f"\n📧 OTP sent to email: {email}")
    print("OTP:", otp)

    user = input("Enter Email OTP: ")
    return user == str(otp)

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
print("📷 Scanner Started")

while True:
    ret, frame = cap.read()

    for obj in decode(frame):

        qr = obj.data.decode("utf-8")

        if "upi://" in qr:
            params = urllib.parse.parse_qs(qr.split("?")[1])
            merchant = params.get("pa", ["unknown"])[0]

            # AMOUNT
            if "am" in params:
                amount = float(params["am"][0])
            else:
                amount = float(input("\nEnter amount: "))

            print(f"\nPay ₹{amount} to {merchant}")
            if input("Confirm (yes/no): ").lower() != "yes":
                print("❌ Cancelled")
                continue

            # ---------------- FEATURES ----------------
            features = np.zeros((1, 30))
            features[0][-1] = amount
            scaled = scaler.transform(features)

            # ---------------- ADAPTIVE ML ----------------
            prob_sgd = model.predict_proba(scaled)[0][1]
            prob_rf = rf_model.predict_proba(scaled)[0][1]
            prob = (0.6 * prob_sgd) + (0.4 * prob_rf)

            # ---------------- ANOMALY ----------------
            anomaly = abs(iso_model.decision_function(scaled)[0])

            # ---------------- BEHAVIOR ----------------
            behavior = behavioral_risk(amount)

            # ---------------- AMOUNT FACTOR ----------------
            amount_factor = min(amount / 10000, 1)

            # ---------------- FINAL RISK SCORE ----------------
            risk_score = (
                0.2 * prob +
                0.2 * anomaly +
                0.4 * behavior +
                0.4 * amount_factor
            )

            risk_score = min(risk_score, 1)

            print("\n--- ANALYSIS ---")
            print("Fraud Prob:", round(prob,2))
            print("Anomaly:", round(anomaly,2))
            print("Behavior:", behavior)
            print("Final Risk Score:", round(risk_score,2))

            # ---------------- DECISION ----------------
            if risk_score < 0.4:
                decision = "APPROVE"

            elif risk_score < 0.7:
                print("\n⚠ MEDIUM RISK → OTP + PIN")
                if otp_verification() and verify_pin():
                    decision = "APPROVE"
                else:
                    decision = "BLOCK"

            else:
                print("\n🚨 HIGH RISK → FULL VERIFICATION")

                if not otp_verification():
                    decision = "BLOCK"

                elif not verify_pin():
                    decision = "BLOCK"

                elif not security_questions():
                    decision = "BLOCK"

                else:
                    print("\n📱 Backup OTP (1 attempt)")
                    if not backup_otp():
                        print("❌ Backup OTP failed → BLOCK")
                        decision = "BLOCK"

                    else:
                        print("\n📧 Email OTP (1 attempt)")
                        if not email_otp():
                            print("❌ Email OTP failed → BLOCK")
                            decision = "BLOCK"
                        else:
                            decision = "APPROVE"

            print("\nFINAL DECISION:", decision)

            # ---------------- SAVE ----------------
            cursor.execute("""
            INSERT INTO transactions VALUES(NULL,?,?,?,?,?)
            """, (merchant, amount, risk_score, decision, str(datetime.now())))
            conn.commit()

    cv2.imshow("UPI Fraud Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()