from fastapi import FastAPI
import pickle
import pandas as pd
import sqlite3
from datetime import datetime

# ---------------- INIT ----------------
app = FastAPI()

# ---------------- LOAD MODELS ----------------
rf_model = pickle.load(open("rf_model.pkl", "rb"))
iso_model = pickle.load(open("iso_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- DATABASE ----------------
conn = sqlite3.connect("fraud.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    merchant TEXT,
    amount REAL,
    risk TEXT,
    fraud INTEGER,
    drift INTEGER,
    risk_score REAL,
    time TEXT
)
""")
conn.commit()

# ---------------- FEATURES ----------------
columns = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# ---------------- DRIFT ----------------
WINDOW_SIZE = 20
recent_scores = []

def detect_drift(score):
    global recent_scores

    drift_flag = 0

    if len(recent_scores) >= WINDOW_SIZE:
        avg = sum(recent_scores) / len(recent_scores)

        if abs(score - avg) > 0.05:
            drift_flag = 1

    recent_scores.append(score)

    if len(recent_scores) > WINDOW_SIZE:
        recent_scores.pop(0)

    return drift_flag

# ---------------- ALERTS ----------------
def send_email_alert(score):
    print("\n📧 EMAIL ALERT")
    print(f"⚠ Concept Drift Detected!")
    print(f"Risk Score: {round(score,2)}\n")

def send_sms_alert(score):
    print("\n📱 SMS ALERT")
    print(f"⚠ Drift detected! Score: {round(score,2)}\n")

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"message": "API Running Successfully 🚀"}

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(data: dict):

    try:
        # ---------------- DEBUG ----------------
        print("\nIncoming Data:", data)

        # ---------------- VALIDATION ----------------
        if "features" not in data:
            return {"error": "Missing 'features' in request"}

        features = data["features"]

        print("Feature length:", len(features))

        if len(features) != 30:
            return {"error": f"Expected 30 features, got {len(features)}"}

        # ---------------- PREPARE DATA ----------------
        amount = float(features[-1])
        merchant = data.get("merchant", "Unknown")

        df = pd.DataFrame([features], columns=columns)
        scaled = scaler.transform(df)

        # ---------------- MODEL ----------------
        rf_pred = rf_model.predict(scaled)[0]
        iso_pred = iso_model.predict(scaled)[0]

        fraud = 1 if (rf_pred == 1 or iso_pred == -1) else 0

        # ---------------- RISK SCORE ----------------
        prob = rf_model.predict_proba(scaled)[0][1]
        anomaly = abs(iso_model.decision_function(scaled)[0])

        risk_score = min((0.6 * prob + 0.4 * anomaly), 1)

        # ---------------- DRIFT ----------------
        drift_flag = detect_drift(risk_score)

        # ---------------- ALERT ----------------
        if drift_flag == 1:
            send_email_alert(risk_score)
            send_sms_alert(risk_score)

        # ---------------- RISK LEVEL ----------------
        if amount <= 2000:
            risk = "LOW"
        elif amount <= 5000:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        # ---------------- SAVE ----------------
        cursor.execute("""
        INSERT INTO transactions (merchant, amount, risk, fraud, drift, risk_score, time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            merchant,
            amount,
            risk,
            fraud,
            drift_flag,
            risk_score,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()

        # ---------------- RESPONSE ----------------
        return {
            "fraud": fraud,
            "risk_level": risk,
            "drift": drift_flag,
            "risk_score": round(risk_score, 3)
        }

    except Exception as e:
        print("\n❌ ERROR:", str(e))
        return {"error": str(e)}

# ---------------- GET TRANSACTIONS ----------------
@app.get("/transactions")
def get_transactions():

    cursor.execute("SELECT * FROM transactions ORDER BY id DESC")
    rows = cursor.fetchall()

    return [
        {
            "id": r[0],
            "merchant": r[1],
            "amount": r[2],
            "risk": r[3],
            "fraud": r[4],
            "drift": r[5],
            "risk_score": r[6],
            "time": r[7]
        }
        for r in rows
    ]