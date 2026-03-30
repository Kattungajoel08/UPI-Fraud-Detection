import numpy as np
import pickle

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "fraud_model.pkl"), "rb"))
rf_model = pickle.load(open(os.path.join(BASE_DIR, "rf_model.pkl"), "rb"))
iso_model = pickle.load(open(os.path.join(BASE_DIR, "iso_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

initialized = False

def compute_risk(amount, user):
    import sqlite3
    from datetime import datetime
    import numpy as np

    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT amount, time FROM transactions
        WHERE sender=? ORDER BY id DESC LIMIT 10
    """, (user,))
    
    rows = cursor.fetchall()
    conn.close()

    amounts = [r[0] for r in rows]

    # -------- REAL FEATURES --------
    avg = sum(amounts)/len(amounts) if amounts else amount
    max_amt = max(amounts) if amounts else amount
    freq = len(amounts)

    deviation = amount - avg
    ratio = amount / (avg + 1)

    # time gap
    if rows:
        last_time = datetime.strptime(rows[0][1], "%Y-%m-%d %H:%M:%S")
        time_gap = (datetime.now() - last_time).seconds
    else:
        time_gap = 9999

    # -------- FEATURE VECTOR --------
    # -------- FEATURE VECTOR --------
    features = np.array([[amount, avg, max_amt, deviation, ratio, freq, time_gap]])

# -------- PAD TO 30 FEATURES --------
    full_features = np.zeros((1, 30))
    full_features[0][:7] = features

# -------- SCALE (IMPORTANT) --------
    features = scaler.transform(full_features)

    
    # -------- ML + ANOMALY --------
    prob = model.predict_proba(features)[0][1]
    anomaly = abs(iso_model.decision_function(features)[0])

    # -------- FINAL SCORE --------
    score = 0.6 * prob + 0.4 * anomaly

    # behavior boost
    if amount > 2 * avg:
        score += 0.2

    if freq >= 3 and time_gap < 60:
        score += 0.2

    score = float(min(score, 1))

    # -------- RISK --------
    if score < 0.35:
        risk = "LOW"
    elif score < 0.65:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {"risk": risk, "risk_score": score}


def update_model(amount, user, label):
    import sqlite3
    import numpy as np

    conn = sqlite3.connect("fraud.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT amount FROM transactions
        WHERE sender=? ORDER BY id DESC LIMIT 10
    """, (user,))
    
    rows = cursor.fetchall()
    conn.close()

    amounts = [r[0] for r in rows]

    avg = sum(amounts)/len(amounts) if amounts else amount
    deviation = amount - avg
    ratio = amount / (avg + 1)

    features = np.array([[amount, avg, max(amounts) if amounts else amount, deviation, ratio, len(amounts), 0]])

# pad to 30
    full_features = np.zeros((1, 30))
    full_features[0][:7] = features

# scale
    features = scaler.transform(full_features)

    model.partial_fit(features, [label], classes=[0,1])

    pickle.dump(model, open(os.path.join(BASE_DIR, "fraud_model.pkl"), "wb"))