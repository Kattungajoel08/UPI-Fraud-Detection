import numpy as np
import pickle
import os
import sqlite3
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

db_path = os.path.join(BASE_DIR, "fraud.db")

model_path = os.path.join(
    BASE_DIR, "fraud_model.pkl"
)

rf_model_path = os.path.join(
    BASE_DIR, "rf_model.pkl"
)

iso_model_path = os.path.join(
    BASE_DIR, "iso_model.pkl"
)

scaler_path = os.path.join(
    BASE_DIR, "scaler.pkl"
)


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    return sqlite3.connect(
        db_path,
        check_same_thread=False
    )


def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            sender TEXT NOT NULL,

            amount REAL NOT NULL,

            time TEXT NOT NULL,

            risk_score REAL,

            risk_level TEXT,

            decision TEXT

        )
    """)

    conn.commit()
    conn.close()


initialize_database()


# ============================================================
# LOAD MODELS
# ============================================================

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(rf_model_path, "rb") as f:
    rf_model = pickle.load(f)

with open(iso_model_path, "rb") as f:
    iso_model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)


# ============================================================
# FEATURE CREATION
# ============================================================

def build_features(
    amount,
    amounts,
    time_gap
):

    avg = (
        sum(amounts) / len(amounts)
        if amounts
        else amount
    )

    max_amt = (
        max(amounts)
        if amounts
        else amount
    )

    freq = len(amounts)

    deviation = amount - avg

    ratio = amount / (avg + 1)

    features = np.array([[
        amount,
        avg,
        max_amt,
        deviation,
        ratio,
        freq,
        time_gap
    ]])

    return features, avg, max_amt, freq, deviation, ratio


# ============================================================
# RISK ENGINE
# ============================================================

def compute_risk(amount, user):

    # --------------------------------------------------------
    # Basic transaction checks
    # --------------------------------------------------------

    if amount < 100:

        return {
            "risk": "LOW",
            "risk_score": 0.20,
            "reason": "Very low transaction amount"
        }


    if amount < 500:

        return {
            "risk": "LOW",
            "risk_score": 0.30,
            "reason": "Low transaction amount"
        }


    # --------------------------------------------------------
    # Retrieve transaction history
    # --------------------------------------------------------

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT amount, time

        FROM transactions

        WHERE sender=?

        ORDER BY id DESC

        LIMIT 10
    """, (user,))

    rows = cursor.fetchall()

    conn.close()


    amounts = [
        row[0]
        for row in rows
    ]


    # --------------------------------------------------------
    # Time gap
    # --------------------------------------------------------

    if rows:

        try:

            last_time = datetime.strptime(
                rows[0][1],
                "%Y-%m-%d %H:%M:%S"
            )

            time_gap = (
                datetime.now() - last_time
            ).total_seconds()

        except (
            ValueError,
            TypeError
        ):

            time_gap = 9999

    else:

        time_gap = 9999


    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    (
        features,
        avg,
        max_amt,
        freq,
        deviation,
        ratio
    ) = build_features(
        amount,
        amounts,
        time_gap
    )


    # --------------------------------------------------------
    # Pad to 30 features
    # --------------------------------------------------------

    full_features = np.zeros(
        (1, 30)
    )

    full_features[0][:7] = features[0]


    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    try:

        if hasattr(
            scaler,
            "feature_names_in_"
        ):

            import pandas as pd

            full_features = pd.DataFrame(
                full_features,
                columns=scaler.feature_names_in_
            )

    except Exception:

        pass


    scaled_features = scaler.transform(
        full_features
    )


    # --------------------------------------------------------
    # Rule-based fraud detection
    # --------------------------------------------------------

    if amount > 3 * avg and amount > 1000:

        return {
            "risk": "HIGH",
            "risk_score": 0.90,
            "reason": "Transaction is significantly higher than user history"
        }


    if amount >= 15000 and ratio > 2:

        return {
            "risk": "HIGH",
            "risk_score": 0.85,
            "reason": "Very high amount and abnormal transaction ratio"
        }


    # --------------------------------------------------------
    # ML fraud probability
    # --------------------------------------------------------

    try:

        ml_probability = model.predict_proba(
            scaled_features
        )[0][1]

    except Exception:

        ml_probability = 0.5


    # --------------------------------------------------------
    # Random Forest probability
    # --------------------------------------------------------

    try:

        rf_probability = rf_model.predict_proba(
            scaled_features
        )[0][1]

    except Exception:

        rf_probability = ml_probability


    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    anomaly = iso_model.decision_function(
        scaled_features
    )[0]

    anomaly_score = (
        1 - (anomaly + 1) / 2
    )

    anomaly_score = max(
        0,
        min(anomaly_score, 1)
    )


    # --------------------------------------------------------
    # Combine ML models
    # --------------------------------------------------------

    ml_score = (
        0.60 * ml_probability
        + 0.40 * rf_probability
    )


    # --------------------------------------------------------
    # Behavioral score
    # --------------------------------------------------------

    behavioral_score = min(
        1,
        amount / (avg + 1)
    )


    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    score = (
        0.40 * ml_score
        + 0.30 * anomaly_score
        + 0.30 * behavioral_score
    )


    # --------------------------------------------------------
    # Behavioral risk boosts
    # --------------------------------------------------------

    if amount > 2 * avg:

        score += 0.15


    if amount > 5 * avg:

        score += 0.25


    if freq >= 3 and time_gap < 60:

        score += 0.25


    score = float(
        max(
            0,
            min(score, 1)
        )
    )


    # --------------------------------------------------------
    # Amount-based safety rules
    # --------------------------------------------------------

    if amount < 2000:

        score = min(
            score,
            0.65
        )


    if amount >= 8000 and score < 0.60:

        return {
            "risk": "MEDIUM",
            "risk_score": max(
                score,
                0.40
            ),
            "reason": "High-value transaction requires additional verification"
        }


    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    if score < 0.40:

        risk = "LOW"

        reason = "Transaction behavior appears normal"

    elif score < 0.70:

        risk = "MEDIUM"

        reason = "Transaction shows moderate risk indicators"

    else:

        risk = "HIGH"

        reason = "Transaction shows strong fraud indicators"


    return {
        "risk": risk,
        "risk_score": score,
        "reason": reason
    }


# ============================================================
# SAVE TRANSACTION
# ============================================================

def save_transaction(
    amount,
    user,
    risk_score=None,
    risk_level=None,
    decision=None
):

    conn = get_connection()
    cursor = conn.cursor()

    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO transactions
        (
            sender,
            amount,
            time,
            risk_score,
            risk_level,
            decision
        )

        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        user,
        amount,
        current_time,
        risk_score,
        risk_level,
        decision
    ))

    conn.commit()
    conn.close()


# ============================================================
# MODEL UPDATE
# ============================================================

def update_model(
    amount,
    user,
    label
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT amount
        FROM transactions
        WHERE sender=?
        ORDER BY id DESC
        LIMIT 10
    """, (user,))

    rows = cursor.fetchall()

    conn.close()

    amounts = [
        row[0]
        for row in rows
    ]

    avg = (
        sum(amounts) / len(amounts)
        if amounts
        else amount
    )

    deviation = amount - avg

    ratio = amount / (avg + 1)

    features = np.array([[
        amount,
        avg,
        max(amounts)
        if amounts
        else amount,
        deviation,
        ratio,
        len(amounts),
        0
    ]])

    full_features = np.zeros(
        (1, 30)
    )

    full_features[0][:7] = features[0]

    scaled_features = scaler.transform(
        full_features
    )

    model.partial_fit(
        scaled_features,
        [label],
        classes=[0, 1]
    )

    with open(
        model_path,
        "wb"
    ) as f:

        pickle.dump(
            model,
            f
        )