import numpy as np
import pickle

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "fraud_model.pkl"), "rb"))
rf_model = pickle.load(open(os.path.join(BASE_DIR, "rf_model.pkl"), "rb"))
iso_model = pickle.load(open(os.path.join(BASE_DIR, "iso_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

initialized = False

def compute_risk(amount):
    features = np.zeros((1, 30))
    features[0][-1] = amount
    features = scaler.transform(features)

    prob = (
        0.6 * model.predict_proba(features)[0][1] +
        0.4 * rf_model.predict_proba(features)[0][1]
    )

    anomaly = abs(iso_model.decision_function(features)[0])

    if amount > 5000:
        behavior = 0.7
    elif amount > 2000:
        behavior = 0.5
    else:
        behavior = 0.2

    amount_factor = min(amount / 10000, 1)

    score = (
        0.2 * prob +
        0.2 * anomaly +
        0.3 * behavior +
        0.3 * amount_factor
    )

    score = float(min(score, 1))

    if score < 0.3:
        risk = "LOW"
    elif score < 0.6:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {"risk": risk, "risk_score": score}


def update_model(amount, label):
    global initialized, model

    features = np.zeros((1, 30))
    features[0][-1] = amount
    features = scaler.transform(features)

    y = np.array([label])

    if not initialized:
        model.partial_fit(features, y, classes=np.array([0,1]))
        initialized = True
    else:
        model.partial_fit(features, y)

    pickle.dump(model, open(os.path.join(BASE_DIR, "fraud_model.pkl"), "wb"))