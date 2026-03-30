import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

# ---------------- LOAD DATA ----------------
BASE_DIR = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE_DIR, "creditcard.csv"))

X = df.drop("Class", axis=1)
y = df["Class"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- MODELS ----------------
sgd_model = SGDClassifier(loss="log_loss", class_weight="balanced")
rf_model = RandomForestClassifier(n_estimators=50, class_weight="balanced")
iso_model = IsolationForest(contamination=0.01)

# ---------------- TRAINING ----------------
sgd_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)
iso_model.fit(X_train_scaled)

# ---------------- EVALUATION ----------------
y_pred_sgd = sgd_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

precision_sgd = precision_score(y_test, y_pred_sgd, zero_division=0)
recall_sgd = recall_score(y_test, y_pred_sgd, zero_division=0)
f1_sgd = f1_score(y_test, y_pred_sgd, zero_division=0)

precision_rf = precision_score(y_test, y_pred_rf, zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)

print("\n--- SGD MODEL ---")
print("Accuracy:", accuracy_sgd)

print("\n--- RANDOM FOREST ---")
print("Accuracy:", accuracy_rf)

# ---------------- SAVE MODELS ----------------
pickle.dump(sgd_model, open("fraud_model.pkl", "wb"))
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(iso_model, open("iso_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# ---------------- SAVE METRICS ----------------
metrics = {
    "SGD": {
        "accuracy": accuracy_sgd,
        "precision": precision_sgd,
        "recall": recall_sgd,
        "f1": f1_sgd
    },
    "RF": {
        "accuracy": accuracy_rf,
        "precision": precision_rf,
        "recall": recall_rf,
        "f1": f1_rf
    }
}

pickle.dump(metrics, open("metrics.pkl", "wb"))

print("\n✅ All models and metrics saved")