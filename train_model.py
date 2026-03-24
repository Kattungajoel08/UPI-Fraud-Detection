import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv(r"C:\Users\kattu\OneDrive\Desktop\UPI_Fraud_Detection_Project\creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- MODEL 1: SGD ----------------
sgd_model = SGDClassifier(loss="log_loss")
sgd_model.fit(X_scaled, y)

# ---------------- MODEL 2: RANDOM FOREST ----------------
rf_model = RandomForestClassifier(n_estimators=50)
rf_model.fit(X_scaled, y)

# ---------------- MODEL 3: ISOLATION FOREST ----------------
iso_model = IsolationForest(contamination=0.01)
iso_model.fit(X_scaled)

# ---------------- EVALUATION ----------------
y_pred_sgd = sgd_model.predict(X_scaled)
y_pred_rf = rf_model.predict(X_scaled)

accuracy_sgd = accuracy_score(y, y_pred_sgd)
accuracy_rf = accuracy_score(y, y_pred_rf)

precision_sgd = precision_score(y, y_pred_sgd)
recall_sgd = recall_score(y, y_pred_sgd)
f1_sgd = f1_score(y, y_pred_sgd)

precision_rf = precision_score(y, y_pred_rf)
recall_rf = recall_score(y, y_pred_rf)
f1_rf = f1_score(y, y_pred_rf)

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