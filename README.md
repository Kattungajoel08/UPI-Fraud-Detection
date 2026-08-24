# Continuous Real-Time UPI Fraud Detection Using Adaptive Machine Learning

An end-to-end UPI fraud detection system combining machine learning,
FastAPI, Flutter, SQLite, and Streamlit to detect suspicious transactions
and apply additional verification based on transaction risk.

## 🚀 Features

- Real-time transaction risk assessment
- QR-based UPI payment interface
- PIN authentication
- OTP verification
- High-risk transaction verification
- Transaction history
- Balance enquiry
- Fraud analytics dashboard
- Adaptive model updates
- REST API integration

## 🏗️ System Architecture

Flutter Mobile App
        ↓
     FastAPI
        ↓
   Risk Engine
        ↓
 Machine Learning Models
        ↓
     SQLite DB
        ↓
 Streamlit Dashboard

## 🧠 Machine Learning

Models used:

- SGD Classifier
- Random Forest
- Isolation Forest
- StandardScaler

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score

## 🛠️ Tech Stack

- Python
- Scikit-learn
- FastAPI
- Flutter / Dart
- SQLite
- Streamlit
- Pandas
- Plotly

## 📱 Flutter Application

The mobile application supports:

- QR code scanning
- Payment amount entry
- PIN verification
- Risk assessment
- OTP verification
- High-risk verification
- Payment receipt
- Transaction history
- Balance enquiry
- Receive-money QR

## 📊 Fraud Detection Dashboard

The Streamlit dashboard provides:

- Total transactions
- Fraud transactions
- Safe transactions
- Fraud rate
- Risk distribution
- Transaction amount distribution
- Approved vs blocked transactions
- Model performance
- ML vs Adaptive comparison

## 🔄 Transaction Flow

1. User scans a QR code or enters a receiver.
2. User enters the transaction amount.
3. Transaction is sent to the FastAPI backend.
4. ML risk engine calculates the risk score.
5. LOW-risk transactions can proceed.
6. MEDIUM-risk transactions require OTP verification.
7. HIGH-risk transactions require additional verification.
8. Transaction details are stored in SQLite.
9. Results are displayed in the Streamlit dashboard.

## 📁 Project Structure

```text
UPI-Fraud-Detection/
│
├── Flutter App/
├── services/
├── fraud_detection_system.py
├── train_model.py
├── dashboard.py
├── api.py
├── fraud_model.pkl
├── rf_model.pkl
├── iso_model.pkl
├── scaler.pkl
├── metrics.pkl
├── requirements.txt
└── README.md
