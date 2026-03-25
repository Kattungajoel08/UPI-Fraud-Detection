import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ---------------- LOGIN ----------------
USERNAME = "admin"
PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 UPI Fraud Monitoring Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------- DASHBOARD ----------------
def dashboard():

    st_autorefresh(interval=3000, key="refresh")

    st.title("💳 UPI Fraud Detection Dashboard")

    # ---------------- DATABASE ----------------
    conn = sqlite3.connect("fraud.db")
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    conn.close()

    if df.empty:
        st.warning("No transactions yet")
        return

    # ---------------- METRICS ----------------
    total = len(df)
    frauds = df["fraud"].sum()
    safe = total - frauds

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Fraud Transactions", frauds)
    col3.metric("Safe Transactions", safe)

    # ---------------- ALERT ----------------
    if frauds > 0:
        st.error("🚨 Fraud Transactions Detected!")
    else:
        st.success("System Secure")

    # ---------------- RISK DISTRIBUTION ----------------
    st.subheader("📊 Risk Distribution")

    fig1 = px.pie(df, names="risk", title="Risk Levels")
    st.plotly_chart(fig1)

    # ---------------- FRAUD VS SAFE ----------------
    st.subheader("⚠️ Fraud vs Safe")

    df["status"] = df["fraud"].apply(lambda x: "Fraud" if x == 1 else "Safe")

    fig2 = px.bar(df, x="risk", color="status", title="Fraud Detection by Risk")
    st.plotly_chart(fig2)

    # ---------------- AMOUNT DISTRIBUTION ----------------
    st.subheader("💰 Transaction Amount Distribution")

    fig3 = px.histogram(df, x="amount", nbins=20)
    st.plotly_chart(fig3)

    # ---------------- TIME ANALYSIS ----------------
    st.subheader("⏰ Transactions Over Time")

    df["time"] = pd.to_datetime(df["time"])
    fig4 = px.line(df, x="time", y="amount", title="Transactions Timeline")
    st.plotly_chart(fig4)

    # ---------------- TABLE ----------------
    st.subheader("📋 Transaction Records")

    def highlight(row):
        if row["risk"] == "HIGH":
            return ["background-color: red"] * len(row)
        elif row["risk"] == "MEDIUM":
            return ["background-color: yellow"] * len(row)
        else:
            return ["background-color: lightgreen"] * len(row)

    st.dataframe(df.style.apply(highlight, axis=1))

# ---------------- FLOW ----------------
if st.session_state.logged_in:
    dashboard()
else:
    login()