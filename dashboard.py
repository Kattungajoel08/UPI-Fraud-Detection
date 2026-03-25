import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from streamlit_autorefresh import st_autorefresh

USERNAME = "admin"
PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

def dashboard():

    st.set_page_config(layout="wide")
    st_autorefresh(interval=3000, key="refresh")

    st.title("💳 UPI Fraud Detection Dashboard")

    API_URL = "http://127.0.0.1:8000/transactions"

    try:
        df = pd.DataFrame(requests.get(API_URL).json())
    except:
        st.error("API not running")
        return

    if df.empty:
        st.warning("No transactions yet")
        return

    # ---------------- METRICS ----------------
    total = len(df)
    frauds = df["fraud"].sum()
    drift_count = df["drift"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Frauds", frauds)
    col3.metric("Drift Alerts", drift_count)

    # ---------------- ALERTS ----------------
    if frauds > 0:
        st.error("🚨 Fraud Detected")

    if drift_count > 0:
        st.warning("⚠ Concept Drift Detected")

    # ---------------- CHARTS ----------------
    col4, col5 = st.columns(2)

    with col4:
        fig1 = px.pie(df, names="risk", title="Risk Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col5:
        df["status"] = df["fraud"].apply(lambda x: "Fraud" if x else "Safe")
        fig2 = px.bar(df, x="risk", color="status", title="Fraud vs Safe")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- DRIFT GRAPH ----------------
    st.subheader("📉 Risk Score Trend")

    df["time"] = pd.to_datetime(df["time"])
    fig3 = px.line(df, x="time", y="risk_score", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

    # ---------------- TABLE ----------------
    def highlight(row):
        if row["drift"] == 1:
            return ["background-color: orange"] * len(row)
        elif row["risk"] == "HIGH":
            return ["background-color: red"] * len(row)
        return [""] * len(row)

    st.dataframe(df.style.apply(highlight, axis=1), use_container_width=True)

if st.session_state.logged_in:
    dashboard()
else:
    login()