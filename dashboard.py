import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fraud Dashboard", layout="wide")

st.title("💳 UPI Fraud Detection Dashboard")

# ---------------- LOAD DATA ----------------
conn = sqlite3.connect("fraud.db")
df = pd.read_sql_query("SELECT * FROM transactions", conn)
conn.close()

if df.empty:
    st.warning("No transactions yet")
else:
    df["Status"] = df["fraud"].apply(lambda x: "Fraud 🚨" if x == 1 else "Safe ✅")

    # ---------------- TABLE ----------------
    st.subheader("Transactions")
    st.dataframe(df)

    # ---------------- CHART ----------------
    st.subheader("Fraud vs Safe")
    fig = px.pie(df, names="Status", title="Fraud Distribution")
    st.plotly_chart(fig)

    # ---------------- RISK ----------------
    st.subheader("Risk Levels")
    fig2 = px.histogram(df, x="risk", color="Status")
    st.plotly_chart(fig2)