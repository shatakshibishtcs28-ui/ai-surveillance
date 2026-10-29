"""Run: streamlit run dashboard.py"""
import os

import pandas as pd
import streamlit as st

CSV_PATH = "alerts.csv"
st.set_page_config(page_title="AI Surveillance Dashboard", layout="wide")
st.title("AI Surveillance & Threat Detection Dashboard")

if not os.path.exists(CSV_PATH):
    st.info("Abhi koi alert nahi hai. Pehle surveillance.py chalao.")
    st.stop()

df = pd.read_csv(CSV_PATH)
if df.empty:
    st.info("Abhi koi alert nahi hai.")
    st.stop()

if st.button("Refresh"):
    st.rerun()

# Summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total alerts", len(df))
for col, lvl in zip((c2, c3, c4), ("Medium", "High", "Critical")):
    col.metric(lvl, int((df["level"] == lvl).sum()))

# Filters
levels = st.multiselect("Risk level filter",
                        ["Low", "Medium", "High", "Critical"],
                        default=["Medium", "High", "Critical"])
view = df[df["level"].isin(levels)].sort_values("risk_score", ascending=False)

COLORS = {"Low": "#c8f7c5", "Medium": "#fff3b0",
          "High": "#ffd1a6", "Critical": "#ff9999"}


def color_row(row):
    return [f"background-color: {COLORS[row['level']]}; color: black"] * len(row)


st.subheader("Alerts (risk ke hisaab se sorted)")
st.dataframe(view.drop(columns=["image"]).style.apply(color_row, axis=1),
             use_container_width=True)

# Operator action (human-in-the-loop)
st.subheader("Operator review")
if not view.empty:
    idx = st.selectbox("Alert chuno (row number)", view.index.tolist())
    row = df.loc[idx]
    left, right = st.columns([2, 1])
    if os.path.exists(row["image"]):
        left.image(row["image"], caption=f"{row['event']} | {row['level']} | {row['risk_score']}")
    with right:
        st.write(f"**Event:** {row['event']}")
        st.write(f"**Time:** {row['time']}")
        st.write(f"**Status:** {row['status']}")
        if st.button("Acknowledge"):
            df.loc[idx, "status"] = "acknowledged"
            df.to_csv(CSV_PATH, index=False)
            st.rerun()
        if st.button("False alarm"):
            df.loc[idx, "status"] = "false_alarm"
            df.to_csv(CSV_PATH, index=False)
            st.rerun()
