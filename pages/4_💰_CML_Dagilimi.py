import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="CML Portfolio", layout="wide")

st.title("💰 Capital Market Line (CML) Portfolio Distribution")

data = st.session_state.get("analysis_data", None)

if not data:
    st.warning("Önce Markowitz sayfasını çalıştırmalısın.")
    st.stop()

mu = data["expected_returns"].values
cov = data["annual_covariance"].values
rf = data["risk_free_rate"]

symbols = data["symbols"]
max_sharpe_weights = data["max_sharpe_weights"].values

# =========================
# Tangency portfolio stats
# =========================

def portfolio_return(w, mu):
    return w @ mu

def portfolio_vol(w, cov):
    return np.sqrt(w.T @ cov @ w)

t_return = portfolio_return(max_sharpe_weights, mu)
t_vol = portfolio_vol(max_sharpe_weights, cov)

# =========================
# CML distribution
# =========================

y_values = np.unique(np.append(np.linspace(0, 2, 60), 1))
y_values.sort()

cml_r, cml_v, rf_w, risky_w = [], [], [], []

for y in y_values:
    rf_w.append(max(0, 1 - y))
    risky_w.append(y)

    # CML FORMÜLÜ
    cml_r.append((rf + y * (t_return - rf)) * 100)
    cml_v.append(y * t_vol * 100)

# =========================
# GRAPH
# =========================

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=cml_v,
    y=cml_r,
    mode="lines+markers",
    name="CML Distribution"
))



fig.add_trace(go.Scatter(
    x=[0],
    y=[rf * 100],
    mode="markers",
    name="Risk-Free",
    marker=dict(size=12)
))

fig.update_layout(
    xaxis_title="Volatility (%)",
    yaxis_title="Return (%)",
    height=550
)
# 🔥 GUARANTEED TANGENCY POINT (y=1)
fig.add_trace(go.Scatter(
    x=[t_vol * 100],
    y=[t_return * 100],
    mode="markers+text",
    name="Tangency (y=1)",
    marker=dict(size=15, color="red"),
    text=["y=1"],
    textposition="top center"
))

st.plotly_chart(fig, use_container_width=True)

# =========================
# TABLE
# =========================

st.markdown("### 📊 CML Distribution Table")

df = pd.DataFrame({
    "y (Risky Weight)": y_values,
    "Risk-Free %": np.array(rf_w) * 100,
    "Risky %": np.array(risky_w) * 100,
    "Return %": cml_r,
    "Volatility %": cml_v
})
df["Is Tangency (y=1)"] = np.isclose(df["y (Risky Weight)"], 1.0)

st.dataframe(df, use_container_width=True)
