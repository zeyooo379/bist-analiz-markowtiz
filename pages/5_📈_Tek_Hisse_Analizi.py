import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

try:
    from utils.constants import get_bist_hisseler
    BIST_HISSELER = get_bist_hisseler(use_remote=True)
except Exception:
    from utils.constants import BIST_HISSELER


# =============================================================================
# SAYFA AYARLARI
# =============================================================================

st.set_page_config(
    page_title="Tek Riskli Varlık + Risk-Free Markowitz",
    page_icon="📈",
    layout="wide",
)

st.markdown("## 📈 Tek Riskli Varlık + Risk-Free Markowitz")
st.markdown(
    """
Bu sayfa:
- Tek bir hisse senedi
- Risk-free asset (hazine bonosu)

ile Markowitz optimizasyonu yapar.

Sistem:
- Kovaryans matrisini oluşturur
- Optimum ağırlıkları hesaplar
- Maksimum Sharpe portföyünü bulur
- Capital Allocation Line (CAL) çizer
"""
)

st.markdown("---")


# =============================================================================
# SESSION STATE KONTROLÜ
# =============================================================================

analysis_data = st.session_state.get("analysis_data", {})

if not analysis_data:
    st.warning(
        "⚠️ Önce Veri İşleme sayfasında analiz çalıştırılmalı."
    )
    st.stop()

if minimize is None:
    st.error(
        "❌ scipy gerekli.\n\nKurulum:\n\npip install scipy"
    )
    st.stop()


# =============================================================================
# VERİLERİ ÇEK
# =============================================================================

expected_returns = analysis_data["expected_returns"]
annual_covariance = analysis_data["annual_covariance"]
symbols = analysis_data["symbols"]
risk_free_rate = analysis_data["risk_free_rate"]


# =============================================================================
# HİSSE SEÇİMİ
# =============================================================================

st.markdown("### ⚙️ Model Ayarları")

selected_stock = st.selectbox(
    "Hisse seç",
    options=symbols,
    format_func=lambda x: f"{x} — {BIST_HISSELER.get(x, x)}",
)


# =============================================================================
# HİSSE VERİLERİ
# =============================================================================

stock_return_decimal = expected_returns[selected_stock]

stock_variance = annual_covariance.loc[
    selected_stock,
    selected_stock,
]

stock_volatility_decimal = np.sqrt(stock_variance)

rf_decimal = risk_free_rate


# =============================================================================
# RETURN VE COVARIANCE MATRIX
# =============================================================================

asset_returns = np.array([
    stock_return_decimal,
    rf_decimal,
])

cov_matrix = np.array([
    [stock_variance, 0],
    [0, 0],
])

asset_names = [
    selected_stock,
    "Risk-Free Asset",
]


# =============================================================================
# PORTFÖY FONKSİYONLARI
# =============================================================================

def portfolio_return(weights, returns):
    return float(weights @ returns)


def portfolio_variance(weights, covariance):
    return float(weights.T @ covariance @ weights)


def portfolio_volatility(weights, covariance):
    variance = portfolio_variance(weights, covariance)
    return float(np.sqrt(max(variance, 0)))


def portfolio_sharpe(weights, returns, covariance, rf):
    p_return = portfolio_return(weights, returns)
    p_volatility = portfolio_volatility(weights, covariance)

    if p_volatility == 0:
        return -999

    return (p_return - rf) / p_volatility


def negative_sharpe(weights, returns, covariance, rf):
    return -portfolio_sharpe(
        weights,
        returns,
        covariance,
        rf,
    )


# =============================================================================
# OPTİMİZASYON
# =============================================================================

initial_weights = np.array([0.5, 0.5])

bounds = (
    (0, 1),
    (0, 1),
)

constraints = {
    "type": "eq",
    "fun": lambda w: np.sum(w) - 1,
}

result = minimize(
    fun=negative_sharpe,
    x0=initial_weights,
    args=(
        asset_returns,
        cov_matrix,
        rf_decimal,
    ),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

if not result.success:
    st.error("❌ Optimizasyon başarısız.")
    st.stop()

optimal_weights = result.x

stock_weight = optimal_weights[0]
rf_weight = optimal_weights[1]


# =============================================================================
# PORTFÖY SONUÇLARI
# =============================================================================

portfolio_return_decimal = portfolio_return(
    optimal_weights,
    asset_returns,
)

portfolio_volatility_decimal = portfolio_volatility(
    optimal_weights,
    cov_matrix,
)

portfolio_sharpe_ratio = portfolio_sharpe(
    optimal_weights,
    asset_returns,
    cov_matrix,
    rf_decimal,
)


# =============================================================================
# YÜZDE FORMATLARI
# =============================================================================

stock_return_pct = stock_return_decimal * 100
stock_volatility_pct = stock_volatility_decimal * 100

rf_pct = rf_decimal * 100

portfolio_return_pct = portfolio_return_decimal * 100
portfolio_volatility_pct = portfolio_volatility_decimal * 100


# =============================================================================
# OPTİMAL AĞIRLIKLAR
# =============================================================================

st.markdown("### 🧠 Optimum Portföy")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    f"{selected_stock} Ağırlığı",
    f"%{stock_weight * 100:.2f}",
)

c2.metric(
    "Risk-Free Ağırlığı",
    f"%{rf_weight * 100:.2f}",
)

c3.metric(
    "Beklenen Getiri",
    f"%{portfolio_return_pct:.2f}",
)

c4.metric(
    "Volatilite",
    f"%{portfolio_volatility_pct:.2f}",
)


# =============================================================================
# HİSSE DETAYLARI
# =============================================================================

st.markdown("---")
st.markdown("### 📊 Hisse Detayları")

d1, d2, d3 = st.columns(3)

d1.metric(
    "Hisse Beklenen Getirisi",
    f"%{stock_return_pct:.2f}",
)

d2.metric(
    "Hisse Volatilitesi",
    f"%{stock_volatility_pct:.2f}",
)

d3.metric(
    "Sharpe Ratio",
    f"{portfolio_sharpe_ratio:.3f}",
)


# =============================================================================
# KOVARYANS MATRİSİ
# =============================================================================

st.markdown("---")
st.markdown("### 🔢 Kovaryans Matrisi")

cov_df = pd.DataFrame(
    cov_matrix,
    index=asset_names,
    columns=asset_names,
)

st.dataframe(
    cov_df.round(6),
    use_container_width=True,
)


# =============================================================================
# RETURN VE RİSK TABLOSU
# =============================================================================

st.markdown("---")
st.markdown("### 📋 Asset Özeti")

summary_df = pd.DataFrame({
    "Asset": asset_names,
    "Beklenen Getiri (%)": [
        stock_return_pct,
        rf_pct,
    ],
    "Volatilite (%)": [
        stock_volatility_pct,
        0,
    ],
})

st.dataframe(
    summary_df.round(4),
    use_container_width=True,
    hide_index=True,
)


# =============================================================================
# CAPITAL ALLOCATION LINE
# =============================================================================

st.markdown("---")
st.markdown("### 📈 Capital Allocation Line (CAL)")

if go is not None:

    x_line = np.linspace(
        0,
        stock_volatility_pct * 1.5,
        150,
    )

    slope = (
        (stock_return_pct - rf_pct)
        / stock_volatility_pct
        if stock_volatility_pct != 0
        else 0
    )

    y_line = rf_pct + slope * x_line

    fig = go.Figure()

    # -------------------------------------------------------------------------
    # CAL ÇİZGİSİ
    # -------------------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Capital Allocation Line",
            hovertemplate=(
                "Volatilite: %{x:.2f}%<br>"
                "Beklenen Getiri: %{y:.2f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # RISK-FREE ASSET
    # -------------------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[rf_pct],
            mode="markers",
            name="Risk-Free Asset",
            marker=dict(
                size=14,
                symbol="diamond",
            ),
        )
    )

    # -------------------------------------------------------------------------
    # HİSSE
    # -------------------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[stock_volatility_pct],
            y=[stock_return_pct],
            mode="markers",
            name=selected_stock,
            marker=dict(
                size=16,
            ),
        )
    )

    # -------------------------------------------------------------------------
    # OPTİMAL PORTFÖY
    # -------------------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=[portfolio_volatility_pct],
            y=[portfolio_return_pct],
            mode="markers",
            name="Optimal Portföy",
            marker=dict(
                size=18,
                symbol="star",
            ),
        )
    )

    # -------------------------------------------------------------------------
    # LAYOUT
    # -------------------------------------------------------------------------

    fig.update_layout(
        xaxis_title="Volatilite / Risk (%)",
        yaxis_title="Beklenen Getiri (%)",
        hovermode="closest",
        height=600,
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:

    st.error(
        "❌ Plotly gerekli.\n\nKurulum:\n\npip install plotly"
    )


# =============================================================================
# MATEMATİKSEL MODEL
# =============================================================================

st.markdown("---")
st.markdown("### 🧮 Matematiksel Gösterim")

st.latex(r"E(R_p) = w_s E(R_s) + w_f r_f")

st.latex(r"\sigma_p = w_s \sigma_s")

st.latex(r"w_s + w_f = 1")

st.markdown(
    """
Burada:

- \( w_s \) → hisse ağırlığı
- \( w_f \) → risk-free asset ağırlığı
- \( r_f \) → risk-free getiri
- \( E(R_s) \) → hisse beklenen getirisi
- \( \sigma_s \) → hisse volatilitesi
"""
)