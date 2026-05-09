import streamlit as st
import pandas as pd
from utils.stock_data import endeks_verisi_cek
from utils.constants import RENKLER

# Sayfa ayarları
st.set_page_config(
    page_title="BIST Analiz - Yeni Proje",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state başlat
if "stock_data" not in st.session_state:
    st.session_state["stock_data"] = {}
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = {}
if "analysis_data" not in st.session_state:
    st.session_state["analysis_data"] = {}
if "selected_symbols" not in st.session_state:
    st.session_state["selected_symbols"] = []

# --- Özel CSS ---
st.markdown("""
<style>
    /* Genel */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Ana başlık */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #00d4aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #8892a4;
        font-size: 1.15rem;
        font-weight: 300;
        margin-bottom: 2.5rem;
    }
    
    /* KPI kartları */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.95), rgba(30, 36, 54, 0.9));
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        flex: 1;
        min-width: 200px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    .kpi-label {
        font-size: 0.8rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
        margin-bottom: 0.4rem;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e1e5ee;
        margin-bottom: 0.2rem;
    }
    
    .kpi-change-up {
        color: #00d4aa;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .kpi-change-down {
        color: #ff4757;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #131825 100%);
    }
    
    .sidebar-header {
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 1rem;
    }
    
    .sidebar-logo {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #00d4aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">📈 BIST Yeni</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Durum göstergesi
    veri_var = len(st.session_state.get("stock_data", {})) > 0
    if veri_var:
        semboller = list(st.session_state["stock_data"].keys())
        st.markdown(f'<span class="status-badge badge-active">● Veri Yüklü</span>', unsafe_allow_html=True)
        st.caption(f"📊 {len(semboller)} hisse")
    else:
        st.markdown(f'<span class="status-badge badge-inactive">○ Veri Yok</span>', unsafe_allow_html=True)
        st.caption("Başlamak için Veri Çekme sayfasını kullanın")

# --- Ana Sayfa İçeriği ---
st.markdown('<div class="hero-title">BIST Analiz Merkezi</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Yeni Proje: Veri Çekme ve İşleme Modülleri</div>', unsafe_allow_html=True)

# BIST 100 endeks bilgisi
with st.spinner("BIST 100 endeks verisi yükleniyor..."):
    endeks = endeks_verisi_cek("^XU100", "5d")

if endeks is not None and len(endeks) >= 2:
    son_kapanis = endeks["Close"].iloc[-1]
    onceki_kapanis = endeks["Close"].iloc[-2]
    degisim = ((son_kapanis - onceki_kapanis) / onceki_kapanis) * 100
    
    degisim_sinif = "kpi-change-up" if degisim >= 0 else "kpi-change-down"
    degisim_ok = "▲" if degisim >= 0 else "▼"
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">BIST 100</div>
            <div class="kpi-value">{son_kapanis:,.0f}</div>
            <div class="{degisim_sinif}">{degisim_ok} %{abs(degisim):.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.info("👈 Analize başlamak için sol menüden bir sayfa seçin.")
