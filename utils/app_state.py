import streamlit as st


def init_portfolio_state():
    """
    Sayfalar arası veri kaybını önlemek için tüm state key'lerini
    sadece yoksa initialize eder. Var olan veriyi asla ezmez.
    """
    defaults = {
        "raw_stock_data": {},
        "stock_data": {},
        "selected_symbols": [],
        "analysis_symbols": [],
        "data_quality_report": None,
        "processed_data": {},
        "analysis_data": {},
        "_last_applied_analysis_symbols": [],
        "selected_symbols_ui": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_outputs():
    """
    Yeni veri çekildiğinde veya analize dahil edilen hisseler değiştiğinde
    eski analiz çıktıları temizlenir.
    """
    st.session_state["processed_data"] = {}
    st.session_state["analysis_data"] = {}


def clear_all_portfolio_state():
    """
    Kullanıcı tamamen temizlemek isterse kullanılır.
    """
    keys = [
        "raw_stock_data",
        "stock_data",
        "selected_symbols",
        "analysis_symbols",
        "data_quality_report",
        "processed_data",
        "analysis_data",
        "_last_applied_analysis_symbols",
        "selected_symbols_ui",
    ]

    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

    init_portfolio_state()