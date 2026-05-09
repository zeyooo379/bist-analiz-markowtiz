import streamlit as st
import pandas as pd
import numpy as np

try:
    from utils.constants import get_bist_hisseler
    BIST_HISSELER = get_bist_hisseler(use_remote=True)
except Exception:
    from utils.constants import BIST_HISSELER

try:
    from utils.app_state import init_portfolio_state
except Exception:
    def init_portfolio_state():
        pass

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


st.set_page_config(
    page_title="Veri İşleme — Markowitz Portföy Optimizasyonu",
    page_icon="🔧",
    layout="wide",
)

init_portfolio_state()

st.markdown("## 🔧 Veri İşleme")
st.markdown("Markowitz Modern Portföy Optimizasyonu için veri hazırlama, risk/getiri analizi ve portföy optimizasyonu.")
st.markdown("---")


# =============================================================================
# 1. VERİ KONTROLÜ
# =============================================================================

veriler = st.session_state.get("stock_data", {})

# Sayfa geçişi recovery:
# stock_data boşsa ama raw_stock_data + analysis_symbols varsa stock_data yeniden oluşturulur.
if not veriler:
    raw_veriler = st.session_state.get("raw_stock_data", {})
    analysis_symbols = st.session_state.get("analysis_symbols", [])

    if raw_veriler and analysis_symbols:
        veriler = {
            sembol: raw_veriler[sembol]
            for sembol in analysis_symbols
            if sembol in raw_veriler
        }

        st.session_state["stock_data"] = veriler
        st.session_state["selected_symbols"] = list(veriler.keys())

if not veriler:
    st.warning("⚠️ Henüz veri yüklenmemiş. Lütfen önce **📥 Veri Çekme** sayfasından veri çekin.")
    st.stop()

raw_symbols = list(veriler.keys())
st.success(f"✅ {len(raw_symbols)} hisse yüklü: {', '.join(raw_symbols)}")


# =============================================================================
# 2. MARKOWITZ İÇİN VERİ HAZIRLAMA
# =============================================================================

st.markdown("### 🧹 Markowitz Veri Hazırlığı")

valid_series = {}
data_quality_rows = []

for symbol, df in veriler.items():
    if df is None or df.empty:
        data_quality_rows.append({
            "Sembol": symbol,
            "Durum": "Atlandı",
            "Sebep": "Veri boş",
            "Ham Satır": 0,
            "Geçerli Close": 0,
        })
        continue

    if "Close" not in df.columns:
        data_quality_rows.append({
            "Sembol": symbol,
            "Durum": "Atlandı",
            "Sebep": "Close kolonu yok",
            "Ham Satır": len(df),
            "Geçerli Close": 0,
        })
        continue

    temp = df.copy()
    temp.index = pd.to_datetime(temp.index, errors="coerce")
    temp = temp[~temp.index.isna()]
    temp = temp.sort_index()
    temp = temp[~temp.index.duplicated(keep="last")]

    close = pd.to_numeric(temp["Close"], errors="coerce").dropna()

    if close.empty:
        data_quality_rows.append({
            "Sembol": symbol,
            "Durum": "Atlandı",
            "Sebep": "Geçerli Close verisi yok",
            "Ham Satır": len(df),
            "Geçerli Close": 0,
        })
        continue

    valid_series[symbol] = close

    data_quality_rows.append({
        "Sembol": symbol,
        "Durum": "Kullanıldı",
        "Sebep": "-",
        "Ham Satır": len(df),
        "Geçerli Close": len(close),
    })


if not valid_series:
    st.error("❌ Markowitz modeli için geçerli kapanış verisi bulunamadı.")
    st.stop()

prices_raw = pd.DataFrame(valid_series).sort_index()
prices_raw = prices_raw.dropna(how="all")

# Markowitz kovaryans matrisi için ortak tarihli veri kullanıyoruz.
prices = prices_raw.dropna(how="any")

symbols = list(prices.columns)
asset_count = len(symbols)

if asset_count < 2:
    st.error("❌ Markowitz portföy optimizasyonu için en az 2 geçerli hisse gerekiyor.")
    st.stop()

raw_row_count = len(prices_raw)
common_row_count = len(prices)
dropped_row_count = raw_row_count - common_row_count

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Yüklenen Hisse", len(raw_symbols))
col_b.metric("Kullanılan Hisse", asset_count)
col_c.metric("Ham Tarih Sayısı", raw_row_count)
col_d.metric("Ortak Tarih Sayısı", common_row_count)

if dropped_row_count > 0:
    st.warning(
        f"⚠️ Ortak tarih seti oluşturulurken {dropped_row_count} tarih çıkarıldı. "
        "Bu normal olabilir ama çok yüksekse bazı hisselerde veri eksikliği olabilir."
    )

with st.expander("📋 Veri Kalitesi Detayı"):
    data_quality_df = pd.DataFrame(data_quality_rows)
    st.dataframe(data_quality_df, use_container_width=True, hide_index=True)

if len(prices) < 30:
    st.error("❌ Optimizasyon için yeterli ortak fiyat verisi yok. En az 30 ortak işlem günü önerilir.")
    st.stop()


# =============================================================================
# 3. MODEL AYARLARI
# =============================================================================

st.markdown("---")
st.markdown("### ⚙️ Model Ayarları")

c1, c2, c3, c4 = st.columns(4)

with c1:
    trading_days = st.number_input(
        "Yıllık işlem günü",
        min_value=200,
        max_value=300,
        value=252,
        step=1,
        help="BIST için genelde 252 kullanılır.",
    )

with c2:
    return_type = st.radio(
        "Getiri tipi",
        ["Basit Getiri", "Log Getiri"],
        horizontal=False,
        help="Markowitz için başlangıçta Basit Getiri kullanmak daha anlaşılırdır.",
    )

with c3:
    risk_free_rate_pct = st.number_input(
        "Risksiz faiz yıllık (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5,
        help="Sharpe hesaplamasında kullanılır.",
    )

with c4:
    max_weight_pct = st.number_input(
        "Tek hisse max ağırlık (%)",
        min_value=1.0,
        max_value=100.0,
        value=100.0,
        step=1.0,
        help="100 olursa tek hisseye kadar yoğunlaşmaya izin verir.",
    )

risk_free_rate = risk_free_rate_pct / 100
max_weight = max_weight_pct / 100

if asset_count * max_weight < 1:
    st.error(
        f"❌ Tek hisse max ağırlığı %{max_weight_pct:.0f} çok düşük. "
        f"{asset_count} hisseyle toplam ağırlık 1'e ulaşamıyor."
    )
    st.stop()


# =============================================================================
# 4. GETİRİ, BEKLENEN GETİRİ VE KOVARYANS
# =============================================================================

if return_type == "Basit Getiri":
    returns = prices.pct_change(fill_method=None).dropna()
else:
    returns = np.log(prices / prices.shift(1)).dropna()

if len(returns) < 30:
    st.error("❌ Optimizasyon için yeterli getiri verisi yok. Daha uzun tarih aralığı seçmelisin.")
    st.stop()

expected_returns = returns.mean() * trading_days
daily_covariance = returns.cov()
annual_covariance = daily_covariance * trading_days
correlation = returns.corr()

st.markdown("---")
st.markdown("### 📈 Markowitz Girdileri")

col1, col2, col3 = st.columns(3)
col1.metric("Getiri Gözlem Sayısı", len(returns))
col2.metric("Beklenen Getiri Ölçeği", "Yıllık")
col3.metric("Kovaryans Ölçeği", "Yıllık")

expected_returns_df = pd.DataFrame({
    "Sembol": expected_returns.index,
    "Şirket": [BIST_HISSELER.get(s, s) for s in expected_returns.index],
    "Beklenen Yıllık Getiri (%)": (expected_returns.values * 100).round(2),
    "Yıllık Volatilite / Standart Sapma (%)": (np.sqrt(np.diag(annual_covariance.values)) * 100).round(2),
})

st.dataframe(expected_returns_df, use_container_width=True, hide_index=True)


# =============================================================================
# 5. KOVARYANS / KORELASYON MATRİSİ
# =============================================================================

st.markdown("---")
st.markdown("### 🔢 Risk Matrisi")

matrix_type = st.radio(
    "Matris tipi:",
    [
        "Yıllık Kovaryans Matrisi",
        "Günlük Kovaryans Matrisi",
        "Korelasyon Matrisi",
    ],
    horizontal=True,
)

if matrix_type == "Yıllık Kovaryans Matrisi":
    matrix = annual_covariance
    matrix_caption = "Markowitz optimizasyonunda kullanılan yıllık kovaryans matrisi."
elif matrix_type == "Günlük Kovaryans Matrisi":
    matrix = daily_covariance
    matrix_caption = "Günlük getiriler üzerinden hesaplanan kovaryans matrisi."
else:
    matrix = correlation
    matrix_caption = "Getiriler arası korelasyon matrisi."

st.caption(f"📌 {matrix_caption}")
st.dataframe(matrix.round(6), use_container_width=True)

csv_matrix = matrix.round(8).to_csv(encoding="utf-8-sig")

st.download_button(
    label=f"⬇️ {matrix_type} İndir",
    data=csv_matrix,
    file_name=f"{matrix_type.lower().replace(' ', '_')}.csv",
    mime="text/csv",
)


# =============================================================================
# 6. TEK HİSSE İSTATİSTİKLERİ
# =============================================================================

st.markdown("---")
st.markdown("### 📊 Tek Hisse Getiri İstatistikleri")

selected_symbol = st.selectbox(
    "Hisse seçin:",
    options=symbols,
    format_func=lambda x: f"{x} — {BIST_HISSELER.get(x, x)}",
)

single_returns = returns[selected_symbol].dropna()
single_prices = prices[[selected_symbol]].dropna().copy()
single_prices.columns = ["Kapanış"]

single_annual_return = single_returns.mean() * trading_days
single_annual_volatility = single_returns.std() * np.sqrt(trading_days)
single_sharpe = (
    (single_annual_return - risk_free_rate) / single_annual_volatility
    if single_annual_volatility != 0
    else np.nan
)

s1, s2, s3 = st.columns(3)
s1.metric("Beklenen Yıllık Getiri", f"%{single_annual_return * 100:.2f}")
s2.metric("Yıllık Volatilite / Std Sapma", f"%{single_annual_volatility * 100:.2f}")
s3.metric("Sharpe", f"{single_sharpe:.3f}" if not np.isnan(single_sharpe) else "-")

single_stats = pd.DataFrame({
    "Değer": {
        "Ortalama Günlük Getiri (%)": round(single_returns.mean() * 100, 4),
        "Medyan Günlük Getiri (%)": round(single_returns.median() * 100, 4),
        "Günlük Std Sapma (%)": round(single_returns.std() * 100, 4),
        "Günlük Varyans": round(single_returns.var(), 8),
        "Min Günlük Getiri (%)": round(single_returns.min() * 100, 4),
        "Max Günlük Getiri (%)": round(single_returns.max() * 100, 4),
        "Çarpıklık": round(single_returns.skew(), 4),
        "Basıklık": round(single_returns.kurt(), 4),
    }
})

st.dataframe(single_stats, use_container_width=True)


# =============================================================================
# 7. PORTFÖY FONKSİYONLARI
# =============================================================================

def clean_weights(weights, tolerance=1e-8):
    weights = np.array(weights, dtype=float)
    weights[np.abs(weights) < tolerance] = 0

    total = weights.sum()
    if total != 0:
        weights = weights / total

    return weights


def portfolio_return(weights, mu):
    return float(weights @ mu)


def portfolio_variance(weights, cov_matrix):
    return float(weights.T @ cov_matrix @ weights)


def portfolio_volatility(weights, cov_matrix):
    variance = portfolio_variance(weights, cov_matrix)
    return float(np.sqrt(max(variance, 0)))


def portfolio_sharpe(weights, mu, cov_matrix, rf):
    p_return = portfolio_return(weights, mu)
    p_volatility = portfolio_volatility(weights, cov_matrix)

    if p_volatility == 0:
        return np.nan

    return (p_return - rf) / p_volatility


def negative_sharpe(weights, mu, cov_matrix, rf):
    sharpe = portfolio_sharpe(weights, mu, cov_matrix, rf)

    if np.isnan(sharpe):
        return 0

    return -sharpe


def negative_return(weights, mu):
    return -portfolio_return(weights, mu)


def min_volatility_objective(weights, cov_matrix):
    return portfolio_volatility(weights, cov_matrix)


def build_weight_table(weights, label):
    return pd.DataFrame({
        "Sembol": symbols,
        "Şirket": [BIST_HISSELER.get(s, s) for s in symbols],
        f"{label} Ağırlık (%)": (weights * 100).round(2),
    })


def build_portfolio_summary(weights, label):
    p_return = portfolio_return(weights, expected_returns.values)
    p_volatility = portfolio_volatility(weights, annual_covariance.values)
    p_sharpe = portfolio_sharpe(
        weights,
        expected_returns.values,
        annual_covariance.values,
        risk_free_rate,
    )

    return {
        "Portföy": label,
        "Beklenen Yıllık Getiri (%)": round(p_return * 100, 2),
        "Yıllık Volatilite / Std Sapma (%)": round(p_volatility * 100, 2),
        "Sharpe": round(p_sharpe, 4) if not np.isnan(p_sharpe) else np.nan,
    }


# =============================================================================
# 8. MARKOWITZ OPTİMİZASYONU
# =============================================================================

st.markdown("---")
st.markdown("### 🧠 Markowitz Portföy Optimizasyonu")

if minimize is None:
    st.error("❌ scipy yüklü değil. Optimizasyon için `pip install scipy` çalıştırmalısın.")
    st.stop()

mu = expected_returns.values
cov = annual_covariance.values
n_assets = len(symbols)

bounds = tuple((0, max_weight) for _ in range(n_assets))

constraints_sum = {
    "type": "eq",
    "fun": lambda w: np.sum(w) - 1,
}

initial_weights = np.array([1 / n_assets] * n_assets)


min_vol_result = minimize(
    fun=min_volatility_objective,
    x0=initial_weights,
    args=(cov,),
    method="SLSQP",
    bounds=bounds,
    constraints=(constraints_sum,),
)

max_sharpe_result = minimize(
    fun=negative_sharpe,
    x0=initial_weights,
    args=(mu, cov, risk_free_rate),
    method="SLSQP",
    bounds=bounds,
    constraints=(constraints_sum,),
)

max_return_result = minimize(
    fun=negative_return,
    x0=initial_weights,
    args=(mu,),
    method="SLSQP",
    bounds=bounds,
    constraints=(constraints_sum,),
)

if not min_vol_result.success:
    st.warning(f"⚠️ Minimum volatilite optimizasyonu tam başarılı olmadı: {min_vol_result.message}")

if not max_sharpe_result.success:
    st.warning(f"⚠️ Max Sharpe optimizasyonu tam başarılı olmadı: {max_sharpe_result.message}")

if not max_return_result.success:
    st.warning(f"⚠️ Maksimum getiri optimizasyonu tam başarılı olmadı: {max_return_result.message}")

min_vol_weights = clean_weights(min_vol_result.x)
max_sharpe_weights = clean_weights(max_sharpe_result.x)
max_return_weights = clean_weights(max_return_result.x)

portfolio_summary_df = pd.DataFrame([
    build_portfolio_summary(min_vol_weights, "Minimum Volatilite"),
    build_portfolio_summary(max_sharpe_weights, "Maksimum Sharpe"),
    build_portfolio_summary(max_return_weights, "Maksimum Getiri"),
])

st.markdown("#### 📌 Optimize Portföy Özeti")
st.dataframe(portfolio_summary_df, use_container_width=True, hide_index=True)

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown("#### 🛡️ Minimum Volatilite Ağırlıkları")
    min_vol_table = build_weight_table(min_vol_weights, "Min Vol")
    st.dataframe(min_vol_table, use_container_width=True, hide_index=True)

with p2:
    st.markdown("#### 🚀 Maksimum Sharpe Ağırlıkları")
    max_sharpe_table = build_weight_table(max_sharpe_weights, "Max Sharpe")
    st.dataframe(max_sharpe_table, use_container_width=True, hide_index=True)

with p3:
    st.markdown("#### 📈 Maksimum Getiri Ağırlıkları")
    max_return_table = build_weight_table(max_return_weights, "Max Return")
    st.dataframe(max_return_table, use_container_width=True, hide_index=True)


# =============================================================================
# 9. EFFICIENT FRONTIER
# =============================================================================

st.markdown("---")
st.markdown("### 📉 Efficient Frontier")

frontier_points = st.slider(
    "Frontier nokta sayısı",
    min_value=10,
    max_value=200,
    value=100,
    step=5,
)

min_frontier_return = portfolio_return(min_vol_weights, mu)
max_frontier_return = portfolio_return(max_return_weights, mu)

frontier_df = pd.DataFrame()
frontier_weights_df = pd.DataFrame()
frontier_weights_long_df = pd.DataFrame()

if max_frontier_return <= min_frontier_return:
    st.warning(
        "⚠️ Efficient frontier için geçerli getiri aralığı oluşmadı. "
        "Hisse sayısını, tarih aralığını veya ağırlık sınırını kontrol et."
    )
else:
    target_returns = np.linspace(
        min_frontier_return,
        max_frontier_return,
        frontier_points,
    )

    frontier_rows = []
    previous_weights = min_vol_weights.copy()

    for target_return in target_returns:
        constraints_frontier = (
            constraints_sum,
            {
                "type": "eq",
                "fun": lambda w, target=target_return: portfolio_return(w, mu) - target,
            },
        )

        result = minimize(
            fun=min_volatility_objective,
            x0=previous_weights,
            args=(cov,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_frontier,
            options={
                "maxiter": 1000,
                "ftol": 1e-10,
            },
        )

        if not result.success:
            continue

        weights = clean_weights(result.x)
        p_return = portfolio_return(weights, mu)
        p_volatility = portfolio_volatility(weights, cov)
        p_sharpe = portfolio_sharpe(weights, mu, cov, risk_free_rate)

        if (
            np.isnan(p_return)
            or np.isnan(p_volatility)
            or p_volatility <= 0
        ):
            continue

        previous_weights = weights.copy()

        portfolio_no = len(frontier_rows) + 1
        portfolio_name = f"P{portfolio_no:02d}"

        row = {
            "Portföy": portfolio_name,
            "Beklenen Getiri (%)": p_return * 100,
            "Volatilite / Std Sapma (%)": p_volatility * 100,
            "Sharpe": p_sharpe,
        }

        for i, symbol in enumerate(symbols):
            row[f"{symbol} (%)"] = weights[i] * 100

        frontier_rows.append(row)

    frontier_full_df = pd.DataFrame(frontier_rows)

    if frontier_full_df.empty:
        st.warning("⚠️ Efficient frontier üretilemedi. Ağırlık sınırlarını veya hisse sayısını kontrol et.")
    else:
        frontier_full_df = frontier_full_df.drop_duplicates(
            subset=["Volatilite / Std Sapma (%)", "Beklenen Getiri (%)"]
        )
        frontier_full_df = frontier_full_df.sort_values("Volatilite / Std Sapma (%)").reset_index(drop=True)

        frontier_full_df["Portföy"] = [
            f"P{i + 1:02d}"
            for i in range(len(frontier_full_df))
        ]

        metric_columns = [
            "Portföy",
            "Beklenen Getiri (%)",
            "Volatilite / Std Sapma (%)",
            "Sharpe",
        ]

        weight_columns = [
            f"{symbol} (%)"
            for symbol in symbols
        ]

        frontier_df = frontier_full_df[metric_columns].copy()
        frontier_weights_df = frontier_full_df[metric_columns + weight_columns].copy()

        long_rows = []

        for _, row in frontier_weights_df.iterrows():
            for symbol in symbols:
                long_rows.append({
                    "Portföy": row["Portföy"],
                    "Sembol": symbol,
                    "Şirket": BIST_HISSELER.get(symbol, symbol),
                    "Ağırlık (%)": row[f"{symbol} (%)"],
                    "Beklenen Getiri (%)": row["Beklenen Getiri (%)"],
                    "Volatilite / Std Sapma (%)": row["Volatilite / Std Sapma (%)"],
                    "Sharpe": row["Sharpe"],
                })

        frontier_weights_long_df = pd.DataFrame(long_rows)

        min_vol_summary = build_portfolio_summary(min_vol_weights, "Minimum Volatilite")
        max_sharpe_summary = build_portfolio_summary(max_sharpe_weights, "Maksimum Sharpe")
        max_return_summary = build_portfolio_summary(max_return_weights, "Maksimum Getiri")

        if go is not None:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=frontier_df["Volatilite / Std Sapma (%)"],
                    y=frontier_df["Beklenen Getiri (%)"],
                    mode="lines+markers",
                    name="Efficient Frontier",
                    customdata=frontier_df[["Portföy", "Sharpe"]],
                    hovertemplate=(
                        "Portföy: %{customdata[0]}<br>"
                        "Volatilite / Std Sapma: %{x:.2f}%<br>"
                        "Beklenen Getiri: %{y:.2f}%<br>"
                        "Sharpe: %{customdata[1]:.3f}<br>"
                        "<extra></extra>"
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[min_vol_summary["Yıllık Volatilite / Std Sapma (%)"]],
                    y=[min_vol_summary["Beklenen Yıllık Getiri (%)"]],
                    mode="markers",
                    name="Minimum Volatilite",
                    marker=dict(size=12),
                    hovertemplate=(
                        "Minimum Volatilite<br>"
                        "Volatilite / Std Sapma: %{x:.2f}%<br>"
                        "Beklenen Getiri: %{y:.2f}%<br>"
                        "<extra></extra>"
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[max_sharpe_summary["Yıllık Volatilite / Std Sapma (%)"]],
                    y=[max_sharpe_summary["Beklenen Yıllık Getiri (%)"]],
                    mode="markers",
                    name="Maksimum Sharpe",
                    marker=dict(size=12),
                    hovertemplate=(
                        "Maksimum Sharpe<br>"
                        "Volatilite / Std Sapma: %{x:.2f}%<br>"
                        "Beklenen Getiri: %{y:.2f}%<br>"
                        "<extra></extra>"
                    ),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[max_return_summary["Yıllık Volatilite / Std Sapma (%)"]],
                    y=[max_return_summary["Beklenen Yıllık Getiri (%)"]],
                    mode="markers",
                    name="Maksimum Getiri",
                    marker=dict(size=12),
                    hovertemplate=(
                        "Maksimum Getiri<br>"
                        "Volatilite / Std Sapma: %{x:.2f}%<br>"
                        "Beklenen Getiri: %{y:.2f}%<br>"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                xaxis_title="Yıllık Standart Sapma / Volatilite (%)",
                yaxis_title="Beklenen Yıllık Getiri (%)",
                hovermode="closest",
                height=520,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("ℹ️ Daha iyi grafik için `pip install plotly` kurabilirsin.")
            st.line_chart(
                frontier_df.set_index("Volatilite / Std Sapma (%)")["Beklenen Getiri (%)"],
                use_container_width=True,
            )

        with st.expander("📋 Efficient Frontier Verisi"):
            st.dataframe(
                frontier_df.round(4),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### 🧮 Efficient Frontier Portföy Ağırlık Matrisi")
        st.caption(
            "Her satır efficient frontier üzerindeki bir portföyü, "
            "her hisse kolonu ise o portföydeki ağırlığı gösterir."
        )

        st.dataframe(
            frontier_weights_df.round(2),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        csv_frontier_weights = frontier_weights_df.round(6).to_csv(
            index=False,
            encoding="utf-8-sig",
        )

        st.download_button(
            label="⬇️ Frontier Ağırlık Matrisi İndir",
            data=csv_frontier_weights,
            file_name="efficient_frontier_agirlik_matrisi.csv",
            mime="text/csv",
        )

        st.markdown("### 🔎 Tek Portföy Ağırlık Detayı")

        selected_frontier_portfolio = st.selectbox(
            "Detayını görmek istediğin frontier portföyü:",
            options=frontier_df["Portföy"].tolist(),
            format_func=lambda p: (
                f"{p} — "
                f"Getiri %{frontier_df.loc[frontier_df['Portföy'] == p, 'Beklenen Getiri (%)'].iloc[0]:.2f}, "
                f"Std Sapma %{frontier_df.loc[frontier_df['Portföy'] == p, 'Volatilite / Std Sapma (%)'].iloc[0]:.2f}"
            ),
        )

        selected_weights_detail_df = frontier_weights_long_df[
            frontier_weights_long_df["Portföy"] == selected_frontier_portfolio
        ].copy()

        selected_weights_detail_df = selected_weights_detail_df[
            selected_weights_detail_df["Ağırlık (%)"] > 0.01
        ].sort_values("Ağırlık (%)", ascending=False)

        st.dataframe(
            selected_weights_detail_df.round(2),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# 10. TOPLU HİSSE KARŞILAŞTIRMA TABLOSU
# =============================================================================

st.markdown("---")
st.markdown("### 📋 Tüm Hisseler — Karşılaştırmalı İstatistikler")

comparison_rows = {}

for symbol in symbols:
    r = returns[symbol].dropna()
    annual_return = r.mean() * trading_days
    annual_volatility = r.std() * np.sqrt(trading_days)
    sharpe = (
        (annual_return - risk_free_rate) / annual_volatility
        if annual_volatility != 0
        else np.nan
    )

    comparison_rows[symbol] = {
        "Şirket": BIST_HISSELER.get(symbol, symbol),
        "Ort. Günlük Getiri (%)": round(r.mean() * 100, 4),
        "Std Sapma Günlük (%)": round(r.std() * 100, 4),
        "Yıllık Beklenen Getiri (%)": round(annual_return * 100, 2),
        "Yıllık Volatilite / Std Sapma (%)": round(annual_volatility * 100, 2),
        "Sharpe": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
        "Min Günlük Getiri (%)": round(r.min() * 100, 2),
        "Max Günlük Getiri (%)": round(r.max() * 100, 2),
        "Çarpıklık": round(r.skew(), 4),
        "Basıklık": round(r.kurt(), 4),
    }

comparison_df = pd.DataFrame(comparison_rows).T
comparison_df.index.name = "Sembol"

st.dataframe(comparison_df, use_container_width=True)

csv_comparison = comparison_df.to_csv(encoding="utf-8-sig")

st.download_button(
    label="⬇️ Karşılaştırmalı Tablo İndir",
    data=csv_comparison,
    file_name="markowitz_hisse_karsilastirma.csv",
    mime="text/csv",
)


# =============================================================================
# 11. SESSION STATE'E KAYDET
# =============================================================================

st.session_state["analysis_data"] = {
    "prices_raw": prices_raw,
    "prices": prices,
    "returns": returns,
    "return_type": return_type,
    "expected_returns": expected_returns,
    "daily_covariance": daily_covariance,
    "annual_covariance": annual_covariance,
    "correlation": correlation,
    "trading_days": trading_days,
    "risk_free_rate": risk_free_rate,
    "symbols": symbols,
    "min_vol_weights": pd.Series(min_vol_weights, index=symbols),
    "max_sharpe_weights": pd.Series(max_sharpe_weights, index=symbols),
    "max_return_weights": pd.Series(max_return_weights, index=symbols),
    "portfolio_summary": portfolio_summary_df,
    "efficient_frontier": frontier_df,
    "efficient_frontier_weights": frontier_weights_df,
    "efficient_frontier_weights_long": frontier_weights_long_df,
}

st.success("✅ Markowitz analiz verileri session state'e kaydedildi.")
