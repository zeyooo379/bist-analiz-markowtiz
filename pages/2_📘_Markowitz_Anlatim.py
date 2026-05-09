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
    page_title="Markowitz Anlatım — Modern Portföy Teorisi",
    page_icon="📘",
    layout="wide",
)

init_portfolio_state()

st.markdown("## 📘 Markowitz Anlatım")
st.markdown("Modern Portföy Teorisi'nin matematiksel temelleri ve adım adım portföy optimizasyonu süreci.")
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

st.markdown("---")
st.markdown("## 📘 MARKOWITZ MATEMATİĞİ — ADIM ADIM AÇIKLAMA")

st.markdown("""
Bu bölümde Modern Portföy Teorisi'nin temel matematiksel hesaplamaları yapılır.

Amaç:
- Hisselerin günlük getirilerini hesaplamak
- Beklenen getiriyi bulmak
- Riski (varyans / volatilite) hesaplamak
- Hisseler arası ilişkiyi (kovaryans / korelasyon) ölçmek
- Bu verilerle optimum portföy oluşturmak
""")

st.markdown("---")
st.markdown("## 1️⃣ Günlük Getiri Hesabı")

st.markdown("""
Markowitz modeli doğrudan fiyatlarla çalışmaz.

Çünkü:
- Fiyat seviyeleri farklıdır
- 10 TL ile 1000 TL aynı ölçekte değildir

Bu yüzden fiyat değişim oranları yani GETİRİ kullanılır.
""")

st.markdown("### Basit Getiri Formülü")

st.latex(r"R_t = \frac{P_t - P_{t-1}}{P_{t-1}}")

st.markdown("""
Burada:

- \(R_t\) → Günlük getiri
- \(P_t\) → Bugünkü fiyat
- \(P_{t-1}\) → Önceki günün fiyatı

Örnek:
- Dün fiyat = 100
- Bugün fiyat = 105

Getiri:
""")

st.latex(r"R_t = \frac{105 - 100}{100} = 0.05 = \%5")

st.markdown("---")

st.markdown("### Logaritmik Getiri Formülü")

st.latex(r"R_t = \ln\left(\frac{P_t}{P_{t-1}}\right)")

st.markdown("""
Log getiri özellikle akademik çalışmalarda kullanılır.

Avantajları:
- İstatistiksel olarak daha düzgündür
- Uzun dönem analizlerde tercih edilir
- Sürekli bileşik getiriyi temsil eder

Ancak başlangıç seviyesinde basit getiri daha anlaşılırdır.
""")

# =============================================================================
# GETİRİ HESABI
# =============================================================================

if return_type == "Basit Getiri":

    st.info("📌 Şu anda BASİT GETİRİ yöntemi kullanılıyor.")

    returns = prices.pct_change(fill_method=None).dropna()

else:

    st.info("📌 Şu anda LOG GETİRİ yöntemi kullanılıyor.")

    returns = np.log(prices / prices.shift(1)).dropna()

# =============================================================================
# GETİRİ VERİSİ KONTROL
# =============================================================================

if len(returns) < 30:
    st.error("""
❌ Optimizasyon için yeterli getiri verisi yok.

Sebep:
Markowitz modeli istatistiksel bir modeldir.
Az veri kullanılırsa:
- Ortalama yanlış hesaplanabilir
- Risk yanlış hesaplanabilir
- Kovaryans matrisi bozulabilir

Bu yüzden en az 30 gözlem önerilir.
""")
    st.stop()

# =============================================================================
# BEKLENEN GETİRİ
# =============================================================================

st.markdown("---")
st.markdown("## 2️⃣ Beklenen Getiri Hesabı")

st.markdown("""
Beklenen getiri:
Bir hissenin ortalama olarak gelecekte ne kadar kazandırmasının beklendiğini gösterir.

Markowitz modeli için her hisseye ait:
- Ortalama getiri
- Risk
hesaplanmalıdır.
""")

st.markdown("### Beklenen Getiri Formülü")

st.latex(r"E[R] = \bar r \times 252")

st.markdown("""
Burada:

- \(E[R]\) → Beklenen yıllık getiri
- \(\bar r\) → Ortalama günlük getiri
- 252 → Yıllık işlem günü sayısı

BIST için genellikle 252 işlem günü kullanılır.
""")

expected_returns = returns.mean() * trading_days

st.markdown("""
Kodun yaptığı işlem:

1. Her hissenin günlük getirilerinin ortalamasını alır
2. Yıllıklaştırmak için 252 ile çarpar
""")

# =============================================================================
# VARYANS
# =============================================================================

st.markdown("---")
st.markdown("## 3️⃣ Risk Hesabı — Varyans")

st.markdown("""
Finansta risk:
Getirilerin ne kadar dalgalandığını gösterir.

En temel risk ölçüsü:
VARYANS'tır.
""")

st.markdown("### Varyans Formülü")

st.latex(r"\sigma^2 = \frac{1}{n-1}\sum_{i=1}^{n}(R_i-\bar R)^2")

st.markdown("""
Burada:

- \(R_i\) → Her günün getirisi
- \(\bar R\) → Ortalama getiri
- \(n\) → Gözlem sayısı
- \(\sigma^2\) → Varyans

Mantık:
- Ortalama getiriden uzaklaşma ölçülür
- Büyük dalgalanma = yüksek risk
""")

# =============================================================================
# VOLATİLİTE
# =============================================================================

st.markdown("---")
st.markdown("## 4️⃣ Volatilite (Standart Sapma)")

st.markdown("""
Varyansın karekökü alınırsa:
VOLATİLİTE elde edilir.

Finansta en yaygın risk ölçüsü budur.
""")

st.markdown("### Volatilite Formülü")

st.latex(r"\sigma = \sqrt{\sigma^2}")

st.markdown("""
Volatilite:
- Yüksekse → hisse daha riskli
- Düşükse → hisse daha stabil
""")

daily_covariance = returns.cov()

annual_covariance = daily_covariance * trading_days

# =============================================================================
# KOVARYANS
# =============================================================================

st.markdown("---")
st.markdown("## 5️⃣ Kovaryans Matrisi")

st.markdown("""
Markowitz modelinin en önemli kısmı:
KOVARYANS MATRİSİ'dir.

Çünkü portföy riski:
Sadece tek tek hisselerin riskine bağlı değildir.

Hisselerin birlikte nasıl hareket ettiği de önemlidir.
""")

st.markdown("### Kovaryans Formülü")

st.latex(r"Cov(X,Y)=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\bar X)(Y_i-\bar Y)")

st.markdown("""
Anlamı:

- Pozitif kovaryans:
  Hisseler birlikte yükselir/düşer

- Negatif kovaryans:
  Biri yükselirken diğeri düşebilir

Negatif ilişki:
Portföy riskini azaltır.
""")

st.markdown("""
Kodun yaptığı işlem:

- Her iki hisse arasındaki ilişki hesaplanır
- Tüm hisseler için büyük bir matris oluşturulur
""")

# =============================================================================
# KORELASYON
# =============================================================================

correlation = returns.corr()

st.markdown("---")
st.markdown("## 6️⃣ Korelasyon Matrisi")

st.markdown("""
Korelasyon:
Kovaryansın normalize edilmiş halidir.

Değer aralığı:
-1 ile +1 arasındadır.
""")

st.markdown("""
- +1 → Tam aynı hareket
- 0 → İlişki yok
- -1 → Tam ters hareket
""")

# =============================================================================
# MARKOWITZ GİRDİLERİ
# =============================================================================

st.markdown("---")
st.markdown("## 📈 Markowitz Modelinde Kullanılan Nihai Veriler")

col1, col2, col3 = st.columns(3)

col1.metric("Getiri Gözlem Sayısı", len(returns))
col2.metric("Beklenen Getiri Ölçeği", "Yıllık")
col3.metric("Kovaryans Ölçeği", "Yıllık")

expected_returns_df = pd.DataFrame({
    "Sembol": expected_returns.index,
    "Şirket": [BIST_HISSELER.get(s, s) for s in expected_returns.index],
    "Beklenen Yıllık Getiri (%)": (
        expected_returns.values * 100
    ).round(2),

    "Yıllık Volatilite / Standart Sapma (%)": (
        np.sqrt(np.diag(annual_covariance.values)) * 100
    ).round(2),
})

st.dataframe(
    expected_returns_df,
    use_container_width=True,
    hide_index=True,
)

# =============================================================================
# 7. PORTFÖY MATEMATİĞİ
# =============================================================================

st.markdown("---")
st.markdown("# 🧠 PORTFÖY MATEMATİĞİ")

st.markdown("""
Artık her hisse için:
- Beklenen getiri
- Risk
- Kovaryans

hesaplandı.

Şimdi sıra:
Bu hisseleri birleştirip optimum portföy oluşturmaya geldi.
""")

# =============================================================================
# PORTFÖY GETİRİSİ
# =============================================================================

st.markdown("---")
st.markdown("## 7️⃣ Portföy Beklenen Getirisi")

st.latex(r"E[R_p]=\sum_{i=1}^{n} w_i E[R_i]")

st.markdown("""
Burada:

- \(w_i\) → Hissenin portföy ağırlığı
- \(E[R_i]\) → Hissenin beklenen getirisi

Portföy getirisi:
Ağırlıklı ortalamadır.
""")

# =============================================================================
# PORTFÖY VARYANSI
# =============================================================================

st.markdown("---")
st.markdown("## 8️⃣ Portföy Riski")

st.latex(r"\sigma_p^2 = w^T \Sigma w")

st.markdown("""
Bu:
Modern Portföy Teorisi'nin en önemli formülüdür.

Burada:

- \(w\) → ağırlık vektörü
- \(\Sigma\) → kovaryans matrisi

Bu formül:
Tüm hisselerin birlikte oluşturduğu toplam portföy riskini hesaplar.
""")

# =============================================================================
# PORTFÖY VOLATİLİTESİ
# =============================================================================

st.markdown("---")
st.markdown("## 9️⃣ Portföy Volatilitesi")

st.latex(r"\sigma_p = \sqrt{w^T \Sigma w}")

st.markdown("""
Portföy volatilitesi:
Portföyün toplam riskidir.
""")

# =============================================================================
# SHARPE
# =============================================================================

st.markdown("---")
st.markdown("## 🔟 Sharpe Oranı")

st.latex(r"Sharpe = \frac{R_p - R_f}{\sigma_p}")

st.markdown("""
Sharpe oranı:
Bir yatırımın risk başına ne kadar getiri sağladığını gösterir.

- Büyük Sharpe → daha verimli portföy
- Küçük Sharpe → riskine göre zayıf getiri
""")

# =============================================================================
# FONKSİYONLAR
# =============================================================================

def clean_weights(weights, tolerance=1e-8):

    weights = np.array(weights, dtype=float)

    weights[np.abs(weights) < tolerance] = 0

    total = weights.sum()

    if total != 0:
        weights = weights / total

    return weights


def portfolio_return(weights, mu):

    """
    Portföy beklenen getirisi hesabı

    Formül:
    E[R_p]=∑w_iE[R_i]
    """

    return float(weights @ mu)


def portfolio_variance(weights, cov_matrix):

    """
    Portföy varyansı hesabı

    Formül:
    σ² = wᵀΣw
    """

    return float(weights.T @ cov_matrix @ weights)


def portfolio_volatility(weights, cov_matrix):

    """
    Portföy volatilitesi hesabı

    Formül:
    σ = √(wᵀΣw)
    """

    variance = portfolio_variance(weights, cov_matrix)

    return float(np.sqrt(max(variance, 0)))


def portfolio_sharpe(weights, mu, cov_matrix, rf):

    """
    Sharpe oranı hesabı

    Formül:
    Sharpe = (Rp - Rf) / σ
    """

    p_return = portfolio_return(weights, mu)

    p_volatility = portfolio_volatility(weights, cov_matrix)

    if p_volatility == 0:
        return np.nan

    return (p_return - rf) / p_volatility


# =============================================================================
# MARKOWITZ OPTİMİZASYONU
# =============================================================================

st.markdown("---")
st.markdown("# 🎯 MARKOWITZ OPTİMİZASYONU")

st.markdown("""
Markowitz modelinin amacı:

📌 Aynı getiri için en düşük riski bulmak

veya

📌 Aynı risk için en yüksek getiriyi bulmaktır.
""")

st.markdown("## Minimum Risk Problemi")

st.latex(r"\min_w \; w^T\Sigma w")

st.markdown("""
Amaç:
Portföy varyansını minimum yapmak.
""")

st.markdown("## Ağırlık Kısıtı")

st.latex(r"\sum_{i=1}^{n} w_i = 1")

st.markdown("""
Portföydeki tüm ağırlıkların toplamı:
%100 olmak zorundadır.
""")

st.markdown("""
Ek olarak:

Her hisse için:
""")

st.latex(r"0 \leq w_i \leq w_{max}")

st.markdown("""
Bu da:
Bir hisseye aşırı yüklenmeyi engeller.
""")
