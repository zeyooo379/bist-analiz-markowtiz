import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from utils.constants import get_bist_hisseler
from utils.stock_data import hisse_verisi_cek, hisse_bilgisi_cek
from utils.app_state import init_portfolio_state, reset_analysis_outputs


st.set_page_config(
    page_title="Veri Çekme — BIST Analiz",
    page_icon="📥",
    layout="wide",
)

init_portfolio_state()

st.markdown("## 📥 Veri Çekme")
st.markdown("BIST hisse senetlerinin fiyat verilerini ve şirket bilgilerini çekin.")
st.markdown("---")


# =============================================================================
# BIST HİSSE LİSTESİ
# =============================================================================

BIST_HISSELER = get_bist_hisseler(use_remote=True)


# =============================================================================
# AYARLAR
# =============================================================================

MIN_DAYS_REQUIRED = 30
RISK_DAYS_THRESHOLD = 120
RECOMMENDED_DAYS = 252
MIN_COVERAGE_PCT = 70


# =============================================================================
# HELPER FONKSİYONLAR
# =============================================================================

def symbol_label(symbol):
    return f"{symbol} — {BIST_HISSELER.get(symbol, symbol)}"


def set_quick_selection(symbols):
    valid_symbols = [
        symbol
        for symbol in symbols
        if symbol in BIST_HISSELER
    ]

    st.session_state["selected_symbols_ui"] = valid_symbols


def prepare_close_series(df):
    """
    DataFrame içinden Markowitz için kullanılabilir Close serisi üretir.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype="float64")

    temp = df.copy()
    temp.index = pd.to_datetime(temp.index, errors="coerce")
    temp = temp[~temp.index.isna()]
    temp = temp.sort_index()
    temp = temp[~temp.index.duplicated(keep="last")]

    close = pd.to_numeric(temp["Close"], errors="coerce").dropna()
    close.name = "Close"

    return close


def build_close_dataframe(veriler):
    """
    Seçilen hisselerden kapanış fiyatı DataFrame'i oluşturur.
    """
    close_series = {}

    for sembol, df in veriler.items():
        close = prepare_close_series(df)

        if not close.empty:
            close_series[sembol] = close

    if not close_series:
        return pd.DataFrame()

    return pd.DataFrame(close_series).sort_index()


def calculate_calendar_stats(veriler):
    """
    Ham tarih sayısı ve ortak tarih sayısını hesaplar.
    """
    close_df = build_close_dataframe(veriler)

    if close_df.empty:
        return 0, 0, 0

    raw_day_count = len(close_df.dropna(how="all"))
    common_day_count = len(close_df.dropna(how="any"))
    dropped_day_count = raw_day_count - common_day_count

    return raw_day_count, common_day_count, dropped_day_count


def build_data_quality_report(veriler):
    """
    Veri çekildikten sonra her hisse için kalite raporu oluşturur.

    Durumlar:
    - Uygun: Analiz için yeterli.
    - Dikkat: Kullanılabilir ama 1 yıldan az veri var.
    - Riskli: Markowitz için veri sayısı/kapsama düşük.
    - Kritik: Çok az veri var; kullanıcı isterse çıkarabilir.

    Not:
    Az verili hisseler otomatik çıkarılmaz.
    Sadece kullanıcıya uyarı verilir.
    """
    rows = []
    close_counts = {}

    for sembol, df in veriler.items():
        close = prepare_close_series(df)
        close_counts[sembol] = len(close)

    max_count = max(close_counts.values()) if close_counts else 0

    for sembol, df in veriler.items():
        close = prepare_close_series(df)
        veri_sayisi = len(close)

        if veri_sayisi == 0:
            rows.append({
                "Sembol": sembol,
                "Şirket": BIST_HISSELER.get(sembol, sembol),
                "Veri Sayısı": 0,
                "Kapsama (%)": 0.0,
                "İlk Tarih": "-",
                "Son Tarih": "-",
                "Durum": "Geçersiz",
                "Açıklama": "Geçerli Close verisi yok",
                "Öneri": "Bu hisse analizde kullanılamaz",
                "Varsayılan Dahil": False,
            })
            continue

        coverage_pct = (veri_sayisi / max_count * 100) if max_count > 0 else 0

        if veri_sayisi < MIN_DAYS_REQUIRED:
            durum = "Kritik"
            aciklama = f"{MIN_DAYS_REQUIRED} günden az veri var"
            oneri = "Modeli ciddi daraltabilir; istersen çıkar"
            varsayilan_dahil = True
        elif veri_sayisi < RISK_DAYS_THRESHOLD:
            durum = "Riskli"
            aciklama = "Markowitz için veri sayısı düşük"
            oneri = "Dahil edilebilir ama sonuçlar daha oynak olabilir"
            varsayilan_dahil = True
        elif coverage_pct < MIN_COVERAGE_PCT:
            durum = "Riskli"
            aciklama = "Diğer hisselere göre veri kapsamı düşük"
            oneri = "Ortak tarih sayısını düşürebilir; istersen çıkar"
            varsayilan_dahil = True
        elif veri_sayisi < RECOMMENDED_DAYS:
            durum = "Dikkat"
            aciklama = "Kullanılabilir ama 1 yıldan az veri var"
            oneri = "Analizde kullanılabilir"
            varsayilan_dahil = True
        else:
            durum = "Uygun"
            aciklama = "Analiz için uygun"
            oneri = "Analizde kullanılabilir"
            varsayilan_dahil = True

        rows.append({
            "Sembol": sembol,
            "Şirket": BIST_HISSELER.get(sembol, sembol),
            "Veri Sayısı": veri_sayisi,
            "Kapsama (%)": round(coverage_pct, 2),
            "İlk Tarih": close.index[0].strftime("%Y-%m-%d"),
            "Son Tarih": close.index[-1].strftime("%Y-%m-%d"),
            "Durum": durum,
            "Açıklama": aciklama,
            "Öneri": oneri,
            "Varsayılan Dahil": varsayilan_dahil,
        })

    return pd.DataFrame(rows)


def apply_analysis_selection(raw_veriler, selected_symbols):
    """
    Kullanıcının analize dahil ettiği hisseleri session_state'e uygular.
    """
    selected_symbols = [
        sembol
        for sembol in selected_symbols
        if sembol in raw_veriler
    ]

    final_veriler = {
        sembol: raw_veriler[sembol]
        for sembol in selected_symbols
    }

    previous_selection = st.session_state.get("_last_applied_analysis_symbols", [])

    st.session_state["stock_data"] = final_veriler
    st.session_state["selected_symbols"] = list(final_veriler.keys())

    if previous_selection != list(final_veriler.keys()):
        reset_analysis_outputs()
        st.session_state["_last_applied_analysis_symbols"] = list(final_veriler.keys())

    return final_veriler


# =============================================================================
# 1. HİSSE SEÇİMİ
# =============================================================================

st.markdown("### 🏢 Hisse Seçimi")

all_symbols = list(BIST_HISSELER.keys())

if "selected_symbols_ui" not in st.session_state:
    st.session_state["selected_symbols_ui"] = []

# Remote liste değişirse session'daki eski/geçersiz sembolleri temizle
st.session_state["selected_symbols_ui"] = [
    symbol
    for symbol in st.session_state["selected_symbols_ui"]
    if symbol in BIST_HISSELER
]

col_b1, col_b2, col_b3, col_b4 = st.columns(4)

with col_b1:
    if st.button("🔝 BIST 30 Popüler", use_container_width=True):
        set_quick_selection([
            "THYAO", "GARAN", "ASELS", "AKBNK", "EREGL",
            "KCHOL", "SAHOL", "TUPRS", "SISE", "FROTO",
        ])

with col_b2:
    if st.button("🏦 Bankalar", use_container_width=True):
        set_quick_selection([
            "GARAN", "AKBNK", "ISCTR", "YKBNK", "VAKBN", "HALKB",
        ])

with col_b3:
    if st.button("🏭 Sanayi", use_container_width=True):
        set_quick_selection([
            "EREGL", "TUPRS", "PETKM", "SISE", "KRDMD", "BRISA",
        ])

with col_b4:
    if st.button("🧹 Temizle", use_container_width=True):
        st.session_state["selected_symbols_ui"] = []


secili_semboller = st.multiselect(
    "Hisse senedi seçin:",
    options=all_symbols,
    key="selected_symbols_ui",
    format_func=symbol_label,
    placeholder="Sembol veya şirket adı yazın: THYAO, Garanti, Aselsan...",
    help=(
        "Arama kutusuna sembol veya şirket adı yazarak hisse seçebilirsiniz. "
        "Seçilen hisseler veri çekme ve Markowitz analizi için aday olarak alınır."
    ),
)

if secili_semboller:
    st.caption(
        f"Seçili hisseler ({len(secili_semboller)}): "
        f"`{'  |  '.join(secili_semboller)}`"
    )

st.markdown("---")


# =============================================================================
# 2. TARİH ARALIĞI
# =============================================================================

st.markdown("### 📅 Tarih Aralığı")

DONEM_SECENEKLERI = {
    "1 Hafta": 7,
    "1 Ay": 30,
    "3 Ay": 90,
    "6 Ay": 180,
    "1 Yıl": 365,
    "2 Yıl": 730,
    "5 Yıl": 1825,
    "10 Yıl": 3650,
}

bugun = datetime.now()
en_eski = bugun - timedelta(days=3650)

secili_donem = st.select_slider(
    "Dönem seçin:",
    options=list(DONEM_SECENEKLERI.keys()),
    value="1 Yıl",
)

gun_sayisi = DONEM_SECENEKLERI[secili_donem]
varsayilan_baslangic = bugun - timedelta(days=gun_sayisi)

col_t1, col_t2 = st.columns(2)

with col_t1:
    baslangic_tarih = st.date_input(
        "Başlangıç tarihi:",
        value=varsayilan_baslangic.date(),
        min_value=en_eski.date(),
        max_value=bugun.date(),
    )

with col_t2:
    bitis_tarih = st.date_input(
        "Bitiş tarihi:",
        value=bugun.date(),
        min_value=en_eski.date(),
        max_value=bugun.date(),
    )

baslangic = baslangic_tarih.strftime("%Y-%m-%d")
bitis = bitis_tarih.strftime("%Y-%m-%d")

tarih_gecerli = True

if baslangic_tarih >= bitis_tarih:
    st.error("❌ Başlangıç tarihi bitiş tarihinden önce olmalı.")
    tarih_gecerli = False

st.caption(f"📅 {baslangic}  →  {bitis}  (maks. 10 yıl)")

st.markdown("---")


# =============================================================================
# 3. VERİ ÇEKME
# =============================================================================

if secili_semboller:
    st.markdown(
        f"**Verisi çekilecek hisseler ({len(secili_semboller)}):** "
        f"`{'  |  '.join(secili_semboller)}`"
    )

    verileri_cek = st.button(
        "🚀 Verileri Çek",
        type="primary",
        use_container_width=True,
        disabled=not tarih_gecerli,
    )

    if verileri_cek:
        with st.spinner(f"📡 {len(secili_semboller)} hisse verisi çekiliyor..."):
            raw_veriler = hisse_verisi_cek(
                secili_semboller,
                baslangic,
                bitis,
            )

        if raw_veriler:
            kalite_df = build_data_quality_report(raw_veriler)

            # Az verili hisseler otomatik çıkarılmaz.
            # Geçerli veri gelen tüm hisseler default seçili gelir.
            default_analysis_symbols = [
                sembol
                for sembol in raw_veriler.keys()
                if sembol in kalite_df["Sembol"].values
            ]

            st.session_state["raw_stock_data"] = raw_veriler
            st.session_state["data_quality_report"] = kalite_df
            st.session_state["analysis_symbols"] = default_analysis_symbols
            st.session_state["_last_applied_analysis_symbols"] = []

            reset_analysis_outputs()

            st.success(f"✅ {len(raw_veriler)} hisse verisi başarıyla çekildi!")

            basarili_semboller = list(raw_veriler.keys())
            basarisiz_semboller = [
                sembol
                for sembol in secili_semboller
                if sembol not in basarili_semboller
            ]

            if basarisiz_semboller:
                st.warning(
                    f"⚠️ Şu semboller için geçerli veri alınamadı: "
                    f"{', '.join(basarisiz_semboller)}"
                )
        else:
            st.error(
                "❌ Hiçbir hisse verisi çekilemedi. "
                "Sembolleri ve tarih aralığını kontrol edip tekrar deneyin."
            )

else:
    st.warning("⚠️ Lütfen en az bir hisse senedi seçin.")
    st.info("💡 Arama kutusuna sembol veya şirket adı yazarak hisse seçebilirsiniz.")


# =============================================================================
# 4. VERİ KALİTESİ VE ANALİZ SEÇİMİ
# =============================================================================

raw_veriler = st.session_state.get("raw_stock_data", {})
kalite_df = st.session_state.get("data_quality_report", pd.DataFrame())

if raw_veriler:
    st.markdown("---")
    st.markdown("### 🧪 Veri Kalitesi Kontrolü")

    if kalite_df is None or kalite_df.empty:
        kalite_df = build_data_quality_report(raw_veriler)
        st.session_state["data_quality_report"] = kalite_df

    display_kalite_df = kalite_df.drop(columns=["Varsayılan Dahil"], errors="ignore")

    st.dataframe(
        display_kalite_df,
        use_container_width=True,
        hide_index=True,
    )

    kritik_hisseler = kalite_df[
        kalite_df["Durum"] == "Kritik"
    ]["Sembol"].tolist()

    riskli_hisseler = kalite_df[
        kalite_df["Durum"] == "Riskli"
    ]["Sembol"].tolist()

    dikkat_hisseler = kalite_df[
        kalite_df["Durum"] == "Dikkat"
    ]["Sembol"].tolist()

    gecersiz_hisseler = kalite_df[
        kalite_df["Durum"] == "Geçersiz"
    ]["Sembol"].tolist()

    if kritik_hisseler:
        st.warning(
            "⚠️ Bazı hisselerde çok az veri var. "
            "Bu hisseler otomatik çıkarılmadı; istersen aşağıdaki seçimden çıkarabilirsin. "
            f"Kritik: {', '.join(kritik_hisseler)}"
        )

    if riskli_hisseler:
        st.warning(
            "⚠️ Bazı hisselerde veri sayısı veya veri kapsamı düşük. "
            "Bu hisseler Markowitz modelinde ortak tarih sayısını düşürebilir. "
            f"Riskli: {', '.join(riskli_hisseler)}"
        )

    if dikkat_hisseler:
        st.info(
            "ℹ️ Bazı hisselerde 1 yıldan az veri var ama analizde kullanılabilir: "
            f"{', '.join(dikkat_hisseler)}"
        )

    if gecersiz_hisseler:
        st.error(
            "❌ Bazı hisselerde geçerli Close verisi yok, bu hisseler analizde kullanılamaz: "
            f"{', '.join(gecersiz_hisseler)}"
        )

    available_symbols = [
        sembol
        for sembol in raw_veriler.keys()
        if sembol not in gecersiz_hisseler
    ]

    if "analysis_symbols" not in st.session_state:
        st.session_state["analysis_symbols"] = available_symbols

    # Raw veri değiştiyse analysis_symbols içinde artık olmayan sembolleri temizle
    st.session_state["analysis_symbols"] = [
        symbol
        for symbol in st.session_state["analysis_symbols"]
        if symbol in available_symbols
    ]

    # Hiç seçim yoksa ama veri varsa, tüm geçerli hisseleri default olarak getir.
    # Böylece az verili hisseler otomatik çıkarılmaz.
    if not st.session_state["analysis_symbols"] and available_symbols:
        st.session_state["analysis_symbols"] = available_symbols

    analize_dahil_edilecekler = st.multiselect(
        "Markowitz analizine dahil edilecek hisseler:",
        options=available_symbols,
        key="analysis_symbols",
        format_func=symbol_label,
        help=(
            "Az verili hisseler otomatik çıkarılmaz. "
            "Uyarıları kontrol edip istemediğin hisseleri buradan çıkarabilirsin."
        ),
    )

    final_veriler = apply_analysis_selection(
        raw_veriler,
        analize_dahil_edilecekler,
    )

    if len(final_veriler) < 2:
        st.error("❌ Markowitz için en az 2 hisse seçmelisin.")
        st.stop()

    raw_day_count, common_day_count, dropped_day_count = calculate_calendar_stats(final_veriler)

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    col_q1.metric("Analize Dahil Hisse", len(final_veriler))
    col_q2.metric("Ham Tarih Sayısı", raw_day_count)
    col_q3.metric("Ortak Tarih Sayısı", common_day_count)
    col_q4.metric("Çıkarılan Tarih", dropped_day_count)

    if common_day_count < MIN_DAYS_REQUIRED:
        st.warning(
            f"⚠️ Seçili hisselerle ortak işlem günü {common_day_count}. "
            f"Markowitz için en az {MIN_DAYS_REQUIRED} ortak işlem günü önerilir. "
            "Az verili hisseleri seçimden çıkararak ortak tarih sayısını artırabilirsin."
        )

    elif common_day_count < RISK_DAYS_THRESHOLD:
        st.warning(
            f"⚠️ Ortak işlem günü {common_day_count}. "
            "Optimizasyon çalışır ama sonuçlar veri azlığı nedeniyle daha oynak olabilir."
        )

    st.success(
        f"✅ Markowitz analizine hazır hisseler: {', '.join(final_veriler.keys())}"
    )

    # =========================================================================
    # 5. GÜNLÜK KAPANIŞ FİYATLARI TABLOSU
    # =========================================================================

    st.markdown("---")
    st.markdown("### 📅 Günlük Kapanış Fiyatları")

    kapanis_df = build_close_dataframe(final_veriler)

    if not kapanis_df.empty:
        kapanis_gosterim_df = kapanis_df.copy()
        kapanis_gosterim_df = kapanis_gosterim_df.sort_index(ascending=False)
        kapanis_gosterim_df = kapanis_gosterim_df.round(2)
        kapanis_gosterim_df.index = kapanis_gosterim_df.index.strftime("%Y-%m-%d")
        kapanis_gosterim_df.index.name = "Tarih"

        st.dataframe(
            kapanis_gosterim_df,
            use_container_width=True,
            height=400,
        )

        st.caption(
            f"📊 Toplam {len(kapanis_gosterim_df)} tarih gösteriliyor. "
            "En yeni tarih üstte."
        )
    else:
        st.info("ℹ️ Gösterilecek kapanış fiyatı bulunamadı.")

    # =========================================================================
    # 6. DÖNEM ÖZETİ
    # =========================================================================

    st.markdown("---")
    st.markdown("### 📋 Dönem Özeti")

    ozet_liste = []

    for sembol, df in final_veriler.items():
        close = prepare_close_series(df)

        if close.empty:
            continue

        ilk_fiyat = close.iloc[0]
        son_fiyat = close.iloc[-1]

        if ilk_fiyat != 0:
            degisim = ((son_fiyat - ilk_fiyat) / ilk_fiyat) * 100
        else:
            degisim = 0

        ozet_liste.append({
            "Sembol": sembol,
            "Şirket": BIST_HISSELER.get(sembol, sembol),
            "İlk Tarih": close.index[0].strftime("%Y-%m-%d"),
            "Son Tarih": close.index[-1].strftime("%Y-%m-%d"),
            "Veri Sayısı": len(close),
            "İlk Fiyat (₺)": f"{ilk_fiyat:.2f}",
            "Son Fiyat (₺)": f"{son_fiyat:.2f}",
            "Dönem Getiri (%)": f"{degisim:+.2f}",
        })

    if ozet_liste:
        ozet_df = pd.DataFrame(ozet_liste)
        st.dataframe(
            ozet_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("ℹ️ Dönem özeti oluşturulamadı.")

    # =========================================================================
    # 7. HİSSE BAZLI DETAY
    # =========================================================================

    st.markdown("---")
    st.markdown("### 📊 Hisse Detayları (Analize Dahil Edilenler)")

    tabs = st.tabs([f"📈 {sembol}" for sembol in final_veriler.keys()])

    for tab, (sembol, df) in zip(tabs, final_veriler.items()):
        with tab:
            col_info, col_table = st.columns([1, 2])

            with col_info:
                st.markdown(f"#### {BIST_HISSELER.get(sembol, sembol)}")

                with st.spinner("Şirket bilgisi yükleniyor..."):
                    bilgi = hisse_bilgisi_cek(sembol)

                for anahtar, deger in bilgi.items():
                    if deger is None or anahtar == "Hata":
                        continue

                    if isinstance(deger, (int, float)):
                        if anahtar == "Piyasa Değeri":
                            deger = f"₺{deger / 1e9:.1f}B"
                        elif anahtar == "Temettü Verimi (%)":
                            deger = f"%{deger * 100:.2f}"
                        elif anahtar == "Ortalama Hacim":
                            deger = f"{deger / 1e6:.1f}M"
                        else:
                            deger = (
                                f"{deger:.2f}"
                                if isinstance(deger, float)
                                else str(deger)
                            )

                    st.markdown(f"**{anahtar}:** {deger}")

            with col_table:
                st.markdown("**Son 10 İşlem Günü:**")

                if df is None or df.empty:
                    st.info("ℹ️ Gösterilecek veri bulunamadı.")
                    continue

                gosterim_df = df.copy()
                gosterim_df.index = pd.to_datetime(
                    gosterim_df.index,
                    errors="coerce",
                )
                gosterim_df = gosterim_df[~gosterim_df.index.isna()]
                gosterim_df = gosterim_df.sort_index()
                gosterim_df = gosterim_df.tail(10).sort_index(ascending=False)

                gosterim_df.index = gosterim_df.index.strftime("%Y-%m-%d")
                gosterim_df.index.name = "Tarih"

                numeric_cols = gosterim_df.select_dtypes(
                    include=["number"]
                ).columns

                gosterim_df[numeric_cols] = gosterim_df[numeric_cols].round(2)

                st.dataframe(
                    gosterim_df,
                    use_container_width=True,
                )