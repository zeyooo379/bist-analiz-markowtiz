"""
yfinance ile BIST hisse senedi verilerini çekme modülü.

Markowitz Modern Portföy Optimizasyonu için:
- auto_adjust=True kullanılır.
- Close serisi düzeltilmiş fiyat mantığıyla kullanılır.
- Tarih index'i normalize edilir.
- Veriler eski tarihten yeni tarihe sıralanır.
- Duplicate tarihler temizlenir.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import streamlit as st


def _normalize_symbol(sembol: str) -> str:
    """
    Kullanıcıdan gelen sembolü normalize eder.
    Örn: ' thyao ' -> 'THYAO'
    """
    return str(sembol).strip().upper()


def _to_yahoo_symbol(sembol: str) -> str:
    """
    BIST sembolünü Yahoo Finance formatına çevirir.
    Örn: THYAO -> THYAO.IS
    """
    temiz_sembol = _normalize_symbol(sembol)

    if temiz_sembol.endswith(".IS"):
        return temiz_sembol

    return f"{temiz_sembol}.IS"


def _inclusive_end_date(bitis: str) -> str:
    """
    yfinance end parametresi çoğu kullanımda bitiş tarihini dışlar.
    Kullanıcının seçtiği bitiş tarihini dahil etmek için 1 gün eklenir.
    """
    bitis_dt = pd.to_datetime(bitis)
    bitis_dt = bitis_dt + timedelta(days=1)
    return bitis_dt.strftime("%Y-%m-%d")


def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance'tan gelen fiyat datasını Markowitz için güvenli hale getirir.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # yfinance bazı durumlarda MultiIndex kolon döndürebiliyor.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Index datetime olmalı.
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # Timezone varsa kaldır.
    try:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
    except Exception:
        pass

    # Tarih sırası Markowitz için kritik.
    df = df.sort_index()

    # Duplicate tarih varsa son kaydı tut.
    df = df[~df.index.duplicated(keep="last")]

    # Kolonları numeric yap.
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Close yoksa bu veri Markowitz için kullanılamaz.
    if "Close" not in df.columns:
        return pd.DataFrame()

    # Close tamamen boşsa kullanma.
    df = df.dropna(subset=["Close"])

    return df


@st.cache_data(ttl=300, show_spinner=False)
def hisse_verisi_cek(
    semboller: List[str],
    baslangic: str,
    bitis: str,
) -> Dict[str, pd.DataFrame]:
    """
    Birden fazla BIST hissesi için fiyat verisi çeker.

    Args:
        semboller: Hisse sembol listesi. Örn: ["THYAO", "GARAN"]
        baslangic: Başlangıç tarihi. Format: YYYY-MM-DD
        bitis: Bitiş tarihi. Format: YYYY-MM-DD

    Returns:
        {
            "THYAO": DataFrame,
            "GARAN": DataFrame
        }
    """

    veriler: Dict[str, pd.DataFrame] = {}

    if not semboller:
        return veriler

    try:
        baslangic_dt = pd.to_datetime(baslangic)
        bitis_dt = pd.to_datetime(bitis)
    except Exception:
        st.warning("⚠️ Tarih formatı hatalı. Başlangıç ve bitiş tarihi YYYY-MM-DD olmalı.")
        return veriler

    if baslangic_dt >= bitis_dt:
        st.warning("⚠️ Başlangıç tarihi bitiş tarihinden önce olmalı.")
        return veriler

    yahoo_bitis = _inclusive_end_date(bitis)

    # Sıra koruyarak duplicate sembolleri temizle.
    temiz_semboller = []
    for sembol in semboller:
        temiz = _normalize_symbol(sembol)
        if temiz and temiz not in temiz_semboller:
            temiz_semboller.append(temiz)

    for sembol in temiz_semboller:
        yahoo_sembol = _to_yahoo_symbol(sembol)

        try:
            df = yf.download(
                yahoo_sembol,
                start=baslangic,
                end=yahoo_bitis,
                progress=False,
                auto_adjust=True,
                actions=False,
                threads=False,
            )

            df = _normalize_price_df(df)

            if df.empty:
                st.warning(f"⚠️ {sembol} için geçerli fiyat verisi bulunamadı.")
                continue

            if len(df) < 2:
                st.warning(f"⚠️ {sembol} için getiri hesaplamaya yetecek veri yok.")
                continue

            veriler[sembol] = df

        except Exception as e:
            st.warning(f"⚠️ {sembol} verisi çekilemedi: {str(e)}")

    return veriler


@st.cache_data(ttl=600, show_spinner=False)
def hisse_bilgisi_cek(sembol: str) -> Dict:
    """
    Hisse senedinin temel bilgilerini çeker.

    Args:
        sembol: Hisse sembolü. Örn: "THYAO"

    Returns:
        Şirket bilgileri sözlüğü.
    """

    yahoo_sembol = _to_yahoo_symbol(sembol)

    try:
        ticker = yf.Ticker(yahoo_sembol)
        info = ticker.info

        return {
            "Şirket Adı": info.get("longName", info.get("shortName", sembol)),
            "Sektör": info.get("sector", "Bilinmiyor"),
            "Endüstri": info.get("industry", "Bilinmiyor"),
            "Piyasa Değeri": info.get("marketCap", None),
            "F/K Oranı": info.get("trailingPE", None),
            "PD/DD": info.get("priceToBook", None),
            "Temettü Verimi (%)": info.get("dividendYield", None),
            "52H En Yüksek": info.get("fiftyTwoWeekHigh", None),
            "52H En Düşük": info.get("fiftyTwoWeekLow", None),
            "Ortalama Hacim": info.get("averageVolume", None),
            "Beta": info.get("beta", None),
            "Para Birimi": info.get("currency", "TRY"),
        }

    except Exception:
        return {
            "Şirket Adı": _normalize_symbol(sembol),
            "Hata": "Bilgi alınamadı",
        }


@st.cache_data(ttl=600, show_spinner=False)
def finansal_tablo_cek(sembol: str) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Şirketin finansal tablolarını çeker.

    Args:
        sembol: Hisse sembolü.

    Returns:
        {
            "gelir_tablosu": df veya None,
            "bilanco": df veya None,
            "nakit_akis": df veya None
        }
    """

    yahoo_sembol = _to_yahoo_symbol(sembol)

    try:
        ticker = yf.Ticker(yahoo_sembol)

        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow

        return {
            "gelir_tablosu": financials if financials is not None and not financials.empty else None,
            "bilanco": balance_sheet if balance_sheet is not None and not balance_sheet.empty else None,
            "nakit_akis": cashflow if cashflow is not None and not cashflow.empty else None,
        }

    except Exception:
        return {
            "gelir_tablosu": None,
            "bilanco": None,
            "nakit_akis": None,
        }


@st.cache_data(ttl=300, show_spinner=False)
def endeks_verisi_cek(
    endeks_sembol: str = "^XU100",
    donem: str = "1y",
) -> Optional[pd.DataFrame]:
    """
    BIST endeks verisini çeker.

    Args:
        endeks_sembol: Endeks sembolü. Varsayılan: ^XU100
        donem: Dönem. Örn: "1y", "6mo"

    Returns:
        Endeks fiyat verisi DataFrame veya None.
    """

    try:
        df = yf.download(
            endeks_sembol,
            period=donem,
            progress=False,
            auto_adjust=True,
            actions=False,
            threads=False,
        )

        df = _normalize_price_df(df)

        if not df.empty:
            return df

    except Exception:
        pass

    return None


def donem_tarihlerini_hesapla(donem: str) -> Tuple[str, str]:
    """
    Dönem kodundan başlangıç ve bitiş tarihlerini hesaplar.

    Args:
        donem: Dönem kodu. Örn: "1mo", "1y"

    Returns:
        (başlangıç, bitiş)
    """

    bitis = datetime.now()

    donem_harita = {
        "5d": timedelta(days=7),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
        "5y": timedelta(days=1825),
        "10y": timedelta(days=3650),
        "max": timedelta(days=7300),
    }

    delta = donem_harita.get(donem, timedelta(days=365))
    baslangic = bitis - delta

    return baslangic.strftime("%Y-%m-%d"), bitis.strftime("%Y-%m-%d")