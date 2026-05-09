"""
BIST Hisse Senedi Listesi ve Sabit Değerler

Not:
- BIST_HISSELER fallback listedir.
- Uygulamada autocomplete için mümkünse get_bist_hisseler() kullanılmalıdır.
- get_bist_hisseler(), Borsa İstanbul resmi CSV kaynağından güncel hisse listesini çeker.
"""

from functools import lru_cache
from typing import Dict

import pandas as pd


# =============================================================================
# RESMİ BIST KAYNAKLARI
# =============================================================================

BIST_ALL_SHARES_CSV_URL = "https://www.borsaistanbul.com/datum/hisse_endeks_ds.csv"


# =============================================================================
# FALLBACK HİSSE LİSTESİ
# =============================================================================
# Bu liste internet bağlantısı/API kaynağı çalışmazsa kullanılır.
# Uygulamada doğrudan BIST_HISSELER yerine get_bist_hisseler() kullanmak daha doğrudur.

BIST_HISSELER = {
    # BIST 30 / popüler
    "AKBNK": "Akbank",
    "ARCLK": "Arçelik",
    "ASELS": "Aselsan",
    "BIMAS": "BİM Mağazalar",
    "EKGYO": "Emlak Konut GYO",
    "ENKAI": "Enka İnşaat",
    "EREGL": "Ereğli Demir Çelik",
    "FROTO": "Ford Otosan",
    "GARAN": "Garanti BBVA",
    "GUBRF": "Gübre Fabrikaları",
    "HEKTS": "Hektaş",
    "ISCTR": "İş Bankası C",
    "KCHOL": "Koç Holding",
    "KOZAA": "Koza Altın",
    "KOZAL": "Koza Anadolu Metal",
    "KRDMD": "Kardemir D",
    "MGROS": "Migros",
    "OYAKC": "Oyak Çimento",
    "PETKM": "Petkim",
    "PGSUS": "Pegasus",
    "SAHOL": "Sabancı Holding",
    "SASA": "SASA Polyester",
    "SISE": "Şişecam",
    "SOKM": "Şok Marketler",
    "TAVHL": "TAV Havalimanları",
    "TCELL": "Turkcell",
    "THYAO": "Türk Hava Yolları",
    "TKFEN": "Tekfen Holding",
    "TOASO": "Tofaş",
    "TUPRS": "Tüpraş",
    "TTKOM": "Türk Telekom",
    "VAKBN": "Vakıfbank",
    "VESTL": "Vestel Elektronik",
    "YKBNK": "Yapı Kredi",

    # Ek popüler
    "AEFES": "Anadolu Efes",
    "AKSA": "Aksa Akrilik",
    "ALARK": "Alarko Holding",
    "AYGAZ": "Aygaz",
    "BAGFS": "Bagfaş",
    "BRISA": "Brisa",
    "CCOLA": "Coca-Cola İçecek",
    "CIMSA": "Çimsa",
    "DOHOL": "Doğan Holding",
    "ECILC": "Eczacıbaşı İlaç",
    "EGEEN": "Ege Endüstri",
    "GESAN": "Girişim Elektrik",
    "HALKB": "Halkbank",
    "ISGYO": "İş GYO",
    "KARSN": "Karsan Otomotiv",
    "KONTR": "Kontrolmatik",
    "LOGO": "Logo Yazılım",
    "MAVI": "Mavi Giyim",
    "NETAS": "Netaş Telekom",
    "ODAS": "Odaş Elektrik",
    "OTKAR": "Otokar",
    "PAPIL": "Papilon Savunma",
    "SARKY": "Sarkuysan",
    "SMRTG": "Smart Güneş Enerjisi",
    "TATGD": "Tat Gıda",
    "TMSN": "Tümosan",
    "TTRAK": "Türk Traktör",
    "TURSG": "Türkiye Sigorta",
    "ULKER": "Ülker Bisküvi",
    "VESBE": "Vestel Beyaz Eşya",
    "YEOTK": "Yeo Teknoloji",
}


# =============================================================================
# BIST HİSSE LİSTESİ YÜKLEYİCİ
# =============================================================================

def _normalize_bist_symbol(symbol: str) -> str:
    """
    Borsa İstanbul CSV içindeki sembolü uygulama formatına çevirir.

    Örnek:
    AEFES.E -> AEFES
    THYAO.E -> THYAO
    """
    if not symbol:
        return ""

    normalized = str(symbol).strip().upper()

    if normalized.endswith(".E"):
        normalized = normalized[:-2]

    return normalized


def _normalize_company_name(name: str) -> str:
    """
    Şirket adını sadeleştirir.
    """
    if not name:
        return ""

    return str(name).strip()


def _read_bist_csv(url: str) -> pd.DataFrame:
    """
    Borsa İstanbul CSV dosyasını farklı encoding ihtimallerine göre okumayı dener.
    """
    last_error = None

    for encoding in ["utf-8", "utf-8-sig", "cp1254", "latin5", "iso-8859-9"]:
        try:
            return pd.read_csv(
                url,
                sep=";",
                header=None,
                encoding=encoding,
            )
        except Exception as error:
            last_error = error

    raise last_error


@lru_cache(maxsize=1)
def get_bist_hisseler_from_official() -> Dict[str, str]:
    """
    Borsa İstanbul resmi hisse/endeks CSV kaynağından güncel hisse listesini çeker.

    CSV formatı pratikte şu yapıda gelir:
    symbol;company;index_code;index_name_tr;index_name_en;date

    Aynı hisse farklı endekslerde geçebileceği için sembole göre tekilleştirme yapılır.
    """
    df = _read_bist_csv(BIST_ALL_SHARES_CSV_URL)

    if df.empty or df.shape[1] < 2:
        return {}

    # İlk iki kolon: sembol ve şirket adı
    df = df.iloc[:, :2].copy()
    df.columns = ["symbol", "company"]

    df["symbol"] = df["symbol"].apply(_normalize_bist_symbol)
    df["company"] = df["company"].apply(_normalize_company_name)

    df = df[
        (df["symbol"] != "")
        & (df["company"] != "")
    ]

    # Aynı sembol birden fazla endekste geçebilir; ilkini tutuyoruz.
    df = df.drop_duplicates(subset=["symbol"], keep="first")

    hisseler = dict(zip(df["symbol"], df["company"]))

    return dict(sorted(hisseler.items()))


def get_bist_hisseler(use_remote: bool = True) -> Dict[str, str]:
    """
    Uygulamanın kullanacağı hisse listesini döndürür.

    Öncelik:
    1. Borsa İstanbul resmi CSV kaynağı
    2. Fallback BIST_HISSELER listesi

    Args:
        use_remote: False verilirse direkt fallback liste döner.

    Returns:
        {"THYAO": "TÜRK HAVA YOLLARI", ...}
    """
    if not use_remote:
        return dict(sorted(BIST_HISSELER.items()))

    try:
        remote_hisseler = get_bist_hisseler_from_official()

        if remote_hisseler:
            return remote_hisseler

    except Exception:
        pass

    return dict(sorted(BIST_HISSELER.items()))


# =============================================================================
# BIST ENDEKSLERİ
# =============================================================================

BIST_ENDEKSLER = {
    "^XU100": "BIST 100",
    "^XU030": "BIST 30",
    "^XU050": "BIST 50",
    "^XBANK": "BIST Bankacılık",
    "^XHOLD": "BIST Holding",
    "^XTEKS": "BIST Tekstil",
    "^XGIDA": "BIST Gıda",
    "^XMANA": "BIST Metal Ana",
    "^XKMYA": "BIST Kimya",
    "^XUTEK": "BIST Teknoloji",
}


# =============================================================================
# VARSAYILAN AYARLAR
# =============================================================================

VARSAYILAN_PERIYOT = "1y"
VARSAYILAN_SMA_PERIYOTLAR = [20, 50, 200]
VARSAYILAN_RSI_PERIYOT = 14
VARSAYILAN_MACD = {
    "fast": 12,
    "slow": 26,
    "signal": 9,
}
VARSAYILAN_BOLLINGER = {
    "period": 20,
    "std": 2,
}


# =============================================================================
# HAZIR DÖNEMLER
# =============================================================================

DONEM_SECENEKLERI = {
    "1 Hafta": "5d",
    "1 Ay": "1mo",
    "3 Ay": "3mo",
    "6 Ay": "6mo",
    "1 Yıl": "1y",
    "2 Yıl": "2y",
    "5 Yıl": "5y",
    "10 Yıl": "10y",
    "Tümü": "max",
}


# =============================================================================
# RENK PALETİ
# =============================================================================

RENKLER = {
    "yukselis": "#00d4aa",
    "dusus": "#ff4757",
    "vurgu": "#667eea",
    "vurgu2": "#764ba2",
    "arka_plan": "#0e1117",
    "kart": "#1a1f2e",
    "metin": "#e1e5ee",
    "metin_soluk": "#8892a4",
    "grafik_1": "#667eea",
    "grafik_2": "#f7971e",
    "grafik_3": "#00d4aa",
    "grafik_4": "#ff4757",
    "grafik_5": "#a855f7",
    "grafik_6": "#06b6d4",
    "sma_20": "#f7971e",
    "sma_50": "#667eea",
    "sma_200": "#a855f7",
    "bollinger": "rgba(102, 126, 234, 0.15)",
}

GRAFIK_RENK_LISTESI = [
    "#667eea",
    "#f7971e",
    "#00d4aa",
    "#ff4757",
    "#a855f7",
    "#06b6d4",
    "#f43f5e",
    "#84cc16",
    "#fb923c",
    "#38bdf8",
]