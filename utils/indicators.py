"""
Teknik Gösterge Hesaplamaları
SMA, RSI, MACD, Bollinger Bantları, Getiri hesaplamaları
"""

import pandas as pd
import numpy as np


def sma_hesapla(df, periyot, sutun="Close"):
    """Basit Hareketli Ortalama (SMA) hesaplar."""
    return df[sutun].rolling(window=periyot, min_periods=1).mean()


def ema_hesapla(df, periyot, sutun="Close"):
    """Üstel Hareketli Ortalama (EMA) hesaplar."""
    return df[sutun].ewm(span=periyot, adjust=False).mean()


def rsi_hesapla(df, periyot=14, sutun="Close"):
    """
    Relative Strength Index (RSI) hesaplar.
    
    Args:
        df: Fiyat verileri
        periyot: RSI periyodu (varsayılan: 14)
        sutun: Hesaplanacak sütun
    
    Returns:
        RSI değerleri (0-100 arası)
    """
    delta = df[sutun].diff()
    kazanc = delta.where(delta > 0, 0.0)
    kayip = -delta.where(delta < 0, 0.0)
    
    ort_kazanc = kazanc.ewm(alpha=1/periyot, min_periods=periyot).mean()
    ort_kayip = kayip.ewm(alpha=1/periyot, min_periods=periyot).mean()
    
    rs = ort_kazanc / ort_kayip
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd_hesapla(df, hizli=12, yavas=26, sinyal=9, sutun="Close"):
    """
    MACD (Moving Average Convergence Divergence) hesaplar.
    
    Returns:
        DataFrame: MACD, Sinyal, Histogram sütunları
    """
    ema_hizli = df[sutun].ewm(span=hizli, adjust=False).mean()
    ema_yavas = df[sutun].ewm(span=yavas, adjust=False).mean()
    
    macd_cizgi = ema_hizli - ema_yavas
    sinyal_cizgi = macd_cizgi.ewm(span=sinyal, adjust=False).mean()
    histogram = macd_cizgi - sinyal_cizgi
    
    sonuc = pd.DataFrame({
        "MACD": macd_cizgi,
        "Sinyal": sinyal_cizgi,
        "Histogram": histogram,
    }, index=df.index)
    
    return sonuc


def bollinger_hesapla(df, periyot=20, std_carpan=2.0, sutun="Close"):
    """
    Bollinger Bantlarını hesaplar.
    
    Returns:
        DataFrame: Üst Bant, Orta Bant (SMA), Alt Bant sütunları
    """
    orta = df[sutun].rolling(window=periyot, min_periods=1).mean()
    std = df[sutun].rolling(window=periyot, min_periods=1).std()
    
    sonuc = pd.DataFrame({
        "BB_Ust": orta + (std * std_carpan),
        "BB_Orta": orta,
        "BB_Alt": orta - (std * std_carpan),
    }, index=df.index)
    
    return sonuc


def gunluk_getiri_hesapla(df, sutun="Close"):
    """Günlük yüzde getiriyi hesaplar."""
    return df[sutun].pct_change() * 100


def kumulatif_getiri_hesapla(df, sutun="Close"):
    """Kümülatif getiriyi hesaplar (ilk güne göre %)."""
    return ((df[sutun] / df[sutun].iloc[0]) - 1) * 100


def frekansi_donustur(df, frekans="W"):
    """
    Veri frekansını dönüştürür.
    
    Args:
        df: Günlük fiyat verisi
        frekans: Hedef frekans ("W": Haftalık, "ME": Aylık)
    
    Returns:
        Dönüştürülmüş DataFrame
    """
    agg_dict = {}
    if "Open" in df.columns:
        agg_dict["Open"] = "first"
    if "High" in df.columns:
        agg_dict["High"] = "max"
    if "Low" in df.columns:
        agg_dict["Low"] = "min"
    if "Close" in df.columns:
        agg_dict["Close"] = "last"
    if "Volume" in df.columns:
        agg_dict["Volume"] = "sum"
    
    if not agg_dict:
        return df
    
    return df.resample(frekans).agg(agg_dict).dropna()


def tum_gostergeleri_hesapla(df, sma_periyotlar=None, rsi_periyot=14,
                              macd_params=None, bollinger_params=None):
    """
    Tüm seçilen teknik göstergeleri hesaplar ve DataFrame'e ekler.
    """
    sonuc = df.copy()
    
    # SMA
    if sma_periyotlar:
        for p in sma_periyotlar:
            sonuc[f"SMA_{p}"] = sma_hesapla(df, p)
    
    # RSI
    if rsi_periyot:
        sonuc["RSI"] = rsi_hesapla(df, rsi_periyot)
    
    # MACD
    if macd_params:
        macd_df = macd_hesapla(
            df,
            hizli=macd_params.get("fast", 12),
            yavas=macd_params.get("slow", 26),
            sinyal=macd_params.get("signal", 9),
        )
        sonuc = pd.concat([sonuc, macd_df], axis=1)
    
    # Bollinger
    if bollinger_params:
        bb_df = bollinger_hesapla(
            df,
            periyot=bollinger_params.get("period", 20),
            std_carpan=bollinger_params.get("std", 2.0),
        )
        sonuc = pd.concat([sonuc, bb_df], axis=1)
    
    # Getiriler
    sonuc["Günlük_Getiri"] = gunluk_getiri_hesapla(df)
    sonuc["Kümülatif_Getiri"] = kumulatif_getiri_hesapla(df)
    
    return sonuc
