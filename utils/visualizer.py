"""
Plotly ile Grafik Oluşturma Modülü
Mum grafik, çizgi grafik, RSI, MACD, karşılaştırma, korelasyon vb.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.constants import RENKLER, GRAFIK_RENK_LISTESI


def _grafik_tema_uygula(fig, baslik="", yukseklik=500):
    """Tüm grafiklere ortak koyu tema uygular."""
    fig.update_layout(
        title=dict(text=baslik, font=dict(size=18, color=RENKLER["metin"]), x=0.5),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=RENKLER["metin"], family="sans-serif"),
        height=yukseklik,
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(
            bgcolor="rgba(26, 31, 46, 0.8)",
            bordercolor="rgba(102, 126, 234, 0.3)",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor="rgba(136, 146, 164, 0.15)",
            zerolinecolor="rgba(136, 146, 164, 0.15)",
        ),
        yaxis=dict(
            gridcolor="rgba(136, 146, 164, 0.15)",
            zerolinecolor="rgba(136, 146, 164, 0.15)",
        ),
        hovermode="x unified",
    )
    return fig


def mum_grafik_olustur(df, baslik="Fiyat Grafiği", sma_sutunlar=None, bollinger=False):
    """
    Candlestick (mum) grafik oluşturur.
    İsteğe bağlı SMA ve Bollinger overlay'leri ekler.
    """
    fig = go.Figure()

    # Mum grafik
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Fiyat",
        increasing_line_color=RENKLER["yukselis"],
        decreasing_line_color=RENKLER["dusus"],
        increasing_fillcolor=RENKLER["yukselis"],
        decreasing_fillcolor=RENKLER["dusus"],
    ))

    # SMA çizgileri
    if sma_sutunlar:
        sma_renkleri = [RENKLER["sma_20"], RENKLER["sma_50"], RENKLER["sma_200"]]
        for i, sutun in enumerate(sma_sutunlar):
            if sutun in df.columns:
                renk = sma_renkleri[i % len(sma_renkleri)]
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[sutun],
                    mode="lines",
                    name=sutun,
                    line=dict(color=renk, width=1.5),
                ))

    # Bollinger Bantları
    if bollinger and "BB_Ust" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Ust"],
            mode="lines", name="Bollinger Üst",
            line=dict(color=RENKLER["vurgu"], width=1, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Alt"],
            mode="lines", name="Bollinger Alt",
            line=dict(color=RENKLER["vurgu"], width=1, dash="dot"),
            fill="tonexty",
            fillcolor=RENKLER["bollinger"],
        ))

    fig.update_layout(xaxis_rangeslider_visible=False)
    return _grafik_tema_uygula(fig, baslik, 500)


def cizgi_grafik_olustur(df, sutunlar, baslik="Çizgi Grafik", y_etiketi="Değer"):
    """Çoklu sütun çizgi grafik oluşturur."""
    fig = go.Figure()

    for i, sutun in enumerate(sutunlar):
        if sutun in df.columns:
            renk = GRAFIK_RENK_LISTESI[i % len(GRAFIK_RENK_LISTESI)]
            fig.add_trace(go.Scatter(
                x=df.index, y=df[sutun],
                mode="lines",
                name=sutun,
                line=dict(color=renk, width=2),
            ))

    fig.update_yaxes(title_text=y_etiketi)
    return _grafik_tema_uygula(fig, baslik, 400)


def hacim_grafik_olustur(df, baslik="İşlem Hacmi"):
    """Hacim bar grafiği oluşturur. Yeşil/kırmızı renkli."""
    renkler = [
        RENKLER["yukselis"] if row["Close"] >= row["Open"] else RENKLER["dusus"]
        for _, row in df.iterrows()
    ]
    
    fig = go.Figure(go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=renkler,
        name="Hacim",
        opacity=0.7,
    ))

    fig.update_yaxes(title_text="Hacim")
    return _grafik_tema_uygula(fig, baslik, 250)


def rsi_grafik_olustur(df, baslik="RSI Göstergesi"):
    """RSI grafiği oluşturur. 30/70 aşırı alım-satım çizgileri ile."""
    if "RSI" not in df.columns:
        return go.Figure()

    fig = go.Figure()

    # RSI çizgisi
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        mode="lines",
        name="RSI",
        line=dict(color=RENKLER["vurgu"], width=2),
    ))

    # Aşırı alım/satım bölgeleri
    fig.add_hline(y=70, line_dash="dash", line_color=RENKLER["dusus"],
                  annotation_text="Aşırı Alım (70)", annotation_position="right")
    fig.add_hline(y=30, line_dash="dash", line_color=RENKLER["yukselis"],
                  annotation_text="Aşırı Satım (30)", annotation_position="right")
    fig.add_hline(y=50, line_dash="dot", line_color=RENKLER["metin_soluk"], opacity=0.5)

    # Aşırı alım/satım fill alanları
    fig.add_hrect(y0=70, y1=100, fillcolor=RENKLER["dusus"], opacity=0.05)
    fig.add_hrect(y0=0, y1=30, fillcolor=RENKLER["yukselis"], opacity=0.05)

    fig.update_yaxes(range=[0, 100], title_text="RSI")
    return _grafik_tema_uygula(fig, baslik, 300)


def macd_grafik_olustur(df, baslik="MACD Göstergesi"):
    """MACD grafiği oluşturur — çizgiler + histogram."""
    if "MACD" not in df.columns:
        return go.Figure()

    fig = go.Figure()

    # MACD histogram
    renkler = [
        RENKLER["yukselis"] if v >= 0 else RENKLER["dusus"]
        for v in df["Histogram"]
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Histogram"],
        name="Histogram",
        marker_color=renkler,
        opacity=0.6,
    ))

    # MACD çizgisi
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        mode="lines", name="MACD",
        line=dict(color=RENKLER["vurgu"], width=2),
    ))

    # Sinyal çizgisi
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Sinyal"],
        mode="lines", name="Sinyal",
        line=dict(color=RENKLER["grafik_2"], width=2),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color=RENKLER["metin_soluk"], opacity=0.5)
    fig.update_yaxes(title_text="MACD")
    return _grafik_tema_uygula(fig, baslik, 300)


def karsilastirma_grafik_olustur(veriler, baslik="Hisse Karşılaştırma (Normalize)"):
    """
    Birden fazla hisseyi normalize ederek karşılaştırma grafiği oluşturur.
    İlk gün = 100 baz alınır.
    """
    fig = go.Figure()

    for i, (sembol, df) in enumerate(veriler.items()):
        if "Close" in df.columns and len(df) > 0:
            normalize = (df["Close"] / df["Close"].iloc[0]) * 100
            renk = GRAFIK_RENK_LISTESI[i % len(GRAFIK_RENK_LISTESI)]
            fig.add_trace(go.Scatter(
                x=df.index, y=normalize,
                mode="lines", name=sembol,
                line=dict(color=renk, width=2),
            ))

    fig.add_hline(y=100, line_dash="dot", line_color=RENKLER["metin_soluk"], opacity=0.5)
    fig.update_yaxes(title_text="Normalize Fiyat (Baz=100)")
    return _grafik_tema_uygula(fig, baslik, 450)


def korelasyon_heatmap_olustur(veriler, baslik="Hisseler Arası Korelasyon"):
    """Seçilen hisseler arası kapanış fiyatı korelasyon heatmap'i."""
    if len(veriler) < 2:
        fig = go.Figure()
        fig.add_annotation(text="En az 2 hisse seçiniz", showarrow=False, font=dict(size=16))
        return _grafik_tema_uygula(fig, baslik, 400)

    # Kapanış fiyatlarını birleştir
    kapanis_df = pd.DataFrame()
    for sembol, df in veriler.items():
        if "Close" in df.columns:
            kapanis_df[sembol] = df["Close"]

    # Korelasyon matrisi
    kor = kapanis_df.corr()

    fig = go.Figure(go.Heatmap(
        z=kor.values,
        x=kor.columns,
        y=kor.index,
        colorscale=[
            [0, RENKLER["dusus"]],
            [0.5, RENKLER["kart"]],
            [1, RENKLER["yukselis"]],
        ],
        zmin=-1, zmax=1,
        text=np.round(kor.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=14, color=RENKLER["metin"]),
        hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Korelasyon: %{z:.3f}<extra></extra>",
    ))

    return _grafik_tema_uygula(fig, baslik, 450)


def getiri_dagilim_grafik_olustur(df, sembol="", baslik=""):
    """Günlük getiri dağılımı histogramı oluşturur."""
    if "Günlük_Getiri" not in df.columns:
        return go.Figure()

    baslik = baslik or f"{sembol} Günlük Getiri Dağılımı"
    getiri = df["Günlük_Getiri"].dropna()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=getiri,
        nbinsx=50,
        name="Getiri Dağılımı",
        marker_color=RENKLER["vurgu"],
        opacity=0.75,
    ))

    # Ortalama çizgisi
    ort = getiri.mean()
    fig.add_vline(x=ort, line_dash="dash", line_color=RENKLER["grafik_2"],
                  annotation_text=f"Ortalama: {ort:.2f}%")

    fig.update_xaxes(title_text="Günlük Getiri (%)")
    fig.update_yaxes(title_text="Frekans")
    return _grafik_tema_uygula(fig, baslik, 350)


def fiyat_ve_hacim_grafik_olustur(df, baslik="Fiyat & Hacim", sma_sutunlar=None, bollinger=False):
    """Fiyat (mum) ve hacim grafiğini alt alta oluşturur."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Mum grafik
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Fiyat",
        increasing_line_color=RENKLER["yukselis"],
        decreasing_line_color=RENKLER["dusus"],
        increasing_fillcolor=RENKLER["yukselis"],
        decreasing_fillcolor=RENKLER["dusus"],
    ), row=1, col=1)

    # SMA
    if sma_sutunlar:
        sma_renkleri = [RENKLER["sma_20"], RENKLER["sma_50"], RENKLER["sma_200"]]
        for i, sutun in enumerate(sma_sutunlar):
            if sutun in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[sutun],
                    mode="lines", name=sutun,
                    line=dict(color=sma_renkleri[i % len(sma_renkleri)], width=1.5),
                ), row=1, col=1)

    # Bollinger
    if bollinger and "BB_Ust" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Ust"],
            mode="lines", name="BB Üst",
            line=dict(color=RENKLER["vurgu"], width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Alt"],
            mode="lines", name="BB Alt",
            line=dict(color=RENKLER["vurgu"], width=1, dash="dot"),
            fill="tonexty", fillcolor=RENKLER["bollinger"],
        ), row=1, col=1)

    # Hacim
    renkler = [
        RENKLER["yukselis"] if row["Close"] >= row["Open"] else RENKLER["dusus"]
        for _, row in df.iterrows()
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Hacim", marker_color=renkler, opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Fiyat (₺)", row=1, col=1)
    fig.update_yaxes(title_text="Hacim", row=2, col=1)
    return _grafik_tema_uygula(fig, baslik, 600)
