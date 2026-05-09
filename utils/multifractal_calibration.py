"""
Multifraktal Coklu Hisse Kalibrasyonu

Bu modul, Muzy-Sornette-Delour-Arneodo makalesindeki structure function
mantigini Streamlit projesine uygun ve Excel sablonundaki akisa yakin sekilde
uygular.

Ana fikir:
- eta_i(t, tau) = ln(P_i(t) / P_i(t - tau))
- eta'_i = eta_i - mean(eta_i)
- M_i(q, tau) = E(|eta'_i|^q)
- ln M_i(q, tau) = const + zeta_i(q) ln(tau)
- zeta_i(q) ~= h_i q - 0.5 lambda_i^2 q^2

Coklu hisse icin:
- M_ij(1,1,tau) = E(|eta'_i| |eta'_j|)
- ln M_ij = const + zeta_ij(1,1) ln(tau)
- lambda_ij^2 = zeta_i(1) + zeta_j(1) - zeta_ij(1,1)

Not:
Bu modul kalibrasyon uretir. Negatif/off-diagonal lambda^2 degerlerini otomatik
silmez; gosterir ve uyari icin tabloya koyar. Sonraki portfoy modelinde istersek
kullanim matrisi icin clip uygulanabilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass
class RegressionResult:
    slope: float
    intercept: float
    r2: float
    std_error: float
    n: int


@dataclass
class ZetaFitResult:
    h: float
    lambda_squared: float
    lambda_value: float
    quadratic_a: float
    quadratic_b: float
    quadratic_c: float
    r2: float
    n: int
    fit_mode: str


def _as_clean_prices(prices: pd.DataFrame, symbols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Fiyat verisini numeric, finite ve ortak tarihli hale getirir."""
    if prices is None or prices.empty:
        return pd.DataFrame()

    clean = prices.copy()

    if symbols:
        existing = [s for s in symbols if s in clean.columns]
        clean = clean[existing]

    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(how="all")
    clean = clean.dropna(how="any")
    clean = clean.sort_index()

    return clean


def calculate_multiscale_log_returns(prices: pd.DataFrame, tau: int) -> pd.DataFrame:
    """Verilen zaman olcegi icin continuous/log return hesaplar."""
    tau = int(tau)
    if tau <= 0:
        raise ValueError("tau pozitif tam sayi olmali")

    if prices is None or prices.empty or len(prices) <= tau:
        return pd.DataFrame(index=getattr(prices, "index", None))

    log_returns = np.log(prices / prices.shift(tau))
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan)
    return log_returns.dropna(how="all")


def centered_abs_returns(multiscale_returns: pd.DataFrame) -> pd.DataFrame:
    """eta -> eta' -> |eta'| donusumu."""
    centered = multiscale_returns - multiscale_returns.mean(axis=0, skipna=True)
    abs_centered = centered.abs().replace([np.inf, -np.inf], np.nan)
    return abs_centered


def linear_regression(x: Iterable[float], y: Iterable[float]) -> RegressionResult:
    """Basit OLS: y = intercept + slope*x."""
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    n = int(len(x_arr))
    if n < 2:
        return RegressionResult(np.nan, np.nan, np.nan, np.nan, n)

    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    ss_xx = float(np.sum((x_arr - x_mean) ** 2))

    if ss_xx <= EPS:
        return RegressionResult(np.nan, np.nan, np.nan, np.nan, n)

    slope = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)) / ss_xx)
    intercept = float(y_mean - slope * x_mean)

    y_hat = intercept + slope * x_arr
    residuals = y_arr - y_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_arr - y_mean) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > EPS else np.nan

    if n > 2:
        std_error = float(np.sqrt((ss_res / (n - 2)) / ss_xx))
    else:
        std_error = np.nan

    return RegressionResult(slope, intercept, r2, std_error, n)


def fit_zeta_curve(
    q_values: Iterable[float],
    zeta_values: Iterable[float],
    fit_mode: str = "pdf_constrained",
) -> ZetaFitResult:
    """
    zeta(q) egirisini parabole oturtur.

    fit_mode:
    - "pdf_constrained": zeta(q) = h*q - 0.5*lambda^2*q^2, intercept 0.
    - "excel_free_intercept": zeta(q) = a*q^2 + b*q + c, intercept serbest.
    """
    q = np.asarray(list(q_values), dtype=float)
    zeta = np.asarray(list(zeta_values), dtype=float)
    mask = np.isfinite(q) & np.isfinite(zeta)
    q = q[mask]
    zeta = zeta[mask]
    n = int(len(q))

    if n < 2:
        return ZetaFitResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n, fit_mode)

    if fit_mode == "excel_free_intercept":
        if n < 3:
            return ZetaFitResult(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n, fit_mode)
        a, b, c = np.polyfit(q, zeta, 2)
        zeta_hat = a * q ** 2 + b * q + c
        h = float(b)
        lambda_squared = float(-2 * a)
        quadratic_a = float(a)
        quadratic_b = float(b)
        quadratic_c = float(c)
    else:
        # PDF'e daha yakin olan kisitli fit: intercept yok, katsayilar h ve lambda^2.
        # zeta = [q, -0.5*q^2] @ [h, lambda^2]
        design = np.column_stack([q, -0.5 * q ** 2])
        beta, *_ = np.linalg.lstsq(design, zeta, rcond=None)
        h = float(beta[0])
        lambda_squared = float(beta[1])
        quadratic_a = float(-0.5 * lambda_squared)
        quadratic_b = h
        quadratic_c = 0.0
        zeta_hat = design @ beta

    ss_res = float(np.sum((zeta - zeta_hat) ** 2))
    ss_tot = float(np.sum((zeta - np.mean(zeta)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > EPS else np.nan
    lambda_value = float(np.sqrt(lambda_squared)) if lambda_squared >= 0 else np.nan

    return ZetaFitResult(
        h=h,
        lambda_squared=lambda_squared,
        lambda_value=lambda_value,
        quadratic_a=quadratic_a,
        quadratic_b=quadratic_b,
        quadratic_c=quadratic_c,
        r2=r2,
        n=n,
        fit_mode=fit_mode,
    )


def calculate_univariate_moments(
    prices: pd.DataFrame,
    symbols: Sequence[str],
    time_scales: Sequence[int],
    q_values: Sequence[int],
    min_obs: int = 30,
) -> pd.DataFrame:
    """Her hisse, her tau ve her q icin M_i(q,tau) hesaplar."""
    clean_prices = _as_clean_prices(prices, symbols)
    rows: List[Dict[str, float]] = []

    for tau in sorted({int(t) for t in time_scales if int(t) > 0}):
        returns_tau = calculate_multiscale_log_returns(clean_prices, tau)
        abs_centered = centered_abs_returns(returns_tau)

        for symbol in symbols:
            if symbol not in abs_centered.columns:
                continue
            series = pd.to_numeric(abs_centered[symbol], errors="coerce").dropna()
            series = series[np.isfinite(series)]
            n_obs = int(len(series))

            for q in q_values:
                q_int = int(q)
                if n_obs < min_obs:
                    moment = np.nan
                else:
                    moment = float(np.mean(np.power(series.values, q_int)))
                    if not np.isfinite(moment) or moment <= 0:
                        moment = np.nan

                rows.append({
                    "Sembol": symbol,
                    "Tau": tau,
                    "q": q_int,
                    "N": n_obs,
                    "M(q,tau)": moment,
                    "ln M(q,tau)": float(np.log(moment)) if np.isfinite(moment) and moment > 0 else np.nan,
                    "ln Tau": float(np.log(tau)),
                })

    return pd.DataFrame(rows)


def estimate_univariate_zeta(
    moments_df: pd.DataFrame,
    min_scale_points: int = 3,
) -> pd.DataFrame:
    """ln M(q,tau) ~ ln(tau) regresyonu ile zeta_i(q) hesaplar."""
    if moments_df is None or moments_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, float]] = []

    grouped = moments_df.dropna(subset=["ln M(q,tau)", "ln Tau"]).groupby(["Sembol", "q"])
    for (symbol, q), group in grouped:
        group = group.sort_values("Tau")
        if len(group) < min_scale_points:
            reg = RegressionResult(np.nan, np.nan, np.nan, np.nan, int(len(group)))
        else:
            reg = linear_regression(group["ln Tau"].values, group["ln M(q,tau)"].values)

        rows.append({
            "Sembol": symbol,
            "q": int(q),
            "zeta(q)": reg.slope,
            "Intercept": reg.intercept,
            "R2": reg.r2,
            "Std Error": reg.std_error,
            "Kullanılan Ölçek": reg.n,
        })

    return pd.DataFrame(rows)


def fit_univariate_parameters(
    zeta_df: pd.DataFrame,
    fit_mode: str = "pdf_constrained",
) -> pd.DataFrame:
    """Her hisse icin h, lambda^2 ve lambda parametrelerini fit eder."""
    if zeta_df is None or zeta_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, float]] = []

    for symbol, group in zeta_df.groupby("Sembol"):
        valid = group.dropna(subset=["q", "zeta(q)"]).sort_values("q")
        fit = fit_zeta_curve(valid["q"].values, valid["zeta(q)"].values, fit_mode=fit_mode)

        zeta_1_rows = valid[valid["q"] == 1]
        if not zeta_1_rows.empty:
            zeta_1 = float(zeta_1_rows["zeta(q)"].iloc[0])
            zeta_1_r2 = float(zeta_1_rows["R2"].iloc[0]) if np.isfinite(zeta_1_rows["R2"].iloc[0]) else np.nan
        elif np.isfinite(fit.h) and np.isfinite(fit.lambda_squared):
            zeta_1 = float(fit.h - 0.5 * fit.lambda_squared)
            zeta_1_r2 = np.nan
        else:
            zeta_1 = np.nan
            zeta_1_r2 = np.nan

        avg_r2 = float(valid["R2"].mean()) if "R2" in valid and not valid.empty else np.nan
        h_minus_lambda2 = fit.h - fit.lambda_squared if np.isfinite(fit.h) and np.isfinite(fit.lambda_squared) else np.nan

        rows.append({
            "Sembol": symbol,
            "h": fit.h,
            "lambda^2": fit.lambda_squared,
            "lambda": fit.lambda_value,
            "h - lambda^2": h_minus_lambda2,
            "zeta(1)": zeta_1,
            "zeta(1) R2": zeta_1_r2,
            "Curve Fit R2": fit.r2,
            "Ortalama Zeta R2": avg_r2,
            "Quadratic a": fit.quadratic_a,
            "Quadratic b": fit.quadratic_b,
            "Quadratic c": fit.quadratic_c,
            "Fit Modu": fit.fit_mode,
            "Zeta Nokta Sayısı": fit.n,
        })

    return pd.DataFrame(rows)


def calculate_pairwise_joint_moments(
    prices: pd.DataFrame,
    symbols: Sequence[str],
    time_scales: Sequence[int],
    min_obs: int = 30,
) -> pd.DataFrame:
    """Her hisse cifti icin M_ij(1,1,tau)=E(|eta'_i||eta'_j|) hesaplar."""
    clean_prices = _as_clean_prices(prices, symbols)
    rows: List[Dict[str, float]] = []
    pairs = list(combinations(symbols, 2))

    if not pairs:
        return pd.DataFrame()

    for tau in sorted({int(t) for t in time_scales if int(t) > 0}):
        returns_tau = calculate_multiscale_log_returns(clean_prices, tau)
        abs_centered = centered_abs_returns(returns_tau)

        for left_symbol, right_symbol in pairs:
            if left_symbol not in abs_centered.columns or right_symbol not in abs_centered.columns:
                continue

            pair_df = abs_centered[[left_symbol, right_symbol]].dropna(how="any")
            pair_df = pair_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
            n_obs = int(len(pair_df))

            if n_obs < min_obs:
                joint_moment = np.nan
            else:
                joint_moment = float(np.mean(pair_df[left_symbol].values * pair_df[right_symbol].values))
                if not np.isfinite(joint_moment) or joint_moment <= 0:
                    joint_moment = np.nan

            rows.append({
                "Sembol i": left_symbol,
                "Sembol j": right_symbol,
                "Pair": f"{left_symbol}-{right_symbol}",
                "Tau": tau,
                "N": n_obs,
                "M_ij(1,1,tau)": joint_moment,
                "ln M_ij": float(np.log(joint_moment)) if np.isfinite(joint_moment) and joint_moment > 0 else np.nan,
                "ln Tau": float(np.log(tau)),
            })

    return pd.DataFrame(rows)


def estimate_pairwise_zeta(
    pair_moments_df: pd.DataFrame,
    min_scale_points: int = 3,
) -> pd.DataFrame:
    """ln M_ij(1,1,tau) ~ ln(tau) regresyonu ile zeta_ij(1,1) hesaplar."""
    if pair_moments_df is None or pair_moments_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, float]] = []
    grouped = pair_moments_df.dropna(subset=["ln M_ij", "ln Tau"]).groupby(["Sembol i", "Sembol j", "Pair"])

    for (left_symbol, right_symbol, pair_name), group in grouped:
        group = group.sort_values("Tau")
        if len(group) < min_scale_points:
            reg = RegressionResult(np.nan, np.nan, np.nan, np.nan, int(len(group)))
        else:
            reg = linear_regression(group["ln Tau"].values, group["ln M_ij"].values)

        rows.append({
            "Sembol i": left_symbol,
            "Sembol j": right_symbol,
            "Pair": pair_name,
            "zeta_ij(1,1)": reg.slope,
            "Intercept": reg.intercept,
            "R2": reg.r2,
            "Std Error": reg.std_error,
            "Kullanılan Ölçek": reg.n,
        })

    return pd.DataFrame(rows)


def build_lambda2_matrix(
    symbols: Sequence[str],
    asset_params_df: pd.DataFrame,
    pair_zeta_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    lambda^2 matrisi kurar.

    Diagonal: tek hisse fitinden gelen lambda_i^2.
    Off-diagonal: lambda_ij^2 = zeta_i(1) + zeta_j(1) - zeta_ij(1,1).
    """
    symbols = list(symbols)
    lambda_matrix = pd.DataFrame(np.nan, index=symbols, columns=symbols, dtype=float)
    r2_matrix = pd.DataFrame(np.nan, index=symbols, columns=symbols, dtype=float)

    if asset_params_df is None or asset_params_df.empty:
        return lambda_matrix, r2_matrix, pd.DataFrame()

    params = asset_params_df.set_index("Sembol")
    zeta1: Dict[str, float] = {}

    for symbol in symbols:
        if symbol not in params.index:
            continue
        lambda_value = params.loc[symbol, "lambda^2"] if "lambda^2" in params.columns else np.nan
        zeta_1 = params.loc[symbol, "zeta(1)"] if "zeta(1)" in params.columns else np.nan
        zeta_1_r2 = params.loc[symbol, "zeta(1) R2"] if "zeta(1) R2" in params.columns else np.nan
        lambda_matrix.loc[symbol, symbol] = float(lambda_value) if np.isfinite(lambda_value) else np.nan
        r2_matrix.loc[symbol, symbol] = float(zeta_1_r2) if np.isfinite(zeta_1_r2) else np.nan
        zeta1[symbol] = float(zeta_1) if np.isfinite(zeta_1) else np.nan

    pair_rows: List[Dict[str, float]] = []

    if pair_zeta_df is not None and not pair_zeta_df.empty:
        for _, row in pair_zeta_df.iterrows():
            i = row.get("Sembol i")
            j = row.get("Sembol j")
            if i not in symbols or j not in symbols:
                continue

            zij = float(row.get("zeta_ij(1,1)", np.nan))
            zi = zeta1.get(i, np.nan)
            zj = zeta1.get(j, np.nan)

            if np.isfinite(zi) and np.isfinite(zj) and np.isfinite(zij):
                lambda_ij = zi + zj - zij
            else:
                lambda_ij = np.nan

            lambda_matrix.loc[i, j] = lambda_ij
            lambda_matrix.loc[j, i] = lambda_ij

            pair_r2 = float(row.get("R2", np.nan)) if np.isfinite(row.get("R2", np.nan)) else np.nan
            r2_matrix.loc[i, j] = pair_r2
            r2_matrix.loc[j, i] = pair_r2

            pair_rows.append({
                "Sembol i": i,
                "Sembol j": j,
                "Pair": row.get("Pair", f"{i}-{j}"),
                "zeta_i(1)": zi,
                "zeta_j(1)": zj,
                "zeta_ij(1,1)": zij,
                "lambda^2_ij": lambda_ij,
                "Pair R2": pair_r2,
                "Kullanılan Ölçek": row.get("Kullanılan Ölçek", np.nan),
                "Durum": "Uygun" if np.isfinite(lambda_ij) and lambda_ij >= 0 else "Dikkat",
            })

    pair_lambda_df = pd.DataFrame(pair_rows)
    return lambda_matrix, r2_matrix, pair_lambda_df


def build_clipped_usage_matrix(
    lambda_matrix: pd.DataFrame,
    lower: float = 0.0,
    upper: Optional[float] = None,
) -> pd.DataFrame:
    """Sonraki portfoy modellerinde kullanmak icin opsiyonel clip matrisi."""
    usage = lambda_matrix.copy().astype(float)
    usage = usage.clip(lower=lower)
    if upper is not None:
        usage = usage.clip(upper=upper)
    return usage


def run_multifractal_calibration(
    prices: pd.DataFrame,
    symbols: Sequence[str],
    time_scales: Sequence[int],
    q_values: Sequence[int],
    min_obs: int = 30,
    min_scale_points: int = 3,
    fit_mode: str = "pdf_constrained",
) -> Dict[str, object]:
    """Tum kalibrasyon akisini tek fonksiyonda calistirir."""
    clean_prices = _as_clean_prices(prices, symbols)
    clean_symbols = [symbol for symbol in symbols if symbol in clean_prices.columns]

    if clean_prices.empty or len(clean_symbols) == 0:
        raise ValueError("Kalibrasyon icin gecerli fiyat verisi bulunamadi.")

    valid_time_scales = sorted({int(t) for t in time_scales if int(t) > 0 and int(t) < len(clean_prices)})
    if len(valid_time_scales) < min_scale_points:
        raise ValueError("Zeta regresyonu icin yeterli zaman olcegi yok.")

    valid_q_values = sorted({int(q) for q in q_values if int(q) > 0})
    if len(valid_q_values) < 2:
        raise ValueError("Curve fitting icin en az 2 q degeri gerekiyor.")

    moments_df = calculate_univariate_moments(
        prices=clean_prices,
        symbols=clean_symbols,
        time_scales=valid_time_scales,
        q_values=valid_q_values,
        min_obs=min_obs,
    )

    zeta_df = estimate_univariate_zeta(
        moments_df=moments_df,
        min_scale_points=min_scale_points,
    )

    asset_params_df = fit_univariate_parameters(
        zeta_df=zeta_df,
        fit_mode=fit_mode,
    )

    pair_moments_df = calculate_pairwise_joint_moments(
        prices=clean_prices,
        symbols=clean_symbols,
        time_scales=valid_time_scales,
        min_obs=min_obs,
    )

    pair_zeta_df = estimate_pairwise_zeta(
        pair_moments_df=pair_moments_df,
        min_scale_points=min_scale_points,
    )

    lambda_matrix, lambda_r2_matrix, pair_lambda_df = build_lambda2_matrix(
        symbols=clean_symbols,
        asset_params_df=asset_params_df,
        pair_zeta_df=pair_zeta_df,
    )

    usage_lambda_matrix = build_clipped_usage_matrix(lambda_matrix, lower=0.0)

    warnings: List[str] = []
    if pair_lambda_df is not None and not pair_lambda_df.empty:
        negative_count = int((pair_lambda_df["lambda^2_ij"] < 0).sum())
        low_r2_count = int((pair_lambda_df["Pair R2"] < 0.70).sum())
        if negative_count > 0:
            warnings.append(f"{negative_count} hisse ciftinde lambda^2_ij negatif hesaplandi; otomatik silinmedi.")
        if low_r2_count > 0:
            warnings.append(f"{low_r2_count} hisse ciftinde pair regresyon R2 degeri 0.70 altinda.")

    if asset_params_df is not None and not asset_params_df.empty:
        negative_diag = int((asset_params_df["lambda^2"] < 0).sum())
        low_curve_r2 = int((asset_params_df["Curve Fit R2"] < 0.70).sum())
        if negative_diag > 0:
            warnings.append(f"{negative_diag} hissede diagonal lambda^2 negatif hesaplandi; veri/olcek secimi kontrol edilmeli.")
        if low_curve_r2 > 0:
            warnings.append(f"{low_curve_r2} hissede zeta curve fit R2 degeri 0.70 altinda.")

    settings = {
        "symbols": clean_symbols,
        "time_scales": valid_time_scales,
        "q_values": valid_q_values,
        "min_obs": min_obs,
        "min_scale_points": min_scale_points,
        "fit_mode": fit_mode,
        "price_observations": int(len(clean_prices)),
    }

    methodology_notes = [
        "eta_i(t,tau)=ln(P_i(t)/P_i(t-tau)) continuous/log return kullanildi.",
        "M_i(q,tau)=mean(|eta_i - mean(eta_i)|^q) structure function olarak hesaplandi.",
        "zeta_i(q), ln M_i(q,tau) ile ln(tau) regresyonunun egimi olarak alindi.",
        "Diagonal lambda_i^2, zeta_i(q) parabol fitinden hesaplandi.",
        "Off-diagonal lambda_ij^2, zeta_i(1)+zeta_j(1)-zeta_ij(1,1) formuluyle hesaplandi.",
        "Negatif/duyarsiz degerler otomatik kaldirilmadi; raporda uyari olarak gosterilir.",
    ]

    return {
        "settings": settings,
        "methodology_notes": methodology_notes,
        "prices": clean_prices,
        "moments": moments_df,
        "zeta": zeta_df,
        "asset_params": asset_params_df,
        "pair_moments": pair_moments_df,
        "pair_zeta": pair_zeta_df,
        "lambda_matrix": lambda_matrix,
        "lambda_r2_matrix": lambda_r2_matrix,
        "usage_lambda_matrix": usage_lambda_matrix,
        "pair_lambda": pair_lambda_df,
        "warnings": warnings,
    }
