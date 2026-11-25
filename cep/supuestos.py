"""
Módulo 4. Validación de supuestos.

Los gráficos de Shewhart y los índices de capacidad suponen que, con el proceso
en control, las observaciones son (1) independientes y (2) aproximadamente
normales (para variables). Este módulo verifica:

  * Normalidad: Shapiro-Wilk, Anderson-Darling (con p-valor de D'Agostino y
    Stephens, 1986), Jarque-Bera, asimetría y curtosis, gráfico de probabilidad normal.
  * Independencia: prueba de rachas sobre la mediana (Wald-Wolfowitz),
    autocorrelación muestral con bandas ±1.96/√N, Ljung-Box y Durbin-Watson.
  * Estabilidad preliminar y patrones no aleatorios: se toman de la Fase I de los
    gráficos de control (reglas de Western Electric/Nelson).
  * Transformación de Box-Cox como alternativa cuando no hay normalidad.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

ALFA = 0.05


def _limpiar(x) -> np.ndarray:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    return x[np.isfinite(x)]


# ---------------------------------------------------------------------------
# Normalidad
# ---------------------------------------------------------------------------

def anderson_darling(x) -> tuple[float, float, float]:
    """Estadístico A², A²* ajustado y p-valor para normalidad con μ y σ estimados.

    p-valor por las fórmulas de D'Agostino y Stephens (1986, tabla 4.9), las mismas
    que usan Minitab y el paquete nortest de R.
    """
    x = np.sort(_limpiar(x))
    n = len(x)
    if n < 8:
        return np.nan, np.nan, np.nan
    z = stats.norm.cdf((x - x.mean()) / x.std(ddof=1))
    z = np.clip(z, 1e-300, 1 - 1e-16)
    i = np.arange(1, n + 1)
    a2 = -n - np.sum((2 * i - 1) * (np.log(z) + np.log(1 - z[::-1]))) / n
    a2s = a2 * (1 + 0.75 / n + 2.25 / n ** 2)
    if a2s >= 0.6:
        p = math.exp(1.2937 - 5.709 * a2s + 0.0186 * a2s ** 2)
    elif a2s >= 0.34:
        p = math.exp(0.9177 - 4.279 * a2s - 1.38 * a2s ** 2)
    elif a2s >= 0.2:
        p = 1 - math.exp(-8.318 + 42.796 * a2s - 59.938 * a2s ** 2)
    else:
        p = 1 - math.exp(-13.436 + 101.14 * a2s - 223.73 * a2s ** 2)
    return float(a2), float(a2s), float(min(max(p, 0.0), 1.0))


def pruebas_normalidad(x, alfa: float = ALFA) -> dict:
    x = _limpiar(x)
    n = len(x)
    filas = []
    if 3 <= n <= 5000:
        w, p = stats.shapiro(x)
        filas.append(("Shapiro-Wilk", "W", w, p))
    a2, a2s, p_ad = anderson_darling(x)
    if np.isfinite(a2):
        filas.append(("Anderson-Darling", "A² (A²* ajustado = %.4f)" % a2s, a2, p_ad))
    if n >= 8:
        jb = stats.jarque_bera(x)
        filas.append(("Jarque-Bera", "JB", float(jb.statistic), float(jb.pvalue)))
    tabla = pd.DataFrame(filas, columns=["Prueba", "Estadístico", "Valor", "p-valor"])
    tabla["Decisión (α = %.2f)" % alfa] = np.where(tabla["p-valor"] < alfa, "Se rechaza normalidad",
                                                    "No se rechaza normalidad")
    rechazos = int((tabla["p-valor"] < alfa).sum()) if len(tabla) else 0
    asim = float(stats.skew(x, bias=False)) if n > 2 else np.nan
    curt = float(stats.kurtosis(x, bias=False)) if n > 3 else np.nan
    distintos = int(len(np.unique(x)))
    # Muchos empates (pocos valores distintos) indican resolución gruesa del instrumento: Shapiro-Wilk y
    # Anderson-Darling rechazan con facilidad aunque la forma de la distribución sea normal.
    empates = n >= 20 and distintos < max(10, 0.3 * n)
    principal = tabla.iloc[0] if len(tabla) else None
    if principal is None:
        conclusion = "Datos insuficientes para probar normalidad (se requieren al menos 8)."
        normal = None
    elif principal["p-valor"] >= alfa:
        conclusion = (f"No hay evidencia para rechazar la normalidad (Shapiro-Wilk p = {principal['p-valor']:.4f} ≥ {alfa}). "
                      "Es razonable usar límites 3σ e índices de capacidad basados en la normal.")
        normal = True
    else:
        conclusion = (f"Se rechaza la normalidad (Shapiro-Wilk p = {principal['p-valor']:.4f} < {alfa}). Revise si hay "
                      "causas especiales o mezcla de poblaciones; las cartas X̄ son robustas por el teorema "
                      "central del límite, pero para capacidad considere la transformación de Box-Cox.")
        normal = False
    nota_resolucion = ""
    if empates:
        nota_resolucion = (f"Los {n} datos solo toman {distintos} valores distintos: la resolución del instrumento genera "
                           "empates, y con ellos Shapiro-Wilk y Anderson-Darling tienden a rechazar la normalidad aunque la "
                           "forma sea normal. Revise la asimetría, la curtosis, Jarque-Bera y el gráfico de probabilidad; "
                           "si la forma es simétrica, el rechazo se atribuye a la resolución de la medición.")
        if normal is False:
            conclusion += " " + nota_resolucion
    return {"tabla": tabla, "n": n, "asimetria": asim, "curtosis": curt, "normal": normal,
            "rechazos": rechazos, "conclusion": conclusion, "valores_distintos": distintos,
            "empates": empates, "nota_resolucion": nota_resolucion}


def datos_qq(x) -> dict:
    """Puntos del gráfico de probabilidad normal (posiciones de Blom) y recta de referencia."""
    x = np.sort(_limpiar(x))
    n = len(x)
    p = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    teoricos = stats.norm.ppf(p)
    pendiente, intercepto = x.std(ddof=1), x.mean()
    return {"teoricos": teoricos, "observados": x, "pendiente": pendiente, "intercepto": intercepto}


# ---------------------------------------------------------------------------
# Independencia
# ---------------------------------------------------------------------------

def prueba_rachas(x) -> dict:
    """Prueba de rachas por encima/debajo de la mediana (Wald-Wolfowitz), aproximación normal."""
    x = _limpiar(x)
    med = np.median(x)
    signos = np.sign(x - med)
    signos = signos[signos != 0]
    n1, n2 = int(np.sum(signos > 0)), int(np.sum(signos < 0))
    if n1 == 0 or n2 == 0:
        return {"rachas": np.nan, "esperadas": np.nan, "z": np.nan, "p": np.nan, "n1": n1, "n2": n2}
    rachas = 1 + int(np.sum(signos[1:] != signos[:-1]))
    N = n1 + n2
    media = 2 * n1 * n2 / N + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - N) / (N ** 2 * (N - 1))
    z = (rachas - media) / math.sqrt(var) if var > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return {"rachas": rachas, "esperadas": media, "z": z, "p": p, "n1": n1, "n2": n2}


def autocorrelacion(x, max_rezago: int | None = None) -> pd.DataFrame:
    x = _limpiar(x)
    n = len(x)
    if max_rezago is None:
        max_rezago = int(min(20, max(1, n // 4)))
    xc = x - x.mean()
    den = np.sum(xc ** 2)
    filas = []
    for k in range(1, max_rezago + 1):
        r = float(np.sum(xc[:-k] * xc[k:]) / den) if den > 0 else np.nan
        filas.append({"Rezago": k, "r_k": r})
    tabla = pd.DataFrame(filas)
    tabla["Banda ±"] = 1.96 / math.sqrt(n)
    tabla["Significativa"] = np.where(tabla["r_k"].abs() > tabla["Banda ±"], "Sí", "")
    return tabla


def ljung_box(x, rezagos: int | None = None) -> dict:
    x = _limpiar(x)
    n = len(x)
    acf = autocorrelacion(x, rezagos)
    h = len(acf)
    q = n * (n + 2) * float(np.sum(acf["r_k"].to_numpy() ** 2 / (n - acf["Rezago"].to_numpy())))
    p = float(stats.chi2.sf(q, h))
    return {"Q": q, "gl": h, "p": p}


def durbin_watson(x) -> float:
    x = _limpiar(x)
    e = x - x.mean()
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))


def pruebas_independencia(x, alfa: float = ALFA) -> dict:
    x = _limpiar(x)
    rachas = prueba_rachas(x)
    acf = autocorrelacion(x)
    lb = ljung_box(x)
    dw = durbin_watson(x)
    tabla = pd.DataFrame([
        ("Rachas sobre la mediana (Wald-Wolfowitz)", f"{rachas['rachas']} rachas (esperadas {rachas['esperadas']:.1f})",
         rachas["z"], rachas["p"]),
        (f"Ljung-Box Q({lb['gl']})", f"Q = {lb['Q']:.3f}", lb["Q"], lb["p"]),
        ("Autocorrelación de rezago 1", f"r₁ = {acf['r_k'].iloc[0]:.3f} (banda ±{acf['Banda ±'].iloc[0]:.3f})",
         acf["r_k"].iloc[0], np.nan),
        ("Durbin-Watson", "d ≈ 2(1 − r₁); cerca de 2 = sin autocorrelación", dw, np.nan),
    ], columns=["Prueba", "Detalle", "Estadístico", "p-valor"])
    problemas = []
    if np.isfinite(rachas["p"]) and rachas["p"] < alfa:
        problemas.append("la prueba de rachas indica un orden no aleatorio "
                         f"({'pocas' if rachas['rachas'] < rachas['esperadas'] else 'demasiadas'} rachas)")
    if lb["p"] < alfa:
        problemas.append(f"Ljung-Box detecta autocorrelación conjunta (p = {lb['p']:.4f})")
    # Solo el rezago 1 entra en la decisión: revisar los 20 rezagos uno por uno daría falsas alarmas
    # por comparaciones múltiples (en promedio 1 de cada 20 sale significativo por azar).
    r1, banda = float(acf["r_k"].iloc[0]), float(acf["Banda ±"].iloc[0])
    if abs(r1) > banda:
        problemas.append(f"la autocorrelación de rezago 1 es significativa (r₁ = {r1:.3f}, banda ±{banda:.3f})")
    if problemas:
        conclusion = ("Hay evidencia de dependencia entre observaciones: " + "; ".join(problemas) +
                      ". Con autocorrelación positiva las cartas de Shewhart dan falsas alarmas; considere ampliar "
                      "el intervalo de muestreo, modelar la serie o usar cartas de residuales.")
        independiente = False
    else:
        conclusion = ("No hay evidencia de dependencia: las observaciones se comportan como independientes, "
                      "lo que respalda el uso de límites de Shewhart.")
        independiente = True
    return {"tabla": tabla, "acf": acf, "rachas": rachas, "ljung_box": lb, "dw": dw,
            "independiente": independiente, "conclusion": conclusion}


# ---------------------------------------------------------------------------
# Transformación de Box-Cox
# ---------------------------------------------------------------------------

def box_cox(x) -> dict:
    """Estima λ por máxima verosimilitud. Requiere datos positivos."""
    import warnings

    x = _limpiar(x)
    if len(x) < 8:
        return {"posible": False, "motivo": "Se necesitan al menos 8 datos para estimar λ."}
    if np.any(x <= 0):
        return {"posible": False, "motivo": "Box-Cox requiere que todos los datos sean positivos."}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xt, lam = stats.boxcox(x)
        if abs(lam) > 5:
            cv = 100 * x.std(ddof=1) / x.mean()
            return {"posible": False, "lambda": float(lam),
                    "motivo": (f"El λ de máxima verosimilitud es extremo (λ ≈ {lam:.0f}) porque la variación es muy "
                               f"pequeña frente al nivel de la variable (CV = {cv:.2f} %). En ese rango la "
                               "transformación es prácticamente lineal y no corrige la falta de normalidad: revise "
                               "si hay causas especiales, mezcla de poblaciones o poca resolución del instrumento.")}
        try:
            _, _, ic = stats.boxcox(x, alpha=0.05)
            ic = (float(ic[0]), float(ic[1]))
        except (RuntimeError, ValueError):
            ic = (np.nan, np.nan)
    redondeado = min([-2, -1, -0.5, 0, 0.5, 1, 2], key=lambda r: abs(r - lam))
    return {"posible": True, "lambda": float(lam), "lambda_redondeado": float(redondeado),
            "ic95": ic, "incluye_1": bool(np.isfinite(ic[0]) and ic[0] <= 1 <= ic[1]),
            "transformados": xt}


def transformar(x, lam: float):
    x = np.asarray(x, dtype=float)
    return np.log(x) if abs(lam) < 1e-12 else (x ** lam - 1) / lam
