"""
Módulo 6. Diseño de gráficos de control: ARL0, ARL1 y ATS.

Definiciones (Montgomery, 2020, sección 6.2.6):
    ARL  = 1/P(señal)                 longitud promedio de corrida (en muestras)
    ARL0 = 1/α                        con el proceso en control (falsas alarmas)
    ARL1 = 1/(1 − β)                  con el proceso fuera de control (detección)
    ATS  = ARL · h                    tiempo promedio hasta la señal (h = horas entre muestras)
    Carga muestral r = n/h            unidades inspeccionadas por hora

Cálculo analítico:
  * X̄ (Shewhart, L sigmas), corrimiento de la media de δ desviaciones estándar del proceso:
        β = Φ(L − δ√n) − Φ(−L − δ√n)
  * S, cambio de σ0 a λσ0:   (n−1)S²/σ1² ~ χ²(n−1)
  * R, cambio de σ0 a λσ0:   distribución del rango relativo W = R/σ
        F_W(w) = n ∫ φ(x) [Φ(x+w) − Φ(x)]^(n−1) dx
  * p y np (binomial exacta), c y u (Poisson exacta)

Simulación Monte Carlo (cuando no hay fórmula cerrada): Shewhart con reglas
suplementarias, EWMA y CUSUM. Se usa semilla fija para que el resultado sea reproducible.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import integrate, stats

from . import constantes as K

# ---------------------------------------------------------------------------
# Analítico: carta X̄ / I
# ---------------------------------------------------------------------------


def beta_xbar(n: int, delta: float, L: float = 3.0) -> float:
    """Probabilidad de NO detectar en una muestra un corrimiento de δσ (riesgo β)."""
    r = delta * math.sqrt(n)
    return float(stats.norm.cdf(L - r) - stats.norm.cdf(-L - r))


def arl_xbar(n: int, delta: float, L: float = 3.0) -> float:
    return 1.0 / (1.0 - beta_xbar(n, delta, L))


def tabla_arl_xbar(n_lista, delta_lista, L: float = 3.0, h: float = 1.0) -> pd.DataFrame:
    filas = []
    for n in n_lista:
        for d in delta_lista:
            b = beta_xbar(n, d, L)
            arl = 1 / (1 - b)
            filas.append({"n": n, "δ (en σ)": d, "β": b, "Potencia 1−β": 1 - b, "ARL": arl, "ATS (h)": arl * h})
    return pd.DataFrame(filas)


def tabla_diseno(n_lista, h_lista, delta: float, L: float = 3.0) -> pd.DataFrame:
    """Compara diseños (n, h): ARL0, ATS0, ARL1, ATS1 y carga muestral n/h."""
    filas = []
    arl0 = arl_xbar(1, 0.0, L)
    for n in n_lista:
        arl1 = arl_xbar(n, delta, L)
        for h in h_lista:
            filas.append({"n": n, "h (horas)": h, "Carga muestral n/h (unid./hora)": n / h,
                          "ARL0": arl0, "ATS0 (horas)": arl0 * h, "ARL1": arl1, "ATS1 (horas)": arl1 * h,
                          "Unidades inspeccionadas hasta detectar": arl1 * n})
    return pd.DataFrame(filas)


def mejor_diseno(n_lista, h_lista, delta: float, L: float = 3.0, carga_maxima: float | None = None,
                 ats0_minimo: float | None = None) -> pd.DataFrame:
    """Diseños factibles ordenados por ATS1 (menor = detecta más rápido)."""
    t = tabla_diseno(n_lista, h_lista, delta, L)
    if carga_maxima is not None:
        t = t[t["Carga muestral n/h (unid./hora)"] <= carga_maxima + 1e-9]
    if ats0_minimo is not None:
        t = t[t["ATS0 (horas)"] >= ats0_minimo - 1e-9]
    return t.sort_values(["ATS1 (horas)", "Carga muestral n/h (unid./hora)"]).reset_index(drop=True)


def n_para_arl1(delta: float, arl1_objetivo: float, L: float = 3.0, n_max: int = 100) -> int | None:
    """Menor n que logra ARL1 <= objetivo para un corrimiento δ."""
    for n in range(1, n_max + 1):
        if arl_xbar(n, delta, L) <= arl1_objetivo:
            return n
    return None


def L_para_arl0(arl0_objetivo: float) -> float:
    """Ancho L de los límites para un ARL0 deseado (carta X̄ bilateral): α = 1/ARL0."""
    alfa = 1.0 / arl0_objetivo
    return float(stats.norm.isf(alfa / 2))


# ---------------------------------------------------------------------------
# Analítico: cartas de dispersión (S y R) ante un aumento de σ
# ---------------------------------------------------------------------------

def arl_s(n: int, lam: float, L: float = 3.0) -> float:
    """ARL de la carta S cuando σ pasa de σ0 a λ·σ0 (límites c4σ0 ± Lσ0√(1−c4²))."""
    c4 = K.c4(n)
    ucl = c4 + L * math.sqrt(1 - c4 ** 2)
    lcl = max(0.0, c4 - L * math.sqrt(1 - c4 ** 2))
    nu = n - 1
    p = stats.chi2.sf(nu * (ucl / lam) ** 2, nu) + stats.chi2.cdf(nu * (lcl / lam) ** 2, nu)
    return float(1.0 / p)


@lru_cache(maxsize=4096)
def _cdf_rango(w: float, n: int) -> float:
    if w <= 0:
        return 0.0
    phi, Phi = stats.norm.pdf, stats.norm.cdf
    f = lambda x: phi(x) * (Phi(x + w) - Phi(x)) ** (n - 1)
    v, _ = integrate.quad(f, -9, 9, limit=200)
    return min(1.0, n * v)


def arl_r(n: int, lam: float, L: float = 3.0) -> float:
    """ARL de la carta R cuando σ pasa de σ0 a λ·σ0 (límites (d2 ± L·d3)σ0)."""
    d2, d3 = K.d2(n), K.d3(n)
    ucl, lcl = d2 + L * d3, max(0.0, d2 - L * d3)
    p = (1.0 - _cdf_rango(round(ucl / lam, 10), n)) + _cdf_rango(round(lcl / lam, 10), n)
    return float(1.0 / p)


# ---------------------------------------------------------------------------
# Analítico: cartas por atributos (distribuciones exactas)
# ---------------------------------------------------------------------------

def arl_p(n: int, p0: float, p1: float, L: float = 3.0) -> dict:
    """ARL exacto (binomial) de la carta p/np con límites calculados con p0."""
    s = math.sqrt(p0 * (1 - p0) / n)
    ucl, lcl = p0 + L * s, max(0.0, p0 - L * s)
    d_sup = math.floor(n * ucl + 1e-9)          # se acepta D <= d_sup
    d_inf = math.ceil(n * lcl - 1e-9)           # se acepta D >= d_inf
    acepta = stats.binom.cdf(d_sup, n, p1) - (stats.binom.cdf(d_inf - 1, n, p1) if d_inf > 0 else 0.0)
    p_senal = 1 - acepta
    return {"LCS": ucl, "LCI": lcl, "P(señal)": p_senal, "ARL": 1 / p_senal if p_senal > 0 else np.inf,
            "Región de aceptación (defectuosos)": f"{d_inf} a {d_sup}"}


def arl_c(c0: float, c1: float, L: float = 3.0) -> dict:
    """ARL exacto (Poisson) de la carta c con límites calculados con c0 (para u, use c = n·u)."""
    ucl, lcl = c0 + L * math.sqrt(c0), max(0.0, c0 - L * math.sqrt(c0))
    d_sup = math.floor(ucl + 1e-9)
    d_inf = math.ceil(lcl - 1e-9)
    acepta = stats.poisson.cdf(d_sup, c1) - (stats.poisson.cdf(d_inf - 1, c1) if d_inf > 0 else 0.0)
    p_senal = 1 - acepta
    return {"LCS": ucl, "LCI": lcl, "P(señal)": p_senal, "ARL": 1 / p_senal if p_senal > 0 else np.inf,
            "Región de aceptación (defectos)": f"{d_inf} a {d_sup}"}


# ---------------------------------------------------------------------------
# Simulación Monte Carlo (vectorizada)
# ---------------------------------------------------------------------------

def simular_arl(tipo: str, delta: float = 0.0, n: int = 1, L: float = 3.0, reglas=(1,), lam: float = 0.2,
                k: float = 0.5, h: float = 5.0, repeticiones: int = 5000, max_longitud: int = 20000,
                semilla: int = 2026) -> dict:
    """Simula longitudes de corrida y devuelve ARL, error estándar y percentiles.

    tipo: 'shewhart' (X̄ o I con las reglas indicadas), 'ewma' (límites exactos) o 'cusum' (bilateral).
    El estadístico se simula estandarizado: x̄ ~ N(δ√n, 1) en unidades de σ_x̄.
    """
    rng = np.random.default_rng(semilla)
    R = int(repeticiones)
    desplaz = delta * math.sqrt(n)
    rl = np.zeros(R, dtype=np.int64)
    activo = np.ones(R, dtype=bool)
    hist = np.zeros((R, 5))        # últimos 5 valores (reglas 2 y 3)
    racha_lado = np.zeros(R)       # racha con signo (regla 4)
    tend = np.zeros(R)             # racha de incrementos con signo (regla 5)
    prev = np.zeros(R)
    z = np.zeros(R)
    cp = np.zeros(R)
    cm = np.zeros(R)
    reglas = set(reglas)
    for t in range(1, max_longitud + 1):
        idx = np.where(activo)[0]
        if idx.size == 0:
            break
        x = rng.standard_normal(idx.size) + desplaz
        if tipo == "shewhart":
            senal = np.abs(x) > L if 1 in reglas else np.zeros(idx.size, dtype=bool)
            hist[idx] = np.column_stack([hist[idx, 1:], x])
            if 2 in reglas and t >= 2:
                h3 = hist[idx, 2:]
                senal |= ((x > 2) & (np.sum(h3 > 2, axis=1) >= 2)) | ((x < -2) & (np.sum(h3 < -2, axis=1) >= 2))
            if 3 in reglas and t >= 4:
                h5 = hist[idx]
                senal |= ((x > 1) & (np.sum(h5 > 1, axis=1) >= 4)) | ((x < -1) & (np.sum(h5 < -1, axis=1) >= 4))
            if 4 in reglas:
                rs = racha_lado[idx]
                rs = np.where(x > 0, np.where(rs > 0, rs + 1, 1), np.where(rs < 0, rs - 1, -1))
                racha_lado[idx] = rs
                senal |= np.abs(rs) >= 8
            if 5 in reglas and t >= 2:
                d = x - prev[idx]
                tr = tend[idx]
                tr = np.where(d > 0, np.where(tr > 0, tr + 1, 1), np.where(tr < 0, tr - 1, -1))
                tend[idx] = tr
                senal |= np.abs(tr) >= 5
            prev[idx] = x
        elif tipo == "ewma":
            z[idx] = lam * x + (1 - lam) * z[idx]
            lim = L * math.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
            senal = np.abs(z[idx]) > lim
        elif tipo == "cusum":
            cp[idx] = np.maximum(0, x - k + cp[idx])
            cm[idx] = np.maximum(0, -k - x + cm[idx])
            senal = (cp[idx] > h) | (cm[idx] > h)
        else:
            raise ValueError(f"Tipo de simulación desconocido: {tipo}")
        rl[idx[senal]] = t
        activo[idx[senal]] = False
    censuradas = int(activo.sum())
    rl[activo] = max_longitud
    return {"ARL": float(rl.mean()), "EE": float(rl.std(ddof=1) / math.sqrt(R)),
            "SDRL": float(rl.std(ddof=1)), "Mediana": float(np.median(rl)),
            "P5": float(np.percentile(rl, 5)), "P95": float(np.percentile(rl, 95)),
            "censuradas": censuradas, "longitudes": rl}


def perfil_arl(tipos: dict, deltas, n: int = 1, repeticiones: int = 3000, semilla: int = 2026) -> pd.DataFrame:
    """ARL simulado de varios esquemas para una lista de corrimientos (comparación de cartas)."""
    filas = []
    for nombre, cfg in tipos.items():
        for d in deltas:
            r = simular_arl(delta=d, n=n, repeticiones=repeticiones, semilla=semilla, **cfg)
            filas.append({"Esquema": nombre, "δ (en σ)": d, "ARL simulado": r["ARL"], "EE": r["EE"]})
    return pd.DataFrame(filas)


def interpretar_diseno(n: int, h: float, L: float, delta: float) -> list[str]:
    arl0 = arl_xbar(n, 0.0, L)
    arl1 = arl_xbar(n, delta, L)
    b = beta_xbar(n, delta, L)
    return [
        f"Con límites a {L:g}σ, el ARL0 es {arl0:.1f}: en promedio habrá una falsa alarma cada {arl0:.0f} muestras, "
        f"es decir, cada {arl0 * h:.0f} horas de operación si se muestrea cada {h:g} h (ATS0).",
        f"Ante un corrimiento de la media de {delta:g}σ, con n = {n} la probabilidad de detectarlo en la primera muestra "
        f"es {1 - b:.3f} (β = {b:.3f}); el ARL1 es {arl1:.2f} muestras y el ATS1 es {arl1 * h:.2f} horas.",
        f"La carga muestral es de {n / h:.2f} unidades por hora; hasta detectar el corrimiento se inspeccionan en "
        f"promedio {arl1 * n:.1f} unidades.",
    ]
