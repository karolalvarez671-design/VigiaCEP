"""
Módulo 5. Capacidad del proceso.

Índices de capacidad potencial y real (corto plazo, σ dentro de subgrupos):
    Cp  = (LSE − LIE) / 6σ̂
    Cpu = (LSE − μ̂) / 3σ̂        Cpl = (μ̂ − LIE) / 3σ̂        Cpk = min(Cpu, Cpl)
Índices de desempeño (largo plazo, σ total = desviación estándar de todos los datos):
    Pp, Ppu, Ppl, Ppk con la misma fórmula y s total
Índice de Taguchi (si hay valor objetivo T):
    Cpm = (LSE − LIE) / 6·√(s² + (μ̂ − T)²)

Intervalos de confianza (Montgomery, 2020, ecs. 8.19 y 8.21):
    Cp:  Ĉp·√(χ²_{α/2, N−1}/(N−1))  ≤ Cp ≤ Ĉp·√(χ²_{1−α/2, N−1}/(N−1))
    Cpk: Ĉpk·[1 ∓ z_{α/2}·√(1/(9N·Ĉpk²) + 1/(2(N−1)))]

Fracción no conforme esperada bajo normalidad:
    p = Φ((LIE − μ̂)/σ) + 1 − Φ((LSE − μ̂)/σ), expresada también en ppm.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import constantes as K
from .supuestos import transformar


def sigma_dentro(largo: pd.DataFrame, metodo: str) -> float:
    """σ dentro de subgrupos con el mismo estimador del gráfico de control.

    metodo: 'R' (R̄/d2), 'S' (s̄/c4 o desviación agrupada si n varía) o 'MR' (MR̄/1.128).
    """
    datos = largo[largo["valor"].notna()]
    if metodo == "MR":
        x = datos.sort_values("orden")["valor"].to_numpy(float)
        return float(np.mean(np.abs(np.diff(x))) / K.d2(2))
    g = datos.groupby("orden")["valor"]
    n = g.count().to_numpy()
    if metodo == "R":
        rango = (g.max() - g.min()).to_numpy()
        if len(np.unique(n)) != 1:
            return float(np.mean(rango / np.array([K.d2(int(k)) for k in n])))
        return float(rango.mean() / K.d2(int(n[0])))
    s = g.std(ddof=1).to_numpy()
    if len(np.unique(n)) == 1:
        return float(s.mean() / K.c4(int(n[0])))
    gl = int(np.sum(n - 1))
    sp = math.sqrt(float(np.sum((n - 1) * s ** 2)) / gl)
    return sp / K.c4(gl + 1)


def _clase_cp(cp: float) -> tuple[str, str]:
    """Clasificación del proceso según Cp (adaptada de Gutiérrez Pulido y De la Vara, 2013)."""
    if not np.isfinite(cp):
        return "—", ""
    if cp >= 2:
        return "Clase mundial", "Calidad de nivel Seis Sigma."
    if cp > 1.33:
        return "Clase 1", "Adecuado."
    if cp > 1:
        return "Clase 2", "Parcialmente adecuado: requiere un control estricto."
    if cp > 0.67:
        return "Clase 3", "No adecuado: requiere análisis del proceso y modificaciones serias."
    return "Clase 4", "No adecuado: requiere modificaciones muy serias."


# Valores mínimos recomendados de Cpk (Montgomery, 2020, tabla 8.3)
MINIMOS_RECOMENDADOS = pd.DataFrame([
    ("Proceso existente", 1.33, 1.25),
    ("Proceso nuevo", 1.50, 1.45),
    ("Existente, parámetro crítico o de seguridad", 1.50, 1.45),
    ("Nuevo, parámetro crítico o de seguridad", 1.67, 1.60),
    ("Proceso Seis Sigma", 2.00, 2.00),
], columns=["Situación", "Especificación bilateral", "Especificación unilateral"])


def _ic_cp(cp, N, alfa):
    if not np.isfinite(cp) or N < 3:
        return np.nan, np.nan
    nu = N - 1
    return (cp * math.sqrt(stats.chi2.ppf(alfa / 2, nu) / nu),
            cp * math.sqrt(stats.chi2.ppf(1 - alfa / 2, nu) / nu))


def _ic_cpk(cpk, N, alfa):
    if not np.isfinite(cpk) or cpk <= 0 or N < 3:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alfa / 2)
    r = z * math.sqrt(1 / (9 * N * cpk ** 2) + 1 / (2 * (N - 1)))
    return cpk * (1 - r), cpk * (1 + r)


def _fracciones(mu, sigma, lie, lse):
    bajo = stats.norm.cdf((lie - mu) / sigma) if lie is not None else 0.0
    sobre = stats.norm.sf((lse - mu) / sigma) if lse is not None else 0.0
    return float(bajo), float(sobre)


def calcular(valores, sigma_w: float, lie: float | None = None, lse: float | None = None,
             objetivo: float | None = None, alfa: float = 0.05, cpk_minimo: float = 1.33) -> dict:
    """Índices de capacidad, fracción no conforme e interpretación."""
    x = np.asarray(pd.Series(valores).dropna(), dtype=float)
    N = len(x)
    if N < 2:
        raise ValueError("Se necesitan datos para calcular la capacidad.")
    if lie is None and lse is None:
        raise ValueError("Indique al menos un límite de especificación (LIE o LSE).")
    if lie is not None and lse is not None and lie >= lse:
        raise ValueError("El LIE debe ser menor que el LSE.")
    mu = float(x.mean())
    s_t = float(x.std(ddof=1))
    bilateral = lie is not None and lse is not None

    def indices(sig):
        cp = (lse - lie) / (6 * sig) if bilateral else np.nan
        cpu = (lse - mu) / (3 * sig) if lse is not None else np.nan
        cpl = (mu - lie) / (3 * sig) if lie is not None else np.nan
        cpk = np.nanmin([cpu, cpl])
        return cp, cpu, cpl, cpk

    cp, cpu, cpl, cpk = indices(sigma_w)
    pp, ppu, ppl, ppk = indices(s_t)
    cpm = np.nan
    if bilateral and objetivo is not None:
        cpm = (lse - lie) / (6 * math.sqrt(s_t ** 2 + (mu - objetivo) ** 2))

    filas = []
    def fila(nombre, valor, ic=(np.nan, np.nan), interpretacion=""):
        filas.append({"Índice": nombre, "Valor": valor, "IC inferior": ic[0], "IC superior": ic[1],
                      "Interpretación": interpretacion})

    if bilateral:
        clase, texto = _clase_cp(cp)
        fila("Cp (potencial, σ dentro)", cp, _ic_cp(cp, N, alfa), f"{clase}. {texto}")
    if lse is not None:
        fila("Cpu (lado superior)", cpu, interpretacion="Capacidad respecto al LSE.")
    if lie is not None:
        fila("Cpl (lado inferior)", cpl, interpretacion="Capacidad respecto al LIE.")
    fila("Cpk (real, σ dentro)", cpk, _ic_cpk(cpk, N, alfa),
         "Cumple el mínimo recomendado." if cpk >= cpk_minimo else f"Por debajo del mínimo recomendado ({cpk_minimo:g}).")
    if bilateral:
        fila("Pp (desempeño, σ total)", pp, _ic_cp(pp, N, alfa), "Usa la variación total (largo plazo).")
    fila("Ppk (desempeño, σ total)", ppk, _ic_cpk(ppk, N, alfa), "Usa la variación total (largo plazo).")
    if np.isfinite(cpm):
        fila("Cpm (Taguchi, respecto al objetivo)", cpm,
             interpretacion="Penaliza la distancia entre la media y el valor objetivo.")
    tabla_indices = pd.DataFrame(filas)

    # Fracción no conforme
    b_w, s_w = _fracciones(mu, sigma_w, lie, lse)
    b_t, s_tt = _fracciones(mu, s_t, lie, lse)
    obs_b = float(np.mean(x < lie)) if lie is not None else 0.0
    obs_s = float(np.mean(x > lse)) if lse is not None else 0.0
    def fila_fr(fuente, b, s):
        tot = b + s
        return {"Fuente": fuente, "Bajo el LIE (ppm)": b * 1e6, "Sobre el LSE (ppm)": s * 1e6,
                "Total (ppm)": tot * 1e6, "Total (%)": tot * 100,
                "Z benchmark": stats.norm.isf(tot) if 0 < tot < 1 else np.inf}
    tabla_fraccion = pd.DataFrame([
        fila_fr("Esperada, corto plazo (σ dentro)", b_w, s_w),
        fila_fr("Esperada, largo plazo (σ total)", b_t, s_tt),
        fila_fr(f"Observada en la muestra (N = {N})", obs_b, obs_s),
    ])
    p_centrado = 2 * stats.norm.sf(3 * cp) if bilateral else np.nan

    centro_spec = (lie + lse) / 2 if bilateral else None
    k_descentrado = abs(mu - centro_spec) / ((lse - lie) / 2) if bilateral else np.nan

    resumen = {
        "N": N, "media": mu, "sigma_dentro": sigma_w, "sigma_total": s_t, "LIE": lie, "LSE": lse,
        "objetivo": objetivo, "Cp": cp, "Cpk": cpk, "Cpu": cpu, "Cpl": cpl, "Pp": pp, "Ppk": ppk, "Cpm": cpm,
        "k_descentrado": k_descentrado, "ppm_corto": (b_w + s_w) * 1e6, "ppm_largo": (b_t + s_tt) * 1e6,
        "ppm_observado": (obs_b + obs_s) * 1e6, "ppm_si_centrado": p_centrado * 1e6 if bilateral else np.nan,
        "bilateral": bilateral, "cpk_minimo": cpk_minimo,
        "relacion_sigmas": s_t / sigma_w if sigma_w > 0 else np.nan,
    }
    return {"indices": tabla_indices, "fraccion": tabla_fraccion, "resumen": resumen,
            "conclusiones": interpretar(resumen)}


def interpretar(r: dict) -> list[str]:
    t = []
    if r["bilateral"]:
        clase, texto = _clase_cp(r["Cp"])
        t.append(f"Cp = {r['Cp']:.3f} ({clase}): la amplitud de la especificación equivale a {r['Cp']:.2f} veces la "
                 f"variación natural del proceso (6σ̂). {texto}")
        if r["k_descentrado"] > 0.05:
            lado = "superior" if r["Cpu"] < r["Cpl"] else "inferior"
            t.append(f"El proceso está descentrado (k = {100 * r['k_descentrado']:.1f} % de la semiamplitud), "
                     f"más cerca del límite {lado}: Cpk = {r['Cpk']:.3f} < Cp. Centrar el proceso en el valor nominal "
                     f"llevaría la fracción no conforme a ≈ {r['ppm_si_centrado']:.0f} ppm.")
    cumple = r["Cpk"] >= r["cpk_minimo"]
    t.append(f"Cpk = {r['Cpk']:.3f}: el proceso {'CUMPLE' if cumple else 'NO cumple'} el mínimo recomendado de "
             f"{r['cpk_minimo']:g} para un proceso existente.")
    t.append(f"Fracción no conforme esperada: {r['ppm_corto']:.0f} ppm a corto plazo y {r['ppm_largo']:.0f} ppm a largo "
             f"plazo; observada en la muestra: {r['ppm_observado']:.0f} ppm.")
    if np.isfinite(r["relacion_sigmas"]) and r["relacion_sigmas"] > 1.2:
        t.append(f"La variación total es {r['relacion_sigmas']:.2f} veces la variación dentro de subgrupos "
                 "(Ppk notablemente menor que Cpk): hay variación entre subgrupos, señal de causas especiales o "
                 "cambios de nivel entre periodos de muestreo.")
    return t


def dictamen_aptitud(estable: bool, cpk: float, cpk_minimo: float = 1.33) -> tuple[str, str]:
    """Combina estabilidad (control estadístico) y capacidad en un dictamen de aptitud.

    Adaptado de los cuatro estados de un proceso (Gutiérrez Pulido y De la Vara, 2013).
    """
    capaz = np.isfinite(cpk) and cpk >= cpk_minimo
    if estable and capaz:
        return "Estable y capaz", ("El proceso es apto: mantenga los gráficos de control en Fase II y evalúe "
                                   "reducir la frecuencia de inspección.")
    if estable and not capaz:
        return "Estable pero no capaz", ("La variación es de causas comunes y excede lo que permite la especificación: "
                                         "se requieren cambios en el sistema (método, equipo, material) para reducir "
                                         "la variación o centrar el proceso. Mientras tanto, inspección 100 % o muestreo.")
    if not estable and capaz:
        return "Capaz pero inestable", ("Los índices son buenos, pero hay causas especiales: elimínelas para que la "
                                        "capacidad calculada sea confiable y sostenible.")
    return "Ni estable ni capaz", ("Primero identifique y elimine las causas especiales (Fase I) y luego ataque la "
                                   "variación por causas comunes; los índices actuales no son confiables porque el "
                                   "proceso no está en control.")


def calcular_boxcox(largo: pd.DataFrame, lam: float, metodo_sigma: str, lie=None, lse=None,
                    objetivo=None, alfa: float = 0.05, cpk_minimo: float = 1.33) -> dict:
    """Capacidad sobre datos transformados con Box-Cox (datos y especificaciones > 0)."""
    datos = largo[largo["valor"].notna()].copy()
    if (datos["valor"] <= 0).any():
        raise ValueError("Box-Cox requiere datos positivos.")
    for nombre, v in (("LIE", lie), ("LSE", lse), ("objetivo", objetivo)):
        if v is not None and v <= 0:
            raise ValueError(f"El {nombre} debe ser positivo para transformarlo con Box-Cox.")
    datos["valor"] = transformar(datos["valor"].to_numpy(float), lam)
    t = lambda v: None if v is None else float(transformar(np.array([v]), lam)[0])
    sw = sigma_dentro(datos, metodo_sigma)
    res = calcular(datos["valor"], sw, t(lie), t(lse), t(objetivo), alfa, cpk_minimo)
    res["lambda"] = lam
    res["especificaciones_transformadas"] = (t(lie), t(lse))
    return res
