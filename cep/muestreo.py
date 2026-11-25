"""
Módulo 7. Muestreo de aceptación.

Planes por atributos (Montgomery, 2020, capítulo 15):
  * Plan simple (n, c): se acepta el lote si en la muestra hay d <= c defectuosos.
        Pa(p) = Σ_{d=0..c} P(d; n, p)   (binomial, hipergeométrica si el lote es pequeño, o Poisson)
        AOQ(p) = Pa·p·(N−n)/N           calidad promedio de salida (con inspección rectificadora)
        AOQL   = máx_p AOQ(p)           límite de la calidad promedio de salida
        ATI(p) = n + (1 − Pa)(N − n)    inspección total promedio
  * Diseño del plan simple por dos puntos de la curva OC:
        Pa(NCA) >= 1 − α   (riesgo del productor)   y   Pa(NCL) <= β   (riesgo del consumidor)
  * Plan doble (n1, c1, n2, c2): acepta si d1 <= c1; rechaza si d1 > c2; si no, toma una
    segunda muestra y acepta si d1 + d2 <= c2.  ASN = n1 + n2·P(c1 < d1 <= c2).

Plan por variables (método k, una especificación, σ conocida o desconocida):
        n = ((z_α + z_β) / (z_{p1} − z_{p2}))²        k = (z_{p1}·z_β + z_{p2}·z_α) / (z_α + z_β)
        con σ desconocida: n_s = n·(1 + k²/2)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, stats

MODELOS = ("binomial", "hipergeometrico", "poisson")


def prob_aceptacion(n: int, c: int, p, N: int | None = None, modelo: str = "binomial"):
    """Pa(p) para un plan simple (vectorizado en p)."""
    p = np.asarray(p, dtype=float)
    if modelo == "hipergeometrico":
        if not N:
            raise ValueError("El modelo hipergeométrico requiere el tamaño del lote N.")
        D = np.round(p * N).astype(int)
        return np.array([stats.hypergeom.cdf(c, N, d, n) for d in np.atleast_1d(D)]).reshape(p.shape)
    if modelo == "poisson":
        return stats.poisson.cdf(c, n * p)
    return stats.binom.cdf(c, n, p)


def curva_oc(n: int, c: int, N: int | None = None, modelo: str = "binomial", p_max: float | None = None,
             puntos: int = 200) -> pd.DataFrame:
    if p_max is None:
        p_max = min(1.0, max(0.05, 3.0 * (c + 1) / n))
    p = np.linspace(0, p_max, puntos)
    pa = prob_aceptacion(n, c, p, N, modelo)
    tabla = pd.DataFrame({"p": p, "Pa": pa})
    if N:
        tabla["AOQ"] = pa * p * (N - n) / N
        tabla["ATI"] = n + (1 - pa) * (N - n)
    return tabla


def aoql(n: int, c: int, N: int, modelo: str = "binomial") -> tuple[float, float]:
    """Máximo de la curva AOQ y el p donde ocurre."""
    f = lambda p: -float(prob_aceptacion(n, c, p, N, modelo if modelo != "hipergeometrico" else "binomial")
                         * p * (N - n) / N)
    res = optimize.minimize_scalar(f, bounds=(1e-9, min(1.0, 5.0 * (c + 1) / n)), method="bounded")
    return -res.fun, res.x


def disenar_plan_simple(nca: float, alfa: float, ncl: float, beta: float, N: int | None = None,
                        modelo: str = "binomial", c_max: int = 60, n_max: int = 20000) -> dict:
    """Plan (n, c) de menor tamaño que cumple ambos riesgos.

    nca: nivel de calidad aceptable (fracción, p. ej. 0.01) — riesgo del productor α
    ncl: nivel de calidad límite / LTPD (fracción, p. ej. 0.06) — riesgo del consumidor β
    """
    if not (0 < nca < ncl < 1):
        raise ValueError("Debe cumplirse 0 < NCA < NCL < 1.")
    tope = min(n_max, N) if N else n_max
    for c in range(0, c_max + 1):
        # menor n que cumple el riesgo del consumidor para este c
        n_lo, n_hi = c + 1, tope
        if prob_aceptacion(n_hi, c, ncl, N, modelo) > beta:
            continue
        while n_lo < n_hi:
            mid = (n_lo + n_hi) // 2
            if prob_aceptacion(mid, c, ncl, N, modelo) <= beta:
                n_hi = mid
            else:
                n_lo = mid + 1
        n = n_lo
        pa_nca = float(prob_aceptacion(n, c, nca, N, modelo))
        if pa_nca >= 1 - alfa:
            pa_ncl = float(prob_aceptacion(n, c, ncl, N, modelo))
            return {"n": n, "c": c, "Pa(NCA)": pa_nca, "Pa(NCL)": pa_ncl,
                    "α real": 1 - pa_nca, "β real": pa_ncl, "modelo": modelo}
    raise ValueError("No se encontró un plan con los parámetros dados; amplíe la distancia entre NCA y NCL.")


def evaluar_plan_simple(n: int, c: int, N: int | None, nca: float | None = None, ncl: float | None = None,
                        modelo: str = "binomial") -> dict:
    r = {"n": n, "c": c}
    if nca is not None:
        r["Pa(NCA)"] = float(prob_aceptacion(n, c, nca, N, modelo))
        r["Riesgo del productor α"] = 1 - r["Pa(NCA)"]
    if ncl is not None:
        r["Pa(NCL)"] = float(prob_aceptacion(n, c, ncl, N, modelo))
        r["Riesgo del consumidor β"] = r["Pa(NCL)"]
    if N:
        q, p_q = aoql(n, c, N, modelo)
        r["AOQL"] = q
        r["p en el AOQL"] = p_q
    # p con Pa = 0.5 (calidad de indiferencia)
    try:
        r["p de indiferencia (Pa = 0.5)"] = optimize.brentq(
            lambda p: float(prob_aceptacion(n, c, p, None, "binomial")) - 0.5, 1e-9, 1 - 1e-9)
    except ValueError:
        pass
    return r


def pa_doble(n1: int, c1: int, n2: int, c2: int, p) -> tuple[np.ndarray, np.ndarray]:
    """Pa y ASN de un plan doble (binomial)."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    pa = stats.binom.cdf(c1, n1, p)
    p_segunda = np.zeros_like(p)
    for d1 in range(c1 + 1, c2 + 1):
        prob_d1 = stats.binom.pmf(d1, n1, p)
        pa = pa + prob_d1 * stats.binom.cdf(c2 - d1, n2, p)
        p_segunda = p_segunda + prob_d1
    asn = n1 + n2 * p_segunda
    return pa, asn


def curva_doble(n1, c1, n2, c2, p_max=None, puntos=200) -> pd.DataFrame:
    if p_max is None:
        p_max = min(1.0, max(0.05, 3.0 * (c2 + 1) / (n1 + n2)))
    p = np.linspace(0, p_max, puntos)
    pa, asn = pa_doble(n1, c1, n2, c2, p)
    return pd.DataFrame({"p": p, "Pa": pa, "ASN": asn})


def plan_variables(nca: float, alfa: float, ncl: float, beta: float, sigma_conocida: bool = True) -> dict:
    """Plan por variables (método k) para una especificación unilateral."""
    if not (0 < nca < ncl < 1):
        raise ValueError("Debe cumplirse 0 < NCA < NCL < 1.")
    za, zb = stats.norm.isf(alfa), stats.norm.isf(beta)
    z1, z2 = stats.norm.isf(nca), stats.norm.isf(ncl)
    n = ((za + zb) / (z1 - z2)) ** 2
    k = (z1 * zb + z2 * za) / (za + zb)
    if not sigma_conocida:
        n = n * (1 + k ** 2 / 2)
    n_ent = int(math.ceil(n))
    return {"n (exacto)": n, "n": n_ent, "k": k, "σ": "conocida" if sigma_conocida else "desconocida (se usa s)",
            "Regla": f"Aceptar el lote si (LSE − x̄)/{'σ' if sigma_conocida else 's'} ≥ {k:.3f} "
                     f"(o si (x̄ − LIE)/{'σ' if sigma_conocida else 's'} ≥ {k:.3f} para el límite inferior)."}


def interpretar_plan(plan: dict, nca: float, ncl: float, N: int | None) -> list[str]:
    t = [f"Plan de muestreo simple: tomar n = {plan['n']} unidades al azar de cada lote"
         + (f" de N = {N}" if N else "") + f" y aceptar el lote si hay como máximo c = {plan['c']} defectuosas; "
         f"si hay {plan['c'] + 1} o más, rechazarlo."]
    if "Pa(NCA)" in plan:
        t.append(f"Con calidad aceptable ({100 * nca:.2f} % defectuoso) el lote se acepta con probabilidad "
                 f"{plan['Pa(NCA)']:.3f} (riesgo del productor α = {1 - plan['Pa(NCA)']:.3f}).")
    if "Pa(NCL)" in plan:
        t.append(f"Con calidad límite ({100 * ncl:.2f} % defectuoso) el lote se acepta con probabilidad "
                 f"{plan['Pa(NCL)']:.3f} (riesgo del consumidor β).")
    if plan.get("AOQL") is not None and np.isfinite(plan.get("AOQL", np.nan)):
        t.append(f"Con inspección rectificadora (los lotes rechazados se inspeccionan al 100 %), la calidad promedio "
                 f"de salida nunca supera el AOQL = {100 * plan['AOQL']:.3f} % defectuoso.")
    return t
