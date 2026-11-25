"""
Constantes de los gráficos de control para subgrupos de tamaño n.

d2(n) y d3(n) son la media y la desviación estándar del rango relativo
W = R/σ de n observaciones normales independientes. Se calcularon por
integración numérica de la distribución del rango y coinciden con la tabla del
Apéndice VI de Montgomery (2020) a tres decimales (ver pruebas/test_validacion.py).

c4(n) se calcula de forma exacta con la función gamma:
    c4 = sqrt(2/(n-1)) * Γ(n/2) / Γ((n-1)/2)

Las constantes derivadas se expresan en función de L (número de sigmas de los
límites, L = 3 por defecto) para poder diseñar gráficos con otros anchos de límite:
    A2 = L / (d2 √n)          X̄ con R̄
    A3 = L / (c4 √n)          X̄ con s̄
    D3 = max(0, 1 - L d3/d2)  D4 = 1 + L d3/d2
    B3 = max(0, 1 - L √(1-c4²)/c4)  B4 = 1 + L √(1-c4²)/c4
"""

from __future__ import annotations

import math
from functools import lru_cache

from scipy import integrate, special, stats

# d2(n) y d3(n) para n = 2..25 (integración numérica, 5 decimales)
_D2 = {
    2: 1.12838, 3: 1.69257, 4: 2.05875, 5: 2.32593, 6: 2.53441, 7: 2.70436,
    8: 2.84720, 9: 2.97003, 10: 3.07751, 11: 3.17287, 12: 3.25846, 13: 3.33598,
    14: 3.40676, 15: 3.47183, 16: 3.53198, 17: 3.58788, 18: 3.64006, 19: 3.68896,
    20: 3.73495, 21: 3.77834, 22: 3.81938, 23: 3.85832, 24: 3.89535, 25: 3.93063,
}
_D3 = {
    2: 0.85250, 3: 0.88837, 4: 0.87981, 5: 0.86408, 6: 0.84804, 7: 0.83321,
    8: 0.81983, 9: 0.80783, 10: 0.79705, 11: 0.78731, 12: 0.77848, 13: 0.77042,
    14: 0.76302, 15: 0.75621, 16: 0.74991, 17: 0.74405, 18: 0.73859, 19: 0.73348,
    20: 0.72869, 21: 0.72417, 22: 0.71991, 23: 0.71589, 24: 0.71207, 25: 0.70844,
}

N_MAX_RANGO = 25  # Las cartas R se limitan a n <= 25 (para n > 10 se recomienda S)


def _validar_n(n: int, minimo: int = 2) -> int:
    n = int(n)
    if n < minimo:
        raise ValueError(f"El tamaño de subgrupo debe ser al menos {minimo} (se recibió n = {n}).")
    return n


@lru_cache(maxsize=None)
def c4(n: int) -> float:
    """Factor de sesgo de la desviación estándar muestral: E[s] = c4·σ."""
    n = _validar_n(n)
    return math.sqrt(2.0 / (n - 1)) * math.exp(special.gammaln(n / 2.0) - special.gammaln((n - 1) / 2.0))


@lru_cache(maxsize=None)
def d2(n: int) -> float:
    """Media del rango relativo: E[R] = d2·σ. Para n > 25 se integra numéricamente."""
    n = _validar_n(n)
    if n in _D2:
        return _D2[n]
    phi, Phi = stats.norm.pdf, stats.norm.cdf
    integrando = lambda x: 1.0 - Phi(x) ** n - (1.0 - Phi(x)) ** n
    valor, _ = integrate.quad(integrando, -12, 12, limit=200)
    return float(valor)


def d3(n: int) -> float:
    """Desviación estándar del rango relativo: σ_R = d3·σ (solo n = 2..25)."""
    n = _validar_n(n)
    if n not in _D3:
        raise ValueError("d3 solo está tabulado para 2 <= n <= 25; use la carta S para subgrupos grandes.")
    return _D3[n]


def A2(n: int, L: float = 3.0) -> float:
    return L / (d2(n) * math.sqrt(n))


def A3(n: int, L: float = 3.0) -> float:
    return L / (c4(n) * math.sqrt(n))


def D3(n: int, L: float = 3.0) -> float:
    return max(0.0, 1.0 - L * d3(n) / d2(n))


def D4(n: int, L: float = 3.0) -> float:
    return 1.0 + L * d3(n) / d2(n)


def B3(n: int, L: float = 3.0) -> float:
    c = c4(n)
    return max(0.0, 1.0 - L * math.sqrt(1.0 - c * c) / c)


def B4(n: int, L: float = 3.0) -> float:
    c = c4(n)
    return 1.0 + L * math.sqrt(1.0 - c * c) / c


def tabla_constantes(n_max: int = 25, L: float = 3.0):
    """Tabla de constantes (como la del apéndice de los textos) para mostrar en la app."""
    import pandas as pd

    filas = []
    for n in range(2, n_max + 1):
        filas.append({
            "n": n, "d2": d2(n), "d3": d3(n), "c4": c4(n),
            "A2": A2(n, L), "A3": A3(n, L), "D3": D3(n, L), "D4": D4(n, L),
            "B3": B3(n, L), "B4": B4(n, L),
        })
    return pd.DataFrame(filas)
