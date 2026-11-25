"""
Estructuras de datos comunes a todos los gráficos de control.

Un gráfico de control se representa con la clase `Carta`: los valores graficados,
la línea central y los límites (uno por punto, para soportar tamaños de muestra
variables), la fase de cada punto (I = estimación, II = monitoreo) y los puntos
excluidos del cálculo de los límites por tener una causa asignable identificada.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Senal:
    """Una señal de falta de control detectada por alguna regla."""

    posicion: int          # posición del punto en la carta (0, 1, 2, ...)
    etiqueta: str          # etiqueta del subgrupo o muestra
    regla: int             # número de regla (1 = punto fuera de límites)
    descripcion: str       # qué patrón se detectó
    causas: str            # causas típicas asociadas al patrón
    fase: str              # 'I' o 'II'


@dataclass
class Carta:
    """Resultado de un gráfico de control listo para graficar, interpretar y exportar."""

    clave: str                       # 'xbar', 'r', 's', 'i', 'mr', 'p', 'np', 'c', 'u', 'ewma', 'cusum_sup', 'cusum_inf'
    titulo: str                      # nombre para mostrar, p. ej. "Gráfico X̄"
    etiqueta_y: str                  # nombre del eje vertical
    etiquetas: list                  # etiqueta de cada subgrupo o muestra (eje horizontal)
    valores: np.ndarray              # estadístico graficado
    lc: np.ndarray                   # línea central por punto
    lcs: np.ndarray                  # límite de control superior por punto
    lci: np.ndarray                  # límite de control inferior por punto (NaN si no aplica)
    fase: np.ndarray                 # 'I' o 'II' por punto
    excluido: np.ndarray             # True si el punto se excluyó del cálculo de límites
    L: float = 3.0                   # ancho de los límites en sigmas
    parametros: dict = field(default_factory=dict)   # parámetros estimados (x̿, R̄, σ̂, p̄, ...)
    reglas_por_defecto: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    senales: list = field(default_factory=list)      # lista de Senal (la llena reglas.evaluar)
    notas: list = field(default_factory=list)        # advertencias técnicas para el usuario

    # ------------------------------------------------------------------ utilidades
    @property
    def m(self) -> int:
        return len(self.valores)

    def sigma_puntos(self) -> np.ndarray:
        """Desviación estándar del estadístico en cada punto, deducida de los límites."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.lcs - self.lc) / self.L

    def fuera_de_limites(self) -> np.ndarray:
        v = self.valores
        sobre = np.where(np.isfinite(self.lcs), v > self.lcs + 1e-12, False)
        bajo = np.where(np.isfinite(self.lci), v < self.lci - 1e-12, False)
        return np.where(np.isfinite(v), sobre | bajo, False)

    def posiciones_con_senal(self, solo_regla_1: bool = False) -> set:
        return {s.posicion for s in self.senales if (s.regla == 1 or not solo_regla_1)}

    def indices_fase(self, fase: str) -> np.ndarray:
        return np.where(self.fase == fase)[0]

    def tabla(self) -> pd.DataFrame:
        """Tabla punto a punto para mostrar o exportar."""
        fuera = self.fuera_de_limites()
        reglas = {}
        for s in self.senales:
            reglas.setdefault(s.posicion, []).append(str(s.regla))
        return pd.DataFrame({
            "Subgrupo/muestra": self.etiquetas,
            "Fase": self.fase,
            self.etiqueta_y: self.valores,
            "LCI": self.lci,
            "LC": self.lc,
            "LCS": self.lcs,
            "Fuera de límites": np.where(fuera, "Sí", ""),
            "Reglas violadas": [", ".join(reglas.get(i, [])) for i in range(self.m)],
            "Excluido (causa asignable)": np.where(self.excluido, "Sí", ""),
        })

    def resumen_limites(self) -> dict:
        """Límites como texto (constantes o 'variables' cuando dependen de n)."""
        def fmt(a):
            a = np.asarray(a, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return "—"
            if np.allclose(a, a[0], rtol=0, atol=1e-12):
                return a[0]
            return f"variable ({a.min():.4g} a {a.max():.4g})"

        return {"LCI": fmt(self.lci), "LC": fmt(self.lc), "LCS": fmt(self.lcs)}


def a_arreglo(x, m: int | None = None, dtype=float) -> np.ndarray:
    """Convierte escalares o listas a un arreglo de longitud m."""
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 0 and m is not None:
        arr = np.full(m, arr, dtype=dtype)
    return arr
