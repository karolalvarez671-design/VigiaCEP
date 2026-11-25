"""
Reglas de sensibilidad para detectar patrones no aleatorios (Western Electric, 1956;
Nelson, 1984; Montgomery, 2020, sección 5.3.6).

Cada regla se evalúa sobre el estadístico estandarizado z = (y - LC) / σ_y, donde
σ_y se deduce de los límites (σ_y = (LCS - LC) / L). Así las reglas funcionan
igual cuando los límites varían de un punto a otro (cartas p y u con n variable).
La regla 1 usa directamente los límites calculados (incluidos los truncados en 0).

Las reglas se evalúan por separado en la Fase I y en la Fase II.
"""

from __future__ import annotations

import numpy as np

from .modelos import Carta, Senal

# Número de puntos consecutivos del mismo lado para la regla 4.
# Western Electric y Montgomery usan 8; Nelson y Minitab usan 9.
RACHA_LADO_POR_DEFECTO = 8

DESCRIPCIONES = {
    1: "1 punto fuera de los límites de control",
    2: "2 de 3 puntos consecutivos más allá de 2σ del mismo lado",
    3: "4 de 5 puntos consecutivos más allá de 1σ del mismo lado",
    4: "{k} puntos consecutivos del mismo lado de la línea central",
    5: "6 puntos consecutivos en tendencia creciente o decreciente",
    6: "14 puntos consecutivos alternando hacia arriba y hacia abajo",
    7: "15 puntos consecutivos dentro de ±1σ (zona C)",
    8: "8 puntos consecutivos fuera de ±1σ, a ambos lados",
}

CAUSAS = {
    1: "Causa especial puntual: error de medición o de registro, material defectuoso, "
       "falla o ajuste de máquina, cambio de operario.",
    2: "Desplazamiento moderado de la media: cambio de lote de materia prima, ajuste de máquina, "
       "cambio de método.",
    3: "Desplazamiento pequeño y sostenido de la media.",
    4: "Cambio sostenido de nivel: nuevo lote o proveedor, nuevo operario, recalibración o cambio "
       "de método.",
    5: "Tendencia: desgaste de herramienta, fatiga, deriva térmica o deterioro gradual del equipo.",
    6: "Sobreajuste (el operario corrige cada medición) o dos fuentes que se alternan "
       "(máquinas, turnos, operarios).",
    7: "Estratificación: el subgrupo mezcla fuentes de variación distintas o los límites están mal "
       "calculados (variación menor a la esperada).",
    8: "Mezcla: los datos provienen de dos poblaciones distintas (dos máquinas, turnos o materiales).",
}

NOMBRES_CORTOS = {
    1: "Fuera de límites",
    2: "2 de 3 > 2σ",
    3: "4 de 5 > 1σ",
    4: "Racha de un lado",
    5: "Tendencia",
    6: "Alternancia",
    7: "Estratificación",
    8: "Mezcla",
}


def descripcion(regla: int, racha_lado: int = RACHA_LADO_POR_DEFECTO) -> str:
    return DESCRIPCIONES[regla].format(k=racha_lado)


def _segmentos_por_fase(fase: np.ndarray):
    """Devuelve (fase, índices) para cada fase en el orden en que aparecen."""
    for f in ("I", "II"):
        idx = np.where(fase == f)[0]
        if idx.size:
            yield f, idx


def _evaluar_segmento(y, z, lcs, lci, reglas, racha_lado):
    """Devuelve lista de (posición_local, regla) para un segmento de puntos consecutivos."""
    hallazgos = []
    n = len(y)
    finito = np.isfinite(z)

    # Regla 1: fuera de los límites calculados
    if 1 in reglas:
        sobre = np.isfinite(lcs) & (y > lcs + 1e-12)
        bajo = np.isfinite(lci) & (y < lci - 1e-12)
        for i in np.where((sobre | bajo) & np.isfinite(y))[0]:
            hallazgos.append((int(i), 1))

    # Reglas 2 y 3: k de m puntos más allá de cierto número de sigmas, mismo lado
    def k_de_m(k, m, umbral, regla):
        for i in range(n):
            if not finito[i]:
                continue
            ini = max(0, i - m + 1)
            ventana = z[ini:i + 1]
            ventana = ventana[np.isfinite(ventana)]
            if z[i] > umbral and np.sum(ventana > umbral) >= k:
                hallazgos.append((i, regla))
            elif z[i] < -umbral and np.sum(ventana < -umbral) >= k:
                hallazgos.append((i, regla))

    if 2 in reglas:
        k_de_m(2, 3, 2.0, 2)
    if 3 in reglas:
        k_de_m(4, 5, 1.0, 3)

    # Regla 4: racha del mismo lado de la línea central
    if 4 in reglas:
        racha, lado = 0, 0
        for i in range(n):
            s = 0 if not finito[i] else (1 if z[i] > 0 else (-1 if z[i] < 0 else 0))
            if s != 0 and s == lado:
                racha += 1
            else:
                racha, lado = (1, s) if s != 0 else (0, 0)
            if racha >= racha_lado:
                hallazgos.append((i, 4))

    # Regla 5: 6 puntos en tendencia (5 incrementos o decrementos seguidos)
    if 5 in reglas:
        racha, signo = 0, 0
        for i in range(1, n):
            if not (finito[i] and finito[i - 1]):
                racha, signo = 0, 0
                continue
            d = y[i] - y[i - 1]
            s = 1 if d > 0 else (-1 if d < 0 else 0)
            if s != 0 and s == signo:
                racha += 1
            else:
                racha, signo = (1, s) if s != 0 else (0, 0)
            if racha >= 5:
                hallazgos.append((i, 5))

    # Regla 6: 14 puntos alternando (13 cambios de dirección seguidos)
    if 6 in reglas:
        racha, signo_prev = 0, 0
        for i in range(1, n):
            if not (finito[i] and finito[i - 1]):
                racha, signo_prev = 0, 0
                continue
            d = y[i] - y[i - 1]
            s = 1 if d > 0 else (-1 if d < 0 else 0)
            if s == 0:
                racha, signo_prev = 0, 0
                continue
            if signo_prev != 0 and s == -signo_prev:
                racha += 1
            else:
                racha = 1
            signo_prev = s
            if racha >= 13:
                hallazgos.append((i, 6))

    # Regla 7: 15 puntos seguidos dentro de ±1σ
    if 7 in reglas:
        racha = 0
        for i in range(n):
            racha = racha + 1 if (finito[i] and abs(z[i]) < 1.0) else 0
            if racha >= 15:
                hallazgos.append((i, 7))

    # Regla 8: 8 puntos seguidos fuera de ±1σ (a cualquier lado)
    if 8 in reglas:
        racha = 0
        for i in range(n):
            racha = racha + 1 if (finito[i] and abs(z[i]) > 1.0) else 0
            if racha >= 8:
                hallazgos.append((i, 8))

    return hallazgos


def evaluar(carta: Carta, reglas=None, racha_lado: int = RACHA_LADO_POR_DEFECTO) -> Carta:
    """Evalúa las reglas sobre la carta y guarda las señales en carta.senales."""
    reglas = tuple(sorted(set(reglas if reglas is not None else carta.reglas_por_defecto)))
    y = np.asarray(carta.valores, dtype=float)
    sig = carta.sigma_puntos()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(np.isfinite(sig) & (sig > 0), (y - carta.lc) / sig, np.nan)

    senales = []
    for f, idx in _segmentos_por_fase(np.asarray(carta.fase)):
        locales = _evaluar_segmento(y[idx], z[idx], carta.lcs[idx], carta.lci[idx], reglas, racha_lado)
        for pos_local, regla in locales:
            pos = int(idx[pos_local])
            senales.append(Senal(
                posicion=pos,
                etiqueta=str(carta.etiquetas[pos]),
                regla=regla,
                descripcion=descripcion(regla, racha_lado),
                causas=CAUSAS[regla],
                fase=f,
            ))
    # Una señal por (punto, regla), ordenadas por posición
    unicas = {(s.posicion, s.regla): s for s in senales}
    carta.senales = sorted(unicas.values(), key=lambda s: (s.posicion, s.regla))
    return carta


def tabla_senales(cartas):
    """Tabla consolidada de señales de una o varias cartas."""
    import pandas as pd

    filas = []
    for c in cartas:
        for s in c.senales:
            filas.append({
                "Gráfico": c.titulo,
                "Fase": s.fase,
                "Subgrupo/muestra": s.etiqueta,
                "Regla": s.regla,
                "Patrón detectado": s.descripcion,
                "Causas típicas a investigar": s.causas,
            })
    return pd.DataFrame(filas, columns=["Gráfico", "Fase", "Subgrupo/muestra", "Regla",
                                         "Patrón detectado", "Causas típicas a investigar"])
