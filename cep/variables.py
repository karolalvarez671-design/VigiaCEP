"""
Módulo 2. Gráficos de control por variables.

  * X̄-R   subgrupos de tamaño constante 2 <= n <= 25 (recomendado n <= 10)
  * X̄-S   subgrupos de cualquier tamaño, constante o variable (recomendado n > 10)
  * I-MR  mediciones individuales (n = 1)
  * EWMA y CUSUM tabular, para detectar corrimientos pequeños (Fase II)

Fase I: los límites se estiman con los subgrupos de Fase I que no estén excluidos.
Cuando un punto tiene una causa asignable identificada, se excluye (con su
justificación) y los límites se recalculan; el proceso se repite hasta lograr
control estadístico (límites de prueba -> límites definitivos).
Fase II: los nuevos subgrupos se comparan con los límites congelados de la Fase I
(o con valores estándar μ0, σ0 dados por la empresa).

Estimadores de σ (variación dentro de subgrupos, "corto plazo"):
    X̄-R:  σ̂ = R̄ / d2(n)
    X̄-S:  σ̂ = s̄ / c4(n)                       (n constante)
           σ̂ = s_p / c4(Σ(n_i-1)+1)            (n variable, desviación agrupada)
    I-MR:  σ̂ = MR̄ / d2(2) = MR̄ / 1.128
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import constantes as K
from .datos import resumen_subgrupos
from .modelos import Carta
from .reglas import RACHA_LADO_POR_DEFECTO, evaluar

REGLAS_DISPERSION = (1,)  # en R, S y MR se evalúa por defecto solo la regla 1


@dataclass
class Estandar:
    """Valores estándar conocidos del proceso (por ejemplo, metas de la empresa)."""

    mu0: float
    sigma0: float


@dataclass
class ResultadoVariables:
    """Par de cartas (posición y dispersión) más la información de la estimación."""

    tipo: str                      # 'xbar_r', 'xbar_s', 'i_mr'
    carta_pos: Carta               # X̄ o I
    carta_disp: Carta              # R, S o MR
    media: float                   # x̿ o media de individuales (Fase I)
    sigma: float                   # σ̂ dentro de subgrupos (corto plazo)
    n: int                         # tamaño de subgrupo típico
    n_constante: bool
    metodo_sigma: str              # texto: cómo se estimó σ
    excluidos: list = field(default_factory=list)
    estandar: Estandar | None = None
    notas: list = field(default_factory=list)

    def cartas(self):
        return [self.carta_pos, self.carta_disp]

    def senales_fase(self, fase: str):
        return [s for c in self.cartas() for s in c.senales if s.fase == fase]

    def en_control_fase1(self) -> bool:
        return len(self.senales_fase("I")) == 0


def recomendar_carta(n: int, n_constante: bool) -> tuple[str, str]:
    """Recomienda el tipo de carta por variables según el tamaño de subgrupo."""
    if n <= 1:
        return "i_mr", ("Con mediciones individuales (n = 1) se usa la carta I-MR: la variación "
                        "dentro de subgrupo se estima con el rango móvil de dos observaciones consecutivas.")
    if not n_constante:
        return "xbar_s", ("El tamaño de subgrupo varía; la carta X̄-S admite n variable con límites "
                          "calculados para cada subgrupo.")
    if n <= 10:
        return "xbar_r", (f"Con n = {n} (pequeño y constante) la carta X̄-R es adecuada: el rango es "
                          "casi tan eficiente como s para estimar σ y es más fácil de calcular en planta.")
    return "xbar_s", (f"Con n = {n} (> 10) el rango pierde eficiencia; se recomienda la carta X̄-S.")


def _excluidos_mask(etiquetas, excluidos) -> np.ndarray:
    ex = {str(e) for e in (excluidos or [])}
    return np.array([str(e) in ex for e in etiquetas], dtype=bool)


def _finalizar(carta: Carta, reglas, racha_lado) -> Carta:
    return evaluar(carta, reglas if reglas is not None else carta.reglas_por_defecto, racha_lado)


# ---------------------------------------------------------------------------
# X̄-R
# ---------------------------------------------------------------------------

def carta_xbar_r(largo: pd.DataFrame, L: float = 3.0, excluidos=(), estandar: Estandar | None = None,
                 reglas_pos=None, reglas_disp=REGLAS_DISPERSION,
                 racha_lado: int = RACHA_LADO_POR_DEFECTO) -> ResultadoVariables:
    sub = resumen_subgrupos(largo)
    if sub.empty:
        raise ValueError("No hay subgrupos con datos.")
    if sub["n"].nunique() != 1:
        raise ValueError("La carta X̄-R requiere tamaño de subgrupo constante. Use X̄-S o elimine los "
                         "subgrupos incompletos en Gestión de datos.")
    n = int(sub["n"].iloc[0])
    if n < 2 or n > K.N_MAX_RANGO:
        raise ValueError(f"La carta X̄-R requiere 2 <= n <= {K.N_MAX_RANGO} (n = {n}).")

    etiquetas = sub["subgrupo"].tolist()
    fase = sub["fase"].to_numpy()
    excl = _excluidos_mask(etiquetas, excluidos)
    usa = (fase == "I") & ~excl
    if estandar is None and usa.sum() < 2:
        raise ValueError("Se necesitan al menos 2 subgrupos de Fase I no excluidos para estimar los límites.")

    d2, d3 = K.d2(n), K.d3(n)
    if estandar is not None:
        media, sigma = estandar.mu0, estandar.sigma0
        metodo = "Valores estándar dados (μ0, σ0)"
        rbar = d2 * sigma
    else:
        media = float(sub.loc[usa, "media"].mean())
        rbar = float(sub.loc[usa, "rango"].mean())
        sigma = rbar / d2
        metodo = f"σ̂ = R̄/d2 = {rbar:.6g}/{d2:.5f}"

    m = len(sub)
    ancho = L * sigma / math.sqrt(n)
    c_pos = Carta(
        clave="xbar", titulo="Gráfico X̄ (medias)", etiqueta_y="Media del subgrupo",
        etiquetas=etiquetas, valores=sub["media"].to_numpy(float),
        lc=np.full(m, media), lcs=np.full(m, media + ancho), lci=np.full(m, media - ancho),
        fase=fase, excluido=excl, L=L,
        parametros={"x̿ (media de medias)": media, "R̄": rbar, "σ̂": sigma, "n": n, "L": L,
                    "A2 (con L)": K.A2(n, L)},
    )
    lcs_r = (d2 + L * d3) * sigma
    lci_r = max(0.0, (d2 - L * d3) * sigma)
    c_disp = Carta(
        clave="r", titulo="Gráfico R (rangos)", etiqueta_y="Rango del subgrupo",
        etiquetas=etiquetas, valores=sub["rango"].to_numpy(float),
        lc=np.full(m, d2 * sigma), lcs=np.full(m, lcs_r), lci=np.full(m, lci_r),
        fase=fase, excluido=excl, L=L, reglas_por_defecto=REGLAS_DISPERSION,
        parametros={"R̄": d2 * sigma, "D3 (con L)": K.D3(n, L), "D4 (con L)": K.D4(n, L), "n": n},
    )
    notas = []
    if n > 10:
        notas.append("Con n > 10 el rango pierde eficiencia; considere la carta X̄-S.")
    if lci_r == 0:
        c_disp.notas.append("El LCI de R es 0 (n <= 6 con L = 3): la carta no detecta disminuciones de variabilidad.")
    _finalizar(c_pos, reglas_pos, racha_lado)
    _finalizar(c_disp, reglas_disp, racha_lado)
    return ResultadoVariables("xbar_r", c_pos, c_disp, media, sigma, n, True, metodo,
                              list(map(str, np.array(etiquetas)[excl])), estandar, notas)


# ---------------------------------------------------------------------------
# X̄-S (n constante o variable)
# ---------------------------------------------------------------------------

def carta_xbar_s(largo: pd.DataFrame, L: float = 3.0, excluidos=(), estandar: Estandar | None = None,
                 reglas_pos=None, reglas_disp=REGLAS_DISPERSION,
                 racha_lado: int = RACHA_LADO_POR_DEFECTO) -> ResultadoVariables:
    sub = resumen_subgrupos(largo)
    if sub.empty:
        raise ValueError("No hay subgrupos con datos.")
    if (sub["n"] < 2).any():
        raise ValueError("La carta X̄-S requiere al menos 2 mediciones en cada subgrupo.")
    etiquetas = sub["subgrupo"].tolist()
    fase = sub["fase"].to_numpy()
    excl = _excluidos_mask(etiquetas, excluidos)
    usa = (fase == "I") & ~excl
    if estandar is None and usa.sum() < 2:
        raise ValueError("Se necesitan al menos 2 subgrupos de Fase I no excluidos para estimar los límites.")

    n_i = sub["n"].to_numpy(int)
    constante = len(np.unique(n_i)) == 1
    n_tip = int(pd.Series(n_i).mode().iloc[0])

    if estandar is not None:
        media, sigma = estandar.mu0, estandar.sigma0
        metodo = "Valores estándar dados (μ0, σ0)"
        sbar = None
    elif constante:
        media = float(sub.loc[usa, "media"].mean())
        sbar = float(sub.loc[usa, "desv"].mean())
        sigma = sbar / K.c4(n_tip)
        metodo = f"σ̂ = s̄/c4 = {sbar:.6g}/{K.c4(n_tip):.5f}"
    else:
        nu = n_i[usa]
        media = float(np.sum(nu * sub.loc[usa, "media"]) / np.sum(nu))
        gl = int(np.sum(nu - 1))
        sp = math.sqrt(float(np.sum((nu - 1) * sub.loc[usa, "desv"] ** 2)) / gl)
        sigma = sp / K.c4(gl + 1)
        sbar = sp
        metodo = f"σ̂ = s_p/c4(gl+1) = {sp:.6g}/{K.c4(gl + 1):.5f} (desviación agrupada, n variable)"

    c4i = np.array([K.c4(int(k)) for k in n_i])
    ancho = L * sigma / np.sqrt(n_i)
    c_pos = Carta(
        clave="xbar", titulo="Gráfico X̄ (medias)", etiqueta_y="Media del subgrupo",
        etiquetas=etiquetas, valores=sub["media"].to_numpy(float),
        lc=np.full(len(sub), media), lcs=media + ancho, lci=media - ancho,
        fase=fase, excluido=excl, L=L,
        parametros={"x̿ (media de medias)": media, "s̄": sbar, "σ̂": sigma, "n": n_tip if constante else "variable", "L": L},
    )
    lc_s = c4i * sigma
    c_disp = Carta(
        clave="s", titulo="Gráfico S (desviaciones estándar)", etiqueta_y="Desviación estándar del subgrupo",
        etiquetas=etiquetas, valores=sub["desv"].to_numpy(float),
        lc=lc_s, lcs=lc_s + L * sigma * np.sqrt(1 - c4i ** 2),
        lci=np.maximum(0.0, lc_s - L * sigma * np.sqrt(1 - c4i ** 2)),
        fase=fase, excluido=excl, L=L, reglas_por_defecto=REGLAS_DISPERSION,
        parametros={"s̄ (línea central)": float(np.mean(lc_s)), "σ̂": sigma},
    )
    notas = []
    if not constante:
        notas.append("Tamaño de subgrupo variable: los límites cambian con n_i en cada subgrupo.")
    _finalizar(c_pos, reglas_pos, racha_lado)
    _finalizar(c_disp, reglas_disp, racha_lado)
    return ResultadoVariables("xbar_s", c_pos, c_disp, media, sigma, n_tip, constante, metodo,
                              list(map(str, np.array(etiquetas)[excl])), estandar, notas)


# ---------------------------------------------------------------------------
# I-MR
# ---------------------------------------------------------------------------

def carta_i_mr(largo: pd.DataFrame, L: float = 3.0, excluidos=(), estandar: Estandar | None = None,
               reglas_pos=None, reglas_disp=REGLAS_DISPERSION,
               racha_lado: int = RACHA_LADO_POR_DEFECTO) -> ResultadoVariables:
    """Carta de individuales y rango móvil (rango de 2 observaciones consecutivas)."""
    datos = largo[largo["valor"].notna()].sort_values("orden", kind="stable")
    if datos.groupby("orden")["valor"].count().max() > 1:
        raise ValueError("La carta I-MR es para mediciones individuales (un dato por subgrupo). "
                         "Use el modo 'individual' en Gestión de datos.")
    x = datos["valor"].to_numpy(float)
    etiquetas = datos["subgrupo"].tolist()
    fase = datos["fase"].to_numpy()
    m = len(x)
    if m < 3:
        raise ValueError("Se necesitan al menos 3 mediciones individuales.")
    excl = _excluidos_mask(etiquetas, excluidos)
    usa = (fase == "I") & ~excl

    mr = np.full(m, np.nan)
    mr[1:] = np.abs(np.diff(x))
    usa_mr = np.zeros(m, dtype=bool)
    usa_mr[1:] = usa[1:] & usa[:-1]   # el MR solo cuenta si ambas observaciones son válidas

    d2, d3 = K.d2(2), K.d3(2)
    if estandar is not None:
        media, sigma = estandar.mu0, estandar.sigma0
        mrbar = d2 * sigma
        metodo = "Valores estándar dados (μ0, σ0)"
    else:
        if usa.sum() < 2 or usa_mr.sum() < 1:
            raise ValueError("Se necesitan al menos 2 observaciones consecutivas de Fase I no excluidas.")
        media = float(x[usa].mean())
        mrbar = float(mr[usa_mr].mean())
        sigma = mrbar / d2
        metodo = f"σ̂ = MR̄/d2 = {mrbar:.6g}/{d2:.5f}"

    c_pos = Carta(
        clave="i", titulo="Gráfico I (individuales)", etiqueta_y="Medición individual",
        etiquetas=etiquetas, valores=x, lc=np.full(m, media),
        lcs=np.full(m, media + L * sigma), lci=np.full(m, media - L * sigma),
        fase=fase, excluido=excl, L=L,
        parametros={"x̄": media, "MR̄": mrbar, "σ̂": sigma, "L": L},
    )
    c_disp = Carta(
        clave="mr", titulo="Gráfico MR (rango móvil)", etiqueta_y="Rango móvil |xᵢ − xᵢ₋₁|",
        etiquetas=etiquetas, valores=mr, lc=np.full(m, mrbar),
        lcs=np.full(m, (d2 + L * d3) * sigma), lci=np.full(m, max(0.0, (d2 - L * d3) * sigma)),
        fase=fase, excluido=excl, L=L, reglas_por_defecto=REGLAS_DISPERSION,
        parametros={"MR̄": mrbar, "D4 (con L)": K.D4(2, L)},
    )
    notas = ["En la carta MR los puntos consecutivos están correlacionados; las reglas de rachas no se "
             "aplican a ella (solo la regla 1)."]
    _finalizar(c_pos, reglas_pos, racha_lado)
    _finalizar(c_disp, reglas_disp, racha_lado)
    return ResultadoVariables("i_mr", c_pos, c_disp, media, sigma, 1, True, metodo,
                              list(map(str, np.array(etiquetas)[excl])), estandar, notas)


FUNCIONES = {"xbar_r": carta_xbar_r, "xbar_s": carta_xbar_s, "i_mr": carta_i_mr}
NOMBRES = {"xbar_r": "X̄-R", "xbar_s": "X̄-S", "i_mr": "I-MR"}


def calcular(tipo: str, largo: pd.DataFrame, **kw) -> ResultadoVariables:
    return FUNCIONES[tipo](largo, **kw)


def fase1_iterativa(tipo: str, largo: pd.DataFrame, excluidos_iniciales=(), max_iter: int = 10,
                    **kw) -> tuple[ResultadoVariables, list[dict]]:
    """Excluye de forma iterativa los puntos de Fase I fuera de límites y recalcula.

    Solo se recomienda cuando cada punto excluido tiene una causa asignable
    identificada; la app le pide al usuario documentar esa causa.
    Devuelve el resultado final y el historial de iteraciones.
    """
    excluidos = [str(e) for e in excluidos_iniciales]
    historial = []
    for it in range(1, max_iter + 1):
        res = calcular(tipo, largo, excluidos=excluidos, **kw)
        nuevos = []
        for c in res.cartas():
            fuera = c.fuera_de_limites()
            for i in np.where(fuera & (c.fase == "I") & ~c.excluido)[0]:
                et = str(c.etiquetas[i])
                if et not in excluidos and et not in nuevos:
                    nuevos.append(et)
        historial.append({
            "Iteración": it,
            "Centro (posición)": res.media,
            "σ̂": res.sigma,
            "LCS (posición)": float(np.nanmax(res.carta_pos.lcs)),
            "LCI (posición)": float(np.nanmin(res.carta_pos.lci)),
            "Puntos fuera de límites": ", ".join(nuevos) if nuevos else "ninguno",
        })
        if not nuevos:
            break
        excluidos.extend(nuevos)
    else:
        res = calcular(tipo, largo, excluidos=excluidos, **kw)
    return res, historial


# ---------------------------------------------------------------------------
# EWMA y CUSUM (cartas con memoria, para corrimientos pequeños)
# ---------------------------------------------------------------------------

def _serie_para_memoria(largo: pd.DataFrame):
    sub = resumen_subgrupos(largo)
    if sub["n"].nunique() != 1:
        raise ValueError("EWMA y CUSUM se implementan para tamaño de subgrupo constante.")
    n = int(sub["n"].iloc[0])
    return sub, n


def carta_ewma(largo: pd.DataFrame, mu0: float, sigma: float, lam: float = 0.2, L: float = 2.86,
               racha_lado: int = RACHA_LADO_POR_DEFECTO) -> Carta:
    """EWMA: z_i = λ·x̄_i + (1-λ)·z_{i-1}, z_0 = μ0, con límites exactos que varían con i.

    μ0 y σ se toman de la Fase I (o de valores estándar). Con λ = 0.2 y L = 2.86 el ARL0 es
    aproximadamente 370, igual al de un Shewhart 3σ, y con L = 2.962 es aproximadamente 500
    (Lucas y Saccucci, 1990; verificado por simulación en el módulo de diseño).
    """
    if not (0 < lam <= 1):
        raise ValueError("λ debe estar en (0, 1].")
    sub, n = _serie_para_memoria(largo)
    x = sub["media"].to_numpy(float)
    m = len(x)
    sx = sigma / math.sqrt(n)
    z = np.empty(m)
    previo = mu0
    for i in range(m):
        previo = lam * x[i] + (1 - lam) * previo
        z[i] = previo
    t = np.arange(1, m + 1)
    ancho = L * sx * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    carta = Carta(
        clave="ewma", titulo=f"Gráfico EWMA (λ = {lam:g}, L = {L:g})", etiqueta_y="Estadístico EWMA zᵢ",
        etiquetas=sub["subgrupo"].tolist(), valores=z, lc=np.full(m, mu0),
        lcs=mu0 + ancho, lci=mu0 - ancho, fase=sub["fase"].to_numpy(),
        excluido=np.zeros(m, dtype=bool), L=L, reglas_por_defecto=(1,),
        parametros={"μ0": mu0, "σ (individual)": sigma, "n": n, "λ": lam, "L": L},
    )
    carta.notas.append("Las reglas de rachas no se aplican al EWMA: sus puntos están autocorrelacionados por construcción.")
    return evaluar(carta, (1,), racha_lado)


def carta_cusum(largo: pd.DataFrame, mu0: float, sigma: float, k: float = 0.5, h: float = 5.0) -> tuple[Carta, Carta]:
    """CUSUM tabular (Montgomery, 2020, sección 9.1):
        C⁺_i = max(0, x̄_i − (μ0 + K) + C⁺_{i−1})
        C⁻_i = max(0, (μ0 − K) − x̄_i + C⁻_{i−1})
    con K = k·σ_x̄ y H = h·σ_x̄. Señal cuando C⁺ > H o C⁻ > H.
    Con k = 0.5 y h = 5 el ARL0 bilateral es ≈ 465 y detecta corrimientos de 1σ en ≈ 10 muestras.
    """
    sub, n = _serie_para_memoria(largo)
    x = sub["media"].to_numpy(float)
    m = len(x)
    sx = sigma / math.sqrt(n)
    Kv, Hv = k * sx, h * sx
    cp, cm = np.zeros(m), np.zeros(m)
    sup, inf = 0.0, 0.0
    for i in range(m):
        sup = max(0.0, x[i] - (mu0 + Kv) + sup)
        inf = max(0.0, (mu0 - Kv) - x[i] + inf)
        cp[i], cm[i] = sup, inf
    comunes = dict(etiquetas=sub["subgrupo"].tolist(), fase=sub["fase"].to_numpy(),
                   excluido=np.zeros(m, dtype=bool), L=h, reglas_por_defecto=(1,),
                   parametros={"μ0": mu0, "σ (individual)": sigma, "n": n, "k": k, "h": h,
                               "K": Kv, "H": Hv})
    c_sup = Carta(clave="cusum_sup", titulo="CUSUM superior C⁺", etiqueta_y="C⁺", valores=cp,
                  lc=np.zeros(m), lcs=np.full(m, Hv), lci=np.full(m, np.nan), **comunes)
    c_inf = Carta(clave="cusum_inf", titulo="CUSUM inferior C⁻", etiqueta_y="C⁻", valores=cm,
                  lc=np.zeros(m), lcs=np.full(m, Hv), lci=np.full(m, np.nan), **comunes)
    evaluar(c_sup, (1,))
    evaluar(c_inf, (1,))
    return c_sup, c_inf


def interpretar(res: ResultadoVariables) -> list[str]:
    """Interpretación automática (preliminar) de un par de cartas por variables."""
    textos = []
    nombre = NOMBRES[res.tipo]
    for fase in ("I", "II"):
        hay_fase = any((c.fase == fase).any() for c in res.cartas())
        if not hay_fase:
            continue
        etiqueta_fase = "Fase I (estimación de límites)" if fase == "I" else "Fase II (monitoreo)"
        # Los subgrupos ya excluidos por causa asignable siguen fuera de los límites revisados, como se
        # espera: se informan aparte y no cuentan como señales pendientes (igual que en el módulo 5).
        por_carta, excl_senal = {}, set()
        for c in res.cartas():
            for s in c.senales:
                if s.fase != fase:
                    continue
                if c.excluido[s.posicion]:
                    excl_senal.add(s.etiqueta)
                else:
                    por_carta.setdefault(c.titulo, []).append(s)
        txt_excl = ", ".join(sorted(excl_senal, key=_clave_orden))
        if not por_carta:
            if excl_senal:
                textos.append(f"{etiqueta_fase}: sin contar los subgrupos excluidos por causa asignable ({txt_excl}), "
                              f"en las cartas {nombre} no hay puntos fuera de los límites ni patrones no aleatorios "
                              f"con las reglas aplicadas; con los límites revisados el proceso se comporta como "
                              f"un sistema estable de causas comunes.")
            else:
                textos.append(f"{etiqueta_fase}: en las cartas {nombre} no hay puntos fuera de los límites ni "
                              f"patrones no aleatorios con las reglas aplicadas; el proceso se comporta como "
                              f"un sistema estable de causas comunes en este periodo.")
            continue
        partes = []
        for titulo, ss in por_carta.items():
            reg1 = sorted({s.etiqueta for s in ss if s.regla == 1}, key=_clave_orden)
            otras = sorted({(s.regla, s.etiqueta) for s in ss if s.regla != 1}, key=lambda t: (t[0], _clave_orden(t[1])))
            frase = titulo
            if reg1:
                frase += f": {len(reg1)} punto(s) fuera de límites (subgrupos {', '.join(reg1)})"
            if otras:
                reglas_txt = sorted({r for r, _ in otras})
                frase += ("; " if reg1 else ": ") + f"patrones no aleatorios (reglas {', '.join(map(str, reglas_txt))})"
            partes.append(frase)
        nota = (f" Los subgrupos excluidos por causa asignable ({txt_excl}) no se cuentan como señales."
                if excl_senal else "")
        accion = (" Investigue las causas asignables antes de usar estos límites para monitorear." if fase == "I"
                  else " Investigue la causa asignable y corrija el proceso.")
        textos.append(f"{etiqueta_fase}: el proceso NO está en control estadístico. " + ". ".join(partes) + "." +
                      nota + accion)
    if res.excluidos:
        textos.append(f"Subgrupos excluidos del cálculo de límites por causa asignable: {', '.join(res.excluidos)}.")
    textos.append(f"Estimación de la variación dentro de subgrupos: {res.metodo_sigma} → σ̂ = {res.sigma:.5g}.")
    return textos


def _clave_orden(etiqueta: str):
    try:
        return (0, float(etiqueta))
    except ValueError:
        return (1, etiqueta)
