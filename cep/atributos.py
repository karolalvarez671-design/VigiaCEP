"""
Módulo 3. Gráficos de control por atributos.

  * p   proporción de unidades defectuosas (no conformes); n constante o variable
  * np  número de unidades defectuosas; n constante
  * c   número de defectos (no conformidades) por unidad de inspección constante
  * u   defectos por unidad cuando el tamaño de la unidad de inspección varía

Modelos: binomial para defectuosos (p, np) y Poisson para defectos (c, u).
Límites a L sigmas (L = 3 por defecto); un LCI negativo se trunca en 0.
Fase I con exclusión justificada de muestras y Fase II con límites congelados o
con un valor estándar (p0, c0, u0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .modelos import Carta
from .reglas import RACHA_LADO_POR_DEFECTO, evaluar

NOMBRES = {"p": "p (proporción de defectuosos)", "np": "np (número de defectuosos)",
           "c": "c (número de defectos)", "u": "u (defectos por unidad)"}


@dataclass
class ResultadoAtributos:
    tipo: str
    carta: Carta
    centro: float              # p̄, n·p̄, c̄ o ū
    parametro: float           # p̄, c̄ o ū (el parámetro del modelo)
    n_constante: bool
    excluidos: list = field(default_factory=list)
    estandar: float | None = None
    notas: list = field(default_factory=list)

    def senales_fase(self, fase):
        return [s for s in self.carta.senales if s.fase == fase]


def recomendar_carta(tipo_dato: str, n_constante: bool) -> tuple[str, str]:
    """Selección justificada del gráfico por atributos.

    tipo_dato: 'defectuosos' (cada unidad se clasifica conforme / no conforme) o
               'defectos' (se cuentan las no conformidades en cada unidad o área).
    """
    if tipo_dato == "defectuosos":
        if n_constante:
            return "np", ("Se clasifican unidades como conformes o no conformes (modelo binomial) y el tamaño de "
                          "muestra es constante: la carta np grafica directamente el número de defectuosos, que es "
                          "lo que el personal de planta cuenta. La carta p es equivalente.")
        return "p", ("Se clasifican unidades como conformes o no conformes (modelo binomial) y el tamaño de "
                     "muestra varía: la carta p usa la proporción de defectuosos con límites que cambian con nᵢ.")
    if n_constante:
        return "c", ("Se cuentan no conformidades (una unidad puede tener varios defectos; modelo de Poisson) "
                     "y la unidad de inspección es constante: corresponde la carta c.")
    return "u", ("Se cuentan no conformidades (modelo de Poisson) pero el tamaño de la unidad de inspección "
                 "varía: la carta u normaliza a defectos por unidad, con límites que cambian con nᵢ.")


def _excl(etiquetas, excluidos):
    ex = {str(e) for e in (excluidos or [])}
    return np.array([str(e) in ex for e in etiquetas], dtype=bool)


def _base(tabla: pd.DataFrame, excluidos):
    etiquetas = tabla["muestra"].tolist()
    fase = tabla["fase"].to_numpy()
    excl = _excl(etiquetas, excluidos)
    usa = (fase == "I") & ~excl
    return etiquetas, fase, excl, usa


def carta_p(tabla: pd.DataFrame, L: float = 3.0, excluidos=(), p0: float | None = None,
            reglas=None, racha_lado: int = RACHA_LADO_POR_DEFECTO) -> ResultadoAtributos:
    etiquetas, fase, excl, usa = _base(tabla, excluidos)
    n = tabla["n"].to_numpy(float)
    d = tabla["conteo"].to_numpy(float)
    if (d > n).any():
        raise ValueError("Hay muestras con más defectuosos que unidades inspeccionadas; revise los datos "
                         "(si cuenta defectos y no unidades defectuosas, use la carta c o u).")
    if p0 is None:
        if usa.sum() < 2:
            raise ValueError("Se necesitan al menos 2 muestras de Fase I no excluidas.")
        pbar = float(d[usa].sum() / n[usa].sum())
    else:
        pbar = float(p0)
    if not (0 < pbar < 1):
        raise ValueError("p̄ debe estar entre 0 y 1 (no hay defectuosos o todo es defectuoso en la Fase I).")
    sig = np.sqrt(pbar * (1 - pbar) / n)
    carta = Carta(
        clave="p", titulo="Gráfico p (proporción de defectuosos)", etiqueta_y="Proporción defectuosa pᵢ",
        etiquetas=etiquetas, valores=d / n, lc=np.full(len(n), pbar),
        lcs=pbar + L * sig, lci=np.maximum(0.0, pbar - L * sig),
        fase=fase, excluido=excl, L=L, parametros={"p̄": pbar, "L": L},
    )
    evaluar(carta, reglas, racha_lado)
    constante = len(np.unique(n)) == 1
    notas = _notas_binomial(pbar, n, L)
    return ResultadoAtributos("p", carta, pbar, pbar, constante, _lista(etiquetas, excl), p0, notas)


def carta_np(tabla: pd.DataFrame, L: float = 3.0, excluidos=(), p0: float | None = None,
             reglas=None, racha_lado: int = RACHA_LADO_POR_DEFECTO) -> ResultadoAtributos:
    n_vals = tabla["n"].to_numpy(float)
    if len(np.unique(n_vals)) != 1:
        raise ValueError("La carta np requiere tamaño de muestra constante; use la carta p.")
    etiquetas, fase, excl, usa = _base(tabla, excluidos)
    n = float(n_vals[0])
    d = tabla["conteo"].to_numpy(float)
    if (d > n).any():
        raise ValueError("Hay muestras con más defectuosos que unidades inspeccionadas.")
    pbar = float(p0) if p0 is not None else float(d[usa].sum() / (n * usa.sum()))
    if not (0 < pbar < 1):
        raise ValueError("p̄ debe estar entre 0 y 1.")
    centro = n * pbar
    s = math.sqrt(n * pbar * (1 - pbar))
    m = len(d)
    carta = Carta(
        clave="np", titulo="Gráfico np (número de defectuosos)", etiqueta_y="Defectuosos por muestra",
        etiquetas=etiquetas, valores=d, lc=np.full(m, centro), lcs=np.full(m, centro + L * s),
        lci=np.full(m, max(0.0, centro - L * s)), fase=fase, excluido=excl, L=L,
        parametros={"p̄": pbar, "n": n, "n·p̄": centro, "L": L},
    )
    evaluar(carta, reglas, racha_lado)
    return ResultadoAtributos("np", carta, centro, pbar, True, _lista(etiquetas, excl), p0,
                              _notas_binomial(pbar, np.array([n]), L))


def carta_c(tabla: pd.DataFrame, L: float = 3.0, excluidos=(), c0: float | None = None,
            reglas=None, racha_lado: int = RACHA_LADO_POR_DEFECTO) -> ResultadoAtributos:
    etiquetas, fase, excl, usa = _base(tabla, excluidos)
    c = tabla["conteo"].to_numpy(float)
    n_vals = tabla["n"].to_numpy(float)
    notas = []
    if len(np.unique(n_vals)) != 1:
        notas.append("El tamaño de la unidad de inspección varía; la carta u es la adecuada.")
    cbar = float(c0) if c0 is not None else float(c[usa].mean())
    if cbar <= 0:
        raise ValueError("c̄ debe ser mayor que 0.")
    s = math.sqrt(cbar)
    m = len(c)
    carta = Carta(
        clave="c", titulo="Gráfico c (número de defectos)", etiqueta_y="Defectos por unidad de inspección",
        etiquetas=etiquetas, valores=c, lc=np.full(m, cbar), lcs=np.full(m, cbar + L * s),
        lci=np.full(m, max(0.0, cbar - L * s)), fase=fase, excluido=excl, L=L,
        parametros={"c̄": cbar, "L": L},
    )
    evaluar(carta, reglas, racha_lado)
    if cbar - L * s <= 0:
        notas.append(f"LCI = 0 porque c̄ = {cbar:.3g} ≤ L² = {L*L:g}: la carta no puede señalar mejoras "
                     "(disminución de defectos).")
    return ResultadoAtributos("c", carta, cbar, cbar, len(np.unique(n_vals)) == 1, _lista(etiquetas, excl), c0, notas)


def carta_u(tabla: pd.DataFrame, L: float = 3.0, excluidos=(), u0: float | None = None,
            reglas=None, racha_lado: int = RACHA_LADO_POR_DEFECTO) -> ResultadoAtributos:
    etiquetas, fase, excl, usa = _base(tabla, excluidos)
    c = tabla["conteo"].to_numpy(float)
    n = tabla["n"].to_numpy(float)
    ubar = float(u0) if u0 is not None else float(c[usa].sum() / n[usa].sum())
    if ubar <= 0:
        raise ValueError("ū debe ser mayor que 0.")
    sig = np.sqrt(ubar / n)
    carta = Carta(
        clave="u", titulo="Gráfico u (defectos por unidad)", etiqueta_y="Defectos por unidad uᵢ",
        etiquetas=etiquetas, valores=c / n, lc=np.full(len(n), ubar), lcs=ubar + L * sig,
        lci=np.maximum(0.0, ubar - L * sig), fase=fase, excluido=excl, L=L,
        parametros={"ū": ubar, "L": L},
    )
    evaluar(carta, reglas, racha_lado)
    return ResultadoAtributos("u", carta, ubar, ubar, len(np.unique(n)) == 1, _lista(etiquetas, excl), u0, [])


def _lista(etiquetas, excl):
    return [str(e) for e, x in zip(etiquetas, excl) if x]


def _notas_binomial(pbar: float, n: np.ndarray, L: float) -> list[str]:
    notas = []
    n_min = float(np.min(n))
    if n_min * pbar < 5:
        notas.append(f"n·p̄ = {n_min * pbar:.2f} < 5: la aproximación normal de los límites es pobre; "
                     "interprete con cuidado o aumente el tamaño de muestra.")
    n_lci = (1 - pbar) * L ** 2 / pbar
    if np.any(pbar - L * np.sqrt(pbar * (1 - pbar) / n) <= 0):
        notas.append(f"LCI = 0: para tener LCI > 0 (y poder detectar mejoras) se requiere n > {n_lci:.0f} "
                     f"(n > (1−p̄)L²/p̄).")
    return notas


FUNCIONES = {"p": carta_p, "np": carta_np, "c": carta_c, "u": carta_u}


def calcular(tipo: str, tabla: pd.DataFrame, L: float = 3.0, excluidos=(), estandar: float | None = None,
             reglas=None, racha_lado: int = RACHA_LADO_POR_DEFECTO) -> ResultadoAtributos:
    f = FUNCIONES[tipo]
    clave_estandar = {"p": "p0", "np": "p0", "c": "c0", "u": "u0"}[tipo]
    return f(tabla, L=L, excluidos=excluidos, reglas=reglas, racha_lado=racha_lado,
             **{clave_estandar: estandar})


def fase1_iterativa(tipo: str, tabla: pd.DataFrame, excluidos_iniciales=(), max_iter: int = 10, **kw):
    excluidos = [str(e) for e in excluidos_iniciales]
    historial = []
    for it in range(1, max_iter + 1):
        res = calcular(tipo, tabla, excluidos=excluidos, **kw)
        c = res.carta
        fuera = c.fuera_de_limites()
        nuevos = [str(c.etiquetas[i]) for i in np.where(fuera & (c.fase == "I") & ~c.excluido)[0]]
        historial.append({"Iteración": it, "Línea central": res.centro,
                          "Muestras fuera de límites": ", ".join(nuevos) if nuevos else "ninguna"})
        if not nuevos:
            break
        excluidos.extend(nuevos)
    else:
        res = calcular(tipo, tabla, excluidos=excluidos, **kw)
    return res, historial


def tamano_para_detectar(pbar: float, delta: float, L: float = 3.0) -> float:
    """n para que un corrimiento de p̄ a p̄+δ se detecte con probabilidad 0.5 (Montgomery, ec. 7.10)."""
    return (L / delta) ** 2 * pbar * (1 - pbar)


def interpretar(res: ResultadoAtributos) -> list[str]:
    textos = []
    c = res.carta
    for fase in ("I", "II"):
        if not (c.fase == fase).any():
            continue
        etiqueta_fase = "Fase I (estimación de límites)" if fase == "I" else "Fase II (monitoreo)"
        # Las muestras ya excluidas por causa asignable no cuentan como señales pendientes.
        todas = res.senales_fase(fase)
        ss = [s for s in todas if not c.excluido[s.posicion]]
        txt_excl = ", ".join(e for _, e in sorted({(s.posicion, s.etiqueta) for s in todas if c.excluido[s.posicion]}))
        if not ss:
            if txt_excl:
                textos.append(f"{etiqueta_fase}: sin contar las muestras excluidas por causa asignable ({txt_excl}), "
                              f"la carta {res.tipo} no muestra puntos fuera de límites ni patrones no aleatorios; "
                              "con los límites revisados la fracción de no conformidades es estable en el periodo.")
            else:
                textos.append(f"{etiqueta_fase}: la carta {res.tipo} no muestra puntos fuera de límites ni patrones "
                              "no aleatorios; la fracción de no conformidades es estable en el periodo.")
            continue
        reg1_altos = [s.etiqueta for s in ss if s.regla == 1 and c.valores[s.posicion] > c.lcs[s.posicion]]
        reg1_bajos = [s.etiqueta for s in ss if s.regla == 1 and c.valores[s.posicion] < c.lci[s.posicion]]
        otras = sorted({s.regla for s in ss if s.regla != 1})
        partes = []
        if reg1_altos:
            partes.append(f"muestras por encima del LCS: {', '.join(reg1_altos)} (deterioro: investigar causa)")
        if reg1_bajos:
            partes.append(f"muestras por debajo del LCI: {', '.join(reg1_bajos)} (mejora real o error de inspección)")
        if otras:
            partes.append(f"patrones no aleatorios (reglas {', '.join(map(str, otras))})")
        nota = f" Las muestras excluidas por causa asignable ({txt_excl}) no se cuentan como señales." if txt_excl else ""
        textos.append(f"{etiqueta_fase}: el proceso NO está en control. " + "; ".join(partes) + "." + nota)
    nombre_param = {"p": "p̄", "np": "p̄", "c": "c̄", "u": "ū"}[res.tipo]
    textos.append(f"Parámetro estimado: {nombre_param} = {res.parametro:.5g}"
                  + (f" (equivale a {100 * res.parametro:.2f} % defectuoso)." if res.tipo in ("p", "np") else "."))
    textos.extend(res.notas)
    return textos


def pareto(conteos) -> pd.DataFrame:
    """Tabla de Pareto: tipos de defecto ordenados de mayor a menor con % y % acumulado.

    conteos: Series o dict {tipo de defecto: total}. Sirve para priorizar las mejoras
    (principio de Pareto: pocos tipos explican la mayoría de los defectos).
    """
    serie = pd.Series(conteos, dtype=float)
    serie = serie[serie > 0].sort_values(ascending=False)
    total = float(serie.sum())
    if total == 0:
        return pd.DataFrame(columns=["Tipo de defecto", "Conteo", "%", "% acumulado"])
    tabla = pd.DataFrame({"Tipo de defecto": serie.index.astype(str), "Conteo": serie.values})
    tabla["%"] = 100 * tabla["Conteo"] / total
    tabla["% acumulado"] = tabla["%"].cumsum()
    return tabla.reset_index(drop=True)


_NO_TIPO_DEFECTO = ("total", "inspecc", "muestra", "subgrupo", "fase", "defectuosos", "defectuosas")


def sugerir_tipos_defecto(df: pd.DataFrame, usadas=()) -> tuple[list[str], list[str]]:
    """Columnas candidatas a tipos de defecto para el Pareto y las que se sugieren de entrada.

    Candidatas: columnas numéricas que no usa la carta. Se sugieren todas menos las de totales,
    unidades inspeccionadas, unidades defectuosas («defectuosos/defectuosas», en plural), muestra,
    subgrupo y fase. Un tipo de defecto como «pintura_defectuosa» sí se sugiere.
    """
    usadas = {u for u in usadas if u}
    candidatas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in usadas]
    sugeridas = [c for c in candidatas
                 if not any(k in str(c).lower() for k in _NO_TIPO_DEFECTO) and str(c).lower() != "n"]
    return candidatas, sugeridas
