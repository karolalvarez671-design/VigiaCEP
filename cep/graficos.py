"""
Figuras interactivas (Plotly) para la aplicación y los reportes.

Convenciones visuales:
  * datos en azul, línea de 2 px con marcadores;
  * límites de control en rojo discontinuo (son umbrales), línea central en gris oscuro;
  * zonas de ±1σ y ±2σ como líneas finas grises (referencia para las reglas de patrones);
  * punto fuera de límites: rombo rojo; patrón no aleatorio: triángulo naranja;
    punto excluido por causa asignable: círculo hueco gris;
  * línea vertical que separa la Fase I de la Fase II.
Todas las figuras se pueden descargar como PNG desde el ícono de cámara de la barra de Plotly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from .modelos import Carta

# Paleta (superficie clara)
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AZUL = SERIES[0]
CRITICO = "#d03b3b"
SERIO = "#ec835a"
BIEN = "#0ca30c"
AZUL_CLARO = "#9ec5f4"
AZUL_OSCURO = "#184f95"
SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FUENTE = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def _fmt(v, dec=None):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if dec is not None:
        return f"{v:.{dec}f}"
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 1000:
        return f"{v:,.1f}"
    if av >= 1:
        return f"{v:.4g}" if av < 100 else f"{v:.2f}"
    return f"{v:.4g}"


def estilo(fig: go.Figure, titulo: str | None = None, alto: int = 380, leyenda: bool = True,
           margen_derecho: int = 30) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=alto,
        font=dict(family=FUENTE, size=13, color=TINTA_2),
        title=dict(text=titulo, font=dict(size=15, color=TINTA), x=0, xanchor="left") if titulo else None,
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        margin=dict(l=60, r=margen_derecho, t=70 if titulo else 40, b=50),
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font=dict(family=FUENTE, size=12, color=TINTA)),
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12, color=TINTA_2), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, linecolor=EJE, tickcolor=EJE, ticks="outside", zeroline=False,
                     tickfont=dict(color=TENUE))
    fig.update_yaxes(gridcolor=REJILLA, gridwidth=1, linecolor=EJE, zeroline=False,
                     tickfont=dict(color=TENUE))
    return fig


# ---------------------------------------------------------------------------
# Gráficos de control
# ---------------------------------------------------------------------------

def decimales_para(valores) -> int:
    """Decimales suficientes para distinguir valores dentro de la amplitud dada (p. ej. 74.0146 vs 73.9878)."""
    v = np.asarray(valores, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 3
    amplitud = float(v.max() - v.min()) or abs(float(v.max())) or 1.0
    return int(min(8, max(0, 2 - np.floor(np.log10(amplitud)))))


def _agregar_carta(fig, c: Carta, fila=None, mostrar_leyenda=True, zonas=True, decimales=None):
    m = c.m
    if decimales is None:
        decimales = decimales_para(np.concatenate([c.lcs, c.lci, c.lc]))
    x = np.arange(1, m + 1)
    kw = dict(row=fila, col=1) if fila else {}
    variable = not (np.allclose(c.lcs[np.isfinite(c.lcs)], c.lcs[np.isfinite(c.lcs)][0])
                    if np.isfinite(c.lcs).any() else True)
    forma = "hvh" if variable else "linear"

    sigma = c.sigma_puntos()
    if zonas and np.isfinite(sigma).any():
        for k in (1, 2):
            for signo in (1, -1):
                y = c.lc + signo * k * sigma
                if signo < 0 and np.isfinite(c.lci).any():
                    y = np.where(y < np.nan_to_num(c.lci, nan=-np.inf), np.nan, y)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=REJILLA, width=1, shape=forma),
                                         hoverinfo="skip", showlegend=False), **kw)

    fig.add_trace(go.Scatter(x=x, y=c.lcs, mode="lines", name="Límites de control",
                             legendgroup="limites", showlegend=mostrar_leyenda,
                             line=dict(color=CRITICO, width=1.5, dash="dash", shape=forma), hoverinfo="skip"), **kw)
    if np.isfinite(c.lci).any():
        fig.add_trace(go.Scatter(x=x, y=c.lci, mode="lines", legendgroup="limites", showlegend=False,
                                 line=dict(color=CRITICO, width=1.5, dash="dash", shape=forma), hoverinfo="skip"), **kw)
    fig.add_trace(go.Scatter(x=x, y=c.lc, mode="lines", name="Línea central", legendgroup="lc",
                             showlegend=mostrar_leyenda, line=dict(color=TINTA_2, width=1.5, shape=forma),
                             hoverinfo="skip"), **kw)

    reglas_por_punto = {}
    for s in c.senales:
        reglas_por_punto.setdefault(s.posicion, []).append(s.regla)
    texto_reglas = [", ".join(map(str, sorted(reglas_por_punto.get(i, [])))) or "ninguna" for i in range(m)]
    datos_hover = np.column_stack([
        np.array(c.etiquetas, dtype=object), c.fase,
        np.round(c.lci.astype(float), 6), np.round(c.lcs.astype(float), 6), texto_reglas,
        np.where(c.excluido, "sí", "no"),
    ])
    plantilla = ("<b>Subgrupo %{customdata[0]}</b> (Fase %{customdata[1]})<br>" + c.etiqueta_y +
                 ": %{y:.5g}<br>LCI: %{customdata[2]} · LCS: %{customdata[3]}<br>Reglas violadas: %{customdata[4]}"
                 "<br>Excluido: %{customdata[5]}<extra></extra>")
    fig.add_trace(go.Scatter(
        x=x, y=c.valores, mode="lines+markers", name="Datos", legendgroup="datos", showlegend=mostrar_leyenda,
        line=dict(color=AZUL, width=2), marker=dict(color=AZUL, size=7, line=dict(color=SUPERFICIE, width=1.5)),
        customdata=datos_hover, hovertemplate=plantilla, connectgaps=False), **kw)

    fuera = c.fuera_de_limites()
    otras = np.array([any(r != 1 for r in reglas_por_punto.get(i, [])) for i in range(m)]) & ~fuera
    if c.excluido.any():
        fig.add_trace(go.Scatter(x=x[c.excluido], y=c.valores[c.excluido], mode="markers",
                                 name="Excluido (causa asignable)", legendgroup="excl", showlegend=mostrar_leyenda,
                                 marker=dict(symbol="circle-open", size=15, color=TENUE, line=dict(width=2)),
                                 hoverinfo="skip"), **kw)
    if otras.any():
        fig.add_trace(go.Scatter(x=x[otras], y=c.valores[otras], mode="markers", name="Patrón no aleatorio",
                                 legendgroup="patron", showlegend=mostrar_leyenda,
                                 marker=dict(symbol="triangle-up", size=12, color=SERIO,
                                             line=dict(color=SUPERFICIE, width=1.5)),
                                 customdata=datos_hover[otras], hovertemplate=plantilla), **kw)
    if fuera.any():
        fig.add_trace(go.Scatter(x=x[fuera], y=c.valores[fuera], mode="markers", name="Fuera de límites",
                                 legendgroup="fuera", showlegend=mostrar_leyenda,
                                 marker=dict(symbol="diamond", size=12, color=CRITICO,
                                             line=dict(color=SUPERFICIE, width=1.5)),
                                 customdata=datos_hover[fuera], hovertemplate=plantilla), **kw)

    # Etiquetas directas de los límites (solo si son constantes)
    yref = "y" if not fila or fila == 1 else f"y{fila}"
    if not variable:
        etiquetas = [("LCS", c.lcs[0]), ("LC", c.lc[0])]
        if np.isfinite(c.lci).any():
            etiquetas.append(("LCI", c.lci[0]))
        for nombre, valor in etiquetas:
            fig.add_annotation(x=1.0, xref="paper", y=valor, yref=yref, xanchor="left", showarrow=False,
                               text=f" {nombre} = {_fmt(valor, decimales)}",
                               font=dict(size=11, color=TINTA_2))

    # División Fase I / Fase II
    idx2 = np.where(c.fase == "II")[0]
    if idx2.size and idx2[0] > 0:
        xd = idx2[0] + 0.5
        fig.add_vline(x=xd, line=dict(color=TENUE, width=1), **({"row": fila, "col": 1} if fila else {}))
        if not fila or fila == 1:
            fig.add_annotation(x=xd, y=1.0, xref="x", yref="paper", text="Fase I ◂ ▸ Fase II", showarrow=False,
                               yanchor="bottom", font=dict(size=11, color=TENUE))

    ejex = dict(tickmode="array", tickvals=x, ticktext=[str(e) for e in c.etiquetas]) if m <= 40 else {}
    if fila:
        fig.update_xaxes(**ejex, row=fila, col=1)
        fig.update_yaxes(title_text=c.etiqueta_y, row=fila, col=1)
    else:
        fig.update_xaxes(**ejex)
        fig.update_yaxes(title_text=c.etiqueta_y)


def fig_carta(c: Carta, zonas: bool = True, decimales=None, alto: int = 380, titulo: str | None = None) -> go.Figure:
    fig = go.Figure()
    _agregar_carta(fig, c, None, True, zonas, decimales)
    estilo(fig, c.titulo if titulo is None else titulo, alto, margen_derecho=120)
    fig.update_xaxes(title_text="Subgrupo / muestra (orden en el tiempo)")
    return fig


def fig_par_cartas(c1: Carta, c2: Carta, titulo: str | None = None, zonas: bool = True, decimales=None,
                   alto: int = 640) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=(c1.titulo, c2.titulo))
    _agregar_carta(fig, c1, 1, True, zonas, decimales)
    _agregar_carta(fig, c2, 2, False, False, decimales)
    estilo(fig, titulo, alto, margen_derecho=120)
    # La leyenda (hasta 6 elementos, dos filas) va por encima de los títulos de los paneles
    fig.update_layout(legend=dict(y=1.09), margin=dict(t=120 if titulo else 95))
    fig.update_xaxes(title_text="Subgrupo (orden en el tiempo)", row=2, col=1)
    for anot in fig.layout.annotations:
        if anot.text in (c1.titulo, c2.titulo):
            anot.update(font=dict(size=13, color=TINTA), x=0, xanchor="left")
    return fig


def fig_cusum(c_sup: Carta, c_inf: Carta, alto: int = 400, titulo: str | None = None) -> go.Figure:
    x = np.arange(1, c_sup.m + 1)
    H = c_sup.lcs[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=c_sup.valores, mode="lines+markers", name="C⁺ (corrimiento hacia arriba)",
                             line=dict(color=SERIES[0], width=2), marker=dict(size=7, line=dict(color=SUPERFICIE, width=1.5))))
    fig.add_trace(go.Scatter(x=x, y=-c_inf.valores, mode="lines+markers", name="−C⁻ (corrimiento hacia abajo)",
                             line=dict(color=SERIES[1], width=2), marker=dict(size=7, line=dict(color=SUPERFICIE, width=1.5))))
    for y, nombre in ((H, "H"), (-H, "−H")):
        fig.add_hline(y=y, line=dict(color=CRITICO, width=1.5, dash="dash"))
        fig.add_annotation(x=1.0, xref="paper", y=y, xanchor="left", showarrow=False, text=f" {nombre} = {_fmt(y)}",
                           font=dict(size=11, color=TINTA_2))
    fig.add_hline(y=0, line=dict(color=TINTA_2, width=1.2))
    senal = (c_sup.valores > H) | (c_inf.valores > H)
    if senal.any():
        ys = np.where(c_sup.valores > H, c_sup.valores, -c_inf.valores)
        fig.add_trace(go.Scatter(x=x[senal], y=ys[senal], mode="markers", name="Señal",
                                 marker=dict(symbol="diamond", size=12, color=CRITICO, line=dict(color=SUPERFICIE, width=1.5))))
    idx2 = np.where(c_sup.fase == "II")[0]
    if idx2.size and idx2[0] > 0:
        fig.add_vline(x=idx2[0] + 0.5, line=dict(color=TENUE, width=1))
    estilo(fig, f"CUSUM tabular (k = {c_sup.parametros['k']:g}, h = {c_sup.parametros['h']:g})" if titulo is None else titulo,
           alto, margen_derecho=110)
    fig.update_xaxes(title_text="Subgrupo")
    fig.update_yaxes(title_text="Suma acumulada")
    return fig


# ---------------------------------------------------------------------------
# Datos y supuestos
# ---------------------------------------------------------------------------

def fig_corrida(largo: pd.DataFrame, alto: int = 320) -> go.Figure:
    """Gráfico de corrida de todas las mediciones en orden, coloreado por fase."""
    d = largo[largo["valor"].notna()].sort_values(["orden"], kind="stable").reset_index(drop=True)
    d["pos"] = np.arange(1, len(d) + 1)
    fig = go.Figure()
    for i, (f, nombre) in enumerate((("I", "Fase I"), ("II", "Fase II"))):
        sel = d[d["fase"] == f]
        if sel.empty:
            continue
        fig.add_trace(go.Scatter(x=sel["pos"], y=sel["valor"], mode="markers", name=nombre,
                                 marker=dict(color=SERIES[i], size=6, line=dict(color=SUPERFICIE, width=1)),
                                 customdata=sel[["subgrupo"]].to_numpy(),
                                 hovertemplate="Subgrupo %{customdata[0]}<br>Valor: %{y:.5g}<extra></extra>"))
    estilo(fig, "Mediciones en orden de recolección", alto, leyenda=d["fase"].nunique() > 1)
    fig.update_xaxes(title_text="Medición n.º")
    fig.update_yaxes(title_text="Valor")
    return fig


def fig_histograma(x, titulo="Distribución de los datos", alto=340) -> go.Figure:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    fig = go.Figure(go.Histogram(x=x, marker=dict(color=AZUL_CLARO, line=dict(color=SUPERFICIE, width=2)),
                                 name="Frecuencia", hovertemplate="%{x}<br>Frecuencia: %{y}<extra></extra>"))
    if len(x) > 2:
        mu, s = x.mean(), x.std(ddof=1)
        xs = np.linspace(x.min() - s, x.max() + s, 300)
        ancho = (x.max() - x.min()) / max(1, int(np.ceil(np.sqrt(len(x)))))
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, mu, s) * len(x) * ancho, mode="lines",
                                 name="Normal ajustada", line=dict(color=AZUL_OSCURO, width=2), hoverinfo="skip"))
        fig.update_traces(selector=dict(type="histogram"), xbins=dict(size=ancho) if ancho > 0 else None)
    estilo(fig, titulo, alto, leyenda=False)
    fig.update_layout(bargap=0)
    fig.update_xaxes(title_text="Valor")
    fig.update_yaxes(title_text="Frecuencia")
    return fig


def fig_qq(qq: dict, alto=360, titulo: str | None = None) -> go.Figure:
    t, o = qq["teoricos"], qq["observados"]
    fig = go.Figure()
    linea_x = np.array([t.min(), t.max()])
    fig.add_trace(go.Scatter(x=linea_x, y=qq["intercepto"] + qq["pendiente"] * linea_x, mode="lines",
                             name="Referencia normal", line=dict(color=TINTA_2, width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=o, mode="markers", name="Datos",
                             marker=dict(color=AZUL, size=7, line=dict(color=SUPERFICIE, width=1.5)),
                             hovertemplate="Cuantil teórico: %{x:.3f}<br>Dato: %{y:.5g}<extra></extra>"))
    estilo(fig, "Gráfico de probabilidad normal" if titulo is None else titulo, alto, leyenda=False)
    fig.update_xaxes(title_text="Cuantil normal estándar teórico")
    fig.update_yaxes(title_text="Dato observado")
    return fig


def fig_acf(acf: pd.DataFrame, alto=320, titulo: str | None = None) -> go.Figure:
    banda = float(acf["Banda ±"].iloc[0])
    colores = np.where(acf["r_k"].abs() > banda, SERIO, AZUL)
    fig = go.Figure(go.Bar(x=acf["Rezago"], y=acf["r_k"], marker=dict(color=colores), width=0.35,
                           hovertemplate="Rezago %{x}<br>r = %{y:.3f}<extra></extra>"))
    for y in (banda, -banda):
        fig.add_hline(y=y, line=dict(color=CRITICO, width=1.2, dash="dash"))
    fig.add_hline(y=0, line=dict(color=EJE, width=1))
    estilo(fig, "Función de autocorrelación (bandas ±1.96/√N)" if titulo is None else titulo, alto, leyenda=False)
    fig.update_xaxes(title_text="Rezago k", dtick=1)
    fig.update_yaxes(title_text="Autocorrelación r_k", range=[-1, 1])
    return fig


# ---------------------------------------------------------------------------
# Capacidad
# ---------------------------------------------------------------------------

def fig_capacidad(x, media, sigma_w, sigma_t, lie=None, lse=None, objetivo=None, alto=420,
                  titulo="Histograma de capacidad") -> go.Figure:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    bordes = [v for v in (lie, lse) if v is not None]
    lo = min([x.min(), media - 4 * sigma_t] + bordes)
    hi = max([x.max(), media + 4 * sigma_t] + bordes)
    nbins = max(6, int(np.ceil(np.sqrt(n))) + 2)
    ancho = (x.max() - x.min()) / nbins if x.max() > x.min() else 1
    fig = go.Figure(go.Histogram(x=x, xbins=dict(start=x.min() - ancho / 2, end=x.max() + ancho, size=ancho),
                                 marker=dict(color=AZUL_CLARO, line=dict(color=SUPERFICIE, width=2)),
                                 name="Datos", hovertemplate="%{x}<br>Frecuencia: %{y}<extra></extra>"))
    xs = np.linspace(lo, hi, 400)
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, media, sigma_w) * n * ancho, mode="lines",
                             name="Normal con σ dentro (corto plazo)", line=dict(color=AZUL_OSCURO, width=2),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, media, sigma_t) * n * ancho, mode="lines",
                             name="Normal con σ total (largo plazo)", line=dict(color=SERIES[1], width=2, dash="dot"),
                             hoverinfo="skip"))
    for valor, nombre, color in ((lie, "LIE", CRITICO), (lse, "LSE", CRITICO), (objetivo, "Objetivo", TINTA_2)):
        if valor is None:
            continue
        fig.add_vline(x=valor, line=dict(color=color, width=2 if nombre != "Objetivo" else 1.5,
                                         dash="solid" if nombre != "Objetivo" else "dot"))
        fig.add_annotation(x=valor, y=1.0, yref="paper", text=f"{nombre} = {_fmt(valor)}", showarrow=False,
                           yanchor="bottom", font=dict(size=11, color=TINTA_2))
    estilo(fig, titulo, alto)
    fig.update_layout(bargap=0, legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
                      margin=dict(b=110))
    fig.update_xaxes(title_text="Valor de la característica de calidad", range=[lo, hi])
    fig.update_yaxes(title_text="Frecuencia")
    return fig


# ---------------------------------------------------------------------------
# Diseño (ARL / ATS)
# ---------------------------------------------------------------------------

def fig_curvas_arl(tabla: pd.DataFrame, columna: str = "ARL", titulo="ARL según el corrimiento de la media",
                   log=True, alto=400) -> go.Figure:
    fig = go.Figure()
    for i, (n, g) in enumerate(tabla.groupby("n")):
        fig.add_trace(go.Scatter(x=g["δ (en σ)"], y=g[columna], mode="lines+markers", name=f"n = {n}",
                                 line=dict(color=SERIES[i % len(SERIES)], width=2),
                                 marker=dict(size=7, line=dict(color=SUPERFICIE, width=1.5)),
                                 hovertemplate=f"n = {n}<br>δ = %{{x}}σ<br>{columna} = %{{y:.3f}}<extra></extra>"))
    estilo(fig, titulo, alto)
    fig.update_xaxes(title_text="Corrimiento δ (en σ del proceso)")
    if log:
        marcas = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
                  if v <= max(2.0, float(tabla[columna].max()) * 1.5)]
        fig.update_yaxes(title_text=columna, type="log", tickvals=marcas, ticktext=[f"{v:g}" for v in marcas])
    else:
        fig.update_yaxes(title_text=columna)
    return fig


def fig_curva_oc_carta(tabla: pd.DataFrame, alto=380) -> go.Figure:
    fig = go.Figure()
    for i, (n, g) in enumerate(tabla.groupby("n")):
        fig.add_trace(go.Scatter(x=g["δ (en σ)"], y=g["β"], mode="lines", name=f"n = {n}",
                                 line=dict(color=SERIES[i % len(SERIES)], width=2),
                                 hovertemplate=f"n = {n}<br>δ = %{{x}}σ<br>β = %{{y:.4f}}<extra></extra>"))
    estilo(fig, "Curva característica de operación del gráfico X̄ (β vs. δ)", alto)
    fig.update_xaxes(title_text="Corrimiento δ (en σ)")
    fig.update_yaxes(title_text="β (prob. de no detectar)", range=[0, 1.02])
    return fig


def fig_mapa_ats(tabla: pd.DataFrame, columna: str = "ATS1 (horas)", alto=420, titulo: str | None = None) -> go.Figure:
    piv = tabla.pivot(index="n", columns="h (horas)", values=columna)
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[f"h = {h:g}" for h in piv.columns], y=[f"n = {n}" for n in piv.index],
        colorscale=[[i / (len(SECUENCIAL) - 1), c] for i, c in enumerate(SECUENCIAL)],
        colorbar=dict(title=dict(text="horas")), xgap=2, ygap=2,
        text=np.vectorize(lambda v: f"{v:.2f}")(piv.values), texttemplate="%{text}",
        hovertemplate="%{y}, %{x}<br>" + columna + ": %{z:.2f}<extra></extra>"))
    estilo(fig, f"{columna} para cada diseño (n, h)" if titulo is None else titulo, alto, leyenda=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def fig_perfil_arl(tabla: pd.DataFrame, alto=400, titulo: str | None = None) -> go.Figure:
    fig = go.Figure()
    for i, (esq, g) in enumerate(tabla.groupby("Esquema", sort=False)):
        fig.add_trace(go.Scatter(x=g["δ (en σ)"], y=g["ARL simulado"], mode="lines+markers", name=esq,
                                 line=dict(color=SERIES[i % len(SERIES)], width=2),
                                 marker=dict(size=7, line=dict(color=SUPERFICIE, width=1.5)),
                                 error_y=dict(type="data", array=1.96 * g["EE"], visible=True, thickness=1, width=3),
                                 hovertemplate=f"{esq}<br>δ = %{{x}}σ<br>ARL = %{{y:.2f}}<extra></extra>"))
    estilo(fig, "Comparación de esquemas por simulación (ARL ± IC 95 %)" if titulo is None else titulo, alto)
    fig.update_xaxes(title_text="Corrimiento δ (en σ)")
    marcas = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if v <= max(2.0, float(tabla["ARL simulado"].max()) * 1.5)]
    fig.update_yaxes(title_text="ARL simulado", type="log", tickvals=marcas, ticktext=[f"{v:g}" for v in marcas])
    return fig


# ---------------------------------------------------------------------------
# Muestreo de aceptación
# ---------------------------------------------------------------------------

def fig_curva_oc_plan(tablas: dict, nca=None, ncl=None, alfa=None, beta=None, alto=420,
                      titulo="Curva característica de operación (OC)") -> go.Figure:
    fig = go.Figure()
    for i, (nombre, t) in enumerate(tablas.items()):
        fig.add_trace(go.Scatter(x=100 * t["p"], y=t["Pa"], mode="lines", name=nombre,
                                 line=dict(color=SERIES[i % len(SERIES)], width=2),
                                 hovertemplate=nombre + "<br>p = %{x:.2f} %<br>Pa = %{y:.4f}<extra></extra>"))
    puntos = []
    if nca is not None and alfa is not None:
        puntos.append((nca, 1 - alfa, f"NCA = {100 * nca:g} %, 1−α = {1 - alfa:g}"))
    if ncl is not None and beta is not None:
        puntos.append((ncl, beta, f"NCL = {100 * ncl:g} %, β = {beta:g}"))
    for p, pa, txt in puntos:
        fig.add_trace(go.Scatter(x=[100 * p], y=[pa], mode="markers+text", text=[txt], textposition="top right",
                                 marker=dict(symbol="x", size=11, color=TINTA), showlegend=False,
                                 textfont=dict(size=11, color=TINTA_2), hoverinfo="skip"))
    estilo(fig, titulo, alto, leyenda=len(tablas) > 1)
    fig.update_xaxes(title_text="Fracción defectuosa del lote p (%)")
    fig.update_yaxes(title_text="Probabilidad de aceptación Pa", range=[0, 1.03])
    return fig


def fig_linea_simple(x, y, titulo, x_titulo, y_titulo, alto=340, x_pct=True, marcar_max=False) -> go.Figure:
    x = np.asarray(x)
    y = np.asarray(y)
    fig = go.Figure(go.Scatter(x=100 * x if x_pct else x, y=y, mode="lines", line=dict(color=AZUL, width=2),
                               hovertemplate="%{x:.3f}<br>%{y:.4g}<extra></extra>"))
    if marcar_max and len(y):
        i = int(np.nanargmax(y))
        fig.add_trace(go.Scatter(x=[100 * x[i] if x_pct else x[i]], y=[y[i]], mode="markers+text",
                                 text=[f"máx = {y[i]:.4g}"], textposition="top center",
                                 marker=dict(size=9, color=AZUL, line=dict(color=SUPERFICIE, width=2)),
                                 textfont=dict(size=11, color=TINTA_2), hoverinfo="skip"))
    estilo(fig, titulo, alto, leyenda=False)
    fig.update_xaxes(title_text=x_titulo)
    fig.update_yaxes(title_text=y_titulo)
    return fig


def fig_pareto(tabla: pd.DataFrame, alto: int = 400, titulo: str = "Pareto de tipos de defecto") -> go.Figure:
    """Barras con el % de cada tipo y línea del % acumulado, en un solo eje de 0 a 100 %."""
    etiquetas = [t.replace("_", " ") for t in tabla["Tipo de defecto"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=etiquetas, y=tabla["%"], name="% del total de defectos", marker=dict(color=AZUL),
                         width=0.55, customdata=tabla[["Conteo"]].to_numpy(),
                         hovertemplate="%{x}<br>%{y:.1f} % (%{customdata[0]:.0f} defectos)<extra></extra>"))
    fig.add_trace(go.Scatter(x=etiquetas, y=tabla["% acumulado"], mode="lines+markers", name="% acumulado",
                             line=dict(color=SERIES[1], width=2),
                             marker=dict(size=7, color=SERIES[1], line=dict(color=SUPERFICIE, width=1.5)),
                             hovertemplate="%{x}<br>acumulado %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=80, line=dict(color=TENUE, width=1, dash="dot"))
    fig.add_annotation(x=1.0, xref="paper", y=80, xanchor="left", showarrow=False, text=" 80 %",
                       font=dict(size=11, color=TINTA_2))
    estilo(fig, titulo, alto, margen_derecho=60)
    fig.update_yaxes(title_text="Porcentaje (%)", range=[0, 105])
    fig.update_xaxes(title_text="Tipo de defecto")
    return fig
