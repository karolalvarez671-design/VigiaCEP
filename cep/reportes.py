"""
Módulo 8. Reportes.

Genera, a partir de los análisis realizados en la sesión:
  * un libro de Excel con todas las tablas (datos, límites, señales, capacidad, ARL, muestreo);
  * un reporte HTML autocontenido con figuras interactivas y conclusiones preliminares
    (se puede abrir sin internet e imprimir a PDF desde el navegador);
  * un archivo JSON con los límites de la Fase I para reutilizarlos en el monitoreo (Fase II).
"""

from __future__ import annotations

import datetime as dt
import html
import io
import json

import numpy as np
import pandas as pd

from . import __nombre__, __version__


def _limpiar_nombre_hoja(nombre: str, usados: set) -> str:
    base = "".join(ch for ch in nombre if ch not in '[]:*?/\\')[:31] or "Hoja"
    nombre_final, i = base, 2
    while nombre_final in usados:
        sufijo = f" ({i})"
        nombre_final = base[:31 - len(sufijo)] + sufijo
        i += 1
    usados.add(nombre_final)
    return nombre_final


def exportar_excel(tablas: dict, metadatos: dict, conclusiones: list[str] | None = None) -> bytes:
    """Libro de Excel: hoja 'Resumen' + una hoja por tabla."""
    buffer = io.BytesIO()
    usados: set = set()
    with pd.ExcelWriter(buffer, engine="openpyxl") as escritor:
        filas = [("Software", f"{__nombre__} v{__version__}"),
                 ("Fecha de generación", dt.datetime.now().strftime("%Y-%m-%d %H:%M"))]
        filas += [(k, v) for k, v in metadatos.items() if v not in (None, "")]
        resumen = pd.DataFrame(filas, columns=["Campo", "Valor"])
        if conclusiones:
            resumen = pd.concat([resumen, pd.DataFrame([("", "")], columns=["Campo", "Valor"]),
                                 pd.DataFrame([(f"Conclusión {i + 1}", c) for i, c in enumerate(conclusiones)],
                                              columns=["Campo", "Valor"])], ignore_index=True)
        resumen.to_excel(escritor, sheet_name=_limpiar_nombre_hoja("Resumen", usados), index=False)
        for nombre, tabla in tablas.items():
            if tabla is None or len(tabla) == 0:
                continue
            tabla.to_excel(escritor, sheet_name=_limpiar_nombre_hoja(nombre, usados), index=False)
        for hoja in escritor.book.worksheets:
            for col in hoja.columns:
                largo = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                hoja.column_dimensions[col[0].column_letter].width = min(60, max(10, largo + 2))
    return buffer.getvalue()


def limites_json(resultado_variables=None, resultado_atributos=None, metadatos=None) -> str:
    """Límites de Fase I en JSON (para documentar y reutilizar en Fase II)."""
    salida = {"software": f"{__nombre__} v{__version__}", "generado": dt.datetime.now().isoformat(timespec="seconds"),
              "metadatos": metadatos or {}}
    if resultado_variables is not None:
        r = resultado_variables
        salida["variables"] = {
            "tipo": r.tipo, "n": r.n, "media": r.media, "sigma_dentro": r.sigma,
            "metodo_sigma": r.metodo_sigma, "excluidos_fase_I": r.excluidos,
            "carta_posicion": r.carta_pos.resumen_limites(), "carta_dispersion": r.carta_disp.resumen_limites(),
        }
    if resultado_atributos is not None:
        a = resultado_atributos
        salida["atributos"] = {"tipo": a.tipo, "parametro": a.parametro, "centro": a.centro,
                               "excluidos_fase_I": a.excluidos, "limites": a.carta.resumen_limites()}

    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    return json.dumps(salida, ensure_ascii=False, indent=2, default=conv)


# ---------------------------------------------------------------------------
# Reporte HTML
# ---------------------------------------------------------------------------

_CSS = """
:root { --tinta:#0b0b0b; --tinta2:#52514e; --tenue:#898781; --rejilla:#e1e0d9; --sup:#fcfcfb; --azul:#2a78d6; }
* { box-sizing: border-box; }
body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: var(--tinta);
       background: #f9f9f7; margin: 0; line-height: 1.5; }
main { max-width: 1060px; margin: 0 auto; padding: 32px 28px 64px; background: var(--sup); }
header { border-bottom: 1px solid var(--rejilla); margin-bottom: 24px; padding-bottom: 16px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 20px; margin: 36px 0 8px; padding-top: 8px; border-top: 1px solid var(--rejilla); }
h3 { font-size: 16px; margin: 20px 0 6px; color: var(--tinta2); }
.meta { color: var(--tinta2); font-size: 14px; }
.meta td { padding: 2px 16px 2px 0; vertical-align: top; }
table.datos { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0 4px; }
table.datos th { text-align: left; color: var(--tinta2); border-bottom: 1px solid #c3c2b7; padding: 6px 8px; font-weight: 600; }
table.datos td { border-bottom: 1px solid var(--rejilla); padding: 5px 8px; font-variant-numeric: tabular-nums; }
.nota { font-size: 13px; color: var(--tenue); }
ul.conclusiones li { margin-bottom: 6px; }
.figura { margin: 8px 0 16px; }
.pie { margin-top: 48px; font-size: 12px; color: var(--tenue); }
@media print { body { background: white; } main { padding: 0; } h2 { break-before: auto; } .figura { break-inside: avoid; } }
"""


def _tabla_html(df: pd.DataFrame, decimales: int = 4, max_filas: int = 200) -> str:
    df = df.copy()
    recorte = len(df) > max_filas
    if recorte:
        df = df.head(max_filas)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return "—"
            if float(v).is_integer() and abs(v) < 1e12:
                return f"{int(v):,}"
            if abs(v) >= 1e5:
                return f"{v:,.0f}"
            return f"{v:.6g}"
        if v is None:
            return "—"
        return html.escape(str(v))
    cab = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    filas = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in fila) + "</tr>" for fila in df.itertuples(index=False))
    extra = f'<p class="nota">Se muestran las primeras {max_filas} filas; la tabla completa está en el Excel.</p>' if recorte else ""
    return f'<table class="datos"><thead><tr>{cab}</tr></thead><tbody>{filas}</tbody></table>{extra}'


def reporte_html(metadatos: dict, secciones: list, incluir_plotly: str = "inline") -> str:
    """Construye el reporte.

    secciones: lista de (titulo, bloques). Cada bloque es una tupla:
        ('texto', str) | ('lista', [str]) | ('tabla', DataFrame[, título]) | ('figura', go.Figure) | ('subtitulo', str)
    incluir_plotly: 'inline' (funciona sin internet) o 'cdn' (archivo más liviano).
    """
    partes = []
    primera_figura = True
    for titulo, bloques in secciones:
        partes.append(f"<h2>{html.escape(titulo)}</h2>")
        for bloque in bloques:
            tipo = bloque[0]
            if tipo == "texto":
                partes.append(f"<p>{html.escape(bloque[1])}</p>")
            elif tipo == "subtitulo":
                partes.append(f"<h3>{html.escape(bloque[1])}</h3>")
            elif tipo == "lista":
                partes.append('<ul class="conclusiones">' + "".join(f"<li>{html.escape(t)}</li>" for t in bloque[1]) + "</ul>")
            elif tipo == "tabla":
                if len(bloque) > 2 and bloque[2]:
                    partes.append(f"<h3>{html.escape(bloque[2])}</h3>")
                partes.append(_tabla_html(bloque[1]))
            elif tipo == "figura":
                fig = bloque[1]
                partes.append('<div class="figura">' + fig.to_html(
                    full_html=False, include_plotlyjs=(incluir_plotly if primera_figura else False),
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}) + "</div>")
                primera_figura = False
    meta_filas = "".join(f"<tr><td><b>{html.escape(str(k))}</b></td><td>{html.escape(str(v))}</td></tr>"
                         for k, v in metadatos.items() if v not in (None, ""))
    ahora = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte CEP — {html.escape(str(metadatos.get('Empresa', '')))}</title><style>{_CSS}</style></head>
<body><main>
<header><h1>Reporte de Control Estadístico de Procesos</h1>
<table class="meta">{meta_filas}<tr><td><b>Generado</b></td><td>{ahora} con {__nombre__} v{__version__}</td></tr></table></header>
{''.join(partes)}
<p class="pie">Las conclusiones de este reporte son preliminares y se generan automáticamente a partir de los
resultados estadísticos; deben revisarse con criterio de ingeniería y con el conocimiento del proceso.</p>
</main></body></html>"""
