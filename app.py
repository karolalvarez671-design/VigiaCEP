"""
Vigía CEP — Software de Control Estadístico de Procesos.

Interfaz gráfica (Streamlit). Los cálculos están en el paquete `cep/`, organizado
por los mismos 8 módulos de la guía del trabajo final:

    1. Gestión de datos            5. Capacidad del proceso
    2. Gráficos por variables      6. Diseño de gráficos (ARL0, ARL1, ATS)
    3. Gráficos por atributos      7. Muestreo de aceptación
    4. Validación de supuestos     8. Reportes

Ejecución:   streamlit run app.py      (o doble clic en INICIAR.bat en Windows)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from cep import __nombre__, __version__
from cep import atributos as ATR
from cep import capacidad as CAP
from cep import constantes as K
from cep import datos as D
from cep import diseno as DIS
from cep import graficos as G
from cep import muestreo as MUE
from cep import reglas as REG
from cep import reportes as REP
from cep import supuestos as SUP
from cep import variables as VAR

RAIZ = Path(__file__).resolve().parent
CARPETA_PRUEBA = RAIZ / "datos_prueba"
CARPETA_PLANTILLAS = RAIZ / "plantillas"
CASO_TALLER = "caso_taller_a/datos/Taller_A_datos.xlsx"
META_TALLER = {"Empresa": "Taller metalmecánico A (Santa Marta)",
                "Proceso / operación": "Corte del brazo horizontal en la Tronzadora 1",
                "Producto": "Soporte para unidad condensadora de aire acondicionado",
                "Característica de calidad": "Longitud del brazo horizontal (nominal 450 mm)",
                "Unidad de medida": "mm"}

st.set_page_config(page_title=f"{__nombre__} — Control Estadístico de Procesos", page_icon="📈", layout="wide")

PAGINAS = [
    "Inicio",
    "1. Gestión de datos",
    "2. Gráficos por variables",
    "3. Gráficos por atributos",
    "4. Validación de supuestos",
    "5. Capacidad del proceso",
    "6. Diseño de gráficos (ARL/ATS)",
    "7. Muestreo de aceptación",
    "8. Reportes",
    "Ayuda y glosario",
]

DATOS_PRUEBA = {
    "Caso Taller A — longitud del brazo, X̄-R": dict(
        ruta=CASO_TALLER, hoja="Corte_variables", tipo="variables", modo="ancho",
        cols=["x1", "x2", "x3", "x4", "x5"], col_sub="subgrupo", col_fase="fase", meta=META_TALLER,
        espec={"lie": 447.0, "lse": 453.0, "objetivo": 450.0}),
    "Caso Taller A — soportes defectuosos, carta p": dict(
        ruta=CASO_TALLER, hoja="Inspeccion_atributos", tipo="atributos", col_conteo="soportes_defectuosos",
        col_n="soportes_inspeccionados", col_muestra="muestra", col_fase="fase", conteo="defectuosos",
        meta=META_TALLER),
    "Caso Taller A — defectos por soporte, carta u": dict(
        ruta=CASO_TALLER, hoja="Inspeccion_atributos", tipo="atributos", col_conteo="defectos_totales",
        col_n="soportes_inspeccionados", col_muestra="muestra", col_fase="fase", conteo="defectos",
        meta=META_TALLER),
    "Anillos de pistón (Montgomery) — variables, X̄-R": dict(
        archivo="anillos_piston_montgomery.csv", tipo="variables", modo="ancho",
        cols=["x1", "x2", "x3", "x4", "x5"], col_sub="muestra"),
    "Prueba de detección: corrimiento conocido en Fase II — variables, X̄-R": dict(
        archivo="prueba_corrimiento_fase2.csv", tipo="variables", modo="largo",
        col_valor="valor", col_sub="subgrupo", col_fase="fase"),
    "Envases de jugo (Montgomery) — atributos, carta p": dict(
        archivo="jugo_naranja_montgomery.csv", tipo="atributos", col_conteo="defectuosos", col_n="n",
        col_muestra="muestra", conteo="defectuosos"),
    "Tarjetas de circuito (Montgomery) — atributos, carta c": dict(
        archivo="tarjetas_circuito_montgomery.csv", tipo="atributos", col_conteo="defectos",
        col_n="unidades_inspeccion", col_muestra="muestra", conteo="defectos"),
}

CONFIG_PLOTLY = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}

# ---------------------------------------------------------------------------
# Estado de la sesión y utilidades
# ---------------------------------------------------------------------------

RESULTADOS = ["res_var", "res_attr", "res_sup", "res_cap", "ewma", "cusum", "hist_var", "hist_attr", "pareto"]


def iniciar_estado():
    valores = {
        "df": None, "fuente": None, "archivo_bytes": None, "archivo_nombre": None,
        "tipo_datos": None, "largo_base": None, "largo": None, "tabla_attr": None, "tipo_conteo": "defectuosos",
        "notas_datos": [], "registro_limpieza": [], "excl_var": {}, "excl_attr": {},
        "res_var": None, "res_attr": None, "res_sup": None, "res_cap": None, "res_dis": None, "res_mue": None,
        "ewma": None, "cusum": None, "hist_var": None, "hist_attr": None, "pareto": None, "cfg": {},
        "meta": {"Empresa": "", "Proceso / operación": "", "Producto": "", "Característica de calidad": "",
                 "Unidad de medida": "", "Responsables": ""},
        "conclusiones_editadas": None,
    }
    for k, v in valores.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reiniciar_resultados(todo: bool = True):
    for k in RESULTADOS:
        st.session_state[k] = None
    st.session_state.excl_var = {}
    st.session_state.excl_attr = {}
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and (k.startswith("ms_excl") or k.startswith("causa_")):
            del st.session_state[k]
    st.session_state.conclusiones_editadas = None


def reiniciar_configuracion():
    """Borra el estado de los controles de configuración y limpieza para que se vuelvan a sugerir
    con los datos nuevos (otro archivo u otra hoja del libro)."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and (k.startswith("cfg_") or k.startswith("limp_")):
            del st.session_state[k]


def hoja_sugerida(hojas) -> int:
    """Hoja «Datos» si existe; si no, la primera que no sea de instrucciones (LEEME, README…)."""
    if "Datos" in hojas:
        return list(hojas).index("Datos")
    for i, h in enumerate(hojas):
        if not str(h).upper().startswith(("LEEME", "README", "INSTRUC", "LÉAME", "LEAME")):
            return i
    return 0


def redondear(df: pd.DataFrame, dec: int = 4) -> pd.DataFrame:
    """Redondea para mostrar; las celdas vacías (NaN/None) se muestran como «—»."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            if df[c].isna().any():
                df[c] = df[c].map(lambda v: "—" if pd.isna(v) else (f"{v:,.{dec}f}" if np.isfinite(v) else "∞"))
            else:
                df[c] = df[c].round(dec)
        elif df[c].dtype == object and df[c].isna().any():
            df[c] = df[c].map(lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else v)
    return df


def tabla(df: pd.DataFrame, dec: int = 4, **kw):
    st.dataframe(redondear(df, dec), hide_index=True, **kw)


def figura(fig, clave: str):
    st.plotly_chart(fig, theme=None, config=CONFIG_PLOTLY, key=clave)


def tabla_medidas(d: dict, dec: int = 4):
    """Muestra un diccionario de resultados como tabla de dos columnas (texto formateado)."""
    filas = [{"Medida": k, "Valor": num(v, dec) if isinstance(v, (int, float, np.integer, np.floating))
              and not isinstance(v, bool) else str(v)} for k, v in d.items()]
    st.dataframe(pd.DataFrame(filas), hide_index=True)


def vinetas(textos):
    if textos:
        st.markdown("\n".join(f"- {t}" for t in textos))


def num(v, dec=4):
    if v is None:
        return "—"
    try:
        if not np.isfinite(v):
            return "∞" if v > 0 else "—"
    except TypeError:
        return str(v)
    return f"{v:,.{dec}f}"


def idx(opciones, valor, defecto=0):
    try:
        return list(opciones).index(valor)
    except ValueError:
        return defecto


def aviso_sin_datos(tipo: str):
    st.info(f"Primero cargue datos por **{tipo}** en el módulo **1. Gestión de datos** "
            "(o use uno de los conjuntos de datos de prueba de la página de inicio).")


def cargar_prueba(nombre: str):
    cfg = DATOS_PRUEBA[nombre]
    ruta = RAIZ / cfg["ruta"] if cfg.get("ruta") else CARPETA_PRUEBA / cfg["archivo"]
    df = D.leer_archivo(ruta, hoja=cfg.get("hoja"))
    reiniciar_resultados()
    reiniciar_configuracion()
    st.session_state.update(df=df, fuente=f"{'Caso' if cfg.get('ruta') else 'Datos de prueba'}: {nombre}",
                            archivo_bytes=None, archivo_nombre=ruta.name, tipo_datos=cfg["tipo"], firma_datos=None)
    # datos del caso para los reportes y especificación sugerida para el módulo 5 (se aplican en la siguiente
    # ejecución, antes de crear los campos, para no modificar un widget ya dibujado)
    st.session_state.meta_pendiente = cfg.get("meta") or {k: "" for k in st.session_state.meta}
    st.session_state.espec_sugerida = cfg.get("espec") or {}
    for k in ("cap_lie", "cap_lse", "cap_obj"):
        st.session_state.pop(k, None)
    if cfg["tipo"] == "variables":
        largo, notas = D.construir_subgrupos(df, cfg["modo"], col_valor=cfg.get("col_valor"),
                                             cols_valor=cfg.get("cols"), col_subgrupo=cfg.get("col_sub"),
                                             col_fase=cfg.get("col_fase"))
        largo, registro = D.aplicar_limpieza(largo)
        st.session_state.update(largo_base=largo, largo=largo, tabla_attr=None, notas_datos=notas,
                                registro_limpieza=registro)
    else:
        tabla_attr, notas = D.construir_atributos(df, cfg["col_conteo"], col_n=cfg["col_n"],
                                                  col_muestra=cfg["col_muestra"], col_fase=cfg.get("col_fase"))
        st.session_state.update(tabla_attr=tabla_attr, largo=None, largo_base=None, notas_datos=notas,
                                tipo_conteo=cfg["conteo"], registro_limpieza=[])
    st.session_state.cfg = {"prueba": nombre, **cfg}


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

def barra_lateral() -> str:
    pendiente = st.session_state.pop("meta_pendiente", None)
    if pendiente is not None:
        for campo in list(st.session_state.meta):
            st.session_state.meta[campo] = pendiente.get(campo, "")
            st.session_state.pop(f"meta_{campo}", None)
    with st.sidebar:
        st.markdown(f"## 📈 {__nombre__}")
        st.caption("Software de Control Estadístico de Procesos")
        pagina = st.radio("Módulos", PAGINAS, key="pagina", label_visibility="collapsed")
        st.divider()
        if st.session_state.largo is not None:
            lg = st.session_state.largo
            n, const = D.tamano_subgrupo(lg)
            m1 = lg.loc[lg["fase"] == "I", "orden"].nunique()
            m2 = lg.loc[lg["fase"] == "II", "orden"].nunique()
            st.success(f"**Variables** · {lg['orden'].nunique()} subgrupos (Fase I: {m1}, Fase II: {m2}) · "
                       f"n = {n}{'' if const else ' (variable)'}")
        if st.session_state.tabla_attr is not None:
            ta = st.session_state.tabla_attr
            st.success(f"**Atributos** ({st.session_state.tipo_conteo}) · {len(ta)} muestras "
                       f"(Fase I: {(ta['fase'] == 'I').sum()}, Fase II: {(ta['fase'] == 'II').sum()})")
        if st.session_state.fuente:
            st.caption(f"Fuente: {st.session_state.fuente}")
        with st.expander("Datos del caso (para los reportes)"):
            meta = st.session_state.meta
            for campo in list(meta):
                meta[campo] = st.text_input(campo, value=meta[campo], key=f"meta_{campo}")
        st.caption(f"v{__version__}")
    return pagina


# ---------------------------------------------------------------------------
# Inicio
# ---------------------------------------------------------------------------

def pagina_inicio():
    st.title(f"{__nombre__}")
    st.markdown("#### Control estadístico de procesos para convertir datos de planta en decisiones")
    st.write("Cargue los datos de su proceso y obtenga, en un solo flujo, gráficos de control en Fase I y Fase II, "
             "validación de supuestos, índices de capacidad, diseño del esquema de muestreo con ARL y ATS, "
             "planes de muestreo de aceptación y un reporte listo para compartir.")
    col = st.columns(4)
    pasos = [("1 · Datos", "Excel o CSV, subgrupos racionales, limpieza documentada."),
             ("2-3 · Control", "X̄-R, X̄-S, I-MR, EWMA, CUSUM, p, np, c y u, con reglas de patrones."),
             ("4-5 · Diagnóstico", "Normalidad, independencia, estabilidad, Cp, Cpk y fracción no conforme."),
             ("6-8 · Decisión", "ARL/ATS para diseñar el muestreo, planes de aceptación y reportes.")]
    for c, (t, d) in zip(col, pasos):
        with c.container(border=True):
            st.markdown(f"**{t}**")
            st.caption(d)
    st.subheader("Caso aplicado y datos de prueba")
    st.write("El caso Taller A trae los datos del taller analizados en el informe: el corte del brazo de los "
             "soportes (variables) y la inspección final de los soportes terminados (atributos). Los demás conjuntos "
             "sirven para verificar el software: tres provienen de ejemplos publicados (Montgomery) con resultados "
             "conocidos y el último tiene un corrimiento conocido de 1.5σ desde el subgrupo 31, para comprobar que la "
             "Fase II lo detecta.")
    c1, c2 = st.columns([3, 1])
    eleccion = c1.selectbox("Conjunto de datos", list(DATOS_PRUEBA), key="inicio_prueba")
    c2.write("")
    c2.write("")
    c2.button("Cargar", type="primary", key="btn_inicio_prueba", on_click=cargar_prueba, args=(eleccion,))
    if (st.session_state.cfg or {}).get("prueba") == eleccion:
        st.success(f"Datos cargados: {eleccion}. Continúe con el módulo correspondiente en el menú de la izquierda.")
    with st.expander("¿Cómo usar el software con datos de su empresa?"):
        st.markdown(
            "1. Descargue la plantilla de recolección en **1. Gestión de datos** y llénela en planta.\n"
            "2. Cargue el archivo, defina los subgrupos racionales y la Fase I / Fase II.\n"
            "3. En **2** o **3** revise los gráficos de Fase I; investigue cada señal, excluya solo los puntos con "
            "causa asignable identificada y documente la causa.\n"
            "4. Valide supuestos (**4**) y calcule la capacidad (**5**) con los límites de especificación.\n"
            "5. Diseñe la frecuencia y el tamaño de muestra con ARL/ATS (**6**) y, si aplica, el plan de "
            "aceptación de lotes (**7**).\n"
            "6. Descargue el reporte en **8** y complete las conclusiones con su criterio de ingeniería.")


# ---------------------------------------------------------------------------
# Módulo 1. Gestión de datos
# ---------------------------------------------------------------------------

def pagina_datos():
    st.header("1. Gestión de datos")
    st.caption("Carga desde Excel/CSV · identificación de variables · subgrupos racionales · limpieza · tablas resumen")

    with st.container(border=True):
        st.subheader("Cargar archivo")
        c1, c2 = st.columns([3, 2])
        archivo = c1.file_uploader("Archivo de Excel (.xlsx) o CSV", type=["xlsx", "xlsm", "xls", "csv", "txt"],
                                   key="uploader")
        if archivo is not None and (archivo.name, archivo.size) != st.session_state.get("ultimo_archivo"):
            st.session_state.ultimo_archivo = (archivo.name, archivo.size)
            reiniciar_resultados()
            reiniciar_configuracion()
            st.session_state.pop("hoja_excel", None)
            st.session_state.update(archivo_bytes=archivo.getvalue(), archivo_nombre=archivo.name,
                                    fuente=f"Archivo: {archivo.name}", largo=None, largo_base=None,
                                    tabla_attr=None, cfg={}, firma_datos=None, espec_sugerida={})
            for k in ("cap_lie", "cap_lse", "cap_obj"):
                st.session_state.pop(k, None)
        with c2:
            eleccion = st.selectbox("…o use el caso aplicado o datos de prueba", ["—"] + list(DATOS_PRUEBA), key="datos_prueba_sel")
            if st.button("Cargar", disabled=eleccion == "—", key="btn_datos_prueba"):
                cargar_prueba(eleccion)
                st.rerun()

        if st.session_state.archivo_bytes is not None:
            buffer = io.BytesIO(st.session_state.archivo_bytes)
            hojas = D.hojas_excel(buffer, st.session_state.archivo_nombre)
            hoja = None
            if len(hojas) > 1:
                hoja = st.selectbox("Hoja del libro", hojas, index=hoja_sugerida(hojas), key="hoja_excel")
            firma = (st.session_state.archivo_nombre, hoja)
            if st.session_state.get("firma_datos") not in (None, firma):
                # otra hoja = otros datos: se reinician los resultados y la configuración sugerida
                reiniciar_resultados()
                reiniciar_configuracion()
                st.session_state.update(largo=None, largo_base=None, tabla_attr=None, cfg={})
            st.session_state.firma_datos = firma
            try:
                st.session_state.df = D.leer_archivo(io.BytesIO(st.session_state.archivo_bytes),
                                                     st.session_state.archivo_nombre, hoja)
            except Exception as e:  # noqa: BLE001 - mostrar cualquier error de lectura al usuario
                st.error(f"No se pudo leer el archivo: {e}")
                return

        st.markdown("**Plantillas de recolección** (llénelas en la visita y cárguelas aquí):")
        p1, p2 = st.columns(2)
        for col, nombre, etiqueta in ((p1, "plantilla_datos_variables.xlsx", "Plantilla de variables (mediciones)"),
                                      (p2, "plantilla_datos_atributos.xlsx", "Plantilla de atributos (conteos)")):
            ruta = CARPETA_PLANTILLAS / nombre
            if ruta.exists():
                col.download_button(etiqueta, ruta.read_bytes(), file_name=nombre, key=f"dl_{nombre}",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    df = st.session_state.df
    if df is None:
        st.info("Cargue un archivo o un conjunto de datos de prueba para continuar.")
        return

    st.subheader("Vista previa e identificación de variables")
    st.caption(f"{st.session_state.fuente} · {len(df)} filas × {df.shape[1]} columnas")
    t1, t2 = st.tabs(["Datos", "Identificación de variables"])
    with t1:
        st.dataframe(df.head(200), hide_index=True, height=260)
    with t2:
        tabla(D.identificar_variables(df))

    configurar_datos(df)

    if st.session_state.largo_base is not None:
        limpieza_y_resumen_variables()
    elif st.session_state.tabla_attr is not None:
        resumen_atributos()


def configurar_datos(df: pd.DataFrame):
    cfg = st.session_state.cfg or {}
    cols = list(df.columns)
    numericas = D.columnas_numericas(df)
    ninguna = "(ninguna)"
    with st.container(border=True):
        st.subheader("Subgrupos racionales y fases")
        tipo = st.radio("Tipo de datos", ["variables", "atributos"], horizontal=True,
                        index=idx(["variables", "atributos"], cfg.get("tipo", st.session_state.tipo_datos or "variables")),
                        format_func={"variables": "Variables (mediciones continuas)",
                                     "atributos": "Atributos (conteos de defectuosos o defectos)"}.get, key="cfg_tipo")
        col_orden = st.selectbox("Ordenar cronológicamente por (opcional)", [ninguna] + cols, key="cfg_orden",
                                 help="Use una columna de fecha u hora si las filas no están en orden de recolección.")
        col_orden = None if col_orden == ninguna else col_orden

        if tipo == "variables":
            modos = {"ancho": "Ancho: una fila por subgrupo (x1, x2, …)",
                     "largo": "Largo: una columna de mediciones y otra con el subgrupo",
                     "consecutivo": "Consecutivo: una columna; agrupar cada n filas seguidas",
                     "individual": "Individual: un dato por subgrupo (carta I-MR)"}
            detectado = "ancho" if sum(str(c).lower().startswith("x") for c in numericas) >= 2 else "largo"
            modo = st.radio("Formato de los datos", list(modos), format_func=modos.get, key="cfg_modo",
                            index=idx(list(modos), cfg.get("modo", detectado)))
            c1, c2 = st.columns(2)
            col_valor = cols_valor = col_sub = tamano = None
            if modo == "ancho":
                sugeridas = cfg.get("cols") or [c for c in numericas if str(c).lower().startswith("x")] or numericas[:5]
                cols_valor = c1.multiselect("Columnas con las mediciones del subgrupo", numericas,
                                            default=[c for c in sugeridas if c in numericas], key="cfg_cols", placeholder="Elija las columnas x1, x2, …")
                opciones_sub = [ninguna] + cols
                col_sub = c2.selectbox("Columna que identifica el subgrupo (opcional)", opciones_sub,
                                       index=idx(opciones_sub, cfg.get("col_sub") or _sugerir(cols, ("subgrupo", "muestra")) or ninguna),
                                       key="cfg_sub_ancho")
            else:
                col_valor = c1.selectbox("Columna con las mediciones", numericas,
                                         index=idx(numericas, cfg.get("col_valor") or _sugerir(numericas, ("valor", "medic", "peso", "long"))),
                                         key="cfg_valor")
                if modo == "largo":
                    col_sub = c2.selectbox("Columna que identifica el subgrupo", cols,
                                           index=idx(cols, cfg.get("col_sub") or _sugerir(cols, ("subgrupo", "muestra", "lote"))),
                                           key="cfg_sub")
                elif modo == "consecutivo":
                    tamano = c2.number_input("Tamaño de subgrupo n", 2, 25, int(cfg.get("tamano", 5)), key="cfg_n")
            col_sub = None if col_sub == ninguna else col_sub
        else:
            tipo_conteo = st.radio("¿Qué se cuenta en cada muestra?", ["defectuosos", "defectos"], horizontal=True,
                                   index=idx(["defectuosos", "defectos"], cfg.get("conteo", st.session_state.tipo_conteo)),
                                   format_func={"defectuosos": "Unidades defectuosas (cada unidad pasa o no pasa)",
                                                "defectos": "Defectos o no conformidades (una unidad puede tener varios)"}.get,
                                   key="cfg_conteo")
            c1, c2, c3 = st.columns(3)
            # la sugerencia depende de lo que se cuenta: «defectos_totales» para defectos, «defectuosos» para unidades
            preferidas = ("defectos", "no_conform", "noconform") if tipo_conteo == "defectos" else ("defectuos", "rechaz")
            sugerida = cfg.get("col_conteo") if cfg.get("conteo") == tipo_conteo else None
            sugerida = sugerida or _sugerir(numericas, preferidas) or _sugerir(numericas, ("defect", "conform", "rechaz"))
            col_conteo = c1.selectbox("Columna del conteo", numericas, key=f"cfg_col_conteo_{tipo_conteo}",
                                      index=idx(numericas, sugerida))
            opciones_n = ["(tamaño constante)"] + numericas
            col_n = c2.selectbox("Columna del tamaño de muestra n", opciones_n, key="cfg_col_n",
                                 index=idx(opciones_n, cfg.get("col_n") or _sugerir(numericas, ("inspecc", "unidades", "tama")) or opciones_n[0]))
            n_const = None
            if col_n == "(tamaño constante)":
                col_n = None
                n_const = c2.number_input("Tamaño constante n", 1, 100000, 50, key="cfg_n_const")
            opciones_m = [ninguna] + cols
            col_muestra = c3.selectbox("Columna que identifica la muestra (opcional)", opciones_m, key="cfg_col_muestra",
                                       index=idx(opciones_m, cfg.get("col_muestra") or _sugerir(cols, ("muestra", "lote")) or ninguna))
            col_muestra = None if col_muestra == ninguna else col_muestra

        st.markdown("**Fase I (estimación de límites) y Fase II (monitoreo)**")
        col_fase_sug = _sugerir(cols, ("fase",))
        metodo_fase = st.radio("Cómo se definen las fases", ["columna", "primeros"], horizontal=True,
                               index=0 if col_fase_sug else 1, key="cfg_metodo_fase",
                               format_func={"columna": "Con una columna del archivo (valores I / II)",
                                            "primeros": "Los primeros m subgrupos son Fase I"}.get)
        col_fase, m1 = None, None
        if metodo_fase == "columna":
            col_fase = st.selectbox("Columna de fase", cols, index=idx(cols, col_fase_sug), key="cfg_col_fase")
        else:
            m1 = st.number_input("Número de subgrupos (o muestras) de Fase I", 2, 100000, int(cfg.get("m1", 25)),
                                 key="cfg_m1", help="Si todo es Fase I, escriba un número mayor o igual al total.")

        if st.button("Aplicar configuración", type="primary", key="btn_aplicar_datos"):
            try:
                reiniciar_resultados()
                if tipo == "variables":
                    largo, notas = D.construir_subgrupos(df, modo, col_valor=col_valor, cols_valor=cols_valor,
                                                         col_subgrupo=col_sub, tamano=tamano, col_fase=col_fase,
                                                         col_orden=col_orden)
                    largo, registro = D.aplicar_limpieza(largo)
                    if m1:
                        largo = D.asignar_fases(largo, int(m1))
                    st.session_state.update(largo_base=largo, largo=largo, tabla_attr=None, notas_datos=notas,
                                            registro_limpieza=registro, tipo_datos="variables")
                    st.session_state.cfg = dict(tipo="variables", modo=modo, col_valor=col_valor, cols=cols_valor,
                                                col_sub=col_sub, tamano=tamano, m1=m1)
                else:
                    ta, notas = D.construir_atributos(df, col_conteo, col_n=col_n, n_constante=n_const,
                                                      col_muestra=col_muestra, col_fase=col_fase, col_orden=col_orden)
                    if m1:
                        ta = D.asignar_fases_atributos(ta, int(m1))
                    st.session_state.update(tabla_attr=ta, largo=None, largo_base=None, notas_datos=notas,
                                            tipo_conteo=tipo_conteo, tipo_datos="atributos", registro_limpieza=[])
                    st.session_state.cfg = dict(tipo="atributos", conteo=tipo_conteo, col_conteo=col_conteo,
                                                col_n=col_n, col_muestra=col_muestra, m1=m1)
                st.success("Configuración aplicada.")
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo aplicar la configuración: {e}")
        for nota in st.session_state.notas_datos:
            st.caption(f"ℹ️ {nota}")


def _sugerir(columnas, patrones):
    for c in columnas:
        if any(p in str(c).lower() for p in patrones):
            return c
    return None


def limpieza_y_resumen_variables():
    base = st.session_state.largo_base
    diag = D.diagnosticar_limpieza(base)
    with st.container(border=True):
        st.subheader("Limpieza de datos")
        c1, c2, c3 = st.columns(3)
        c1.metric("Celdas vacías o no numéricas", len(diag["faltantes"]))
        c2.metric("Subgrupos con tamaño distinto", len(diag["subgrupos_incompletos"]), help=f"n modal = {diag['n_modal']}")
        c3.metric("Valores atípicos (criterio de Tukey)", len(diag["atipicos"]),
                  help=f"Fuera de [{num(diag['cercas_tukey'][0])}, {num(diag['cercas_tukey'][1])}]")
        if len(diag["subgrupos_incompletos"]):
            st.caption("Subgrupos con tamaño distinto del modal: " +
                       ", ".join(f"{o} (n = {k})" for o, k in diag["subgrupos_incompletos"].items()))
        quitar_inc = st.checkbox("Eliminar subgrupos con tamaño distinto del modal", key="limp_incompletos",
                                 help="Necesario para X̄-R. La carta X̄-S sí admite n variable.")
        filas_excluir, justificacion = [], ""
        if len(diag["atipicos"]):
            st.dataframe(diag["atipicos"][["fila", "subgrupo", "valor", "fase"]], hide_index=True)
            filas_excluir = st.multiselect("Excluir mediciones atípicas (por fila del archivo)",
                                           diag["atipicos"]["fila"].tolist(), key="limp_filas", placeholder="Ninguna",
                                           help="Excluya solo si comprobó un error de medición o registro.")
            if filas_excluir:
                justificacion = st.text_input("Justificación de la exclusión (queda en el reporte)", key="limp_just")
        if st.button("Aplicar limpieza", key="btn_limpieza"):
            largo, registro = D.aplicar_limpieza(base, True, quitar_inc, filas_excluir)
            if filas_excluir:
                registro.append(f"Justificación: {justificacion or 'sin justificar (complete antes del informe)'}")
            reiniciar_resultados()
            st.session_state.largo = largo
            st.session_state.registro_limpieza = registro or ["No se requirieron cambios."]
        for r in st.session_state.registro_limpieza:
            st.caption(f"✔ {r}")

    largo = st.session_state.largo
    st.subheader("Tablas resumen")
    n, const = D.tamano_subgrupo(largo)
    c = st.columns(4)
    c[0].metric("Mediciones", int(largo["valor"].notna().sum()))
    c[1].metric("Subgrupos", largo["orden"].nunique())
    c[2].metric("Tamaño de subgrupo n", f"{n}" + ("" if const else " (variable)"))
    c[3].metric("Fase I / Fase II", f"{largo.loc[largo.fase == 'I', 'orden'].nunique()} / "
                                     f"{largo.loc[largo.fase == 'II', 'orden'].nunique()}")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Estadística descriptiva (todos los datos)**")
        tabla(D.resumen_descriptivo(largo["valor"]), 5)
    with c2:
        figura(G.fig_corrida(largo), "fig_corrida")
        figura(G.fig_histograma(largo["valor"]), "fig_hist_datos")
    with st.expander("Resumen por subgrupo"):
        tabla(D.resumen_subgrupos(largo), 5)


def resumen_atributos():
    ta = st.session_state.tabla_attr
    st.subheader("Tabla resumen de muestras")
    c = st.columns(4)
    c[0].metric("Muestras", len(ta))
    c[1].metric("Unidades inspeccionadas", f"{ta['n'].sum():,.0f}")
    c[2].metric(f"Total de {st.session_state.tipo_conteo}", f"{ta['conteo'].sum():,.0f}")
    tasa = ta["conteo"].sum() / ta["n"].sum()
    c[3].metric("Proporción global" if st.session_state.tipo_conteo == "defectuosos" else "Defectos por unidad",
                f"{tasa:.4f}")
    tabla(ta.drop(columns=["fila"]), 4)


# ---------------------------------------------------------------------------
# Exclusiones en Fase I (común a variables y atributos)
# ---------------------------------------------------------------------------

def exclusiones_actuales(prefijo: str, guardadas: dict) -> dict:
    """Lee la selección de exclusiones desde el estado de los widgets (antes de dibujarlos)."""
    clave_ms = f"ms_excl_{prefijo}"
    if clave_ms not in st.session_state:
        st.session_state[clave_ms] = list(guardadas)
    seleccion = st.session_state[clave_ms]
    return {e: st.session_state.get(f"causa_{prefijo}_{e}", guardadas.get(e, "")) for e in seleccion}


def panel_exclusiones(prefijo: str, opciones: list, senales_f1: dict, callback_auto, args_auto: tuple):
    clave_ms = f"ms_excl_{prefijo}"
    guardadas = st.session_state[f"excl_{prefijo}"]
    with st.container(border=True):
        st.subheader("Fase I: causas asignables y recálculo de límites")
        st.write("Investigue cada señal de la Fase I. Si encuentra su **causa asignable** (error de medición, cambio "
                 "de lote, ajuste de máquina…), excluya ese punto del cálculo y escriba la causa: los límites se "
                 "recalculan con el resto. No excluya puntos sin una causa identificada.")
        if senales_f1:
            st.caption("Puntos con señales en la Fase I: " +
                       "; ".join(f"{e} ({'regla' if len(r) == 1 else 'reglas'} {', '.join(map(str, sorted(r)))})"
                                 for e, r in senales_f1.items()))
        st.multiselect("Subgrupos o muestras a excluir del cálculo de límites", opciones, key=clave_ms,
                       placeholder="Ninguno excluido (elija si encontró una causa asignable)")
        nuevas = {}
        for e in st.session_state[clave_ms]:
            if f"causa_{prefijo}_{e}" not in st.session_state:
                st.session_state[f"causa_{prefijo}_{e}"] = guardadas.get(e, "")
            nuevas[e] = st.text_input(f"Causa asignable del punto {e}", key=f"causa_{prefijo}_{e}",
                                      placeholder="Ej.: cambio de lote de materia prima a las 10:40")
        st.session_state[f"excl_{prefijo}"] = nuevas
        sin_causa = [e for e, c in nuevas.items() if not str(c).strip()]
        if sin_causa:
            st.warning("Falta documentar la causa de: " + ", ".join(sin_causa) +
                       ". El informe debe justificar cada exclusión.")
        st.button("Excluir los puntos fuera de límites y recalcular (una iteración)", key=f"btn_auto_{prefijo}",
                  on_click=callback_auto, args=args_auto,
                  help="Excluye los puntos de Fase I que hoy están fuera de los límites y recalcula. Repita solo si "
                       "los nuevos puntos fuera de límites también tienen causa asignable; luego escriba las causas.")
        hist = st.session_state.get(f"hist_{prefijo}")
        if hist:
            st.markdown("**Historial de iteraciones (límites de prueba → definitivos)**")
            tabla(pd.DataFrame(hist), 5)


def _una_iteracion(prefijo: str, funcion, datos, tipo, kw):
    """Excluye los puntos de Fase I que hoy están fuera de límites y recalcula (una iteración por clic)."""
    guardadas = st.session_state[f"excl_{prefijo}"]
    res, hist = funcion(tipo, datos, excluidos_iniciales=list(guardadas), max_iter=1, **kw)
    st.session_state[f"excl_{prefijo}"] = {e: guardadas.get(e, "") for e in res.excluidos}
    st.session_state[f"ms_excl_{prefijo}"] = list(st.session_state[f"excl_{prefijo}"])
    previo = st.session_state.get(f"hist_{prefijo}") or []
    for h in hist:
        h["Iteración"] = len(previo) + 1
        previo.append(h)
    st.session_state[f"hist_{prefijo}"] = previo


def _auto_var(tipo, kw):
    _una_iteracion("var", VAR.fase1_iterativa, st.session_state.largo, tipo, kw)


def _auto_attr(tipo, kw):
    _una_iteracion("attr", ATR.fase1_iterativa, st.session_state.tabla_attr, tipo, kw)


def senales_por_punto(cartas, fase="I") -> dict:
    salida = {}
    for c in cartas:
        for s in c.senales:
            if s.fase == fase:
                salida.setdefault(s.etiqueta, set()).add(s.regla)
    return salida


def tabla_limites(cartas) -> pd.DataFrame:
    filas = []
    for c in cartas:
        r = c.resumen_limites()
        filas.append({"Gráfico": c.titulo, "LCI": r["LCI"], "LC": r["LC"], "LCS": r["LCS"]})
    df = pd.DataFrame(filas)
    for col in ("LCI", "LC", "LCS"):
        df[col] = df[col].map(lambda v: num(v, 5) if isinstance(v, (float, np.floating)) else v)
    return df


# ---------------------------------------------------------------------------
# Módulo 2. Gráficos por variables
# ---------------------------------------------------------------------------

def pagina_variables():
    st.header("2. Gráficos de control por variables")
    largo = st.session_state.largo
    if largo is None:
        aviso_sin_datos("variables")
        return
    n, const = D.tamano_subgrupo(largo)
    recomendado, razon = VAR.recomendar_carta(n, const)
    opciones = ["xbar_r", "xbar_s", "i_mr"]

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        tipo = c1.selectbox("Tipo de gráfico", opciones, index=idx(opciones, recomendado),
                            format_func=lambda t: VAR.NOMBRES[t] + (" (recomendado)" if t == recomendado else ""),
                            key="var_tipo")
        c1.caption(razon)
        L = c2.number_input("Ancho de límites L (σ)", 1.0, 5.0, 3.0, 0.05, key="var_L")
        racha = c3.selectbox("Puntos para la regla 4", [8, 9], key="var_racha",
                             help="8 según Western Electric y Montgomery; 9 según Nelson y Minitab.")
        c4, c5 = st.columns(2)
        reglas_pos = c4.multiselect("Reglas en el gráfico de posición (X̄ o I)", list(range(1, 9)), default=[1, 2, 3, 4],
                                    format_func=lambda r: f"{r}. {REG.NOMBRES_CORTOS[r]}", key="var_reglas", placeholder="Ninguna regla")
        reglas_disp = c5.multiselect("Reglas en el gráfico de dispersión (R, S o MR)", list(range(1, 9)), default=[1],
                                     format_func=lambda r: f"{r}. {REG.NOMBRES_CORTOS[r]}", key="var_reglas_disp", placeholder="Ninguna regla")
        c6, c7 = st.columns(2)
        fuente = c6.radio("Límites para monitorear la Fase II", ["fase1", "estandar"], horizontal=True,
                          format_func={"fase1": "Estimados en la Fase I", "estandar": "Valores estándar (μ0, σ0)"}.get,
                          key="var_fuente")
        zonas = c7.toggle("Mostrar zonas ±1σ y ±2σ", True, key="var_zonas")
        estandar = None
        if fuente == "estandar":
            e1, e2 = st.columns(2)
            mu0 = e1.number_input("μ0 (media objetivo)", value=float(largo["valor"].mean()), format="%.6f", key="var_mu0")
            s0 = e2.number_input("σ0 (desviación estándar conocida)", value=float(largo["valor"].std()), format="%.6f",
                                 min_value=1e-12, key="var_s0")
            estandar = VAR.Estandar(mu0, s0)

    excluidos = exclusiones_actuales("var", st.session_state.excl_var)
    kw = dict(L=L, estandar=estandar, reglas_pos=reglas_pos, reglas_disp=reglas_disp, racha_lado=racha)
    try:
        res = VAR.calcular(tipo, largo, excluidos=list(excluidos), **kw)
    except ValueError as e:
        st.error(str(e))
        return
    st.session_state.res_var = res

    figura(G.fig_par_cartas(res.carta_pos, res.carta_disp, titulo=f"Gráficos {VAR.NOMBRES[tipo]}", zonas=zonas),
           "fig_var")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Límites de control**")
        st.dataframe(tabla_limites(res.cartas()), hide_index=True)
        st.caption(f"{res.metodo_sigma} → σ̂ = {num(res.sigma, 6)} · centro = {num(res.media, 6)}")
    with c2:
        st.markdown("**Interpretación preliminar**")
        vinetas(VAR.interpretar(res) + res.notas + [n for c in res.cartas() for n in c.notas])

    senales = REG.tabla_senales(res.cartas())
    st.markdown("**Señales de falta de control y patrones no aleatorios**")
    if senales.empty:
        st.success("No se detectaron señales con las reglas seleccionadas.")
    else:
        tabla(senales)

    opciones_f1 = [str(e) for e, f in zip(res.carta_pos.etiquetas, res.carta_pos.fase) if f == "I"]
    panel_exclusiones("var", opciones_f1, senales_por_punto(res.cartas()), _auto_var, (tipo, kw))

    with st.expander("Tabla punto a punto"):
        for c in res.cartas():
            st.markdown(f"**{c.titulo}**")
            tabla(c.tabla(), 5)

    with st.expander("Gráficos con memoria para corrimientos pequeños: EWMA y CUSUM (avanzado)"):
        if not const:
            st.info("EWMA y CUSUM se implementan para tamaño de subgrupo constante.")
        else:
            st.caption(f"Usan la media y la σ estimadas en la Fase I (μ0 = {num(res.media, 5)}, σ = {num(res.sigma, 5)}). "
                       "Son más sensibles que X̄ a corrimientos de 0.5σ a 1.5σ.")
            a1, a2, a3, a4 = st.columns(4)
            lam = a1.number_input("λ (EWMA)", 0.01, 1.0, 0.2, 0.01, key="ewma_lam")
            Lw = a2.number_input("L (EWMA)", 1.0, 4.0, 2.86, 0.01, key="ewma_L",
                                 help="λ = 0.2 y L = 2.86 → ARL0 ≈ 370 (igual que Shewhart 3σ).")
            k = a3.number_input("k (CUSUM)", 0.05, 2.0, 0.5, 0.05, key="cusum_k")
            h = a4.number_input("h (CUSUM)", 1.0, 10.0, 5.0, 0.5, key="cusum_h",
                                help="k = 0.5 y h = 5 → ARL0 ≈ 465 y ARL1(1σ) ≈ 10.4.")
            c_ewma = VAR.carta_ewma(largo, res.media, res.sigma, lam, Lw, racha)
            c_sup, c_inf = VAR.carta_cusum(largo, res.media, res.sigma, k, h)
            st.session_state.ewma = c_ewma
            st.session_state.cusum = (c_sup, c_inf)
            figura(G.fig_carta(c_ewma, zonas=False), "fig_ewma")
            figura(G.fig_cusum(c_sup, c_inf), "fig_cusum")
            sen_ewma = [s.etiqueta for s in c_ewma.senales]
            sen_cusum = sorted({s.etiqueta for s in c_sup.senales + c_inf.senales}, key=VAR._clave_orden)
            vinetas([f"EWMA: {'señales en los subgrupos ' + ', '.join(sen_ewma) if sen_ewma else 'sin señales'}.",
                     f"CUSUM: {'señales en los subgrupos ' + ', '.join(sen_cusum) if sen_cusum else 'sin señales'}."])


# ---------------------------------------------------------------------------
# Módulo 3. Gráficos por atributos
# ---------------------------------------------------------------------------

def pagina_atributos():
    st.header("3. Gráficos de control por atributos")
    ta = st.session_state.tabla_attr
    if ta is None:
        aviso_sin_datos("atributos")
        return
    n_const = ta["n"].nunique() == 1
    recomendado, razon = ATR.recomendar_carta(st.session_state.tipo_conteo, n_const)
    opciones = ["p", "np", "c", "u"]
    with st.container(border=True):
        st.markdown("**Selección justificada del gráfico**")
        st.info(f"Datos: **{st.session_state.tipo_conteo}**, tamaño de muestra "
                f"{'constante (n = ' + format(ta['n'].iloc[0], 'g') + ')' if n_const else 'variable'}. "
                f"Recomendado: **carta {recomendado}**. {razon}")
        c1, c2, c3 = st.columns([2, 1, 1])
        tipo = c1.selectbox("Tipo de gráfico", opciones, index=idx(opciones, recomendado),
                            format_func=lambda t: ATR.NOMBRES[t] + (" (recomendado)" if t == recomendado else ""),
                            key="attr_tipo")
        L = c2.number_input("Ancho de límites L (σ)", 1.0, 5.0, 3.0, 0.05, key="attr_L")
        racha = c3.selectbox("Puntos para la regla 4", [8, 9], key="attr_racha")
        reglas = st.multiselect("Reglas para detectar patrones", list(range(1, 9)), default=[1, 2, 3, 4],
                                format_func=lambda r: f"{r}. {REG.NOMBRES_CORTOS[r]}", key="attr_reglas", placeholder="Ninguna regla")
        estandar = None
        if st.toggle("Usar un valor estándar dado (p0, c0 o u0) en lugar de estimarlo en la Fase I", key="attr_usa_std"):
            etiqueta = {"p": "p0", "np": "p0", "c": "c0", "u": "u0"}[tipo]
            estandar = st.number_input(etiqueta, min_value=1e-9, value=float(ta["conteo"].sum() / ta["n"].sum())
                                       if tipo in ("p", "np", "u") else float(ta["conteo"].mean()),
                                       format="%.6f", key="attr_std")
        if tipo != recomendado:
            st.warning(f"Eligió la carta {tipo}, distinta de la recomendada ({recomendado}). Justifique la elección en el informe.")

    excluidos = exclusiones_actuales("attr", st.session_state.excl_attr)
    kw = dict(L=L, estandar=estandar, reglas=reglas, racha_lado=racha)
    try:
        res = ATR.calcular(tipo, ta, excluidos=list(excluidos), **kw)
    except ValueError as e:
        st.error(str(e))
        return
    st.session_state.res_attr = res
    figura(G.fig_carta(res.carta, zonas=True, alto=420), "fig_attr")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Límites de control**")
        st.dataframe(tabla_limites([res.carta]), hide_index=True)
    with c2:
        st.markdown("**Interpretación preliminar**")
        vinetas(ATR.interpretar(res))
    senales = REG.tabla_senales([res.carta])
    st.markdown("**Señales**")
    if senales.empty:
        st.success("No se detectaron señales con las reglas seleccionadas.")
    else:
        tabla(senales)
    opciones_f1 = [str(e) for e, f in zip(res.carta.etiquetas, res.carta.fase) if f == "I"]
    panel_exclusiones("attr", opciones_f1, senales_por_punto([res.carta]), _auto_attr, (tipo, kw))
    with st.expander("Tabla punto a punto"):
        tabla(res.carta.tabla(), 5)
    with st.expander("Pareto de tipos de defecto (para priorizar mejoras)"):
        df_orig = st.session_state.df
        cfg = st.session_state.cfg or {}
        usadas = {cfg.get("col_conteo"), cfg.get("col_n"), cfg.get("col_muestra")}
        candidatas, sugeridas = ATR.sugerir_tipos_defecto(df_orig, usadas) if df_orig is not None else ([], [])
        tipos_def = st.multiselect("Columnas con el conteo de cada tipo de defecto", candidatas, default=sugeridas,
                                   key="attr_pareto_cols", placeholder="Elija las columnas de tipos de defecto")
        if tipos_def:
            tp = ATR.pareto(df_orig[tipos_def].sum(numeric_only=True))
            figura(G.fig_pareto(tp), "fig_pareto")
            tabla(tp, 1)
            vitales = tp[tp["% acumulado"].shift(fill_value=0) < 80]["Tipo de defecto"].tolist()
            st.caption("Pocos vitales (hasta acumular cerca del 80 %): " + ", ".join(v.replace("_", " ") for v in vitales))
            st.session_state.pareto = tp
        else:
            st.caption("El archivo debe tener una columna por tipo de defecto (como la plantilla de atributos).")
    if tipo in ("p", "np"):
        with st.expander("Tamaño de muestra para detectar un corrimiento (Montgomery, ec. 7.10)"):
            delta = st.number_input("Corrimiento a detectar δ en la proporción (p. ej. 0.05)", 0.001, 0.5, 0.05, 0.005,
                                    format="%.3f", key="attr_delta")
            n_req = ATR.tamano_para_detectar(res.parametro, delta, L)
            st.write(f"Para detectar que p pasa de {res.parametro:.4f} a {res.parametro + delta:.4f} con probabilidad "
                     f"0.5 en la primera muestra se requiere n ≈ **{np.ceil(n_req):.0f}** unidades por muestra.")


# ---------------------------------------------------------------------------
# Módulo 4. Validación de supuestos
# ---------------------------------------------------------------------------

def pagina_supuestos():
    st.header("4. Validación de supuestos")
    largo = st.session_state.largo
    res = st.session_state.res_var
    if largo is None and st.session_state.res_attr is None:
        aviso_sin_datos("variables")
        return
    resultado = {}
    if largo is not None:
        excl = set(st.session_state.excl_var)
        base = st.radio("Datos a evaluar", ["f1", "todos"], horizontal=True, key="sup_base",
                        format_func={"f1": "Fase I sin subgrupos excluidos (recomendado)", "todos": "Todos los datos"}.get)
        datos = largo[(largo["fase"] == "I") & ~largo["subgrupo"].astype(str).isin(excl)] if base == "f1" else largo
        t1, t2, t3, t4 = st.tabs(["Normalidad", "Independencia", "Estabilidad y patrones", "Causas especiales"])
        with t1:
            rn = SUP.pruebas_normalidad(datos["valor"])
            resultado["normalidad"] = rn
            c = st.columns(3)
            c[0].metric("N", rn["n"])
            c[1].metric("Asimetría", num(rn["asimetria"], 3))
            c[2].metric("Curtosis (exceso)", num(rn["curtosis"], 3))
            tabla(rn["tabla"])
            (st.success if rn["normal"] else st.warning)(rn["conclusion"])
            g1, g2 = st.columns(2)
            with g1:
                figura(G.fig_histograma(datos["valor"], "Histograma con normal ajustada"), "fig_sup_hist")
            with g2:
                figura(G.fig_qq(SUP.datos_qq(datos["valor"])), "fig_sup_qq")
            if rn["normal"] is False:
                bc = SUP.box_cox(datos["valor"])
                if bc["posible"]:
                    st.info(f"Transformación de Box-Cox sugerida: λ = {bc['lambda']:.3f} (IC 95 %: {bc['ic95'][0]:.2f} a "
                            f"{bc['ic95'][1]:.2f}; valor práctico {bc['lambda_redondeado']:g}). "
                            + ("El intervalo incluye λ = 1: la transformación no es indispensable." if bc["incluye_1"]
                               else "Puede usarla en el módulo 5 para calcular la capacidad."))
                    resultado["boxcox"] = {k: v for k, v in bc.items() if k != "transformados"}
                else:
                    st.info(bc["motivo"])
        with t2:
            n_sub, _ = D.tamano_subgrupo(datos)
            serie = st.radio("Serie a evaluar", ["individuales", "medias"], horizontal=True, key="sup_serie",
                             index=0 if n_sub == 1 else 1,
                             format_func={"individuales": "Mediciones individuales en orden de toma",
                                          "medias": "Medias de subgrupo"}.get)
            ordenados = datos.sort_values(["orden"], kind="stable")
            x = ordenados["valor"] if serie == "individuales" else ordenados.groupby("orden")["valor"].mean()
            if len(x) < 8:
                st.info("Se necesitan al menos 8 datos para evaluar la independencia.")
            else:
                ri = SUP.pruebas_independencia(x)
                resultado["independencia"] = ri
                tabla(ri["tabla"])
                (st.success if ri["independiente"] else st.warning)(ri["conclusion"])
                figura(G.fig_acf(ri["acf"]), "fig_sup_acf")
        with t3:
            if res is None:
                st.info("Calcule primero los gráficos del módulo 2 para evaluar la estabilidad.")
            else:
                resultado["estable"] = mostrar_estabilidad(res.cartas(), "variables")
        with t4:
            mostrar_causas()
    if st.session_state.res_attr is not None:
        st.subheader("Estabilidad del gráfico por atributos")
        resultado["estable_attr"] = mostrar_estabilidad([st.session_state.res_attr.carta], "atributos")
    st.session_state.res_sup = resultado


def mostrar_estabilidad(cartas, nombre) -> bool:
    filas = []
    for c in cartas:
        activos = [s for s in c.senales if s.fase == "I" and not c.excluido[s.posicion]]
        filas.append({"Gráfico": c.titulo, "Puntos de Fase I": int((c.fase == "I").sum()),
                      "Excluidos": int(c.excluido.sum()),
                      "Señales activas (Fase I)": len(activos),
                      "Reglas violadas": ", ".join(map(str, sorted({s.regla for s in activos}))) or "ninguna"})
    t = pd.DataFrame(filas)
    tabla(t)
    estable = bool((t["Señales activas (Fase I)"] == 0).all())
    if estable:
        st.success(f"Estabilidad preliminar ({nombre}): con las exclusiones justificadas, los puntos de Fase I no "
                   "presentan señales; los límites pueden adoptarse para el monitoreo (Fase II).")
    else:
        st.warning(f"Estabilidad preliminar ({nombre}): persisten señales en la Fase I. Investigue sus causas; si no "
                   "hay causa asignable, conserve el punto y documente el análisis.")
    detalle = REG.tabla_senales(cartas)
    if not detalle.empty:
        st.markdown("**Patrones no aleatorios detectados y causas típicas a investigar**")
        tabla(detalle)
    return estable


def mostrar_causas():
    filas = [{"Tipo": "Variables", "Subgrupo/muestra": e, "Causa asignable documentada": c or "⚠ sin documentar"}
             for e, c in st.session_state.excl_var.items()]
    filas += [{"Tipo": "Atributos", "Subgrupo/muestra": e, "Causa asignable documentada": c or "⚠ sin documentar"}
              for e, c in st.session_state.excl_attr.items()]
    if filas:
        tabla(pd.DataFrame(filas))
    else:
        st.info("No hay puntos excluidos. Las exclusiones se hacen en los módulos 2 y 3 (Fase I).")
    if st.session_state.registro_limpieza:
        st.markdown("**Registro de limpieza de datos**")
        vinetas(st.session_state.registro_limpieza)


# ---------------------------------------------------------------------------
# Módulo 5. Capacidad
# ---------------------------------------------------------------------------

def pagina_capacidad():
    st.header("5. Capacidad del proceso")
    largo = st.session_state.largo
    if largo is None:
        aviso_sin_datos("variables")
        return
    res = st.session_state.res_var
    if res is None:
        n, const = D.tamano_subgrupo(largo)
        res = VAR.calcular(VAR.recomendar_carta(n, const)[0], largo)
        st.caption("Se usó el gráfico recomendado para estimar σ; ajústelo en el módulo 2 si es necesario.")
    metodo = {"xbar_r": "R", "xbar_s": "S", "i_mr": "MR"}[res.tipo]
    espec = st.session_state.get("espec_sugerida") or {}   # especificación del conjunto de ejemplo cargado

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        lie = c1.number_input("LIE (límite inferior de especificación)", value=espec.get("lie"), format="%.6f", key="cap_lie",
                              placeholder="vacío si es unilateral")
        lse = c2.number_input("LSE (límite superior de especificación)", value=espec.get("lse"), format="%.6f", key="cap_lse",
                              placeholder="vacío si es unilateral")
        objetivo = c3.number_input("Valor objetivo / nominal (opcional)", value=espec.get("objetivo"), format="%.6f", key="cap_obj")
        c4, c5, c6 = st.columns(3)
        minimos = {1.33: "1.33 · proceso existente, bilateral", 1.25: "1.25 · proceso existente, unilateral",
                   1.50: "1.50 · proceso nuevo o parámetro crítico", 1.67: "1.67 · nuevo y crítico",
                   2.0: "2.00 · Seis Sigma"}
        cpk_min = c4.selectbox("Cpk mínimo exigido", list(minimos), format_func=minimos.get, key="cap_min")
        alfa = c5.selectbox("Nivel de significancia (IC)", [0.05, 0.10, 0.01], key="cap_alfa",
                            format_func=lambda a: f"{a:g} (confianza {100 * (1 - a):.0f} %)")
        base = c6.radio("Datos", ["f1", "todos"], key="cap_base",
                        format_func={"f1": "Fase I sin excluidos", "todos": "Todos"}.get)
    if lie is None and lse is None:
        st.info("Escriba al menos un límite de especificación (LIE o LSE). Deben venir de la ficha técnica, "
                "del cliente o de una norma, no de los límites de control.")
        return
    excl = set(st.session_state.excl_var)
    datos = largo[(largo["fase"] == "I") & ~largo["subgrupo"].astype(str).isin(excl)] if base == "f1" else largo
    try:
        sw = CAP.sigma_dentro(datos, metodo)
        rc = CAP.calcular(datos["valor"], sw, lie, lse, objetivo, alfa, cpk_min)
    except ValueError as e:
        st.error(str(e))
        return
    r = rc["resumen"]
    c = st.columns(5)
    c[0].metric("Cp", num(r["Cp"], 3))
    c[1].metric("Cpk", num(r["Cpk"], 3))
    c[2].metric("Pp", num(r["Pp"], 3))
    c[3].metric("Ppk", num(r["Ppk"], 3))
    c[4].metric("No conformes (largo plazo)", f"{r['ppm_largo']:,.0f} ppm")
    figura(G.fig_capacidad(datos["valor"], r["media"], r["sigma_dentro"], r["sigma_total"], lie, lse, objetivo), "fig_cap")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Índices de capacidad** (σ dentro = %s, σ total = %s)" % (num(r["sigma_dentro"], 6), num(r["sigma_total"], 6)))
        tabla(rc["indices"], 4)
    with c2:
        restantes = {(s.etiqueta, s.regla) for c_ in res.cartas() for s in c_.senales
                     if s.fase == "I" and not c_.excluido[s.posicion]}
        estable = not restantes
        fuera = {e for e, regla in restantes if regla == 1}
        patrones = sorted({e for e, regla in restantes if regla != 1},
                          key=lambda e: (0, int(e), "") if str(e).isdigit() else (1, 0, str(e)))
        if restantes and not fuera:
            # solo quedan señales de patrones: el analista decide si son falsas alarmas tras investigarlas
            st.caption("En la Fase I quedan señales de patrones no aleatorios sin excluir en: " + ", ".join(patrones) + ".")
            if st.checkbox("Se investigaron y no tienen causa asignable: considerarlas falsas alarmas",
                           key="cap_falsas_alarmas",
                           help="Con varias reglas activas aumentan las falsas alarmas (con las reglas 1 a 4, ARL0 ≈ 91). "
                                "Marque la casilla solo si la investigación no encontró una causa asignable."):
                estable = True
        dictamen, accion = CAP.dictamen_aptitud(estable, r["Cpk"], cpk_min)
        if restantes and estable:
            accion += (" Las señales de patrones en " + ", ".join(patrones) +
                       " se investigaron sin hallar causa asignable y se consideran falsas alarmas.")
        st.markdown("**Análisis de aptitud del proceso**")
        (st.success if dictamen == "Estable y capaz" else st.warning)(f"**{dictamen}.** {accion}")
        st.markdown("**Interpretación**")
        vinetas(rc["conclusiones"])
        sup = st.session_state.res_sup or {}
        if sup.get("normalidad") and sup["normalidad"]["normal"] is False:
            st.warning("La prueba de normalidad rechazó el supuesto: la fracción no conforme estimada puede ser "
                       "imprecisa. Compare con la transformación de Box-Cox abajo.")
    st.markdown("**Fracción no conforme**")
    tabla(rc["fraccion"], 2)
    with st.expander("Capacidad con transformación de Box-Cox (datos no normales)"):
        bc = SUP.box_cox(datos["valor"])
        if not bc["posible"]:
            st.info(bc["motivo"])
        else:
            # mismo rango que acepta SUP.box_cox (|λ| ≤ 5); el valor inicial se acota por seguridad
            lam = st.number_input("λ", -5.0, 5.0, float(min(5.0, max(-5.0, round(bc["lambda"], 2)))), 0.05, key="cap_lambda",
                                  help=f"λ de máxima verosimilitud = {bc['lambda']:.3f}")
            try:
                rbc = CAP.calcular_boxcox(datos, lam, metodo, lie, lse, objetivo, alfa, cpk_min)
                tabla(rbc["indices"], 4)
                tabla(rbc["fraccion"], 2)
                rc["boxcox"] = {"lambda": lam, "indices": rbc["indices"], "fraccion": rbc["fraccion"]}
            except ValueError as e:
                st.error(str(e))
    with st.expander("Valores mínimos recomendados de Cpk (Montgomery, 2020)"):
        tabla(CAP.MINIMOS_RECOMENDADOS, 2)
    rc.update(valores=datos["valor"].to_numpy(), lie=lie, lse=lse, objetivo=objetivo, dictamen=(dictamen, accion),
              estable=estable)
    st.session_state.res_cap = rc


# ---------------------------------------------------------------------------
# Módulo 6. Diseño de gráficos (ARL / ATS)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Simulando longitudes de corrida…")
def _perfil_cacheado(esquemas_items: tuple, deltas: tuple, n: int, repeticiones: int):
    esquemas = {nombre: dict(cfg) for nombre, cfg in esquemas_items}
    return DIS.perfil_arl(esquemas, list(deltas), n=n, repeticiones=repeticiones)


def _lista_numeros(texto: str, defecto):
    try:
        valores = [float(t.strip().replace(",", ".")) for t in texto.split(";") if t.strip()]
        return valores or defecto
    except ValueError:
        st.warning("Escriba los valores separados por punto y coma, p. ej. 0.5; 1; 1.5; 2")
        return defecto


def pagina_diseno():
    st.header("6. Diseño de gráficos de control: ARL0, ARL1 y ATS")
    st.caption("Conecte el desempeño estadístico del gráfico con las decisiones de muestreo: tamaño n, frecuencia h "
               "y carga muestral n/h.")
    n_datos = D.tamano_subgrupo(st.session_state.largo)[0] if st.session_state.largo is not None else 5
    t1, t2, t3, t4 = st.tabs(["Gráfico X̄ / I (media)", "Dispersión (R y S)", "Atributos (p y c)",
                              "Comparar esquemas (simulación)"])
    with t1:
        c = st.columns(4)
        n = c[0].number_input("Tamaño de muestra n", 1, 50, max(1, n_datos), key="dis_n")
        h = c[1].number_input("Intervalo entre muestras h (horas)", 0.05, 24.0, 1.0, 0.05, key="dis_h")
        L = c[2].number_input("Límites L (σ)", 1.0, 5.0, 3.0, 0.05, key="dis_L")
        delta = c[3].number_input("Corrimiento a detectar δ (σ)", 0.1, 5.0, 1.0, 0.1, key="dis_delta")
        arl0, arl1 = DIS.arl_xbar(n, 0, L), DIS.arl_xbar(n, delta, L)
        m = st.columns(4)
        m[0].metric("ARL0", num(arl0, 1), help="Muestras promedio entre falsas alarmas")
        m[1].metric("ATS0", f"{num(arl0 * h, 1)} h")
        m[2].metric(f"ARL1 (δ = {delta:g}σ)", num(arl1, 2))
        m[3].metric(f"ATS1 (δ = {delta:g}σ)", f"{num(arl1 * h, 2)} h")
        vinetas(DIS.interpretar_diseno(n, h, L, delta))
        st.markdown("**Curvas ARL y curva OC para varios tamaños de muestra**")
        txt_n = st.text_input("Tamaños de muestra a comparar (separados por ;)", "1; 3; 5; 10", key="dis_ns")
        txt_d = st.text_input("Corrimientos δ (separados por ;)", "0; 0.25; 0.5; 0.75; 1; 1.5; 2; 2.5; 3", key="dis_ds")
        ns = [int(v) for v in _lista_numeros(txt_n, [1, 3, 5, 10]) if v >= 1]
        ds = _lista_numeros(txt_d, [0, 0.5, 1, 1.5, 2, 3])
        t_arl = DIS.tabla_arl_xbar(ns, ds, L, h)
        g1, g2 = st.columns(2)
        with g1:
            figura(G.fig_curvas_arl(t_arl), "fig_arl")
        with g2:
            d_fino = np.round(np.linspace(0, 3, 61), 3)
            figura(G.fig_curva_oc_carta(DIS.tabla_arl_xbar(ns, d_fino, L)), "fig_oc_carta")
        with st.expander("Tabla de ARL y ATS"):
            tabla(t_arl, 4)
        st.markdown("**Diseño del esquema de muestreo (n, h) con restricción de carga muestral**")
        e = st.columns(3)
        carga = e[0].number_input("Carga máxima (unidades por hora)", 0.1, 1000.0, 5.0, 0.5, key="dis_carga")
        ats0_min = e[1].number_input("ATS0 mínimo aceptable (horas entre falsas alarmas)", 1.0, 100000.0, 100.0, 10.0,
                                     key="dis_ats0")
        txt_h = e[2].text_input("Intervalos h a evaluar (horas, separados por ;)", "0.25; 0.5; 1; 2; 4", key="dis_hs")
        hs = [v for v in _lista_numeros(txt_h, [0.25, 0.5, 1, 2, 4]) if v > 0]
        n_rango = list(range(1, 11))
        t_dis = DIS.tabla_diseno(n_rango, hs, delta, L)
        figura(G.fig_mapa_ats(t_dis), "fig_mapa_ats")
        mejores = DIS.mejor_diseno(n_rango, hs, delta, L, carga, ats0_min)
        if mejores.empty:
            st.warning("Ningún diseño cumple las restricciones; aumente la carga permitida o reduzca el ATS0 mínimo.")
        else:
            b = mejores.iloc[0]
            st.success(f"Diseño recomendado: **n = {int(b['n'])} cada h = {b['h (horas)']:g} h** "
                       f"(carga {b['Carga muestral n/h (unid./hora)']:.2f} unid./h). Detecta un corrimiento de "
                       f"{delta:g}σ en {b['ATS1 (horas)']:.2f} h en promedio, con una falsa alarma cada "
                       f"{b['ATS0 (horas)']:.0f} h.")
            tabla(mejores.head(10), 3)
        st.session_state.res_dis = {"n": n, "h": h, "L": L, "delta": delta, "ARL0": arl0, "ARL1": arl1,
                                    "tabla_arl": t_arl, "tabla_diseno": t_dis,
                                    "mejores": mejores.head(10), "interpretacion": DIS.interpretar_diseno(n, h, L, delta),
                                    "carga": carga, "ats0_min": ats0_min}
    with t2:
        c = st.columns(3)
        n2 = c[0].number_input("n", 2, 25, max(2, min(25, n_datos)), key="dis_n_disp")
        L2 = c[1].number_input("L (σ)", 1.0, 5.0, 3.0, 0.05, key="dis_L_disp")
        txt_l = c[2].text_input("Aumentos de σ: λ = σ1/σ0 (separados por ;)", "1; 1.25; 1.5; 2; 2.5; 3", key="dis_lams")
        lams = [v for v in _lista_numeros(txt_l, [1, 1.5, 2, 3]) if v > 0]
        filas = [{"λ = σ1/σ0": l_, "ARL carta R": DIS.arl_r(n2, l_, L2), "ARL carta S": DIS.arl_s(n2, l_, L2)} for l_ in lams]
        t_disp = pd.DataFrame(filas)
        tabla(t_disp, 3)
        st.caption("Los gráficos de dispersión detectan lentamente aumentos pequeños de σ; la carta S es algo más "
                   "eficiente que R para n > 10.")
        if st.session_state.res_dis is not None:
            st.session_state.res_dis["tabla_dispersion"] = t_disp
    with t3:
        c = st.columns(4)
        n3 = c[0].number_input("n (unidades por muestra)", 1, 100000, 50, key="dis_n_p")
        p0 = c[1].number_input("p0 (en control)", 0.0001, 0.9999, 0.05, 0.005, format="%.4f", key="dis_p0")
        txt_p1 = c[2].text_input("p1 a evaluar (separados por ;)", "0.05; 0.075; 0.10; 0.15; 0.20", key="dis_p1")
        L3 = c[3].number_input("L (σ) ", 1.0, 5.0, 3.0, 0.05, key="dis_L_p")
        p1s = [v for v in _lista_numeros(txt_p1, [0.05, 0.1, 0.2]) if 0 < v < 1]
        tabla(pd.DataFrame([{"p1": p1, **DIS.arl_p(n3, p0, p1, L3)} for p1 in p1s]), 4)
        st.caption("ARL exacto con la distribución binomial. Note que el ARL0 de la carta p no es 370: la "
                   "distribución es discreta y la región de aceptación es un rango de enteros.")
        c = st.columns(3)
        c0 = c[0].number_input("c0 (defectos promedio en control)", 0.1, 1000.0, 10.0, 0.5, key="dis_c0")
        txt_c1 = c[1].text_input("c1 a evaluar (separados por ;)", "10; 12; 15; 20; 25", key="dis_c1")
        L4 = c[2].number_input("L (σ)  ", 1.0, 5.0, 3.0, 0.05, key="dis_L_c")
        c1s = [v for v in _lista_numeros(txt_c1, [10, 15, 20]) if v > 0]
        tabla(pd.DataFrame([{"c1": c1, **DIS.arl_c(c0, c1, L4)} for c1 in c1s]), 4)
    with t4:
        st.write("Compara por simulación Monte Carlo (semilla fija, reproducible) el Shewhart con y sin reglas "
                 "suplementarias, el EWMA y el CUSUM. También valida el cálculo analítico del Shewhart.")
        c = st.columns(4)
        n4 = c[0].number_input("n", 1, 50, max(1, n_datos), key="sim_n")
        reps = c[1].selectbox("Repeticiones", [1000, 2000, 5000], index=1, key="sim_reps")
        lam = c[2].number_input("λ EWMA", 0.01, 1.0, 0.2, 0.01, key="sim_lam")
        Le = c[3].number_input("L EWMA", 1.0, 4.0, 2.86, 0.01, key="sim_Le")
        c2 = st.columns(3)
        k = c2[0].number_input("k CUSUM", 0.05, 2.0, 0.5, 0.05, key="sim_k")
        hc = c2[1].number_input("h CUSUM", 1.0, 10.0, 5.0, 0.5, key="sim_h")
        txt = c2[2].text_input("δ a simular (separados por ;)", "0; 0.5; 1; 1.5; 2; 3", key="sim_deltas")
        deltas = tuple(_lista_numeros(txt, [0, 0.5, 1, 1.5, 2, 3]))
        if st.button("Ejecutar simulación", type="primary", key="btn_sim"):
            esquemas = (
                ("Shewhart (regla 1)", (("tipo", "shewhart"), ("reglas", (1,)))),
                ("Shewhart + reglas 2-4 (Western Electric)", (("tipo", "shewhart"), ("reglas", (1, 2, 3, 4)))),
                (f"EWMA (λ = {lam:g}, L = {Le:g})", (("tipo", "ewma"), ("lam", lam), ("L", Le))),
                (f"CUSUM (k = {k:g}, h = {hc:g})", (("tipo", "cusum"), ("k", k), ("h", hc))),
            )
            perfil = _perfil_cacheado(esquemas, deltas, int(n4), int(reps))
            st.session_state.perfil_sim = perfil
        perfil = st.session_state.get("perfil_sim")
        if perfil is not None:
            figura(G.fig_perfil_arl(perfil), "fig_perfil")
            piv = perfil.pivot(index="δ (en σ)", columns="Esquema", values="ARL simulado")
            piv.insert(0, "Shewhart analítico", [DIS.arl_xbar(int(n4), d, 3.0) for d in piv.index])
            tabla(piv.reset_index(), 2)
            st.caption("El Shewhart simulado debe coincidir con el analítico dentro del error de simulación. Las reglas "
                       "suplementarias y las cartas con memoria detectan antes los corrimientos pequeños, a costa de "
                       "más falsas alarmas (menor ARL0) en el caso de las reglas.")
            if st.session_state.res_dis is not None:
                st.session_state.res_dis["perfil"] = perfil


# ---------------------------------------------------------------------------
# Módulo 7. Muestreo de aceptación
# ---------------------------------------------------------------------------

def pagina_muestreo():
    st.header("7. Muestreo de aceptación")
    t1, t2, t3, t4 = st.tabs(["Diseñar plan simple", "Evaluar un plan simple", "Plan doble", "Plan por variables"])
    modelos = {"binomial": "Binomial (lotes grandes)", "hipergeometrico": "Hipergeométrica (lote pequeño, exacta)",
               "poisson": "Poisson (aproximación)"}
    with t1:
        st.write("Encuentra el plan (n, c) de menor tamaño de muestra que cumple el riesgo del productor en el NCA y el "
                 "riesgo del consumidor en el NCL (método de dos puntos de la curva OC).")
        c = st.columns(5)
        nca = c[0].number_input("NCA / AQL (%)", 0.01, 50.0, 1.0, 0.1, key="m_nca") / 100
        alfa = c[1].number_input("α (riesgo del productor)", 0.001, 0.5, 0.05, 0.01, key="m_alfa")
        ncl = c[2].number_input("NCL / LTPD (%)", 0.02, 80.0, 6.0, 0.5, key="m_ncl") / 100
        beta = c[3].number_input("β (riesgo del consumidor)", 0.001, 0.5, 0.10, 0.01, key="m_beta")
        N = c[4].number_input("Tamaño del lote N", 2, 10_000_000, 2000, 100, key="m_N")
        modelo = st.radio("Modelo de probabilidad", list(modelos), format_func=modelos.get, horizontal=True, key="m_modelo")
        try:
            plan = MUE.disenar_plan_simple(nca, alfa, ncl, beta, N if modelo == "hipergeometrico" else None, modelo)
        except ValueError as e:
            st.error(str(e))
            return
        if plan["n"] > N:
            st.error(f"El plan requiere n = {plan['n']} > N = {N}: inspeccione el 100 % o relaje los riesgos.")
        ev = MUE.evaluar_plan_simple(plan["n"], plan["c"], N, nca, ncl, modelo)
        m = st.columns(4)
        m[0].metric("Tamaño de muestra n", plan["n"])
        m[1].metric("Número de aceptación c", plan["c"])
        m[2].metric("α real", num(plan["α real"], 4))
        m[3].metric("β real", num(plan["β real"], 4))
        oc = MUE.curva_oc(plan["n"], plan["c"], N, modelo if modelo != "hipergeometrico" else "binomial",
                          p_max=min(1.0, ncl * 2.5))
        figura(G.fig_curva_oc_plan({f"n = {plan['n']}, c = {plan['c']}": oc}, nca, ncl, alfa, beta), "fig_oc_plan")
        g1, g2 = st.columns(2)
        with g1:
            figura(G.fig_linea_simple(oc["p"], oc["AOQ"] * 100, "Calidad promedio de salida (AOQ)",
                                      "Fracción defectuosa entrante p (%)", "AOQ (%)", marcar_max=True), "fig_aoq")
        with g2:
            figura(G.fig_linea_simple(oc["p"], oc["ATI"], "Inspección total promedio (ATI)",
                                      "Fracción defectuosa entrante p (%)", "Unidades inspeccionadas por lote"), "fig_ati")
        interpretacion = MUE.interpretar_plan({**plan, **ev}, nca, ncl, N)
        vinetas(interpretacion)
        st.session_state.res_mue = {"plan": {**plan, **ev}, "oc": oc, "nca": nca, "ncl": ncl, "alfa": alfa,
                                    "beta": beta, "N": N, "interpretacion": interpretacion}
    with t2:
        c = st.columns(5)
        n_e = c[0].number_input("n", 1, 100000, 89, key="me_n")
        c_e = c[1].number_input("c", 0, 1000, 2, key="me_c")
        N_e = c[2].number_input("N", 2, 10_000_000, 10000, key="me_N")
        nca_e = c[3].number_input("NCA (%)", 0.01, 50.0, 1.0, 0.1, key="me_nca") / 100
        ncl_e = c[4].number_input("NCL (%)", 0.02, 80.0, 5.0, 0.5, key="me_ncl") / 100
        if n_e > N_e:
            st.error("n no puede superar N.")
        else:
            ev = MUE.evaluar_plan_simple(n_e, c_e, N_e, nca_e, ncl_e)
            tabla_medidas(ev, 5)
            oc_e = MUE.curva_oc(n_e, c_e, N_e)
            figura(G.fig_curva_oc_plan({f"n = {n_e}, c = {c_e}": oc_e}), "fig_oc_eval")
    with t3:
        st.write("Plan doble: se acepta si d1 ≤ c1; se rechaza si d1 > c2; en otro caso se toma una segunda muestra y "
                 "se acepta si d1 + d2 ≤ c2.")
        c = st.columns(4)
        n1 = c[0].number_input("n1", 1, 10000, 50, key="md_n1")
        c1 = c[1].number_input("c1", 0, 100, 1, key="md_c1")
        n2 = c[2].number_input("n2", 1, 10000, 100, key="md_n2")
        c2 = c[3].number_input("c2", 0, 200, 3, key="md_c2")
        if c2 < c1:
            st.error("Debe cumplirse c2 ≥ c1.")
        else:
            td = MUE.curva_doble(n1, c1, n2, c2)
            g1, g2 = st.columns(2)
            with g1:
                figura(G.fig_curva_oc_plan({"Plan doble": td}), "fig_oc_doble")
            with g2:
                figura(G.fig_linea_simple(td["p"], td["ASN"], "Número promedio de muestra (ASN)",
                                          "Fracción defectuosa p (%)", "ASN"), "fig_asn")
    with t4:
        st.write("Plan por variables (método k) para una especificación unilateral: requiere menos unidades que un "
                 "plan por atributos con la misma protección, si la característica es normal.")
        c = st.columns(5)
        nca_v = c[0].number_input("NCA (%)", 0.01, 50.0, 1.0, 0.1, key="mv_nca") / 100
        alfa_v = c[1].number_input("α", 0.001, 0.5, 0.05, 0.01, key="mv_alfa")
        ncl_v = c[2].number_input("NCL (%)", 0.02, 80.0, 6.0, 0.5, key="mv_ncl") / 100
        beta_v = c[3].number_input("β", 0.001, 0.5, 0.10, 0.01, key="mv_beta")
        sigma_c = c[4].radio("σ del proceso", ["conocida", "desconocida"], key="mv_sigma")
        try:
            pv = MUE.plan_variables(nca_v, alfa_v, ncl_v, beta_v, sigma_c == "conocida")
            tabla_medidas(pv, 4)
            try:
                pa = MUE.disenar_plan_simple(nca_v, alfa_v, ncl_v, beta_v)
                st.caption(f"Comparación: el plan por atributos equivalente requiere n = {pa['n']} (c = {pa['c']}).")
            except ValueError:
                pass
        except ValueError as e:
            st.error(str(e))


# ---------------------------------------------------------------------------
# Módulo 8. Reportes
# ---------------------------------------------------------------------------

def conclusiones_automaticas() -> list[str]:
    ss = st.session_state
    t = []
    if ss.res_var is not None:
        t += VAR.interpretar(ss.res_var)[:2]
    if ss.res_attr is not None:
        t += ATR.interpretar(ss.res_attr)[:2]
    sup = ss.res_sup or {}
    if sup.get("normalidad"):
        t.append(sup["normalidad"]["conclusion"])
    if sup.get("independencia"):
        t.append(sup["independencia"]["conclusion"])
    if ss.res_cap is not None:
        t += ss.res_cap["conclusiones"]
        d, a = ss.res_cap["dictamen"]
        t.append(f"Dictamen de aptitud: {d}. {a}")
    if ss.res_dis is not None:
        mj = ss.res_dis.get("mejores")
        if mj is not None and len(mj):
            b = mj.iloc[0]
            t.append(f"Diseño de muestreo sugerido: n = {int(b['n'])} cada {b['h (horas)']:g} h (ATS1 = "
                     f"{b['ATS1 (horas)']:.2f} h para δ = {ss.res_dis['delta']:g}σ; ATS0 = {b['ATS0 (horas)']:.0f} h).")
    if ss.res_mue is not None:
        t.append(ss.res_mue["interpretacion"][0])
    return t


def pagina_reportes():
    st.header("8. Reportes")
    ss = st.session_state
    disponibles = {
        "Datos": ss.largo is not None or ss.tabla_attr is not None,
        "Gráficos por variables": ss.res_var is not None,
        "Gráficos por atributos": ss.res_attr is not None,
        "Validación de supuestos": bool(ss.res_sup),
        "Capacidad": ss.res_cap is not None,
        "Diseño ARL/ATS": ss.res_dis is not None,
        "Muestreo de aceptación": ss.res_mue is not None,
    }
    st.write("El reporte incluye los análisis que haya ejecutado en esta sesión (visite cada módulo para generarlos).")
    cols = st.columns(len(disponibles))
    for c, (nombre, ok) in zip(cols, disponibles.items()):
        c.markdown(f"{'✅' if ok else '⬜'} {nombre}")
    if not disponibles["Datos"]:
        aviso_sin_datos("variables o atributos")
        return
    meta = {k: v for k, v in ss.meta.items()}
    meta["Fuente de datos"] = ss.fuente
    if not any(ss.meta.values()):
        st.info("Complete los datos del caso (empresa, proceso, variable…) en la barra lateral para que aparezcan "
                "en el reporte.")
    auto = conclusiones_automaticas()
    st.subheader("Conclusiones preliminares")
    st.caption("Generadas automáticamente a partir de los resultados. Edítelas y compleméntelas con su juicio de "
               "ingeniería y el conocimiento del proceso.")
    texto = st.text_area("Conclusiones (una por línea)", value="\n".join(auto) if ss.conclusiones_editadas is None
                         else ss.conclusiones_editadas, height=260, key="txt_conclusiones")
    ss.conclusiones_editadas = texto
    conclusiones = [l.strip() for l in texto.splitlines() if l.strip()]

    tablas, secciones = construir_reporte(meta, conclusiones)
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ Descargar Excel con todas las tablas", REP.exportar_excel(tablas, meta, conclusiones),
                       file_name="reporte_cep.xlsx", type="primary", key="dl_excel",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("⬇ Descargar reporte HTML (imprimible a PDF)", REP.reporte_html(meta, secciones).encode("utf-8"),
                       file_name="reporte_cep.html", mime="text/html", key="dl_html")
    c3.download_button("⬇ Límites de Fase I (JSON)",
                       REP.limites_json(ss.res_var, ss.res_attr, meta).encode("utf-8"),
                       file_name="limites_fase1.json", mime="application/json", key="dl_json")
    with st.expander("Vista previa del contenido del reporte"):
        for titulo, bloques in secciones:
            st.markdown(f"**{titulo}**")
            for b in bloques:
                if b[0] == "tabla":
                    tabla(b[1])
                elif b[0] == "lista":
                    vinetas(b[1])
                elif b[0] == "texto":
                    st.write(b[1])


def construir_reporte(meta: dict, conclusiones: list[str]):
    ss = st.session_state
    tablas, secciones = {}, []
    bloques = []
    if ss.largo is not None:
        tablas["Datos (formato largo)"] = ss.largo.drop(columns=[])
        tablas["Descriptiva"] = D.resumen_descriptivo(ss.largo["valor"])
        tablas["Resumen por subgrupo"] = D.resumen_subgrupos(ss.largo)
        n, const = D.tamano_subgrupo(ss.largo)
        bloques += [("texto", f"Datos por variables: {ss.largo['valor'].notna().sum()} mediciones en "
                              f"{ss.largo['orden'].nunique()} subgrupos de tamaño {n}{'' if const else ' (variable)'}."),
                    ("tabla", tablas["Descriptiva"], "Estadística descriptiva"),
                    ("figura", G.fig_corrida(ss.largo))]
    if ss.tabla_attr is not None:
        tablas["Datos atributos"] = ss.tabla_attr
        bloques.append(("texto", f"Datos por atributos ({ss.tipo_conteo}): {len(ss.tabla_attr)} muestras."))
    if ss.registro_limpieza:
        bloques.append(("lista", ss.registro_limpieza))
    secciones.append(("1. Datos", bloques))

    if ss.res_var is not None:
        r = ss.res_var
        tablas["Límites variables"] = tabla_limites(r.cartas())
        tablas["Puntos " + VAR.NOMBRES[r.tipo] + " (posición)"] = r.carta_pos.tabla()
        tablas["Puntos " + VAR.NOMBRES[r.tipo] + " (dispersión)"] = r.carta_disp.tabla()
        tablas["Señales variables"] = REG.tabla_senales(r.cartas())
        excl = pd.DataFrame([{"Subgrupo": e, "Causa asignable": c} for e, c in ss.excl_var.items()])
        if len(excl):
            tablas["Exclusiones variables"] = excl
        b = [("figura", G.fig_par_cartas(r.carta_pos, r.carta_disp, f"Gráficos {VAR.NOMBRES[r.tipo]}")),
             ("tabla", tablas["Límites variables"], "Límites de control"),
             ("texto", f"{r.metodo_sigma} → σ̂ = {r.sigma:.6g}."),
             ("lista", VAR.interpretar(r))]
        if len(tablas["Señales variables"]):
            b.append(("tabla", tablas["Señales variables"], "Señales y patrones"))
        if len(excl):
            b.append(("tabla", excl, "Causas asignables documentadas (Fase I)"))
        if ss.ewma is not None:
            b.append(("figura", G.fig_carta(ss.ewma, zonas=False)))
        if ss.cusum is not None:
            b.append(("figura", G.fig_cusum(*ss.cusum)))
        secciones.append(("2. Gráficos de control por variables", b))
    if ss.res_attr is not None:
        a = ss.res_attr
        tablas["Límites atributos"] = tabla_limites([a.carta])
        tablas["Puntos atributos"] = a.carta.tabla()
        tablas["Señales atributos"] = REG.tabla_senales([a.carta])
        b = [("figura", G.fig_carta(a.carta)), ("tabla", tablas["Límites atributos"], "Límites de control"),
             ("lista", ATR.interpretar(a))]
        excl = pd.DataFrame([{"Muestra": e, "Causa asignable": c} for e, c in ss.excl_attr.items()])
        if len(excl):
            tablas["Exclusiones atributos"] = excl
            b.append(("tabla", excl, "Causas asignables documentadas (Fase I)"))
        if ss.get("pareto") is not None and len(ss.pareto):
            tablas["Pareto defectos"] = ss.pareto
            b += [("figura", G.fig_pareto(ss.pareto)), ("tabla", ss.pareto, "Pareto de tipos de defecto")]
        secciones.append(("3. Gráficos de control por atributos", b))
    sup = ss.res_sup or {}
    if sup:
        b = []
        if sup.get("normalidad"):
            tablas["Normalidad"] = sup["normalidad"]["tabla"]
            b += [("tabla", sup["normalidad"]["tabla"], "Pruebas de normalidad"), ("texto", sup["normalidad"]["conclusion"])]
            if ss.largo is not None:
                excl = set(ss.excl_var)
                d = ss.largo[(ss.largo["fase"] == "I") & ~ss.largo["subgrupo"].astype(str).isin(excl)]
                b.append(("figura", G.fig_qq(SUP.datos_qq(d["valor"]))))
        if sup.get("independencia"):
            tablas["Independencia"] = sup["independencia"]["tabla"]
            b += [("tabla", sup["independencia"]["tabla"], "Pruebas de independencia"),
                  ("texto", sup["independencia"]["conclusion"]), ("figura", G.fig_acf(sup["independencia"]["acf"]))]
        secciones.append(("4. Validación de supuestos", b))
    if ss.res_cap is not None:
        rc = ss.res_cap
        r = rc["resumen"]
        tablas["Capacidad"] = rc["indices"]
        tablas["Fracción no conforme"] = rc["fraccion"]
        d, acc = rc["dictamen"]
        secciones.append(("5. Capacidad del proceso", [
            ("figura", G.fig_capacidad(rc["valores"], r["media"], r["sigma_dentro"], r["sigma_total"], rc["lie"],
                                       rc["lse"], rc["objetivo"])),
            ("tabla", rc["indices"], "Índices de capacidad"), ("tabla", rc["fraccion"], "Fracción no conforme"),
            ("lista", rc["conclusiones"]), ("texto", f"Dictamen de aptitud: {d}. {acc}")]))
    if ss.res_dis is not None:
        rd = ss.res_dis
        tablas["ARL y ATS"] = rd["tabla_arl"]
        tablas["Diseños (n, h)"] = rd["tabla_diseno"]
        b = [("lista", rd["interpretacion"]), ("figura", G.fig_curvas_arl(rd["tabla_arl"]))]
        if rd.get("mejores") is not None and len(rd["mejores"]):
            b.append(("tabla", rd["mejores"], f"Mejores diseños (carga ≤ {rd['carga']:g} unid./h, ATS0 ≥ {rd['ats0_min']:g} h)"))
        if rd.get("tabla_dispersion") is not None:
            tablas["ARL dispersión"] = rd["tabla_dispersion"]
        if rd.get("perfil") is not None:
            tablas["Simulación ARL"] = rd["perfil"]
            b.append(("figura", G.fig_perfil_arl(rd["perfil"])))
        secciones.append(("6. Diseño del gráfico (ARL/ATS)", b))
    if ss.res_mue is not None:
        rm = ss.res_mue
        plan = pd.DataFrame([rm["plan"]]).T.reset_index().rename(columns={"index": "Medida", 0: "Valor"})
        plan["Valor"] = plan["Valor"].astype(str)
        tablas["Plan de muestreo"] = plan
        tablas["Curva OC"] = rm["oc"]
        secciones.append(("7. Muestreo de aceptación", [
            ("lista", rm["interpretacion"]), ("tabla", plan, "Plan diseñado"),
            ("figura", G.fig_curva_oc_plan({f"n = {rm['plan']['n']}, c = {rm['plan']['c']}": rm["oc"]},
                                           rm["nca"], rm["ncl"], rm["alfa"], rm["beta"]))]))
    secciones.append(("Conclusiones preliminares", [("lista", conclusiones)]))
    return tablas, secciones


# ---------------------------------------------------------------------------
# Ayuda
# ---------------------------------------------------------------------------

GLOSARIO = [
    ("Subgrupo racional", "Unidades tomadas de forma que la variación dentro del subgrupo sea solo de causas comunes "
                          "(p. ej., 5 piezas consecutivas de la misma máquina) y las diferencias entre subgrupos "
                          "revelen causas especiales."),
    ("Fase I", "Análisis retrospectivo de datos históricos para estimar límites de control y llevar el proceso a "
               "control estadístico (límites de prueba → límites definitivos)."),
    ("Fase II", "Monitoreo en línea de datos nuevos con los límites congelados de la Fase I."),
    ("Causa común / especial", "Común: variación natural e inherente del proceso. Especial (asignable): perturbación "
                               "identificable que se puede eliminar (error, material, máquina, método)."),
    ("LCS, LC, LCI", "Límite de control superior, línea central y límite de control inferior. NO son límites de "
                     "especificación."),
    ("LIE, LSE", "Límites de especificación inferior y superior: los define el cliente, el diseño o una norma."),
    ("Cp, Cpk", "Capacidad potencial y real con σ dentro de subgrupos (corto plazo)."),
    ("Pp, Ppk", "Desempeño con σ total (largo plazo)."),
    ("Cpl, Cpu", "Capacidad respecto a cada límite de especificación."),
    ("ppm", "Partes por millón no conformes (fracción × 1 000 000)."),
    ("ARL", "Longitud promedio de corrida: número promedio de muestras hasta una señal. ARL0 = en control "
            "(falsas alarmas); ARL1 = fuera de control (detección)."),
    ("ATS", "Tiempo promedio hasta la señal = ARL × h, con h el intervalo entre muestras."),
    ("Carga muestral", "Unidades inspeccionadas por unidad de tiempo, n/h."),
    ("NCA (AQL) / NCL (LTPD)", "Nivel de calidad aceptable y nivel de calidad límite de un plan de muestreo."),
    ("α / β", "Riesgo del productor (rechazar un lote bueno) y riesgo del consumidor (aceptar un lote malo)."),
    ("Curva OC", "Probabilidad de aceptar un lote (o de no detectar un cambio) en función de la calidad real."),
    ("AOQ / AOQL", "Calidad promedio de salida con inspección rectificadora y su valor máximo."),
    ("ASN / ATI", "Número promedio de muestra (planes dobles) e inspección total promedio por lote."),
    ("EWMA / CUSUM", "Gráficos con memoria, más sensibles que Shewhart a corrimientos pequeños."),
]

REFERENCIAS = [
    "Gutiérrez Pulido, H., & De la Vara Salazar, R. (2013). Control estadístico de la calidad y Seis Sigma (3.ª ed.). "
    "McGraw-Hill.",
    "Lucas, J. M., & Saccucci, M. S. (1990). Exponentially weighted moving average control schemes: Properties and "
    "enhancements. Technometrics, 32(1), 1–12.",
    "Montgomery, D. C. (2020). Introduction to statistical quality control (8th ed.). Wiley.",
    "Nelson, L. S. (1984). The Shewhart control chart—Tests for special causes. Journal of Quality Technology, 16(4), "
    "237–239.",
    "Western Electric Company. (1956). Statistical quality control handbook. Western Electric.",
]


def pagina_ayuda():
    st.header("Ayuda y glosario")
    t1, t2, t3, t4 = st.tabs(["Formato de datos", "Glosario", "Constantes", "Referencias y créditos"])
    with t1:
        st.markdown(
            "**Variables, formato ancho** (recomendado, es el de la plantilla): una fila por subgrupo, columnas `x1…x5` "
            "con las mediciones, `subgrupo` y `fase` (I o II).\n\n"
            "**Variables, formato largo:** una fila por medición, columnas `subgrupo`, `valor` y `fase`.\n\n"
            "**Atributos:** una fila por muestra, columnas `muestra`, `unidades_inspeccionadas`, `defectuosos` (o "
            "`defectos`) y `fase`.\n\n"
            "Se aceptan Excel (.xlsx) y CSV con coma o punto y coma como separador y coma o punto decimal.")
        st.markdown("**Errores frecuentes**")
        vinetas([
            "«La carta X̄-R requiere tamaño de subgrupo constante»: elimine subgrupos incompletos en el módulo 1 o use X̄-S.",
            "«Se necesitan al menos 2 subgrupos de Fase I»: revise la columna de fase o el número m de Fase I.",
            "Una columna numérica no aparece: tiene texto mezclado (p. ej. «12,5 g»); deje solo el número.",
            "«Hay muestras con más defectuosos que unidades inspeccionadas»: está contando defectos; use la carta c o u.",
            "El navegador no abre: entre manualmente a http://localhost:8501 mientras la ventana negra esté abierta.",
        ])
    with t2:
        tabla(pd.DataFrame(GLOSARIO, columns=["Término", "Significado"]))
    with t3:
        L = st.number_input("L", 1.0, 5.0, 3.0, 0.05, key="ayuda_L")
        tabla(K.tabla_constantes(25, L), 5)
        st.caption("d2 y d3 calculados por integración numérica de la distribución del rango; c4 exacto con la función gamma.")
    with t4:
        vinetas(REFERENCIAS)
        st.caption(f"{__nombre__} v{__version__}. Desarrollado como trabajo final de Control Estadístico de Procesos, "
                   "Universidad del Magdalena. Complete los datos de contacto del equipo en el manual de usuario.")


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main():
    iniciar_estado()
    pagina = barra_lateral()
    {
        "Inicio": pagina_inicio,
        "1. Gestión de datos": pagina_datos,
        "2. Gráficos por variables": pagina_variables,
        "3. Gráficos por atributos": pagina_atributos,
        "4. Validación de supuestos": pagina_supuestos,
        "5. Capacidad del proceso": pagina_capacidad,
        "6. Diseño de gráficos (ARL/ATS)": pagina_diseno,
        "7. Muestreo de aceptación": pagina_muestreo,
        "8. Reportes": pagina_reportes,
        "Ayuda y glosario": pagina_ayuda,
    }[pagina]()


main()
