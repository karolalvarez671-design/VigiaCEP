"""
Módulo 1. Gestión de datos.

Funciones para:
  * cargar datos desde Excel o CSV (detecta separador y coma decimal),
  * identificar el tipo y el rol probable de cada columna,
  * definir subgrupos racionales (formato largo, ancho, consecutivo o individual),
  * separar Fase I (estimación) y Fase II (monitoreo),
  * limpiar los datos y documentar lo que se hizo,
  * generar tablas resumen.

Formato interno para variables (formato "largo"), una fila por medición:
    subgrupo  etiqueta del subgrupo (texto)
    orden     posición del subgrupo en el tiempo (1, 2, 3, ...)
    valor     medición
    fase      'I' o 'II'
    fila      fila del archivo original (trazabilidad)

Formato interno para atributos, una fila por muestra:
    muestra, orden, n (unidades inspeccionadas o unidades de inspección), conteo, fase, fila
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Lectura de archivos
# ---------------------------------------------------------------------------

EXTENSIONES_EXCEL = (".xlsx", ".xlsm", ".xls")
EXTENSIONES_TEXTO = (".csv", ".txt", ".tsv")


def _nombre(archivo, nombre=None) -> str:
    return nombre or getattr(archivo, "name", None) or str(archivo)


def hojas_excel(archivo, nombre=None) -> list[str]:
    """Lista las hojas de un libro de Excel (vacía si el archivo no es Excel)."""
    if Path(_nombre(archivo, nombre)).suffix.lower() not in EXTENSIONES_EXCEL:
        return []
    if hasattr(archivo, "seek"):
        archivo.seek(0)
    libro = pd.ExcelFile(archivo)
    return list(libro.sheet_names)


def _leer_texto(archivo) -> str:
    if hasattr(archivo, "read"):
        if hasattr(archivo, "seek"):
            archivo.seek(0)
        crudo = archivo.read()
    else:
        crudo = Path(archivo).read_bytes()
    if isinstance(crudo, str):
        return crudo
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            return crudo.decode(codificacion)
        except UnicodeDecodeError:
            continue
    return crudo.decode("utf-8", errors="replace")


def _detectar_separador(texto: str) -> str:
    muestra = "\n".join(texto.splitlines()[:30])
    try:
        return csv.Sniffer().sniff(muestra, delimiters=",;\t|").delimiter
    except csv.Error:
        conteos = {s: muestra.count(s) for s in (";", "\t", "|", ",")}
        return max(conteos, key=conteos.get)


def leer_archivo(archivo, nombre: str | None = None, hoja=None) -> pd.DataFrame:
    """Carga un archivo Excel o CSV y convierte a número las columnas numéricas.

    Acepta coma o punto decimal y separadores coma, punto y coma o tabulador
    (los CSV generados por Excel en español suelen usar ';' y coma decimal).
    """
    ext = Path(_nombre(archivo, nombre)).suffix.lower()
    if ext in EXTENSIONES_EXCEL:
        if hasattr(archivo, "seek"):
            archivo.seek(0)
        df = pd.read_excel(archivo, sheet_name=hoja if hoja is not None else 0)
    elif ext in EXTENSIONES_TEXTO or ext == "":
        texto = _leer_texto(archivo)
        sep = _detectar_separador(texto)
        df = pd.read_csv(io.StringIO(texto), sep=sep, dtype=str, skipinitialspace=True)
    else:
        raise ValueError(f"Formato no soportado: '{ext}'. Use Excel (.xlsx) o CSV (.csv).")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, [not c.lower().startswith("unnamed") for c in df.columns]]
    return convertir_numericos(df).reset_index(drop=True)


def _es_texto(serie: pd.Series) -> bool:
    return serie.dtype == object or pd.api.types.is_string_dtype(serie)


def convertir_numericos(df: pd.DataFrame, umbral: float = 0.8) -> pd.DataFrame:
    """Convierte a número las columnas de texto que son mayoritariamente numéricas.

    Reemplaza la coma decimal por punto. Una columna se convierte si al menos el
    `umbral` (80 %) de sus celdas no vacías se pueden leer como número.
    """
    df = df.copy()
    for col in df.columns:
        if not _es_texto(df[col]):
            continue
        s = df[col].astype("string").str.strip().str.replace(" ", "", regex=False)
        no_vacias = s.notna() & (s != "") & (s.str.lower() != "nan")
        if no_vacias.sum() == 0:
            continue
        candidata = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
        if candidata[no_vacias].notna().mean() >= umbral:
            df[col] = candidata.astype(float)
    return df


# ---------------------------------------------------------------------------
# Identificación de variables
# ---------------------------------------------------------------------------

_PATRONES_ROL = [
    ("Fase", r"\bfase\b|^phase"),
    ("Fecha/hora", r"fecha|hora|date|time|tiempo"),
    ("Tamaño de muestra", r"^n$|^n_|tama[nñ]o|inspeccionad|unidades|lote_n|sample.?size"),
    ("Conteo (defectuosos/defectos)", r"defect|no.?conform|rechaz|conteo|fallas?|errores"),
    ("Subgrupo / muestra", r"subgrupo|muestra|^lote|grupo|^id$|turno|corrida|sample"),
    ("Etiqueta", r"operari|maquina|máquina|linea|línea|observ|nota|causa"),
]


def _rol_sugerido(nombre: str, tipo: str) -> str:
    nom = nombre.strip().lower()
    for rol, patron in _PATRONES_ROL:
        if re.search(patron, nom):
            return rol
    if tipo == "Numérica continua":
        return "Medición (variable de calidad)"
    if tipo == "Numérica entera":
        return "Medición o conteo"
    return "Etiqueta"


def identificar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica cada columna (tipo de dato y rol probable dentro del análisis)."""
    filas = []
    for col in df.columns:
        s = df[col]
        validos = s.dropna()
        if pd.api.types.is_datetime64_any_dtype(s):
            tipo = "Fecha/hora"
        elif pd.api.types.is_numeric_dtype(s):
            enteros = len(validos) > 0 and np.allclose(validos, np.round(validos))
            tipo = "Numérica entera" if enteros else "Numérica continua"
        else:
            convertida = pd.to_datetime(validos.astype(str), errors="coerce", dayfirst=True, format="mixed") \
                if len(validos) else pd.Series(dtype="datetime64[ns]")
            tipo = "Fecha/hora" if len(validos) and convertida.notna().mean() > 0.8 else "Categórica / texto"
        filas.append({
            "Columna": col,
            "Tipo detectado": tipo,
            "Rol sugerido": _rol_sugerido(str(col), tipo),
            "Valores válidos": int(validos.shape[0]),
            "Faltantes": int(s.isna().sum()),
            "Valores distintos": int(validos.nunique()),
            "Ejemplo": str(validos.iloc[0]) if len(validos) else "",
        })
    return pd.DataFrame(filas)


def columnas_numericas(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Subgrupos racionales y fases
# ---------------------------------------------------------------------------

def normalizar_fase(valor) -> str | None:
    """Interpreta etiquetas de fase como 'I', 'II', 1, 2, 'Fase 1', 'F2'..."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    t = str(valor).strip().upper().replace("FASE", "").replace("F", "").strip()
    if t in ("I", "1", "1.0"):
        return "I"
    if t in ("II", "2", "2.0"):
        return "II"
    return None


def _orden_de_aparicion(etiquetas: pd.Series) -> pd.Series:
    vistos = {}
    for e in etiquetas:
        if e not in vistos:
            vistos[e] = len(vistos) + 1
    return etiquetas.map(vistos)


def _etiqueta_texto(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def construir_subgrupos(df: pd.DataFrame, modo: str, col_valor: str | None = None,
                        cols_valor: list[str] | None = None, col_subgrupo: str | None = None,
                        tamano: int | None = None, col_fase: str | None = None,
                        col_orden: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Arma los subgrupos racionales en formato largo.

    modo:
      'largo'        una fila por medición y una columna que identifica el subgrupo
      'ancho'        una fila por subgrupo; las mediciones están en varias columnas
      'consecutivo'  una columna de mediciones agrupada en bloques de `tamano` filas seguidas
      'individual'   cada medición es un subgrupo de tamaño 1 (carta I-MR)

    Devuelve (datos_largos, notas) donde notas describe decisiones tomadas.
    """
    notas: list[str] = []
    base = df.copy()
    base["_fila"] = np.arange(1, len(base) + 1) + 1  # fila de Excel (encabezado = fila 1)
    if col_orden:
        base = base.sort_values(col_orden, kind="stable")
        notas.append(f"Datos ordenados cronológicamente por la columna '{col_orden}'.")

    if modo == "ancho":
        if not cols_valor:
            raise ValueError("Seleccione las columnas que contienen las mediciones de cada subgrupo.")
        etiquetas = base[col_subgrupo].map(_etiqueta_texto) if col_subgrupo else \
            pd.Series([str(i) for i in range(1, len(base) + 1)], index=base.index)
        largo = base.assign(subgrupo=etiquetas.values).melt(
            id_vars=["subgrupo", "_fila"] + ([col_fase] if col_fase else []),
            value_vars=list(cols_valor), var_name="medicion", value_name="valor")
        largo["_pos"] = largo.groupby("subgrupo").cumcount()
        orden_sub = {e: i + 1 for i, e in enumerate(pd.unique(etiquetas.values))}
        largo["orden"] = largo["subgrupo"].map(orden_sub)
        largo = largo.sort_values(["orden", "medicion"], kind="stable")
    else:
        if not col_valor:
            raise ValueError("Seleccione la columna con las mediciones.")
        largo = base[[col_valor, "_fila"] + ([col_subgrupo] if col_subgrupo and modo == "largo" else [])
                     + ([col_fase] if col_fase else [])].copy()
        largo = largo.rename(columns={col_valor: "valor"})
        if modo == "largo":
            if not col_subgrupo:
                raise ValueError("Seleccione la columna que identifica el subgrupo.")
            largo["subgrupo"] = largo[col_subgrupo].map(_etiqueta_texto)
            largo["orden"] = _orden_de_aparicion(largo["subgrupo"])
        elif modo == "consecutivo":
            if not tamano or tamano < 2:
                raise ValueError("Indique un tamaño de subgrupo n >= 2.")
            largo = largo[largo["valor"].notna()].copy()
            completos = (len(largo) // tamano) * tamano
            sobrantes = len(largo) - completos
            if sobrantes:
                notas.append(f"Se descartaron las últimas {sobrantes} mediciones porque no completan "
                             f"un subgrupo de n = {tamano}.")
            largo = largo.iloc[:completos].copy()
            largo["orden"] = np.arange(completos) // tamano + 1
            largo["subgrupo"] = largo["orden"].astype(str)
        elif modo == "individual":
            largo["orden"] = np.arange(1, len(largo) + 1)
            largo["subgrupo"] = largo["orden"].astype(str)
        else:
            raise ValueError(f"Modo de subgrupos desconocido: {modo}")

    largo["valor"] = pd.to_numeric(largo["valor"], errors="coerce")
    if col_fase:
        largo["fase"] = largo[col_fase].map(normalizar_fase)
        sin_fase = largo["fase"].isna().sum()
        if sin_fase:
            notas.append(f"{sin_fase} filas sin fase reconocible se asignaron a la Fase I.")
        largo["fase"] = largo["fase"].fillna("I")
        # Toda la fila del subgrupo debe quedar en una sola fase (se toma la primera)
        largo["fase"] = largo.groupby("orden")["fase"].transform("first")
    else:
        largo["fase"] = "I"

    largo = largo.rename(columns={"_fila": "fila"})
    salida = largo[["subgrupo", "orden", "valor", "fase", "fila"]].sort_values(["orden"], kind="stable")
    return salida.reset_index(drop=True), notas


def asignar_fases(largo: pd.DataFrame, m_fase1: int) -> pd.DataFrame:
    """Los primeros `m_fase1` subgrupos quedan en Fase I y el resto en Fase II."""
    largo = largo.copy()
    largo["fase"] = np.where(largo["orden"] <= m_fase1, "I", "II")
    return largo


def construir_atributos(df: pd.DataFrame, col_conteo: str, col_n: str | None = None,
                        n_constante: float | None = None, col_muestra: str | None = None,
                        col_fase: str | None = None, col_orden: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Arma la tabla de muestras para gráficos por atributos."""
    notas: list[str] = []
    base = df.copy()
    base["_fila"] = np.arange(1, len(base) + 1) + 1
    if col_orden:
        base = base.sort_values(col_orden, kind="stable")
    conteo = pd.to_numeric(base[col_conteo], errors="coerce")
    if col_n:
        n = pd.to_numeric(base[col_n], errors="coerce")
    elif n_constante:
        n = pd.Series(float(n_constante), index=base.index)
    else:
        raise ValueError("Indique la columna con el tamaño de muestra o un tamaño constante.")
    salida = pd.DataFrame({
        "muestra": base[col_muestra].map(_etiqueta_texto) if col_muestra
        else [str(i) for i in range(1, len(base) + 1)],
        "n": n.values,
        "conteo": conteo.values,
        "fila": base["_fila"].values,
    })
    if col_fase:
        fase = base[col_fase].map(normalizar_fase)
        salida["fase"] = fase.fillna("I").values
    else:
        salida["fase"] = "I"
    validas = salida["n"].notna() & salida["conteo"].notna() & (salida["n"] > 0)
    if (~validas).sum():
        notas.append(f"Se descartaron {(~validas).sum()} filas sin tamaño de muestra o sin conteo válido.")
    salida = salida[validas].copy()
    negativos = salida["conteo"] < 0
    if negativos.any():
        notas.append(f"Se descartaron {negativos.sum()} filas con conteo negativo.")
        salida = salida[~negativos]
    salida["orden"] = np.arange(1, len(salida) + 1)
    salida["proporcion"] = salida["conteo"] / salida["n"]
    return salida[["muestra", "orden", "n", "conteo", "proporcion", "fase", "fila"]].reset_index(drop=True), notas


def asignar_fases_atributos(tabla: pd.DataFrame, m_fase1: int) -> pd.DataFrame:
    tabla = tabla.copy()
    tabla["fase"] = np.where(tabla["orden"] <= m_fase1, "I", "II")
    return tabla


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------

def diagnosticar_limpieza(largo: pd.DataFrame) -> dict:
    """Revisa faltantes, subgrupos incompletos y valores atípicos (criterio de Tukey).

    No elimina nada por su cuenta: devuelve el diagnóstico para que el usuario decida
    y justifique cada exclusión (requisito de documentación del informe).
    """
    faltantes = largo[largo["valor"].isna()]
    validos = largo[largo["valor"].notna()]
    tamanos = validos.groupby("orden")["valor"].count()
    moda_n = int(tamanos.mode().iloc[0]) if len(tamanos) else 0
    incompletos = tamanos[tamanos != moda_n]
    q1, q3 = np.percentile(validos["valor"], [25, 75]) if len(validos) else (np.nan, np.nan)
    riq = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * riq, q3 + 1.5 * riq
    atipicos = validos[(validos["valor"] < lim_inf) | (validos["valor"] > lim_sup)]
    return {
        "faltantes": faltantes,
        "n_modal": moda_n,
        "subgrupos_incompletos": incompletos,
        "atipicos": atipicos,
        "cercas_tukey": (lim_inf, lim_sup),
    }


def aplicar_limpieza(largo: pd.DataFrame, quitar_faltantes: bool = True,
                     quitar_incompletos: bool = False, filas_excluir=()) -> tuple[pd.DataFrame, list[str]]:
    """Aplica las decisiones de limpieza y devuelve el registro de lo realizado."""
    registro = []
    datos = largo.copy()
    if quitar_faltantes:
        k = int(datos["valor"].isna().sum())
        if k:
            datos = datos[datos["valor"].notna()]
            registro.append(f"Se eliminaron {k} celdas vacías o no numéricas.")
    if filas_excluir:
        filas_excluir = set(int(f) for f in filas_excluir)
        k = int(datos["fila"].isin(filas_excluir).sum())
        datos = datos[~datos["fila"].isin(filas_excluir)]
        if k:
            registro.append(f"Se excluyeron {k} mediciones marcadas por el usuario (filas {sorted(filas_excluir)}).")
    if quitar_incompletos:
        tamanos = datos.groupby("orden")["valor"].count()
        if len(tamanos):
            moda_n = int(tamanos.mode().iloc[0])
            malos = tamanos[tamanos != moda_n].index
            if len(malos):
                datos = datos[~datos["orden"].isin(malos)]
                registro.append(f"Se eliminaron {len(malos)} subgrupos con tamaño distinto de n = {moda_n}.")
    # Re-numerar el orden para que sea consecutivo
    mapa = {o: i + 1 for i, o in enumerate(sorted(datos["orden"].unique()))}
    datos["orden"] = datos["orden"].map(mapa)
    return datos.reset_index(drop=True), registro


# ---------------------------------------------------------------------------
# Tablas resumen
# ---------------------------------------------------------------------------

def resumen_descriptivo(valores) -> pd.DataFrame:
    """Estadística descriptiva de una variable continua."""
    x = np.asarray(pd.Series(valores).dropna(), dtype=float)
    n = len(x)
    if n == 0:
        return pd.DataFrame({"Estadístico": [], "Valor": []})
    media = x.mean()
    desv = x.std(ddof=1) if n > 1 else np.nan
    filas = [
        ("Número de datos (N)", n),
        ("Media", media),
        ("Mediana", np.median(x)),
        ("Desviación estándar (s)", desv),
        ("Varianza", desv ** 2 if n > 1 else np.nan),
        ("Coeficiente de variación (%)", 100 * desv / media if media != 0 else np.nan),
        ("Mínimo", x.min()),
        ("Cuartil 1", np.percentile(x, 25)),
        ("Cuartil 3", np.percentile(x, 75)),
        ("Máximo", x.max()),
        ("Rango", x.max() - x.min()),
        ("Asimetría", stats.skew(x, bias=False) if n > 2 else np.nan),
        ("Curtosis (exceso)", stats.kurtosis(x, bias=False) if n > 3 else np.nan),
    ]
    return pd.DataFrame(filas, columns=["Estadístico", "Valor"])


def resumen_subgrupos(largo: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos por subgrupo: n, media, rango y desviación estándar."""
    g = largo[largo["valor"].notna()].groupby("orden", sort=True)
    tabla = pd.DataFrame({
        "subgrupo": g["subgrupo"].first(),
        "fase": g["fase"].first(),
        "n": g["valor"].count(),
        "media": g["valor"].mean(),
        "rango": g["valor"].max() - g["valor"].min(),
        "desv": g["valor"].std(ddof=1),
        "minimo": g["valor"].min(),
        "maximo": g["valor"].max(),
    }).reset_index()
    return tabla


def tamano_subgrupo(largo: pd.DataFrame) -> tuple[int, bool]:
    """Devuelve (n típico, ¿es constante?)."""
    tam = largo[largo["valor"].notna()].groupby("orden")["valor"].count()
    if tam.empty:
        return 0, True
    return int(tam.mode().iloc[0]), bool(tam.nunique() == 1)
