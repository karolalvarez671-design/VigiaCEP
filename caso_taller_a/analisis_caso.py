"""
Caso aplicado (Taller metalmecánico A): análisis completo con el núcleo de Vigía CEP.

Lee los datos tomados en el taller (datos/Taller_A_datos.xlsx) y reproduce, paso a paso,
el análisis del informe técnico:

    1. Caracterización de los datos (módulo 1)
    2. Gráficos X̄-R de Fase I con límites de prueba y revisados, Fase II, EWMA y CUSUM (módulo 2)
    3. Gráficos p y u y Pareto de defectos (módulo 3)
    4. Normalidad e independencia (módulo 4)
    5. Capacidad y fracción no conforme (módulo 5)
    6. ARL, ATS y diseño del esquema de muestreo (módulo 6)
    7. Planes de muestreo de aceptación (módulo 7)
    8. Reportes en Excel, HTML y JSON (módulo 8)

Uso, desde la carpeta VigiaCEP y con el entorno del programa activo:
    python caso_taller_a/analisis_caso.py

Escribe en caso_taller_a/resultados/ las figuras PNG del informe, los resultados numéricos
(resultados.json y resumen_resultados.txt) y los reportes del módulo 8. Las figuras PNG
requieren la librería kaleido y Google Chrome (pip install kaleido); sin ellas, el resto se
genera igual.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ_APP = Path(sys.argv[1]) if len(sys.argv) > 1 else AQUI.parent                                  # carpeta VigiaCEP
DATOS = Path(sys.argv[2]) if len(sys.argv) > 2 else AQUI / "datos" / "Taller_A_datos.xlsx"     # libro de datos
SALIDA = Path(sys.argv[3]) if len(sys.argv) > 3 else AQUI / "resultados"                           # resultados
SALIDA.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RAIZ_APP))

from cep import atributos as ATR, capacidad as CAP, datos as D, diseno as DIS, graficos as G  # noqa: E402
from cep import muestreo as MUE, reglas as REG, reportes as REP, supuestos as SUP, variables as VAR  # noqa: E402

# ---------------------------------------------------------------- especificación (hoja Especificaciones)
esp = pd.read_excel(DATOS, sheet_name="Especificaciones", header=0, usecols=[0, 1], names=["campo", "valor"])
esp = dict(zip(esp["campo"].astype(str).str.strip(), esp["valor"]))
NOMINAL, LIE, LSE = float(esp["Medida nominal del brazo (mm)"]), float(esp["LIE (mm)"]), float(esp["LSE (mm)"])

# Causas asignables documentadas en la bitácora de campo y en la planilla del inspector
EXCL_VAR = {"9": "Cambio de disco a las 10:00 sin reajustar el tope (bitácora de campo)",
            "17": "Error de lectura o de anotación en la pieza 3, anotada como 442 mm (bitácora de campo)"}
EXCL_ATR = {"12": "Lote nuevo de electrodos E6013 sin resguardo contra la humedad (planilla del inspector)"}
REGLAS = [1, 2, 3, 4]                         # Western Electric
TIPOS_DEFECTO = ["porosidad", "cordon_incompleto", "salpicadura_escoria", "rebaba_filo", "pintura_defectuosa",
                 "perforacion_desalineada", "falta_escuadra", "otros"]
H_ACTUAL, N_ACTUAL, DELTA = 1 / 3, 5, 1.0      # esquema de la Fase I: n = 5 cada 20 minutos; corrimiento de 1σ
LOTE = 200                                    # lote de despacho: unas dos semanas de producción (95 a 115 por semana)

R = {"especificacion": {"nominal": NOMINAL, "LIE": LIE, "LSE": LSE}, "exclusiones_variables": EXCL_VAR,
     "exclusiones_atributos": EXCL_ATR}

# ---------------------------------------------------------------- 1. datos por variables
dfv = D.leer_archivo(DATOS, hoja="Corte_variables")
largo, notas_var = D.construir_subgrupos(dfv, "ancho", cols_valor=["x1", "x2", "x3", "x4", "x5"],
                                         col_subgrupo="subgrupo", col_fase="fase")
largo, registro = D.aplicar_limpieza(largo)
f1 = largo[largo.fase == "I"]
R["descriptiva_f1"] = D.resumen_descriptivo(f1["valor"]).set_index("Estadístico")["Valor"].to_dict()
R["descriptiva_todo"] = D.resumen_descriptivo(largo["valor"]).set_index("Estadístico")["Valor"].to_dict()
diag = D.diagnosticar_limpieza(largo)
R["atipicos"] = diag["atipicos"][["fila", "subgrupo", "valor"]].to_dict("records")
R["cercas"] = diag["cercas_tukey"]

# ---------------------------------------------------------------- 2. gráficos por variables
prueba = VAR.carta_xbar_r(largo, reglas_pos=REGLAS)
rev = VAR.carta_xbar_r(largo, excluidos=list(EXCL_VAR), reglas_pos=REGLAS)


def resumen_xbar_r(res):
    return {"media": res.media, "rbar": float(res.carta_disp.lc[0]), "sigma": res.sigma,
            "lim_x": (float(res.carta_pos.lci[0]), float(res.carta_pos.lcs[0])),
            "lim_r": (float(res.carta_disp.lci[0]), float(res.carta_disp.lcs[0])),
            "senales": REG.tabla_senales(res.cartas()).to_dict("records")}


R["prueba"], R["revisada"] = resumen_xbar_r(prueba), resumen_xbar_r(rev)
ewma = VAR.carta_ewma(largo, rev.media, rev.sigma, 0.2, 2.86)
c_sup, c_inf = VAR.carta_cusum(largo, rev.media, rev.sigma, 0.5, 5.0)
R["ewma_senales"] = [s.etiqueta for s in ewma.senales]
R["cusum_senales"] = sorted({s.etiqueta for s in c_sup.senales + c_inf.senales}, key=int)
R["primera_senal_fase2"] = {
    "Shewhart regla 1": min([int(s.etiqueta) for s in rev.carta_pos.senales if s.fase == "II" and s.regla == 1], default=None),
    "Shewhart reglas 1-4": min([int(s.etiqueta) for s in rev.carta_pos.senales if s.fase == "II"], default=None),
    "EWMA": min([int(e) for e in R["ewma_senales"] if int(e) > 25], default=None),
    "CUSUM": min([int(e) for e in R["cusum_senales"] if int(e) > 25], default=None),
}

# ---------------------------------------------------------------- 4. supuestos (Fase I depurada)
limpio = largo[(largo.fase == "I") & ~largo.subgrupo.isin(list(EXCL_VAR))]
rn = SUP.pruebas_normalidad(limpio["valor"])
R["normalidad"] = {"tabla": rn["tabla"].to_dict("records"), "asimetria": rn["asimetria"], "curtosis": rn["curtosis"],
                   "conclusion": rn["conclusion"], "distintos": rn["valores_distintos"], "n": rn["n"]}
R["descriptiva_f1_depurada"] = D.resumen_descriptivo(limpio["valor"]).set_index("Estadístico")["Valor"].to_dict()
medias = limpio.groupby("orden")["valor"].mean()
ri_ind = SUP.pruebas_independencia(limpio.sort_values("orden")["valor"])
ri_med = SUP.pruebas_independencia(medias)
R["independencia_individuales"] = {"tabla": ri_ind["tabla"].to_dict("records"), "conclusion": ri_ind["conclusion"]}
R["independencia_medias"] = {"tabla": ri_med["tabla"].to_dict("records"), "conclusion": ri_med["conclusion"]}
R["normalidad_medias"] = {"tabla": SUP.pruebas_normalidad(medias)["tabla"].to_dict("records")}

# ---------------------------------------------------------------- 5. capacidad
sw = CAP.sigma_dentro(limpio, "R")
rc = CAP.calcular(limpio["valor"], sw, LIE, LSE, NOMINAL)
R["capacidad"] = {"resumen": rc["resumen"], "indices": rc["indices"].to_dict("records"),
                  "fraccion": rc["fraccion"].to_dict("records"), "conclusiones": rc["conclusiones"]}
R["dictamen"] = CAP.dictamen_aptitud(True, rc["resumen"]["Cpk"])


def ppm(mu, sg):
    from scipy import stats
    return 1e6 * (stats.norm.cdf((LIE - mu) / sg) + stats.norm.sf((LSE - mu) / sg))


semi = (LSE - LIE) / 2
R["escenarios_mejora"] = {   # cálculos con el modelo normal (no son datos observados)
    "actual": {"media": rc["resumen"]["media"], "sigma": sw, "Cpk": rc["resumen"]["Cpk"], "ppm": ppm(rc["resumen"]["media"], sw)},
    "centrado": {"media": NOMINAL, "sigma": sw, "Cpk": semi / (3 * sw), "ppm": ppm(NOMINAL, sw)},
    "centrado_y_sigma_0.6": {"media": NOMINAL, "sigma": 0.6, "Cpk": semi / (3 * 0.6), "ppm": ppm(NOMINAL, 0.6)},
}

# ---------------------------------------------------------------- 6. ARL / ATS
R["arl_actual"] = {"ARL0": DIS.arl_xbar(N_ACTUAL, 0), "ARL1": DIS.arl_xbar(N_ACTUAL, DELTA),
                   "ATS0_h": DIS.arl_xbar(N_ACTUAL, 0) * H_ACTUAL, "ATS1_h": DIS.arl_xbar(N_ACTUAL, DELTA) * H_ACTUAL,
                   "carga": N_ACTUAL / H_ACTUAL, "delta_mm": DELTA * sw}
t_arl = DIS.tabla_arl_xbar([1, 3, 5, 10], [0, 0.5, 1, 1.5, 2, 3], 3.0, 1.0)
R["tabla_arl"] = t_arl.to_dict("records")
hs = [0.25, 0.5, 1, 2]
t_dis = DIS.tabla_diseno(list(range(1, 11)), hs, DELTA, 3.0)
mejores = DIS.mejor_diseno(list(range(1, 11)), hs, DELTA, 3.0, carga_maxima=6, ats0_minimo=100)
R["mejores_disenos"] = mejores.head(6).to_dict("records")
perfil = DIS.perfil_arl({
    "Shewhart X̄ (regla 1)": {"tipo": "shewhart", "reglas": (1,)},
    "Shewhart + reglas 2-4": {"tipo": "shewhart", "reglas": (1, 2, 3, 4)},
    "EWMA (λ = 0.2, L = 2.86)": {"tipo": "ewma", "lam": 0.2, "L": 2.86},
    "CUSUM (k = 0.5, h = 5)": {"tipo": "cusum", "k": 0.5, "h": 5.0},
}, [0, 0.5, 1, 1.5, 2], n=5, repeticiones=4000)
R["perfil"] = perfil.to_dict("records")

# ---------------------------------------------------------------- 3. atributos
dfa = D.leer_archivo(DATOS, hoja="Inspeccion_atributos")
tp_, _ = D.construir_atributos(dfa, "soportes_defectuosos", col_n="soportes_inspeccionados", col_muestra="muestra",
                               col_fase="fase")
tu_, _ = D.construir_atributos(dfa, "defectos_totales", col_n="soportes_inspeccionados", col_muestra="muestra",
                               col_fase="fase")
cp_prueba = ATR.carta_p(tp_, reglas=REGLAS)
cu_prueba = ATR.carta_u(tu_, reglas=REGLAS)
cp_rev = ATR.carta_p(tp_, excluidos=list(EXCL_ATR), reglas=REGLAS)
cu_rev = ATR.carta_u(tu_, excluidos=list(EXCL_ATR), reglas=REGLAS)
R["p"] = {"prueba": cp_prueba.parametro, "revisada": cp_rev.parametro, "notas": cp_rev.notas,
          "senales_prueba": REG.tabla_senales([cp_prueba.carta]).to_dict("records"),
          "senales_rev": REG.tabla_senales([cp_rev.carta]).to_dict("records"),
          "lcs_rango": (float(np.min(cp_rev.carta.lcs)), float(np.max(cp_rev.carta.lcs)))}
R["u"] = {"prueba": cu_prueba.parametro, "revisada": cu_rev.parametro,
          "senales_prueba": REG.tabla_senales([cu_prueba.carta]).to_dict("records"),
          "senales_rev": REG.tabla_senales([cu_rev.carta]).to_dict("records"),
          "lcs_rango": (float(np.min(cu_rev.carta.lcs)), float(np.max(cu_rev.carta.lcs)))}
R["n_promedio_attr"] = float(tp_["n"].mean())
tpar = ATR.pareto(dfa[TIPOS_DEFECTO].sum())
R["pareto"] = tpar.to_dict("records")
R["n_para_lci"] = (1 - cp_rev.parametro) * 9 / cp_rev.parametro

# ---------------------------------------------------------------- 7. muestreo de aceptación
planes = {}
for nombre, (aql, ltpd) in {"A: NCA 2.5 % / NCL 10 %": (0.025, 0.10), "B: NCA 2.5 % / NCL 15 %": (0.025, 0.15)}.items():
    pl = MUE.disenar_plan_simple(aql, 0.05, ltpd, 0.10, LOTE, "hipergeometrico")
    ev = MUE.evaluar_plan_simple(pl["n"], pl["c"], LOTE, aql, ltpd, "hipergeometrico")
    ev["Pa con p = p̄ actual"] = float(MUE.prob_aceptacion(pl["n"], pl["c"], cp_rev.parametro, LOTE, "hipergeometrico"))
    planes[nombre] = {**pl, **ev, "N": LOTE}
R["planes"] = planes

# ---------------------------------------------------------------- figuras PNG del informe
_sin_png = []


def png(fig, nombre, ancho=1000, alto=None):
    if _sin_png:
        return
    try:
        fig.write_image(SALIDA / f"{nombre}.png", width=ancho, height=alto or fig.layout.height or 420, scale=2)
    except Exception as e:  # kaleido o Chrome no disponibles: se omiten solo las figuras
        _sin_png.append(str(e))
        print("Aviso: no se exportaron las figuras PNG (instale kaleido y Google Chrome). Detalle:", str(e)[:200])


png(G.fig_par_cartas(prueba.carta_pos, prueba.carta_disp, None), "f1_xbar_r_prueba", alto=600)
png(G.fig_par_cartas(rev.carta_pos, rev.carta_disp, None), "f2_xbar_r_revisada_fase2", alto=600)
png(G.fig_carta(ewma, zonas=False, titulo=""), "f3_ewma", alto=380)
png(G.fig_cusum(c_sup, c_inf, titulo=""), "f4_cusum", alto=380)
png(G.fig_histograma(limpio["valor"], ""), "f5a_histograma", ancho=620, alto=380)
png(G.fig_qq(SUP.datos_qq(limpio["valor"]), titulo=""), "f5b_qq", ancho=620, alto=380)
png(G.fig_acf(ri_ind["acf"], titulo=""), "f6_acf", alto=320)
r = rc["resumen"]
png(G.fig_capacidad(limpio["valor"], r["media"], r["sigma_dentro"], r["sigma_total"], LIE, LSE, NOMINAL, titulo=""),
    "f7_capacidad", alto=440)
png(G.fig_curvas_arl(t_arl, titulo=""), "f8_arl", alto=400)
png(G.fig_mapa_ats(t_dis, titulo=""), "f9_mapa_ats", alto=380)
png(G.fig_perfil_arl(perfil, titulo=""), "f10_perfil_arl", alto=400)
png(G.fig_carta(cp_rev.carta, titulo=""), "f11_p", alto=400)
png(G.fig_carta(cu_rev.carta, titulo=""), "f12_u", alto=400)
png(G.fig_pareto(tpar, titulo=""), "f13_pareto", alto=420)
pl = planes["A: NCA 2.5 % / NCL 10 %"]
oc = MUE.curva_oc(pl["n"], pl["c"], LOTE, "binomial", p_max=0.25)
png(G.fig_curva_oc_plan({f"n = {pl['n']}, c = {pl['c']}": oc}, 0.025, 0.10, 0.05, 0.10, titulo=""), "f14_oc", alto=400)

# ---------------------------------------------------------------- 8. reportes (mismas funciones del módulo 8)
meta = {"Empresa": "Taller metalmecánico A (Santa Marta)",
        "Proceso / operación": "Corte del brazo horizontal en la Tronzadora 1",
        "Producto": "Soporte para unidad condensadora de aire acondicionado",
        "Característica de calidad": "Longitud del brazo horizontal (nominal 450 mm, LIE 447 mm, LSE 453 mm)",
        "Unidad de medida": "mm", "Fuente de datos": f"Archivo: {DATOS.name}"}


def limites(cartas):
    filas = []
    for c in cartas:
        x = c.resumen_limites()
        filas.append({"Gráfico": c.titulo, "LCI": x["LCI"], "LC": x["LC"], "LCS": x["LCS"]})
    return pd.DataFrame(filas)


excl_v = pd.DataFrame([{"Subgrupo": k, "Causa asignable": v} for k, v in EXCL_VAR.items()])
excl_a = pd.DataFrame([{"Muestra": k, "Causa asignable": v} for k, v in EXCL_ATR.items()])
plan_tab = pd.DataFrame([{"Plan": k, "N": v["N"], "n": v["n"], "c": v["c"], "α real": v["α real"],
                          "β real": v["β real"], "AOQL": v["AOQL"], "Pa con p̄ actual": v["Pa con p = p̄ actual"]}
                         for k, v in planes.items()])
d_ap, acc = R["dictamen"]
conclusiones = (VAR.interpretar(rev)[:2] + ATR.interpretar(cp_rev)[:2] + [rn["conclusion"], ri_ind["conclusion"]]
                + rc["conclusiones"] + [f"Dictamen de aptitud: {d_ap}. {acc}"])
tablas = {
    "Datos (formato largo)": largo, "Descriptiva": D.resumen_descriptivo(largo["valor"]),
    "Resumen por subgrupo": D.resumen_subgrupos(largo),
    "Límites de prueba": limites(prueba.cartas()), "Límites revisados": limites(rev.cartas()),
    "Señales variables": REG.tabla_senales(rev.cartas()), "Exclusiones variables": excl_v,
    "Datos atributos (p)": tp_, "Límites carta p": limites([cp_rev.carta]),
    "Señales atributos": REG.tabla_senales([cp_rev.carta]), "Exclusiones atributos": excl_a,
    "Límites carta u": limites([cu_rev.carta]), "Pareto defectos": tpar,
    "Normalidad": rn["tabla"], "Independencia": ri_ind["tabla"],
    "Capacidad": rc["indices"], "Fracción no conforme": rc["fraccion"],
    "ARL y ATS": t_arl, "Diseños (n, h)": mejores.head(10), "Comparación de esquemas": perfil,
    "Planes de aceptación": plan_tab, "Curva OC plan A": oc,
}
secciones = [
    ("1. Datos", [("texto", f"Datos por variables: {largo['valor'].notna().sum()} mediciones en "
                           f"{largo['orden'].nunique()} subgrupos de tamaño 5 (25 de Fase I y 15 de Fase II)."),
                  ("tabla", tablas["Descriptiva"], "Estadística descriptiva"), ("figura", G.fig_corrida(largo)),
                  ("texto", f"Datos por atributos: {len(tp_)} jornadas de inspección final, del 13/10/2026 al 11/11/2026 (20 de Fase I y 10 de Fase II).")]),
    ("2. Gráficos de control por variables", [
        ("figura", G.fig_par_cartas(rev.carta_pos, rev.carta_disp, "Gráficos X̄-R")),
        ("tabla", tablas["Límites revisados"], "Límites de control revisados"),
        ("texto", f"{rev.metodo_sigma} → σ̂ = {rev.sigma:.6g}."), ("lista", VAR.interpretar(rev)),
        ("tabla", excl_v, "Causas asignables documentadas (Fase I)"),
        ("figura", G.fig_carta(ewma, zonas=False)), ("figura", G.fig_cusum(c_sup, c_inf))]),
    ("3. Gráficos de control por atributos", [
        ("figura", G.fig_carta(cp_rev.carta)), ("tabla", tablas["Límites carta p"], "Límites de control"),
        ("lista", ATR.interpretar(cp_rev)), ("tabla", excl_a, "Causas asignables documentadas (Fase I)"),
        ("figura", G.fig_carta(cu_rev.carta)), ("figura", G.fig_pareto(tpar)),
        ("tabla", tpar, "Pareto de tipos de defecto")]),
    ("4. Validación de supuestos", [
        ("tabla", rn["tabla"], "Pruebas de normalidad"), ("texto", rn["conclusion"]),
        ("figura", G.fig_qq(SUP.datos_qq(limpio["valor"]))), ("tabla", ri_ind["tabla"], "Pruebas de independencia"),
        ("texto", ri_ind["conclusion"]), ("figura", G.fig_acf(ri_ind["acf"]))]),
    ("5. Capacidad del proceso", [
        ("figura", G.fig_capacidad(limpio["valor"], r["media"], r["sigma_dentro"], r["sigma_total"], LIE, LSE, NOMINAL)),
        ("tabla", rc["indices"], "Índices de capacidad"), ("tabla", rc["fraccion"], "Fracción no conforme"),
        ("lista", rc["conclusiones"]), ("texto", f"Dictamen de aptitud: {d_ap}. {acc}")]),
    ("6. Diseño de gráficos (ARL/ATS)", [
        ("tabla", t_arl, "ARL del gráfico X̄"), ("figura", G.fig_mapa_ats(t_dis)),
        ("tabla", mejores.head(6), "Mejores diseños con carga de hasta 6 piezas por hora"),
        ("figura", G.fig_perfil_arl(perfil))]),
    ("7. Muestreo de aceptación", [
        ("tabla", plan_tab, f"Planes simples para lotes de {LOTE} soportes"),
        ("figura", G.fig_curva_oc_plan({f"n = {pl['n']}, c = {pl['c']}": oc}, 0.025, 0.10, 0.05, 0.10))]),
    ("Conclusiones preliminares", [("lista", conclusiones)]),
]
(SALIDA / "reporte_caso.xlsx").write_bytes(REP.exportar_excel(tablas, meta, conclusiones))
(SALIDA / "reporte_caso.html").write_text(REP.reporte_html(meta, secciones), encoding="utf-8")
(SALIDA / "limites_fase1.json").write_text(REP.limites_json(rev, cp_rev, meta), encoding="utf-8")


# ---------------------------------------------------------------- resultados numéricos
def conv(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


(SALIDA / "resultados.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=conv), encoding="utf-8")
cap = rc["resumen"]
lineas = [
    "RESULTADOS PRINCIPALES — CASO TALLER METALMECÁNICO A (Vigía CEP)",
    f"Fase I, límites de prueba: x̿ = {prueba.media:.3f} mm; R̄ = {R['prueba']['rbar']:.3f} mm; σ̂ = {prueba.sigma:.3f} mm; "
    f"X̄ de {R['prueba']['lim_x'][0]:.3f} a {R['prueba']['lim_x'][1]:.3f} mm; R hasta {R['prueba']['lim_r'][1]:.3f} mm.",
    f"Fase I, límites revisados (sin subgrupos {', '.join(EXCL_VAR)}): x̿ = {rev.media:.3f} mm; R̄ = {R['revisada']['rbar']:.3f} mm; "
    f"σ̂ = {rev.sigma:.3f} mm; X̄ de {R['revisada']['lim_x'][0]:.3f} a {R['revisada']['lim_x'][1]:.3f} mm; "
    f"R hasta {R['revisada']['lim_r'][1]:.3f} mm.",
    f"Fase II, primera señal: {R['primera_senal_fase2']}.",
    f"Capacidad: Cp = {cap['Cp']:.3f}; Cpk = {cap['Cpk']:.3f}; Pp = {cap['Pp']:.3f}; Ppk = {cap['Ppk']:.3f}; "
    f"ppm esperados a corto plazo = {cap['ppm_corto']:.0f}; a largo plazo = {cap['ppm_largo']:.0f}. Dictamen: {d_ap}.",
    f"Carta p: p̄ de prueba = {cp_prueba.parametro:.4f}; revisada (sin día 12) = {cp_rev.parametro:.4f}. "
    f"Carta u: ū revisada = {cu_rev.parametro:.4f}.",
    f"ARL del esquema n = 5 cada 20 min: ARL0 = {R['arl_actual']['ARL0']:.1f}; ARL1(1σ) = {R['arl_actual']['ARL1']:.2f}; "
    f"ATS1 = {R['arl_actual']['ATS1_h']:.2f} h; carga = {R['arl_actual']['carga']:.0f} piezas/h.",
] + [f"Plan {k}: n = {v['n']}, c = {v['c']}, α = {v['α real']:.3f}, β = {v['β real']:.3f}, "
     f"Pa con p̄ actual = {v['Pa con p = p̄ actual']:.3f}." for k, v in planes.items()]
(SALIDA / "resumen_resultados.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
print("\n".join(lineas))
print("Listo: resultados en", SALIDA)
