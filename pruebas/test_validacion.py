"""
Validación del núcleo estadístico contra resultados publicados y propiedades teóricas.

Ejecutar desde la carpeta del proyecto:   python -m pytest pruebas -v
Cada prueba compara un cálculo del software con un valor conocido (Montgomery, 2020)
o con una propiedad teórica (ARL0 = 370.4 con límites 3σ, simulación vs. fórmula, etc.).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from cep import atributos as ATR  # noqa: E402
from cep import capacidad as CAP  # noqa: E402
from cep import constantes as K  # noqa: E402
from cep import datos as D  # noqa: E402
from cep import diseno as DIS  # noqa: E402
from cep import muestreo as MUE  # noqa: E402
from cep import reglas as REG  # noqa: E402
from cep import supuestos as SUP  # noqa: E402
from cep import variables as VAR  # noqa: E402

PRUEBA = RAIZ / "datos_prueba"
CASO = RAIZ / "caso_taller_a" / "datos" / "Taller_A_datos.xlsx"

# Tabla del apéndice de Montgomery (3 decimales)
D2_TABLA = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078,
            15: 3.472, 20: 3.735, 25: 3.931}
D3_TABLA = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797,
            15: 0.756, 20: 0.729, 25: 0.708}
C4_TABLA = {2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 10: 0.9727, 25: 0.9896}


def anillos():
    df = D.leer_archivo(PRUEBA / "anillos_piston_montgomery.csv")
    largo, _ = D.construir_subgrupos(df, "ancho", cols_valor=["x1", "x2", "x3", "x4", "x5"], col_subgrupo="muestra")
    return largo


def jugo():
    df = D.leer_archivo(PRUEBA / "jugo_naranja_montgomery.csv")
    return D.construir_atributos(df, "defectuosos", col_n="n", col_muestra="muestra")[0]


def tarjetas():
    df = D.leer_archivo(PRUEBA / "tarjetas_circuito_montgomery.csv")
    return D.construir_atributos(df, "defectos", col_n="unidades_inspeccion", col_muestra="muestra")[0]


# --------------------------------------------------------------------------- constantes
@pytest.mark.parametrize("n", sorted(D2_TABLA))
def test_constantes_coinciden_con_tabla(n):
    assert K.d2(n) == pytest.approx(D2_TABLA[n], abs=6e-4)
    assert K.d3(n) == pytest.approx(D3_TABLA[n], abs=6e-4)


@pytest.mark.parametrize("n", sorted(C4_TABLA))
def test_c4(n):
    assert K.c4(n) == pytest.approx(C4_TABLA[n], abs=6e-5)


def test_constantes_derivadas_n5():
    assert K.A2(5) == pytest.approx(0.577, abs=1e-3)
    assert K.D4(5) == pytest.approx(2.114, abs=1e-3)
    assert K.D3(5) == 0.0
    assert K.A3(5) == pytest.approx(1.427, abs=1e-3)
    assert K.B4(5) == pytest.approx(2.089, abs=1e-3)
    assert K.D3(7) == pytest.approx(0.076, abs=1e-3)


# --------------------------------------------------------------------------- variables
def test_xbar_r_anillos_montgomery():
    """Montgomery: x̿ = 74.001, R̄ = 0.023, X̄ [73.988, 74.014], R LCS = 0.049; todo en control.

    El texto redondea x̿ y R̄ antes de multiplicar por A2 (74.001 + 0.577·0.023 = 74.0143); el software usa
    los valores sin redondear (74.00118 + 0.5768·0.02324 = 74.0146), por eso la tolerancia es 0.001.
    """
    res = VAR.carta_xbar_r(anillos())
    assert res.media == pytest.approx(74.00118, abs=1e-5)
    rbar = res.carta_disp.lc[0]
    assert rbar == pytest.approx(0.02324, abs=1e-5)
    assert res.carta_pos.lcs[0] == pytest.approx(74.014, abs=1e-3)
    assert res.carta_pos.lci[0] == pytest.approx(73.988, abs=1e-3)
    assert res.carta_disp.lcs[0] == pytest.approx(0.049, abs=5e-4)
    assert not res.carta_pos.fuera_de_limites().any()
    assert not res.carta_disp.fuera_de_limites().any()


def test_xbar_s_anillos():
    """X̄-S con los mismos datos: s̄ ≈ 0.0094, límites X̄ ≈ [73.988, 74.014], S LCS ≈ 0.0196 (Montgomery, ej. 6.3)."""
    res = VAR.carta_xbar_s(anillos())
    sbar = res.carta_disp.lc[0]
    assert sbar == pytest.approx(0.0094, abs=1e-4)
    assert res.carta_pos.lcs[0] == pytest.approx(74.014, abs=1e-3)
    assert res.carta_pos.lci[0] == pytest.approx(73.988, abs=1e-3)
    assert res.carta_disp.lcs[0] == pytest.approx(0.0196, abs=2e-4)


def test_xbar_s_tamano_variable_coincide_con_constante():
    """Con n constante, la rama de n variable (desviación agrupada) da casi lo mismo."""
    largo = anillos()
    quitar = largo.index[(largo["orden"] == 3)][:1]   # subgrupo 3 queda con n = 4
    res = VAR.carta_xbar_s(largo.drop(index=quitar))
    assert not res.n_constante
    assert res.sigma == pytest.approx(VAR.carta_xbar_s(largo).sigma, rel=0.05)


def test_i_mr():
    x = np.array([10.0, 12, 11, 13, 12, 14, 13, 12, 11, 12])
    largo = pd.DataFrame({"subgrupo": [str(i) for i in range(1, 11)], "orden": range(1, 11), "valor": x,
                          "fase": "I", "fila": range(2, 12)})
    res = VAR.carta_i_mr(largo)
    mrbar = np.mean(np.abs(np.diff(x)))
    assert res.media == pytest.approx(x.mean())
    assert res.sigma == pytest.approx(mrbar / 1.128, rel=1e-3)
    assert res.carta_disp.lcs[0] == pytest.approx(3.267 * mrbar, rel=1e-3)


def test_fase2_detecta_corrimiento_conocido():
    df = D.leer_archivo(PRUEBA / "prueba_corrimiento_fase2.csv")
    largo, _ = D.construir_subgrupos(df, "largo", col_valor="valor", col_subgrupo="subgrupo", col_fase="fase")
    res = VAR.carta_xbar_r(largo, reglas_pos=[1, 2, 3, 4])
    f2 = [s for s in res.carta_pos.senales if s.fase == "II"]
    assert f2, "La Fase II debe señalar el corrimiento de +1.5σ"
    assert min(int(s.etiqueta) for s in f2) >= 31, "No debe haber señales antes del corrimiento (subgrupo 31)"
    # Los límites se estiman solo con la Fase I
    assert res.carta_pos.lc[0] == pytest.approx(largo[largo.fase == "I"]["valor"].mean(), abs=1e-9)


def test_exclusion_recalcula_limites():
    largo = anillos()
    base = VAR.carta_xbar_r(largo)
    excl = VAR.carta_xbar_r(largo, excluidos=["14"])
    sub = D.resumen_subgrupos(largo)
    esperado = sub[sub["subgrupo"] != "14"]["media"].mean()
    assert excl.media == pytest.approx(esperado)
    assert excl.media != base.media


# --------------------------------------------------------------------------- atributos
def test_carta_p_jugo_montgomery():
    res = ATR.carta_p(jugo())
    assert res.parametro == pytest.approx(0.2313, abs=1e-4)
    assert res.carta.lcs[0] == pytest.approx(0.4102, abs=1e-4)
    assert res.carta.lci[0] == pytest.approx(0.0524, abs=1e-4)
    fuera = [res.carta.etiquetas[i] for i in np.where(res.carta.fuera_de_limites())[0]]
    assert fuera == ["15", "23"]


def test_carta_p_jugo_revisada():
    res = ATR.carta_p(jugo(), excluidos=["15", "23"])
    assert res.parametro == pytest.approx(0.2150, abs=1e-4)
    assert res.carta.lcs[0] == pytest.approx(0.3893, abs=1e-4)
    assert res.carta.lci[0] == pytest.approx(0.0407, abs=1e-4)


def test_carta_np_equivale_a_p():
    p = ATR.carta_p(jugo())
    np_ = ATR.carta_np(jugo())
    assert np_.carta.lcs[0] == pytest.approx(50 * p.carta.lcs[0])


def test_carta_c_tarjetas_montgomery():
    res = ATR.carta_c(tarjetas())
    assert res.parametro == pytest.approx(19.85, abs=1e-2)
    assert res.carta.lcs[0] == pytest.approx(33.22, abs=1e-2)
    assert res.carta.lci[0] == pytest.approx(6.48, abs=1e-2)
    fuera = [res.carta.etiquetas[i] for i in np.where(res.carta.fuera_de_limites())[0]]
    assert fuera == ["6", "20"]
    rev = ATR.carta_c(tarjetas(), excluidos=["6", "20"])
    assert rev.parametro == pytest.approx(19.67, abs=1e-2)
    assert rev.carta.lcs[0] == pytest.approx(32.97, abs=1e-2)
    assert rev.carta.lci[0] == pytest.approx(6.36, abs=1e-2)


def test_carta_u_con_unidades_iguales_es_c():
    c = ATR.carta_c(tarjetas())
    u = ATR.carta_u(tarjetas())
    assert u.carta.lcs[0] == pytest.approx(c.carta.lcs[0])


def test_recomendacion_atributos():
    assert ATR.recomendar_carta("defectuosos", False)[0] == "p"
    assert ATR.recomendar_carta("defectuosos", True)[0] == "np"
    assert ATR.recomendar_carta("defectos", True)[0] == "c"
    assert ATR.recomendar_carta("defectos", False)[0] == "u"


# --------------------------------------------------------------------------- reglas
def test_reglas_patrones():
    def carta(valores):
        m = len(valores)
        from cep.modelos import Carta
        return Carta("x", "t", "y", [str(i + 1) for i in range(m)], np.array(valores, float), np.zeros(m),
                     np.full(m, 3.0), np.full(m, -3.0), np.array(["I"] * m), np.zeros(m, bool))
    c = REG.evaluar(carta([0.5] * 8), reglas=[4])
    assert [s.posicion for s in c.senales] == [7]                    # 8 del mismo lado
    c = REG.evaluar(carta([0, 2.5, 2.4, 0]), reglas=[2])
    assert [s.posicion for s in c.senales] == [2]                    # 2 de 3 más allá de 2σ
    c = REG.evaluar(carta([-1, -0.5, 0, 0.5, 1, 1.5, 0]), reglas=[5])
    assert [s.posicion for s in c.senales] == [5]                    # 6 crecientes
    c = REG.evaluar(carta([0, 3.5, 0]), reglas=[1])
    assert [s.posicion for s in c.senales] == [1]
    c = REG.evaluar(carta([0.1, -0.1] * 8), reglas=[7])
    assert c.senales and c.senales[0].posicion == 14                # 15 dentro de ±1σ


# --------------------------------------------------------------------------- supuestos
def test_anderson_darling_vs_scipy():
    x = np.random.default_rng(1).normal(size=60)
    a2, _, p = SUP.anderson_darling(x)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a2_scipy = stats.anderson(x, dist="norm").statistic
    assert a2 == pytest.approx(a2_scipy, rel=1e-6)
    assert 0 < p <= 1


def test_normalidad_detecta_exponencial():
    x = np.random.default_rng(2).exponential(size=200)
    assert SUP.pruebas_normalidad(x)["normal"] is False


def test_independencia_detecta_ar1():
    rng = np.random.default_rng(3)
    e = rng.normal(size=300)
    x = np.zeros(300)
    for t in range(1, 300):
        x[t] = 0.8 * x[t - 1] + e[t]
    assert SUP.pruebas_independencia(x)["independiente"] is False
    assert SUP.pruebas_independencia(rng.normal(size=300))["independiente"] is True


# --------------------------------------------------------------------------- capacidad
def test_capacidad_formulas():
    rng = np.random.default_rng(4)
    x = rng.normal(10, 1, 500)
    r = CAP.calcular(x, sigma_w=1.0, lie=7, lse=13)["resumen"]
    assert r["Cp"] == pytest.approx(1.0)
    assert r["Cpk"] == pytest.approx(min(13 - x.mean(), x.mean() - 7) / 3)
    esperado = stats.norm.cdf((7 - x.mean()) / 1.0) + stats.norm.sf((13 - x.mean()) / 1.0)
    assert r["ppm_corto"] == pytest.approx(esperado * 1e6)


def test_capacidad_unilateral():
    x = np.random.default_rng(5).normal(50, 2, 100)
    r = CAP.calcular(x, 2.0, lse=58)["resumen"]
    assert np.isnan(r["Cp"])
    assert r["Cpk"] == pytest.approx((58 - x.mean()) / 6)


def test_ic_cp_contiene_estimacion():
    x = np.random.default_rng(6).normal(0, 1, 100)
    tabla = CAP.calcular(x, 1.0, -4, 4)["indices"]
    fila = tabla[tabla["Índice"].str.startswith("Cp ")].iloc[0]
    assert fila["IC inferior"] < fila["Valor"] < fila["IC superior"]


# --------------------------------------------------------------------------- diseño (ARL/ATS)
def test_arl0_shewhart():
    assert DIS.arl_xbar(5, 0, 3) == pytest.approx(370.4, abs=0.1)
    assert DIS.L_para_arl0(370.4) == pytest.approx(3.0, abs=1e-3)


def test_arl1_montgomery():
    # n = 5, δ = 1σ: β = Φ(3 − √5) − Φ(−3 − √5) ≈ 0.7775 → ARL1 ≈ 4.5
    assert DIS.beta_xbar(5, 1, 3) == pytest.approx(0.7775, abs=1e-3)
    assert DIS.arl_xbar(5, 1, 3) == pytest.approx(4.5, abs=0.05)


def test_simulacion_concuerda_con_formula():
    sim = DIS.simular_arl("shewhart", delta=1.0, n=1, repeticiones=20000, semilla=7)
    teo = DIS.arl_xbar(1, 1.0, 3)
    assert abs(sim["ARL"] - teo) < 4 * sim["EE"]


def test_cusum_arl0_publicado():
    # Montgomery, tabla 9.3: k = 0.5, h = 5 → ARL0 = 465, ARL1(δ=1) = 10.4
    sim0 = DIS.simular_arl("cusum", delta=0.0, k=0.5, h=5, repeticiones=8000, semilla=11)
    assert abs(sim0["ARL"] - 465) < 4 * sim0["EE"]
    sim1 = DIS.simular_arl("cusum", delta=1.0, k=0.5, h=5, repeticiones=8000, semilla=12)
    assert sim1["ARL"] == pytest.approx(10.4, abs=0.4)


def test_reglas_western_electric_arl0():
    # Con las reglas 1-4 el ARL0 baja a ≈ 91 (Champ y Woodall, 1987)
    sim = DIS.simular_arl("shewhart", delta=0.0, reglas=(1, 2, 3, 4), repeticiones=8000, semilla=13)
    assert 80 < sim["ARL"] < 100


def test_arl_s_y_r_en_control():
    # En control (λ = 1) ambos ARL deben ser grandes y R/S razonables
    assert DIS.arl_s(5, 1.0) > 200
    assert DIS.arl_r(5, 1.0) > 200
    assert DIS.arl_r(5, 2.0) < DIS.arl_r(5, 1.5) < DIS.arl_r(5, 1.0)


def test_arl_p_discreto():
    r = DIS.arl_p(50, 0.2313, 0.2313)
    assert 100 < r["ARL"] < 2000


# --------------------------------------------------------------------------- muestreo
def test_plan_n89_c2():
    # Montgomery, sección 15.2: n = 89, c = 2, Pa(p = 0.01) ≈ 0.9397
    assert float(MUE.prob_aceptacion(89, 2, 0.01)) == pytest.approx(0.9397, abs=1e-4)


def test_disenar_plan_cumple_riesgos():
    plan = MUE.disenar_plan_simple(0.01, 0.05, 0.06, 0.10)
    assert plan["Pa(NCA)"] >= 0.95 and plan["Pa(NCL)"] <= 0.10
    # minimalidad: con n − 1 ya no se cumple el riesgo del consumidor
    assert float(MUE.prob_aceptacion(plan["n"] - 1, plan["c"], 0.06)) > 0.10


def test_plan_doble_y_asn():
    pa, asn = MUE.pa_doble(50, 1, 100, 3, [0.0, 0.05])
    assert pa[0] == pytest.approx(1.0)
    assert 50 <= asn[1] <= 150


def test_plan_variables():
    pv = MUE.plan_variables(0.01, 0.05, 0.06, 0.10)
    assert pv["n"] >= 1 and 1.5 < pv["k"] < 2.4
    pa = MUE.disenar_plan_simple(0.01, 0.05, 0.06, 0.10)
    assert pv["n"] < pa["n"]


# --------------------------------------------------------------------------- datos
def test_lectura_csv_punto_y_coma_coma_decimal(tmp_path):
    ruta = tmp_path / "datos.csv"
    ruta.write_text("subgrupo;valor\n1;10,5\n1;10,7\n2;10,4\n2;10,9\n", encoding="utf-8")
    df = D.leer_archivo(ruta)
    assert df["valor"].tolist() == [10.5, 10.7, 10.4, 10.9]


def test_plantilla_variables_diligenciada_se_lee(tmp_path):
    """Se llena la plantilla con 3 subgrupos de n = 4 (x5 vacía) y se arma la carta."""
    from openpyxl import load_workbook
    wb = load_workbook(RAIZ / "plantillas" / "plantilla_datos_variables.xlsx")
    ws = wb["Datos"]
    valores = [[10.1, 10.0, 9.9, 10.2], [10.0, 10.3, 10.1, 9.8], [9.9, 10.0, 10.2, 10.1]]
    for i, fila in enumerate(valores, start=2):
        for j, v in enumerate(fila, start=6):      # columnas F..I = x1..x4
            ws.cell(row=i, column=j, value=v)
    ruta = tmp_path / "llena.xlsx"
    wb.save(ruta)
    df = D.leer_archivo(ruta, hoja="Datos")
    assert {"subgrupo", "x1", "x4", "fase"} <= set(df.columns) and "x5" not in df.columns
    largo, _ = D.construir_subgrupos(df, "ancho", cols_valor=["x1", "x2", "x3", "x4"], col_subgrupo="subgrupo",
                                     col_fase="fase")
    largo, _ = D.aplicar_limpieza(largo)
    assert largo["orden"].nunique() == 3 and D.tamano_subgrupo(largo) == (4, True)
    assert VAR.carta_xbar_r(largo).n == 4


# ---------------------------------------------------------------------------
# Caso aplicado (Taller metalmecánico A): el software lee el libro del taller y reproduce las cifras del informe
# ---------------------------------------------------------------------------

def caso_corte():
    df = D.leer_archivo(CASO, hoja="Corte_variables")
    largo, _ = D.construir_subgrupos(df, "ancho", cols_valor=["x1", "x2", "x3", "x4", "x5"],
                                     col_subgrupo="subgrupo", col_fase="fase")
    return largo


def test_caso_taller_lectura_y_fases():
    largo = caso_corte()
    assert largo["orden"].nunique() == 40
    assert largo.loc[largo.fase == "I", "orden"].nunique() == 25
    assert largo.loc[largo.fase == "II", "orden"].nunique() == 15
    assert largo["valor"].notna().sum() == 200


def test_caso_taller_limites_de_prueba_y_revisados():
    largo = caso_corte()
    prueba = VAR.carta_xbar_r(largo)
    assert prueba.media == pytest.approx(450.628, abs=1e-6)
    assert prueba.carta_disp.lc[0] == pytest.approx(2.060, abs=1e-6)
    rev = VAR.carta_xbar_r(largo, excluidos=["9", "17"])
    assert rev.media == pytest.approx(450.600, abs=1e-6)
    assert rev.sigma == pytest.approx((41 / 23) / 2.326, rel=1e-3)      # R̄ = 41/23 mm; σ̂ = R̄/d2
    assert rev.carta_pos.lcs[0] == pytest.approx(rev.media + 3 * rev.sigma / np.sqrt(5), rel=1e-9)


def test_caso_taller_carta_p_excluye_dia_12():
    df = D.leer_archivo(CASO, hoja="Inspeccion_atributos")
    tabla, _ = D.construir_atributos(df, "soportes_defectuosos", col_n="soportes_inspeccionados",
                                     col_muestra="muestra", col_fase="fase")
    res = ATR.carta_p(tabla, excluidos=["12"])
    f1 = tabla[(tabla["fase"] == "I") & (tabla["muestra"].astype(str) != "12")]
    assert res.parametro == pytest.approx(f1["conteo"].sum() / f1["n"].sum(), rel=1e-12)
    assert res.parametro == pytest.approx(0.1006, abs=5e-4)
    # Con las reglas 1 a 4 (las del programa), el día 12 ya excluido no cuenta como señal pendiente
    res4 = ATR.carta_p(tabla, excluidos=["12"], reglas=(1, 2, 3, 4))
    assert "sin contar las muestras excluidas por causa asignable (12)" in ATR.interpretar(res4)[0]
    # El Pareto del módulo 3 sugiere los 8 tipos de defecto (incluida «pintura_defectuosa») y nada más
    _, sugeridas = ATR.sugerir_tipos_defecto(df, {"soportes_defectuosos", "soportes_inspeccionados", "muestra"})
    assert set(sugeridas) == {"porosidad", "cordon_incompleto", "salpicadura_escoria", "rebaba_filo",
                              "pintura_defectuosa", "perforacion_desalineada", "falta_escuadra", "otros"}
    assert ATR.pareto(df[sugeridas].sum())["Conteo"].sum() == 210
