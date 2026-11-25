# Vigía CEP — Software de Control Estadístico de Procesos

Vigía CEP convierte datos de producción en decisiones: construye gráficos de control por
variables y por atributos en Fase I y Fase II, valida supuestos, calcula índices de capacidad y
fracción no conforme, diseña el esquema de muestreo con ARL0, ARL1 y ATS, diseña planes de
muestreo de aceptación y genera reportes en Excel y HTML.

Trabajo final de Control Estadístico de Procesos — Facultad de Ingeniería, Universidad del Magdalena.

---

## 1. Instalación

**Requisito:** Python 3.10 a 3.13 ([python.org/downloads](https://www.python.org/downloads/)); se recomienda
3.12 o 3.13, porque las versiones recién lanzadas pueden tardar en tener disponibles todas las librerías.
Al instalarlo en Windows marque la casilla **"Add python.exe to PATH"**.

### Windows (recomendado)
1. Descomprima la carpeta del proyecto.
2. Haga doble clic en **`INICIAR.bat`**.
   La primera vez crea un entorno virtual e instala las librerías (requiere internet, 2 a 5 minutos).
3. El programa se abre en el navegador en `http://localhost:8501`.
   No cierre la ventana negra mientras use el programa.

### macOS o Linux
```bash
bash iniciar.sh
```

### Manual (cualquier sistema)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (en macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

### Librerías
`streamlit`, `plotly`, `pandas`, `numpy`, `scipy`, `openpyxl` (y `pytest` para las pruebas).
Ver `requirements.txt`.

---

## 2. Uso rápido

1. **Inicio**: cargue el caso aplicado (Taller A) o un conjunto de datos de prueba para conocer el programa.
2. **1. Gestión de datos**: cargue su Excel/CSV (o llene primero la plantilla de `plantillas/`),
   defina los subgrupos racionales y la Fase I / Fase II, revise la limpieza y las tablas resumen.
3. **2 y 3. Gráficos de control**: revise la Fase I, investigue cada señal y excluya únicamente los
   puntos con causa asignable documentada; los límites se recalculan. La Fase II se monitorea con los
   límites congelados.
4. **4. Supuestos** y **5. Capacidad** (con LIE y LSE de la ficha técnica o del cliente).
5. **6. Diseño ARL/ATS** para decidir n y la frecuencia de muestreo; **7. Muestreo de aceptación**.
6. **8. Reportes**: descargue el Excel, el reporte HTML (imprimible a PDF) y los límites en JSON.

El manual de usuario describe cada función paso a paso.

---

## 3. Estructura del código (organizado por módulos)

```
VigiaCEP/
├── app.py                  Interfaz gráfica (Streamlit): un menú por cada módulo
├── cep/                    Núcleo estadístico, independiente de la interfaz
│   ├── datos.py            Módulo 1. Carga, identificación de variables, subgrupos, limpieza, resumen
│   ├── variables.py        Módulo 2. X̄-R, X̄-S, I-MR, EWMA y CUSUM, Fase I y Fase II
│   ├── atributos.py        Módulo 3. p, np, c y u con selección justificada; Pareto de defectos
│   ├── supuestos.py        Módulo 4. Normalidad (con aviso de resolución), independencia, Box-Cox
│   ├── reglas.py           Módulo 4. Reglas de Western Electric / Nelson (patrones no aleatorios)
│   ├── capacidad.py        Módulo 5. Cp, Cpk, Cpl, Cpu, Pp, Ppk, Cpm, fracción no conforme
│   ├── diseno.py           Módulo 6. ARL0, ARL1, ATS (analítico y simulación), carga muestral
│   ├── muestreo.py         Módulo 7. Planes simples, dobles y por variables; curvas OC, AOQ, ATI, ASN
│   ├── reportes.py         Módulo 8. Exportación a Excel, HTML y JSON
│   ├── graficos.py         Figuras interactivas (Plotly)
│   ├── constantes.py       d2, d3, c4 y constantes derivadas
│   └── modelos.py          Estructura común de un gráfico de control
├── datos_prueba/           Datos de prueba con resultados conocidos (ver LEEME.txt)
├── caso_taller_a/          Caso aplicado del informe: datos del taller, script de análisis y resultados
├── plantillas/             Plantillas de Excel para recolectar datos en planta
├── pruebas/                Pruebas automáticas de validación (pytest)
├── docs/                   Evidencia de ejecución (capturas y resultado de las pruebas)
├── requirements.txt        Librerías requeridas
├── INICIAR.bat / iniciar.sh  Inicio con un clic
└── .streamlit/config.toml  Tema visual
```

### Cambios de la versión 1.2
- Inicio: el caso aplicado (Taller A) se carga con un clic (longitud del brazo con X̄-R, soportes defectuosos con
  carta p y defectos por soporte con carta u), con los datos del caso para los reportes y la especificación
  (LIE 447 mm, LSE 453 mm) ya sugerida en el módulo 5.
- Carpeta `caso_taller_a/`: libro de datos del taller, script `analisis_caso.py` que reproduce todas las
  cifras y figuras del informe, y los reportes del módulo 8 (Excel, HTML y límites en JSON).
- Tres pruebas nuevas comprueban que el software lee el libro del taller y reproduce los límites de control
  y la proporción de defectuosos del informe (54 pruebas en total).
- Módulo 3: el Pareto sugiere de entrada todos los tipos de defecto, también los que se llaman, por ejemplo,
  «pintura_defectuosa»; solo descarta las columnas de totales, unidades inspeccionadas y unidades defectuosas.
- Módulos 2 y 3: la interpretación ya no cuenta como señales pendientes los puntos excluidos por causa
  asignable; los menciona aparte, como hace el módulo 5.
- El libro del caso trae la fecha de cada jornada de la planilla de inspección (13/10/2026 a 11/11/2026).

### Cambios de la versión 1.1
- Módulo 3: diagrama de Pareto de tipos de defecto (también entra en el reporte).
- Módulo 4: si la medición tiene poca resolución (muchos valores repetidos), la prueba de normalidad
  lo advierte, porque los empates hacen rechazar a Shapiro-Wilk y Anderson-Darling aunque la forma sea normal.
- Módulo 1: al cambiar de archivo o de hoja, la configuración sugerida (formato, columnas, fases) se recalcula
  con los datos nuevos, y en libros con varias hojas se propone la primera hoja con datos (se omite la de LEEME).
- Módulo 3: la columna del conteo se sugiere según lo que se cuenta (defectos o unidades defectuosas).
- Módulo 5: casilla para tratar como falsas alarmas las señales de patrones de la Fase I sin causa asignable,
  y rango de λ de Box-Cox ampliado a ±5 (corrige un error con datos de poca resolución).
- Ajustes de presentación en las figuras (leyendas que no se superponen, títulos opcionales para el informe).

---

## 4. Validación

```bash
python -m pytest pruebas -v
```

Las 54 pruebas comparan el software con resultados publicados (Montgomery, 2020: anillos de pistón
X̄-R y X̄-S, envases de jugo con carta p, tarjetas de circuito con carta c), con propiedades teóricas
(ARL0 = 370.4 con límites 3σ, ARL0 ≈ 465 del CUSUM con k = 0.5 y h = 5, ARL0 ≈ 91 con las reglas de
Western Electric), con un caso de detección (un corrimiento conocido de 1.5σ desde el subgrupo 31 debe
señalarse en la Fase II) y con las cifras del caso aplicado. Resultado de referencia:
`docs/evidencia/resultado_pruebas.txt`.

---

## 5. Fórmulas principales

| Elemento | Fórmula |
|---|---|
| σ dentro (X̄-R / X̄-S / I-MR) | R̄/d2 · s̄/c4 · MR̄/1.128 |
| Límites X̄ | x̿ ± L·σ̂/√n |
| Límites R | (d2 ± L·d3)·σ̂, LCI truncado en 0 |
| Límites p / u | p̄ ± L√(p̄(1−p̄)/nᵢ) · ū ± L√(ū/nᵢ) |
| Capacidad | Cp = (LSE−LIE)/6σ̂ · Cpk = mín[(LSE−μ̂)/3σ̂, (μ̂−LIE)/3σ̂] |
| Fracción no conforme | Φ((LIE−μ̂)/σ̂) + 1 − Φ((LSE−μ̂)/σ̂) |
| ARL del X̄ | β = Φ(L − δ√n) − Φ(−L − δ√n), ARL = 1/(1−β), ATS = ARL·h |
| Plan simple | Pa = Σ_{d≤c} C(n,d) pᵈ(1−p)ⁿ⁻ᵈ, AOQ = Pa·p(N−n)/N, ATI = n + (1−Pa)(N−n) |

---

## 6. Referencias

- Gutiérrez Pulido, H., & De la Vara Salazar, R. (2013). *Control estadístico de la calidad y Seis Sigma* (3.ª ed.). McGraw-Hill.
- Lucas, J. M., & Saccucci, M. S. (1990). Exponentially weighted moving average control schemes: Properties and enhancements. *Technometrics, 32*(1), 1–12.
- Montgomery, D. C. (2020). *Introduction to statistical quality control* (8th ed.). Wiley.
- Nelson, L. S. (1984). The Shewhart control chart—Tests for special causes. *Journal of Quality Technology, 16*(4), 237–239.
- Western Electric Company. (1956). *Statistical quality control handbook*. Western Electric.

---

## 7. Equipo desarrollador

| Integrante | Correo |
|---|---|
| Santiago José Castro Mateus | sjcastrom@unimagdalena.edu.co |
| Sebastián David Ferreira García | sdferreira@unimagdalena.edu.co |
| Karol Valentina Álvarez Tobón | kvalvarez@unimagdalena.edu.co |
| Juan David Acosta Santana | jdacosta@unimagdalena.edu.co |

Docente: Manuel J. Campuzano, Ph.D. — Control Estadístico de Procesos.

Desarrollado con apoyo de un asistente de inteligencia artificial, según lo autorizado por el docente.
