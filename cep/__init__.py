"""
Vigía CEP — núcleo estadístico.

Paquete con los cálculos de Control Estadístico de Procesos, separados de la
interfaz gráfica para que puedan probarse y reutilizarse de forma independiente.

Módulos (en el mismo orden de la guía del trabajo final):
    datos        Módulo 1. Gestión de datos (carga, subgrupos, limpieza, resumen)
    variables    Módulo 2. Gráficos de control por variables (X̄-R, X̄-S, I-MR, EWMA, CUSUM)
    atributos    Módulo 3. Gráficos de control por atributos (p, np, c, u)
    supuestos    Módulo 4. Validación de supuestos (normalidad, independencia, patrones)
    capacidad    Módulo 5. Capacidad del proceso (Cp, Cpk, Cpl, Cpu, Pp, Ppk, fracción no conforme)
    diseno       Módulo 6. Diseño de gráficos (ARL0, ARL1, ATS, carga muestral)
    muestreo     Módulo 7. Muestreo de aceptación (planes simples, dobles y por variables)
    reportes     Módulo 8. Reportes (Excel, HTML, conclusiones preliminares)

Apoyo:
    constantes   Constantes de gráficos de control (d2, d3, c4 y derivadas)
    reglas       Reglas de Western Electric / Nelson para patrones no aleatorios
    modelos      Estructura común de resultados de un gráfico de control
    graficos     Figuras interactivas (Plotly)
"""

__version__ = "1.2.0"
__nombre__ = "Vigía CEP"
