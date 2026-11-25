@echo off
title Vigia CEP - Control Estadistico de Procesos
cd /d "%~dp0"
echo ==================================================
echo   VIGIA CEP - Control Estadistico de Procesos
echo ==================================================
echo.
set "PY="
rem Se prefieren versiones de Python con todas las librerias disponibles (3.13 a 3.10)
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo No se encontro Python en este computador.
  echo Instale Python 3.12 o 3.13 desde https://www.python.org/downloads/
  echo y durante la instalacion marque la casilla "Add python.exe to PATH".
  echo Luego vuelva a hacer doble clic en INICIAR.bat
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creando entorno virtual - solo la primera vez...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo No se pudo crear el entorno virtual.
    pause
    exit /b 1
  )
)
echo [2/3] Verificando librerias - la primera vez tarda unos minutos y requiere internet...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo Error instalando librerias. Revise su conexion a internet e intente de nuevo.
  pause
  exit /b 1
)
echo [3/3] Abriendo Vigia CEP en el navegador: http://localhost:8501
echo.
echo   NO CIERRE ESTA VENTANA mientras use el programa.
echo   Para salir, cierre esta ventana.
echo.
start "" /min powershell -NoProfile -Command "Start-Sleep -Seconds 6; Start-Process 'http://localhost:8501'"
".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.port 8501
pause
