#!/usr/bin/env bash
# Vigía CEP — inicio en macOS o Linux:  bash iniciar.sh
cd "$(dirname "$0")" || exit 1
PY=python3
command -v "$PY" >/dev/null 2>&1 || { echo "Instale Python 3.10 o superior (https://www.python.org/downloads/)"; exit 1; }
[ -x .venv/bin/python ] || "$PY" -m venv .venv || exit 1
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt || exit 1
echo "Abriendo Vigía CEP en http://localhost:8501  (Ctrl+C para salir)"
( sleep 5; if command -v open >/dev/null 2>&1; then open http://localhost:8501; else xdg-open http://localhost:8501 >/dev/null 2>&1; fi ) &
.venv/bin/python -m streamlit run app.py --server.headless true --server.port 8501
