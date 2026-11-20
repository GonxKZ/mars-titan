#!/usr/bin/env bash
# Lanzador de la campaña de comparación en la DGX Spark (GB10).
#
# Fija el entorno que exige el recorrido cronológico (cuBLAS determinista, salida sin búfer y
# el banco episódico nativo) y ejecuta comparison_campaign con los argumentos recibidos.
# MARS_TITAN_DGX_PYTHON y MARS_TITAN_EPISODIC_NATIVE permiten otras rutas de instalación.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export MARS_TITAN_EPISODIC_NATIVE="${MARS_TITAN_EPISODIC_NATIVE:-$HOME/.local/share/mars-titan/dgx-us-nativo/_episodic_native.cpython-312-aarch64-linux-gnu.so}"
python="${MARS_TITAN_DGX_PYTHON:-$HOME/.local/share/mars-titan/dgx-us-py312/bin/python}"
cd "$(dirname "$0")/.."
exec "$python" -m mars_titan.training.comparison_campaign "$@"
