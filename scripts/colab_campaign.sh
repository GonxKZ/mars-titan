#!/usr/bin/env bash
# Lanzador de la campaña de comparación en una máquina de Google Colab.
#
# Igual que dgx_campaign.sh, con el entorno del repositorio (uv) y la etiqueta del equipo
# obligatoria: cada intento anota en su registro dónde se ajustó (MARS_TITAN_SITE), junto con
# la GPU, su controlador y las versiones. El código que se ejecuta es el de esta copia fija.
set -euo pipefail
: "${MARS_TITAN_SITE:?Declara MARS_TITAN_SITE, por ejemplo google-colab-rtx-pro-6000}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
python="${MARS_TITAN_COLAB_PYTHON:-$PWD/.venv/bin/python}"
[ -x "$python" ] || { echo "No existe el intérprete del entorno ($python)" >&2; exit 1; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$python" -m mars_titan.training.comparison_campaign "$@"
