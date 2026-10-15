#!/usr/bin/env bash
# Compila launch_threshold.cpp con el núcleo del árbol actual, las opciones de coma flotante del
# proyecto y la LibTorch del entorno de Python activo, y lo ejecuta. Se lanza desde la raíz del
# árbol de #628 con `python` apuntando a ese entorno.
#
# Uso: launch_threshold.sh REPETICIONES warm|cold
set -euo pipefail
here=$(dirname "$0")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
torch=$(python -c "import os, torch; print(os.path.dirname(torch.__file__))")
clang++ -std=c++20 -O3 -DNDEBUG -fno-fast-math -ffp-contract=off -pthread -Inative/include \
    -isystem "$torch/include" -isystem "$torch/include/torch/csrc/api/include" \
    "$here/launch_threshold.cpp" native/src/episodic_projection.cpp \
    -L"$torch/lib" -Wl,-rpath,"$torch/lib" -ltorch_cpu -lc10 -o "$work/launch_threshold"
"$work/launch_threshold" "$@"
