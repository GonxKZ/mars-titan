#!/usr/bin/env bash
# Compila pool_threshold.cpp con el núcleo del árbol actual, las opciones de coma flotante del
# proyecto y el OpenMP de GCC (libgomp, el runtime del backend OpenMP de ATen), y lo ejecuta.
# Se lanza desde la raíz del árbol de #628.
#
# Uso: pool_threshold.sh REPETICIONES warm|cold
set -euo pipefail
here=$(dirname "$0")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
g++ -std=c++20 -O3 -DNDEBUG -fno-fast-math -ffp-contract=off -pthread -fopenmp -Inative/include \
    "$here/pool_threshold.cpp" native/src/episodic_projection.cpp -o "$work/pool_threshold"
"$work/pool_threshold" "$@"
