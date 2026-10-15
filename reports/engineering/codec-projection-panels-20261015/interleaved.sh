#!/usr/bin/env bash
# Compila el núcleo de 131d809e y el del árbol actual con las opciones de coma flotante del
# proyecto y ejecuta interleaved.cpp. Se lanza desde la raíz del árbol de #628.
#
# Uso: interleaved.sh REPETICIONES FILAS...
set -euo pipefail
here=$(dirname "$0")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/old/include/mars_titan"
git show 131d809e:native/src/episodic_projection.cpp > "$work/old/episodic_projection.cpp"
git show 131d809e:native/include/mars_titan/episodic_projection.hpp \
    > "$work/old/include/mars_titan/episodic_projection.hpp"
flags=(-std=c++20 -O3 -DNDEBUG -fno-fast-math -ffp-contract=off -pthread)
# El núcleo anterior se compila en otro espacio de nombres para enlazar los dos a la vez.
clang++ "${flags[@]}" -Dmars_titan=mars_titan_old -I"$work/old/include" \
    -c "$work/old/episodic_projection.cpp" -o "$work/old.o"
clang++ "${flags[@]}" -Inative/include "$here/interleaved.cpp" \
    native/src/episodic_projection.cpp "$work/old.o" -o "$work/interleaved"
"$work/interleaved" "$@"
