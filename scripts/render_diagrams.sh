#!/bin/sh
# Valida, genera y exporta con fondo blanco los diagramas de docs/diagrams.
#
# Necesita archify en el PATH, Node 18 o posterior, Chrome o Chromium y ARCHIFY_HOME con la
# carpeta de archify. Los HTML interactivos quedan en build/diagrams y las imágenes junto a
# su especificación. Con nombres de diagrama como argumentos, solo procesa esos.
set -eu
: "${ARCHIFY_HOME:?Indica la carpeta de archify en ARCHIFY_HOME}"
cd "$(dirname "$0")/.."
mkdir -p build/diagrams
pairs=""
for spec in docs/diagrams/*/*.json; do
  name=$(basename "$spec" .json)
  if [ "$#" -gt 0 ]; then
    case " $* " in *" $name "*) ;; *) continue ;; esac
  fi
  kind=$(node -e 'process.stdout.write(JSON.parse(require("fs").readFileSync(process.argv[1], "utf8")).diagram_type)' "$spec")
  # Si falla, se repite sin --json para mostrar los errores.
  archify validate "$kind" "$spec" --quality showcase --json >/dev/null ||
    { archify validate "$kind" "$spec" --quality showcase; exit 1; }
  archify deliver "$kind" "$spec" "build/diagrams/$name.html" --quality showcase --json >/dev/null
  pairs="$pairs build/diagrams/$name.html ${spec%.json}.png"
done
[ -n "$pairs" ] || { echo "Ningún diagrama coincide con: $*" >&2; exit 1; }
# shellcheck disable=SC2086
node scripts/export_diagrams.mjs $pairs
