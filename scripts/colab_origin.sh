#!/usr/bin/env bash
# Un origen de la campaña en una máquina de Google Colab, paso a paso.
#
# Da las mismas órdenes con las que el relevo de la DGX prepara y lanza un origen, en el orden
# en que hay que darlas la primera vez en una máquina nueva. Cada paso imprime su orden antes de
# ejecutarla; con --ensayo solo la imprime. El código que se ejecuta es el de la copia fija.
#
# Uso:
#   colab_origin.sh <paso> --repo <copia fija> --estado <estado> --vistas <vistas> \
#        --declaracion <json> --origen <año> [--receta <json>] [--semillas "42 43 44"] \
#        [--niveles "core controls"] [--trabajadores 4] [--nota "..."] [--nativo <.so>] \
#        [--extra "--no-preflight"] [--ensayo]
#
# Pasos, en este orden:
#   cohortes  cohortes e índices del estado (prepare) y comprobaciones del equipo (doctor)
#   entradas  copia empaquetada de las entradas del origen; necesita --receta
#   permiso   permiso expreso de esta declaración y de este código; necesita --nota
#   lanza     cola del origen con las semillas y los niveles pedidos
#   estado    resumen de la cola
#
# MARS_TITAN_SITE es obligatoria: la etiqueta del equipo con la que cada intento anota dónde
# se ajustó. Una copia fija anterior al kit de Colab (la de la campaña v10) no la anota, pero
# sus intentos llevan la GPU, la plataforma y las versiones.
set -euo pipefail

paso="${1:-}"
[ -n "$paso" ] || { echo "Falta el paso. Consulta la cabecera de este guion." >&2; exit 2; }
shift

repo="" estado="" vistas="" declaracion="" origen="" receta="" nota="" nativo="" extra=""
semillas="42" niveles="core" trabajadores="4" ensayo=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --estado) estado="$2"; shift 2 ;;
    --vistas) vistas="$2"; shift 2 ;;
    --declaracion) declaracion="$2"; shift 2 ;;
    --origen) origen="$2"; shift 2 ;;
    --receta) receta="$2"; shift 2 ;;
    --semillas) semillas="$2"; shift 2 ;;
    --niveles) niveles="$2"; shift 2 ;;
    --trabajadores) trabajadores="$2"; shift 2 ;;
    --nota) nota="$2"; shift 2 ;;
    --nativo) nativo="$2"; shift 2 ;;
    --extra) extra="$2"; shift 2 ;;
    --ensayo) ensayo=1; shift ;;
    *) echo "Opción desconocida: $1" >&2; exit 2 ;;
  esac
done

falta() { echo "FALTA: $*" >&2; exit 1; }

[ -d "$repo/src/mars_titan" ] || falta "--repo debe ser la copia fija del repositorio ($repo)"
[ -n "$estado" ] || falta "--estado"
[ -n "$vistas" ] || falta "--vistas"
[ -n "$declaracion" ] || falta "--declaracion"
[ -f "$repo/$declaracion" ] || falta "la declaración no está en la copia fija ($declaracion)"
[[ "$origen" =~ ^20[0-9]{2}$ ]] && [ "$origen" -ge 2005 ] || falta "--origen debe ser un año desde 2005"
[[ "$trabajadores" =~ ^[1-8]$ ]] || falta "--trabajadores debe estar entre 1 y 8"
: "${MARS_TITAN_SITE:?Declara MARS_TITAN_SITE, por ejemplo google-colab-rtx-pro-6000}"

python="${MARS_TITAN_COLAB_PYTHON:-$repo/.venv/bin/python}"
[ "$ensayo" = 1 ] || [ -x "$python" ] || falta "no existe el intérprete del entorno ($python)"

# El lanzador de la copia fija: el de Colab si lo trae, y si no el de la DGX con este entorno.
entorno=("MARS_TITAN_SITE=$MARS_TITAN_SITE")
if [ -x "$repo/scripts/colab_campaign.sh" ]; then
  lanzador="./scripts/colab_campaign.sh"
  entorno+=("MARS_TITAN_COLAB_PYTHON=$python")
elif [ -x "$repo/scripts/dgx_campaign.sh" ]; then
  lanzador="./scripts/dgx_campaign.sh"
  entorno+=("MARS_TITAN_DGX_PYTHON=$python")
else
  falta "la copia fija no trae un lanzador de campaña en scripts/"
fi
[ -z "$nativo" ] || entorno+=("MARS_TITAN_EPISODIC_NATIVE=$nativo")

orden() {
  # Imprime la orden con su entorno y, salvo en un ensayo, la ejecuta en la copia fija.
  local variables=("$@")
  printf '+ cd %q &&' "$repo"
  printf ' %q' env "${variables[@]}"
  printf '\n'
  [ "$ensayo" = 1 ] || (cd "$repo" && env "${variables[@]}")
}

campana() {
  orden "${entorno[@]}" "$lanzador" "$@" \
    --declaration "$declaracion" --state "$estado" --views "$vistas"
}

paso_cohortes() {
  campana prepare
  campana doctor --machine mars_titan --workers "$trabajadores"
}

paso_entradas() {
  [ -n "$receta" ] || falta "entradas necesita --receta (la del modelo base de la declaración)"
  [ -f "$repo/$receta" ] || falta "la receta no está en la copia fija ($receta)"
  [ "$ensayo" = 1 ] || [ -f "$estado/cohorts/$origen/manifest.json" ] \
    || falta "ejecuta antes el paso cohortes: no existe la cohorte de $origen"
  local fold
  fold="$(printf 'fold-%03d' $((origen - 2005)))"
  orden "${entorno[@]}" \
    "MARS_TITAN_SHARED_INDICES=$estado/indices" \
    "MARS_TITAN_TRAIN_ARCHIVE=$estado/packed-inputs/$origen" \
    "MARS_TITAN_DIGEST_CACHE=$estado/file-digests.json" \
    "MARS_TITAN_DECODE_WORKERS=${MARS_TITAN_DECODE_WORKERS:-4}" \
    "MARS_TITAN_PREFETCH_BATCHES=${MARS_TITAN_PREFETCH_BATCHES:-4}" \
    "MARS_TITAN_GROUP_CACHE_MIB=${MARS_TITAN_GROUP_CACHE_MIB:-2048}" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 "PYTHONPATH=$repo/src" \
    "$python" scripts/dgx_prepare_inputs.py origin \
    --view "$estado/cohorts/$origen/manifest.json" --fold "$fold" --recipe "$receta" --archive
}

paso_permiso() {
  [ -n "$nota" ] || falta "permiso necesita --nota: quién lanza y para qué"
  campana authorize --note "$nota"
}

paso_lanza() {
  local argumentos=() valor
  for valor in $semillas; do argumentos+=(--seed "$valor"); done
  for valor in $niveles; do argumentos+=(--tier "$valor"); done
  # shellcheck disable=SC2086
  campana run --machine mars_titan --workers "$trabajadores" "${argumentos[@]}" \
    --origin "$origen" $extra
}

paso_estado() {
  campana status --machine mars_titan
}

case "$paso" in
  cohortes) paso_cohortes ;;
  entradas) paso_entradas ;;
  permiso) paso_permiso ;;
  lanza) paso_lanza ;;
  estado) paso_estado ;;
  *) echo "Paso desconocido: $paso" >&2; exit 2 ;;
esac
