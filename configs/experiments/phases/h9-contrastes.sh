#!/usr/bin/env bash
# Contrastes de la prueba H9 (rasgos long_horizon_v3 en la conjunta de EE. UU. y China): cada fichero
# de contrastes en todas las sesiones y en las de cada mercado. Espera a las dos referencias de
# árboles y no repite un contraste que ya existe.
# Uso: h9-contrastes.sh <año> <semilla> [semilla...]
set -u
[ "$#" -ge 2 ] || { echo "uso: h9-contrastes.sh <año> <semilla> [semilla...]" >&2; exit 2; }
year="$1"; shift
tools=/home/gpic/.local/share/mars-titan/herramientas
python=/home/gpic/.local/share/mars-titan/dgx-us-py312/bin/python
a="/home/gpic/.local/state/mars-titan/dgx-titans-us-cn-v31/v11-inputs-v2/semillas-42,43,44/todos-cac92bf8"
b="/home/gpic/.local/state/mars-titan/dgx-titans-us-cn-v31/v11-inputs-v3/semillas-42,43,44/todos-cac92bf8"
refa="/home/gpic/.local/state/mars-titan/referencias/h9-us-cn-v2-cac92bf8"
refb="/home/gpic/.local/state/mars-titan/referencias/h9-us-cn-v3-cac92bf8"
out="/home/gpic/.local/state/mars-titan/informes/conjunta-h9/${year}-semillas-$(IFS=-; echo "$*")"
mkdir -p "$(dirname "$out")"
seeds=(); for seed in "$@"; do seeds+=(--seed "$seed"); done
# Las referencias de árboles se ajustan aparte: hasta cuatro horas de espera.
for run in "$refa/runs/lightgbm_technical/$year/seed-42" "$refb/runs/lightgbm_technical_v3/$year/seed-42"; do
  waited=0
  until [ -f "$run/evaluation-predictions.parquet" ] || [ "$waited" -ge 14400 ]; do sleep 60; waited=$((waited + 60)); done
done
status=0
for file in contrasts memory trees; do
  for market in todos CN US; do
    target="$out/$file-$market"
    [ -e "$target" ] && continue
    extra=(); [ "$market" = todos ] || extra=(--market "$market")
    "$python" -I "$tools/dgx_seed_contrasts.py" --state "$b" --year "$year" "${seeds[@]}" \
      --contrasts "$tools/us-cn-campaign-v11-inputs-$file.json" --also "$a" \
      --reference "$refa" --reference "$refb" --reference-seed 42 --offset "${extra[@]}" \
      --output "$target" > "$out-$file-$market.log" 2>&1 || status=1
  done
done
exit "$status"
