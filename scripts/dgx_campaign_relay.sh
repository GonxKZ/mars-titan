#!/usr/bin/env bash
# Relevo de campañas en la DGX: recorre un plan con fases de varias campañas, una cada vez.
#
# Uso: dgx_campaign_relay.sh <plan> [unidad de systemd del usuario a la que esperar antes]
#
# Cada línea del plan:
#   <copia fija>|<estado>|<vistas>|<declaración>|<trabajadores> <argumentos de run>
# Las líneas vacías y las que empiezan por # se ignoran. Cada fase se ejecuta con la copia fija
# de su campaña (su commit autorizado), así que el plan alterna campañas de commits distintos sin
# que dos compitan a la vez por la GPU, la memoria unificada y la caché de páginas de sus orígenes.
#
# El avance se guarda en <plan>.hecho (número de fases terminadas): tras una parada, el relevo
# continúa por la siguiente. El plan se puede ampliar por el final mientras corre.
#
# Pausa: mientras exista <carpeta del relevo>/pausa (la crea la guardia de producción), el relevo
# pide a la fase en curso que pare en su siguiente barrera y espera. Al retirarse la pausa repite
# la fase, que continúa desde sus checkpoints. Con SIGTERM para igual y termina.
#
# Una línea «limpia|<estado>/packed-inputs/<año>» retira las copias empaquetadas de ese origen
# cuando sus fases han terminado: las de todos los orígenes no caben a la vez en el disco y un
# trabajo que las necesite después las vuelve a construir. Solo acepta esa forma de ruta.
#
# Una línea «prepara|<copia fija>|<estado>|<receta>|<año>» construye antes de la fase, con un
# solo proceso, los índices y las copias empaquetadas de ese origen (dgx_prepare_inputs.py de la
# copia fija, con las mismas rutas bajo el estado que usa la cola). Si las construye el primer
# trabajo de la fase, los demás esperan con sus tablas en memoria, y en los orígenes grandes la
# suma supera el tope del grupo. Lo ya construido se reutiliza. Si la preparación falla después
# del margen de arranque, se anota y la fase siguiente construye lo que falte, como antes.
#
# Una fase que falla nada más arrancar (antes de MARS_TITAN_RELAY_MIN_SECONDS, 60 s) detiene el
# relevo sin anotarla: un permiso que falta o una copia rota no deben consumir el plan.
set -u
[ $# -ge 1 ] && [ $# -le 2 ] || { echo "Uso: $0 <plan> [unidad a la que esperar]" >&2; exit 2; }
plan="$1"; espera="${2:-}"
[ -f "$plan" ] || { echo "Falta el plan: $plan" >&2; exit 2; }
carpeta="${MARS_TITAN_RELAY_STATE:-$HOME/.local/state/mars-titan/relevo}"
pausa="$carpeta/pausa"; hecho="$plan.hecho"
sondeo="${MARS_TITAN_RELAY_POLL:-15}"; minimo="${MARS_TITAN_RELAY_MIN_SECONDS:-60}"
python="${MARS_TITAN_DGX_PYTHON:-$HOME/.local/share/mars-titan/dgx-us-py312/bin/python}"
mkdir -p "$carpeta"
hijo=""; parar=0
trap 'parar=1; [ -n "$hijo" ] && kill -TERM "$hijo" 2>/dev/null' TERM INT
nota() { echo "relevo: $* ($(date -u +%Y-%m-%dT%H:%M:%SZ))"; }
dormir() { sleep "$1" & wait $! 2>/dev/null; }

if [ -n "$espera" ]; then
  nota "espera a que termine $espera"
  while [ "$parar" = 0 ] && systemctl --user is-active -q "$espera"; do dormir "$sondeo"; done
fi
while [ "$parar" = 0 ]; do
  if [ -e "$pausa" ]; then
    nota "en pausa"
    while [ -e "$pausa" ] && [ "$parar" = 0 ]; do dormir "$sondeo"; done
    continue
  fi
  terminadas=$(cat "$hecho" 2>/dev/null || echo 0)
  linea=$(grep -v -E '^[[:space:]]*(#|$)' "$plan" | sed -n "$((terminadas + 1))p")
  if [ -z "$linea" ]; then nota "plan terminado tras $terminadas fases"; exit 0; fi
  if [[ "$linea" == limpia\|* ]]; then
    copias="${linea#limpia|}"
    if ! [[ "$copias" =~ ^/.+/packed-inputs/[0-9]{4}$ ]] || [ -L "$copias" ]; then
      nota "fase $((terminadas + 1)) mal escrita: $linea"
      exit 2
    fi
    if [ -d "$copias" ]; then
      rm -rf -- "$copias"
      nota "fase $((terminadas + 1)): copias retiradas de $copias"
    else
      nota "fase $((terminadas + 1)): no hay copias en $copias"
    fi
    echo $((terminadas + 1)) > "$hecho.tmp" && mv "$hecho.tmp" "$hecho"
    continue
  fi
  if [[ "$linea" == prepara\|* ]]; then
    IFS='|' read -r _ copia estado receta origen <<< "$linea"
    if ! [[ "$origen" =~ ^20[0-9]{2}$ ]] || [ "$origen" -lt 2005 ] || [ -z "$estado" ] || [ -z "$receta" ] ||
       [ ! -f "$copia/scripts/dgx_prepare_inputs.py" ]; then
      nota "fase $((terminadas + 1)) mal escrita: $linea"
      exit 2
    fi
    nota "fase $((terminadas + 1)): $linea"
    inicio=$(date +%s)
    ( cd "$copia" && exec env \
        MARS_TITAN_SHARED_INDICES="$estado/indices" \
        MARS_TITAN_TRAIN_ARCHIVE="$estado/packed-inputs/$origen" \
        MARS_TITAN_DIGEST_CACHE="$estado/file-digests.json" \
        MARS_TITAN_DECODE_WORKERS=4 MARS_TITAN_PREFETCH_BATCHES=4 MARS_TITAN_GROUP_CACHE_MIB=2048 \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONPATH="$copia/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$python" scripts/dgx_prepare_inputs.py origin \
          --view "$estado/cohorts/$origen/manifest.json" \
          --fold "$(printf 'fold-%03d' $((origen - 2005)))" --recipe "$receta" --archive ) &
  else
    IFS='|' read -r copia estado vistas declaracion fase <<< "$linea"
    trabajadores="${fase%% *}"; argumentos="${fase#* }"
    if ! [[ "$trabajadores" =~ ^[1-8]$ ]] || [ ! -x "$copia/scripts/dgx_campaign.sh" ] ||
       [ -z "$estado" ] || [ -z "$vistas" ] || [ -z "$declaracion" ] || [ "$argumentos" = "$fase" ]; then
      nota "fase $((terminadas + 1)) mal escrita: $linea"
      exit 2
    fi
    nota "fase $((terminadas + 1)): $linea"
    inicio=$(date +%s)
    # shellcheck disable=SC2086
    ( cd "$copia" && exec ./scripts/dgx_campaign.sh run --machine mars_titan \
        --declaration "$declaracion" --state "$estado" --views "$vistas" \
        --workers "$trabajadores" $argumentos ) &
  fi
  hijo=$!; pausada=0
  while kill -0 "$hijo" 2>/dev/null; do
    if [ "$pausada" = 0 ] && [ -e "$pausa" ]; then
      nota "pausa pedida: la fase para en su siguiente barrera"
      kill -TERM "$hijo" 2>/dev/null
      pausada=1
    fi
    dormir "$sondeo"
  done
  wait "$hijo"; codigo=$?; hijo=""
  [ "$parar" = 1 ] && break
  [ "$pausada" = 1 ] && continue
  if [ "$codigo" != 0 ] && [ "$codigo" != 3 ]; then
    if [ $(( $(date +%s) - inicio )) -lt "$minimo" ]; then
      nota "la fase falló al arrancar (código $codigo): el relevo se detiene sin anotarla"
      exit 1
    fi
    nota "la fase terminó con código $codigo: se anota y el plan sigue"
  fi
  echo $((terminadas + 1)) > "$hecho.tmp" && mv "$hecho.tmp" "$hecho"
done
nota "parada pedida"
exit 0
