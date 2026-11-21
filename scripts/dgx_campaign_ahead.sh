#!/usr/bin/env bash
# Preparación adelantada del relevo de campañas en la DGX (MARS_TITAN_RELAY_AHEAD).
#
# Uso: dgx_campaign_ahead.sh start <copia fija> <estado> <receta> <año> | busy | stop
#
# «start» construye en segundo plano, mientras el relevo entrena otra fase, los índices y las
# copias empaquetadas de un origen: la misma orden que la línea «prepara» del relevo, en una
# unidad transitoria de systemd con poca prioridad de CPU y de disco y su propio tope de memoria.
# Solo arranca si no hay otra en curso, si el relevo no está en pausa y si queda margen: el disco
# libre debe cubrir las copias del origen en curso y las del siguiente sin acercarse al umbral
# de la guardia de producción (MARS_TITAN_AHEAD_MIN_DISK_GIB, 140) y la memoria disponible debe
# cubrir el constructor (MARS_TITAN_AHEAD_MIN_MEMORY_GIB, 40). Devuelve 0 si la lanza.
# «busy» devuelve 0 mientras trabaja y «stop» la detiene; lo construido hasta entonces se
# reutiliza o se rehace al repetir la orden.
set -u
unidad="${MARS_TITAN_AHEAD_UNIT:-mars-titan-adelanto}"
python="${MARS_TITAN_DGX_PYTHON:-$HOME/.local/share/mars-titan/dgx-us-py312/bin/python}"
relevo="${MARS_TITAN_RELAY_STATE:-$HOME/.local/state/mars-titan/relevo}"
nativo="${MARS_TITAN_EPISODIC_NATIVE:-$HOME/.local/share/mars-titan/dgx-us-nativo/_episodic_native.cpython-312-aarch64-linux-gnu.so}"
case "${1:-}" in
  busy)
    systemctl --user is-active -q "$unidad.service"
    ;;
  stop)
    systemctl --user stop "$unidad.service" 2> /dev/null
    exit 0
    ;;
  start)
    [ $# -eq 5 ] || { echo "Uso: $0 start <copia fija> <estado> <receta> <año>" >&2; exit 2; }
    copia="$2"; estado="$3"; receta="$4"; origen="$5"
    [[ "$origen" =~ ^20[0-9]{2}$ ]] && [ "$origen" -ge 2005 ] && [ -d "$estado" ] &&
      [ -f "$copia/scripts/dgx_prepare_inputs.py" ] && [ -f "$copia/$receta" ] || exit 2
    systemctl --user is-active -q "$unidad.service" && exit 1
    [ -e "$relevo/pausa" ] && exit 1
    libre=$(df -BG --output=avail "$estado" | tail -1 | tr -dc 0-9)
    disponible=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
    [ "${libre:-0}" -ge "${MARS_TITAN_AHEAD_MIN_DISK_GIB:-140}" ] &&
      [ "${disponible:-0}" -ge "${MARS_TITAN_AHEAD_MIN_MEMORY_GIB:-40}" ] || exit 1
    systemctl --user reset-failed "$unidad.service" 2> /dev/null
    exec systemd-run --user -q --unit "$unidad" --slice=mars-titan.slice \
      -p MemoryHigh=16G -p MemoryMax=24G -p MemorySwapMax=0 \
      -p CPUWeight=20 -p IOWeight=20 -p Nice=10 --working-directory="$copia" \
      -E MARS_TITAN_SHARED_INDICES="$estado/indices" \
      -E MARS_TITAN_TRAIN_ARCHIVE="$estado/packed-inputs/$origen" \
      -E MARS_TITAN_DIGEST_CACHE="$estado/file-digests.json" \
      -E MARS_TITAN_DECODE_WORKERS=4 -E MARS_TITAN_PREFETCH_BATCHES=4 \
      -E MARS_TITAN_GROUP_CACHE_MIB=2048 -E CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      -E PYTHONUNBUFFERED=1 -E PYTHONPATH="$copia/src" -E MARS_TITAN_EPISODIC_NATIVE="$nativo" \
      "$python" "$copia/scripts/dgx_prepare_inputs.py" origin \
      --view "$estado/cohorts/$origen/manifest.json" \
      --fold "$(printf 'fold-%03d' $((origen - 2005)))" --recipe "$copia/$receta" --archive
    ;;
  *)
    echo "Uso: $0 start <copia fija> <estado> <receta> <año> | busy | stop" >&2
    exit 2
    ;;
esac
