#!/bin/bash
# Conductor de la campaña DGX v2: recorre las fases declaradas en orden, un origen cada vez.
# Con un solo origen en marcha, todos los trabajos leen la misma copia empaquetada, que cabe en
# la caché de página del grupo, y las copias de un origen terminado se pueden borrar antes.
# Cada fase es una orden run de comparison_campaign limitada con --origin, --seed y --tier; la
# cola es idempotente, así que repetir una fase ya terminada no ejecuta nada.
# Uso: conductor-v2.sh <copia fija> <estado> <vistas> <archivo de fases>
# Cada línea del archivo de fases: <trabajadores> <argumentos de run>. Las líneas vacías y las
# que empiezan por # se ignoran. El conductor para si se le pide parar (SIGTERM).
set -u
copia="$1"; estado="$2"; vistas="$3"; fases="$4"
parar=0
trap "parar=1" TERM INT
cd "$copia" || exit 2
while IFS= read -r linea || [ -n "$linea" ]; do
  case "$linea" in ""|\#*) continue ;; esac
  [ "$parar" -eq 0 ] || break
  trabajadores="${linea%% *}"; argumentos="${linea#* }"
  echo "fase: $linea ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  # shellcheck disable=SC2086
  ./scripts/dgx_campaign.sh run --machine mars_titan --declaration configs/experiments/dgx-us-campaign-v2.json --state "$estado" --views "$vistas" --workers "$trabajadores" $argumentos &
  hijo=$!
  wait "$hijo"; codigo=$?
  # Una señal interrumpe wait: se espera a que la cola pause sus trabajos en su barrera.
  while kill -0 "$hijo" 2>/dev/null; do wait "$hijo"; codigo=$?; done
  echo "fase terminada con salida $codigo: $linea ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  # 0: todo completo. 3: quedan trabajos pausados o fallidos; se sigue con la fase siguiente
  # salvo que se haya pedido parar. Otra salida es un error de la orden y detiene el conductor.
  [ "$codigo" -eq 0 ] || [ "$codigo" -eq 3 ] || exit "$codigo"
done < "$fases"
[ "$parar" -eq 0 ] && echo "conductor: todas las fases recorridas" || echo "conductor: parada solicitada"
