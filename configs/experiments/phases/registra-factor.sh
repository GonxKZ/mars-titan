#!/usr/bin/env bash
# Deja constancia de si el factor de mercado que declara la vista está en este equipo. Si falta, los
# rasgos del modelo de mercado de ese mercado valen cero (corpus_inputs.py, _market_factor).
# Uso: registra-factor.sh <manifiesto de la cohorte> <etiqueta> <fichero de registro>
set -u
[ "$#" -eq 3 ] || { echo "uso: registra-factor.sh <manifiesto> <etiqueta> <registro>" >&2; exit 2; }
mkdir -p "$(dirname "$3")"
/home/gpic/.local/share/mars-titan/dgx-us-py312/bin/python -I - "$1" "$2" >> "$3" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
for market, declared in sorted((manifest.get("market_factors") or {}).items()):
    path = Path(declared.get("prices_path") or "")
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state = "presente" if digest == declared.get("prices_sha256") else "presente con otra huella"
    else:
        state = "AUSENTE: rasgos del modelo de mercado apagados"
    print(f"{now} {sys.argv[2]} {market} {declared.get('symbol')} {state} {path}")
PY
