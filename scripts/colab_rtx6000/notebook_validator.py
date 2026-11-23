#!/usr/bin/env python3
"""Generador y validador del cuaderno local `.ipynb` (`nbformat 4`) de arranque en Colab.

El cuaderno generado es estrictamente un lanzador ordenado que invoca los guiones reales
del kit (`scripts/colab_*.{sh,py}` y `scripts/colab_rtx6000/*.py`) sin
lógica de entrenamiento incrustada, sin subidas a red ni a Drive no autorizadas, con FP32
IEEE estricto (`allow_tf32=False`, sin autocast, `torch.use_deterministic_algorithms(True)`)
y con el año 2024 sellado.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_SCRIPT_REF_RE = re.compile(
    r"\b((?:colab_[a-z0-9_]+|campaign_search|concurrency_probe|mps_probe|cu_budget_guard|kernel_sm120_plan)\.(?:sh|py))\b"
)

_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tf32_enabled", re.compile(r"allow_tf32\s*=\s*True", re.IGNORECASE)),
    (
        "low_matmul_precision",
        re.compile(r"set_float32_matmul_precision\(\s*['\"](?:medium|high)['\"]", re.IGNORECASE),
    ),
    ("autocast_usage", re.compile(r"\bautocast\b|\bGradScaler\b", re.IGNORECASE)),
    (
        "reduced_precision_dtype",
        re.compile(r"\bbfloat16\b|\bfloat16\b|\bfp16\b|\bbf16\b|\bfp8\b", re.IGNORECASE),
    ),
    (
        "nondeterministic_algorithms",
        re.compile(r"use_deterministic_algorithms\(\s*False", re.IGNORECASE),
    ),
    (
        "cudnn_nondeterministic",
        re.compile(r"cudnn\.deterministic\s*=\s*False|cudnn\.benchmark\s*=\s*True", re.IGNORECASE),
    ),
    (
        "sealed_2024_reference",
        re.compile(
            r"(?:^|[^0-9])2024(?:[^0-9]|$)|fold-019|final_test_access\s*[:=]\s*true", re.IGNORECASE
        ),
    ),
    (
        "unauthorized_upload",
        re.compile(
            r"\brclone\s+(?:copy|sync)\b|\bgcloud\s+storage\b|\bgit\s+push\b|\bhf\s+upload\b",
            re.IGNORECASE,
        ),
    ),
    (
        "inline_training_loop",
        re.compile(r"\boptimizer\.step\s*\(|\bloss\.backward\s*\(", re.IGNORECASE),
    ),
)

_REQUIRED_GUARDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cublas_workspace", re.compile(r"CUBLAS_WORKSPACE_CONFIG=:4096:8")),
    ("triton_ieee", re.compile(r"TRITON_F32_DEFAULT=ieee")),
    ("strict_allow_tf32_false", re.compile(r"allow_tf32\s*=\s*False")),
    ("deterministic_true", re.compile(r"use_deterministic_algorithms\(\s*True")),
)


def _md_cell(cell_id: str, lines: list[str]) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in lines],
    }


def _code_cell(cell_id: str, lines: list[str]) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else f"{line}\n" for line in lines],
    }


def _lines(text: str) -> list[str]:
    """Las líneas de una celda escrita tal cual, sin el blanco inicial ni el final."""
    return text.strip("\n").split("\n")


def build_bootstrap_notebook() -> dict[str, Any]:
    """Construye el cuaderno `.ipynb` (`nbformat 4.5`) ordenado de arranque para Colab G4."""
    cells = [
        _md_cell(
            "hdr-00",
            [
                "# Arranque operativo en Google Colab RTX PRO 6000 Blackwell (`sm_120`)",
                "",
                "Cuaderno local de ejecución por pasos ordenados sobre el kit de `mars-titan`.",
                "- **Contrato numérico:** FP32 IEEE estricto (sin TF32 ni precisión mixta, "
                "`TRITON_F32_DEFAULT=ieee`) y algoritmos deterministas.",
                "- **Datos:** restaurados en `/content` desde el paquete que ya está en Google "
                "Drive, "
                "con la huella SHA-256 de cada volumen. Desde aquí solo sale el estado de la "
                "campaña "
                "hacia su espejo.",
                "- **Seguridad temporal:** solo orígenes hasta 2023; el año de la evaluación final "
                "sigue sellado.",
                "- **Relevos:** una sesión dura menos de 24 h. El paso 6 para la cola en su "
                "barrera, "
                "copia el estado al espejo y solo entonces deja soltar la máquina. La sesión "
                "siguiente "
                "repite los pasos 0 a 4 y la cola continúa donde quedó.",
            ],
        ),
        _code_cell(
            "step-00-config",
            _lines(
                r"""
# Paso 0: variables del entorno y montaje de Drive (solo en Colab)
import os

os.environ['MARS_TITAN_SITE'] = 'google-colab-rtx-pro-6000'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TRITON_F32_DEFAULT'] = 'ieee'

# Ajustar antes de ejecutar (rutas dentro de Drive montado en /content/drive):
os.environ['P'] = '/content/drive/MyDrive/mars-titan-paquete'
os.environ['B'] = '/content/drive/MyDrive/mars-titan-paquete/mars-titan-colab.bundle'
os.environ['RAMA_KIT'] = 'dgx-trainning'  # la rama de las herramientas, no la del modelo
# COMMIT_HEX falla en cerrado mientras no sea el commit autorizado, con sus 40 hex
os.environ['COMMIT_HEX'] = 'REEMPLAZAR_CON_COMMIT_40_HEX'
os.environ['T'] = '/content/mars-titan'
os.environ['K'] = '/content/mars-titan/kit/scripts'
os.environ['ADDON'] = '/content/mars-titan/kit/scripts/colab_rtx6000'
os.environ['ESTADO'] = '/content/mars-titan/estado/v11'
os.environ['MIRROR_DRIVE'] = '/content/drive/MyDrive/mars-titan-estado-colab/v11'
os.environ['DESTINO_DATOS'] = '/home/gonzalo'
vistas = '/home/gonzalo/.local/state/mars-titan/campaign-a-v2-views-v31-20261013/views'
os.environ['VISTAS'] = vistas + '/US'  # '/US+CN' para la campaña conjunta
os.environ['FACTOR_CN_ZIP'] = ''  # con vistas US+CN: zip del factor de mercado de China
os.environ['DECLARACION'] = 'configs/experiments/us-campaign-v11-screen1.json'
os.environ['RECETA'] = 'configs/titans/dgx-us-core-v2.json'  # la de las entradas, como en la DGX
os.environ['ORIGIN_YEAR'] = '2013'
os.environ['SEMILLAS'] = '42 43 44'
os.environ['NIVELES'] = 'core'
os.environ['TRABAJADORES'] = '4'
os.environ['NOTA_PERMISO'] = 'Arranque en Colab RTX PRO 6000'
os.environ['COLAB_CU_PANEL'] = '7.8 CU/h'  # sustituir por la lectura real del panel de recursos
os.environ['SESSION_ID'] = 'relay-01'
os.environ['ESPERA_PARADA_S'] = '1800'  # margen para que la cola llegue a su barrera

try:
    from google.colab import drive  # type: ignore[import-not-found]
    drive.mount('/content/drive')
except ImportError:
    print('Fuera de Colab: se omite drive.mount')
"""
            ),
        ),
        _md_cell(
            "hdr-01-bootstrap",
            [
                (
                    "## Paso 1 — Arranque repetible del equipo (`colab_bootstrap.sh` y "
                    "`colab_site.py`)"
                ),
                "Clona el kit desde el bundle en Drive, instala las herramientas, crea el entorno "
                "bloqueado, restaura los datos con su huella (y el factor de mercado de China si "
                "se "
                "da su zip), compila el enlace episódico nativo y anota la ficha del equipo.",
            ],
        ),
        _code_cell(
            "step-01-bootstrap",
            _lines(
                r"""
%%bash
set -euo pipefail
[ -d "$T/kit" ] || git clone --quiet --branch "$RAMA_KIT" "$B" "$T/kit"
bash "$K/colab_bootstrap.sh" sistema
bash "$K/colab_bootstrap.sh" codigo --bundle "$B" --commit "$COMMIT_HEX" --trabajo "$T"
bash "$K/colab_bootstrap.sh" entorno --trabajo "$T"
datos=(--paquete "$P" --trabajo "$T" --destino "$DESTINO_DATOS")
[ -z "${FACTOR_CN_ZIP:-}" ] || datos+=(--factor-cn "$FACTOR_CN_ZIP")
bash "$K/colab_bootstrap.sh" datos "${datos[@]}"
bash "$K/colab_bootstrap.sh" nativo --trabajo "$T"
bash "$K/colab_bootstrap.sh" comprobar --trabajo "$T"
"$T/repo/.venv/bin/python" "$K/colab_site.py" --output "$T/equipo.json" --disk "$T"
"""
            ),
        ),
        _md_cell(
            "hdr-02-fp32-parity",
            [
                "## Paso 2 — FP32 IEEE estricto y la escritura fusionada en `sm_120`",
                "Comprueba, con el intérprete del entorno de la campaña, que el contrato numérico "
                "se "
                "puede imponer, y pasa el nivel 0 (bit a bit) y el nivel 1 (frente a FP64) del "
                "kernel antes de entrenar.",
            ],
        ),
        _code_cell(
            "step-02-fp32-parity",
            _lines(
                r"""
%%bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_F32_DEFAULT=ieee
# Con el intérprete del entorno de la campaña, no con el del cuaderno.
"$T/repo/.venv/bin/python" - <<'PY'
import torch
import triton

torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
assert triton.knobs.language.fp32_default == 'ieee', triton.knobs.language.fp32_default
assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
print('FP32 IEEE estricto y determinismo: torch', torch.__version__, '| triton', triton.__version__)
PY
"""
            ),
        ),
        _code_cell(
            "step-02b-kernel-plan",
            _lines(
                r"""
%%bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_F32_DEFAULT=ieee
"$T/repo/.venv/bin/python" "$ADDON/kernel_sm120_plan.py" --mode cpu-verify
"$T/repo/.venv/bin/python" "$ADDON/kernel_sm120_plan.py" --mode ptx-audit
"$T/repo/.venv/bin/python" "$ADDON/kernel_sm120_plan.py" --mode gpu-sweep \
  --output "$T/sm120_kernel_report.json"
"""
            ),
        ),
        _md_cell(
            "hdr-03-mps-probe",
            [
                "## Paso 3 — CUDA MPS y sonda de trabajadores (1, 2, 4 y 6)",
                "Si MPS arranca y responde se deja activo y la cola lo usará; si no, la cola "
                "corre con "
                "procesos independientes. La sonda mide memoria y pasos por minuto para elegir "
                "`TRABAJADORES`.",
            ],
        ),
        _code_cell(
            "step-03-mps-and-probe",
            _lines(
                r"""
%%bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_F32_DEFAULT=ieee
"$T/repo/.venv/bin/python" "$ADDON/mps_probe.py" --probe-live --keep-running \
  --workers "$TRABAJADORES" --output "$T/mps_report.json"
# El tope de memoria por cliente no se pasa a la cola: la campaña acota la suya, como en la DGX.
"$T/repo/.venv/bin/python" - "$T/mps_report.json" "$T/mps_entorno.sh" <<'PY'
import json
import shlex
import sys

report = json.load(open(sys.argv[1], encoding='utf-8'))
names = ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY') if report['mps_operational'] else ()
lines = [f"export {name}={shlex.quote(report['active_environment'][name])}" for name in names]
open(sys.argv[2], 'w', encoding='utf-8').write("".join(line + "\n" for line in lines))
print('MPS activo para la cola' if lines else 'Sin MPS: procesos independientes',
      '| trabajadores aconsejados:', report['recommended_workers'])
PY
"$T/repo/.venv/bin/python" "$ADDON/concurrency_probe.py" plan \
  --kit "$K" --repo "$T/repo" --state-root "$T/estado/sonda" \
  --views "$VISTAS" --declaration "$DECLARACION" \
  --origin "$ORIGIN_YEAR" --site "$MARS_TITAN_SITE"
# Tras la sonda en vivo (1, 2, 4 y 6), el análisis aconseja TRABAJADORES:
# "$T/repo/.venv/bin/python" "$ADDON/concurrency_probe.py" analyze \
#   --probe-root "$T/estado/sonda" --site-json "$T/equipo.json" --output "$T/sonda_resumen.json"
"""
            ),
        ),
        _md_cell(
            "hdr-04-sync-origin",
            [
                "## Paso 4 — Reanudación desde el espejo, preparación del origen y cola",
                "Recupera el estado anterior si existe (`colab_sync.py pull`), prepara cohortes, "
                "entradas y permiso (`colab_origin.sh`) y lanza la cola en `tmux`.",
            ],
        ),
        _code_cell(
            "step-04-origin-and-sync",
            _lines(
                r"""
%%bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_F32_DEFAULT=ieee
if [ -d "$MIRROR_DRIVE" ]; then
  "$T/repo/.venv/bin/python" "$K/colab_sync.py" pull --state "$ESTADO" --mirror "$MIRROR_DRIVE"
fi
origen=(--repo "$T/repo" --estado "$ESTADO" --origen "$ORIGIN_YEAR" --vistas "$VISTAS"
        --declaracion "$DECLARACION")
bash "$K/colab_origin.sh" cohortes "${origen[@]}"
bash "$K/colab_origin.sh" entradas "${origen[@]}" --receta "$RECETA"
bash "$K/colab_origin.sh" permiso "${origen[@]}" --nota "$NOTA_PERMISO"
"""
            ),
        ),
        _code_cell(
            "step-04b-launch",
            _lines(
                r"""
%%bash
set -euo pipefail
origen=(--repo "$T/repo" --estado "$ESTADO" --origen "$ORIGIN_YEAR" --vistas "$VISTAS"
        --declaracion "$DECLARACION")
# La cola corre en tmux desde un guion que lleva escrito su entorno: no depende del que tuviera
# un servidor tmux anterior, cada argumento conserva sus espacios y anota su sesión para la parada.
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'export %q\n' "MARS_TITAN_SITE=$MARS_TITAN_SITE" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    TRITON_F32_DEFAULT=ieee
  [ ! -s "$T/mps_entorno.sh" ] || cat "$T/mps_entorno.sh"
  printf 'ps -o sid= -p $$ | tr -d " " > %q\n' "$T/campana.sid"
  printf 'exec bash'
  printf ' %q' "$K/colab_origin.sh" lanza "${origen[@]}" --semillas "$SEMILLAS" \
    --niveles "$NIVELES" --trabajadores "$TRABAJADORES"
  printf '\n'
} > "$T/lanza_campana.sh"
tmux new-session -d -s campana \
  "bash $(printf '%q' "$T/lanza_campana.sh") 2>&1 | tee -a $(printf '%q' "$T/campana.log")"
bash "$K/colab_origin.sh" estado "${origen[@]}"
# Puerta de paridad con la DGX, cuando los dos estados tienen el mismo trabajo:
# "$T/repo/.venv/bin/python" "$K/colab_parity.py" --a <estado de la DGX> --b "$ESTADO" \
#   --model titans_mac --origin 2013 --seed 42 --output "$T/paridad.json"
"""
            ),
        ),
        _md_cell(
            "hdr-05-cu-guard",
            [
                "## Paso 5 — Copia periódica al espejo y guarda de unidades de cómputo",
                "Copia el estado a Drive y decide si se sigue con el siguiente trabajo u origen o "
                "se "
                "para antes del techo de 22,5 h o de la reserva de unidades. Las cifras de pasos "
                "son "
                "las medidas en esta sesión.",
            ],
        ),
        _code_cell(
            "step-05-cu-guard",
            _lines(
                r"""
%%bash
set -euo pipefail
"$T/repo/.venv/bin/python" "$K/colab_sync.py" push --state "$ESTADO" --mirror "$MIRROR_DRIVE"
"$T/repo/.venv/bin/python" "$ADDON/cu_budget_guard.py" decide \
  --panel-text "$COLAB_CU_PANEL" \
  --budget-cu 1000 \
  --session-elapsed-hours 2.0 \
  --remaining-steps 18000 \
  --steps-per-min 320.0 \
  --next-origin "$ORIGIN_YEAR" \
  --ledger "$MIRROR_DRIVE/cu_ledger.json" \
  --session-id "$SESSION_ID" \
  --state-dir "$ESTADO" \
  --mirror-dir "$MIRROR_DRIVE" \
  --kit-dir "$K"
"""
            ),
        ),
        _md_cell(
            "hdr-06-shutdown",
            [
                "## Paso 6 — Cierre seguro antes de 22,5 h o de agotar la reserva de unidades",
                "Pide la parada al proceso de la cola, espera a que llegue a su barrera, copia el "
                "estado al espejo y detiene MPS. Si la cola no para, no copia ni deja soltar "
                "la máquina.",
            ],
        ),
        _code_cell(
            "step-06-safe-shutdown",
            _lines(
                r"""
%%bash
set -euo pipefail
# La parada la recibe el proceso de la cola, que para en su siguiente barrera. Se busca solo en la
# sesión que anotó el lanzador, nunca por nombre en toda la máquina.
rm -f "$T/final_sync.json"
patron='mars_titan.training.comparison_campaig[n] run'
viva() {
  local codigo=0
  pgrep -s "$sid" -f "$patron" >/dev/null || codigo=$?
  [ "$codigo" -le 1 ] || { echo "pgrep falló con código $codigo" >&2; exit 1; }
  [ "$codigo" = 0 ]
}
if [ -s "$T/campana.sid" ]; then
  sid="$(cat "$T/campana.sid")"
  [[ "$sid" =~ ^[1-9][0-9]*$ ]] || { echo "Sesión de la cola no válida: $sid" >&2; exit 1; }
  pkill -TERM -s "$sid" -f "$patron" || true
  for _ in $(seq 1 "${ESPERA_PARADA_S:-1800}"); do
    viva || break
    sleep 1
  done
  if viva; then
    echo "La cola sigue viva tras la espera: no se copia ni se suelta la máquina" >&2
    exit 1
  fi
  rm -f "$T/campana.sid"
fi
"$T/repo/.venv/bin/python" "$K/colab_sync.py" push --state "$ESTADO" --mirror "$MIRROR_DRIVE" \
  > "$T/final_sync.json.tmp"
mv "$T/final_sync.json.tmp" "$T/final_sync.json"
cat "$T/final_sync.json"
if [ -s "$T/mps_entorno.sh" ] && command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  ( . "$T/mps_entorno.sh"; echo quit | nvidia-cuda-mps-control ) || true
fi
"""
            ),
        ),
        _code_cell(
            "step-06b-unassign",
            _lines(
                r"""
# La máquina se suelta solo si la copia final no dejó nada retenido ni pendiente de reintento
import json
import os
from pathlib import Path

AUTO_UNASSIGN = False  # Cambiar a True solo cuando se quiera liberar la máquina al acabar
sync_file = Path(os.environ.get('T', '/content/mars-titan')) / 'final_sync.json'
if AUTO_UNASSIGN and sync_file.is_file():
    report = json.loads(sync_file.read_text(encoding='utf-8'))
    if not report.get('held') and not report.get('retried_later'):
        from google.colab import runtime  # type: ignore[import-not-found]
        runtime.unassign()
"""
            ),
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.3",
            },
            "colab": {
                "name": "colab_rtx6000_bootstrap.ipynb",
                "provenance": [],
                "gpuType": "G4",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate_nbformat4_schema(nb: Any) -> list[str]:
    """Valida el esquema JSON obligatorio de `nbformat 4`."""
    errors: list[str] = []
    if not isinstance(nb, dict):
        return ["El cuaderno debe ser un objeto JSON"]
    if nb.get("nbformat") != 4:
        errors.append(f"nbformat debe ser 4 (recibido {nb.get('nbformat')!r})")
    minor = nb.get("nbformat_minor")
    if type(minor) is not int or minor < 0:
        errors.append(f"nbformat_minor inválido: {minor!r}")
    if not isinstance(nb.get("metadata"), dict):
        errors.append("metadata debe ser un diccionario")
    cells = nb.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("cells debe ser una lista no vacía")
        return errors

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"Celda {idx} no es un objeto")
            continue
        ctype = cell.get("cell_type")
        if ctype not in ("markdown", "code"):
            errors.append(f"Celda {idx} tiene cell_type inválido: {ctype!r}")
        if not isinstance(cell.get("metadata"), dict):
            errors.append(f"Celda {idx} carece de metadata dict")
        src = cell.get("source")
        if not isinstance(src, list) or not all(isinstance(line, str) for line in src):
            errors.append(f"Celda {idx} debe tener source como lista de cadenas")
        if ctype == "code":
            if cell.get("execution_count") is not None:
                errors.append(
                    f"Celda de código {idx} debe tener execution_count=None (plantilla limpia)"
                )
            if cell.get("outputs") != []:
                errors.append(f"Celda de código {idx} debe tener outputs=[] (plantilla limpia)")
    return errors


def extract_invoked_scripts(nb: dict[str, Any]) -> list[str]:
    """Extrae todos los guiones del kit invocados en celdas de código del cuaderno."""
    found: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell.get("source", []))
        for match in _SCRIPT_REF_RE.finditer(text):
            name = match.group(1)
            if name not in found:
                found.append(name)
    return found


def validate_notebook_contract(
    nb: dict[str, Any],
    *,
    repo_scripts_dir: Path,
    addon_dir: Path,
) -> dict[str, Any]:
    """Valida esquema `nbformat 4`, scripts reales del kit, orden de fases y guardas FP32/2024."""
    errors = validate_nbformat4_schema(nb)
    invoked = extract_invoked_scripts(nb) if not errors else []
    phase_indices: dict[str, int] = {}

    if not errors:
        # 1. Todos los guiones invocados deben existir en scripts/ o en scripts/colab_rtx6000
        for script_name in invoked:
            exists = (Path(repo_scripts_dir) / script_name).is_file() or (
                Path(addon_dir) / script_name
            ).is_file()
            if not exists:
                errors.append(f"Guion invocado no existe en el kit: {script_name}")

        all_code_text = ""
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            text = "".join(cell["source"])
            all_code_text += "\n" + text

            # 2. Patrones prohibidos (TF32, autocast, no determinismo, 2024, subidas a red)
            for label, pat in _FORBIDDEN_PATTERNS:
                if pat.search(text):
                    errors.append(f"Violación de contrato ({label}) en celda {idx}")

            # 3. Registrar índice de cada fase obligatoria
            if "colab_bootstrap.sh" in text and "bootstrap" not in phase_indices:
                phase_indices["bootstrap"] = idx
            if "kernel_sm120_plan.py" in text and "fp32_parity" not in phase_indices:
                phase_indices["fp32_parity"] = idx
            if (
                "mps_probe.py" in text
                and "concurrency_probe.py" in text
                and "mps_and_probe" not in phase_indices
            ):
                phase_indices["mps_and_probe"] = idx
            if (
                "colab_origin.sh" in text
                and "colab_sync.py" in text
                and "origin_and_sync" not in phase_indices
            ):
                phase_indices["origin_and_sync"] = idx
            if "cu_budget_guard.py" in text and "cu_guard" not in phase_indices:
                phase_indices["cu_guard"] = idx
            if "runtime.unassign" in text and "safe_shutdown" not in phase_indices:
                phase_indices["safe_shutdown"] = idx

        # 4. Guardas obligatorias FP32 y determinismo
        for label, pat in _REQUIRED_GUARDS:
            if not pat.search(all_code_text):
                errors.append(f"Falta guarda obligatoria FP32/determinismo: {label}")

        # 5. Orden estricto de las 6 fases
        required_order = (
            "bootstrap",
            "fp32_parity",
            "mps_and_probe",
            "origin_and_sync",
            "cu_guard",
            "safe_shutdown",
        )
        missing_phases = [p for p in required_order if p not in phase_indices]
        if missing_phases:
            errors.append(f"Faltan fases obligatorias en el cuaderno: {missing_phases}")
        else:
            seq = [phase_indices[p] for p in required_order]
            if seq != sorted(seq) or len(set(seq)) != len(seq):
                errors.append(f"Orden de fases incorrecto en el cuaderno: {phase_indices}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "invoked_scripts": invoked,
        "phase_indices": phase_indices,
    }


def write_bootstrap_notebook(output_path: Path) -> Path:
    """Escribe el cuaderno `.ipynb` en disco en UTF-8 con formato determinista."""
    nb = build_bootstrap_notebook()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera o valida el cuaderno .ipynb de arranque en Colab."
    )
    parser.add_argument(
        "--generate", type=Path, default=None, help="Escribe el .ipynb en esta ruta."
    )
    parser.add_argument("--validate", type=Path, default=None, help="Valida un .ipynb existente.")
    parser.add_argument(
        "--repo-scripts",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--addon-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)

    if args.generate is not None:
        write_bootstrap_notebook(args.generate)
    target = args.validate or args.generate
    if target is None:
        parser.error("Indica --generate o --validate")
    nb = json.loads(target.read_text(encoding="utf-8"))
    report = validate_notebook_contract(
        nb, repo_scripts_dir=args.repo_scripts, addon_dir=args.addon_dir
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
