#!/usr/bin/env bash
# Arranque de una máquina de Google Colab como segundo equipo de la campaña.
#
# Cada paso comprueba lo que hace y se puede repetir: lo ya hecho se conserva. Nada de este
# guion entrena ni sube datos. Los datos se restauran desde el paquete que ya está en Drive
# (el mismo del que se restauró la DGX), con la huella SHA-256 de cada volumen.
#
# Uso, dentro de Colab y con Drive montado:
#   bash colab_bootstrap.sh <paso> --paquete <carpeta del paquete en Drive> \
#        --bundle <mars-titan.bundle> --commit <40 hex> [--trabajo /content/mars-titan]
#        [--destino /home/gonzalo] [--grupos "A01-preparados-v31 Precios-US"]
#
# Pasos, en este orden:
#   sistema   herramientas del sistema (zstd, tmux, cmake, ninja, git) y uv
#   codigo    copia fija del repositorio en el commit pedido
#   entorno   entorno bloqueado del proyecto (uv sync --locked --all-extras)
#   datos     restaura los grupos de datos que entrenan (EE. UU., China y vistas)
#   nativo    compila el enlace episódico que usan los lectores m1, m2 y m3
#   comprobar ficha del equipo, pruebas de CUDA del núcleo y memoria fusionada
#   todo      todos los anteriores
#   busqueda  Optuna para scripts/campaign_search.py, en una carpeta aparte (no entra en todo)
#
# El año 2024 no forma parte del paquete de entrenamiento y sigue cerrado.
set -euo pipefail

paso="${1:-}"
[ -n "$paso" ] || { echo "Falta el paso. Consulta la cabecera de este guion." >&2; exit 2; }
shift

paquete="" bundle="" commit="" trabajo="/content/mars-titan" destino="/home/gonzalo" grupos=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --paquete) paquete="$2"; shift 2 ;;
    --bundle) bundle="$2"; shift 2 ;;
    --commit) commit="$2"; shift 2 ;;
    --trabajo) trabajo="$2"; shift 2 ;;
    --destino) destino="$2"; shift 2 ;;
    --grupos) grupos="$2"; shift 2 ;;
    *) echo "Opción desconocida: $1" >&2; exit 2 ;;
  esac
done

trabajo="$(realpath -m "$trabajo")"
[ -z "$paquete" ] || paquete="$(realpath -m "$paquete")"
[ -z "$bundle" ] || bundle="$(realpath -m "$bundle")"
[ -z "$destino" ] || destino="$(realpath -m "$destino")"
repo="$trabajo/repo"
# Los guiones de este kit se leen de la copia que contiene a este, que puede ser más nueva
# que el commit fijado de la campaña.
herramientas="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"

# Grupos del paquete que entrenan. Los originales (B01 a B07) no hacen falta.
GRUPOS=(
  A00-macroeconomia A01-preparados-v31 A02-etiquetas-v31 A03-muestras-codificadas-v31
  A04-vistas-US A05-vistas-CN A06-vistas-conjuntas
  Etiquetas-US Muestras-US Precios-US Ventanas-US
)

# Con --grupos se restauran solo los nombrados, por ejemplo los de un mercado para empezar.
if [ -n "$grupos" ]; then
  read -r -a GRUPOS <<< "$grupos"
fi

falta() { echo "FALTA: $*" >&2; exit 1; }

paso_sistema() {
  local paquetes=() orden
  for orden in zstd tmux cmake ninja git rsync; do
    command -v "$orden" >/dev/null 2>&1 && continue
    case "$orden" in
      ninja) paquetes+=(ninja-build) ;;
      *) paquetes+=("$orden") ;;
    esac
  done
  if [ "${#paquetes[@]}" -gt 0 ]; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${paquetes[@]}"
  fi
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  uv --version
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || falta "no hay GPU visible: elige en Colab el entorno con la RTX PRO 6000"
}

paso_codigo() {
  [ -f "$bundle" ] || falta "no existe el bundle de Git ($bundle)"
  [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || falta "--commit debe ser el commit completo de 40 hex"
  mkdir -p "$trabajo"
  if [ ! -d "$repo/.git" ]; then
    git clone --quiet "$bundle" "$repo"
  else
    git -C "$repo" fetch --quiet "$bundle" '+refs/heads/*:refs/remotes/bundle/*'
  fi
  git -C "$repo" checkout --quiet --detach "$commit"
  [ "$(git -C "$repo" rev-parse HEAD)" = "$commit" ] || falta "el repositorio no quedó en $commit"
  [ -z "$(git -C "$repo" status --porcelain)" ] || falta "la copia del repositorio tiene cambios"
  echo "código en $commit"
}

paso_entorno() {
  [ -d "$repo" ] || falta "ejecuta antes el paso codigo"
  (cd "$repo" && uv sync --locked --all-extras)
  (cd "$repo" && uv run --no-sync python -c "
import torch, numpy, pyarrow
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'numpy', numpy.__version__, 'pyarrow', pyarrow.__version__)
assert torch.cuda.is_available(), 'CUDA no responde'
print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
")
}

paso_datos() {
  [ -d "$paquete" ] || falta "no existe la carpeta del paquete de datos ($paquete)"
  [ -f "$paquete/RESTAURAR.py" ] || falta "el paquete no trae RESTAURAR.py"
  mkdir -p "$destino"
  local argumentos=() grupo
  for grupo in "${GRUPOS[@]}"; do
    argumentos+=(--grupo "$grupo")
  done
  # El restaurador comprueba tamaño y SHA-256 de cada volumen, se puede reanudar y se
  # detiene si un archivo existente contiene otros bytes.
  uv run --no-project --python 3.12 "$paquete/RESTAURAR.py" \
    --paquetes "$paquete" --destino "$destino" "${argumentos[@]}"
  df -h "$destino" | tail -1
}

paso_nativo() {
  [ -d "$repo/.venv" ] || falta "ejecuta antes el paso entorno"
  local salida="$trabajo/nativo"
  # Las mismas opciones del enlace de la DGX: solo el banco episódico y su simulación.
  cmake -S "$repo/native" -B "$salida" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON -DMARS_TITAN_BUILD_SIMULATION=ON \
    -DMARS_TITAN_BUILD_RUNNER=OFF -DMARS_TITAN_BUILD_CANDIDATE=OFF \
    -DMARS_TITAN_BUILD_FINANCIAL=OFF -DMARS_TITAN_BUILD_PPO=OFF \
    -DMARS_TITAN_ENABLE_CUDA=OFF -DMARS_TITAN_TORCH_ENVIRONMENT="$repo/.venv"
  cmake --build "$salida" --target _episodic_native -j 4
  ls -la "$salida"/_episodic_native*.so
}

paso_comprobar() {
  [ -d "$repo/.venv" ] || falta "ejecuta antes el paso entorno"
  local nativo=""
  if [ -d "$trabajo/nativo" ]; then
    nativo="$(find "$trabajo/nativo" -maxdepth 1 -name '_episodic_native*.so' 2>/dev/null | head -1 || true)"
  fi
  (
    cd "$repo"
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_REQUIRE_CUDA=1 CUDA_VISIBLE_DEVICES=0
    [ -z "$nativo" ] || export MARS_TITAN_EPISODIC_NATIVE="$nativo"
    uv run --no-sync python "$herramientas/colab_site.py" \
      --output "$trabajo/equipo.json" --disk "$trabajo"
    # Emisión, gradientes y recuperación en CUDA sin pasos de optimizador, y la escritura
    # de memoria fusionada (Triton) frente a su referencia.
    uv run --no-sync pytest -q -rs \
      tests/models/titans/test_financial_learning_scale_cuda.py \
      tests/models/titans/test_fused_memory_write.py
  )
}

paso_busqueda() {
  [ -d "$repo/.venv" ] || falta "ejecuta antes el paso entorno"
  local sitio="$trabajo/busqueda-site"
  # Optuna no forma parte del entorno bloqueado: va a una carpeta que solo ve la búsqueda,
  # con las mismas versiones que en la DGX y sin dependencias, para no tapar las del entorno.
  uv pip install --python "$repo/.venv/bin/python" --target "$sitio" --no-deps \
    optuna==5.1.0 alembic==1.20.1 colorlog==6.12.0 mako==1.4.3 sqlalchemy==2.1.9 tqdm==4.70.1
  PYTHONPATH="$sitio" "$repo/.venv/bin/python" -c "
import optuna
from optuna.storages.journal import JournalFileBackend
print('optuna', optuna.__version__)
"
  echo "Para buscar: PYTHONPATH=$sitio $repo/.venv/bin/python <kit>/campaign_search.py ..."
}

case "$paso" in
  sistema) paso_sistema ;;
  codigo) paso_codigo ;;
  entorno) paso_entorno ;;
  datos) paso_datos ;;
  nativo) paso_nativo ;;
  comprobar) paso_comprobar ;;
  busqueda) paso_busqueda ;;
  todo) paso_sistema; paso_codigo; paso_entorno; paso_datos; paso_nativo; paso_comprobar ;;
  *) echo "Paso desconocido: $paso" >&2; exit 2 ;;
esac
