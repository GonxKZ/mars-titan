#!/usr/bin/env bash
# Entorno optimizado para MARS-TITAN en NVIDIA RTX PRO 6000 Blackwell (sm_120 / GB202, Google Colab G4)
# y compatibilidad automática con DGX Spark (GB10, sm_121).
#
# Uso en Google Colab G4 (RTX PRO 6000 96 GB GDDR7, 188 SMs, 128 MB L2, 120 TFLOPS FP32):
#   source scripts/colab_rtx_pro_6000_env.sh
#   # Opcional para saturar los 96 GB VRAM con 4-8 trabajadores paralelos:
#   start_mars_titan_mps

set -euo pipefail

# 1. Arquitectura Blackwell explícita (sin fallback PTX JIT; sm_120 para RTX PRO 6000)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-120-real}"

# 2. Contrato numérico estricto: FP32 IEEE determinista (sin TF32 ni precisiones reducidas)
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export MARS_TITAN_GPU_PROFILE="${MARS_TITAN_GPU_PROFILE:-auto}"

# 3. Asignador CUDA sin fragmentación para sesiones variables en 96 GB GDDR7
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

# 4. Cache persistente de Triton (evita recompilar kernels entre reinicios de sesión)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/mars-titan-triton-cache-sm120}"
mkdir -p "${TRITON_CACHE_DIR}"

start_mars_titan_mps() {
    export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/mars-titan-mps/pipe}"
    export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/mars-titan-mps/log}"
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
    if ! pgrep -f "nvidia-cuda-mps-control" >/dev/null 2>&1; then
        nvidia-cuda-mps-control -d
        echo "[MARS-TITAN] CUDA MPS iniciado en ${CUDA_MPS_PIPE_DIRECTORY}"
    else
        echo "[MARS-TITAN] CUDA MPS ya estaba activo en ${CUDA_MPS_PIPE_DIRECTORY}"
    fi
}

stop_mars_titan_mps() {
    if [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && [ -d "${CUDA_MPS_PIPE_DIRECTORY}" ]; then
        echo quit | nvidia-cuda-mps-control || true
    fi
}
