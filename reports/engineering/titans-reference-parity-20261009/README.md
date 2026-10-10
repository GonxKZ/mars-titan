# Paridad del núcleo Titans con implementaciones públicas

Medidas del 9 y 10 de octubre de 2026 para [#434](https://github.com/GonxKZ/mars-titan/issues/434). El análisis está en [titans-reference-implementations.md](../../../docs/research/titans-reference-implementations.md).

| Archivo | Contenido |
| --- | --- |
| [`inventory.json`](inventory.json) | Comprobaciones de código oficial e inventario de implementaciones con URL, licencia, commit, fecha, variantes, nivel de revisión y motivo |
| [`parity-cpu.json`](parity-cpu.json) | Paridad con `lucidrains/titans-pytorch@1d40c445` y comprobaciones de `fla@855e5026` en CPU, FP32 y FP64 |
| [`parity-cuda.json`](parity-cuda.json) | La misma paridad con lucidrains en `cuda:0`, FP32 y FP64. Sin las comprobaciones de `fla` |

Ningún recibo ejecuta pasos de optimizador (`optimizer_steps: 0`). Cada uno conserva las tolerancias declaradas, las versiones del entorno aislado, los hashes de los archivos de terceros y del proyecto y, para cada tensor, la diferencia absoluta, la relativa y el cociente frente a la tolerancia.

## Reproducción

Las referencias se descargan fuera del repositorio, cada una en su directorio, en los commits `1d40c445fa1794fb28582c721786482b5b683e47` de `lucidrains/titans-pytorch` y `855e502613d47d8cc139153f0d87ecae5831ac3c` de `fla-org/flash-linear-attention`. El entorno aislado se crea con `uv venv --python 3.12`, `torch==2.14.0+cu130` desde el índice de PyTorch y las dependencias de lucidrains (`assoc-scan`, `axial_positional_embedding`, `einops`, `einx`, `hyper-connections`, `rotary-embedding-torch`, `tensordict`, `tqdm`, `x-transformers`) con `--exclude-newer 2026-10-01T00:00:00Z`.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  memslot light -- <entorno>/bin/python -I benchmarks/titans_reference_parity.py \
  --lucidrains-root <titans-pytorch@1d40c445> --fla-root <flash-linear-attention@855e5026> \
  --output parity-cpu.json
```

Dos ejecuciones CPU dieron exactamente las mismas cifras. La ejecución tarda unos 8 s.

## CUDA

La comparación en `cuda:0` usa la misma orden con `--device cuda:0` y `memslot gpu` en lugar de `memslot light`, sin `CUDA_VISIBLE_DEVICES=-1`. Las comprobaciones de `fla` se limitan a CPU.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  memslot gpu -- <entorno>/bin/python -I benchmarks/titans_reference_parity.py --device cuda:0 \
  --lucidrains-root <titans-pytorch@1d40c445> --fla-root <flash-linear-attention@855e5026> \
  --output parity-cuda.json
```

Se ejecutó el 10 de octubre de 2026 en una RTX 4070 Laptop GPU (capacidad 8.9, CUDA 13.0), con TF32 desactivado y algoritmos deterministas. Tardó 23 s y reservó como máximo 433 MiB. El primer intento falló al reiniciar las estadísticas de memoria antes de crear el contexto CUDA. El arnés lo crea ahora explícitamente y el recibo CPU se regeneró con las mismas cifras.

Los resultados repiten el patrón de CPU. En FP64 pasan los 5 casos, con un cociente máximo de 0,083. En FP32 pasan los 3 casos sin residual y los 2 casos con residual y LayerNorm quedan fuera de la tolerancia declarada en 7 y 9 de 19 tensores. La diferencia FP32 entre implementaciones es como mucho 1,61 veces el redondeo de cada una frente a su evaluación FP64, y las evaluaciones FP64 coinciden hasta 2,2·10⁻¹².
