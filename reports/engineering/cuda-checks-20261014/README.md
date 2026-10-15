# Comprobaciones CUDA explícitas (14 de octubre de 2026)

Este informe acompaña a [#621](https://github.com/GonxKZ/mars-titan/issues/621). Recoge la repetición de todas las comprobaciones CUDA que no forman parte de la batería normal, sobre `develop` 52802c61 y con la GPU libre de otras cargas de cómputo. Ninguna aplica pasos de optimizador. Los entrenadores usan registradores que guardan gradientes sin cambiar pesos, y nada de esto mide utilidad predictiva. La protección local de aprendizaje siguió activa.

## Entorno

| Elemento | Valor |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8.188 MiB, capacidad 8.9, controlador 595.91.07 |
| Perfil de energía | `power-saver`, sin cambios |
| Software | PyTorch 2.14.0+cu130 (CUDA 13.0, cuDNN 9.24) |
| Enlace nativo | `native-candidate-cuda` compilado desde `develop` df7bb523 (`f9e9d13a…`). `native/` no cambia hasta 52802c61 |
| Aritmética | Sin TF32 en cuBLAS ni en cuDNN, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, dos hilos de CPU |
| Plaza | Cada archivo en su propio proceso con la plaza `gpu` de `memslot` y 6 GiB de límite |

## Resultados

| Comprobación | Resultado | Tiempo |
| --- | --- | ---: |
| [`memory/cuda_financial_attributes_check.py`](../../../tests/memory/cuda_financial_attributes_check.py) | 1 pasa | 19,1 s |
| [`memory/cuda_financial_session_check.py`](../../../tests/memory/cuda_financial_session_check.py) | 1 pasa | 152,0 s |
| [`memory/cuda_gru_session_check.py`](../../../tests/memory/cuda_gru_session_check.py) | 1 pasa | 12,2 s |
| [`memory/cuda_mars_titan_correction_check.py`](../../../tests/memory/cuda_mars_titan_correction_check.py) | 1 pasa | 6,0 s |
| [`memory/cuda_mature_error_check.py`](../../../tests/memory/cuda_mature_error_check.py), con `MARS_TITAN_HARDWARE_PROFILE=rtx4070-laptop` | 2 pasan | 53,7 s |
| [`models/candidate/cuda_candidate_check.py`](../../../tests/models/candidate/cuda_candidate_check.py), como guion ([recibo](candidate-cuda.json)) | 6 casos | |
| [`models/titans/cuda_episodic_readout_check.py`](../../../tests/models/titans/cuda_episodic_readout_check.py) | 1 pasa | 2,7 s |
| [`models/titans/cuda_financial_blocks_check.py`](../../../tests/models/titans/cuda_financial_blocks_check.py) | 1 pasa, 1 aviso | 3,3 s |
| [`models/titans/cuda_local_control_check.py`](../../../tests/models/titans/cuda_local_control_check.py) | 1 pasa | 4,0 s |
| [`models/titans/cuda_memory_stability_check.py`](../../../tests/models/titans/cuda_memory_stability_check.py) | 1 pasa | 4,1 s |
| [`posttraining/cuda_candidate_adapters_check.py`](../../../tests/posttraining/cuda_candidate_adapters_check.py), con la corrección de #621 ([recibo](candidate-adapters-cuda.json)) | 2 pasan | 19,4 s |
| [`posttraining/cuda_readout_adapters_check.py`](../../../tests/posttraining/cuda_readout_adapters_check.py) | 2 pasan | 150,2 s |
| [`posttraining/cuda_titans_adapters_check.py`](../../../tests/posttraining/cuda_titans_adapters_check.py) | 4 pasan | 25,7 s |
| [`training/cuda_candidate_run_check.py`](../../../tests/training/cuda_candidate_run_check.py) | 6 pasan | 42,4 s |
| [`training/cuda_cm_v1_control_check.py`](../../../tests/training/cuda_cm_v1_control_check.py) | 8 pasan | 116,5 s |
| [`training/cuda_mars_titan_run_check.py`](../../../tests/training/cuda_mars_titan_run_check.py) | 10 pasan | 83,4 s |
| [`training/cuda_regeneration_check.py`](../../../tests/training/cuda_regeneration_check.py) ([recibo](regeneration-cuda.json)) | 3 pasan | 58,0 s |
| [`training/candidate_memory_check.py`](../../../tests/training/candidate_memory_check.py), en `cuda:0` con 128, 256 y 512 activos ([recibo](candidate-memory-cuda.json)) | 4 pasan | 987,0 s |

Los tiempos son los de pytest e incluyen la construcción de las fixtures en CPU dentro de la plaza.

## Fallos y causas

**Adaptadores de la GRU candidata.** La primera ejecución dio `test_null_adapters_reproduce_the_frozen_parent_on_the_device` como fallida. Los tres casos que recorre (la cabeza con su corrección a cero, la continuación completa y la continuación anclada de reserva) emitieron en `cuda:0` exactamente las filas del padre congelado, pero la última aserción esperaba solo los dos primeros. El auxiliar `cases` de `test_candidate_adapters.py` añade la continuación anclada desde [#540](https://github.com/GonxKZ/mars-titan/pull/540), y la comprobación CUDA no se había vuelto a ejecutar desde entonces. La aserción espera ahora los tres casos. En la repetición, cada caso hace una actualización con el registrador y reproduce las filas del padre, el padre congelado difiere de CPU en 2,1·10⁻¹⁷ como mucho y los gradientes de la cabeza en la ventana del padre (42 actualizaciones) en 5,2·10⁻¹⁸.

**M2 y M3.** Sin `MARS_TITAN_HARDWARE_PROFILE`, las dos pruebas de `cuda_mature_error_check.py` se detienen antes de empezar, como describe la [guía de M3](../../../docs/engineering/m3-write-policy.md). Con el perfil declarado pasan las dos.

**Memoria de la candidata.** `candidate_memory_check.py` usa `cpu` si no se indica otro dispositivo. En la primera tanda se lanzó sin `MARS_TITAN_CANDIDATE_MEMORY_CHECK_DEVICE` y el proceso superó los 6 GiB de la plaza en CPU con 128 y 256 activos, así que el núcleo lo detuvo. No es un fallo de la comprobación. Se repitió el 15 de octubre en `cuda:0` con 128, 256 y 512 activos y pasan las cuatro pruebas. El pico de memoria que PyTorch asignó en la GPU fue de 355, 621 y 1.151 MiB sin acumulación ni recomputación, de 156 a 258 MiB con acumulación y de 144 a 164 MiB con recomputación, y el proceso llegó a 1,9 GiB de memoria residente. La recomputación sola da los mismos gradientes bit a bit. La acumulación cambia el orden de las sumas y los gradientes difieren como mucho en 9·10⁻⁸, con gradientes de hasta 0,1. En esa repetición la GPU tenía cargado el modelo de otra aplicación, con unos 5,9 GiB. Los picos salen de los contadores de PyTorch de este proceso y no incluyen esa memoria, pero los tiempos de esa fila pueden estar afectados.

**Aviso de cuBLAS.** `cuda_financial_blocks_check.py` deja el aviso de PyTorch «Attempting to run cuBLAS, but there was no current CUDA context», que aparece cuando el primer uso de cuBLAS ocurre en el hilo del backward. PyTorch fija entonces el contexto primario y la comprobación pasa con las mismas tolerancias.

## Regeneración

La comprobación de regeneración ajusta Titans-MAC, el lector M1 de MARS-TITAN y la GRU episódica en `cuda:0` con el registrador y las regenera en el mismo dispositivo. Las tablas de validación, calibración y evaluación salen idénticas a las registradas en las tres familias, con el mismo archivo. Es la condición que la retención exige antes de liberar una tabla por fila ([#363](https://github.com/GonxKZ/mars-titan/issues/363)).

## Orden

Desde la raíz, con el enlace episódico compilado con CUDA y cada archivo en su propia ejecución:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MARS_TITAN_EPISODIC_NATIVE=<enlace episódico nativo con CUDA>
export MARS_TITAN_HARDWARE_PROFILE=rtx4070-laptop
export UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA>
for check in $(find tests -name "cuda_*_check.py" ! -name cuda_candidate_check.py | sort); do
  memslot gpu --max 6G -- uv run --no-sync python -m pytest -q "$check"
done
(cd tests/models/candidate && uv run --no-sync python cuda_candidate_check.py --output candidate-cuda.json)
MARS_TITAN_CANDIDATE_MEMORY_CHECK_DEVICE=cuda:0 MARS_TITAN_CANDIDATE_MEMORY_CHECK_ASSETS=128,256,512 \
  uv run --no-sync python -m pytest -q tests/training/candidate_memory_check.py
```
