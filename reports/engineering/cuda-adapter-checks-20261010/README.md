# Comprobaciones CUDA de adaptadores, lectores y GRU candidata (10 de octubre de 2026)

Estas comprobaciones cierran los dos puntos de CUDA que seguían abiertos en [#364](https://github.com/GonxKZ/mars-titan/issues/364) y [#383](https://github.com/GonxKZ/mars-titan/issues/383). La primera pregunta es si los adaptadores a cero de los lectores episódicos y de la GRU candidata reproducen en `cuda:0` al padre congelado. La segunda es si el entrenador de la candidata coincide entre CPU y `cuda:0` con el calentamiento de 12 meses de su receta. La comprobación del lector puro cubre además el modo `first_read`. Ninguna aplica pasos de optimizador. Los entrenadores usan registradores que guardan gradientes sin heredar de `torch.optim.Optimizer` ni cambiar pesos. El bloqueo de aprendizaje siguió vigente y nada de esto mide utilidad predictiva.

El [resumen estructurado](summary.json) recoge el entorno, los tiempos y las huellas. Cada comprobación tiene su recibo en esta carpeta.

## Entorno

| Elemento | Valor |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8.188 MiB, capacidad 8.9, controlador 595.91.07 |
| Perfil de energía | `power-saver`, sin cambios |
| Software | Python 3.12.14, PyTorch 2.14.0+cu130 (CUDA 13.0, cuDNN 9.24), NumPy 2.5.3, PyArrow 25.0.1 |
| Enlace nativo | `native-candidate-cuda` compilado desde `f1aca19e` (`3fdbb5a3…`). `native/` no cambia hasta `b92e07a0` |
| Código | `develop` en `b92e07a0` con las comprobaciones de esta rama. `src/` no cambia |
| Aritmética | Sin TF32 en cuBLAS ni en cuDNN, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, dos hilos de CPU |

Las cuatro comprobaciones se ejecutaron seguidas en una sola plaza `gpu` de `memslot`, entre las 13:43 y las 13:47, sin otra carga de cómputo CUDA. El pico de `nvidia-smi` fue de 904 MiB, con unos 532 MiB del escritorio. La de los lectores se repitió a las 14:02, en otra plaza, después de ajustar su tolerancia a la precisión del padre (ver más abajo).

## Resultados

| Comprobación | Resultado | Tiempo | Pico del asignador |
| --- | --- | ---: | ---: |
| [`cuda_episodic_readout_check.py`](../../../tests/models/titans/cuda_episodic_readout_check.py), lector puro con `per_step` y `first_read` | 1 pasa, 16 configuraciones y 2 controles | 11,6 s | 64 MiB |
| [`cuda_candidate_run_check.py`](../../../tests/training/cuda_candidate_run_check.py), entrenador de la GRU candidata con 12 meses de calentamiento | 6 pasan | 51,6 s | 128 MiB |
| [`cuda_candidate_adapters_check.py`](../../../tests/posttraining/cuda_candidate_adapters_check.py), cabeza y continuación de la GRU candidata | 2 pasan | 32,9 s | 86 MiB |
| [`cuda_readout_adapters_check.py`](../../../tests/posttraining/cuda_readout_adapters_check.py), núcleo y lectura del lector M1 | 2 pasan | 124,3 s | 65 MiB |

**Lector puro ([recibo](episodic-readout-cuda.json)).** Recorre FP32 y FP64, K = 1, 2 y 4 y una memoria vacía en los dos modos de selección. En todos, los identificadores elegidos coinciden exactamente con CPU, el estado queda dentro de las tolerancias declaradas y los gradientes difieren como mucho en 4,8·10⁻⁷ en FP32 y 8,9·10⁻¹⁶ en FP64. Con `first_read`, todos los pasos leen los episodios del primero, y con K = 1 coincide bit a bit con `per_step` también en el dispositivo. Con `per_step` y K > 1 la selección cambia entre pasos en esta fixture, así que la comparación no es trivial. El control con parámetros que fuerzan otra búsqueda reproduce en `cuda:0` lo que ya se comprobaba en CPU: con K = 2 y 4, `per_step` pasa del episodio 10 al 20 y `first_read` se queda en el 10. Los generadores de CPU y CUDA no cambian.

**Entrenador de la GRU candidata ([recibo](candidate-trainer-cuda.json)).** Las cuatro comparaciones del recorrido completo (FP32 y FP64, K = 1 y 4, 11 actualizaciones) repiten las del 9 de octubre. La de la ventana v2 se ejecuta ahora con los 12 meses de calentamiento de la receta de campaña. En FP32 y FP64, el ajuste, el traslado y el traslado en `cuda:0` del estado elegido en CPU registran esos meses en `bank_policy`, `anchor_warmup` los reconstruye a partir de las fases de la ventana y las fases coinciden entre CPU y el dispositivo. La evaluación trasladada observa entradas anteriores a su inicio, así que el calentamiento no es vacío. Las predicciones de los tres pares coinciden en filas y huella, y en valores dentro de las tolerancias de cada precisión.

**Adaptadores de la GRU candidata ([recibo](candidate-adapters-cuda.json)).** El padre es la ventana técnica US+CN de `test_candidate_adapters.py`, en float64 y con 12 meses de calentamiento. En `cuda:0`, la cabeza con su corrección a cero y la continuación completa, ajustadas por etapas con las filas nuevas de la ventana siguiente, emiten exactamente las filas del padre congelado calculado en el mismo dispositivo en validación, calibración y evaluación. El padre congelado de CPU y el de `cuda:0` difieren como mucho en 2,1·10⁻¹⁷. En la ventana del padre, con 42 actualizaciones, la cabeza recibe gradiente solo en su corrección y difiere de CPU en 5,2·10⁻¹⁸ como mucho.

**Adaptadores de los lectores ([recibo](readout-adapters-cuda.json)).** El padre es el lector M1 elegido en la campaña reducida de `test_mars_titan_campaign.py`. En `cuda:0`, los cuatro casos de la matriz v3 para el lector con banco (núcleo, lectura episódica, ambos y continuación completa), ajustados por etapas, emiten exactamente las filas del padre congelado en el mismo dispositivo. El padre congelado de CPU y el de `cuda:0` difieren como mucho en 4,4·10⁻¹⁶. Con núcleo y lectura en la ventana del padre, las 9 actualizaciones tienen los mismos grupos (`episodic_read_adapters` y `core_adapters`) y gradientes a 5,4·10⁻¹⁷ de CPU como mucho.

## Decisiones y límites

- La igualdad exacta se exige frente al padre congelado del mismo dispositivo, como en la comprobación de Titans-MAC. Entre CPU y `cuda:0` solo cabe exigir tolerancias.
- El lector y la candidata de las fixtures son float64, la precisión que el lector hereda del Titans-MAC reducido de las pruebas. La primera ejecución de la comprobación de los lectores usaba las tolerancias de FP32 de su receta de campaña (2·10⁻⁴ y 2·10⁻⁶). Pasó con las mismas diferencias, pero era una cota demasiado holgada para float64. Ahora la tolerancia se deriva de la precisión del padre (10⁻⁸ y 10⁻¹⁰) y el recibo es el de la repetición.
- En las ventanas por etapas de las fixtures cada caso hace una sola actualización, porque hay pocas filas nuevas. Por eso los gradientes se comparan en la ventana del padre, con 42 y 9 actualizaciones.
- Las comprobaciones de los adaptadores construyen en CPU la campaña o la ventana del padre dentro de la plaza de GPU. Con estas fixtures son unos dos minutos en el caso de los lectores y no leen datos reales.
- Son fixtures técnicas pequeñas. No dicen nada del caudal ni de la memoria con la edición real.

## Orden

Desde la raíz, con el enlace nativo compilado con CUDA, cada archivo en su propia ejecución, como se hizo aquí:

```bash
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export MARS_TITAN_EPISODIC_NATIVE=<enlace episódico nativo con CUDA>
export UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA>
MARS_TITAN_EPISODIC_READOUT_CHECK_REPORT=$PWD/episodic-readout-cuda.json \
  uv run --no-sync python -m pytest -q tests/models/titans/cuda_episodic_readout_check.py
MARS_TITAN_CANDIDATE_RUN_CHECK_REPORT=$PWD/candidate-trainer-cuda.json \
  uv run --no-sync python -m pytest -q tests/training/cuda_candidate_run_check.py
MARS_TITAN_CANDIDATE_ADAPTERS_CHECK_REPORT=$PWD/candidate-adapters-cuda.json \
  uv run --no-sync python -m pytest -q tests/posttraining/cuda_candidate_adapters_check.py
MARS_TITAN_READOUT_ADAPTERS_CHECK_REPORT=$PWD/readout-adapters-cuda.json \
  uv run --no-sync python -m pytest -q tests/posttraining/cuda_readout_adapters_check.py
```

Las comprobaciones de los adaptadores y la del entrenador admiten `..._CHECK_DEVICE=cpu` para ensayar la lógica sin GPU, lo que no acredita CUDA. Las tres pasaron así en CPU antes de ejecutarlas en `cuda:0`.
