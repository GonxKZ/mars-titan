# Refinamientos del lector M1 con K = 4

Recibos del 13 de octubre de 2026 para [#548](https://github.com/GonxKZ/mars-titan/issues/548). La rama parte de la de [#549](https://github.com/GonxKZ/mars-titan/issues/549) ([#589](https://github.com/GonxKZ/mars-titan/pull/589)), que ya difiere las comprobaciones de finitud de cada evento, y las cifras de «antes» son las de esa rama (d020ff07). Los refinamientos K = 1, 2 y 4 son un eje experimental propio y no cuentan como actualizaciones de memoria de Titans.

| Archivo | Contenido |
| --- | --- |
| [`shapes.json`](shapes.json) | Lanzamientos e igualdad de bits de una llamada del lector con las formas de la campaña, para K = 1, 2 y 4 |
| [`trainer-counts.json`](trainer-counts.json) | Trabajo contado del tramo de ajuste de M1 en la fixture técnica para K = 1, 2 y 4 |
| [`parity.json`](parity.json) | Igualdad de bits del tramo de ajuste de M1 y M3 con K = 1, 2 y 4 y las dos selecciones |
| [`mutations.json`](mutations.json) | Mutaciones dirigidas y la prueba que detecta cada una |
| [`k_shapes.py`](k_shapes.py), [`k_counts.py`](k_counts.py), [`k_parity_capture.py`](k_parity_capture.py), [`mutations.py`](mutations.py) | Scripts que generan cada recibo |

Ningún recibo ejecuta pasos de optimizador. El tramo de ajuste usa `RecordingOptimizer` de las pruebas, que solo copia los gradientes, con el gancho que prohíbe cualquier paso de `torch.optim` y la comprobación final de que ningún parámetro ha cambiado. Las formas de la campaña solo hacen forward y, en el modo con gradiente, el backward de la suma del estado. La protección local de aprendizaje sigue activa.

## Qué añade cada refinamiento

En el tramo de ajuste de M1 sobre la fixture (88 predicciones, 11 llamadas al optimizador sustituto, `cuda:0` y FP32 estricto), los lanzamientos de núcleos pasan de 24.888 con K = 1 a 33.960 con K = 2 y 52.113 con K = 4. Cada refinamiento más añade unos 9.075 lanzamientos, unos 82 por cada una de las 110 llamadas del lector, y 252 copias. Las sincronizaciones no cambian (344 a 346), porque desde #549 las comprobaciones de todos los refinamientos se resuelven con la misma espera. Con K = 4, los tres refinamientos extra suponen el 52 % de los lanzamientos del tramo. El crecimiento de llamadas de Python se concentra en `_read`, `refine` y sus comprobaciones.

Con las formas de la campaña (512 flujos, `hidden_size` 64, 8 vecinos y 1.024 episodios), un refinamiento sin gradiente añade unos 50 lanzamientos. Entre ellos hay seis reducciones de finitud (consulta, valores seleccionados, valores proyectados, lectura, actualización y estado refinado) y cinco núcleos de la comprobación de relleno cero. El resto es el cálculo del paso: proyección y normalización de la consulta, puntuación en FP64 contra todas las claves, ordenación estable, gather de los vecinos, softmax, proyección de valores, suma ponderada y actualización. La lista completa está en `base_kernels_added_by_second_refinement` de `shapes.json`.

## Qué cambia

Dentro de una llamada a `forward`, la instantánea, el presupuesto de trabajo y la puerta de refinamiento no cambian entre pasos, y cada lectura sale de `_read` con la forma, el tipo y el dispositivo del estado, comprobada finita y con ceros en las filas ausentes. Por eso:

- `_read` valida el estado, el presupuesto y la instantánea solo en el primer paso (`_validate_read`). En los siguientes, el estado es el resultado del refinamiento anterior, que conserva forma, tipo y dispositivo y ya se comprobó finito.
- La puerta se comprueba una vez, en el primer paso y en el mismo orden que antes, después de la primera lectura.
- `forward` llama a `_step`, el cálculo del refinamiento, sin repetir la validación de la lectura ni la comprobación de relleno cero, que se cumplen por construcción.

`read` y `refine` públicos siguen validándolo todo, incluido el relleno cero. El cálculo no cambia, así que los bits tampoco.

## Igualdad de bits

- **Formas de la campaña.** Los estados, las cuatro partes de cada lectura (valores, pesos, IDs y presencia) y, con gradiente, los gradientes de todos los parámetros coinciden con `torch.equal` en las 12 llamadas (K = 1, 2 y 4, `per_step` y `first_read`, con y sin gradiente).
- **Tramo de ajuste.** `k_parity_capture.py` recorre M1 y M3 con K = 1, 2 y 4 y las dos selecciones (10 configuraciones) en CPU y `cuda:0`, con FP64 y FP32 estricto. En los 40 pares coinciden la auditoría completa (predicciones, etiquetas, admisiones y recibos de M3), los gradientes de cada llamada, los estados de los flujos y el banco episódico al cerrar el recorrido, las métricas y la identidad sin el hash de los módulos. Solo cambia el SHA-256 de `episodic_readout`.
- **Prueba nueva.** `test_forward_matches_the_public_read_and_refine_bit_for_bit` exige que `forward` dé los mismos bits que componer `_read` y la `refine` pública, en FP32 y FP64, con K = 1, 2 y 4, las dos selecciones y con y sin gradiente.

## Trabajo retirado

| Llamada con formas de la campaña | Lanzamientos antes | Después |
| --- | --- | --- |
| K = 1, sin gradiente | 58 | 52 |
| K = 1, con gradiente | 130 | 124 |
| K = 4 `per_step`, sin gradiente | 206 | 184 |
| K = 4 `per_step`, con gradiente | 512 | 490 |
| K = 4 `first_read`, sin gradiente | 194 | 172 |
| K = 4 `first_read`, con gradiente | 500 | 478 |

En el tramo de ajuste de la fixture con K = 4, los lanzamientos bajan de 52.113 a 49.867 (−4,3 %) y las llamadas a funciones de `mars_titan` de 95.716 a 82.688 (−13,6 %). Con K = 1 bajan de 24.888 a 24.310. Las sincronizaciones y las copias no cambian. Las seis reducciones de finitud de cada paso se mantienen, porque cada una vigila un resultado nuevo. Por ejemplo, `tanh` convierte una actualización infinita en un valor finito, así que no basta con comprobar el estado final.

Lo que queda por paso es el cálculo propio del refinamiento. Dos candidatos más se descartaron sin implementarlos:

- Calcular `sigmoid` de la puerta una vez por llamada cambiaría los bits del gradiente de la puerta, porque autograd sumaría antes las contribuciones de los pasos.
- Precalcular la parte de `base` en la capa de refinamiento cambiaría el orden de la suma.

## Mutaciones

`mutations.py` cambia un único fragmento de `episodic_readout.py`, ejecuta sus pruebas en CPU con `-x` y sin orden aleatorio y restaura el archivo. Las siete hacen fallar una prueba. Saltar la validación también en el primer paso hace fallar además `test_reader_rejects_context_and_input_mismatches` con el contexto, el corte o el codec cambiados.

| Mutación | Qué rompe | Prueba que falla |
| --- | --- | --- |
| `first_step_without_validation` | el primer paso tampoco valida | `test_refinements_validate_once_and_skip_the_checks_their_reads_satisfy` |
| `every_step_validates` | cada paso vuelve a validar | la misma |
| `public_read_without_validation` | `read` público sin validación | `test_public_read_cannot_bypass_the_working_budget` |
| `forward_without_gate_check` | la puerta no se comprueba | `test_nonfinite_gate_is_rejected_even_if_sigmoid_saturates` |
| `gate_checked_every_step` | la puerta se comprueba en cada paso | `test_refinements_validate_once_and_skip_the_checks_their_reads_satisfy` |
| `forward_through_public_refine` | `forward` vuelve a pasar por `refine` | la misma |
| `public_refine_without_zero_fill` | `refine` público sin relleno cero | `test_refinement_rejects_nonzero_absent_read_values` |

## Medidas de tiempo pendientes

El perfil de punta a punta con py-spy y Nsight Systems y la medida del recorrido completo antes y después siguen pendientes, por los mismos motivos que [#549](../titans-event-path-20261013/README.md#medidas-de-tiempo-pendientes). La GPU tenía la carga ajena de un `llama-server` de otro proyecto (unos 5,4 GB y un 84 % de uso a las 14:36, y 5.816 MiB ocupados en total a las 19:31). Además, los eventos guardados de `US+CN/fold-012` de #363 ya no se pueden abrir desde que la v3.1 sustituyó las muestras de la v3. El issue pide también perfilar con el núcleo del codec de [#520](https://github.com/GonxKZ/mars-titan/issues/520), que sigue abierto en [#578](https://github.com/GonxKZ/mars-titan/pull/578). Las cuentas de este informe muestran qué trabajo sale de cada paso, pero no cuánto tiempo ahorra en la campaña.

Órdenes para repetirlas desde la raíz de cada árbol (la rama de #549 y esta), con `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src`, los eventos de una vista v3.1 (`benchmarks/chronological_events.py VISTA/manifest.json TRABAJO eventos.pkl 100 6`) y un trabajo `memslot gpu` por ronda, alternando los árboles:

- `python benchmarks/chronological_replay.py VISTA/manifest.json TRABAJO --family readout --arm mars_titan_m1_k4 --option precision='"fp32_strict"' --option block_rows=512 --predictor max_batch=1024 --predictor max_state_bytes=134217728 --strict --warmup 2 --segments 8 --events eventos.pkl --capture CAPTURA.pt --output resultados.jsonl --label ETIQUETA`, y lo mismo con `--arm mars_titan_m1` como referencia de K = 1.
- `python reports/engineering/profiling-20261010/compare_capture.py DIRECTORIO salida.json antes:despues` para los bits de las rondas.
- `py-spy record --rate 100 --format raw` y `MARS_TITAN_NVTX=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none` sobre la misma orden, resumidos con `benchmarks/collapsed_profile_summary.py` (raíz `_train_pass@training/mars_titan_run.py`) y `benchmarks/nsys_range_summary.py`.

## Reproducción

- Formas de la campaña: `python k_shapes.py ÁRBOL salida.pt` en cada árbol con `memslot gpu`.
- Tramo de ajuste: `python k_counts.py ÁRBOL salida.json cuda:0 TRABAJO` y `python k_parity_capture.py ÁRBOL salida.pt DISPOSITIVO DTYPE TRABAJO`, con el mismo `TRABAJO` en los dos árboles, comparadas con `parity_compare.py` del informe de #549.
- Mutaciones: `python mutations.py ÁRBOL salida.json`.
- Pruebas: `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src MARS_TITAN_EPISODIC_NATIVE=ENLACE pytest -p no:randomly tests/models/titans tests/training/test_mars_titan_run.py tests/training/test_financial_run.py tests/test_learning_hold_lowering.py tests/training/test_learning_hold_inventory.py`. Con `cuda:0` visible pasaron 796 pruebas en 7 min 10 s, con 2,1 GiB de memoria residente máxima.

Las órdenes del tramo de ajuste necesitan el enlace episódico en `MARS_TITAN_EPISODIC_NATIVE`. Todo se ejecutó en una RTX 4070 Laptop con torch 2.14.0 y CUDA 13.0.
