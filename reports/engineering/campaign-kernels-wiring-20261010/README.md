# Opciones de la campaña hasta el entrenador, medoids y CUDA Graphs en Titans-MAC (10 de octubre de 2026)

Este informe cierra lo que dejó pendiente el [informe de núcleos](../campaign-kernels-20261009/README.md#pendiente-e-impedimentos) de [#482](https://github.com/GonxKZ/mars-titan/pull/482). La rama `perf/campaign-kernels-wiring` (#363 y #446) hace que la campaña A v2 declare y entregue al proceso que entrena la precisión, el paso con CUDA Graphs, el lote y las opciones de memoria. También acelera la selección de medoids de los lectores sin cambiar sus bits y mide CUDA Graphs en la emisión de Titans-MAC.

Ninguna medida ni prueba aplica pasos de optimizador. Las medidas del lector recorren el bucle real con un optimizador que solo guarda gradientes, y las pruebas de la campaña detienen cada entrenador antes de su primer lote de ajuste. La protección de aprendizaje sigue activa. Fijar las opciones de memoria en la configuración no la levanta.

## Entorno

| Elemento | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8.188 MiB, perfil `power-saver` sin cambios |
| CPU | AMD Ryzen 9 8945HS, dos hilos por trabajo |
| Software | Python 3.12.14, PyTorch 2.14.0+cu130, NumPy 2.5.3, glibc 2.43 |
| Datos | Eventos guardados el 10 de octubre de la vista US+CN `fold-012` de la edición v3 (254 eventos desde el 127, 8 tramos medidos y 151.797 filas) |

Las medidas compartieron la máquina con suites y perfiles de otras ramas, con una carga media entre 5 y 18. Los datos de cada medida registran su carga.

## Recorrido hasta el entrenador

La sección `neural` de la [campaña A v2](../../../configs/baselines/historical-masked-campaign-a-v2.json) admite ahora dos campos más, `precision: "fp32_strict"` y `cuda_graphs: true`, y un lote de hasta 4.096 (`MAX_BATCH_SIZE` de `reference_design.py`, el mismo límite que el ajuste y la búsqueda). El lote declarado sigue en 256.

| Opción | Validación en el plan | Llega al entrenador por | Identidad |
|---|---|---|---|
| `precision` | Texto de `PRECISIONS` | El caso de cada búsqueda y finalista. `run_reference_case` aplica la política antes de construir el modelo | `case` del trabajo y del ajuste, y `kernel_policy` del ajuste |
| `cuda_graphs` | Booleano, y verdadero solo si todas las familias de la sección tienen paridad comprobada. Falso equivale a no declararlo | El caso. El ajuste usa `ReferenceStepGraph` con el lote declarado | `case` y la huella de `reference_step_graph.py` en el código del ajuste |
| `batch_size` | Entero de 1 a 4.096 | `JobRun.batch_size`, también en una ranura con su propio proceso | Huella de la campaña en cada trabajo y `batch_size` del ajuste |
| `memory_options` | Igual a la receta en valor y tipo, o `pending`. Un valor fijado exige un recibo con una medida de esa familia y esos valores | La receta que nombra el caso con su huella. El ejecutor la vuelve a comprobar y su entrenador la lee | Huella de la receta en el caso y receta completa en la identidad del entrenador |

El recibo de memoria es [vram-throughput.json](../campaign-kernels-20261009/vram-throughput.json) de #482. Titans-MAC y los núcleos de CM-v1 fijan `accumulation_rows` 1.024 y la GRU candidata 128 sin `recompute`. El plan exige que el recibo contenga una entrada medida del modelo de cada familia (`titans_mac`, `cm_v1_core` y `episodic_gru`) con esos valores. Esa medida usó la edición v3 y hay que repetirla sobre la vista v3.1 (ver «Dependencias»).

`tests/training/test_campaign_launch_options.py` lanza la ventana `fold-006` de una copia reducida de la campaña v2 con los brazos `gru`, `ridge`, `titans_mac_online`, `gru_episodic` y los cuatro de CM-v1 con sus núcleos, sobre el corpus técnico de las pruebas. Los ejecutores son los reales de `masked_campaign.EXECUTORS` hasta la entrada de su entrenador (`tests/training/launch_doubles.py`). La referencia neuronal corre en CPU hasta escribir su primer informe, con la identidad ya fijada. Titans-MAC y los núcleos de CM-v1 se detienen al pasar la receta a `ChronologicalTrainer`, y la GRU candidata al llamar a `fit_window`. Después un doble escribe filas nulas para que la campaña confirme cada recibo. Los lectores de CM-v1 terminan con el doble porque necesitan un núcleo ajustado de verdad, y su receta no tiene opciones de memoria. La copia reducida no incluye los lectores de MARS-TITAN, que necesitan un padre Titans-MAC ajustado. No reciben opciones de la sección `neural` ni de `memory_options`, y su FP32 estricto es el de `numerics`, que `masked_campaign` aplica antes de cada trabajo y exige en su informe, igual que en las demás familias.

| Prueba | Qué comprueba |
|---|---|
| Ventana en dos ranuras | Cada trabajo con entrenador de la ventana, de las cuatro familias, llega una vez a su entrenador, en un proceso distinto del de la campaña. Las referencias reciben el caso del recibo con `fp32_strict` y `cuda_graphs`, el lote 256, la política estricta y la huella del paso con grafo. Titans-MAC y los núcleos de CM-v1 reciben `accumulation_rows` 1.024 y la GRU candidata 128 sin `recompute`, todos con `fp32_strict`. Todos trabajan sin TF32 |
| Valores no declarados | TF32, `cuda_graphs` como texto, un lote de 4.097, una opción desconocida, filas distintas de la receta, 1.024,0 en vez de 1.024, `recompute` como 0, valores fijados sin recibo y una opción pendiente se rechazan antes de leer vistas o crear la salida |
| Identidad | Cada opción por separado cambia la huella de la campaña, y solo cambian los casos de la familia que la recibe |
| Corte y reanudación | Se corta la campaña cuando una búsqueda neuronal de US ya ha llegado a su entrenador. Con cualquiera de las seis opciones cambiada, la salida cortada se rechaza sin lanzar nada. Con la misma configuración, el trabajo cortado vuelve a su entrenador con lo mismo, el resto llega como en la ejecución sin corte y los recibos tienen la misma identidad |
| Opciones cambiadas | Con lote 512, sin grafo, filas 2.048 en Titans-MAC y CM-v1 y 256 con `recompute` en la GRU candidata, cada entrenador recibe el valor nuevo y todos los recibos cambian de identidad |

`test_reference_run_precision.py` comprueba además que el ajuste de una referencia solo se reanuda con el lote, la precisión y el paso con grafo con que empezó. Los entrenadores con receta guardan la receta completa en su identidad y su huella en la petición de la ventana, y ya rechazaban reanudar con otra (`test_candidate_run.py` y `test_titans_walk_forward.py`). `test_campaign_a_joint.py` rechaza un recibo sin medida de la familia, con otro valor, con otro tipo o sin `measured`.

Cuatro mutaciones dirigidas hacen fallar estas pruebas: entregar siempre el lote 256 al ajuste, no añadir `cuda_graphs` al caso, comparar las opciones de memoria sin su tipo y leer el lote de la sección tabular.

## Precisión del padre en el posentrenamiento

El padre congelado de #446 aplica ahora la precisión que declaró el caso de su ajuste antes de reconstruirse, y se rechaza si la política resultante no es la que registró ese ajuste. El adaptador hereda esa precisión, la fija antes de registrar su identidad y la vuelve a exigir al guardar cada checkpoint y en la evaluación final. Las pruebas de `tests/posttraining` cubren el padre y su adaptador con FP32 estricto y cuDNN sin TF32, un proceso que activa TF32 a mitad del ajuste y una evaluación final con TF32, que no se confirma.

## Medoids

Los lectores de MARS-TITAN y de CM-v1 pasaban la mayor parte de su tiempo en `cm/medoids.py`. Se midieron tres opciones que conservan los bits sobre el recorrido real de M1 y BCM, con dos repeticiones intercaladas ([datos](reader-medoids.json)):

- dos hilos que reparten los bloques de clientes,
- `reuse_distances`, que calcula una sola vez la distancia de cada bloque de clientes a un candidato variable,
- `bounded_background`, que calcula el fondo de los centros fijos solo en los pares que una cota inferior no excluye.

La deducción de la cota y sus límites están en [retención con centros fijos](../../../docs/experiments/mars_titan_cm_v1/anchored_retention.md#opciones-que-conservan-los-bits).

| Lector | Opciones | Filas por segundo (a, b) | Segundos de medoids (a, b) |
|---|---|---|---|
| M1 | Referencia | 1.249, 1.435 | 125,1, 117,4 |
| M1 | Dos hilos | 2.124, 2.178 | 64,0, 61,5 |
| M1 | Cota | 5.279, 5.317 | 3,8, 3,9 |
| BCM | Referencia | 1.028, 1.128 | 167,2, 148,3 |
| BCM | Dos hilos | 1.477, 1.520 | 101,6, 97,1 |
| BCM | Reutilización | 1.183, 1.317 | 137,3, 121,2 |
| BCM | Dos hilos y reutilización | 1.678, 2.090 | 80,7, 65,5 |
| BCM | Cota | 2.204, 2.781 | 48,3, 37,6 |
| BCM | Cota y reutilización | 4.414, 4.418 | 8,7, 8,2 |
| BCM | Las tres | 4.536, 4.337 | 8,6, 9,0 |

M1 no tiene candidatos variables, así que solo le afecta el fondo. Los dos hilos no añaden una mejora reproducible sobre la cota y la reutilización (4.536 y 4.337 frente a 4.414 y 4.418), así que se descartaron y su código se retiró. Las 86 selecciones, los gradientes de las 11 llamadas al optimizador y las 1.265 predicciones son idénticos bit a bit en todas las ejecuciones de cada lector.

El banco pide ahora las dos opciones conservadas. Con el código final, M1 da 5.240 filas por segundo (carga 12) y BCM 4.009 (carga 15), frente a 1.249 y 1.155 de sus referencias. Las selecciones, los gradientes y las predicciones vuelven a coincidir bit a bit con esas referencias. Los medoids pasan de unos 150 s a entre 4 y 9 s por recorrido. Si la memoria de las opciones no cabe en el presupuesto de la propuesta, la selección usa la ruta de referencia, y solo cambia `estimated_peak_bytes` del recibo del banco.

[`benchmarks/anchored_medoid_options.py`](../../../benchmarks/anchored_medoid_options.py) repite la selección con la forma de esas llamadas (5.064 episodios, 1.016 fijos y 16 candidatos en 64 coordenadas) sobre dos geometrías sintéticas, con cinco repeticiones alternadas. Con el código final y la máquina muy cargada (carga 18), la mediana baja de 4,08 a 0,215 s en la isótropa (19,0 veces) y de 3,54 a 0,217 s en la de grupos (16,3 veces), con los mismos campos y bits, y la memoria estimada pasa de 10,0 a 18,5 MB ([datos](anchored-medoid-options.json)). La poda depende de la geometría, así que estas cifras no sustituyen a las del recorrido real.

### Memoria de la selección

[`benchmarks/anchored_medoid_memory.py`](../../../benchmarks/anchored_medoid_memory.py) registra con memray todas las reservas de una llamada, también las nativas, y separa por la pila nativa las que hace OpenBLAS ([datos](anchored-medoid-memory.json)):

| Ruta | Estimada | Pico propio en memray | OpenBLAS en el pico |
|---|---|---|---|
| Referencia | 9,98 MB | 6,99 MB | 0 |
| Cota y reutilización | 18,50 MB | 13,98 MB | 0,52 MB |

Los 0,52 MB son la tabla de tareas que OpenBLAS reserva mientras reparte un producto entre sus dos hilos. Aparte, el primer producto de matrices del proceso deja reservada el área de trabajo de OpenBLAS 0.3.34: 32,3 MiB de memoria virtual y entre 0,86 y 0,93 MB residentes, medidos en seis intérpretes nuevos con uno y dos hilos. No pertenece a la selección y no entra en `estimated_peak_bytes`, como se documenta en el contrato.

La primera versión de la cota tenía tres fallos que esta revisión encontró y corrigió:

- El factor final fijo (1 − 2048u) solo cubría el error de la cadena de `hypot` hasta 255 dimensiones. Ahora es (1 − (8D + 2048)u), y una prueba con 300 dimensiones compara los bits con la referencia.
- La estimación no contaba los temporales de NumPy (los buffers del iterador de una ufunc con operandos difundidos y la copia que hace `argmin` de un bloque parcial) ni los índices de los pares supervivientes. Medida aparte, la parte acotada superaba su estimación entre un 3 y un 15 % en cuatro de las cinco formas probadas, aunque la llamada completa quedaba dentro por el margen de la referencia. Ahora los buffers se reservan una vez, los índices son contiguos y la estimación los cuenta. La prueba de memoria con `tracemalloc` cubre cinco formas, entre ellas una mínima, una con un solo centro fijo, una sin poda y una de 300 dimensiones.
- La medida con memray de la primera versión no separaba el área de trabajo de OpenBLAS (47,5 MB de pico frente a 24,3 MB estimados). Ahora se mide aparte y queda documentada.

La cota usa el producto de matrices de OpenBLAS con los hilos que fije el proceso. Sin límite de hilos (16 por defecto) y con carga 11, una medida exploratoria de los productos de un bloque de 256 clientes por 1.016 fijos tardó 204 ms frente a 0,83 ms con un hilo. La campaña fija `OMP_NUM_THREADS=2` en cada ranura, y OpenBLAS respeta esa variable.

## CUDA Graphs en Titans-MAC

En `mac_online` con la receta de la campaña, la emisión (`_observe`) y la repetición con backward (`_replay`, dentro de `_update`) se reparten el tiempo del tramo casi a partes iguales (7,6 y 8,7 s acumulados con sincronización, 13.346 filas por segundo, [datos](titans-graphs.json)). La repetición agrupa los flujos con formas que cambian en cada instante, y rellenarlas cambiaría las reducciones, así que solo la emisión de bloques completos admite un grafo.

[`benchmarks/titans_emission_graph.py`](../../../benchmarks/titans_emission_graph.py) captura la emisión de un bloque de 1.024 flujos reales con `differentiable=True`. Para poder capturarla, el codificador de precios difiere sus dos comprobaciones de finitud en lugar de leerlas en su línea. Las tres rutas dan los mismos bits que `prepare` en predicción, cuantiles, token, salida y estado siguiente de la memoria ([datos](titans-emission-graph.json)):

Medianas por bloque de 20 repeticiones tras tres de calentamiento, en ms:

| Medida | Carga | `prepare`, la emisión del ajuste | Mismo cálculo con comprobaciones diferidas | Ese cálculo con un CUDA Graph | Grafo frente a diferido |
|---|---|---|---|---|---|
| Primera, sin JSON guardado | Sin registrar | | 8,46 | 7,87 | 1,07 |
| Versión anterior del benchmark | Sin registrar | 11,20 | 7,69 | 7,84 | 0,98 |
| Antes de los perfiles | Sin registrar | 11,44 | 9,58 | 7,88 | 1,22 |
| a | 29,7 | 12,47 | 9,23 | 7,92 | 1,17 |
| b | 29,1 | 13,26 | 8,65 | 7,91 | 1,09 |
| c | 29,0 | 13,53 | 8,51 | 7,85 | 1,08 |

El grafo tarda siempre entre 7,84 y 7,92 ms, que es casi el tiempo de sus núcleos en la GPU (7,12 ms según nsys, más las copias de entrada y salida). El cálculo eager diferido varía entre 7,69 y 9,58 ms según la carga del host, así que la ventaja del grafo va de 0,98 a 1,22 veces, con mediana 1,085. Esa ventaja solo alcanza a la emisión de bloques completos, que ocupa la mitad del tramo, así que en el recorrido completo quedaría entre un −1 y un 9 % a cambio de buffers fijos para el estado y un grafo por forma. **CUDA Graphs no se adopta en esta rama** y la rama no contiene código de captura fuera del benchmark. Decidirlo con datos del bucle completo, con grafos por forma exacta en una caché acotada y su tasa de acierto, es la siguiente tarea sobre el camino por evento de Titans-MAC.

La diferencia entre `prepare` y el cálculo diferido (entre 1,19 y 1,59 veces) está en el host. Diferir las comprobaciones no necesita grafo, pero cambia el momento en que se detecta un valor no finito, así que también se decide en la tarea siguiente. Un perfil con cProfile de 20 llamadas atribuye por bloque unos 3,5 ms a esperar en tres sincronizaciones (una es la comprobación inmediata del codificador de precios, que impide adelantar en el host el resto de la emisión), 1,6 ms a lanzar las 32 comprobaciones de finitud, 1,1 ms a validar con expresiones regulares los 1.024 identificadores de flujo (`_check_flows`), 1,1 ms a `NeuralMemory.validate_state` y 0,7 ms a serializar metadatos en JSON (`_state_size` y `_config_id`). Los perfiles de py-spy y nsys de la sección siguiente lo confirman. Es el siguiente cuello de botella medido de la emisión de Titans-MAC, MARS-TITAN y CM-v1. Reducirlo sin cambiar las comprobaciones ni sus mensajes queda para otra rama.

## Perfiles de CPU y memoria

Los perfiles usan py-spy 0.4.2, memray 1.20.0 y Nsight Systems 2026.3.2, sin contadores de hardware. Los resúmenes están en [`profiles/`](profiles/), junto a las pilas plegadas de py-spy comprimidas, y se obtienen con [`collapsed_profile_summary.py`](../../../benchmarks/collapsed_profile_summary.py), [`memray_peak_summary.py`](../../../benchmarks/memray_peak_summary.py) y [`nsys_range_summary.py`](../../../benchmarks/nsys_range_summary.py). Las filas por segundo de una ejecución con perfilador no se usan como caudal. La sobrecarga del muestreo no se midió de forma emparejada en esta rama.

### Lectores

py-spy muestreó 50 veces por segundo los marcos de Python (sin `--native`) del recorrido real de M1 y BCM con el código final y de BCM con la ruta de referencia. La tabla da la parte de las muestras de `_train_pass` que pasa por cada función:

| Perfil | Muestras | Selección de medoids | `_admit` | `_observe` | `encode` del codec | `_update` |
|---|---|---|---|---|---|---|
| BCM, referencia | 12.128 | 80,0 % | 83,0 % | 16,0 % | 9,8 % | 0,8 % |
| BCM, código final | 2.771 | 17,8 % | 29,6 % | 65,9 % | 41,8 % | 3,7 % |
| M1, código final | 3.103 | 9,5 % | 18,6 % | 77,3 % | 41,2 % | 2,8 % |

Con la selección acelerada, el marco propio con más muestras pasa a ser `einsum` dentro de `FrozenEpisodeCodec.encode` (`memory/episodic_codec.py`): un 37,6 % en BCM y un 35,5 % en M1. Es la proyección congelada de cada bloque de entrada, en FP64 y con `einsum(..., optimize=False)`, que no usa la BLAS. Pasarla a un producto de matrices cambiaría el orden de las sumas y los bits de las claves, así que exige una identidad nueva del codec y una tolerancia frente a FP64. Es el siguiente cuello de botella medido de los lectores y no se toca en esta rama.

memray registró los dos lectores completos en modo agregado. Su pico (12,3 GiB) supera los 3,8 GB residentes que midió el propio recorrido, porque memray cuenta el tamaño pedido y no las páginas tocadas. En estos procesos la mayor parte son reservas del controlador y del runtime de CUDA: 4,1 GiB de `malloc` al iniciar CUDA desde `require_cuda` y las proyecciones de memoria de la GPU en el espacio de direcciones (`mmap`). Los resúmenes separan las reservas por función (`by_allocator`), pero no atribuyen la memoria residente del host. Esa atribución queda pendiente.

### Emisión de Titans-MAC

Nsight Systems trazó la API de CUDA, los núcleos y las copias de `benchmarks/titans_emission_graph.py` con 100 repeticiones por ruta, cada una en su rango NVTX. Por bloque de 1.024 flujos:

| Ruta | Tiempo con nsys | Núcleos | Tiempo de núcleos en GPU | GPU ocupada | Lanzamientos (tiempo de API) | Copias a host |
|---|---|---|---|---|---|---|
| `prepare` | 14,79 ms | 398 | 7,30 ms | 49 % | 397 (2,44 ms) | 2 |
| Comprobaciones diferidas | 10,41 ms | 351 | 6,91 ms | 66 % | 350 (1,94 ms) | 1 |
| CUDA Graph | 8,08 ms | 11 fuera del grafo | 7,12 ms en el grafo | 90 % | 1 grafo y 11 núcleos | 1 |

nsys traza el grafo como una sola ejecución. Las 14 copias entre posiciones de la GPU del grafo (74,7 MB) son las entradas y salidas en buffers fijos, y forman parte del coste de esa ruta.

py-spy muestreó 100 veces por segundo, contando las esperas, la misma emisión con 400 repeticiones. Dentro de `prepare`, `check_finite` reúne el 21,1 % de las muestras al lanzar las comprobaciones de finitud, y `resolve` (8,1 %) y `first_flagged` (5,0 %) esperan su resultado en una sincronización. El codificador JSON de los metadatos suma un 6,3 % y `NeuralMemory.validate_state` un 9,4 % inclusivo. Junto con la tabla, confirma que `prepare` está limitado por el host: la GPU pasa la mitad del bloque esperando lanzamientos y sincronizaciones.

## Pruebas en subprocesos y huellas con hilos

`tests/training/test_input_pipeline.py` y `test_information_inputs.py` lanzan subprocesos que importan el paquete. Ahora reciben el árbol `src` en `PYTHONPATH` mediante el accesorio `source_environment` de `tests/conftest.py`, así que no dependen de una instalación editable. Las huellas bit a bit de `test_paper_projections.py` (#474) y `test_memory_stability.py` (#477) dependen del número de hilos de CPU, y sus módulos fijan dos hilos durante cada prueba y restauran después el valor del proceso.

## Dependencias

- **Vista v3.1 ([#428](https://github.com/GonxKZ/mars-titan/issues/428), [edición v3.1](../../../docs/data/edition-v3-1.md)).** La ejecución de la v3.1 sustituye activo a activo las muestras de la vista v3, así que las medidas del lector usaron eventos guardados el mismo día con tres parches que solo existen para medir: no se comprueban los archivos de muestras, los índices de observaciones conservan la huella de código con que se generaron y las anchuras de entrada salen de los eventos. Cuando existan las vistas v3.1 hay que repetir sin parches la medida de memoria que fija `memory_options` en `configs/baselines/historical-masked-campaign-a-v2.json` (recibo `reports/engineering/campaign-kernels-20261009/vram-throughput.json`) y el caudal de los lectores de este informe. Si cambia algún valor de memoria, cambian la receta, el recibo y la configuración a la vez.
- **Protección de aprendizaje.** Sigue activa la protección local que bloquea los ajustes hasta verificar la edición histórica. Ninguna cifra de este informe dice nada de la calidad predictiva.

## Reproducción

- Pruebas: `OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest tests/training/test_campaign_launch_options.py tests/training/test_campaign_a_joint.py tests/training/test_reference_run_precision.py tests/posttraining tests/cm tests/memory/test_retention_bank.py`. Las pruebas del banco necesitan `MARS_TITAN_EPISODIC_NATIVE`.
- Selección de medoids: `OMP_NUM_THREADS=2 PYTHONPATH=src python benchmarks/anchored_medoid_options.py salida.json` y, para la memoria, `OMP_NUM_THREADS=2 PYTHONPATH=src python benchmarks/anchored_medoid_memory.py carpeta`.
- Lectores: `benchmarks/chronological_replay.py` con `--family readout --arm mars_titan_m1` o `--family cm_readout --arm cm_v1_bcm`, `--option precision='"fp32_strict"' --option block_rows=512 --predictor max_batch=1024 --predictor max_state_bytes=134217728 --strict --warmup 2 --segments 8` y los eventos de `benchmarks/chronological_events.py`, como en el [informe de núcleos](../campaign-kernels-20261009/README.md#reproducción). Las opciones de cada fila se forzaron sustituyendo `select_anchored_medoids` en `memory.retention_bank` por una envoltura que fija sus argumentos y registra cada llamada.
- Emisión de Titans-MAC: `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src python benchmarks/titans_emission_graph.py eventos.pkl VISTA/manifest.json salida.json 1024`, con un quinto argumento para el número de repeticiones de los perfiles (400 con py-spy y 100 con nsys).
- Perfiles: `py-spy record --rate 50 --format raw` sobre el lector y `py-spy record --idle --rate 100 --format raw` sobre la emisión, `python -m memray run --aggregate` sobre el lector, y `nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none` seguido de `nsys export --type sqlite` sobre la emisión. Los resúmenes salen de `benchmarks/collapsed_profile_summary.py` con la raíz `_train_pass@training/mars_titan_run.py` (lectores) o `prepared@titans_emission_graph.py` (emisión) y las funciones con su archivo, como `encode@memory/episodic_codec.py`, `benchmarks/memray_peak_summary.py` y `benchmarks/nsys_range_summary.py --repeats 100 prepare deferred graph`.
