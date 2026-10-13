# Proyección nativa del codec episódico y perfil de B+M en CM-v1 (10 de octubre de 2026)

Este informe recoge el trabajo de [#520](https://github.com/GonxKZ/mars-titan/issues/520). Tras la mejora de los medoids de [#363](https://github.com/GonxKZ/mars-titan/issues/363), el primer coste de los lectores con banco episódico era la proyección FP64 de `FrozenEpisodeCodec.encode` (`src/mars_titan/memory/episodic_codec.py`). La rama `perf/cm-v1-codec-projection` la calcula con un núcleo C++20 que da los mismos bits que `numpy.einsum`, deja de repetir la huella de la matriz en cada llamada y mide el recorrido completo de los lectores de CM-v1 (B+M y B+C+M) y de M1 de MARS-TITAN.

Ninguna medida ni prueba aplica pasos de optimizador. Los lectores recorren el bucle real con el optimizador de `chronological_replay.py`, que solo guarda gradientes. La protección de aprendizaje sigue activa.

## Entorno

| Elemento | Valor |
|---|---|
| CPU | AMD Ryzen 9 8945HS (AVX2 y AVX-512F), dos hilos por proceso |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8.188 MiB, perfil `power-saver` sin cambios |
| Software | Python 3.12.14, NumPy 2.5.3 (línea base `X86_V2`), PyTorch 2.14.0+cu130, glibc 2.43 |
| Compiladores | Clang 21.1.8 en Release con `-O3` y `-ffp-contract=off`, sin `-march`. GCC 15.2.0 para las comprobaciones |
| Datos | Eventos guardados de la vista US+CN `fold-012` de la edición v3, los mismos de #363 (254 eventos desde el 127, 8 tramos medidos y 151.797 filas) |

La máquina estuvo compartida con suites y perfiles de otras ramas durante todas las medidas, con una carga media de entre 11 y 33 en 16 hilos. Cada archivo de datos registra su carga.

## Núcleo de proyección

`einsum("bi,oi->bo", optimize=False)` sobre float64 no usa BLAS en NumPy 2.5.3. Su núcleo de la línea base x86-64 tiene dos acumuladores por salida, recorre las entradas en bloques de ocho, del par 6-7 al 0-1, y después el resto por pares, y al final suma los dos acumuladores. Multiplica y suma por separado. Un BLAS o una reducción vectorial cambiarían ese orden y, con él, los bits de las claves del banco.

`native/src/episodic_projection.cpp` repite exactamente esas operaciones para cada salida y vectoriza sobre las salidas, ocho por vector. Cada carril hace las operaciones de una salida, así que el ancho de la instrucción no cambia el resultado. Los mosaicos de cuatro, dos y una fila mantienen ocho cadenas de sumas independientes. La compilación usa `-ffp-contract=off`, que impide fusionar multiplicación y suma, y el código objeto no contiene instrucciones FMA. En x86-64 hay versiones portable, AVX2 y AVX-512 de la misma función. `project_block` usa la más ancha que admite el procesador y el enlace la publica en `codec_projection_isa`. No se usan clones con `ifunc`, porque su resolutor se ejecuta antes de iniciar ThreadSanitizer y lo hace fallar con GCC y con Clang.

`FrozenEpisodeCodec.encode(batch, native=...)` solo usa el núcleo si el enlace lo incluye y si el einsum instalado da los mismos bits que `_declared_order`, la misma suma escrita con operaciones de NumPy, sobre unos datos fijos que se comprueban una vez por proceso. Con otro NumPy, por ejemplo uno que fusione multiplicación y suma, sigue con einsum. Como el resultado no cambia, la identidad del codec tampoco. El lector de MARS-TITAN (`training/mars_titan_run.py`) y la sesión episódica (`memory/episodic_session.py`) entregan su enlace al codec.

El codec guarda además la proyección transpuesta, que el núcleo recorre por filas. La estimación de memoria de cada llamada cuenta las dos matrices. La huella SHA-256 de la proyección se calcula una vez al construir el codec y queda en su identidad. En cada llamada se exige la misma memoria de solo lectura de las dos matrices, que son vistas de objetos `bytes`.

### Comprobaciones

| Comprobación | Resultado |
|---|---|
| `episodic_projection_tests` contra una referencia escalar que también hace las dos sumas de cero que omite el núcleo | Mismos bits en las tres versiones con 10 números de filas, 13 anchuras y 11 números de salidas, con valores normales y con ceros con signo, subnormales y magnitudes entre 2^-300 y 2^300. Cada fila da lo mismo sola que dentro de un mosaico |
| `tests/memory/test_episodic_codec.py` | El einsum instalado sigue el orden declarado, el núcleo coincide con `_declared_order` y `encode` da los mismos bits, huella y memoria declarada con el núcleo y con einsum, también con una fila, 17 filas y valores de 10^30 |
| GCC 15.2.0 y Clang 21.1.8 con avisos como errores, y clang-tidy 21.1.6 | Sin avisos |
| ASan, UBSan, TSan (Clang y GCC) y MSan | Sin errores |

Todas las mutaciones dirigidas hacen fallar alguna prueba ([datos](mutations.json)). En el núcleo son 10: pares en orden creciente, un solo acumulador, bloques de cuatro, resto sin el carril impar, suma fusionada, suma final sin el carril impar, carga parcial incompleta, filas superpuestas en la salida, salida corta aceptada y versión por defecto que no es la más ancha. El procesador de las medidas admite las tres versiones, así que el rechazo de una versión no disponible no se ejercita en esta máquina. En Python son 9: las tres partes del orden de `_declared_order`, elegir el núcleo sin comprobar einsum, no vigilar la transpuesta, no contarla en la memoria, ignorar el enlace en `encode` y no entregarlo desde la sesión o desde el lector.

### Medidas aisladas

`mars-titan-projection-benchmark` mide cada versión con las seis formas del codec (127 salidas) en 41 repeticiones alternadas ([datos](projection-isa.json)). AVX-512 fue la más rápida en todas las filas, así que es la versión por defecto en este procesador. Otro procesador necesita su propia medida.

| Filas | Portable (µs) | AVX2 (µs) | AVX-512 (µs) | AVX-512 frente a portable |
|---|---|---|---|---|
| 1 | 80,1 | 52,8 | 49,9 | 1,61 |
| 16 | 934,9 | 428,4 | 300,7 | 3,11 |
| 128 | 6.776 | 3.241 | 2.263 | 2,99 |
| 512 | 27.520 | 13.258 | 9.083 | 3,03 |

[`benchmarks/episodic_codec_projection.py`](../../../benchmarks/episodic_codec_projection.py) compara `encode` y la proyección sola con einsum y con el núcleo, con las anchuras de la campaña y siete repeticiones alternadas ([datos](codec-encode.json)). Las salidas, la huella y la memoria declarada coinciden en las cuatro formas.

| Filas | `encode` con einsum (ms) | `encode` con el núcleo (ms) | Aceleración de `encode` | Aceleración de la proyección |
|---|---|---|---|---|
| 1 | 0,421 | 0,393 | 1,07 | 1,84 |
| 16 | 2,09 | 1,07 | 1,96 | 3,00 |
| 128 | 12,9 | 4,19 | 3,08 | 4,64 |
| 512 | 51,4 | 14,8 | 3,47 | 5,89 |

## Recorrido completo de los lectores

Los tres lectores recorren los eventos guardados con `chronological_replay.py`, precisión `fp32_strict`, bloques de 512 filas y el mismo enlace nativo, en rondas ABBA: einsum, núcleo, núcleo y einsum, cada una en su propio proceso ([datos](readers.json)). Un envoltorio de la medida elige la proyección del codec y suma el tiempo de reloj y el de CPU del hilo de cada `encode`. Las rondas 1 y 2 guardan los gradientes de las 11 llamadas al optimizador, las 1.265 predicciones pendientes y las 86 selecciones de medoids, y coinciden bit a bit en los tres lectores.

| Lector | Filas por segundo con einsum (rondas 1 y 4) | Filas por segundo con el núcleo (rondas 2 y 3) | CPU de `encode` con einsum (s) | CPU de `encode` con el núcleo (s) |
|---|---|---|---|---|
| B+C+M (`cm_v1_bcm`) | 4.711, 3.072 | 6.519, 4.682 | 19,5, 21,4 | 5,5, 5,8 |
| B+M (`cm_v1_bm`) | 4.097, 4.108 | 6.410, 6.023 | 20,7, 21,2 | 5,4, 5,9 |
| M1 (`mars_titan_m1`) | 4.835, 4.974 | 7.495, 5.194 | 20,0, 20,0 | 5,5, 5,9 |

El tiempo de CPU de `encode` en todo el proceso, con los tramos de calentamiento y sus 452 llamadas, baja de unos 20 s a menos de 6 s en los tres lectores, entre 3,5 y 3,7 veces menos, y apenas varía entre rondas. El caudal de los tramos medidos sube por término medio 1,44 veces en B+C+M, 1,52 en B+M y 1,29 en M1. Su dispersión es grande porque la carga media de la máquina pasó de 11 a 21 durante las rondas. El tiempo de CPU de usuario de todo el proceso baja de 95,8 a 73,0 s en B+C+M, de 92,2 a 70,1 s en B+M y de 73,1 a 66,2 s en M1.

Una primera serie de B+C+M y B+M coincidió con la memoria de la máquina agotada, con carga de hasta 116 y el intercambio lleno, y se repitió entera. Sus capturas también coincidían bit a bit.

## Perfiles con el núcleo

Cada lector se perfiló dos veces con el núcleo, en procesos separados: py-spy 0.4.2 en el modo que no detiene el proceso, a 100 muestras por segundo, y Nsight Systems 2026.3.2 con los rangos NVTX de `readout.*` ([datos](profiles/)). Los resúmenes de py-spy salen de `benchmarks/collapsed_profile_summary.py` con la raíz `_train_pass@training/mars_titan_run.py`, la misma del [informe del cableado](../campaign-kernels-wiring-20261010/README.md). La tabla da la parte inclusiva de cada función dentro de esa raíz.

| Función | B+C+M | B+M | M1 |
|---|---|---|---|
| Admisión (`_admit`) | 42,3 % | 41,9 % | 33,0 % |
| Selección de medoids (`select_anchored_medoids`) | 25,5 % | 25,3 % | 17,0 % |
| `FrozenEpisodeCodec.encode` | 17,4 % | 14,5 % | 19,3 % |
| `prepare` de Titans-MAC | 11,8 % | 9,4 % | 11,8 % |
| Huella de la vista CPU (`CPUDecisionBatch._digest`) | 8,6 % | 7,0 % | 9,4 % |
| Comprobación de finitud (`check_finite`) | 5,9 % | 6,5 % | 7,0 % |
| Instantánea del banco (`bank_snapshot`) | 4,3 % | 6,7 % | 4,9 % |

En B+C+M, `encode` pasa del 41,8 % del perfil con einsum del informe del cableado al 17,4 %. Casi todo lo que queda dentro de `encode` es la llamada al núcleo, que py-spy no desglosa sin pilas nativas.

Los rangos NVTX separan las fases del lector. Son segundos de reloj de todo el proceso con nsys, calentamiento incluido, y la ronda de B+M coincidió con más carga que las otras dos.

| Fase (rango NVTX) | B+C+M (s) | B+M (s) | M1 (s) |
|---|---|---|---|
| Admisión y consolidación (`readout.admit`) | 15,5 | 23,4 | 9,2 |
| Recuperación (`readout.snapshot` y `readout.read`) | 1,5 | 5,7 | 1,4 |
| Emisión (`readout.emit`) | 0,05 | 0,06 | 0,05 |
| Codificación (`readout.encode`) | 5,8 | 6,5 | 5,4 |
| Titans-MAC (`readout.prepare`) | 6,9 | 8,6 | 5,6 |
| Validación de la vista (`readout.validate`) | 1,9 | 2,1 | 1,8 |
| Gradiente del lector (`readout.update` y `readout.replay`) | 7,1 | 7,5 | 6,2 |

### Siguientes cuellos de botella

- **Consolidación.** La selección de medoids ocupa entre el 17 y el 25 % del recorrido y el trabajo propio de `RetentionBank.propose` otro 8 %. Ese trabajo copia en Python las claves de todos los registros del banco en cada propuesta, da formato a sus identificadores y restaura el banco desde una instantánea. Las selecciones ya usan las dos opciones de #363 que conservan los bits.
- **Vista CPU y su paso a la GPU.** `CPUDecisionBatch._digest` vuelve a calcular el SHA-256 de las entradas en cada `verify`, cuando la vista ya es memoria de solo lectura, y `check_finite` y `device_tensor` comprueban y copian cada modalidad por separado. Juntos suman entre el 17 y el 21 %. Es el mismo trabajo repetido que aparece en el recorrido por evento de Titans-MAC.
- **Recuperación y emisión** ya son pequeñas: la instantánea del banco ocupa entre el 4 y el 7 % y la emisión menos del 0,1 %.

## Dependencias

- **Rama `perf/campaign-kernels-wiring` (#363).** Las medidas de los lectores usan sus opciones de medoids (`reuse_distances` y `bounded_background`), que esta rama no contiene. Se midieron sobre la unión de las dos ramas, y esta PR debe integrarse después de aquella.
- **Vista v3.1 ([#428](https://github.com/GonxKZ/mars-titan/issues/428), `docs/data/edition-v3-1.md`).** Los lectores usan los eventos guardados de la edición v3 de #363 con los mismos tres parches que solo existen para medir: no se comprueban los archivos de muestras, los índices de observaciones conservan la huella de código con que se generaron y las anchuras de entrada salen de los eventos. Hay que repetir el caudal sobre la v3.1. El núcleo y las pruebas no dependen de la edición.

## Reproducción

- Núcleo: `ctest -R episodic_projection` en una compilación de `native/` con `MARS_TITAN_BUILD_SIMULATION=ON`, y `mars-titan-projection-benchmark` para los tiempos por versión.
- Codec: `MARS_TITAN_EPISODIC_NATIVE=enlace.so OMP_NUM_THREADS=2 PYTHONPATH=src python benchmarks/episodic_codec_projection.py salida.json`, con un enlace compilado con `MARS_TITAN_BUILD_EPISODIC_PYTHON=ON`.
- Pruebas: `pytest tests/memory/test_episodic_codec.py tests/memory/test_episodic_session.py tests/training/test_mars_titan_run.py` con `MARS_TITAN_EPISODIC_NATIVE`.
- Lectores: `benchmarks/chronological_replay.py` con `--family cm_readout --arm cm_v1_bcm`, `--family cm_readout --arm cm_v1_bm` o `--family readout --arm mars_titan_m1`, `--option precision='"fp32_strict"' --option block_rows=512 --predictor max_batch=1024 --predictor max_state_bytes=134217728 --strict --warmup 2 --segments 8` y los eventos de `benchmarks/chronological_events.py`, como en el [informe del cableado](../campaign-kernels-wiring-20261010/README.md#reproducción). Una envoltura de la medida sustituye `projection_backend` para elegir einsum o el núcleo, suma el tiempo de reloj y de CPU del hilo de cada `encode`, registra cada selección de medoids y el uso de CPU del proceso al terminar.
- Perfiles: `py-spy record --nonblocking --rate 100 --threads --format raw` y `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none` con `MARS_TITAN_NVTX=1`, seguidos de `nsys stats --report nvtx_pushpop_sum`.

## Límites

- La máquina estuvo compartida y el caudal de reloj de los tramos medidos varía mucho entre rondas. El tiempo de CPU de `encode` es la medida más estable de la mejora.
- La versión por defecto se eligió con medidas de un solo procesador. En otro, `mars-titan-projection-benchmark` indica si la más ancha sigue siendo la más rápida.
- No se mide energía ni se usan contadores de hardware. Ninguna cifra de este informe dice nada de la calidad predictiva.
