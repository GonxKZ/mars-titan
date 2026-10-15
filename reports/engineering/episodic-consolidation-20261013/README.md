# Consolidación del banco episódico

Recibos del 13 al 15 de octubre de 2026 para [#547](https://github.com/GonxKZ/mars-titan/issues/547). La primera parte ([#593](https://github.com/GonxKZ/mars-titan/pull/593)) parte de `develop` en 7012da85, que ya integra la proyección nativa del codec de [#520](https://github.com/GonxKZ/mars-titan/issues/520) ([#578](https://github.com/GonxKZ/mars-titan/pull/578)). La segunda, descrita en [Núcleos nativos y exportaciones del enlace](#núcleos-nativos-y-exportaciones-del-enlace), parte de `develop` en 131d809e.

| Archivo | Contenido |
| --- | --- |
| [`counts.json`](counts.json) | Trabajo contado de cuatro propuestas con el banco lleno e igualdad de bits de todo el flujo, antes y después |
| [`memory.json`](memory.json) | Pico de asignaciones de Python y NumPy de cada propuesta frente a `estimated_peak_bytes` |
| [`mutations.json`](mutations.json) | Mutaciones dirigidas y la prueba que detecta cada una |
| [`consolidation_counts.py`](consolidation_counts.py), [`consolidation_memory.py`](consolidation_memory.py), [`mutations.py`](mutations.py) | Scripts que generan cada recibo |
| [`abba-20261014.json`](abba-20261014.json) | Rondas del recorrido de M1, B+M y B+C+M con eventos reales |
| [`proposal-times-20261015.json`](proposal-times-20261015.json) | Rondas de las propuestas y las instantáneas con los núcleos nativos, de [`proposal_times.py`](proposal_times.py) |
| [`anchored-options-20261015.json`](anchored-options-20261015.json) | `benchmarks/anchored_medoid_options.py` con los núcleos nativos |
| [`kernel-mutations-20261015.json`](kernel-mutations-20261015.json) | Mutaciones de los núcleos y de las exportaciones, de [`kernel_mutations.py`](kernel_mutations.py) |

Una propuesta del banco es retención de episodios, no ajuste. Ningún recibo construye un modelo ni ejecuta pasos de optimizador, y la protección local de aprendizaje sigue activa.

## Qué se repetía

`RetentionBank.propose` hacía en cada propuesta, sobre todo el banco, dos trabajos que no dependen de los episodios nuevos:

- Convertía desde el enlace la clave de cada registro retenido, 64 números de Python por registro, para formar la matriz de puntos. La clave de un registro no cambia una vez escrito.
- Copiaba el banco con `restore(snapshot())`. Eso pasaba el generador y el recibo por JSON dos veces, volvía a leer los registros retenidos y los metadatos del enlace y validaba con `_check_receipt` un recibo que la propuesta sustituye al terminar.

## Qué cambia

- **`_fork`.** Crea el banco de la propuesta con los bytes de la memoria nativa y el estado del generador, sin el recibo y sin volver a validarlo. El estado es el mismo que da `restore(snapshot())`. `restore` sigue validándolo todo cuando el estado viene de fuera, por ejemplo al reanudar.
- **Claves por ID.** `_keys` guarda la clave FP32 de cada registro retenido como una fila de solo lectura. Cada clave se convierte desde el enlace una vez, al entrar en el banco o en la primera propuesta después de una recuperación. Un banco y sus propuestas comparten las filas, y cada propuesta conserva solo las de los registros que retiene.

No cambian la selección (`select_anchored_medoids`), el formato de los identificadores ni el SHA-256 de la geometría, que forma parte del contrato del recibo.

## Igualdad de bits

- **Pruebas.** `test_proposals_match_the_snapshot_copy_and_full_key_conversion_bit_for_bit` recorre cinco propuestas con las cuatro políticas y compara recibos, instantáneas (identidad, bytes de la memoria, generador y recibo) e IDs retenidos con la ruta anterior, `restore(snapshot())` más la conversión completa de claves.
- **Escala de la campaña.** `consolidation_counts.py` llena un banco de capacidad 1.024 con 12 propuestas de 128 episodios, con `reservoir` y con `anchored` (frontera 8 y 8 candidatos nuevos, como CM-v1), y con los contratos `legacy_v1` y `causal_v2`. En las 48 propuestas coinciden el recibo completo (selecciones, objetivo, contadores y huella de la geometría), la instantánea (bytes de la memoria, generador y recibo) y los registros retenidos (ID, clave, valor, etiqueta y fechas).

La identidad del banco incluye el SHA-256 de `retention_bank.py`, `anchored_medoids.py` y `medoids.py`, así que la comparación fija el mismo valor en los dos árboles. Con `legacy_v1`, esa identidad también siembra el reservorio nativo a través de `scope.representation`. Por tanto, cualquier cambio en esos archivos, aunque sea un comentario, cambia los sorteos de un banco `legacy_v1`. Sin fijar el valor, la propuesta 12 de `legacy_v1` con `reservoir` ya retiene otros registros. Es un comportamiento previo a esta rama. Los lectores de la campaña A usan `causal_v2`, cuyo reservorio no depende del ámbito, y en ellos el hash solo cambia la identidad y los bytes de la instantánea.

## Trabajo contado

Cuatro propuestas con el banco lleno (1.024 registros retenidos y 128 nuevos, 1.152 clientes), `develop` frente a la rama:

| Recuento en las cuatro propuestas | Antes | Después |
| --- | --- | --- |
| Lecturas de claves de registros ya retenidos | 4.096 | 0 |
| Llamadas a `json.dumps` | 32 | 4 |
| Caracteres de `json.dumps` | 115.826 a 119.448 | 4.404 a 4.848 |
| Llamadas a `json.loads` | 20 | 0 |
| Validaciones del recibo copiado (`_check_receipt_selection`, que solo llama `_check_receipt`) | 4 | 0 |
| Llamadas a SHA-256 | 8 (`reservoir`) y 520 (`anchored`) | iguales |

Cada propuesta deja de crear unos 65.536 números de Python para las claves. Las cuatro serializaciones que quedan son la identidad del banco nuevo. Según el código, cada propuesta deja de pedir al enlace una segunda lectura de los registros retenidos y `snapshot_metadata`, que solo servían para validar el recibo copiado. La copia de la memoria nativa (`snapshot_bytes` y `restore_bytes`) se mantiene. El SHA-256 no cambia: son la identidad, la huella de la geometría y, con `anchored`, el orden de los candidatos nuevos de `_centers`.

## Memoria

Con tracemalloc, el pico de asignaciones de Python y NumPy de cada propuesta con el banco lleno está entre 4,85 y 4,87 MB con `reservoir` y entre 5,09 y 5,10 MB con `anchored`. Los valores declarados en `estimated_peak_bytes` son 20,4 y 20,5 MB y no cambian. La rama sube el pico entre 1 y 10 KB, un 0,2 % como mucho. El pico llega durante la selección, cuando la lista de claves de Python ya se ha liberado, y la propuesta añade el diccionario de claves del banco nuevo y las filas de los episodios nuevos. Las filas de un banco lleno ocupan unos 0,4 MB que se conservan entre propuestas. Caben de sobra en los 4.096 bytes por registro que reserva el presupuesto propio de la propuesta. tracemalloc no ve las asignaciones del enlace C++, así que el pico medido es una cota inferior del total.

## Mutaciones

`mutations.py` cambia un único fragmento de `retention_bank.py`, ejecuta `tests/memory/test_retention_bank.py` con `-x` y sin orden aleatorio y restaura el archivo. Las cinco hacen fallar una prueba.

| Mutación | Qué rompe | Prueba que falla |
| --- | --- | --- |
| `fork_without_generator` | la propuesta no hereda el generador | `test_proposals_match_the_snapshot_copy_and_full_key_conversion_bit_for_bit[uniform]` |
| `fork_without_memory` | la propuesta no hereda la memoria | `test_proposals_keep_original_records_and_leave_previous_snapshot_unchanged[reservoir]` |
| `proposal_without_key_cache` | la propuesta no guarda las claves | `test_proposals_convert_each_retained_key_once[reservoir]` |
| `cache_keeps_every_client` | se guardan también las claves descartadas | la misma |
| `writable_cached_rows` | las filas compartidas se pueden escribir | la misma |

## Medidas de tiempo

Las rondas se tomaron el 14 de octubre de 2026 con los mismos eventos, opciones y condiciones que las de [#549](../titans-event-path-20261013/README.md#medidas-de-tiempo): vista `US+CN/fold-012` de las vistas v3.1 de A v2, 88.028 filas y 11 llamadas a `_NoStepOptimizer` sin ningún paso, FP32 estricto, dos hilos, perfil de energía de ahorro y un trabajo `memslot gpu` por ronda. El recibo es [`abba-20261014.json`](abba-20261014.json). Los árboles son A = bb19f7be (`develop` con #591) y B = 89913902, el commit de [#593](https://github.com/GonxKZ/mars-titan/pull/593), sin más cambios entre ellos.

| Modelo | Filas/s antes | Filas/s después | Cociente de medias | Peor par |
| --- | --- | --- | --- | --- |
| `mars_titan_m1` | 6.878 y 6.972 | 6.671 y 7.078 | 0,99 | 0,96 |
| `cm_v1_bm` (B+M) | 5.822 y 5.708 | 6.026 y 6.039 | 1,05 | 1,04 |
| `cm_v1_bcm` (B+C+M) | 5.908 y 5.683 | 6.026 y 5.661 | 1,01 | 0,96 |

El peor par divide la ronda más lenta de después entre la más rápida de antes. La memoria no cambia en ningún modelo (402 MiB de pico asignado y 492 o 494 MiB reservados). Las capturas de las cuatro rondas de cada modelo coinciden bit a bit entre árboles y dentro de cada árbol: nombres, gradientes de las 11 llamadas (77 tensores) y las 3.347 predicciones pendientes al terminar.

Solo B+M mejora de forma clara: sus dos rondas de después superan a las dos de antes. En M1 y B+C+M las rondas de los dos árboles se solapan, así que el cambio queda dentro de la variación entre rondas. Las cuentas de [Trabajo contado](#trabajo-contado) siguen siendo ciertas, pero en este tramo las propuestas del banco no pesan lo bastante para que retirar ese trabajo se note en el caudal de M1 o de B+C+M.

## Núcleos nativos y exportaciones del enlace

El perfil del lector con K = 4 ([#548](../mars-titan-k4-path-20261013/README.md#perfil-del-recorrido)) atribuía a `propose` un 27,6 % de `_train_pass` y a `bank_snapshot` un 9 %. Dentro de la propuesta dominaban `_bounded_minimum` (el fondo de los medoids anclados), la conversión de registros entre Python y el enlace y la copia del banco por su archivo. La segunda parte de #547 mueve ese trabajo al enlace episódico sin cambiar ningún bit.

- **Fondo de los medoids.** El producto de las coordenadas centradas sigue en la BLAS de NumPy, porque la cota de `_bounded_minimum` vale para cualquier orden de sumas, también con FMA. `nearest_reference_block` (`native/src/medoid_kernels.cpp`) hace en C++ el resto de cada bloque de clientes: el cuadrado con el mismo orden de operaciones que NumPy, el primer mínimo, la cota y la cadena de `hypot` de referencia solo en los pares que la cota no descarta. Devuelve el mínimo exacto de la cadena, así que el resultado es el de la referencia. La cota exige coordenadas que vengan de FP32 y el núcleo lo comprueba en cada llamada.
- **Sumas de las propuestas.** `min_column_sums` suma por columnas el mínimo entre la distancia y la base con `AccurateSum`, el mismo algoritmo de Shewchuk con redondeo final al par que `math.fsum`. Las dos sumas están correctamente redondeadas, así que coinciden en cualquier orden. Si aparece un NaN, un infinito o un desbordamiento, devuelve `None` y `_Costs` repite la suma en Python, que conserva sus errores. Con métricas enteras sigue la suma de Python.
- **Copia del banco.** `restore_from` copia la memoria nativa de un banco a otro sin pasar por su archivo. El estado es el mismo que da `restore_bytes(snapshot_bytes())`.
- **Claves y columnas.** `record_keys` devuelve las claves FP32 de muchos registros en una matriz, y `retained_arrays` las columnas de los registros retenidos (claves, valores, etiquetas, IDs y fechas) ordenadas por ID. `bank_snapshot` las usa en lugar de recorrer los registros uno a uno.

Con un enlace anterior que no tenga alguna de estas funciones se usa la ruta de Python de siempre, que da los mismos bits. Así la campaña pausada puede reanudarse con sus binarios fijados. `select_anchored_medoids` registra en `native_background` y `native_sums` qué núcleos ha usado.

### Igualdad de bits

- `native/tests/medoid_kernels_tests.cpp` compara el fondo con una búsqueda exhaustiva de la cadena de referencia en nubes de 1 a 120 centros y 1 a 65 dimensiones, con el producto calculado en cuatro órdenes (hacia delante, hacia atrás, por pares y con `std::fma`), centros en la media o en el origen, empates, duplicados, cancelación lejos del origen (bases de 10^6, 10^30 y -3·10^38), subnormales de FP32 y escalas mezcladas. Las sumas se comparan con valores conocidos, columnas con paso, bases infinitas y entradas no finitas.
- `tests/cm/test_anchored_medoids.py` compara la selección completa con el enlace y sin él en todas las geometrías de la prueba de bits y con más de 255 dimensiones. Si el búfer de productos de un bloque no llega a una fila entera de centros, la selección usa la cota de NumPy y lo registra. `test_native_column_sums_match_math_fsum` compara 300 casos sorteados con `math.fsum`.
- `tests/memory/test_retention_bank.py` recorre cinco propuestas con las cuatro políticas y exige el mismo recibo, la misma instantánea y los mismos IDs con el enlace nuevo, con un enlace sin estas funciones y con la ruta anterior de `restore(snapshot())` y conversión completa de claves. Otras pruebas comparan la copia directa con la del archivo, las claves exportadas con las de NumPy y las columnas de `bank_snapshot` con las listas de los registros.

Con el enlace de la rama y sin GPU pasaron 925 pruebas de `tests/cm`, `tests/memory` y `tests/training/test_cm_v1_factorial.py` en d07964e8, y otras 118 de los archivos de lectores, adaptadores y protección de aprendizaje de la reproducción. El commit final, 9ce26bae, solo añade pruebas nativas y la función que exponen. Con su enlace pasaron las 296 de `tests/cm`, `test_retention_bank.py` y `test_episodic_session.py`.

### Mutaciones de los núcleos

[`kernel_mutations.py`](kernel_mutations.py) aplica 23 mutaciones dirigidas sobre una copia del árbol ([`kernel-mutations-20261015.json`](kernel-mutations-20261015.json)). Doce cambian el núcleo C++: los dos márgenes y la contracción de la cota, el descarte de candidatos, el orden de la cadena de `hypot`, las comprobaciones de FP32, de valores finitos y de formas, y las sumas con su base, sus NaN, su escritura tras un fallo y su forma. Las otras once cambian el código de Python que los usa: el requisito de una fila entera de centros, el registro de los núcleos usados, las sumas sin los clientes salientes o con métricas enteras, la copia del banco y las claves y columnas exportadas. Las 23 hacen fallar alguna prueba.

En la primera tanda, sobre d07964e8, sobrevivieron tres. Quitar el margen absoluto o la contracción de la cota no cambiaba ninguna selección de las nubes de prueba, porque esos términos solo deciden pares en el borde de la cota. `bound_follows_the_proven_formula` compara ahora `background_bound` bit a bit con la fórmula escrita en el orden de NumPy, con 2.000 sorteos para cada una de las dimensiones 1, 64, 300 y 4.096. La tercera aceptaba un NaN en la distancia cuando la base era finita, y la prueba de sumas incluye ahora ese caso, en el que `numpy.minimum` propaga el NaN.

### Tiempos de las propuestas

[`proposal_times.py`](proposal_times.py) llena un banco de capacidad 1.024 con 7 propuestas de 4.040 episodios, la forma de las llamadas de los lectores, y cronometra cada `propose` y cada `bank_snapshot` en CPU. Guarda una huella SHA-256 de los recibos, de las instantáneas del banco y de los tensores de lectura, con la identidad del banco fijada igual en los dos árboles. Las rondas se tomaron el 15 de octubre de 2026 en orden A B B A dos veces por política, cada una en su proceso y su trabajo `memslot light`, con dos hilos y el perfil de energía de ahorro. A es `develop` en f164b4fd con el enlace de df7bb523 y B es la rama en d07964e8 con su enlace. El equipo tenía otras cargas (carga media entre 8,6 y 9,4), que quedan en el recibo [`proposal-times-20261015.json`](proposal-times-20261015.json).

| Política | Seis propuestas con el banco lleno, antes | Después | Cociente de medias | Peor par | Seis instantáneas, antes | Después |
| --- | --- | --- | --- | --- | --- | --- |
| `reservoir` | 2,07 a 2,19 s | 1,10 a 1,13 s | 1,90 | 1,83 | 162 a 175 ms | 22 a 24 ms |
| `anchored` | 3,27 a 3,43 s | 1,87 a 1,96 s | 1,74 | 1,67 | 157 a 172 ms | 20,5 a 20,7 ms |

Las huellas coinciden en las ocho rondas de cada política. El pico de memoria residente del proceso baja de 610 a 600 MiB. En `reservoir` no hay medoids, así que su mejora sale de la copia directa del banco y de las claves exportadas.

[`anchored-options-20261015.json`](anchored-options-20261015.json) es `benchmarks/anchored_medoid_options.py` con la rama. Con la forma de las llamadas de los lectores, la selección con cota y distancias reutilizadas tarda 0,38 y 0,37 s en la mediana de cinco repeticiones con las dos geometrías, y 0,22 s con los núcleos nativos, con los mismos campos y bits del objetivo que la referencia (7,8 a 8,2 s).

### Lo que queda en la propuesta

Un perfil con py-spy de las propuestas ancladas de B reparte así el tiempo de `propose`: la selección de medoids un 71 %, con un 35 % en `_compute` (las distancias de los candidatos variables, dimensión a dimensión con NumPy) y un 24 % en el fondo nativo con su producto. `validate_batch` y `retain_batch`, que reciben cada uno la lista de los 4.040 registros nuevos, suman un 19 %.

`_compute` no pasa a C++. Casi todo su tiempo es la propia cadena de `hypot`: con NumPy 2.5.3 y glibc 2.43, `np.hypot` tardó unos 100 ns por elemento y `np.subtract` unos 2,6 ns, en vectores de 2^20 elementos y con el equipo cargado. Un núcleo nativo llamaría a la misma función de la libm, que es lo que garantiza los mismos bits, así que solo ahorraría la resta y el recorrido de NumPy, en torno a un 3 % de `_compute`. Calcular menos distancias exactas exigiría cotas también en la tabla de distancias reutilizadas, que hoy guarda valores exactos para todas las evaluaciones.

### Alternativa descartada

En una medida exploratoria, sin recibo, un micronúcleo propio de AVX-512 sin FMA para el producto tardó entre 46 y 52 ms con 3.000 × 1.024 × 64, frente a 24 ms de la BLAS de NumPy con un hilo y 13,6 ms con dos. La cota no necesita un orden concreto de sumas, así que el producto se queda en la BLAS.

## Pendiente

- **Recorrido completo.** Faltan las rondas A B B A de M1, B+M y B+C+M con el enlace nuevo sobre los eventos de `US+CN/fold-012`, que esperan turno de GPU. Las órdenes son las de abajo.

Órdenes para repetir las medidas del recorrido desde la raíz de cada árbol, con `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src`, el enlace en `MARS_TITAN_EPISODIC_NATIVE`, los eventos de una vista v3.1 (`benchmarks/chronological_events.py VISTA/manifest.json TRABAJO eventos.pkl 100 6`) y un trabajo `memslot gpu` por ronda, alternando los árboles:

- `python benchmarks/chronological_replay.py VISTA/manifest.json TRABAJO --family readout --arm mars_titan_m1 --option precision='"fp32_strict"' --option block_rows=512 --predictor max_batch=1024 --predictor max_state_bytes=134217728 --strict --warmup 2 --segments 8 --events eventos.pkl --capture CAPTURA.pt --output resultados.jsonl --label ETIQUETA`. Para B+M y B+C+M se usa `--family cm_readout --arm cm_v1_bm` y `--arm cm_v1_bcm` con las mismas opciones.
- `python reports/engineering/profiling-20261010/compare_capture.py DIRECTORIO salida.json antes:despues` para los bits de las rondas.
- `py-spy record --rate 50 --format raw` y `MARS_TITAN_NVTX=1 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none` sobre la misma orden, resumidos con `benchmarks/collapsed_profile_summary.py` (raíz `_train_pass@training/mars_titan_run.py` y funciones como `propose@memory/retention_bank.py` y `select_anchored_medoids@cm/anchored_medoids.py`) y `benchmarks/nsys_range_summary.py`.

## Reproducción

- Cuentas e igualdad de bits: `python consolidation_counts.py ÁRBOL salida.pkl` en cada árbol, con la misma semilla, y comparación de `states` entre los dos archivos.
- Memoria: `python consolidation_memory.py ÁRBOL salida.json`.
- Mutaciones: `python mutations.py ÁRBOL salida.json` y `python kernel_mutations.py ÁRBOL salida.json`, cada uno sobre una copia del árbol y con el enlace compilado desde ella en `MARS_TITAN_EPISODIC_NATIVE`.
- Propuestas: `python proposal_times.py ÁRBOL POLÍTICA salida.json` en cada árbol, en rondas A B B A con un trabajo `memslot light` por ronda.
- Pruebas: `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 PYTHONPATH=src MARS_TITAN_EPISODIC_NATIVE=ENLACE pytest -p no:randomly tests/memory tests/cm tests/posttraining/test_readout_adapters.py tests/training/test_candidate_run.py tests/training/test_cm_v1_factorial.py tests/training/test_mars_titan_run.py tests/test_learning_hold_lowering.py tests/training/test_learning_hold_inventory.py`. Con `cuda:0` visible y el enlace compilado desde 7012da85 pasaron 962 pruebas en 34 min 48 s.
