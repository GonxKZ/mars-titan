# Proyección del codec por paneles (15 de octubre de 2026)

Este informe acompaña a [#628](https://github.com/GonxKZ/mars-titan/issues/628). Recoge qué cambia en la proyección del codec episódico de [#520](https://github.com/GonxKZ/mars-titan/issues/520) (`native/src/episodic_projection.cpp` y su enlace), cómo se ha comprobado que da los mismos bits y los tiempos medidos. La referencia es `develop` en 131d809e. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto, y la protección local de aprendizaje sigue activa.

| Archivo | Contenido |
| --- | --- |
| [`interleaved.cpp`](interleaved.cpp), [`interleaved.sh`](interleaved.sh), [`interleaved-20261015.jsonl`](interleaved-20261015.jsonl) | Núcleo de 131d809e y núcleo nuevo con un hilo, alternados en el mismo proceso |
| [`launch_threshold.cpp`](launch_threshold.cpp), [`launch_threshold.sh`](launch_threshold.sh), [`launch-threshold-20261015.jsonl`](launch-threshold-20261015.jsonl) | Un hilo frente a dos con el pool inter-op de ATen, como reparte el enlace |
| [`pool_threshold.cpp`](pool_threshold.cpp), [`pool_threshold.sh`](pool_threshold.sh), [`pool-threshold-20261015.jsonl`](pool-threshold-20261015.jsonl) | Un hilo frente a dos con un pool de OpenMP, la alternativa que no se puede usar |
| [`projection_times.py`](projection_times.py), [`projection-times-20261015.json`](projection-times-20261015.json), [`projection-times-loaded-20261015.json`](projection-times-loaded-20261015.json) | `project_codec_block` con el enlace de `develop` y con el de la rama, alternados muestra a muestra, con la máquina libre y con carga |
| [`benchmark-20261015.jsonl`](benchmark-20261015.jsonl) | `mars-titan-projection-benchmark`: versiones portable, AVX2 y AVX-512 del núcleo |
| [`mutations.py`](mutations.py), [`mutations.json`](mutations.json) | Mutaciones dirigidas del núcleo y su resultado |
| [`binding_mutations.py`](binding_mutations.py), [`binding-mutations.json`](binding-mutations.json) | Mutaciones dirigidas del reparto en el enlace y su resultado |

## Resumen

Con 32 filas o más, el núcleo copia los pesos de cada grupo de ocho salidas en un panel contiguo y lo recorre con todas las filas. Con un hilo y la máquina libre es entre 1,34 y 1,46 veces más rápido que el de 131d809e desde 64 filas, y por debajo de 32 filas, donde sigue leyendo la transpuesta, queda a menos de un 2 % de él. Desde 2^21 productos el enlace reparte los grupos con el pool inter-op de ATen. En `project_codec_block` con los seis bloques del codec, alternando los enlaces muestra a muestra con la máquina libre, la rama es 1,42 veces más rápida que `develop` con 64 filas, 1,85 con 278, 1,87 con 512 y 2,26 con 1.000, e igual de rápida con 1, 8 y 31. Todas las variantes dan los mismos bits que `einsum` en las pruebas y los mismos bytes en cada medida.

La primera versión del reparto usaba `at::parallel_for` y nunca calculaba en paralelo. La plantilla se expande en el enlace y, sin OpenMP al compilarlo, ejecuta todo el rango en el hilo que llama. Las pruebas de bits no podían verlo y `codec_projection_threads` decía dos hilos. Ahora una prueba comprueba en un proceso nuevo que el pool se usa desde el umbral, y una mutación que devuelve el cálculo al hilo que llama hace que falle.

## Qué cambia

`EpisodicCodec.encode` proyecta cada modalidad con `project_codec_block`. En el perfil del lector con K = 4 de [#548](../mars-titan-k4-path-20261013/README.md#perfil-del-recorrido), `encode` es el 16 % de `_train_pass` y la llamada nativa reúne el 70 % de `encode`, con bloques de cientos de filas. El núcleo de #520 leía cada vector de ocho salidas de la transpuesta (anchura × 127) con un paso de 127 dobles, así que casi todas las cargas cruzaban dos líneas de caché, y calculaba en un solo hilo.

- **Paneles.** Con 32 filas o más (`projection_packed_rows`), el núcleo recorre las salidas en grupos de ocho. Copia los pesos de cada grupo en un panel contiguo de `width · 8` dobles y recorre con él todas las filas, en mosaicos de cuatro filas y restos de dos y de una. El panel de la modalidad más ancha (512 entradas) ocupa 32 KiB.
- **Recorrido directo.** Con menos filas, copiar los paneles cuesta más de lo que ahorran y el núcleo lee la transpuesta como antes, con los mosaicos de #520.
- **Orden de las sumas.** Los dos recorridos llaman a la misma función, `add_in_einsum_order`, que fija el orden de einsum: bloques de ocho entradas, dentro de cada uno del par 6-7 al 0-1, y el resto por pares. Cada salida tiene sus dos acumuladores, multiplica y después suma, sin FMA. El recorrido solo cambia de dónde se leen los pesos y cuántas filas comparten cada carga.
- **Tramos de grupos.** `project_block_groups` calcula los grupos de salidas `[first, last)` con el recorrido que corresponde a las filas y un panel propio, y escribe solo sus columnas. `project_block` es el tramo completo.
- **Hilos.** Desde 2^21 productos (`projection_parallel_work`), el enlace reparte los grupos en tantos tramos como hilos tenga ATen (`OMP_NUM_THREADS` o `torch.set_num_threads`). Quien llama calcula el primero y una tarea de `at::launch` cada uno de los demás. Espera a todas antes de volver, también si alguna falla, y el estado de la espera vive en el montón para que la última tarea pueda avisar aunque quien llama ya haya vuelto. Un proceso hijo de `fork` hereda el pool sin sus hilos, así que un aviso registrado con `pthread_atfork` lo deja en un hilo. `codec_projection_threads` devuelve esa decisión para registrarla en las medidas. El enlace libera el GIL durante el cálculo, como antes. Si todos los hilos del pool inter-op estuvieran ocupados por tareas que llaman a la proyección, sus tramos no tendrían hilo libre y esas llamadas no terminarían. Ningún código del proyecto usa ese pool aparte de este enlace, que lo usa desde el hilo de Python.

Cada salida la calcula un solo hilo con el orden declarado, así que ni el recorrido ni el reparto cambian los bits. La transpuesta del codec, su huella y su identidad no cambian, y los enlaces anteriores siguen sirviendo para reanudar la campaña pausada con sus binarios fijados.

## Igualdad de bits

| Prueba | Qué comprueba |
| --- | --- |
| `native/tests/episodic_projection_tests.cpp`, orden declarado | Filas de 0 a 7, 9 y 13, y de 31 a 35 (los dos recorridos y todos sus restos), 13 anchuras de 1 a 513 y 11 números de salidas de 1 a 127, con valores normales y con ceros con signo, subnormales y magnitudes de 2^-300 a 2^300. La proyección entera y la calculada en tres tramos, uno vacío, dan los bits de una referencia escalar que suma en el orden de einsum, también las sumas de 0 · 0 que el núcleo omite |
| Filas independientes | Cada fila sola da los mismos bits que dentro de un bloque de 11 filas (recorrido directo) y de 43 (paneles) |
| Tramos a la vez | Con 31 y 37 filas y las anchuras 5, 78, 384 y 512 del codec, 2, 3 y 16 tramos calculados a la vez desde hilos distintos sobre la misma salida dan los bits de la proyección entera |
| Columnas de cada tramo | Un tramo escribe solo las columnas de sus grupos, también el último, de siete salidas |
| Regla de hilos | `projection_threads` en el borde del trabajo con 1, 3, 32 y 1.000 filas, sin filas y sin hilos disponibles |
| Formas y versiones | Rechazo de formas incoherentes, de tramos invertidos y de grupos que no existen antes de escribir, y versión por defecto igual a la más ancha |
| `tests/memory/test_episodic_codec.py`, hilos | `project_codec_block` con 1, 2, 3 y 4 hilos de ATen da los bits de `_declared_order` en siete formas a ambos lados del umbral y en los dos recorridos, y `codec_projection_threads` coincide con la regla |
| Uso del pool | En un proceso nuevo, después de proyectar 64 × 512 con dos hilos torch ya no deja fijar los hilos inter-op, porque el pool está en uso. Con 32 × 512, justo por debajo del umbral, sí deja |
| Hijo de `fork` | El padre usa el pool y un hijo de `fork` proyecta con un hilo y los mismos bits, sin quedarse esperando |

El resto de `test_episodic_codec.py` compara `encode` con el enlace, con un enlace anterior y sin enlace. Cada prueba nativa se ejecuta con las versiones portable, AVX2 y AVX-512. Las pruebas nativas pasan compiladas con GCC y con Clang, y con ThreadSanitizer y AddressSanitizer más UndefinedBehaviorSanitizer. `tests/memory` y `tests/cm` pasan con el enlace de la rama (884 pruebas en 201b927c, antes del cambio del reparto). Tras ese cambio pasan las 56 de `test_episodic_codec.py`. Después, la rama se rebasó sobre e867e30f, que añade al mismo enlace los núcleos de los medoids de #547 sin tocar la proyección, y 537b102b corresponde a b61c41f4 con el mismo parche. Con el enlace recompilado tras el rebase pasan las pruebas nativas de la proyección y de los medoids, las 56 de `test_episodic_codec.py`, las 199 de `tests/cm` y las 81 de `tests/memory/test_retention_bank.py`.

[`mutations.py`](mutations.py) aplica 20 mutaciones dirigidas sobre una copia del árbol: el orden compartido, los acumuladores, las cargas y la escritura de cada recorrido, la copia de los paneles, los límites de los tramos, la regla de hilos, la versión por defecto y la compilación con `-ffp-contract=fast`. Las 20 hacen fallar las pruebas nativas ([`mutations.json`](mutations.json)). 18 fallan con el mensaje de su comprobación, `writes_every_lane` corrompe el montón y `inverted_range_accepted` termina con una violación de segmento al recorrer el tramo invertido, antes de llegar a la comprobación que lo rechaza. La compilación con `-ffp-contract=fast` solo cambia los bits de la versión AVX-512.

[`binding_mutations.py`](binding_mutations.py) compila el enlace con cada una de siete mutaciones del reparto y ejecuta las pruebas de la proyección: tareas calculadas en el hilo que llama (el fallo de la primera versión), tareas o primer tramo sin calcular, último tramo más allá del final, sin esperar a las tareas, pool desactivado y aviso de `fork` vacío. Las siete compilan y las siete hacen fallar alguna prueba ([`binding-mutations.json`](binding-mutations.json)). Devolver las tareas al hilo que llama solo lo detecta la prueba del pool, porque los bits no cambian. Sin la espera, el proceso termina con un fallo de memoria, y con el aviso de `fork` vacío el hijo se queda esperando a un pool sin hilos hasta la alarma de la prueba.

## Tiempos

Las medidas finales se tomaron el 15 de octubre de 2026 desde 537b102b, en el Ryzen 9 8945HS con el perfil de energía de ahorro y cada una en su trabajo `memslot light`. Empezaron con el equipo libre (carga media de 1,2) y, a partir de las 14:10, otras aplicaciones volvieron a cargarlo. Con carga, los núcleos se quedan en 1,4 GHz y los tiempos absolutos llegan a duplicarse, así que solo se comparan variantes alternadas dentro de la misma medida. La carga al empezar cada medida queda en su recibo o en la tabla de abajo.

### Mosaicos

En una medida exploratoria, sin recibo y con carga media de 15 a 17, se compararon paneles con mosaicos de 4 × 1, 8 × 1, 4 × 2, 6 × 2, 4 × 3 y 3 × 4 filas × vectores, con los mismos bits. El de 4 × 1 fue el más rápido o empató con 8 × 1 en 64, 278, 512 y 1.000 filas, y los de dos o más vectores perdieron entre un 15 y un 50 % frente a él. Empaquetar cada grupo justo antes de usarlo, en vez de toda la transpuesta al principio, rindió igual con muchas filas y mejor con 64, porque el panel sigue en la caché.

### Núcleo con un hilo

[`interleaved.sh`](interleaved.sh) alterna los dos núcleos en el mismo proceso, con los seis bloques del codec por llamada y 31 repeticiones, con carga media de 1,2. Las dos salidas son iguales bit a bit en todas las filas.

| Filas | 131d809e | Rama | Razón (p10 a p90) |
| ---: | ---: | ---: | ---: |
| 1 | 49,3 µs | 50,3 µs | 0,98 (0,88 a 1,00) |
| 8 | 144,0 µs | 144,2 µs | 0,98 (0,92 a 1,09) |
| 16 | 281,4 µs | 284,9 µs | 1,00 (0,96 a 1,04) |
| 24 | 413,8 µs | 417,8 µs | 0,99 (0,96 a 1,04) |
| 31 | 579,5 µs | 591,3 µs | 1,01 (0,93 a 1,06) |
| 32 | 554,2 µs | 500,5 µs | 1,11 (0,97 a 1,21) |
| 48 | 846,2 µs | 690,2 µs | 1,25 (1,17 a 1,31) |
| 64 | 1.130,5 µs | 841,3 µs | 1,34 (1,29 a 1,43) |
| 128 | 2.264,6 µs | 1.674,8 µs | 1,39 (1,25 a 1,47) |
| 278 | 5.177,9 µs | 3.697,9 µs | 1,44 (1,28 a 1,53) |
| 512 | 9.030,1 µs | 6.216,1 µs | 1,46 (1,40 a 1,55) |
| 1.000 | 17.910,4 µs | 12.477,7 µs | 1,46 (1,35 a 1,58) |

Por debajo de 32 filas los dos núcleos siguen el mismo recorrido y la rama añade una capa de llamadas y la comprobación de los tramos. Eso cuesta hasta 1 µs por llamada de seis bloques, que en el enlace completo queda dentro de la variación. Con 1,4 GHz y carga, un bloque de 64 × 384 tarda unos 380 µs, unas 11,7 operaciones de coma flotante por ciclo, cerca del 73 % de lo que admite Zen 4 sin FMA con multiplicaciones y sumas separadas.

### Hilos

La primera versión de la rama creaba hilos con `std::jthread` en cada llamada. Con carga media de 13 a 20, `mars-titan-projection-benchmark` midió un coste fijo de entre 170 y 250 µs por llamada solo por crearlos. Esa medida es exploratoria y no tiene recibo, porque el código ya no existe. La segunda usaba `at::parallel_for` y, como se explica en el resumen, calculaba en un hilo.

Un pool de OpenMP sería lo más barato. [`pool_threshold.sh`](pool_threshold.sh), compilado con GCC y su libgomp, gana a un hilo desde unos 50.000 productos, porque sus hilos esperan activamente un tiempo antes de dormirse. Clang, que compila el proyecto, no genera OpenMP para libgomp, y añadir el runtime de OpenMP de LLVM metería un segundo runtime en el proceso de torch. Queda como referencia.

[`launch_threshold.sh`](launch_threshold.sh) reparte como el enlace, con `at::launch`. Los hilos del pool inter-op duermen en cuanto acaban una tarea, así que despertarlos cuesta unos 14 µs en llamadas seguidas con la máquina libre, 80 µs tras 3 ms de reposo y entre 90 y 115 µs con carga. Las razones son un hilo frente a dos, alternados en el mismo proceso, y los bits coinciden en las 80 formas.

| Productos | Formas | Llamadas seguidas (carga de 1,4 a 4) | Tras 3 ms de reposo (carga de 9 a 6) |
| ---: | --- | ---: | ---: |
| 0,5 a 1,3 millones | 8 × 512, 16 × 384, 16 × 512, 64 × 78, 128 × 78 y 1.000 × 5 | 0,87 a 1,28 | 0,72 a 1,24 |
| 1,5 a 2,1 millones | 31 y 32 × 384, 31 y 32 × 512 | 1,06 a 1,31 | 1,13 a 1,25 |
| 2,1 a 4,2 millones | 64 × 384, 64 × 512 y 278 × 78 | 1,36 a 1,47 | 1,23 a 1,34 |
| 5 a 18 millones | 128 y 278 × 384 y 512, 512 y 1.000 × 78 | 1,45 a 1,69 | 1,49 a 1,68 |
| 25 a 65 millones | 512 y 1.000 × 384 y 512 | 1,75 a 1,84 | 1,75 a 1,90 |

Una medida exploratoria anterior, sin recibo, con el mismo núcleo y carga media de 13, dio lo mismo: desde 2^21 productos, entre 1,26 y 1,80 veces en caliente y entre 1,36 y 1,83 en frío, y entre 0,5 y 2,1 millones, razones de 0,64 a 1,34. Por eso el umbral es 2^21, algo más de 2 millones de productos. En el codec reparte el bloque de gráficos desde 33 filas, el de macro desde 40, los de precios y noticias desde 44, el de fundamentales desde 212 y el de presencia desde 3.303.

### Enlace completo

[`projection_times.py`](projection_times.py) carga cada enlace en su propio proceso y pide una muestra a cada uno por turnos, en un orden que rota en cada ronda, con 42 rondas por número de filas. Una muestra proyecta los seis bloques en el orden de `encode`, repetidos hasta unos 5 ms. Las tres variantes son el enlace de `develop` (`df7bb523`, con el núcleo de 131d809e), la rama con `torch.set_num_threads(1)` y la rama con los dos hilos de `OMP_NUM_THREADS=2`. Las tres dan los mismos bytes. La carga media fue de 1,2.

| Filas | `develop` | Rama, un hilo | Rama, dos hilos | Razón con un hilo (IC 95 %) | Razón con dos hilos (IC 95 %) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 58 µs | 58 µs | 57 µs | 1,00 (0,99 a 1,01) | 1,02 (1,00 a 1,03) |
| 8 | 162 µs | 166 µs | 163 µs | 0,98 (0,97 a 1,00) | 0,99 (0,97 a 1,01) |
| 31 | 590 µs | 597 µs | 594 µs | 0,99 (0,97 a 1,01) | 1,00 (0,98 a 1,01) |
| 64 | 1.243 µs | 887 µs | 894 µs | 1,40 (1,38 a 1,42) | 1,42 (1,31 a 1,53) |
| 278 | 5.702 µs | 3.815 µs | 3.067 µs | 1,47 (1,42 a 1,52) | 1,85 (1,77 a 1,96) |
| 512 | 9.715 µs | 6.658 µs | 5.226 µs | 1,49 (1,46 a 1,52) | 1,87 (1,77 a 2,06) |
| 1.000 | 19.582 µs | 13.385 µs | 8.926 µs | 1,47 (1,41 a 1,52) | 2,26 (2,12 a 2,30) |

Las razones son la mediana de las 42 razones emparejadas de cada ronda y el intervalo es un bootstrap de esa mediana. Con la máquina libre, el cociente de medianas entre dos hilos y uno es 0,99 con 64 filas y de 1,24 a 1,50 desde 278. Media hora después, con otras aplicaciones cargando el equipo (carga media de 3,5), la misma medida con las dos variantes de la rama ([recibo](projection-times-loaded-20261015.json)) dio 1,31 veces con 64 filas y entre 1,74 y 1,76 desde 278, y lo mismo con 1, 8 y 31 filas. Una comprobación sin recibo, alternando uno y dos hilos dentro de un solo proceso con esa carga, dio 1,31 veces con 64 filas, 1,35 con 128 y 1,66 con 278. Una explicación probable es que, con el equipo libre, el núcleo que despierta la tarea arranca a una frecuencia más baja que la del hilo que llama, que ya está a plena velocidad. No se midió la frecuencia de cada núcleo durante las llamadas, así que queda como hipótesis.

Una primera tanda midió cada enlace en un proceso distinto, en rondas A B B A de 31 llamadas, con carga media de 13. El mismo enlace varió hasta tres veces entre rondas (de 138 a 367 µs con una fila y de 43 a 129 ms con 1.000), así que se sustituyó por el diseño emparejado. Esa tanda usaba además la versión con `at::parallel_for`.

### Versiones del núcleo

`mars-titan-projection-benchmark` mide las versiones portable, AVX2 y AVX-512 del núcleo de la rama con un hilo, los seis bloques del codec y 41 repeticiones, con carga media de 4 a 8. AVX-512 es la más rápida con todas las filas: 1,04 veces más rápida que AVX2 con una fila y entre 1,30 y 1,60 desde 16, y entre 1,44 y 3,84 veces más rápida que la portable. Sigue siendo la versión por defecto en este procesador. Las pruebas comprueban que las tres dan los mismos bits.

## Pendiente

- **Recorrido del lector.** Faltan las rondas A B B A de M1, B+M y B+C+M con eventos reales de `US+CN/fold-012`, como las de [#547](../episodic-consolidation-20261013/README.md#pendiente), que esperan turno de GPU.

## Reproducción

Desde la raíz del árbol, con `OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=` y cada orden en su trabajo `memslot`:

- Pruebas nativas: preset `native-candidate-cuda` con `-DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON` y `ctest --preset native-candidate-cuda -R episodic_projection`.
- Pruebas de Python: `PYTHONPATH=src MARS_TITAN_EPISODIC_NATIVE=ENLACE pytest -p no:randomly tests/memory tests/cm`.
- Mutaciones del núcleo: `python reports/engineering/codec-projection-panels-20261015/mutations.py COPIA salida.json`, sobre una copia del árbol.
- Mutaciones del enlace: `PYTHONPATH=src python reports/engineering/codec-projection-panels-20261015/binding_mutations.py COPIA COMPILACIÓN salida.json`, con COMPILACIÓN configurada con ese preset en la copia.
- Núcleo con un hilo: `reports/engineering/codec-projection-panels-20261015/interleaved.sh 31 1 8 16 24 31 32 48 64 128 278 512 1000`.
- Umbral del reparto: `launch_threshold.sh 21 warm` y `launch_threshold.sh 11 cold`, con `python` del entorno con PyTorch, y lo mismo con `pool_threshold.sh` para OpenMP.
- Enlace completo: `PYTHONPATH=src python reports/engineering/codec-projection-panels-20261015/projection_times.py salida.json develop=ENLACE_DEVELOP rama-un-hilo=ENLACE:1 rama=ENLACE`.
- Versiones del núcleo: `mars-titan-projection-benchmark`, del mismo preset.
