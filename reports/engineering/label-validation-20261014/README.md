# Validación de etiquetas por columnas (14 de octubre de 2026)

Este informe acompaña a [#625](https://github.com/GonxKZ/mars-titan/issues/625). Recoge qué cambia en la lectura y validación de las etiquetas de cada activo (`CorpusDataset._labels`), cómo se ha comprobado que el resultado y los errores no cambian, y el tiempo y la memoria medidos en vistas reales de A v2. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto, y no se ha abierto el test final.

## Resumen

La validación de etiquetas pasa de recorrer motivos, particiones y cohortes fila a fila en Python a comprobarlos por columnas con PyArrow y NumPy, y la caché guarda la partición de cada etiqueta como un código de un byte. Las etiquetas devueltas, sus tipos, los mensajes de error y su orden no cambian: un sorteo de 800 tablas lo comprueba contra la revisión anterior, y en las vistas reales las huellas de lo devuelto coinciden en las cuatro particiones.

En `US+CN/fold-018`, la vista mayor de A v2 (5.008 activos), la caché de antes no cabía en su límite de 1 GiB: guardaba 3.891 activos y cada partición siguiente volvía a leer y validar los demás. Con los códigos ocupa 579 MiB y caben todos. La pasada de etiquetas de las cuatro particiones tardó 325 y 250 s en las dos rondas de antes y 36 y 40 s en las de después, y el pico de memoria del proceso bajó de 1.330 y 1.296 MiB a 812 MiB. En `CN/fold-012` la caché de etiquetas pasa de 200,8 a 87,3 MiB y el pico del proceso de 346 y 353 MiB a 236 MiB.

## Qué cambia

Tras #622, el perfil de la construcción de índices de `US+CN/fold-018` atribuía a `_labels` el 54,6 % del hilo principal. La función convertía motivos, particiones y cohortes a listas de Python fila a fila, ordenaba posiciones y fechas con `np.unique` para buscar duplicados y recibía de la vista temporal los motivos como un vector de objetos. Un perfil de cProfile de las etiquetas de los 810 activos de `CN/fold-012` mostraba además que la asignación temporal gastaba unos 3 ms por activo en vectores de objetos: `np.full` con `dtype=object` tarda 1,4 ms con 4.309 filas y copiar los motivos de anchura fija a objetos otros 1,6 ms.

- `validate_cohort_rows` compara el conjunto de valores distintos de la columna (`unique`) con la cohorte. Una fila ausente aparece como `None` y se rechaza igual que antes.
- Los motivos se comprueban con sus valores distintos, y las filas aceptadas salen de `pyarrow.compute.equal`. Una etiqueta excluida con partición se detecta con la validez de la columna. Una tabla vacía no compara motivos, porque su columna puede no ser de texto.
- Los duplicados de posición y de fecha se buscan con una comparación lineal cuando el vector ya es estrictamente creciente, que es el caso habitual. Si no lo es, se usa `np.unique` como antes.
- Las particiones se codifican como diccionario. Si alguna etiqueta aceptada no tiene partición se conserva la conversión anterior, porque la comprobación siguiente la rechaza siempre. Una columna ya codificada se decodifica antes, para que su diccionario no repita valores.
- La caché guarda las particiones como códigos de un byte y su vector de nombres, en vez de un texto de 10 u 11 caracteres por fila. Cada fila ocupa 33 bytes en vez de 72 a 76, y `_partition_labels` selecciona por código. Los recuentos por partición salen de `np.bincount` sobre los códigos.
- `TemporalInputs.partition_reasons` devuelve tramo y motivo como texto de anchura fija (U13 y U32), los mismos que daba el particionador. `assign` la usa y convierte los motivos a objetos una sola vez al final, porque la preparación de vistas (`fold_rows`) escribe después motivos de origen de cualquier longitud. `_labels` usa la variante de anchura fija.
- Al filtrar las filas aceptadas solo se copian las columnas que se leen después (maduración, objetivo y partición), y los años de la regla fija se calculan solo sin vista temporal.

Ninguna etiqueta, motivo ni receta cambia. Los mensajes de error y su orden son los mismos.

## Equivalencia

La referencia es `dd0a5d68`, la revisión de `develop` anterior a #625. `tests/training/test_label_validation.py` la carga con `git show` y compara con ella el resultado completo de `_labels` (arrays, tipos y mensaje de error) en 400 tablas sorteadas por modo, con la caché llena y sin ella. Las tablas cubren motivos desconocidos o ausentes, particiones de etiquetas excluidas, duplicados, posiciones fuera de rango, maduraciones no posteriores, objetivos no finitos, cohortes ajenas, recuentos que no concilian y filas aceptadas en el tramo reservado de la vista temporal. El sorteo recorre todos los mensajes de error.

La vista temporal de la prueba reserva el final del periodo como la real, con el tramo `test_reserved` y el motivo `final_test_reserved`. Sin esa reserva, la condición de que la vista acepte la fila no se comprobaba: la primera tanda de mutaciones dejó viva la que la quita, y tras añadirla el sorteo la detecta.

En los datos reales, [`label_passes.py`](label_passes.py) guarda una huella SHA-256 de los tipos y bytes de lo que devuelve `_labels` en cada partición. Las cuatro rondas de `CN/fold-012` (2.770.574 etiquetas) y de `US+CN/fold-018` (18.368.012) dan las mismas huellas en los dos árboles. La construcción de índices da los mismos archivos de eventos, con la misma huella SHA-256 por fase, en todas las rondas de las tres vistas medidas.

## Tiempo y memoria

Todas las medidas se tomaron el 15 de octubre de 2026 en rondas A B B A, con A = `develop` en f164b4fd (ya con #622) y B = la rama en f31ee5ea. Después, la rama se rebasó sobre cd6b2c56 sin cambiar ningún parche, y f31ee5ea corresponde a a6cc4c94. El commit siguiente, 6f2d5286, solo añade una prueba. Cada ronda se ejecutó en su propio trabajo `memslot heavy`, con `OMP_NUM_THREADS=2` y sin GPU, en el Ryzen 9 8945HS con el perfil de ahorro. El equipo tenía otras cargas y la carga media de cada ronda queda en [`abba-20261015.json`](abba-20261015.json), junto con todos los tiempos, picos y huellas.

| Pasada de etiquetas (`label_passes.py`) | A (rondas 1 y 4) | B (rondas 2 y 3) |
| --- | ---: | ---: |
| `CN/fold-012`, 810 activos, carga 12 a 15 | 42,9 y 11,7 s | 12,9 y 6,4 s |
| Pico, `CN/fold-012` | 346 y 353 MiB | 236 MiB |
| `US+CN/fold-018`, 5.008 activos, carga 7 a 10 | 324,7 y 249,9 s | 36,4 y 39,6 s |
| Pico, `US+CN/fold-018` | 1.330 y 1.296 MiB | 812 MiB |

La primera ronda de `CN/fold-012` leyó los archivos sin la caché de páginas del sistema y tardó casi cuatro veces lo que la cuarta. En `US+CN/fold-018`, con A, el tramo de ajuste tarda lo mismo que cada uno de los otros tres (de 43 a 101 s), porque la caché de 1 GiB no guarda todas las etiquetas y cada tramo vuelve a leer y validar las que faltan. Con B solo el tramo de ajuste lee archivos (33 y 37 s) y los otros tres salen de la caché en 1 s.

| Construcción de los cuatro índices | A (rondas 1 y 4) | B (rondas 2 y 3) | Pico A | Pico B |
| --- | ---: | ---: | ---: | ---: |
| `CN/fold-012`, carga 9 a 11 | 55,6 y 52,8 s | 30,4 y 38,7 s | 1.202 y 1.266 MiB | 1.098 y 1.150 MiB |
| `US+CN/fold-000`, rondas 3 y 4, carga 2 a 3 | 90,6 s | 77,8 s | 1.160 MiB | 1.093 MiB |
| `US+CN/fold-018`, carga 2 a 3 | 253,5 y 256,5 s | 99,4 y 112,6 s | 2.203 y 2.212 MiB | 1.754 y 1.747 MiB |

Los archivos de eventos tienen la misma huella SHA-256 por fase en las cuatro rondas de cada vista. En `US+CN/fold-000`, la primera pareja coincidió con compilaciones nativas de otra tarea (carga de 10 a 15) y tardó 374,3 s con A y 497,3 s con B, con aperturas de 9,8 y 16,5 s frente a 2,4 y 3,9 s en la segunda pareja. Esa pareja no permite comparar y se deja en el recibo sin usarla.

| Lectura de `US+CN/fold-018` (`train`, 300 eventos y 501.810 filas desde el cursor 4.709) | A | B |
| --- | ---: | ---: |
| Lectura, rondas de tiempo (carga 2 a 3) | 85,2 y 85,1 s | 71,7 y 71,5 s |
| Filas por segundo en régimen estable | 16.079 y 15.932 | 16.409 y 15.736 |
| Pico | 5.243 y 5.303 MiB | 5.262 y 5.264 MiB |

Las dos lecturas con huellas, una por árbol, dan la misma huella combinada de los lotes. El caudal en régimen estable no cambia, así que la diferencia de unos 14 s está en el tramo inicial de la lectura, antes de alcanzar ese régimen.

## Alternativas medidas y descartadas

Leer cada archivo grupo a grupo con `read_row_group`, en vez de `iter_batches`, devuelve las mismas tablas, los mismos fragmentos y los mismos bytes contados. En rondas intercaladas sobre 400 archivos de etiquetas de `CN/fold-012` la mediana fue 1,98 ms frente a 2,04 ms por archivo, y en 40 archivos de muestras 186 ms frente a 162 ms. No compensa cambiar `read_bounded_table`, que usan 20 llamadas del código.

## Pruebas y mutaciones

| Prueba | Qué comprueba |
| --- | --- |
| `tests/training/test_label_validation.py::test_labels_match_the_row_by_row_reading` | 400 tablas sorteadas por modo (regla fija y vista temporal), la mitad con un defecto: motivos desconocidos o ausentes, particiones de etiquetas excluidas o ausentes en aceptadas, duplicados, posiciones fuera de rango, maduración no posterior, objetivos no finitos, cohortes ajenas o ausentes, recuentos que no concilian, sin columna de motivos y etiquetas todas excluidas. Las filas se escriben desordenadas o no, en varios grupos, con motivos `large_string` y particiones como diccionario. Resultado, tipos y mensaje coinciden con `dd0a5d68`, con la caché y sin ella, y el sorteo recorre todos los mensajes |
| `test_cached_labels_take_less_than_half_the_memory` | La caché guarda códigos `uint8` y ocupa menos de la mitad que antes con las mismas etiquetas |
| `test_cohort_rows_match_the_row_by_row_check` | La comprobación de cohorte coincide con la anterior con columnas vacías, ausentes, nulas, de diccionario, de texto largo, enteras y binarias |
| `test_repeated_values_match_unique` | La búsqueda de duplicados coincide con `np.unique` en vectores ordenados, invertidos y sorteados |
| `test_an_empty_label_file_keeps_any_reason_type` | Un archivo sin filas con motivos enteros se acepta igual que antes |
| `test_temporal_assignment_matches_the_object_reasons_before` | En las vistas temporales reales de la fixture histórica, `assign` y `partition_reasons` dan los mismos tramos y motivos que `assign` en `dd0a5d68`, con sus tipos |
| `test_dictionary_codes_rebuild_the_column` | Los códigos reconstruyen la columna, también con fragmentos, texto largo y un diccionario con valores repetidos |

[`mutations.py`](mutations.py) aplica 19 mutaciones dirigidas sobre una copia del árbol ([`mutations.json`](mutations.json)). Las 19 hacen fallar alguna prueba. La tabla de `mutations.json` recoge, para cada una, la primera prueba que falla.

La batería completa pasa en ede86385, el mismo parche que 6f2d5286, por carpetas en trabajos `memslot suite` de 6 GiB, sin GPU y con el enlace episódico de `develop` en df7bb523: 10.829 pruebas aprobadas, 1.053 omitidas y ningún fallo. `tests/memory` (703) se ejecutó aparte en el mismo árbol y `tests/training` (2.534) tardó 2 h 2 min. Las omisiones son las que esperan CUDA (190), los ejecutables y la biblioteca de simulación no declarados en esta batería (328) y el bloqueo de aprendizaje. Tras el último rebase, las 32 pruebas de `tests/training/test_label_validation.py` vuelven a pasar.

## Reproducción

[`label_passes.py`](label_passes.py) abre una vista y pide `_labels` de cada activo en los cuatro tramos, en el orden de la construcción de índices. Guarda el tiempo de cada pasada, los bytes de etiquetas en la caché, el pico de memoria y una huella de tipos y bytes de lo devuelto. Se ejecuta desde la raíz de cada árbol con `PYTHONPATH=src`, en su propio trabajo `memslot heavy`:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python reports/engineering/label-validation-20261014/label_passes.py \
  VISTA/manifest.json salida.json
```

La construcción de índices usa [`index_build.py`](../observation-index-build-20261014/index_build.py) de #622 y la lectura [`reader_parity.py`](../calendar-price-windows-20261014/reader_parity.py) de #619, cada una desde la raíz del árbol medido y en su propio trabajo `memslot heavy`. Las rondas de tiempo de la lectura reutilizan los índices de la lectura con huellas del mismo árbol y fijan `READER_TIMING_ONLY=1`:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src python reports/engineering/observation-index-build-20261014/index_build.py \
  VISTA/manifest.json ÍNDICES salida.json
OMP_NUM_THREADS=2 MARS_TITAN_DECODE_WORKERS=2 MARS_TITAN_PREFETCH_BATCHES=2 MARS_TITAN_GROUP_CACHE_MIB=4096 \
  PYTHONPATH=src python reports/engineering/calendar-price-windows-20261014/reader_parity.py \
  VISTA/manifest.json ÍNDICES salida.json train mitad 300
```

Las mutaciones se aplican sobre una copia del árbol con `PYTHONPATH=src:. python reports/engineering/label-validation-20261014/mutations.py COPIA salida.json`.
