# Ventanas de precios del calendario por bloque (14 de octubre de 2026)

Este informe acompaña a [#619](https://github.com/GonxKZ/mars-titan/issues/619). Recoge qué cambia en el lector cronológico por bloques, la comparación bit a bit con `develop` (06e69db6) en dos vistas reales de A v2, el caudal y la memoria medidos y las pruebas. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto, y no se ha abierto el test final.

## Resumen

- Con el contrato de ventanas por sesión del calendario de la v3.1, `develop` calculaba la ventana de precios de cada fila por separado y repetía en cada una la validación de todas las posiciones de calendario del activo. En el perfil de `CN/fold-012`, `price_windows` ocupaba el 67 % del hilo productor, que es el único que trabaja en serie.
- Ahora las posiciones de cada activo se validan una vez por recorrido. Las ventanas sin huecos de un bloque se forman juntas con la misma transformación que el camino sin calendario, y solo las que contienen sesiones ausentes de todo el mercado siguen la ruta fila a fila. La agrupación por instante del índice convierte columnas enteras en lugar de valor a valor.
- Los lotes de `batched_events` coinciden bit a bit con `develop` en seis tramos de `CN/fold-012` y `US+CN/fold-018`, de ajuste y de evaluación, con 47.570 filas con huecos en CN y 39.990 en la vista conjunta.
- En rondas alternas, el caudal estable del lector pasa de unas 3.600 a 3.800 filas/s a entre 14.300 y 21.100 (entre 3,9 y 5,5 veces en el peor par). El pico de memoria baja entre 100 y 190 MiB en esas rondas, y con un consumidor que calcula huellas queda dentro de la variación entre rondas, aunque las posiciones guardadas ocupan 145 MiB en la vista conjunta.
- Las 10 mutaciones dirigidas hacen fallar alguna prueba.

## Qué cambia

`CorpusDataset.calendar_positions` separa de `price_windows` la validación de las posiciones de calendario de un activo: cada precio debe ser una decisión del calendario de su mercado, en orden y fuera de las sesiones ausentes de todo el mercado. Las comprobaciones y los mensajes son los mismos. `price_windows` la sigue usando para la lectura por observación, que es la referencia de paridad.

`_BlockReader` guarda esas posiciones por activo durante el recorrido, como ya hacía con precios y etiquetas. Al montar un bloque, `_block_calendar_contexts` separa las filas cuya ventana no tiene huecos. Una ventana sin huecos ocupa exactamente las 64 filas de precios que terminan en la suya, y eso se sabe comparando dos posiciones (`positions[end] - positions[end - 63] == 63`), porque las posiciones son estrictamente crecientes. Esas filas se forman juntas con `_block_window_contexts`, la misma función del camino sin calendario, y reciben el canal de presencia a uno. Las demás filas pasan por `window_rows` y `_calendar_price_contexts`, como antes, con sus mismas comprobaciones.

`_instants` agrupa por instante las filas del índice. Antes convertía cada valor con `int` y comprobaba el orden fila a fila. Ahora convierte cada columna con `tolist`, localiza los cambios de fecha con NumPy y copia cada instante de una vez. Los errores (orden perdido, instante por encima de su presupuesto y posiciones ausentes) son los mismos y se lanzan en el mismo punto, después de entregar los instantes completos anteriores.

## Comparación bit a bit

[`reader_parity.py`](reader_parity.py) lee un tramo con la tubería de la campaña (2 hilos de decodificación, 2 lotes adelantados y 4 GiB de caché de grupos) y guarda una huella SHA-256 por instante de todo lo que entrega: cada array con su tipo y su forma, identificadores, mercados, etiquetas y cierre. Cada árbol construye sus propios índices. Los resultados están en [`parity.json`](parity.json).

| Vista | Tramo | Instantes | Filas | Filas con huecos | Huellas iguales |
| --- | --- | --- | --- | --- | --- |
| `CN/fold-012` | Ajuste, desde la mitad (instante 1.942) | 300 | 198.635 | 0 | Sí |
| `CN/fold-012` | Ajuste, desde el 20 de abril de 2019 (instante 3.169) | 120 | 90.834 | 47.570 | Sí |
| `CN/fold-012` | Evaluación, desde el inicio | 400 | 323.221 | 0 | Sí |
| `US+CN/fold-018` | Ajuste, desde la mitad (instante 4.709) | 300 | 501.810 | 0 | Sí |
| `US+CN/fold-018` | Ajuste, desde el 20 de abril de 2019 (instante 7.960) | 120 | 280.437 | 39.990 | Sí |
| `US+CN/fold-018` | Evaluación, desde el inicio | 200 | 508.068 | 0 | Sí |

## Caudal y memoria

Las rondas A B B A alternan `develop` (06e69db6, A) y esta rama (B) en el mismo tramo, cada una en su propio proceso con la plaza `heavy` de `memslot`, con la misma tubería y con los índices ya construidos. En estas rondas el consumidor solo cuenta filas (`READER_TIMING_ONLY=1`), así que mide el lector sin la huella. El caudal estable descarta los primeros instantes, en los que cada activo lee por primera vez sus etiquetas, precios y posiciones. La carga media de un minuto estuvo entre 1,9 y 3,4. Los resultados están en [`abba-20261014.json`](abba-20261014.json).

| Vista y tramo | Instantes | Filas | Lectura A | Lectura B | Caudal estable A | Caudal estable B | Pico de memoria A | Pico de memoria B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `CN/fold-012`, ajuste desde la mitad | 600 | 403.560 | 117,0 y 116,1 s | 27,9 y 27,8 s | 3.829 y 3.797 filas/s | 21.015 y 21.065 filas/s | 1.824 y 1.809 MiB | 1.692 y 1.667 MiB |
| `CN/fold-012`, evaluación desde el inicio | 400 | 323.221 | 93,3 y 90,7 s | 25,0 y 25,1 s | 3.561 y 3.665 filas/s | 14.912 y 14.306 filas/s | 1.908 y 1.894 MiB | 1.788 y 1.791 MiB |
| `US+CN/fold-018`, ajuste desde la mitad | 300 | 501.810 | 205,9 y 206,9 s | 92,0 y 89,2 s | 3.592 y 3.581 filas/s | 15.671 y 15.931 filas/s | 5.311 y 5.298 MiB | 5.133 y 5.120 MiB |

El peor par (la ronda B más lenta frente a la A más rápida) da 5,5 veces el caudal estable en el ajuste de CN, 3,9 veces en su evaluación y 4,4 veces en la vista conjunta. El tiempo total de lectura baja menos en la vista conjunta (de 206 s a 90 s), porque los 300 instantes medidos incluyen la primera lectura de unos 4.000 activos, que este cambio no toca. El pico de memoria residente baja entre 100 y 190 MiB en los tres tramos.

La caché de posiciones ocupa memoria nueva: 8 bytes por fila de precios de cada activo leído, 145 MiB en `US+CN/fold-018` con 4.985 activos. En las ejecuciones de paridad, con la huella completa de cada instante y una sola ejecución por árbol, el pico de esta rama superó al de `develop` en dos de los tres tramos de la vista conjunta (912 y 790 MiB) y quedó por debajo en el tercero. Para separar ese efecto de la variación entre ejecuciones, [`reader_memory.py`](reader_memory.py) repitió los 200 primeros instantes de evaluación de `US+CN/fold-018` en rondas A B B A, con un consumidor que calcula la huella de las entradas, y al terminar desglosó la memoria ([`memory-20261014.json`](memory-20261014.json)).

| Ronda | Árbol | Pico residente | Caché de grupos | Precios | Posiciones | Montón de malloc en uso |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | A | 5.818 MiB | 3.411 MiB | 871 MiB | 0 | 898 MiB |
| 2 | B | 5.777 MiB | 3.411 MiB | 871 MiB | 145 MiB | 1.044 MiB |
| 3 | B | 5.673 MiB | 3.411 MiB | 871 MiB | 145 MiB | 1.043 MiB |
| 4 | A | 5.675 MiB | 3.411 MiB | 871 MiB | 0 | 898 MiB |

Las cachés de grupos y de precios son idénticas, y el montón de B lleva los 145 MiB de las posiciones. Aun así, el pico residente de B queda dentro de la variación entre rondas del mismo árbol (143 MiB en A). El desglose no explica la diferencia de las ejecuciones de paridad, que no se reproduce en estas rondas. Guardar las posiciones en `int32` ahorraría 73 MiB, pero la cifra no justifica cambiar el tipo que comparten la ruta por bloques y la de referencia.

## Perfil

En `develop`, el perfil de py-spy de 300 instantes desde la mitad del tramo de ajuste de `CN/fold-012` atribuía al montaje de bloques (`_block`) el 87 % del hilo productor, y a `price_windows` el 67 %. Con este cambio, el mismo tramo deja `_block` en el 34 % del productor, con un 9 % para las ventanas de calendario. El resto del productor se reparte entre la primera lectura de etiquetas, precios y pies Parquet de cada activo en cada recorrido, la búsqueda de cada maduración y la programación de las decodificaciones. Los hilos de decodificación pasan casi la mitad de su tiempo leyendo grupos Parquet y otra parte abriendo cada archivo, porque el lector no conserva sus metadatos para no retener varios GiB en la ventana conjunta.

## Pruebas y mutaciones

| Prueba | Qué comprueba |
| --- | --- |
| `tests/memory/test_financial_observation_blocks.py::test_calendar_blocks_give_each_row_the_bits_of_its_window_alone` | Bloques de 600 filas, más de 256 sin huecos, con sesiones ausentes y precios en FP32: cada fila sale igual que su ventana calculada sola |
| `tests/memory/test_financial_observation_blocks.py::test_calendar_blocks_reject_windows_like_window_rows` | Una ventana que termina fuera de los precios o con una ausencia solo del activo se rechaza con el mismo error |
| `tests/training/test_calendar_price_windows.py::test_block_reader_checks_the_calendar_positions_of_each_asset_once_per_pass` | Las posiciones se validan una vez por activo y recorrido, y la lectura por bloques coincide con la lectura por observación |
| `tests/memory/test_observation_instants.py` | `_instants` coincide con la agrupación fila a fila en lotes que cortan instantes, con cursor, con lotes vacíos y en cada error |

Las pruebas de paridad entre la lectura por bloques y la lectura por observación (`test_pipeline_events.py`, `test_financial_observation_blocks.py` y `test_financial_observation_source.py`) siguen pasando.

[`mutations.py`](mutations.py) aplica 10 mutaciones dirigidas sobre una copia del árbol y todas hacen fallar alguna prueba ([`mutations.json`](mutations.json)): la condición de ventana completa, el canal de presencia, la caché de posiciones, el desplazamiento de las filas completas, la ruta de las filas con huecos, el presupuesto de filas por instante, la comprobación de orden, el cursor, el corte por fecha y el lote vacío.

## Siguiente coste

Tras este cambio, el productor dedica la mayor parte de su tiempo a la primera lectura de cada activo en cada recorrido (etiquetas, precios, posiciones y pies Parquet), que con 30 épocas se repite en cada pasada, y a la búsqueda de cada maduración. Dentro de esa primera lectura domina la validación de las etiquetas de `CorpusDataset._labels`, que también domina la construcción de los índices tras [#622](https://github.com/GonxKZ/mars-titan/issues/622). Conservar estas lecturas entre pasadas aumentaría la memoria residente.
