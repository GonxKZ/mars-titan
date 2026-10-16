# Validación de la vista CPU en el hilo del prefetch

Issue: [#638](https://github.com/GonxKZ/mars-titan/issues/638). Trabajo técnico sin pasos de optimizador ni evaluaciones científicas, con la protección local de aprendizaje activa.

| Archivo | Contenido |
| --- | --- |
| [`host_costs.py`](host_costs.py), [`host-costs-20261016.json`](host-costs-20261016.json) | Coste de cada pieza de la vista CPU y de los metadatos del estado con los eventos reales, antes del cambio |
| [`mutations.py`](mutations.py), [`mutations.json`](mutations.json) | Mutaciones dirigidas y la prueba que detecta cada una |
| [`abba-20261016.json`](abba-20261016.json) | Dos tandas A B B A del recorrido con eventos reales, de [`full_path_rounds.py`](../episodic-consolidation-20261013/full_path_rounds.py) |

## Punto de partida

Los perfiles de [#549](../titans-event-path-20261013/README.md#perfiles) muestran que el recorrido por evento de `titans_mac_online` está limitado por el anfitrión: la GPU solo está ocupada el 19,5 % de la ventana. En py-spy, `validated_cpu_batch` se lleva el 17,1 % de las muestras de `_train_pass`. Dentro de ella, el SHA-256 de la vista (`_digest`) se lleva el 9,7 % y `validate_historical_vectors` el 4,0 %. Las tres son garantías que deben conservarse, no cálculo del modelo.

`host_costs.py` mide cada pieza por separado, en CPU y con un hilo, sobre los 100 eventos guardados de `US+CN/fold-012` que usan las rondas de #549, troceados en 204 bloques de hasta 1.024 filas (140.843 filas). Cada medida es el mínimo de tres repeticiones.

| Pieza | Tiempo en los 204 bloques |
| --- | --- |
| `validated_cpu_batch` completo | 0,957 s |
| SHA-256 de la vista (1,00 GB a 1,69 GB/s, con SHA-NI) | 0,594 s |
| `validate_historical_vectors` | 0,127 s |
| `_state_size` (del que `json.dumps` 0,035 s) | 0,053 s |
| `_usage` (del que `json.dumps` 0,034 s) | 0,053 s |

## Qué cambia

La validación de los bloques de un evento no depende del estado del ajuste, solo de las entradas del evento y de la especificación del predictor. Por eso se puede adelantar. Cuando la tubería declara prefetch, `batched_events` ya leía los eventos en un hilo productor. Ahora, en los dos entrenadores (`ChronologicalTrainer` de Titans-MAC y `ReadoutTrainer` de los lectores), un segundo hilo encadenado al primero y con la misma profundidad valida los bloques del evento siguiente mientras el hilo principal calcula el actual.

- `input_pipeline.prepared(items, prepare, depth)` entrega parejas (elemento, resultado). Con `depth` positivo, `prepare` corre en el hilo de `background` y el resultado es un `Future` ya resuelto que guarda el valor o el error. El error solo se lanza cuando el consumidor pide el resultado, así que llega en la misma posición que en la ruta secuencial, después de las etiquetas del evento. Con `depth` 0 no hay hilo y `prepare` se calcula al pedir el resultado, como antes.
- `financial_inputs.validated_cpu_batches` valida en orden los bloques de un evento.
- `_observe` de los dos entrenadores recibe los bloques ya validados. Sin ellos, como en la inferencia, la evaluación y los adaptadores, valida igual que antes. `ReadoutAdapterTrainer` y `CorrectionInference` reenvían el argumento.
- La profundidad sale de `pipeline.prefetch_batches` de la fuente, que la campaña ya declara con 2 en `configs/baselines/historical-masked-campaign-execution.json`.

Las funciones de validación no cambian, así que las huellas, los errores y sus mensajes son los mismos por construcción. El SHA-256 libera el GIL con búferes de más de 2 KiB y las comparaciones de NumPy también lo sueltan en sus bucles, así que la parte más cara de la validación puede solaparse con el hilo principal. La memoria adicional son como mucho `depth` + 1 eventos validados en espera.

## Lo que no se cambia

- **Resumir en paralelo los bloques de un evento.** Con el prefetch el resumen ya sale del hilo principal. Repartirlo además entre hilos competiría con los dos hilos por proceso de la campaña.
- **Retirar comprobaciones de `validate_historical_vectors`.** Se comparte con otros llamadores y sus errores deben conservar el orden. Una vez fuera del hilo principal, su 4 % deja de estar en el camino crítico.
- **Contar los bytes de metadatos sin `json.dumps`.** En el perfil de #549, `canonical` llamado desde `_state_size` y `_usage` dentro de `prepare` suma 40 de 1.088 muestras (3,7 %). Una caché por identidad de las tuplas de flujos, IDs y cortes solo evitaría repetir la serialización de las tuplas que pasan del lote al estado, en torno a la mitad. Una ganancia del 2 % queda por debajo de la dispersión entre rondas y añadiría una caché que depende de que nadie sustituya los elementos de esas tuplas. Se descarta.

## Pruebas

- `tests/training/test_input_pipeline.py`: siete pruebas de `prepared`, con profundidad 0 y positiva. Comprueban el orden, que la preparación corre en el hilo productor solo con prefetch, que un error de preparación espera a que el consumidor lo pida, que un error de lectura llega después de los elementos anteriores, que la cola respeta su profundidad y que cerrar el consumidor cierra la fuente y detiene al productor.
- `tests/training/test_financial_run.py`: el recorrido con prefetch reproduce el secuencial y su reanudación, registro a registro. Un error de validación adelantado detiene el recorrido en el mismo punto que el secuencial, con profundidad 0 y 2, y `_observe` usa los bloques adelantados sin volver a validar.
- `tests/training/test_mars_titan_run.py`: lo mismo para los lectores M1 y M3.

Las tres pruebas que pasan de un segundo están en `tests/slow_tests.txt`.

El 16 de octubre, con la rama en ac3b5797, los enlaces nativos de `develop` en cd6b2c56 y CUDA oculta, pasaron sin fallos `tests/training` (2.523 pruebas), `tests/memory`, `tests/posttraining`, `tests/models` y `tests/test_learning_hold_lowering.py` (2.785) y `tests/training/test_titans_campaign.py` (31), cada parte en su plaza `memslot suite`. Las 465 omisiones se deben todas al bloqueo de aprendizaje.

## Mutaciones

`mutations.py` aplica nueve mutaciones dirigidas, cada una con las pruebas que deberían detectarla. Las nueve hacen fallar alguna prueba: errores lanzados al preparar en lugar de al pedir, prefetch sin hilo productor, preparación adelantada sin prefetch, `_observe` que vuelve a validar y recorridos que no pasan los bloques adelantados, en los dos entrenadores.

## Rondas A B B A

Las rondas usan los eventos, las opciones y las condiciones de las de [#549](../titans-event-path-20261013/README.md#medidas-de-tiempo): vista `US+CN/fold-012` de las vistas v3.1 de A v2, 88.028 filas y 11 llamadas a `_NoStepOptimizer` sin ningún paso, FP32 estricto, dos hilos, perfil de energía de ahorro y un trabajo `memslot gpu` por ronda. Titans-MAC usa la receta de la campaña y M1 las opciones de los lectores (`precision="fp32_strict"`, `block_rows=512`, `max_batch=1024` y `max_state_bytes=134217728`). A es `develop` en d0c1045d y B es la rama en ac3b5797. Los dos lados usan `MARS_TITAN_PREFETCH_BATCHES=2`, como la campaña, y el enlace episódico de `develop` en cd6b2c56. En A, los eventos guardados no pasan por el hilo de la fuente y la validación se hace en el hilo principal, como antes. En B, `prepared` la adelanta en su hilo. Una ronda está limpia si el llama-server de Kaizten no calculó en la GPU mientras duraba. El recibo es [`abba-20261016.json`](abba-20261016.json).

- **Tanda 1**, el 16 de octubre de 07:11 a 07:29, con carga media de 1,7 a 3,0.
- **Tanda 2**, de 08:27 a 08:45, con carga media de 1,1 a 4,4. Durante la tercera ronda de Titans-MAC, que es de B, se borraron árboles de trabajo en el mismo disco y la carga subió a 4,4.

Las dieciséis rondas están limpias.

| Modelo | Tanda | Filas/s antes | Filas/s después | Cociente de medias | Peor par |
| --- | --- | --- | --- | --- | --- |
| `titans_mac_online` | 1 | 18.726 y 18.164 | 19.577 y 19.281 | 1,05 | 1,03 |
| `titans_mac_online` | 2 | 17.951 y 18.293 | 18.861 y 18.620 | 1,03 | 1,02 |
| `mars_titan_m1` | 1 | 9.551 y 9.889 | 10.435 y 10.231 | 1,06 | 1,03 |
| `mars_titan_m1` | 2 | 9.890 y 9.903 | 10.553 y 10.303 | 1,05 | 1,04 |

En las dos tandas cada ronda de después supera a cada ronda de antes. La ganancia es del 3 al 6 %, bastante menos que el 17,1 % de las muestras que se llevaba `validated_cpu_batch`. Una explicación probable es el GIL: solo el SHA-256 y los bucles de NumPy lo sueltan, y el resto de la validación es código de Python que compite con el hilo principal. No se ha perfilado el reparto entre hilos, así que queda como hipótesis.

Los picos de la GPU no cambian (2.113 MiB asignados y 2.420 reservados en Titans-MAC, 402 y 492 MiB en M1). La memoria residente máxima sube entre 30 y 50 MiB, de 3.343 a 3.384 MiB a entre 3.382 y 3.410 en Titans-MAC y de 3.509 a 3.523 MiB a entre 3.553 y 3.576 en M1, que es lo que ocupan los eventos validados en espera.

Las capturas de las ocho rondas de cada modelo coinciden bit a bit con la primera ronda de antes. Son 16 comparaciones, cada una con los nombres de los parámetros, los gradientes de las 11 llamadas (442 tensores en Titans-MAC y 77 en M1) y las 3.347 predicciones pendientes, con `torch.equal` y el mismo tipo.

## Decisión

Se integra. La ganancia es pequeña pero se repite en las dos tandas y en cada pareja de rondas, con los mismos bits, el mismo comportamiento ante errores y un coste de memoria acotado por la profundidad del prefetch. Sin prefetch (`prefetch_batches` 0) el recorrido es el de antes.

## Reproducción

- Costes de partida: `python host_costs.py ÁRBOL EVENTOS VISTA/manifest.json salida.json 3`, con los eventos de `benchmarks/chronological_events.py VISTA/manifest.json TRABAJO eventos.pkl 100 6`.
- Mutaciones: `python mutations.py ÁRBOL salida.json`, sobre una copia del árbol y con el enlace episódico en `MARS_TITAN_EPISODIC_NATIVE`.
- Rondas: las órdenes del [informe de #547](../episodic-consolidation-20261013/README.md#reproducción), sin `admission_timer.py` y con `MARS_TITAN_PREFETCH_BATCHES=2`. Titans-MAC usa `--family titans --arm titans_mac_online` sin opciones del lector. Los bits salen de `reports/engineering/profiling-20261010/compare_capture.py` y el recibo de `python ../episodic-consolidation-20261013/full_path_rounds.py 638 DIRECTORIO bits.json salida.json`.
