# Ejecución de la campaña: lectura, ranuras GPU y memoria

Este documento describe cómo se leen las vistas de la campaña con máscaras y cómo se reparten sus trabajos entre la GPU, la CPU y la RAM. Nada de lo descrito cambia qué datos ve un trabajo ni en qué orden. Las opciones de esta sección no forman parte de la identidad científica de ningún ajuste. Las medidas que justifican cada decisión están en el [informe del 9 de octubre](../../reports/engineering/campaign-pipeline-20261009/README.md).

## Lectura del corpus por lotes

`CorpusDataset` recorre los activos y sus grupos Parquet en el orden de su plan, que fija la semilla, la época y el cursor. La tubería se declara con `PipelineOptions` o con las variables `MARS_TITAN_DECODE_WORKERS`, `MARS_TITAN_PREFETCH_BATCHES` y `MARS_TITAN_GROUP_CACHE_MIB`. Con cero trabajadores y sin prefetch, la lectura es la secuencial.

- **Lectura por activo.** En la edición con máscaras, los grupos pendientes de un activo se leen con una sola llamada a `read_row_groups`, con los hilos de Arrow si la tubería tiene trabajadores. Las comprobaciones de `_sample_group` son por fila, así que la tabla del activo las supera si y solo si las supera cada grupo. Cada grupo es un tramo de esa tabla con los mismos valores que su lectura aislada. Si algo falla, el activo se lee grupo a grupo con el código anterior, que lanza el mismo error en la misma posición. La edición estricta y la caché de tablas siguen leyendo por grupo.
- **Decodificación en hilos.** `ordered_map` reparte los activos (o los grupos) entre hilos y entrega los resultados en el orden del plan, con un número acotado de unidades pendientes. Los cursores y los errores se calculan en el plan, en el mismo orden que la ruta secuencial.
- **Prefetch.** `background` monta los lotes en un hilo productor con una cola acotada. Cerrar el consumidor detiene al productor. Al terminar el intérprete, los productores abandonados se detienen y cierran sus archivos antes de que el proceso salga.
- **Memoria.** Por activo, cada unidad pendiente retiene un activo decodificado (como mucho 64 MiB, y unos 34 MiB con los activos más largos de la edición v3). El adelanto es de `decode_workers + 1` activos. Por grupo, retiene un grupo y el adelanto es de `2 · decode_workers + 2` grupos.

Las pruebas de `tests/training/test_pipeline_batches.py` comparan cada configuración con el lector de la revisión `14c6b1dd`, cargado desde git, con vectores no finitos, disponibilidad futura, ventanas de precios inválidas, grupos vacíos, cursores intermedios y las tres ablaciones de modalidades.

## Lectura cronológica

`FinancialObservationSource.batched_events` conserva el último grupo decodificado de cada activo dentro de un presupuesto de bytes (4 GiB por defecto) y decodifica en hilos los grupos de los `EVENTS_AHEAD` instantes siguientes. Cada bloque monta sus ventanas de precios con una sola transformación vectorizada, cuyo resultado por fila no depende del bloque. El contador `redecoded_groups` informa de los grupos que hubo que decodificar de nuevo porque la caché los había descartado.

Cada instante recorre los activos de su mercado en el mismo orden. Con ese acceso cíclico, descartar el grupo usado hace más tiempo descarta justo el siguiente que se pedirá, y con un presupuesto menor que el conjunto activo no se reutilizaría ningún grupo. Por eso, al superar el presupuesto, salen primero los grupos de activos que no aparecen desde hace 16 instantes y después los usados más recientemente, y la parte que cabe se sigue reutilizando. En `US+CN/fold-012` los grupos tienen 128 filas y ocupan unos 0,7 MB decodificados, así que un grupo por cada uno de los cerca de 5.000 activos del recorrido son unos 3,2 GiB y caben en el presupuesto por defecto.

Esta lectura retiene también las etiquetas y los precios de cada activo que aparece en el recorrido. Con la ventana conjunta `US+CN/fold-012`, un recorrido cronológico ocupa varios GiB de RAM, que deben figurar en la RAM declarada de su modelo.

## Huellas compartidas

Abrir una vista comprueba el SHA-256 de todos sus archivos. Con la ventana conjunta son unos 15.000 archivos y decenas de GB, así que cada trabajo que abría su vista volvía a leerlos completos. Con `MARS_TITAN_DIGEST_CACHE`, los procesos guardan las huellas en un archivo común, con cerrojo y escritura atómica, y cada una solo vale para la ruta y la firma de stat exactas con que se calculó (dispositivo, inodo, tamaño, mtime y ctime). Un archivo modificado cambia de ctime y se vuelve a leer completo. La campaña fija esa variable en `file-digests.json` de su salida.

## Declaración de ejecución

`configs/baselines/historical-masked-campaign-execution.json` declara las ranuras GPU, los trabajadores CPU, los hilos por trabajo, los presupuestos de VRAM y RAM, CUDA MPS y la tubería de lectura, con los recursos de cada modelo y, si hace falta, de cada ámbito. La VRAM de un modelo es la de todo su proceso, con el contexto de CUDA (unos 512 MiB fuera del asignador). Se pasa con `run --execution`. Sin ella la campaña conserva su ejecución anterior.

Antes de empezar se rechazan los planes fuera de orden topológico, los trabajos cuya estimación no cabe sola en su presupuesto y los dispositivos que no coinciden con su ejecutor.

Un modelo CUDA puede declarar `"campaign_process": true` para ejecutarse en un hilo del proceso de la campaña aunque haya ranuras. Es la opción de XGBoost y Ridge: los ajustes de una misma ventana comparten en ese proceso la matriz cuantizada de XGBoost y las estadísticas de Ridge, que en procesos separados se recalcularían. Esos trabajos se ejecutan de uno en uno y sin bloquear el bucle de la campaña, así que conviven con las ranuras si su VRAM cabe. El proceso de la campaña no acota su VRAM (XGBoost reserva con CuPy y RMM, fuera del asignador de PyTorch), de modo que su declaración debe ser una cota medida. XGBoost declara todo el presupuesto hasta medir sus rondas y se ejecuta sin otros trabajos GPU al lado. Desde el primer trabajo de este tipo, la admisión cuenta los 512 MiB del contexto CUDA que conserva el proceso de la campaña, salvo mientras corre uno de ellos, cuya declaración ya lo incluye.

## Ranuras GPU

Con más de una ranura, cada trabajo GPU se ejecuta en su propio proceso creado con `spawn`:

- **Admisión.** `ResourcePool` admite un trabajo si su dispositivo tiene una ranura libre, la suma de VRAM y RAM declaradas de los trabajos en curso más la suya cabe en el presupuesto y la memoria disponible del anfitrión cubre su RAM más la reserva. La guardia de disco de `campaign_storage` se aplica igual: con trabajos en curso, uno que no cabe en disco espera a que liberen su reserva.
- **Orden.** Entre los 16 primeros trabajos pendientes del plan con sus dependencias confirmadas, se intentan primero los de mayor VRAM estimada y los pequeños rellenan lo que queda. Con la misma VRAM se conserva el orden del plan. Si el primer trabajo listo de esa ventana no cabe, su dispositivo queda reservado para él y no empieza otro trabajo de ese dispositivo hasta que los que están en curso le dejen sitio. Así un trabajo grande, como un ajuste XGBoost, no espera indefinidamente detrás de pequeños que lo adelantan.
- **GPU libre al empezar.** La campaña lee `nvidia-smi -q -x` y no empieza si hay otro proceso de cómputo (tipo «C», «M» o «M+C») en la GPU, salvo el servidor MPS cuando se declara. Los procesos gráficos del escritorio («G» y «C+G») no cuentan, igual que en `GpuLease`, aunque su memoria sí resta de la libre.
- **Aislamiento.** El proceso limita su asignador a la VRAM declarada menos el contexto, sin que el trabajo pueda ampliarla, y con MPS su cliente queda limitado por `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`. Un trabajo que agota su VRAM falla solo. Toma un cerrojo exclusivo de su trabajo, ignora SIGINT y SIGTERM, muere con la campaña (`PR_SET_PDEATHSIG`) y atiende la parada por un evento compartido en su siguiente barrera confirmada. Solo el proceso de la campaña escribe recibos y resumen.
- **Picos observados.** El pico reservado de cada trabajo terminado, más el contexto, sustituye a la estimación de su modelo y ámbito si es mayor. Nunca baja de lo declarado. Se guarda en `observed-resources.json` de la salida.
- **Agotamiento de VRAM.** Si un proceso termina con `OutOfMemoryError`, la estimación de su tipo sube la mitad (al menos 1 GiB) sin pasar del presupuesto y el trabajo se repite desde su intento, como mucho dos veces.
- **Aritmética.** Cada trabajo CUDA, en serie o en su ranura, se ejecuta en FP32 estricto, sin TF32 en matmul ni en cuDNN, con la precisión más alta de matmul y cuDNN determinista. `CUBLAS_WORKSPACE_CONFIG` se hereda de la campaña, y los entrenadores lo exigen.
- **Hilos.** `threads_per_job` fija `OMP_NUM_THREADS`, `MKL_NUM_THREADS` y `OPENBLAS_NUM_THREADS` de cada proceso. El pool de Arrow toma el mismo valor.

Las ranuras de una familia solo se suben cuando una medida con sus formas reales muestra que el caudal agregado aumenta y que cada trabajo da los mismos bits solo y acompañado. La declaración actual usa tres ranuras con MPS a partir de las medidas del informe, que también recoge cómo cae el caudal cuando otros procesos ocupan la CPU.

MPS necesita su demonio antes de empezar la campaña, con el mismo directorio que declara la ejecución:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/run/user/1000/nvidia-mps
nvidia-cuda-mps-control -d      # arrancar
echo quit | nvidia-cuda-mps-control   # detener al terminar
```

## CPU compartida

La lectura de cada trabajo GPU usa sus hilos de decodificación y su prefetch, así que el caudal de la campaña depende de la CPU libre. Con ocho procesos de Python ocupados al lado, el caudal agregado de tres ranuras cayó a entre la mitad y un tercio del medido sin esa carga, y con cuatro no se distinguió de la variación entre repeticiones. Un trabajo CPU largo en paralelo (por ejemplo la etapa RL en CPU, que se lanza con su propia orden) debe acotar sus procesos para no quitar a la lectura los hilos que necesita.

## Limitaciones

- El número de ranuras es una decisión declarada a partir de medidas previas. El ejecutor no mide todavía el caudal agregado durante la campaña para añadir o quitar ranuras, porque los ejecutores no publican su progreso por filas. Hacerlo exige un contador de progreso en cada entrenador.
- La transferencia a la GPU con memoria fijada y un stream propio no está en esta capa, sino en los entrenadores.
- Las estimaciones de VRAM por modelo proceden de las medidas de memoria de la ventana más poblada. XGBoost declara el presupuesto completo hasta medir sus rondas reales.
- La admisión no cuenta los hilos de CPU. Las ranuras, los hilos por trabajo y la tubería se declaran juntos con las medidas del informe.
