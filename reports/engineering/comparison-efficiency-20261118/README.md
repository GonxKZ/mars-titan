# Lectura cronológica y concurrencia en la RTX 4070

En la comparación de noviembre, cada época vuelve a montar las entradas de miles de archivos Parquet organizados por activo. La prueba materializó los lotes una vez, por sesión, en un archivo Arrow IPC con compresión LZ4. Los bloques conservan fechas de disponibilidad, máscaras, identificadores, valores y etiquetas en su orden de maduración. Las repeticiones verifican el índice, el origen y la huella del archivo.

| Recorrido medido | Referencia | Entrada preparada | Factor de caudal |
| --- | ---: | ---: | ---: |
| Preparar y validar en CPU una época completa del origen 2021 | 279,26 s | 67,93 s | 4,11× |
| Tramo de Titans-MAC, con cálculo hacia delante y hacia atrás | 10,30 s | 4,00 s | 2,57× |
| El mismo tramo con el lector M2 | 16,15 s | 11,10 s | 1,45× |
| Validación completa de M2 con el calentamiento ya residente | 34,77 s | 21,31 s | 1,63× |

La época contiene 2.957.359 observaciones y 2.892.903 etiquetas. Coincidió la huella ordenada de todos los bloques y de sus etiquetas en cuatro pasadas. Construir la copia de 10,08 GiB costó 315,29 s. En la lectura aislada se recupera ese coste en la segunda reutilización. El máximo de RSS del proceso incluye páginas del archivo mapeado. Subir de dos a cuatro hilos de decodificación con prefetch de dos tampoco redujo el tiempo del tramo tardío original.

Los tramos CUDA usan los pesos seleccionados del origen 2021, semilla 44, FP32 estricto y 1.024 activos en la cohorte. El tramo tardío empieza en el evento 4.527 de 5.031. Se inicializan los flujos del tramo y solo se resuelven decisiones emitidas dentro del propio recorrido. Tras cuatro segmentos de calentamiento se miden doce segmentos de ocho sesiones, con 88.888 observaciones y 87.007 etiquetas. La comprobación de paridad se ejecuta aparte del cronometraje. Coinciden las 117.820 emisiones del recorrido comprobado, las dieciséis colecciones de gradientes tras el recorte y los contadores. El sustituto del optimizador libera gradientes sin modificar pesos. Las cifras excluyen el paso de AdamW y la escritura de checkpoints. Las cinco pasadas completas de validación de M2 reprodujeron su Parquet original de 128.898 filas, incluidos los cuantiles. El primer ensayo rechazó una huella de ajuste al abrir validación. La repetición utilizó el índice de validación.

## Dos procesos y coste de los kernels

Se compararon Titans-MAC con semilla 42 y M2 con semilla 44, dos trabajos independientes del mismo origen. M2 utilizó su padre terminado de la semilla 44. Cada proceso cargó pesos propios. Tras dos segmentos de calentamiento, se midió el tiempo de terminar ambos recorridos, con 87.660 observaciones en total. En serie se sumó el tiempo de ambos. En paralelo se tomó desde el primero que empezó hasta el último que terminó. Hubo dos repeticiones cronometradas por configuración y otras independientes para verificar gradientes y emisiones. Se usó CUDA sin MPS.

| Entrada | Dos recorridos en serie | Dos a la vez | Factor de caudal |
| --- | ---: | ---: | ---: |
| Original | 11,83 s | 10,56 s | 1,12× |
| Preparada | 7,50 s | 5,83 s | 1,29× |

La combinación de entradas preparadas y dos procesos dio 2,03× respecto a la referencia original en serie en esos tramos. Con la entrada original, los dos procesos llegaron en conjunto a 4,90 GiB de memoria anónima dentro del grupo que les reservaba 8 GiB. El mínimo observado de VRAM libre fue de 2,75 GiB. Los ajustes completos de distintas arquitecturas tienen otros picos y costes. La medición corresponde a la RTX 4070 Laptop con el perfil de energía bajo utilizado en la campaña.

En un segmento perfilado con la entrada preparada, los kernels ocuparon el 32,85 % de 509 ms en Titans-MAC y el 7,68 % de 1.048 ms en M2. Se midió la unión de intervalos CUDA dentro de la ventana marcada. Los tiempos del perfil tienen la sobrecarga del trazador. Resolver en CPU los indicadores escalares de finitud que se trasladan al anfitrión dio 1,07× en el tramo del núcleo. Se comprobaron 171 combinaciones que incluyen NaN, infinitos y orden de los errores. Agrupar las cinco transferencias de entradas FP32 al dispositivo en una sola copia, además de la máscara booleana, dio 0,99× en M2. Estas alternativas se ensayaron por separado. [Resultados, dispersión, memoria y kernels](measurements.json)

## Apertura del corpus

El lanzador de recursos permite a los procesos compartir las huellas de los archivos por estado. En la apertura de la cohorte de 2021, la utilidad versionada tardó 18,94 y 18,04 s hasta validar el primer lote con las huellas por proceso. Con las huellas compartidas ya calculadas tardó 7,02 y 6,80 s. Evitó volver a calcular el SHA-256 de 3.072 archivos, con 13.428.746.691 bytes en total. Los archivos modificados se vuelven a comprobar y una caché ilegible se reconstruye.

## Reproducción

El [ensayo de apertura](../../../benchmarks/comparison_file_digests.py) recibe el manifiesto de la cohorte y el del índice de ajuste. Se ejecuta en procesos nuevos, alternando la variable `MARS_TITAN_DIGEST_CACHE` vacía y la ruta de un archivo de huellas ya poblado. El [ensayo de lectura](../../../benchmarks/comparison_input_archive.py) recibe los mismos manifiestos y un directorio de trabajo con espacio para una copia temporal. Sin `--events` recorre toda la fase y elimina la copia al terminar.

Para reproducir los tramos CUDA se necesitan los resultados locales indicados, el enlace episódico de la campaña, la revisión científica `e23a44034c8584682397d3d5f7ac76dbc7bd8e12` y PyTorch 2.14.0 con CUDA 13.0. Los [programas del ensayo](../../../benchmarks/comparison_efficiency_20261118/) se ejecutan con el mismo entorno virtual y una reserva exclusiva de GPU. La prueba de dos procesos comprueba el límite de RAM de su cgroup v2, mantiene 4 GiB disponibles para el sistema y vigila la VRAM libre. La preparación de los archivos se hace primero en CPU. El directorio de trabajo debe ser nuevo, estar fuera del repositorio y residir en disco.

```bash
repo="$(git rev-parse --show-toplevel)"
read -r -p "Ruta de la copia científica fija: " runtime
test "$(git -C "$runtime" rev-parse HEAD)" = e23a44034c8584682397d3d5f7ac76dbc7bd8e12 || exit 1
test -d "$MARS_TITAN_COMPARISON_STATE" && test -d "$MARS_TITAN_BENCHMARK_WORK"
export PYTHONPATH="$repo:$runtime/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MARS_TITAN_DECODE_WORKERS=2 MARS_TITAN_PREFETCH_BATCHES=2 MARS_TITAN_GROUP_CACHE_MIB=1536
export MARS_TITAN_RESIDENT_VALIDATION_WARMUP=1
ensayo="$repo/benchmarks/comparison_efficiency_20261118"
cd "$runtime"
CUDA_VISIBLE_DEVICES=-1 uv run --no-sync python "$ensayo/entrada.py"
export CUDA_VISIBLE_DEVICES=0
uv run --no-sync python "$ensayo/recorrido.py" --model titans_mac --kernels --flags
uv run --no-sync python "$ensayo/recorrido.py" --model mars_titan_m2 --kernels --copies --validation
uv run --no-sync python "$ensayo/paralelos.py"
```

La variable de shell `runtime` identifica la copia científica fija y `repo` la copia que contiene los programas del ensayo. `MARS_TITAN_COMPARISON_STATE` apunta al estado que contiene los checkpoints y `MARS_TITAN_BENCHMARK_WORK` al directorio de trabajo. El informe diferencia tiempos con auditoría de paridad, tiempos de cálculo tras el calentamiento, preparación y tiempo del par completo. La energía no se midió.
