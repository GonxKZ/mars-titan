# Medición de los cambios de PC2 en la RTX 4070 Laptop

El 19 de noviembre de 2026 se comparó la etiqueta de entradas verificadas (`f52cc3c33`) con los cambios de PC2 (`559bfa0f8`) en la misma NVIDIA GeForce RTX 4070 Laptop GPU, SM 8.9, con PyTorch 2.14.0+cu130 y Triton 3.8.0. El despacho publicado por PC2 selecciona SM 12.0. Para medir cada kernel en SM 8.9 se habilitó solo en el proceso de prueba. El modelo, los pesos, los índices y la configuración FP32 fueron comunes a cada pareja.

Como operaciones aisladas, la escritura de 1.024 matrices de 64 por 64 alcanzó 2,12× y la selección de ocho vecinos para 512 consultas entre 1.024 episodios alcanzó 5,48×. El microbenchmark repitió 48 veces cada operación por cronometraje. Pasaron los 25 casos CUDA del archivo de PC2 y los dos de CPU e importación. Cubren redondeo, derivadas de segundo orden, empates, valores no finitos y huellas de parámetros.

## Tramos con los modelos guardados

La tabla muestra la media de dos procesos por configuración en el origen de 2023, semilla 43. El tiempo del recorrido incluye la apertura del archivo y los cuatro segmentos de calentamiento. El tramo estable reúne las 468.708 observaciones posteriores. Los cambios portables agrupan las copias de los parámetros, evitan una segunda máscara de precios cuando el escalador ya la aplicó y difieren dos comprobaciones escalares en CUDA.

| Modelo | Opción ensayada | Recorrido completo | Tramo estable | Proceso completo |
| --- | --- | ---: | ---: | ---: |
| Titans-MAC | PC1 | 50,41 s | 20,95 s | 69,30 s |
| Titans-MAC | Cambios portables | 50,59 s | 21,22 s | 71,40 s |
| Titans-MAC | Escritura MAC | 51,83 s | 21,23 s | 72,35 s |
| M2 | PC1 | 85,44 s | 57,40 s | 105,80 s |
| M2 | Cambios portables | 86,01 s | 58,13 s | 106,80 s |
| M2 | Vecinos | 87,73 s | 58,81 s | 108,40 s |
| M2 | Escritura y vecinos | 86,91 s | 58,00 s | 107,60 s |

En M2, el pico de memoria asignada por PyTorch fue de 274,22 MiB con la referencia y de 264,28 MiB con la combinación completa. La memoria reservada por el asignador fue de 366 y 336 MiB respectivamente. Son medidas del proceso de ensayo.

Titans-MAC solo ejecutó el kernel de escritura. En M2, la pasada con ambos kernels realizó 512 lanzamientos de escritura y 504 de selección. En ambos modelos se cotejaron 127.894 emisiones y 16 conjuntos de gradientes, además de las 127.875 filas de sus respectivos archivos de validación con la combinación completa. La paridad de una combinación se informa por separado de sus tiempos.

## Kernels con la preparación original

Este control mantiene el escalador y el predictor de la versión de PC1 para distinguir el coste del kernel del resto del cambio. En el núcleo y en M2 se mide una pareja. Si el candidato mejora el tramo estable o el tiempo desde la apertura se repite en orden inverso. Las pasadas opcionales de M1 y M3 se cerraron al devolver la GPU a la cola. Los tiempos largos corresponden a Titans-MAC y M2.

| Modelo | Opción | Parejas largas | Recorrido PC1 | Recorrido de prueba | Tramo estable PC1 | Tramo estable de prueba |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Titans-MAC | Escritura MAC | 2 | 50,04 s | 51,39 s | 21,40 s | 21,40 s |
| M2 | Escritura y vecinos | 2 | 85,12 s | 86,90 s | 57,95 s | 58,25 s |

La cola de PC1 continúa con la etiqueta `comparison-campaign-packed-inputs-v1`. En las dos familias medidas, ninguna configuración de PC2 redujo de forma repetida el tiempo del recorrido completo y el del tramo estable. Se comprobó la paridad numérica de los kernels en esta GPU.

Las decisiones de rendimiento se basan en el recorrido y el tramo estable con el modelo completo. Cuando hay dos parejas se comprueba la dirección de ambas y el tiempo del proceso. Se conservan las medidas individuales, la memoria de PyTorch, el pico RSS y las opciones que solo se comprobaron en la pasada auditada. La energía y el rendimiento predictivo de mercado quedan fuera de este ensayo. [Mediciones en SM 8.9](measurements.json)

## Verificación de checkpoints existentes

Se utilizó también la función de huellas por campo publicada por PC2 en `f189d4502`. Coincidieron los 10 checkpoints iniciales con sus copias ya migradas en PC1. La comparación incluye los pesos, AdamW, los generadores aleatorios y el cursor. Se comprobó por separado el checkpoint ya guardado de la LSTM de 2023, semilla 44, en el paso 6.850. Esta lectura no ejecuta el optimizador. [Utilidad de PC2](https://github.com/GonxKZ/mars-titan/blob/f189d4502bd40b2f831e0c45d67078bd735a9979/benchmarks/comparison_efficiency_20261118/ajustes_con_checkpoint.py)

## Alcance del control de parejas de PC2

El recibo de PC2 identifica una RTX 5050 Laptop, SM 12.0. En ese equipo se midieron dos parejas cortas de modelos sobre la revisión del calentamiento. El selector consulta memoria de WSL y de Windows y la VRAM de esa GPU. La revisión de entradas verificadas utiliza un trabajador también en PC2. En PC1 se ejecutó únicamente la evaluación del selector y se conservaron los límites del lanzador de Linux. [Medidas de PC2](https://github.com/GonxKZ/mars-titan/blob/9a094635ac9b27bf6d54501c0a7a95935306fa40/reports/engineering/titans-cuda-pc2-20261119/README.md)

## Reproducción

La comparación usa los cuatro primeros segmentos como calentamiento. En la pasada de paridad se comparan las emisiones, los contadores y los 16 conjuntos de gradientes. La validación coteja el Parquet completo ya guardado cuando se indica `--validation`. En la pasada larga se recorren 68 segmentos y se miden 64 después del calentamiento, con 468.708 observaciones y 460.182 etiquetas. Los pesos se cargan desde el mismo checkpoint de 2023 con semilla 43. El optimizador del ensayo recoge los gradientes y conserva intactos los parámetros.

Desde una copia del repositorio con los dos commits disponibles, estas órdenes preparan fuera del repositorio el código numérico de la combinación. Se necesita el entorno virtual de la campaña y una copia de la referencia en la etiqueta `comparison-campaign-packed-inputs-v1`.

```bash
repo="$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT="$(uv run --no-sync python -c 'import sys; print(sys.prefix)')"
base=f52cc3c332ba358e7afc1772e2038f05505c2285
pc2=559bfa0f849b6f07241b5296e4c44d1f47c4b2c4
git fetch origin perf/pc2-tested-titans
tree="$(git merge-tree --write-tree "$base" "$pc2")"
read -r -p "Directorio nuevo de trabajo en disco: " scratch
mkdir -p "$scratch/combined" "$scratch/results"
git archive "$tree" -- src configs data/catalogs pyproject.toml | tar -xf - -C "$scratch/combined"
read -r -p "Ruta de la copia científica de referencia: " reference
test "$(git -C "$reference" rev-parse HEAD)" = "$base" || exit 1
export MARS_TITAN_BENCHMARK_WORK="$scratch/results"
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MARS_TITAN_DECODE_WORKERS=2 MARS_TITAN_PREFETCH_BATCHES=2 MARS_TITAN_GROUP_CACHE_MIB=1536
export MARS_TITAN_RESIDENT_VALIDATION_WARMUP=1
```

`MARS_TITAN_COMPARISON_STATE` señala la carpeta con los trabajos terminados y los índices, `MARS_TITAN_TRAIN_ARCHIVE` el directorio con las dos copias verificadas de 2023 y `MARS_TITAN_EPISODIC_NATIVE` el enlace compilado de la campaña. `MARS_TITAN_DIGEST_CACHE` puede señalar el archivo de huellas del estado. Se usa una sola GPU y un proceso de medida a la vez.

```bash
bench="$repo/benchmarks/comparison_efficiency_20261118/pc2_sm89.py"
cd "$reference"
PYTHONPATH="$reference/src:$repo" uv run --no-sync python "$bench" --mode reference --kind audit \
  --runtime "$reference" --model mars_titan_m2 --output "$scratch/results/m2-reference.json"
PYTHONPATH="$scratch/combined/src:$repo" uv run --no-sync python "$bench" --mode fused --kind audit \
  --runtime "$scratch/combined" --model mars_titan_m2 --validation --output "$scratch/results/m2-fused.json"
```

Cada pareja debe coincidir en `weights_sha256`, `sample_input_sha256` y `case.parity`. `launches` recoge solo kernels que se han lanzado durante la pasada auditada. Los modos `write` y `neighbors` prueban cada primitiva por separado, y `portable` mantiene el despacho original de PC2 en SM 8.9. Para cronometrar se usa `--kind timing`, una salida nueva por proceso y orden inverso en la segunda repetición. `case.pass_seconds` incluye la apertura del archivo completo, el calentamiento y el cálculo. `case.seconds` corresponde al tramo estable. El tiempo externo de proceso incluye también el arranque de Python, el cotejo de los módulos y la carga del modelo.

El control con preparación original se construye a partir de la combinación. Se restauran exactamente `src/mars_titan/models/learning_scale.py` y `src/mars_titan/models/titans/financial.py` del commit de referencia en otra copia. Se añade `--original-preparation` al modo de kernel elegido. El ensayo comprueba esos dos archivos y el resto del árbol de módulos antes de usar CUDA.

Las pruebas unitarias y el microbenchmark proceden de PC2. En la revisión `9a094635a` se puede repetir la medición de las primitivas añadiendo `--probe-device --iterations 48` a `benchmarks/titans_cuda_updates.py`. Esa opción limita el cambio de despacho al proceso de medida. [Código y recibos de PC2](https://github.com/GonxKZ/mars-titan/tree/9a094635ac9b27bf6d54501c0a7a95935306fa40/reports/engineering/titans-cuda-pc2-20261119)
