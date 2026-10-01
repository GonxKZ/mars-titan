# Construcción directa de lotes del corpus

El lector reserva los arrays del lote y copia en ellos las filas admitidas con `np.take(..., out=...)`. Sustituye las copias por muestra y el `np.stack` posterior. En la carga densa de 16.384 filas, el recorrido completo de lectura, validación y construcción baja un 32,4 % con lote 32 y un 39,4 % con lote 512. Las 60 ejecuciones comparadas conservan exactamente entradas, etiquetas, pesos, fechas, identidades, orden y cursor.

La [medición del 24 de septiembre de 2026](corpus-batches.json) compara el lector de `4f1ae0f`, con SHA-256 `fa231072bbcc3a6b1043581e57140bfc3a60f2891700be0fc355b4da74b3ace8`, y el lector modificado, con SHA-256 `77dacde1de45d4ea393e1ce2a94aa3af6d7acd36d286b1af7b6fd85d977068ef`. La [medición inicial](corpus-batches-baseline.json) permanece separada. Sus cifras no se atribuyen a la nueva versión.

## Cambio y contrato

Las ventanas de precios siguen calculándose en bloques de hasta 256 filas. Las modalidades pasan del grupo Parquet al buffer contiguo del lote. Los índices se comprueban antes de usar el modo `clip` de `np.take`, que evita el buffer adicional del modo `raise`. El trabajo sigue siendo proporcional a las filas y a la anchura de sus características, pero desaparecen los arrays intermedios de cada muestra.

Cada lote posee sus arrays. Avanzar el generador o modificar un lote posterior no cambia los anteriores. Las vistas del grupo se consumen al copiar su bloque y no quedan retenidas en los lotes entregados. El cursor confirma únicamente las filas copiadas, incluso cuando un lote cruza bloques, grupos o activos. La última tanda parcial se entrega después de comprobar los recuentos y las firmas de los archivos.

Las comprobaciones temporales y de valores finitos se aplican a las filas admitidas. Una modalidad excluida no impide consumir una muestra válida. Se conservan el orden determinista, las fechas reales de madurez y la ausencia explícita de disponibilidad cuando el origen no la proporciona.

## Medición

Se usaron un AMD Ryzen 9 8945HS, Linux x86-64, Python 3.12.14, NumPy 2.5.3 y PyArrow 25.0.1. Cada ejecución tuvo un proceso nuevo, una pasada de calentamiento con huella de salida y una pasada cronometrada. Se alternó el orden de referencia y candidato en cinco repeticiones, con un hilo numérico y lectura Arrow sin hilos. La campaña existente y las aplicaciones del equipo siguieron ejecutándose. La caché del sistema estaba caliente.

Las características tienen anchuras 384, 512, 45 y 420, más una ventana de precios de 64 × 5. La carga pequeña contiene 256 filas en cuatro activos y 2.109.745 bytes Parquet. La densa contiene 16.384 filas en 16 activos y 138.599.923 bytes. La dispersa conserva una de cada 16 filas, admite 1.024 y lee 138.356.611 bytes. El descarte no reduce las dimensiones de los grupos del origen.

La tabla recoge la mediana del lector y, entre paréntesis, la desviación típica en milisegundos. El RSS es la mediana de los picos de cada proceso, en MiB.

| Carga | Lote | Referencia, ms | Lotes directos, ms | RSS de referencia | RSS de lotes directos |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pequeña | 32 | 23,28 (0,97) | 20,07 (0,88) | 128,63 | 127,40 |
| Pequeña | 512 | 22,73 (0,57) | 20,55 (0,75) | 131,18 | 128,98 |
| Densa | 32 | 447,77 (11,60) | 302,60 (9,96) | 176,99 | 176,30 |
| Densa | 512 | 473,55 (5,75) | 287,14 (3,51) | 185,28 | 183,11 |
| Dispersa | 32 | 238,84 (6,41) | 232,44 (4,88) | 172,33 | 171,82 |
| Dispersa | 512 | 241,15 (5,77) | 235,07 (4,62) | 181,89 | 178,11 |

En la carga densa, el caudal pasa de 36.591 a 54.143 muestras/s con lote 32 y de 34.598 a 57.059 con lote 512. La suma de inicialización y recorrido pasa de 577,46 a 435,02 ms y de 601,06 a 415,89 ms, respectivamente. El tiempo total de proceso, que también incluye importaciones, calentamiento y huella de salida, pasa de 1,483 a 1,190 s y de 1,520 a 1,129 s. Las medianas de los seis casos bajan tanto en el recorrido como en el tiempo total de proceso. La diferencia del caso disperso está próxima a la variación entre repeticiones y no acredita por sí sola una aceleración clara.

El [perfil con cProfile](corpus-batches-profiles.json) registra otra pasada por carga, lote y versión, con reloj de CPU y sin mezclarla con los tiempos anteriores. En la carga densa con lote 512 desaparecen las 81.920 llamadas a `copy` y las 160 a `stack` del recorrido de referencia. `read_row_group` conserva 64 llamadas y unos 137 ms acumulados en ambos perfiles. En el candidato representa el 31,3 % del tiempo instrumentado, y `_fill_batch` el 10,9 %. En la carga dispersa, la lectura de grupos ocupa el 34,8 %. Queda trabajo de decodificación y validación del origen que este cambio no elimina.

Esta comparación mide el lector CPU completo, sin modelos ni entrenamiento. No se ha medido VRAM, transferencias CPU/GPU, energía ni coste monetario. Esos campos figuran como `null`. Los datos sintéticos permiten comparar el comportamiento del lector, pero no establecen una mejora del entrenamiento acelerado ni de la calidad predictiva. No se ha abierto el test final.

## Comprobaciones

Las cuatro baterías dirigidas pasan 72 pruebas en 10,11 s. Incluyen construcción de etiquetas y lectura, recuperación desde cursor, último lote parcial, propiedad de buffers, filas excluidas, disponibilidad temporal y un caso disperso de 12 grupos con las anchuras reales de las modalidades. Este último comprueba que las asignaciones Arrow no crecen con todos los grupos atravesados. La [mutación dirigida](corpus-batches-mutations.json) detecta cuatro cambios incorrectos sobre la misma huella del lector: fila seleccionada, disponibilidad futura, cursor confirmado y reutilización del buffer.

La [cobertura y complejidad](corpus-batches-quality.json) se obtienen con coverage.py 7.16.1 y Radon 6.0.1. Se cubren el 86,94 % de las sentencias y el 74,32 % de las ramas del lector. CRAP usa `C² × (1 − cobertura de sentencias)³ + C`. El valor más alto sigue en `_blocks`, con 47,36. Son diagnósticos de esta batería, no una comprobación exhaustiva del módulo.

Ruff, su comprobación de formato y `scripts/check_repository.py` pasan. La batería general previa se detuvo en la colección de 33 módulos por dependencias opcionales ausentes. No se ha repetido esa ejecución ni se presenta como aprobada.

## Reproducción

Desde un entorno del proyecto ya sincronizado, la comparación crea sus datos técnicos fuera del repositorio:

```bash
benchmark_dir="$(mktemp -d)/corpus"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python benchmarks/corpus_batches.py \
  --reference 4f1ae0f --repetitions 5 --output "$benchmark_dir"
```

Una pasada instrumentada sobre los mismos datos:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python benchmarks/corpus_batches.py \
  --worker "$benchmark_dir/dense/manifest.json" --batch-size 512 \
  --reference current --profile
```

Las pruebas dirigidas:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync pytest -q tests/training/test_corpus_inputs.py \
  tests/training/test_corpus_targets.py tests/training/test_price_context_blocks.py \
  tests/training/test_corpus_batches_benchmark.py
```
