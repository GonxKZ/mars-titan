# Rendimiento del recorrido cronológico en PC1

Se mantiene el cálculo actual de Titans-MAC. Reutilizar la máscara causal eliminó dos kernels y clonar las filas contiguas redujo el tiempo de una selección completa. En la RTX 4070 Laptop ninguna de las dos opciones redujo de forma repetida el tiempo del tramo estable. [Mediciones por pareja y actividad de las primitivas](medidas.json)

Se usaron los pesos seleccionados del origen estadounidense de 2021, semilla 44, y datos de ajuste en FP32 estricto, desde el evento 4024. El recorrido CUDA emitió 25.495 predicciones y calculó cuatro conjuntos de gradientes con pesos fijos. Coincidieron las huellas de emisiones, gradientes, estado, parámetros y generadores aleatorios, así como el cursor y los contadores. El intervalo estable comprende las dos últimas etapas, con 12.752 observaciones y 12.624 etiquetas. Cada pareja comparte pesos y entradas. El orden se alterna entre parejas. El tiempo de verificar el archivo Arrow de 10.819.114.658 bytes queda fuera de los intervalos. [Datos del corte](medidas.json)

| Variante en CUDA | Recorrido completo, mediana base → variante | Tramo estable, mediana base → variante | Parejas con menor tiempo estable |
| --- | ---: | ---: | ---: |
| Máscara booleana fija de 6 × 6 | 1,662 → 1,550 s | 0,635 → 0,636 s | 1 de 3 |
| Clon de la selección completa | 1,772 → 1,709 s | 0,730 → 0,720 s | 2 de 3 |

La máscara fija elimina una escritura de inicialización y el kernel triangular. La selección se probó aparte sobre las 796 filas de un instante, con 5.665.132 bytes de entrada. El clon sustituye seis copias de índices desde el anfitrión y conserva una copia independiente de los valores. Las actividades CUDA, incluidas las transferencias, pasaron de 23 a 17. La mediana medida con eventos pasó de 1.233 a 223 µs por selección. La última pareja del recorrido estable tardó 0,730 s con la referencia y 0,783 s con el clon. El intervalo incluye el tiempo de despacho y las esperas entre operaciones. [Operadores y tiempos individuales](medidas.json)

La comprobación CPU usó un hilo, las primeras 32 filas de cada uno de ocho instantes y las 224 etiquetas que ya habían madurado en ese corte. Tras una pasada de paridad se midieron seis parejas de emisión y repetición con cada cambio. Las medianas fueron 746,3 → 725,4 ms con la máscara, 546,6 → 577,7 ms con el clon y 752,7 → 748,2 ms al agrupar los escalares de pérdida. Cada variante tuvo su propia serie de referencia. La agrupación conserva la suma en orden y evita la conversión individual a un escalar Python. Este corte cabe en un único grupo de acumulación. La comparación de pérdidas solo se ejecutó en CPU.

El ensayo conserva la huella `selection_v1` y la independencia del almacenamiento de los lotes seleccionados. Devolver el lote de origen rompería ambas propiedades. La máscara se deriva al construir o mover el módulo. Un buffer registrado entraría en la huella de parámetros, incluso con `persistent=False`. Las variantes se activan únicamente en el proceso del benchmark. [Opciones](../../../benchmarks/comparison_efficiency_20261118/opciones_replay.py) · [Pruebas de los contratos](../../../tests/benchmarks/test_comparison_replay_options.py)

## Alimentación de la GPU

Un decodificador de Arrow que conserva la independencia de los datos bajó de 177,23 a 165,68 ms la mediana de CPU para 16 sesiones con bloques de 1.024 filas. El percentil 95 aumentó en esa muestra. Al medir lectura, emisión y cálculo de gradientes en la RTX 4070, el tiempo total pasó de 1,561 a 1,758 s de mediana. Solo una de tres parejas favoreció al candidato. Se conserva el lector de la campaña. [Diseño, comprobaciones y límites](alimentacion.md) · [Medidas individuales](alimentacion.json)

## Alcance de un CUDA Graph en la emisión

El 10 de octubre se capturó el cálculo de un bloque de 1.024 activos, con pesos iniciales y `differentiable=True`. En tres series de 20 repeticiones, el cálculo con comprobaciones diferidas tardó entre 8,506 y 9,227 ms y el grafo entre 7,853 y 7,921 ms. Se obtuvieron los mismos bits y se conservaron 32 indicadores. Se copiaron las entradas de mercado y las salidas, pero todas las repeticiones leyeron el mismo estado inicial de la memoria. El perfil registró 14 copias entre posiciones de GPU, 74,7 MB por bloque. El informe anterior recoge también una serie en la que el grafo fue más lento que el cálculo diferido. [Mediciones de octubre](../campaign-kernels-wiring-20261010/README.md#cuda-graphs-en-titans-mac) · [Captura original](../../../benchmarks/titans_emission_graph.py)

Durante el ajuste de Titans-MAC, R2 emite con `differentiable=True` y separa el estado después de emitir. La repetición para calcular el gradiente también usa `True`. Una captura limitada a `False` sí tendría llamadas durante el calentamiento, la validación y la emisión del padre congelado de los lectores. Las filas siguientes proceden exclusivamente de los índices de ajuste. Los aciertos simulan una caché LRU de cuatro tamaños exactos del núcleo. [Entrenador](../../../src/mars_titan/training/financial_run.py) · [Emisión del padre](../../../src/mars_titan/training/mars_titan_run.py) · [Índices y recuentos](viabilidad-grafos.json)

| Origen | Sesiones | Tamaños distintos | Cobertura de las cuatro formas más frecuentes | Aciertos / fallos con LRU de cuatro formas | Bloques completos del lector de 512 / total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2019 | 4.526 | 550 | 5,39 % | 3.976 / 550 | 2.904 / 7.430 |
| 2020 | 4.777 | 582 | 5,15 % | 4.195 / 582 | 2.963 / 7.740 |
| 2021 | 5.030 | 604 | 4,99 % | 4.426 / 604 | 3.022 / 8.050 |
| 2022 | 5.282 | 619 | 5,87 % | 4.663 / 619 | 2.914 / 8.185 |
| 2023 | 5.535 | 619 | 8,42 % | 4.915 / 620 | 3.156 / 8.687 |

Ninguna sesión alcanzó las 1.024 filas. La repetición de tamaños en sesiones próximas explica el 87,8–88,8 % de aciertos teóricos del núcleo. En 2021, si cada fallo provoca una captura, el ahorro medio de 4.426 aciertos debe cubrir el coste adicional de 604 fallos. Atender solo bloques completos de 512 en el lector cubriría el 36,33 % de sus llamadas en 2023. Ambos cálculos dependen únicamente de la forma.

En tres sesiones reales del padre de M2 (2023, semilla 43) se sustituyeron en CPU las direcciones de trabajo por tensores de tamaño fijo. Al copiar en cada llamada el token, los pesos rápidos, el momentum y los pasos de cada activo, coincidieron todos los campos de las seis emisiones, el estado y el RNG. Si se reutilizaba el estado inicial, las diferencias máximas en los cuantiles de los tres bloques de 512 fueron 0, 0,051399 y 0,100672. El último contador quedó dos pasos por detrás. En otro control, sobrescribir un indicador antes de resolver las comprobaciones ocultó la primera excepción. [Resultados por bloque](viabilidad-grafos.json)

En la RTX 4070 se repitieron las mismas 2.509 filas con estado independiente por activo. El padre congelado de M2 usó el grafo en los tres bloques de 512 y la ruta ordinaria en los de 324, 324 y 325. Coincidieron todos los campos de las seis emisiones y el estado final de los generadores aleatorios. Se conservaron 32 indicadores de comprobación. Ante un desbordamiento seguido de una emisión válida, la ruta ordinaria y el grafo devolvieron el mismo primer error, «El contador de pasos desbordaría int64». [Recibo CUDA y diferencias por bloque](grafo-m2-cuda.json)

Los tres calentamientos, la captura y la sincronización sumaron 40,85 ms. Entre las lecturas anterior y posterior, la VRAM asignada pasó de 225.091.072 a 412.155.904 bytes y la reservada de 281.018.368 a 503.316.480 bytes. Son aumentos de 178,40 y 212,00 MiB, respectivamente. Los buffers de entrada ya estaban creados al tomar la primera lectura. Para cada bloque completo, la copia de los pesos rápidos, el momentum, los pasos, el token y las salidas independientes suma 67.379.200 bytes, antes de indicadores y temporales. El control comparativo duró 0,840 s y el proceso completo, 4,422 s. El contador NVML registró 204,360 J en toda la GPU durante el proceso, incluido el escritorio. Quedan por medir el caudal de M2 de extremo a extremo y el pico de VRAM de ese recorrido. [Medidas y alcance](viabilidad-grafos.json) · [Direcciones y memoria de los grafos en PyTorch 2.14](https://docs.pytorch.org/docs/2.14/notes/cuda.html#graph-memory-management)

## Reproducción

El reproductor usa una cohorte preparada, el índice de ajuste, el archivo de eventos verificado y un checkpoint terminado. Las rutas se toman de `MARS_TITAN_COMPARISON_STATE` y `MARS_TITAN_TRAIN_ARCHIVE`. `MARS_TITAN_BENCHMARK_WORK` debe señalar un directorio de trabajo existente fuera del repositorio. La salida de cada orden debe ser un archivo nuevo. Las tres parejas originales de cada comparación se midieron en un turno exclusivo de cálculo con dos hilos de CPU y 2.197.815.296 bytes de VRAM reservados. El proceso llegó a unos 2 GiB de RSS. La energía y el coste económico de esas parejas quedaron sin medir.

Los resultados principales proceden del ensayo que llamó a `recorrido.build` y `recorrido.profile_slice`. El reproductor reutiliza ambas funciones y coteja el estado en memoria. Las tres variantes se comprobaron en CPU con ocho filas y cuatro instantes. La invocación CUDA de máscara se verificó después, con una sola pareja y el mismo checkpoint. Coincidieron las 25.495 predicciones, cuatro conjuntos de gradientes, el estado y los generadores aleatorios. Esta pareja queda fuera de las medianas principales. El SHA-256 de su recibo es `d4715faa6f6e6c814475a58b791509222c3648acca68e14ad88de0ffc2a6443a`. [Comprobación del reproductor](mascara-cli-cuda.json)

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync python benchmarks/comparison_efficiency_20261118/replay_pc1.py \
  mascara "$MARS_TITAN_BENCHMARK_WORK/mascara.json"
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync python benchmarks/comparison_efficiency_20261118/replay_pc1.py \
  seleccion "$MARS_TITAN_BENCHMARK_WORK/seleccion.json" --primitivas
for opcion in mascara seleccion perdidas; do
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    uv run --no-sync python benchmarks/comparison_efficiency_20261118/replay_cpu.py \
    "$opcion" "$MARS_TITAN_BENCHMARK_WORK/cpu-$opcion.json"
done
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync python benchmarks/comparison_efficiency_20261118/replay_cpu.py \
  seleccion "$MARS_TITAN_BENCHMARK_WORK/primitivas-cpu.json" --filas 1024 --solo-primitivas
uv run --no-sync python benchmarks/comparison_efficiency_20261118/formas_emision.py \
  "$MARS_TITAN_COMPARISON_STATE" "$MARS_TITAN_BENCHMARK_WORK/formas-emision.json"
```
