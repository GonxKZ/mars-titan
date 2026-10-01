# Rendimiento del recorrido RL nativo

Con 65.536 transiciones, la mediana del tiempo de proceso de PPO pasó de 39,516 a 30,768 segundos y la de Double DQN de 170,106 a 155,437 segundos. Son reducciones del 22,1 % y el 8,6 % en dos pares de ejecuciones. En la carga de 8.192 transiciones, Double DQN empeoró un 19,0 %. Esa regresión forma parte del resultado y limita cualquier afirmación general sobre la mejora.

El [informe JSON](adaptive-pipeline-performance.json) conserva las repeticiones, dispersión, tiempos internos, RSS, tamaños de archivos y huellas de los artefactos. Estas medidas comparan coste de ejecución sobre datos sintéticos. No evalúan rentabilidad ni calidad de aprendizaje.

## Condiciones de medida

Se utilizó una RTX 4070 Laptop GPU con 8.188 MiB, driver 595.91.07, un Ryzen 9 8945HS y 16 CPU lógicas. Los recibos registran Python 3.12.14, Linux 7.0.0-34-generic y LibTorch 2.14.0. La identidad conservada de la versión optimizada confirma Clang 21.1.8, C++20, Release, CUDA 13.0 y LibTorch 2.14.0+cu130, con `-O3`, `-fno-fast-math` y `-ffp-contract=off`.

La semilla de política fue 42. El catálogo disponible contiene 256 mundos de entrenamiento y 128 de validación, con 16 activos, 256 sesiones y 11 campos de contexto por mundo. Los Parquet suman 75.873.867 bytes. Se usaron 16 entornos, recorridos de 1.024 observaciones, minibatches de 64 y cuatro épocas PPO. La entrada de las políticas medidas tiene 207 componentes. La forma del recorrido es `[64, 16, 207]`. GRU utiliza secuencias de 16. Una ejecución corta no implica recorrer todo el catálogo de entrenamiento.

Cada repetición arranca un proceso nuevo. La herramienta incorpora un calentamiento por caso y binario y alterna referencia/candidata con candidata/referencia. El calentamiento se registra y queda fuera de las estadísticas. No se vacían las cachés. Los límites de OpenMP, MKL y OpenBLAS son de un hilo. La admisión del lanzador permite una carga científica GPU a la vez.

La CPU se compartió con otras tareas. Parte de la primera comparación coincidió con pruebas CPU instrumentadas. Las series posteriores conservan capturas de carga, que no son seguimiento continuo. El perfil `power-saver` se observó a las 09:49:21 UTC del 29 de septiembre, después de los benchmarks. Ese registro no demuestra que fuese constante durante cada prueba.

## Comprobaciones de finitud

La primera comparación evaluó la reducción de trabajo redundante al comprobar valores finitos. Sus familias se intercalaron en orden alfabético y el checkpoint se guardaba cada 1.024 transiciones. Hubo calentamientos en barridos previos separados, no dentro de la serie intercalada.

Las cifras son mediana ± desviación típica muestral, en segundos. La desviación no es un intervalo de confianza.

| Variante | Pares | Referencia | Candidata | Cambio de tiempo |
|---|---:|---:|---:|---:|
| PPO | 3 | 6,004 ± 0,053 | 5,852 ± 0,036 | −2,5 % |
| Double DQN | 5 | 21,495 ± 5,260 | 18,368 ± 5,101 | −14,6 % |
| PPO GRU | 3 | 8,431 ± 0,167 | 8,122 ± 0,195 | −3,7 % |
| PPO con memoria y HMM | 3 | 7,490 ± 0,073 | 7,501 ± 0,178 | +0,2 % |

La variante con memoria y HMM quedó prácticamente igual. La dispersión de Double DQN fue amplia. La comprobación de paridad registrada conserva parámetros y 11.264 filas de traza exactos en las cuatro variantes.

## GAE en CPU y publicación del índice de trazas

La siguiente versión evita calcular GAE cuando no hay muestras válidas. Para las políticas MLP calcula GAE y selecciona las filas válidas en CPU mediante ATen. GRU conserva su recorrido y Double DQN no utiliza GAE. El escritor de trazas acumula los fragmentos confirmables y difiere la publicación del índice hasta el checkpoint. La publicación del índice completo sigue ocurriendo en cada checkpoint.

Esta comparación usa el orden de familias del coordinador y checkpoints cada 2.048 transiciones. Por esas diferencias de protocolo, sus razones de tiempo no se encadenan con las de la tabla anterior. La referencia de esta serie ya incluye la optimización de finitud.

| Transiciones | Variante | Referencia (s) | Candidata (s) | Cambio de tiempo |
|---:|---|---:|---:|---:|
| 8.192 | PPO | 8,435 ± 0,735 | 7,535 ± 0,230 | −10,7 % |
| 8.192 | Double DQN | 21,866 ± 4,198 | 26,031 ± 1,733 | +19,0 % |
| 8.192 | PPO GRU | 11,092 ± 0,420 | 10,744 ± 0,466 | −3,1 % |
| 8.192 | PPO con memoria y HMM | 9,234 ± 0,258 | 8,487 ± 1,104 | −8,1 % |
| 65.536 | PPO | 39,516 ± 0,077 | 30,768 ± 2,380 | −22,1 % |
| 65.536 | Double DQN | 170,106 ± 5,735 | 155,437 ± 3,861 | −8,6 % |

La carga corta tiene tres pares por variante y la larga dos pares A/B y B/A. Los pares comprobados conservaron exactamente los parámetros y las trazas, con 11.264 y 88.064 decisiones observadas, respectivamente. Se conserva la versión candidata para las cargas posteriores por el resultado del recorrido largo, con la regresión corta de Double DQN registrada.

El JSON separa las fases de preparación, entrenamiento, evaluación y checkpoint. La mediana de entrenamiento interno de PPO largo bajó de 32,549 a 23,858 segundos. La publicación diferida también cambia la fase en la que se contabiliza trabajo de trazas, por lo que esos tiempos no aíslan el efecto de cada cambio.

El RSS es el máximo del proceso nativo registrado por `VmHWM`. En la serie corta varió aproximadamente entre 1,23 y 1,39 GiB. Al cerrar la carga larga, la candidata conservaba una mediana de 15,86 MiB de archivos para PPO y 34,02 MiB para Double DQN. La retención comprende dos checkpoints recientes y el mejor si es distinto, además de las trazas y los recibos. Estos tamaños no son E/S física ni el total acumulado de bytes escritos.

## Número de trabajadores

Se probaron 1, 2, 4 y 8 trabajadores con la candidata, 8.192 transiciones, un calentamiento y tres repeticiones por caso. La tabla recoge medianas de proceso en segundos.

| Variante | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| PPO | 7,182 | 7,934 | 7,834 | 7,733 |
| PPO con memoria y HMM | 9,238 | 10,290 | 9,639 | 9,486 |

Ninguna mediana mejoró al aumentar los trabajadores. Se mantiene uno. En ambas variantes, los valores 2, 4 y 8 conservaron parámetros y 11.264 filas de traza exactos frente a uno. Esta prueba no mide cómo escalan todas las arquitecturas o tamaños posibles.

## Perfil del recorrido optimizado

Los perfiles de Nsight Systems comparan la versión de finitud con la versión que añade GAE CPU y el cambio de trazas. Usan otra carga: 1.024 transiciones de aprendizaje, 2.048 decisiones observadas, recorridos de 256, checkpoints cada 256 y ocho mundos de validación. Hay una ejecución instrumentada por variante y binario.

| Variante | Instancias de kernels antes → después | `cudaStreamSynchronize` antes → después |
|---|---:|---:|
| PPO | 39.348 → 35.940 | 4.003 → 3.863 |
| Double DQN | 162.092 → 162.092 | 11.165 → 11.165 |
| PPO GRU | 56.647 → 56.647 | 5.403 → 5.403 |
| PPO con memoria y HMM | 39.348 → 35.940 | 4.003 → 3.863 |

En las dos MLP, las copias de CPU a GPU pasan de 1.243 a 1.191 y de 10,349 a 9,446 MB exportados. Las copias de GPU a CPU pasan de 2.760 a 2.672, con 1,499 MB en ambas versiones al redondeo del exportador. Los recuentos de Double DQN y GRU permanecen iguales en estos perfiles. No se calcula aceleración de proceso a partir de tiempos instrumentados.

La [recuperación CUDA](adaptive-cuda-recovery-20260929.json) de la versión optimizada conservó parámetros y trazas exactos en las nueve variantes al pausar en 512 y continuar hasta 1.024 transiciones. Las dos auxiliares completaron cuatro pasos y 256 muestras de consolidación. La comprobación final amplió el recorrido a 4.096 transiciones, con pausa en 2.048 y cambio de episodio. Las nueve variantes volvieron a coincidir. El informe conserva las comprobaciones anteriores como historial.

No se han medido energía, pico de VRAM ni contadores físicos de E/S. Las repeticiones son pocas y las condiciones no acreditan un límite de rendimiento del hardware. La [herramienta pública](../../scripts/benchmark_adaptive_rl.py) conserva configuración, versiones y medidas por proceso para nuevas comparaciones.

Las tres series generadas con la herramienta pública conservan la huella del harness que las produjo. Esa revisión verificaba los Parquet al inicio, pero no volvía a comprobarlos después de cada proceso. La comprobación posterior de sus hashes se añadió tras estas mediciones, fuera del intervalo cronometrado. No se atribuye esa protección a las ejecuciones anteriores.
