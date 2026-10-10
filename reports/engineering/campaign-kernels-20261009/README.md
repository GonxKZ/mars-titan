# Núcleos y caudal del ajuste neuronal de la campaña (9 y 10 de octubre de 2026)

Este informe recoge los cambios de cálculo de la rama `perf/campaign-kernels` (#363). Su objetivo es acelerar el ajuste en la RTX 4070 Max-Q de 8 GB sin cambiar la aritmética declarada ni los hiperparámetros, y por igual en todos los brazos que comparten un camino. Cubre las referencias neuronales (RNN, LSTM, GRU, DLinear y Transformer compacto), los cuatro controles de Titans-MAC, los núcleos B y C de CM-v1, los lectores de MARS-TITAN y de CM-v1 y la GRU candidata nativa. RL y las referencias tabulares quedan fuera porque las miden otras ramas.

Ninguna medida aplica pasos de optimizador. Los entrenadores recorren su bucle real hasta el paso con un optimizador que solo cuenta llamadas y guarda gradientes. El gancho de PyTorch que prohíbe `Optimizer.step` sigue instalado y cada medida comprueba al final que ningún parámetro ha cambiado. El bloqueo de aprendizaje sigue vigente y estas cifras no dicen nada de la calidad predictiva.

## Entorno

| Elemento | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8.188 MiB, controlador 595.91.07 |
| CPU | AMD Ryzen 9 8945HS, 8 núcleos y 16 hilos, un nodo NUMA |
| Memoria | 30 GiB, compartida con otras tareas de la sesión |
| Software | Python 3.12.14, PyTorch 2.14.0+cu130, CUDA 13.0, cuDNN 9.24 |
| Energía | Perfil `power-saver` sin cambios. No se tocaron relojes, ventiladores ni límites |
| Datos | Edición v3 con máscaras desde 2000, vista US+CN `fold-012` (14.639.357 filas de ajuste y 609.090 de validación) |

Cada medida GPU se ejecutó en una plaza exclusiva de GPU, con dos hilos de CPU por trabajo. La carga media de la máquina estuvo entre 13 y 149 por las suites de otras ramas, así que las cifras llevan dispersión por tramo y se comparan siempre antes y después en las mismas condiciones.

## Criterio numérico

La orden del usuario es no inventar datos ni perder precisión. Por eso toda la campaña usa FP32 estricto: sin TF32 en cuBLAS ni en cuDNN, `float32_matmul_precision="highest"` y sin autocast. PyTorch deja TF32 activo en cuDNN por defecto, así que las recurrentes lo tenían hasta ahora. La política se declara en cada receta (`precision: "fp32_strict"`), se aplica antes de construir el modelo, entra en la identidad (`kernel_policy`) y se vuelve a comprobar al guardar cada checkpoint.

Solo se conservan cambios que dejan igual la aritmética de cada operación. Los que cambian el orden de una suma (bloques más grandes, acumulación por bloques o el orden del grafo de autograd) se contrastan con la versión anterior y con el estudio de divergencia frente a FP64, y se aceptan cuando su diferencia queda al nivel del redondeo de FP32. TF32, BF16, `torch.compile` y AdamW `fused` cambian el redondeo y quedan descartados.

## Cambios

| Commit | Familias | Cambio | Aritmética |
|---|---|---|---|
| f51c86f3 | GRU candidata | Los cuatro pesos de la GRU nativa forman un único bloque de cuDNN. PyTorch ya no los compacta en cada llamada | Idéntica bit a bit |
| 11e5050f | Todas las cabezas de cuantiles | El tensor de niveles de la pérdida pinball se crea una vez por dispositivo en lugar de copiarse al host en cada pérdida | Idéntica bit a bit |
| 90db4ef7 | Transformer compacto y codificador de precios de Titans-MAC, CM-v1 y MARS-TITAN | La última capa solo calcula consulta, atención y FFN del último token. Las capas se evalúan con sus pesos y SDPA, con la GELU exacta en todos los modos, y las comprobaciones de finitud se agrupan en una sincronización | Cambia el orden de reducción en la última capa (2,4·10⁻⁷ en la salida del codificador) |
| 9ec625ae | Titans-MAC, CM-v1, MARS-TITAN | Las comprobaciones de datos en CUDA se agrupan y se leen juntas al final de cada preparación. Las copias desde el host usan memoria fijada y no bloqueante. Los límites de bloque (4.096 flujos y 1 GiB de estado) se comparten entre núcleo, observaciones y lectores | Idéntica bit a bit |
| 83e1531a | Titans-MAC, CM-v1, MARS-TITAN, GRU candidata | Los estados de los flujos se guardan por bloques y se reúnen sin copia cuando se pide un bloque completo en orden. Las comprobaciones de cada actualización se leen con una sola copia al host. La política FP32 estricta se declara en las recetas y entra en la identidad | Idéntica bit a bit |
| 69ca6b4c | Referencias neuronales | La precisión declarada por caso se aplica antes de construir el modelo y se exige a las anclas trasladadas | Sin cambio |
| d9f54a3e | Transformer compacto | La medida de caudal y la búsqueda admiten lotes mayores de 256 con el mismo contrato que `reference_run` | Sin cambio |
| 189327f0, 00dfa822 | RNN, LSTM, GRU, DLinear, Transformer | Paso de ajuste opcional con CUDA Graphs (`cuda_graphs: true` en el caso) | Idéntica bit a bit, comprobada por familia |
| 5505103c | Titans-MAC, CM-v1, MARS-TITAN | Las recetas declaran FP32 estricto y los bloques medidos | Solo cambia el orden de las sumas del bloque |
| 5d34eb46, 2bac933f | GRU candidata | La receta declara FP32 estricto y acumulación de 128 filas. El ancla trasladada admite identidades anteriores sin precisión | Solo cambia el orden de las sumas del bloque |
| ed15e255, ecb5021c, 52cefce5 | Transformer compacto en el posentrenamiento y en las predicciones trasladadas | El padre y el ancla trasladada se reconstruyen con el lote de su identidad, como en `reference_run`. Una identidad sin lote conserva el contrato por defecto | Sin cambio |
| 895494ad, 9a104e85 | Lector cronológico y adaptadores de los lectores | El lector por bloques transforma las ventanas de precios de la v3 por tramos de 256 filas, y la repetición del núcleo en los adaptadores admite el límite común de bloque. Así los bloques de 512 y 1.024 de las recetas funcionan también con el lector real | Idéntica bit a bit |
| d429bf15 | Ejecución de la campaña | La declaración de #469 recoge la VRAM medida por brazo de Titans-MAC y por modelo de las demás familias cronológicas | Sin cambio |

«Idéntica bit a bit» se refiere a las pruebas de cada cambio y a una bisección de la rama sobre las huellas FP32 que #474 fija para las cuatro variantes financieras de Titans-MAC. En CPU y en CUDA, 90db4ef7 es el único commit que cambia sus bits: los anteriores coinciden con develop y los posteriores con 90db4ef7. Las huellas financieras de PT1 (#477) se comportan igual: su commit anterior coincide con develop y 90db4ef7 con la punta de la rama. El predictor financiero usa el Transformer compacto como codificador de precios, así que el cambio de la última capa llega también a Titans-MAC, a CM-v1 y a los lectores (ver «Validación numérica»).

Los commits a3550ff3 (estudio de divergencia), cc6aa02b, 4a4b87c7, a0276f91, 8039f4f6 y 87664661 añaden las medidas y los arneses de `benchmarks/`. 2a110fe8, 0f340331, 708adefc, 76900cd0 y dd3535e7 adaptan pruebas al lote admitido por el Transformer, a los estados por bloques y a las ventanas de la convolución causal de #474. f209a9bb y 645d5086 recapturan las huellas financieras de #474 y de PT1 (#477) después del codificador de último token. Las huellas de la memoria sola no cambian. 60f4a07f adapta la prueba de la búsqueda con máscaras al lote ampliado del Transformer. 7a2c7af4 hace que la campaña A v2 compare sus opciones de memoria con la acumulación que ya declara la receta de Titans-MAC.

Las medidas repiten eventos ya leídos y no pasan por el lector. Las pruebas con las recetas nuevas encontraron dos límites de 256 filas que la medida no podía ver, uno en el lector por bloques y otro en la repetición del núcleo de los adaptadores de los lectores. Los corrigen 9a104e85 y 895494ad.

## Cómo se mide

Las familias cronológicas (Titans-MAC, núcleos de CM-v1, lectores de MARS-TITAN y de CM-v1 y GRU candidata) se miden con su entrenador real mediante `benchmarks/chronological_replay.py`. El arnés intercepta la construcción que hace `campaign_throughput`, recorre `_train_pass` desde la mitad del tramo de ajuste y marca el tiempo en cada actualización, sin guardar checkpoints. Se descartan dos actualizaciones de calentamiento y se miden ocho tramos. La medida cuenta filas de observación por segundo y registra el pico asignado y reservado de PyTorch, la memoria del proceso según nvidia-smi y la memoria residente en el host.

Con el lector cronológico de develop, el tramo medido de US+CN pasaba entre el 83 % y el 97 % del tiempo leyendo datos, a entre 170 y 230 filas por segundo en todas las familias. Esa cifra no permite ver ningún cambio de cálculo. Por eso `benchmarks/chronological_events.py` materializó una vez en CPU los eventos del tramo (100 eventos completos desde la mitad del tramo, 248.863 filas, con la caché de bloques ampliada a 6 GiB) y cada medida los repite desde memoria con `--events`. Así la medida aísla el cálculo, que es lo que cambia esta rama. El lector lo mejora otra rama (ver «Lectura de datos»).

Antes es `develop` en 14c6b1dd con el enlace nativo de la candidata de develop. Después es esta rama en 7aedbae5, con el enlace nativo empaquetado. Ese árbol contiene los cambios de cálculo de la rama hasta 2a110fe8 y es anterior a los dos rebases sobre develop. Develop trae desde entonces cambios de #431, #435, #445, #446, #460, #469, #473, #474, #476, #477 y #481 en los mismos entrenadores, en la memoria y en las entradas (ganchos de bloque, comprobaciones de precisión, el lector, la convolución causal de la sección 4.4 y PT1, estas dos desactivadas por defecto, y las ventanas de precios de la v3.1 con su bit de presencia), que no se han vuelto a medir. La puerta de presencia de la v3.1 se aplica antes del codificador de precios en las referencias y en Titans-MAC, y el codificador de último token admite el sexto canal. Los dos árboles se miden con FP32 estricto. La receta de develop no cabe en la GPU con US+CN sin acumulación (mac_online y la candidata fallan por memoria con la fracción de 6 GiB), así que «antes» usa `accumulation_rows` 128, el valor que la receta preveía para este caso. Los bloques de «después» se pasan como opciones de receta, las mismas que ahora declaran las recetas.

| Familia y brazo | Opciones antes | Opciones después | Antes (filas/s) | Después (filas/s) | Aceleración | Pico asignado MiB | Pico reservado MiB | Proceso MiB | Horas por época de fold-012 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Titans-MAC transformer_direct | bloque 128, acumulación 128 | bloque 1.024, acumulación 1.024 | 4.320 | 9.225 | 2,14 | 818 | 890 | 1.056 | 0,44 |
| Titans-MAC mac_disabled | bloque 128, acumulación 128 | bloque 1.024, acumulación 1.024 | 3.519 | 18.526 | 5,26 | 4.715 | 5.086 | 5.252 | 0,22 |
| Titans-MAC mac_frozen | bloque 128, acumulación 128 | bloque 1.024, acumulación 1.024 | 1.826 | 15.061 | 8,25 | 2.102 | 2.198 | 2.364 | 0,27 |
| Titans-MAC mac_online | bloque 128, acumulación 128 | bloque 1.024, acumulación 1.024 | 1.336 | 8.360 | 6,26 | 5.308 | 5.764 | 5.930 | 0,49 |
| Titans-MAC mac_online (bloque 128) | bloque 128, acumulación 128 | bloque 128, acumulación 128 | 1.336 | 1.625 | 1,22 | 2.090 | 2.168 | 2.334 | 2,50 |
| CM-v1 núcleo B | bloque 128, acumulación 128 | bloque 1.024, acumulación 1.024 | 1.255 | 5.798 | 4,62 | 5.317 | 5.852 | 6.018 | 0,70 |
| CM-v1 núcleo C | bloque 128, acumulación 128 | bloque 1.024, acumulación 1.024 | 1.005 | 3.445 | 3,43 | 5.316 | 5.800 | 5.978 | 1,18 |
| MARS-TITAN M0 | bloque 128 | bloque 512 | 5.151 | 11.818 | 2,29 | 2.256 | 2.624 | 2.788 | 0,34 |
| MARS-TITAN M1 | bloque 128 | bloque 512 | 756 | 1.156 | 1,53 | 2.261 | 2.632 | 2.798 | 3,52 |
| MARS-TITAN M1 (bloque 128) | bloque 128 | bloque 128 | 756 | 792 | 1,05 | 855 | 884 | 1.050 | 5,14 |
| MARS-TITAN M3 | bloque 128 | bloque 512 | 2.869 | 4.655 | 1,62 | 2.261 | 2.632 | 2.798 | 0,87 |
| MARS-TITAN M1 K=4 | bloque 128 | bloque 512 | 644 | 935 | 1,45 | 2.262 | 2.642 | 2.808 | 4,35 |
| CM-v1 lector BCM | bloque 128 | bloque 512 | 141 | 284 | 2,02 | 2.261 | 2.634 | 2.800 | 14,30 |
| GRU candidata | bloque 128, acumulación 128 | bloque 128, acumulación 128 | 1.589 | 2.825 | 1,78 | 2.714 | 2.808 | 2.978 | 1,44 |
| GRU candidata (bloque 256) | bloque 128, acumulación 128 | bloque 256, acumulación 256 | 1.589 | 2.489 | 1,57 | 3.500 | 3.636 | 3.806 | 1,63 |

Las horas por época son las filas de ajuste de US+CN fold-012 entre el caudal medido, con el cálculo como único límite. No incluyen validación, calentamiento ni lectura.

La mayor parte de la ganancia de Titans-MAC y de los núcleos de CM-v1 viene del bloque. Con el mismo bloque de 128 que develop, los cambios de la rama dan ×1,22 en mac_online. Con 1.024, cada actualización lanza los mismos núcleos sobre ocho veces más flujos y el coste fijo de CPU por actualización (bucle de Python, lanzamientos y sincronizaciones) se reparte entre más filas. mac_disabled y mac_frozen, que no actualizan la memoria por gradiente durante el recorrido, ganan todavía más. En los lectores pasa lo mismo con 512: M1 gana ×1,05 con 128 y ×1,53 con 512. La GRU candidata gana ×1,78 con el mismo bloque de 128, por el empaquetado de cuDNN y los estados por bloques. Con bloque y acumulación de 256 rinde menos (mediana por tramo de 2.714 frente a 3.152 filas por segundo) y ocupa 828 MiB más de proceso, así que su receta se queda en 128.

El lector BCM de CM-v1 se midió con una carga media de 123 a 149 por otras suites, con tramos entre 57 y 793 filas por segundo. Una medida posterior con carga 17, con el perfilador de PyTorch activo, dio 1.124 filas por segundo con bloque 512, del orden de M1. Su aceleración de ×2,02 debe leerse con esa dispersión.

M3 se midió con las escalas estimadas sobre los 40 eventos repetidos (`BENCH_M3_EVENTS`) y no sobre todo el tramo de ajuste, porque ese recorrido previo retenía la plaza de GPU más de 30 minutos sin calcular nada. Antes y después usan las mismas escalas, así que la comparación es justa, pero esas escalas no son las de la campaña. Con bloque 512 pasa de 2.869 a 4.655 filas por segundo (×1,62) y ocupa 2.798 MiB de proceso frente a 1.052.

## Referencias neuronales

El detalle está en [references.md](references.md). Con el lote declarado de 256 y FP32 estricto, el paso de cálculo del Transformer compacto pasa de 53,8 a 59,4 mil muestras por segundo en el caso `-00` y de 44,0 a 60,8 mil en el `-10`, y su evaluación se duplica. RNN, LSTM, GRU y DLinear quedan iguales bit a bit y sin cambio de caudal fuera del ruido.

El paso con CUDA Graphs (`cuda_graphs: true` en el caso) se suma a lo anterior. Captura forward, pérdida y backward con lotes de tamaño fijo y deja fuera del grafo la lectura de las comprobaciones de finitud, el recorte, las estadísticas y el optimizador. El último lote parcial de cada época usa la ruta sin grafo, y el estado del generador se restaura tras el calentamiento para que el dropout capturado avance igual que sin grafo. Sobre 64 lotes reales de 256 de US+CN fold-012, con el bucle de `reference_run` hasta el optimizador, las predicciones y los gradientes de cada lote son idénticos bit a bit a la ruta sin grafo en los seis casos medidos. Recibo en [reference-step-graph.json](reference-step-graph.json).

| Caso | Sin grafo (ms por paso) | Con grafo (ms por paso) | Aceleración |
|---|---:|---:|---:|
| rnn-10 | 4,78 | 2,63 | 1,81 |
| lstm-10 | 5,36 | 3,72 | 1,44 |
| gru-10 | 5,27 | 3,70 | 1,42 |
| dlinear-10 | 4,06 | 1,37 | 2,96 |
| transformer-10 | 8,38 | 3,46 | 2,42 |
| transformer-11 | 11,45 | 7,92 | 1,45 |

Las cifras son medianas de cinco repeticiones de 64 pasos. El lector de las referencias entrega unas 21,8 mil muestras por segundo con un proceso y dos hilos. Con el grafo, un paso de 256 muestras tarda entre 1,4 y 7,9 ms, es decir, entre 32 y 187 mil muestras por segundo de cálculo, así que el ajuste de las referencias queda limitado por la lectura hasta que llegue la mejora de la tubería.

## Validación numérica

| Comprobación | Resultado |
|---|---|
| Estados por bloques, comprobaciones diferidas, copias fijadas, pinball y GRU empaquetada, pruebas con datos sintéticos | Predicciones, estados y gradientes idénticos bit a bit con el mismo bloque |
| Paso con CUDA Graphs en RNN, LSTM, GRU, DLinear y Transformer, con datos sintéticos, dropout, ausencias, pesos por fila, cambio de pesos entre lotes y lotes parciales | Idéntico bit a bit, también el estado del generador al final (20 pruebas) |
| Paso con CUDA Graphs, lotes reales de US+CN | Idéntico bit a bit en los seis casos medidos. Las cuatro mutaciones dirigidas (sin reasignar gradientes, sin restaurar el generador, captura con gradientes del calentamiento y sin leer la finitud) se detectan |
| Transformer de último token frente a la secuencia completa | Igual en FP64 (rtol 1e-12) y 2,4·10⁻⁷ de diferencia en FP32 |
| mac_online, bloques de 128 y 1.024 frente a develop, gradientes de 11 actualizaciones reales | Diferencia relativa global de 1,3·10⁻⁷ y 2,3·10⁻⁷, máxima por tensor de 3,3·10⁻⁶ y 7,0·10⁻⁶ |
| Lector M1, bloque 512 frente a develop | Diferencia relativa global de 2,2·10⁻⁷ |
| Lector M1, bloque 128 (el de develop) frente a develop | Diferencia relativa global de 3,0·10⁻⁶, máxima por tensor de 6,1·10⁻⁵ |
| Repetición del mismo árbol, M1 de develop y M1 con bloque 512 | Gradientes idénticos bit a bit en las 11 actualizaciones. La medida es determinista |
| Bisección de la rama con las huellas FP32 de #474 (cuatro variantes financieras de Titans-MAC, CPU y CUDA) | Solo 90db4ef7 cambia bits. Los commits anteriores coinciden con develop y los posteriores con 90db4ef7. Dos ejecuciones en CUDA del mismo árbol coinciden bit a bit |

El bloque y la acumulación no cambian el lote efectivo de una actualización (el instante completo), solo el orden de las sumas. Sus diferencias quedan en el redondeo de FP32 al principio del recorrido. El estudio de [divergencia](titans-divergence.md) muestra que la memoria con residual y LayerNorm es caótica (exponente de Lyapunov de 0,007 a 0,037 por paso), así que cualquier cambio de redondeo, incluido el de dispositivo o de bloque, lleva tras unos cientos de observaciones a otra trayectoria FP32 tan cercana a FP64 como la anterior. CPU y CUDA en FP32 se alejan de FP64 al mismo ritmo. TF32 multiplica el error a 64 pasos por unas 300 (2,2·10⁻³) y BF16 emulado por unas 4.000 (3,0·10⁻²), por eso se descartan.

Una captura en FP64 del tramo real no sirve de referencia directa porque el modelo FP64 parte de otra inicialización (su diferencia con FP32 es de 1,35 en relativo). Por eso la comparación con FP64 se apoya en el estudio de divergencia y en las huellas de #474, que se recalculan en FP64 con los mismos pesos. Para estas últimas, en CPU, la diferencia máxima relativa al mayor valor de cada tensor es:

| Variante | Gradientes, develop frente a FP64 | Gradientes, rama frente a FP64 | Gradientes, rama frente a develop | Predicciones, rama frente a develop |
|---|---:|---:|---:|---:|
| mac_disabled | 5,5·10⁻⁷ | 6,4·10⁻⁷ | 7,7·10⁻⁷ | 1,8·10⁻⁷ |
| mac_frozen | 5,7·10⁻⁷ | 4,2·10⁻⁷ | 6,6·10⁻⁷ | 0 |
| mac_online | 5,2·10⁻⁷ | 1,0·10⁻⁶ | 7,2·10⁻⁷ | 0 |
| mac_online con la memoria de la campaña | 1,2·10⁻⁵ | 5,1·10⁻⁶ | 1,7·10⁻⁵ | 2,5·10⁻⁶ |

La rama queda tan cerca de FP64 como develop, unas veces algo más y otras algo menos, y su diferencia con develop es del mismo orden que el error de FP32. Como escala, cambiar en develop de dos a uno o cuatro hilos de CPU mueve esos gradientes entre 1,3·10⁻⁷ y 2,0·10⁻⁷. En las trazas de PT1 de #477 (cuatro eventos de mac_online con la receta de 2000 y la cabeza de cuantiles), la rama difiere de develop en 8·10⁻⁸ a 1,6·10⁻⁷ en los cuantiles y en 2·10⁻⁶ a 6·10⁻⁶ en el estado y los gradientes, en CPU y en CUDA.

## Memoria y concurrencia

La fracción de memoria por proceso de `require_cuda` es de 6 GiB. El bloque de 2.048 flujos de mac_online no cabe en ella (falla por memoria), así que 1.024 es el bloque mayor que admite la receta común de Titans-MAC y de los núcleos de CM-v1. Con 1.024, mac_online ocupa 5.930 MiB de proceso y no deja sitio para un segundo trabajo en los 8 GB.

La concurrencia se midió con las recetas recomendadas sobre 40 eventos completos del mismo tramo, con dos hilos de CPU por trabajo, una copia propia del directorio de trabajo para cada proceso y con y sin el servidor MPS. Cada agregado se compara con un trabajo solo en los mismos eventos.

| Familia y bloque | Trabajos | MPS | Agregado (filas/s) | Por trabajo (filas/s) | Proceso MiB por trabajo |
|---|---:|---|---:|---|---|
| transformer_direct, bloque 1.024 | 1 | no | 19.221 | 19.221 | 1.056 |
| transformer_direct, bloque 1.024 | 2 | no | 40.440 | 20.495, 19.944 | 1.056, 1.056 |
| transformer_direct, bloque 1.024 | 2 | sí | 43.082 | 21.746, 21.336 | 1.056, 1.056 |
| transformer_direct, bloque 1.024 | 3 | no | 34.751 | 11.438, 11.576, 11.737 | 1.056, 1.056, 1.056 |
| transformer_direct, bloque 1.024 | 3 | sí | 48.258 | 16.519, 15.949, 15.790 | 1.056, 1.056, 1.056 |
| mac_online, bloque 1.024 | 1 | no | 14.804 | 14.804 | 3.528 |
| mac_online, bloque 1.024, 100 eventos | 1 | no | 15.326 | 15.326 | 5.930 |
| mac_online, bloque 512 | 1 | no | 12.018 | 12.018 | 3.196 |
| mac_online, bloque 512 | 2 | no | 18.258 | 9.306, 8.951 | 3.196, 3.196 |
| mac_online, bloque 512 | 2 | sí | 20.215 | 10.069, 10.146 | 3.196, 3.196 |
| mac_online, bloque 512, 100 eventos | 1 | no | 10.798 | 10.798 | 4.412 |
| mac_online, bloque 128 | 1 | no | 2.597 | 2.597 | 2.094 |
| mac_online, bloque 128 | 2 | no | 4.804 | 2.415, 2.388 | 2.094, 2.094 |
| mac_online, bloque 128 | 2 | sí | 5.861 | 2.937, 2.923 | 2.094, 2.094 |
| mac_online, bloque 128 | 3 | no | 6.383 | 2.123, 2.144, 2.116 | 2.094, 2.094, 2.094 |
| mac_online, bloque 128 | 3 | sí | 9.563 | 3.205, 3.195, 3.162 | 2.094, 2.094, 2.094 |
| M1, bloque 512 | 1 | no | 1.408 | 1.408 | 1.754 |
| M1, bloque 512 | 2 | no | 2.277 | 1.158, 1.119 | 1.754, 1.754 |
| M1, bloque 512 | 2 | sí | 1.972 | 993, 979 | 1.754, 1.754 |
| GRU candidata, acumulación 128 | 1 | no | 3.968 | 3.968 | 2.978 |
| GRU candidata, acumulación 128 | 2 | no | 3.049 | 1.519, 1.530 | 2.978, 2.978 |
| GRU candidata, acumulación 128 | 2 | sí | 1.934 | 979, 955 | 2.978, 2.978 |

- transformer_direct gana con tres trabajos y MPS: 48.258 filas por segundo frente a 19.221 con uno (×2,51). Sin MPS el tercer trabajo empeora el agregado (34.751 frente a 40.440 con dos).
- mac_online se queda en un trabajo con bloque 1.024. Dos trabajos con bloque 512 y MPS suman 20.215 en estos eventos, un 37 % más que uno de 1.024 (14.804), pero en los 100 eventos un trabajo de 512 ya ocupa 4.412 MiB de proceso y dos no caben en la VRAM. Tres trabajos con bloque 128 y MPS suman 9.563, por debajo de un solo trabajo de 1.024.
- El lector M1 gana con dos trabajos sin MPS (2.277 frente a 1.408, ×1,62). Con MPS rinde menos (1.972).
- La GRU candidata va sola. Dos trabajos suman menos que uno (3.049 sin MPS y 1.934 con MPS frente a 3.968) y tres agotan la VRAM.
- Las referencias neuronales se midieron sin MPS y con lotes residentes ([references.md](references.md)). RNN, LSTM y GRU ganan como mucho con dos trabajos, y DLinear y el Transformer con tres.

MPS solo mejoró el agregado en transformer_direct y en mac_online con bloques menores. En M1 y en la GRU candidata rindió menos que los mismos procesos sin MPS. El resto de brazos (mac_disabled, mac_frozen, núcleos de CM-v1, M0, M3, K=4 y BCM) no se midió en paralelo y queda en un trabajo.

El archivo [vram-throughput.json](vram-throughput.json) recoge para el planificador, por familia y brazo, el caudal de un trabajo, los picos de PyTorch, la memoria del proceso, el contexto CUDA (proceso menos reservado) y los agregados con 2 y 3 trabajos, con y sin MPS cuando se midieron. Todas las memorias están en MiB enteros.

## Perfil tras los cambios

Un perfil de dos actualizaciones de mac_online con bloque 1.024 da 1,14 s de cálculo en la GPU frente a unos 6,1 s de recorrido, así que la GPU trabaja en torno al 19 % del tiempo. El resto es CPU: el bucle de Python del entrenador (48 % del tiempo de CPU sin atribuir a operadores), los lanzamientos (8,7 %) y `fill_` (10,4 %), que rellena cada tensor nuevo porque el modo determinista de PyTorch activa `fill_uninitialized_memory`. Entre los núcleos dominan la atención eficiente en memoria de cutlass en FP32 (23 % del tiempo de GPU) y operaciones elementales. No aparece ningún núcleo cuyo cálculo justifique escribir uno propio en CUDA. La ganancia restante está en reducir trabajo de CPU por actualización.

En los lectores el límite es otro. Un perfil de PyTorch de BCM con bloque 512 da 19,8 s de CPU frente a 0,22 s de núcleos en la GPU en las actualizaciones perfiladas, y el 96 % de esa CPU queda fuera de los operadores de PyTorch. Un perfil de Python (cProfile, tres actualizaciones) localiza ese tiempo en `cm/medoids.py::_Costs._distance`, que suma 70 de los 150 s de preparación y recorrido de M1 y 98 de los 155 s de BCM. La función calcula, dimensión a dimensión, la distancia euclídea acumulada con `np.hypot` entre los episodios del banco y sus centros fijos en cada selección de representantes (`anchored_medoids._background`), y BCM añade el intercambio greedy. Una reformulación como producto de matrices cambiaría el redondeo, porque la referencia acumula con `hypot`. Hay dos opciones que conservan los bits. La primera reparte los bloques de clientes entre los dos hilos del trabajo. La segunda deja de recalcular entre actualizaciones las distancias de los pares que siguen en el banco. Las dos cambian la contabilidad de memoria que declara la selección y necesitan sus propias pruebas de paridad, así que quedan como el siguiente paso de los lectores.

## Lectura de datos

El lector cronológico de develop y la verificación de artefactos limitan hoy el ajuste real más que el cálculo:

- `_BlockReader` reabre el archivo y vuelve a decodificar el grupo de filas de cada observación cuando su caché LRU de 1 GiB no cubre todos los activos activos. En US+CN fold-012 hacen falta unos 3,2 GiB, así que el acceso cíclico deja la caché sin aciertos. Con 1 GiB se midieron 36 filas por segundo en un perfil de CPU. Con 6 GiB, tras 205 s de llenado, el mismo lector entrega unas 1.930 filas por segundo con un hilo.
- `CorpusDataset` calculaba el sha256 completo de unos 47 GB de artefactos en cada proceso nuevo, entre 23 y 25 minutos en US+CN.

Ambos se corrigen en la rama `perf/campaign-pipeline` (#469), que conviene fusionar junto a esta: caché de grupos por activo con lectura anticipada en hilos y una política que no colapsa con acceso cíclico, y la caché de huellas `MARS_TITAN_DIGEST_CACHE`, con la que un proceso abre US+CN fold-012 en unos 2 s. En `CN/fold-012` esa rama mide 5.600 filas por segundo en serie y 16.500 con cuatro hilos y lectura anticipada de cuatro instantes. Su cifra en US+CN está pendiente en #469.

## Proyección de horas

Con el cálculo como único límite, una época sobre las 14.639.357 filas de ajuste de US+CN fold-012 y una pasada por las 113.180.071 filas de ajuste de los 13 cortes costarían estas horas por brazo, con un trabajo en la GPU:

| Familia y brazo | Antes, fold-012 (h) | Después, fold-012 (h) | Antes, 13 cortes (h) | Después, 13 cortes (h) |
|---|---:|---:|---:|---:|
| Titans-MAC transformer_direct | 0,9 | 0,4 | 7,3 | 3,4 |
| Titans-MAC mac_disabled | 1,2 | 0,2 | 8,9 | 1,7 |
| Titans-MAC mac_frozen | 2,2 | 0,3 | 17,2 | 2,1 |
| Titans-MAC mac_online | 3,0 | 0,5 | 23,5 | 3,8 |
| CM-v1 núcleo B | 3,2 | 0,7 | 25,1 | 5,4 |
| CM-v1 núcleo C | 4,0 | 1,2 | 31,3 | 9,1 |
| MARS-TITAN M0 | 0,8 | 0,3 | 6,1 | 2,7 |
| MARS-TITAN M1 | 5,4 | 3,5 | 41,6 | 27,2 |
| MARS-TITAN M3 | 1,4 | 0,9 | 11,0 | 6,8 |
| MARS-TITAN M1 K=4 | 6,3 | 4,3 | 48,8 | 33,6 |
| CM-v1 lector BCM | 28,9 | 14,3 | 223,7 | 110,6 |
| GRU candidata | 2,6 | 1,4 | 19,8 | 11,1 |

Son cotas inferiores. Con el lector de develop, a entre 170 y 230 filas por segundo en US+CN, una época de fold-012 tarda entre 18 y 24 horas en cualquier familia, de modo que estas horas solo se alcanzan con la lectura de #469 y cuando la carga de la máquina no compite por la CPU. Tampoco incluyen validación, calentamiento, checkpoints ni la estimación de escalas de M3 antes del ajuste. La fila de BCM usa la medida con carga de 123 a 149. Con la del perfil (1.124 filas por segundo), una época de fold-012 bajaría a unas 3,6 horas.

## Decisiones

| Opción | Decisión | Motivo |
|---|---|---|
| FP32 estricto (sin TF32 en cuBLAS ni cuDNN, `highest`) | Adoptada en todas las familias y en la identidad | Orden del usuario. Sin coste medible en las recurrentes |
| Estados por bloques, comprobaciones diferidas, copias fijadas, niveles de pinball, GRU empaquetada | Adoptadas | Idénticas bit a bit |
| Transformer de último token con capas explícitas | Adoptada | Igual en FP64, 2,4·10⁻⁷ en FP32 y entrenamiento igual a evaluación |
| Bloque y acumulación de 1.024 en Titans-MAC y núcleos CM-v1 | Adoptada en la receta común | Mayor bloque que cabe en 6 GiB. Solo cambia el orden de las sumas |
| Bloque de 512 en los lectores de MARS-TITAN y CM-v1 | Adoptada en la receta común | M1 gana ×1,53 con 512 y solo ×1,05 con 128. Con 512 ocupa 2.798 MiB de proceso, así que deja sitio para otro trabajo. El límite de los lectores está en la selección de medoids en CPU (ver «Perfil tras los cambios»), así que un bloque mayor no se midió |
| Acumulación de 128 en la GRU candidata | Adoptada en su receta | Sin acumulación no cabe en 6 GiB con US+CN. Con 256 rinde menos (mediana de 2.714 frente a 3.152 filas por segundo) y ocupa 3.806 MiB de proceso frente a 2.978 |
| CUDA Graphs en el paso de las referencias | Adoptado como opción `cuda_graphs` por caso | Idéntico bit a bit en las cinco familias, de 1,4 a 3,0 veces más rápido |
| Bloque de 2.048 en Titans-MAC | Descartado | No cabe en la fracción de 6 GiB |
| TF32 y BF16 | Descartados | Cambian la aritmética (2,2·10⁻³ y 3,0·10⁻² frente a FP64 a 64 pasos) |
| `torch.compile` | Descartado | Cambia el redondeo (2,4·10⁻⁷) y necesita recompilar el último lote |
| AdamW `fused` | Descartado | Cambia la aritmética del optimizador. No se mide durante el bloqueo |
| Núcleos propios en C++/CUDA | No justificados por ahora | El perfil no muestra un núcleo dominante que cuBLAS, cuDNN o SDPA no cubran |
| CUDA Graphs en Titans-MAC | Pendiente | Cada instante tiene otro número de flujos y rellenar cambia las reducciones |
| Distancias de medoids en dos hilos o reutilizadas entre actualizaciones | Pendiente | Ocupan la mayor parte del recorrido de los lectores. Conservan los bits, pero cambian la memoria declarada de la selección |
| `fill_uninitialized_memory` desactivado en modo determinista | Pendiente de medir | 10 % del tiempo de CPU en el perfil. No cambia resultados si ningún núcleo lee memoria sin escribir, pero hay que comprobarlo |
| Varios trabajos por GPU | Tres con MPS en transformer_direct, dos sin MPS en M1 y uno en mac_online y en la GRU candidata | Medido con las recetas recomendadas. Varios mac_online con bloques menores rinden menos o agotan la VRAM en la ventana completa. Las demás familias quedan en un trabajo hasta medirlas |

## Opciones recomendadas en las recetas

| Receta | Opciones |
|---|---|
| `configs/titans/chronological-training-historical-masked.json` (Titans-MAC y núcleos de CM-v1) | `precision: fp32_strict`, `block_rows: 1024`, `accumulation_rows: 1024`, `predictor.max_batch: 1024`, `predictor.max_state_bytes: 134217728` |
| `configs/titans/episodic-readout-historical-masked.json` (MARS-TITAN y lectores de CM-v1) | `precision: fp32_strict`, `block_rows: 512` |
| `configs/candidate/chronological-training.json` | `precision: fp32_strict`, `accumulation_rows: 128` (bloque de 128 por defecto) |
| Casos de las referencias neuronales | `precision: fp32_strict` y `cuda_graphs: true` |

Las opciones entran en la identidad de cada trabajo y son comunes a todos los brazos que comparten receta. Los lotes y épocas declarados no cambian. Las tres recetas ya las declaran en esta rama. Los casos de las referencias las admiten, pero la sección neuronal de `campaign_plan` todavía no las propaga (ver «Pendiente»).

## Pendiente e impedimentos

Actualización del 10 de octubre. El [informe de la rama `perf/campaign-kernels-wiring`](../campaign-kernels-wiring-20261010/README.md) resuelve los puntos primero, segundo, cuarto y sexto de esta lista. La sección neuronal admite `precision`, `cuda_graphs` y lotes de hasta 4.096, y una prueba de extremo a extremo comprueba que llegan al ajuste. El padre del posentrenamiento aplica la precisión de su caso. Los módulos con huellas que dependen de los hilos fijan dos hilos durante cada prueba. Las distancias de los medoids se reducen sin cambiar sus bits (de unos 150 s a menos de 10 s por recorrido de los lectores), y CUDA Graphs en Titans-MAC, pendiente en «Decisiones», se midió en la emisión de bloques completos y no se adopta en esa rama (entre 0,98 y 1,22 veces frente al mismo cálculo sin grafo). La medida sobre la v3.1 sigue pendiente. Los puntos siguientes describen el estado del 9 de octubre.

- La sección neuronal de `campaign_plan` todavía no admite `precision` ni `cuda_graphs`, y limita el lote a 256. Hasta que el plan los acepte, las referencias de la campaña se ajustan sin ellos.
- El posentrenamiento de #446 registra las banderas de TF32 de cada ejecución, pero no fija la política que declara el caso del padre. Con esta rama el padre conserva su lote. Falta aplicarle la precisión al cargarlo, como ya hacen el ajuste y las predicciones trasladadas.
- La medida usa la edición v3. La v3.1 (#445) añade un bit de presencia por sesión a los precios. Ninguna optimización depende de la anchura de entrada, y los archivos de eventos y de lotes permiten repetir la medida en pocos minutos cuando exista la vista.
- Las distancias de `cm/medoids.py` limitan los lectores de MARS-TITAN y de CM-v1 (ver «Perfil tras los cambios»).
- Con datos reales y el mismo bloque de 128 que develop, los gradientes de mac_online y de M1 difieren de develop en 1,3·10⁻⁷ y 3,0·10⁻⁶ en relativo, mientras que dos ejecuciones del mismo árbol coinciden bit a bit. La bisección con las huellas de #474 atribuye el cambio de bits al codificador de último token (90db4ef7), que todas esas familias usan. La bisección no se ha repetido commit a commit con los datos reales.
- Las huellas bit a bit de `test_paper_projections` (#474) dependen del número de hilos de CPU. En este equipo coinciden con dos hilos y fallan también en develop con los hilos que PyTorch elige por defecto. Las huellas financieras se recapturaron con dos hilos.
- Las pruebas del grafo muestran un aviso de PyTorch: el nodo que acumula los gradientes no está en el stream del cálculo que los produce, lo que puede añadir una sincronización. La captura y la paridad bit a bit no cambian, y ese coste ya está incluido en los tiempos medidos.
- No se ha medido energía ni se ha evaluado TensorRT, que solo afectaría a la inferencia.
- Las medidas compartieron la máquina con otras suites (carga media de 13 a 149, con los picos durante las medidas de BCM) y con la cola GPU de otras ramas. Las cifras de un solo trabajo tienen una dispersión por tramo del orden del 20 al 40 %.
- Una medida auxiliar de las referencias llegó a ejecutar pasos de AdamW sobre tensores sintéticos, algo que el bloqueo prohíbe. El código y sus cifras se retiraron de la rama (cc6aa02b) y ninguna conclusión de este informe depende de ellos.

## Reproducción

- Eventos de un tramo: `PYTHONPATH=src python benchmarks/chronological_events.py VISTA TRABAJO eventos.pkl 100 6`.
- Familias cronológicas: `CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python benchmarks/chronological_replay.py VISTA TRABAJO --family titans --arm titans_mac_online --option precision='"fp32_strict"' --option block_rows=1024 --option accumulation_rows=1024 --predictor max_batch=1024 --predictor max_state_bytes=134217728 --strict --warmup 2 --segments 8 --events eventos.pkl --output salida.jsonl`. Las demás familias cambian `--family` y `--arm` y usan las opciones de su receta.
- Paso con grafo: `CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:benchmarks python benchmarks/reference_step_graph.py lotes.pt reference-step-graph.json`, con los lotes de `benchmarks/reference_kernels.py materialize`.
- Bisección con las huellas de #474: `OMP_NUM_THREADS=2 pytest tests/models/titans/test_paper_projections.py -k bit_for_bit` en cada commit, y la misma traza en CUDA con los pesos copiados del modelo de CPU.
- Referencias: los comandos de [references.md](references.md).
- Divergencia de Titans: [titans-divergence.md](titans-divergence.md).
