# Precomputación opcional de KronLSTM

La precomputación escalar cumplió el criterio técnico previo en el recorrido completo de `FinancialPredictor.prepare`. Con 256 filas, su mediana de inferencia pasó de 21,034 a 19,182 ms, una reducción del 8,81 %, y ganó cuatro de cinco parejas. El control vectorial no cumplió el criterio y se rechaza como opción financiera. La referencia sigue siendo el backend predeterminado. Estas medidas no evalúan calidad predictiva.

El [resumen](summary.json) conserva las ocho celdas, las ochenta muestras, sus órdenes, caudal, tiempos CUDA, memoria y huellas. La versión medida está archivada en [`6a56f194c`](https://github.com/GonxKZ/mars-titan/commit/6a56f194c62e2ab8d77a4bc4ea24d84f5214c3a4). El HEAD registrado al medir era `0e8d6359c1b9c1bd0bf130f2a8d882d66ec4c680`, con cambios locales fijados por manifiesto. Los 551 casos históricos corresponden a ese prototipo. Después de reducir el contrato financiero a la opción escalar, la regresión final pasó 545 casos y Window acumuló 57 casos únicos sobre fixtures en dos corridas separadas de 56 y 1. No se suman los 551 casos anteriores al recuento final.

## Mecanismo y comparación

El operador precomputa `exp(g)`, `q * scale` y los términos de puertas independientes del estado sobre toda la ventana. Conserva la recurrencia y las comprobaciones de revisión, estado y salida en cada paso. No añade un kernel ni elimina diagnósticos. El operador genérico mantiene las reglas escalar y vectorial para contrastarlas, aunque la opción financiera vectorial precomputada queda descartada por coste.

Se compararon los pares `kronlstm_v1`/`kronlstm_precomputed_v1` y `vector_delta_v1`/`vector_delta_precomputed_v1` con pesos idénticos, 81.325 parámetros por modelo y entradas sintéticas comunes de cinco modalidades con máscaras. El predictor usa `transformer_direct`, dimensión 32, una capa, cuatro cabezas de ancho 8, contexto 64, semilla 42 y lotes de 32/256. La memoria temporal se reinicia por ventana. No se utilizaron objetivos financieros.

El equipo fue una NVIDIA GeForce RTX 4070 Laptop GPU, con PyTorch 2.14.0+cu130 y CUDA 13.0. Se fijaron FP32 estricto, `cuda:0`, TF32 y autocast desactivados, algoritmos deterministas, dos hilos CPU y uno de interoperabilidad. El límite del asignador fue 1 GiB, con 256 MiB adicionales de margen libre exigido. Había una carga externa de inferencia que ocupaba 5.438 MiB y tres procesos gráficos con 100 MiB en conjunto. La concurrencia se contrastó antes y después, sin vigilancia continua ni interrupción de aplicaciones.

Cada variante tuvo un calentamiento y cinco medidas. El orden alternó referencia/candidato y candidato/referencia. Inferencia usa `no_grad` y `differentiable=False`. La otra condición añade la pérdida sintética `mean(point_predictions**2)`, backward y revisión de finitud de los gradientes. Las muestras incluyen `zero_grad`, preparación completa y sincronización CUDA. Construcción, transferencias, hashes, comparaciones y auditorías quedan fuera de la muestra y dentro del tiempo total.

## Resultados completos

Los tiempos son mediana y rango observado en milisegundos. El cociente divide la mediana precomputada por la de referencia. La última columna cuenta parejas en las que el candidato fue más rápido.

| Regla | Filas | Recorrido | Referencia, ms | Precomputed, ms | Cociente | Parejas favorables |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Escalar | 32 | Inferencia | 22,531 [21,320, 23,365] | 19,416 [18,749, 23,847] | 0,861746 | 4/5 |
| Escalar | 32 | Forward + backward | 80,681 [78,881, 87,075] | 66,563 [65,080, 69,397] | 0,825009 | 5/5 |
| Escalar | 256 | Inferencia | 21,034 [20,518, 23,592] | 19,182 [18,559, 21,768] | 0,911939 | 4/5 |
| Escalar | 256 | Forward + backward | 75,442 [62,220, 85,322] | 64,823 [56,258, 67,071] | 0,859244 | 5/5 |
| Vectorial | 32 | Inferencia | 14,611 [14,132, 15,818] | 13,985 [13,486, 14,549] | 0,957146 | 5/5 |
| Vectorial | 32 | Forward + backward | 54,056 [48,811, 58,184] | 48,942 [47,295, 58,631] | 0,905396 | 4/5 |
| Vectorial | 256 | Inferencia | 15,142 [14,411, 16,867] | 15,162 [14,668, 16,816] | 1,001324 | 2/5 |
| Vectorial | 256 | Forward + backward | 51,638 [49,813, 58,099] | 55,276 [45,638, 67,092] | 1,070450 | 2/5 |

El criterio fijado antes de medir exigía un cociente máximo de 0,95 en inferencia con B256, al menos cuatro de cinco parejas favorables, ningún cociente superior a 1,03 en las demás celdas, aumento del pico incremental asignado no mayor de 16 MiB y todos los controles de paridad e invariantes aprobados. El escalar cumplió esas condiciones. El vectorial obtuvo un cociente de 1,001324 y dos de cinco parejas en la celda principal, además de una regresión del 7,04 % con backward y B256. Se conserva ese resultado negativo sin elegir solo sus celdas favorables.

Con B256, el caudal mediano de inferencia escalar pasó de 12.170,84 a 13.346,11 filas/s y el de forward con backward de 3.393,35 a 3.949,23 filas/s. El vectorial pasó de 16.906,55 a 16.884,19 filas/s en inferencia y de 4.957,54 a 4.631,27 filas/s con backward. Son exposiciones repetidas de la fixture.

Cinco muestras por variante permiten describir estas medianas y rangos. No sostienen una estimación fiable de p95/p99, intervalos de confianza ni aceleración universal. Los tiempos del stream CUDA incluyen huecos entre lanzamientos y no equivalen a tiempo exclusivo de kernels.

## Memoria, proceso y verificación

Los máximos absolutos asignados por Torch variaron entre 49.039.360 y 270.932.992 bytes entre celdas. Los reservados variaron entre 62.914.560 y 322.961.408 bytes. En cada celda coincidieron los máximos de referencia y candidato y la diferencia del máximo incremento asignado fue cero. Esto no demuestra que todos sus temporales coincidan. El resumen conserva baseline, pico absoluto e incremento de allocated y reserved de cada muestra, sin sumarlos.

El mayor RSS muestreado fue 1.474.699.264 bytes. El máximo de vida registrado mediante `getrusage` fue 1.473.949.696 bytes y el máximo externo de GNU time fue 1.589.440.512 bytes. Son registros de distinto alcance, afectados también por imports y auditorías. No se suman ni se atribuyen solo al modelo. El muestreo de RSS puede perder picos entre observaciones.

La ejecución completó 96 preparaciones, incluidas 16 de calentamiento, 48 backward y 13.824 exposiciones a filas, con cero intentos de optimizador. Duró 17,713 s internamente y 18,72 s como proceso externo. Las comparaciones de predicciones y gradientes registraron error absoluto máximo cero en estas celdas, con tolerancias previas de `rtol=1e-5, atol=1e-6` para salida y `rtol=2e-5, atol=2e-6` para gradientes. Entradas, RNG, fuentes, pesos y protección global pasaron sus comprobaciones. Se verificaron también los pasos observados y los cortes de los estados inicial y devuelto.

La regresión final del contrato escalar pasó 545 casos, sin fallos, errores ni omisiones, en 17,957 s supervisados. El operador vectorial de bajo nivel se conserva, pero su identificador financiero precomputado se rechaza expresamente. La pasada instrumentada final con Coverage 7.16.1 y Radon 6.0.1 registra 57/57 sentencias y 20/20 ramas en `precomputed.py`. La función recurrente tiene complejidad 21 y CRAP 21. La guarda `CompactPriceKronLSTM._check_execution` tiene complejidad 8 y CRAP 8, con 3/3 sentencias y 2/2 ramas cubiertas. El conjunto de los tres archivos Kron registra 413/436 sentencias y 139/160 ramas, con CRAP máximo 44,0058. Por tanto, la cobertura completa del archivo nuevo no se extiende a todo el núcleo.

Window pasó primero 56 casos y las seis rutas previas con fixtures en 65,321 s supervisados. Después pasó un caso nuevo que carga la receta escalar precomputed y adapta solo las fechas, una época, lote y límites de la fixture. Este caso confirmó tres actualizaciones físicas AdamW mediante espía y checkpoint, seis filas de ajuste, cuatro de validación, sus IDs y el estado recuperable de modelo, optimizador, RNG, selección y cursor. Duró 4,91 s según la salida de pytest y 7,634 s como proceso supervisado. Son 57 casos únicos en dos corridas separadas, sin afirmar una pasada conjunta. Ambas autorizaciones técnicas quedaron cerradas y la protección global permaneció intacta. Los 545 casos de regresión no ejecutaron optimizadores. Las actualizaciones de la fixture no constituyen un ajuste con datos financieros reales.

La convención CRAP es `v²(1−c)³+v`, con complejidad Radon y fracción de sentencias cubiertas dentro del rango de función. Las ramas se informan aparte. Estas cifras son diagnósticos, no garantías de ausencia de defectos.

Dos mutaciones dirigidas se detectaron con original G1 aprobado, fallo CALL de tipo `AssertionError` en M1 y restauración G2 aprobada. P1 multiplica indebidamente por beta la escritura desacoplada. P2 elimina el chequeo del estado actualizado y altera la secuencia exigida de veinte diagnósticos. Las seis fases terminaron sin errores ni omisiones. El ejecutor registró 25,449 s, con originales y protección global intactos y cero intentos de optimizador. No se suman como nuevos casos únicos ni acreditan resistencia a otras mutaciones.

La [receta mensual independiente](../../../configs/experiments/kronlstm-precomputed/window-month.json) declara ajuste en enero de 2015 y validación en febrero de 2015, con extremos finales exclusivos. Solo cambia `temporal_encoder` a `kronlstm_precomputed_v1` frente al [control Kron](../../../configs/experiments/window-variants-pilot-month-v3/kronlstm-bp-adamw.json). Conserva BP, AdamW, MAE, dos épocas, lote 32, selección y límites. No se ha ejecutado sobre la ventana real. La adaptación técnica con fixture corresponde al caso adicional aprobado. Las seis recetas previas siguen intactas.

La medición no incluye datos reales, ajuste, sesión durable, recuperación de checkpoints, coste energético ni MAE financiero. Las seis recetas de ventanas conservan su backend de referencia. La campaña completa, la reanudación automática y el test final siguen cerrados.
