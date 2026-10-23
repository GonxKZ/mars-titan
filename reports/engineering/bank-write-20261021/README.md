# Escritura neuronal con parciales por banco

`MARS-TITAN-BANK-WRITE-v1` añade `mean_partial` y `per_bank` a la memoria neuronal del predictor. Ambas almacenan tres momentum por capa y leen las mismas tres redes. La primera promedia las parciales de la pérdida de mezcla y la segunda conserva cada una. El código está confirmado en `84778ea236558e57e1640cf807d63f36501b7799`. La regla anterior `common_translation` sigue siendo el valor por defecto, con su identidad y estado anteriores.

La [especificación](../../../docs/experiments/bank-write-v1.md) detalla las ecuaciones, propiedad del estado, diferencias de escala, causalidad y límites. Las [diez configuraciones](../../../configs/experiments/bank-write-v1.json) reutilizan la receta cronológica existente y combinan rutas uniforme, constante y por consulta, réplicas y el mismo grafo regional. No se ha entrenado la variante ni medido mejora de MAE. No se activa por defecto ni redefine el modelo B histórico.

## Comprobaciones realizadas

La [regresión](verification.json) pasó 422 casos en 90,81 s. Después se añadieron dos casos adversariales, aprobados en otra ejecución de 1,19 s. No cambió el código de producción entre ambas. Son 424 casos distintos en dos tandas, incluidos 76 marcados CUDA, sin fallos, errores ni omisiones. Incluyen 87 nuevos, 38 CUDA. No es toda la suite del proyecto.

Se conserva un aviso conocido del detector experimental de sincronización de PyTorch en `test_cuda_prepare_waits_for_the_device_once[False]`. Las pasadas previas de 77, 67 y 81 casos y la revisión matemática de cuatro casos están contenidas en la evidencia final y no se suman como pruebas independientes. La primera integración nativa tuvo cuatro fallos de expectativa en M0. El test confundía etiquetas aplicadas con episodios admitidos. Se corrigió según el contrato existente: cuatro etiquetas aplicadas en M0 y M1, pero cero y cuatro admisiones. No se modificó el ejecutor para obtener ese resultado.

Las pruebas contrastan las parciales con una implementación explícita de las tres MLP, diferencias finitas con y sin memoria residual y el gradiente externo de las direcciones a través de la derivada interna. Incluyen Jacobiano escalar de orden seis y término forzado, amplificación con claves sin normalizar y diagnóstico de MAC con todos sus momentum. La cancelación de gradientes opuestos se comprueba en un estado construido. Es una propiedad del mecanismo y no un experimento financiero.

Se comprueban además RNG, parámetros iniciales compartidos, presupuestos, identidad de regla, finitud, ausencia de alias, máscaras, permutación y partición, pérdida media en la primitiva compartida y recuperación. La sesión nativa conserva la siguiente predicción y actualización tras un corte confirmado, con M0/M1 y grafo regional. Dos casos adicionales parten de pérdida finita pero gradiente desbordado y exigen rechazar la actualización, conservando intacto el estado anterior, en CPU y CUDA.

La [captura previa](baseline.json) contiene 16 condiciones pequeñas CPU/CUDA, FP32/FP64, con y sin lector y con rutas uniforme, constante y por consulta. Se capturó antes de editar y se verificaron bit a bit pesos, identidad, lecturas, estados y escritura compartida después del cambio. El RNG tampoco cambió. No demuestra compatibilidad de todas las ejecuciones históricas ni paridad de las reglas nuevas con la anterior, pues estas cambian la dinámica.

La [mutación dirigida](mutations.json) detectó 18 de 18 mutantes. Las referencias inicial y restaurada aprobaron 45 casos CPU cada una. No se contaron errores de importación, sintaxis u omisiones como detecciones. Se alteraron proyección, momentum, olvido, curvatura externa, orden de bancos, identidad, restauración, presupuesto, máscaras, contadores y finitud. No es mutación exhaustiva.

La [cobertura y CRAP](quality.json), con coverage.py 7.16.1, Radon 6.0.1 y el cálculo común del proyecto, registran 1.582/1.762 sentencias y 571/730 ramas en 156 funciones de siete módulos. `_write_banks` cubre 15/15 y 6/6, con CRAP 16. `_associative_problem` cubre 12/12 y 6/6, CRAP 13. La configuración financiera conserva rechazos sin recorrer y CRAP 470,03. Las ramas se publican aparte y no entran en el cálculo CRAP. No se afirma cobertura completa del proyecto ni de todas las combinaciones de operandos.

## Coste sobre fixtures

El [protocolo](cost-protocol.json) se fijó antes de la medida completa. El [resumen legible por máquina](cost.json) conserva 43 condiciones y sus colas, medias, dispersión y extremos. Son 796 llamadas temporizadas. Se reutilizan `raw`, `summarize` y `prepare_financial` del benchmark anterior y `trajectory` del recorrido nativo, sin modificarlos.

El equipo medido fue una RTX 4070 Laptop de 8.188 MiB, driver 595.91.07, con Torch 2.14.0+cu130 y CUDA 13.0. Se seleccionó `cuda:0`, sin alternativa CPU silenciosa, con dos hilos CPU, TF32 y autocast desactivados. Las actualizaciones neuronales usan FP32/FP64. Preparación financiera y sesión usan FP32 y dimensión 64. No hubo pruebas, mutaciones, perfiles o consultas propias en paralelo con las latencias. Se perfiló aparte un grupo de tres actualizaciones FP32 de 256 flujos por regla. No se detuvieron aplicaciones ajenas ni se atribuye exclusividad al equipo.

Los activos lógicos replican un fixture de precios, con noticias, fundamentales y macro ausentes y máscaras explícitas. Sus dimensiones son pequeñas y las etiquetas del recorrido nativo son manuales. No son historias bursátiles independientes ni el corpus FinMultiTime. Se ejecutan actualizaciones asociativas de memoria rápida durante inferencia, con parámetros compartidos congelados. No se han ejecutado pasos de optimizador externo, selección científica o evaluación del test final.

La actualización neuronal se mide desde los mismos pesos iniciales y momentum cero, con una observación por llamada. La preparación financiera usa tres observaciones previas de cada regla, validación CPU, transferencia H2D, codificadores internos, MAC, cabeza y propuesta de estado. Las 256 filas se procesan en cuatro bloques de 64. Cada condición tiene tres calentamientos y veinte repeticiones, en orden fijo. Los resultados no permiten clasificar de forma general qué arquitectura es más rápida.

Estas son medianas en milisegundos de condiciones seleccionadas de 256 flujos. Todas las demás permanecen en el JSON:

| Recorrido | Traslación común | Media de parciales | Parciales por banco |
| --- | ---: | ---: | ---: |
| Actualización neuronal FP32 | 4,476 | 5,714 | 4,645 |
| Actualización neuronal FP64 | 6,659 | 8,279 | 6,199 |
| Preparación financiera FP32, EMA y consulta | 52,360 | 54,460 | 53,957 |
| Preparación financiera FP32, réplicas uniformes | No medido aquí | 42,474 | 41,803 |
| Preparación financiera FP32, EMA, consulta y grafo | No medido aquí | 52,872 | 50,534 |

Las diferencias no son uniformes entre tamaños. Con 64 filas FP32, la actualización por banco tuvo p50 de 4,108 ms frente a 3,326 ms de la media. Con una fila de preparación financiera, los p50 fueron 8,721 y 9,156 ms, pero sus p99 fueron 11,791 y 10,699 ms. No se concluye una aceleración general ni se elige la mejor condición como resultado único.

El estado neuronal FP32 de 256 flujos pasa de 33.556.480 bytes con la regla anterior a 50.333.696 con las nuevas. Las matrices aumentan un 50 %, aunque el contador de pasos no cambia. Con proyecciones completas y ventanas, el estado financiero pasa de 34.148.352 a 50.925.568 bytes. Media y parciales almacenan los mismos tensores. No tienen necesariamente la misma capacidad efectiva ni norma de escritura.

El pico asignado por Torch en la actualización neuronal FP32 fue 155.739.136 bytes con la traslación común, 231.236.608 con la media y 222.848.000 con parciales. En preparación financiera de 256 filas fue 141.859.328, 185.901.568 y 183.804.416 bytes. Incluye modelo, estado, resultado y temporales del proceso, no solo los pesos de memoria. No es la VRAM total.

El [perfil de tres actualizaciones](profiles.json) registra 45 llamadas `bmm` en las tres reglas. La traslación común tiene 3 clones, 462 lanzamientos y 81 sincronizaciones de stream. La media tiene 21, 510 y 105, y las parciales 21, 498 y 105. Son atributos del profiler del framework, sin contadores físicos de ocupación, caché o energía. Las copias y comprobaciones adicionales quedan identificadas para una revisión posterior. No se ha adoptado una optimización por estos recuentos ni se atribuye todo el tiempo a una sola operación.

## Sesión completa, límites y ejecución incompleta

El recorrido nativo incluye apertura, calentamiento, tres emisiones, feedback anterior, cierre, escritura, lectura de trazas, contabilidad de disco, poda y recuperación de la fase cerrada. Usa M1, capacidad episódica 64 y K1, sin lector episódico, PT4, PT2 o selector activos. Mantiene dos consumidores residentes. Se alternan las reglas por repetición, con un calentamiento y cuatro medidas por condición. No se suma este tiempo con el de sus partes anidadas.

| Flujos | p50 media, ms | p50 por banco, ms | p95 media, ms | p95 por banco, ms |
| --- | ---: | ---: | ---: | ---: |
| 4 | 1.895,090 | 1.739,238 | 1.937,282 | 1.878,270 |
| 256 | 16.862,366 | 17.489,455 | 17.587,361 | 18.017,024 |

En 256 flujos, el p99 fue 17.680,784 y 18.074,687 ms. La mediana por banco fue aproximadamente un 3,72 % mayor. Cuatro repeticiones solo ofrecen colas descriptivas. La menor latencia de un componente no se convierte automáticamente en menor coste de sesión.

El pico Torch de esa trayectoria fue 135.212.032 bytes con la media y 133.114.880 con parciales, con ambos consumidores presentes. Los bytes retenidos fueron 53.936.338 y 53.935.058, estables entre repeticiones. Los metadatos de las reglas tienen diferente tamaño. No se interpreta la diferencia como compresión de la memoria ni como reducción general de RAM.

El primer intento se detuvo con salida 1 al preparar 256 flujos nativos. La receta limita `max_batch` a 128 y el consumidor aplica ese límite a la cohorte completa. Tener bloques GPU de 64 no lo elimina. El JSON conserva 41 condiciones y estado incompleto. Su fuente anterior se conserva y coincide con el SHA registrado antes de ejecutar.

La medida completa utiliza un preset nativo con `max_batch=256` en las dos configuraciones, `block_rows=64` y el mismo presupuesto de estado de 128 MiB. Tiene su propia identidad. No se modificó el guardia. La receta de entrenamiento y la preparación financiera por bloques conservan 128. El componente neuronal aislado declara 256 desde el primer intento, en sus propias configuraciones. La ampliación del preset nativo se registró antes de repetir toda la medida. No se presenta el intento incompleto como otra pasada completa ni se promete admitir universos mayores con ese límite.

El [coste de proceso](process-cost.json) fue de 215,84 s internos y 219,91 s externos para la pasada completa. El pico RSS Python fue 1.766.760.448 bytes y el externo 1.758.644 KiB. Incluye preparación, perfilado y limpieza. El intento rechazado tardó 44,99 s externos y se conserva aparte. No se midió energía ni coste monetario, backpropagación externa, selección de modelos o entrenamiento de campaña.

Con el entorno uv del proyecto y su extensión episódica nativa configurados:

```bash
evidence_dir="$(mktemp -d)"
uv run --no-sync python reports/engineering/bank-write-20261021/benchmark.py --output "$evidence_dir/cost.json" --storage-root "$evidence_dir"
```

Las muestras temporizadas, perfiles completos, XML, cobertura, referencia previa y fuentes están fuera de Git. [external-artifacts.json](external-artifacts.json) registra tamaños y huellas. La [auditoría de fuentes](source_audit.md) distingue antecedentes, derivaciones y comprobaciones propias. Siguen pendientes la utilidad predictiva, la especialización aprendida y los contrastes con datos reales. La campaña completa y el test final permanecen cerrados.
