# Coste CUDA de las reglas de captura

Se midieron las tres capturas sobre el predictor financiero con 256 activos reales y ocho sesiones de febrero de 2011, dentro del tramo de ajuste de fold-018. Se usaron también prefijos de 1 y 64 activos. Las entradas conservan máscaras, seis canales de precios y las dimensiones declaradas de noticias, gráficos, fundamentales y macro. No se leyeron objetivos para esta medida.

La preparación vigente terminó en 24,64 segundos y alcanzó 862.088 KiB de RSS. Su archivo de entradas ocupa 14.625.222 bytes y permanece fuera de Git. Se conservan dos rechazos anteriores: una vista antigua refería un artefacto ausente y el materializador asumía cinco canales. La [corrección de canales](materializer/README.md) tiene pruebas y mutación propias. No se modificó el productor científico congelado.

El benchmark reutiliza `benchmarks.neural_plasticity_cost` desde `4eb6b4f89`. Añade dos controles unitarios solo para comprobar paridad técnica, sin nuevos entrenamientos. Se ejecutaron 30 condiciones, cinco modelos, tres tamaños y dos precisiones, con tres calentamientos, diez trayectorias medidas de ocho decisiones y orden rotado. Los dos controles unitarios reproducen sus predicciones de inferencia bit a bit. Todos los parámetros compartidos permanecen intactos y no se construye un optimizador.

| Precisión y activos | Global, p50 ms | Local, p50 ms | Constante, p50 ms |
| --- | ---: | ---: | ---: |
| FP32, 1 | 7,922 | 8,060 | 8,084 |
| FP32, 64 | 9,076 | 8,622 | 8,791 |
| FP32, 256 | 8,366 | 8,329 | 8,214 |
| FP64, 1 | 8,081 | 8,013 | 8,165 |
| FP64, 64 | 8,509 | 8,352 | 8,344 |
| FP64, 256 | 10,887 | 11,150 | 11,222 |

Son latencias por decisión del lote completo con `DecisionBatch` residente. Incluyen el predictor y sus validaciones, no la lectura Parquet por decisión ni los codificadores que produjeron el corpus. La preparación, transferencia e inicialización se registran por separado. Los p95/p99, caudal y recorridos con grafo y backward están en los informes [FP32](financial-cost-float32.json) y [FP64](financial-cost-float64.json). El backward corresponde a la trayectoria de ocho decisiones, no a una sola.

La captura local no es siempre más rápida. Con 256 activos FP32, su p99 es 13,814 ms frente a 11,773 ms de la global, aunque sus medianas son próximas. En FP64 su mediana también es mayor en ese tamaño. Los percentiles proceden de 80 decisiones medidas por condición y no certifican latencias extremas. El escritorio permaneció activo. No se ha adoptado una optimización ni alcanzado un supuesto límite del equipo.

Cada modelo tiene 186.887 parámetros compartidos. Las tres capturas conservan los mismos bytes persistentes de tensores por flujo: 167.184 en FP32 y 334.352 en FP64. Con 256 activos, el máximo de asignación observado para la local es 867.732.480 bytes en FP32 y 1.672.100.352 en FP64. Estos picos incluyen la trayectoria diferenciable y cinco modelos residentes. No describen la memoria mínima de despliegue de un único modelo.

El proceso FP32 consumió 42,070606 segundos, de los que 39,438513 están dentro del cronómetro del benchmark. FP64 consumió 47,739080 y 45,112924, respectivamente. Sus RSS pico son 1.580.318.720 y 1.560.494.080 bytes. Los recibos externos se publican para [FP32](financial-process-float32.json) y [FP64](financial-process-float64.json). No se suman de nuevo sus tiempos inclusivos. No se midieron energía ni contadores físicos de transferencias.

El primer comprobador posterior a FP32 consultó un nombre de campo incorrecto. El trabajo numérico había terminado con código cero y el informe usa `final_test_access`. Se validó ese esquema sin repetir FP32 y se continuó únicamente con FP64. El [recibo de esa comprobación](process-postcheck.json) conserva la incidencia. La protección global no cambió.

La [sonda aislada de memoria](memory-cost.json) contiene otras 24 condiciones y 120 trayectorias con parámetros iniciales, cuatro escrituras y sin datos financieros. El [programa](memory_benchmark.py) coincide con el ejecutado. Registra 16,076038 segundos internos y conserva su configuración propia, sin normalización residual. No es el mismo recorrido que el predictor financiero ni se utiliza para afirmar una mejora de su latencia. La paridad CPU/CUDA se comprueba en las pruebas, dentro de cada precisión. Estas medidas no acreditan equivalencia predictiva entre FP32 y FP64.

Los [artefactos externos](artifacts.json), la [selección de entradas](input-selection.json) y el [catálogo técnico](financial-catalog.json) fijan la procedencia. La medida es local sobre RTX 4070 Laptop, con CUDA explícito y TF32 desactivado. El [resultado predictivo](../../research/local-capture-20261020/development-v1/README.md) es un estudio distinto y no mejora por obtener una mediana de tiempo menor en alguno de estos casos.
