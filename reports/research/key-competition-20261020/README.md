# Comprobaciones de centrado y competencia de claves

La [implementación](../../../docs/research/key-competition.md) dispone de siete configuraciones independientes y desactivadas por defecto. El [protocolo](../../../docs/experiments/key-competition-v1.md) separa el piloto predictivo de estas comprobaciones técnicas. Los resultados predictivos se incorporan cuando termine el piloto, sin atribuir una mejora a estas pruebas.

La [verificación](verification.json) recoge 399 casos distintos aprobados. Los dos casos inicialmente omitidos se ejecutaron después con fixtures y no quedan como omisiones pendientes en esta selección. Se comprobaron CPU y CUDA, FP32 y FP64, ecuaciones de escritura y consulta, gradientes, recuperación, RNG, cuatro actualizaciones del control completo y el análisis integral con archivos simulados. No es una ejecución de toda la batería del repositorio.

Las 18 mutaciones dirigidas fueron detectadas y cada grupo pasó de nuevo al restaurar sus fuentes. Esta cifra cubre los mutantes definidos en el recibo, no todos los fallos posibles. La [medida de calidad](quality.json) usa coverage.py 7.16.1 con ramas y Radon 6.0.1, con la convención de CRAP registrada por función. Los dos módulos nuevos cubren sus 48 sentencias y 22 ramas, y su CRAP máximo es 14. La selección completa cubre 1.597 de 1.928 sentencias y 554 de 756 ramas. Conserva zonas complejas poco cubiertas, incluida la validación general de FinancialConfig. No se afirma una cobertura integral.

[KeyCompetition.lean](../lean/KeyCompetition.lean) comprueba seis propiedades del centrado de pares reales, incluida la ausencia de una contracción uniforme por factor ½. Lean 4.35.0, commit eb6cafb7290e8bd5f2937522d8d804f58f02bafc, aceptó el archivo con Mathlib fijada por el manifiesto existente. El control negativo que intentaba demostrar esa contracción fue rechazado. El recibo incluye salidas, dependencias y axiomas estándar. Se confía en el binario y las dependencias precompiladas instaladas. No se verifican la selección, normalización posterior, coma flotante o red completa.

## Coste CUDA con entradas precodificadas

La [medida](cuda-cost.json) procede del commit 902fe6c6d, con RTX 4070 Laptop de 8 GiB y PyTorch 2.14.0+cu130, FP32 estricto, sin TF32. Son ocho decisiones, diez recorridos medidos y dos de calentamiento por configuración y tamaño. Incluye tamaños 1, 64 y 128. La tabla muestra latencias por decisión con 64 activos y caudal de filas de activos por segundo. Los p99 se estiman con 80 decisiones medidas por caso y tienen esa resolución limitada.

| Configuración | p50 ms | p95 ms | p99 ms | Filas/s |
| --- | ---: | ---: | ---: | ---: |
| original | 5.963 | 7.174 | 10.655 | 10822.7 |
| all_components | 6.029 | 7.271 | 7.863 | 10795.9 |
| centered | 6.114 | 7.172 | 7.548 | 10612.4 |
| competitive | 6.311 | 8.138 | 10.037 | 10119.4 |
| centered_competitive | 6.372 | 7.518 | 8.041 | 10390.4 |
| fixed | 6.461 | 8.142 | 8.399 | 10087.1 |
| centered_fixed | 6.129 | 7.660 | 8.466 | 10290.6 |

Los siete modelos tienen 186.823 parámetros compartidos y 67.856 bytes de estado tensorial por flujo. No hay ahorro persistente por seleccionar coordenadas. El proceso completo medido tardó 40,895 s y su pico de RAM fue 1.598.603.264 bytes. El recibo incluye memoria CUDA por recorrido, preparación y transferencia, derivación de una pérdida artificial sin optimizador y todas las observaciones temporales. Los modelos de cada tamaño residen simultáneamente, por lo que sus picos no describen una aplicación con un único modelo residente.

La decisión medida parte de DecisionBatch residente e incluye prepare, validaciones y actualización neuronal. La inicialización y transferencia se informan por separado. No incluye la lectura Parquet por decisión ni los codificadores anteriores al corpus. No se midieron energía, contadores internos de kernels o bytes físicos transferidos. Había escritorio gráfico y comprobaciones CPU breves concurrentes. No es una medida con CPU aislada. La paridad de las salidas del control completo se comprobó bit a bit y los pesos no cambiaron.

La competencia combinada incrementa la mediana en esta carga y no justifica sustituir una biblioteca por kernels propios. Su p95 en 64 activos aumenta un 4,80 %, dentro del límite de ingeniería fijado para el piloto. Esto no aporta evidencia de menor MAE.
