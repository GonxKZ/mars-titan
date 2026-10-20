# Comprobaciones de separación de señales de cuantiles

La [implementación](../../../docs/research/quantile-routing.md) conserva la función de inferencia para pesos fijos y permite separar dos vías de gradiente durante el ajuste. Son cinco configuraciones con identidad propia y opciones desactivadas por defecto. El [piloto completado](development-v1/README.md) reduce el error de la combinación frente a su referencia, pero sigue perdiendo frente al residual cero. No se promueve ninguna configuración.

La [verificación](verification.json) recoge 378 casos distintos aprobados en una selección de pruebas. Incluye CPU y CUDA, FP32 y FP64, valores y derivadas, gradiente sustituto, paridad con la cabeza nativa, recuperación, RNG, validación de configuraciones, rechazo de cambios de ruta en un consumidor congelado y análisis con fixtures. Las cuatro comprobaciones CUDA separadas de la regresión CPU también pasaron. PyTorch emitió un aviso sobre su modo de diagnóstico de sincronización. No se ejecutó toda la batería del repositorio.

Los 18 mutantes dirigidos fueron detectados y cada grupo volvió a pasar al restaurar las fuentes. Las huellas de los módulos mutados corresponden a las revisiones publicadas. Esta cifra no cubre todos los mutantes posibles. La [medida de calidad](quality.json) utiliza coverage.py 7.16.1 con ramas y Radon 6.0.1. Registra por función la convención `CRAP = C² × (1 − cobertura de sentencias)³ + C`. La cabeza y la configuración de rutas cubren 68 de 68 sentencias y 20 de 20 ramas. El diagnóstico de gradientes cubre sus 25 sentencias y ocho ramas. El conjunto de siete archivos cubre 926 de 1.098 sentencias y 338 de 470 ramas.

La selección conserva una zona poco comprobada en el validador general `FinancialConfig.__post_init__`, con complejidad 61, 34 de 73 sentencias cubiertas y CRAP 628,39. La cobertura focalizada no acredita todas sus combinaciones. No se ha validado esta modificación con Lean. El contrato de derivación y su contraejemplo se comprobaron con autodiferenciación, sin atribuirles una garantía de convergencia.

## Coste CUDA con entradas precodificadas

La [medida](cuda-cost.json) corresponde al commit `7acf7c278226232f5da134bfb91c5121b9ec0dfe`, RTX 4070 Laptop de 8 GiB y PyTorch 2.14.0+cu130, con FP32 estricto y TF32 desactivado. Incluye tamaños 1, 64 y 128, ocho decisiones y diez recorridos medidos tras dos de calentamiento. La tabla recoge latencias por decisión de 64 activos y caudal de filas de activos. Cada p99 tiene solo 80 decisiones medidas por configuración.

| Configuración | p50 ms | p95 ms | p99 ms | Filas/s |
| --- | ---: | ---: | ---: | ---: |
| original | 4,507 | 5,913 | 6,061 | 13.312,6 |
| all_feedback | 4,585 | 5,630 | 5,800 | 13.563,8 |
| median_isolated | 4,471 | 6,696 | 7,616 | 13.221,0 |
| encoder_isolated | 4,597 | 6,303 | 6,734 | 12.984,9 |
| both_isolated | 4,703 | 6,456 | 6,682 | 12.865,1 |

Todas las configuraciones tienen 186.823 parámetros y 67.856 bytes tensoriales por flujo. No hay ahorro de estado persistente. En la combinación, p95 aumenta un 9,19 % frente a la referencia, dentro del límite del 20 % fijado antes del piloto. No se demuestra aceleración. Durante inferencia sin gradientes todas las rutas hacen una sola proyección. La separación de dispersión añade otra proyección cuando se requieren sus derivadas.

El proceso de medición tardó 28,528 segundos y alcanzó 1.521.606.656 bytes de RSS. Los cinco modelos de cada tamaño residen simultáneamente, por lo que sus picos CUDA no describen un despliegue con un solo modelo. El recibo conserva muestras temporales, preparación, transferencia y derivación de una pérdida artificial, sin pasos de optimizador. Esa pérdida no es un experimento predictivo.

La decisión parte de entradas precodificadas residentes e incluye preparación, validación y actualización neuronal. La carga y transferencia se informan por separado. No incluye la lectura Parquet por decisión, los codificadores anteriores al corpus, energía o contadores internos de kernels. Hubo escritorio gráfico y tareas CPU concurrentes. La paridad del control completo se comprobó, sin cambiar los pesos. Estos tiempos no justifican introducir un kernel propio ni cambiar de biblioteca.
