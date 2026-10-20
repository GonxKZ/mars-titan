# Controles del objetivo puntual con cinco semillas

El contraste principal es inconcluyente. La combinación de vías aisladas tiene un MAE medio un 0,41 % menor que su control L1/10, pero el intervalo temporal del 95 % incluye cero. Ningún modelo supera al residual cero en esta ventana. La reducción frente a la cabeza original no permite atribuir una mejora a la separación de señales ni a su analogía neurocientífica.

## Comparación ejecutada

Se completaron los 25 ajustes predeclarados, con cinco configuraciones, semillas 42–46, una época y 685 actualizaciones por ajuste. Se conserva la misma cabeza de cinco salidas, [parámetros iniciales emparejados](paired-initialization.json), K=1, banco desactivado, proyecciones del artículo y disposición temporal contigua. L1/10 y L1 solo ajustan la mediana. Sus colas no son cuantiles aprendidos y no se interpretan como evidencia de incertidumbre.

La cohorte conserva 64 activos estadounidenses, todas sus filas admisibles desde 2000 y las ausencias explícitas. Son 203.350 filas de ajuste, 8.000 de validación, 3.968 de calibración y 15.936 de desarrollo de 2023. El desarrollo ya se ha utilizado. Este estudio no es una confirmación independiente ni representa todo el universo histórico pendiente de verificación.

| Configuración | MAE medio por sesión | Desviación muestral entre semillas | Más error que cero |
| --- | ---: | ---: | ---: |
| Original y control completo | 0,02161406 | 0,00484225 | 36,37 % |
| Ambas vías aisladas | 0,01700958 | 0,00086600 | 7,32 % |
| Solo mediana, L1/10 | 0,01707936 | 0,00166101 | 7,76 % |
| Solo mediana, L1 | 0,01704668 | 0,00100554 | 7,55 % |
| Residual cero | 0,01584945 | No depende de la semilla | 0 % |

Primero se promedian errores de activos dentro de cada sesión y después los errores de las cinco semillas. Las unidades temporales siguen siendo las mismas 249 sesiones. No se promedian predicciones ni se evalúa un ensemble.

## Incertidumbre e interpretación

El principal, `MAE(both_isolated) − MAE(point_component)`, es −0,000069779. Su intervalo temporal del 95 % es [−0,000153996, +0,000020706]. El margen mínimo relevante era 0,00001. La diferencia media no acredita mejora respaldada ni equivalencia dentro de ese margen. El resultado sigue siendo inconcluyente con las longitudes de bloque 8 y 32 predefinidas.

La combinación mejora frente a L1/10 en dos semillas, empeora en dos y empata en MAE en una. La desviación muestral de esas diferencias es 0,00132338. El bootstrap temporal mantiene fijas estas cinco inicializaciones. No estima una población de nuevas semillas ni corrige la reutilización del desarrollo. Se utilizan 2.000 réplicas circulares de bloques de 16 sesiones y semilla 20261020.

Los doce secundarios tienen intervalos simultáneos. Todos los modelos empeoran frente a cero, con sus intervalos por encima de cero. La combinación reduce el error un 21,30 % frente al original, pero los controles L1/10 y L1 también lo reducen un 20,98 % y un 21,13 %. Las comparaciones entre la combinación y los controles puntuales no respaldan una ventaja adicional. No se calcula interacción factorial entre estos objetivos.

No se promueve ninguna variante. La decisión mantiene la referencia y los controles. Estos datos no justifican una mejora sobre el estado del arte, retención útil, generalización financiera o una reproducción de mecanismos cerebrales. Una época tampoco decide la convergencia de cada objetivo.

## Paridad, coste y trazabilidad

Los cinco controles completos reproducen por bytes los 45 tensores compartidos, la inicialización y los Parquet de las tres fases de su referencia. Las tres configuraciones que ya existían reproducen también sus resultados anteriores en las cinco semillas. Son repeticiones de las mismas ventanas, no quince nuevas réplicas temporales.

Los 25 intentos están cerrados, sin fallos nuevos, y suman 3.094,613 segundos externos y 17.125 actualizaciones. Los 89 intentos acumulados suman 11.573,363 segundos, dentro del límite de 14.400. Al cerrar esta ronda quedaban 2.826,637 segundos de ese presupuesto. El análisis añade 6,281 segundos de proceso y las comprobaciones técnicas se contabilizan aparte. Los tiempos internos están incluidos en los externos y no se suman otra vez.

El mayor RSS fue 2.143.989.760 bytes. Los máximos de PyTorch fueron 206.327.808 bytes asignados y 234.881.024 reservados en CUDA. Son métricas distintas del uso total de la GPU. La RTX 4070 Laptop informa de 8.188 MiB. Se usó FP32 estricto, sin TF32 y con una sola carga científica simultánea. El escritorio y tareas técnicas CPU siguieron activos, por lo que estas duraciones no son una medida aislada de aceleración entre modelos. No se midió energía.

El productor y el analizador están fijados en `c122cee7491b86a0343724e28454b4abd292b074`. El [resultado legible por máquina](results.json), el [protocolo preparado](prepared.json), el [manifiesto del análisis](analysis-manifest.json), los [controles completos](unity-controls.json), la [paridad histórica](historical-parity.json), el [resumen de coste](cost-summary.json) y el [cierre](closure.json) conservan sus comprobaciones. El [inventario](artifacts.json) identifica 274 artefactos con rutas relativas, bytes y SHA-256. Incluye predicciones por muestra, métricas por sesión, checkpoints, recetas, fuentes de ejecución y recibos. Los archivos voluminosos permanecen fuera de Git.

La preparación y el análisis reutilizan `benchmarks.neural_plasticity_study` y `benchmarks.quantile_multiseed_analysis`. El manifiesto conserva la forma del comando y las huellas de su ejecución real. Las autorizaciones locales están cerradas y la protección global conserva su huella. La campaña completa y el test final de 2024 permanecen cerrados. El control original de cabeza escalar del Transformer y la comparación de finalistas de #22 siguen pendientes.
