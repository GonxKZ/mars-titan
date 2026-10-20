# Piloto de separación de señales de cuantiles

La separación de ambas vías reduce el MAE frente a Titans-MAC original en este piloto, pero sigue perdiendo frente al residual cero. Aislar solo la ubicación empeora mucho el resultado. Ninguna configuración se promueve ni acredita superar el estado del arte.

Se completaron las cinco configuraciones del [protocolo](../../../../docs/experiments/quantile-routing-v1.md), fijado antes del ajuste. El runtime quedó congelado en `7acf7c278226232f5da134bfb91c5121b9ec0dfe`. Cada modelo utilizó semilla 42, una época, 203.350 filas de entrenamiento y 685 actualizaciones. Se conservaron las mismas 8.000 filas de validación, 3.968 de calibración y 15.936 de desarrollo, estas últimas correspondientes a 249 sesiones de 2023. Son 64 activos estadounidenses con todas sus filas admisibles desde 2000 y sus máscaras. No es el universo completo de FinMultiTime. El test final de 2024 permaneció cerrado.

## Error e incertidumbre

| Configuración | MAE residual por sesión |
| --- | ---: |
| Residual cero | 0,01584945 |
| Original | 0,01747972 |
| Ambas vías originales | 0,01747972 |
| Ubicación aislada | 0,03041158 |
| Dispersión aislada del tronco | 0,01659710 |
| Ambas vías aisladas | 0,01590925 |

El contraste principal, `both_isolated − original`, es −0,00157047, con intervalo temporal del 95 % [−0,00181875, −0,00134364]. La mejora relativa es del 8,98 %, con intervalo [7,85 %, 10,22 %]. La diferencia supera el margen de 0,00001 declarado en este piloto. Frente a cero, la misma variante tiene un error un 0,3773 % mayor, con diferencia +0,00005980 e intervalo simultáneo [+0,00002343, +0,00009618]. La mejora frente a una referencia que pierde con cero no basta para elegir la variante.

El aislamiento de ubicación aumenta el MAE frente a la referencia en +0,01293186, con intervalo simultáneo [+0,00862816, +0,01723555]. Aislar la dispersión del tronco lo reduce en −0,00088262, intervalo [−0,00128352, −0,00048172], pero también pierde frente a cero. La combinación mejora frente a cada componente: −0,01450232 frente a ubicación aislada y −0,00068785 frente a dispersión aislada. Sus intervalos simultáneos son [−0,01902991, −0,00997473] y [−0,00082527, −0,00055043].

La interacción `MAE_both − MAE_median − MAE_encoder + MAE_original` es −0,01361970, con intervalo simultáneo [−0,01786745, −0,00937196]. Este valor recoge en gran parte que la combinación evita el fuerte empeoramiento del aislamiento de ubicación. No prueba causalidad económica ni un beneficio general de combinar mecanismos.

Los intervalos usan remuestreo circular por bloques de 16 sesiones, 2.000 réplicas y semilla 20261020, con sensibilidad predefinida a 8 y 32. Las doce comparaciones secundarias conservan intervalos simultáneos. El [resultado completo](results.json) incluye sensibilidades, métricas adicionales y costes. No hay sesiones excluidas en el contraste principal. Se igualan actualizaciones, no tiempo de cómputo.

La ventana de desarrollo ya se utilizó en otros estudios y solo se ha ejecutado una semilla. El intervalo temporal no incluye variación entre semillas ni la búsqueda acumulada entre estudios. Es evidencia exploratoria de esta cohorte y receta, sin confirmación en otras ventanas o mercados. No se cambió una calibración para favorecer estas rutas.

## Diagnóstico posterior de validación

La [inspección de validación](validation-moments.json) utiliza únicamente las 8.000 etiquetas maduras de 125 sesiones de 2022, sin ajustar transformaciones ni elegir configuraciones nuevas. La desviación típica de las predicciones baja de 0,007723 en el original a 0,001305 con ambas vías aisladas. La media pasa de 0,006366 a −0,000689. Las correlaciones de Pearson con el retorno son −0,01910 y +0,00591, respectivamente. No indican una asociación predictiva apreciable en ese tramo.

El MAE de validación de la combinación es 0,01808159 frente a 0,01804236 de cero. La reducción de dispersión y sesgo es compatible con acercarse al predictor cero. No demuestra que esa sea toda la causa del cambio de MAE ni que el modelo haya extraído más información útil. La identidad `MSE_predicción − MSE_cero = E[predicción²] − 2 E[predicción × etiqueta]` se comprobó con ponderación por sesión. Describe el MSE y no sustituye el criterio principal de MAE.

## Paridad, coste y trazabilidad

El [control con ambas vías originales](unity-control.json) reproduce bit a bit los 45 tensores compartidos y las predicciones de validación, calibración y desarrollo después de 685 actualizaciones. La inicialización coincide. La [comparación histórica](baseline-historical-parity.json) confirma los mismos parámetros y predicciones que `original` del piloto de competencia de claves. No se ha elegido retrospectivamente otra referencia más fácil de superar.

Los cinco trabajos consumieron 605,491 segundos externos, unos 10,09 minutos, y 3.425 actualizaciones. Los 38 intentos acumulados, incluidos los 33 anteriores, suman 4.919,036 segundos externos. El análisis consumió otros 4,727 segundos de proceso, con 0,731 segundos dentro del analizador. Los tiempos internos ya están incluidos en los externos. La cuenta no incluye toda la investigación, las pruebas o la medida técnica de coste, que registra por separado 28,528 segundos.

El mayor RSS de los cinco trabajos fue de 2.069.753.856 bytes. PyTorch registró máximos de 206.327.808 bytes asignados y 236.978.176 reservados en CUDA. Se utilizó una RTX 4070 Laptop de 8 GiB, PyTorch 2.14.0+cu130 y FP32 estricto, con una sola carga científica GPU. Había escritorio gráfico y tareas CPU de documentación. No es una medida con la máquina aislada. La [medida técnica](../README.md#coste-cuda-con-entradas-precodificadas) no muestra ahorro persistente ni aceleración de la combinación.

El [manifiesto preparado](prepared.json) identifica población, entorno, recetas y fuentes. El [inventario](artifacts.json) registra predicciones por muestra, métricas por sesión, checkpoints, manifiestos y recibos conservados fuera de Git. El [cierre](closure.json) confirma los cinco intentos y autorizaciones locales cerrados, con la protección global intacta. No queda un entrenamiento de este piloto ejecutándose en segundo plano.

Se conservan código, controles y resultados con las opciones desactivadas. Este piloto no completa el contraste escalar frente a cuantiles de varias semillas ni la aceptación de los finalistas de [#22](https://github.com/GonxKZ/mars-titan/issues/22). La campaña completa y el test final siguen cerrados.
