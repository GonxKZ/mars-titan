# Latencia de los modelos seleccionados

Se completaron los 12 casos previstos en la RTX 4070 Laptop, usando `cuda:0`.
Cada caso reconstruyó el modelo y el estado para una pasada sin instrumentación
por evento y tres pasadas medidas. Las 48 pasadas conservaron exactamente las
predicciones puntuadas y los contadores de su referencia. Los parámetros y el
RNG compartidos no cambiaron durante las pasadas. No hubo pasos del optimizador
de parámetros compartidos. La actualización interna de memoria sigue la receta
del modelo.

La medida corresponde a `linear_v1`, semilla 42 y lector entrenado con K1.
Compara K1 fijo, K4cc fijo y su selector ya elegido. Las entradas parten del
corpus codificado. No incluye producir los embeddings originales, descargar
datos ni convertir noticias nuevas en vectores.

Cada pasada conserva 251 decisiones de calentamiento histórico. Las primeras
cinco decisiones puntuables se separan del resumen de latencia, dejando 245
decisiones estables. La tabla muestra la mediana, entre las tres pasadas, de
cada percentil por decisión. Son lotes de hasta 64 o 256 activos, no peticiones
individuales de un activo.

| Cohorte | Gráficos | Ruta | p50 (ms) | p95 (ms) | p99 (ms) |
| ---: | --- | --- | ---: | ---: | ---: |
| 64 | Originales | K1 | 10,188 | 12,074 | 13,576 |
| 64 | Originales | K4cc | 11,751 | 14,431 | 17,028 |
| 64 | Originales | Selector | 12,868 | 15,963 | 18,973 |
| 64 | Geometría | K1 | 10,223 | 13,418 | 15,622 |
| 64 | Geometría | K4cc | 11,972 | 16,677 | 19,970 |
| 64 | Geometría | Selector | 11,832 | 13,711 | 15,399 |
| 256 | Originales | K1 | 16,557 | 20,483 | 25,153 |
| 256 | Originales | K4cc | 18,626 | 24,539 | 32,617 |
| 256 | Originales | Selector | 17,833 | 21,822 | 34,510 |
| 256 | Geometría | K1 | 17,294 | 21,485 | 28,273 |
| 256 | Geometría | K4cc | 17,664 | 24,093 | 35,742 |
| 256 | Geometría | Selector | 19,918 | 23,002 | 30,390 |

El tramo medido incluye preparación del lote en la observación, transferencia,
avance del núcleo, consulta, selección, refinamiento, emisión y registro de
pendientes. La lectura que ocurre fuera de ese método y la maduración se
registran en el tiempo completo de fase. No están incluidas en los percentiles
de la tabla.

Los picos RSS estuvieron entre 1,585 y 2,024 GiB. El asignador de PyTorch registró
entre 78,896 y 152,819 MiB de memoria CUDA asignada. El RSS corresponde a la vida
del proceso. La cifra CUDA incluye los modelos residentes y no describe toda
la memoria usada por el dispositivo. Los picos reservados y los rangos entre
pasadas se conservan en [results.json](results.json).

Los modelos usan FP32 estricto sin TF32. Los coeficientes del selector se
conservan en FP64. Hubo actividad del escritorio y comprobaciones CPU acotadas
en paralelo. Estos percentiles describen la carga observada y no garantizan la
latencia de otras configuraciones o de un servicio desplegado.

La ejecución de los 12 casos consumió 830,165 segundos. Un perfil de llamadas
adicional del caso de 256 activos, gráficos originales y K1 consumió otros
116,735 segundos. La suma fue 946,900 segundos, dentro de los veinte minutos
técnicos. El perfil se conserva separado de los percentiles anteriores y sus
cuatro pasadas también verificaron las predicciones originales.

En el perfil, `_event` de `financial_observations.py` acumuló 59,455 segundos de
los 114,811 segundos registrados por cProfile. Incluye llamadas internas de
construcción y decodificación. `_sample_group` acumuló 27,193 segundos. Estos
tiempos se solapan y no se suman. El perfil de llamadas puede incluir esperas
CUDA y no mide el tiempo útil de cada kernel. La preparación de observaciones
es un candidato concreto para la siguiente revisión de rendimiento. Esta
medida no prueba aún que cambiar una caché o un backend acelere el recorrido.

[profile.json](profile.json) conserva funciones, llamadas, tiempos y la huella
del perfil. Las huellas de cada pasada, las identidades de modelos y checkpoints
y las medidas completas están en [results.json](results.json). Las trazas por
evento y el perfil binario permanecen fuera del repositorio.

La latencia no acredita una mejora predictiva. El
[resultado provisional de desarrollo](../../research/refinement-development-20261019/README.md)
mantiene esa comparación por separado. El test final no se abrió.
