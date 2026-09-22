# Supervisión por bloques

El lector recorre activos y grupos Parquet sin cargar todas las etiquetas en un
diccionario global. Conserva precios, noticias, fundamentales, gráficos y contexto
macro. Las pruebas incluyen la continuación desde el consumo confirmado, la
última tanda incompleta y la separación temporal entre mercados.

La revisión detectó que un dato de ejecución cambiaba la identidad del corpus al
reutilizar etiquetas. También encontró vistas NumPy que retenían grupos enteros
hasta completar un lote. Ambas regresiones se reprodujeron y corrigieron. La
preparación rechaza continuar si cambian las dependencias numéricas o temporales.

## Memoria y lectura

Medición local del 22 de septiembre de 2026. Cada tamaño se ejecutó en un proceso
separado. Los grupos sintéticos contienen 128 filas y vectores de dimensiones
384, 512, 45 y 420. Los lotes contienen hasta 128 muestras. La ventana sintética
tiene dos sesiones y la edición real conserva 64.

| Datos | Filas de ajuste | Parquet en disco | Tiempo del lector | Pico RSS (VmHWM) |
| --- | ---: | ---: | ---: | ---: |
| 10 activos sintéticos | 8.000 | 65.504.580 bytes | 0,428 s | 154.435.584 bytes |
| 130 activos sintéticos | 104.000 | 851.559.540 bytes | 4,327 s | 162.594.816 bytes |
| Edición verificada disponible | 160 | No medido en este ensayo | 0,304 s | 162.254.848 bytes |

Multiplicar las filas sintéticas por trece aumentó el pico observado en
8.159.232 bytes. La memoria no creció en proporción a la matriz completa. El
manifiesto y las huellas sí crecen con el número de activos. Cada grupo sigue
limitado a 64 MiB y se copian únicamente los vectores seleccionados antes de
acumular el lote, para no retener grupos ya recorridos.

Son medidas de una sola repetición, con entrenamiento y verificación editorial
concurrentes. La caché del sistema no se controló. El tiempo incluye validación
de artefactos y recorrido, no creación de los datos, arranque de Python ni
entrenamiento. No es una comparación de velocidad entre implementaciones.

## Población real y comprobaciones

La edición de desarrollo tiene 38 activos con muestras. El contrato de etiquetas
admite 160 muestras de entrenamiento hasta 2022 y 135 de validación de 2023.
Los motivos de exclusión y las muestras reservadas permanecen en las etiquetas.
Esta edición no representa el corpus completo ni incluye una población china
ya verificada. El test final sigue cerrado.

La batería completa pasó con 989 pruebas, sin omisiones, en 57,43 segundos.
Las 28 pruebas específicas pasaron también con instrumentación de cobertura.
Tres mutaciones dirigidas detectaron las regresiones de identidad, retención
de grupos y cambio de dependencia. No equivalen a un análisis de mutación
completo.

El [recibo de medidas y calidad](../resources/corpus-inputs-quality.json) conserva
huellas, cobertura por archivo, complejidad, CRAP y límites. Estas métricas ayudan
a localizar riesgos y no garantizan ausencia de errores.
