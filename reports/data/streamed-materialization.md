# Comprobación de la materialización por ventanas

La nueva ruta conserva las 832 decisiones admisibles de AA en la comparación
con el materializador anterior. Las fechas coinciden. La diferencia máxima en
los vectores de texto es 5,96 × 10⁻⁸. Los vectores visuales, contables y macro
coinciden exactamente. La tolerancia fijada para la comparación era 10⁻⁵.
Es una prueba de preparación, sin entrenamiento de modelos predictivos.

Se ejecutaron tres pares alternados en procesos separados, con cachés de
embeddings inicialmente vacías y calentamiento de los codificadores antes de
medir. El equipo tenía una RTX 4070 Laptop de 8.188 MiB y 30.816 MiB de memoria
visible. El escritorio permaneció activo. No coincidieron estas repeticiones
con la normalización del corpus ni con las pruebas locales.

| Ruta | Mediana de materialización | Muestras por segundo |
| --- | ---: | ---: |
| Referencia anterior | 17,081 s | 48,71 |
| Ventanas y contexto macro compartido | 16,653 s | 49,96 |

La razón entre las medianas es 1,026. La diferencia es pequeña y no demuestra
una aceleración general del corpus. La mejora funcional es mantener acotados
los cuerpos textuales y no recalcular el historial contable en cada decisión.
El ensayo de un activo incluye la construcción inicial de los vectores macro,
que se comparte con los siguientes activos durante un recorrido completo.

Los codificadores se mantuvieron en float32 sobre `cuda:0`. La agregación textual
usó float64 antes de guardar float32. El pico CUDA asignado fue de 594.470.400
bytes en ambas rutas. No se introdujeron kernels propios ni una menor precisión
para obtener estos tiempos. No se midieron energía ni contadores de ciclos.

## Pruebas y límites

La batería completa pasó con 1.120 pruebas, sin omisiones, en 122,07 segundos
con instrumentación de cobertura. Incluyó la comprobación de codificadores CUDA
y el analizador `clang-tidy` 21.1.8. Las pruebas específicas de este cambio
sumaron 60 casos. Cuatro mutaciones dirigidas fueron detectadas. Alteraban el
límite temporal de noticias, la actualización contable, la cohorte de una fila
y la reconciliación de la población.

Las regresiones comprueban recuperación, recibos alterados, enlaces intermedios,
metadatos cambiados durante su lectura y mezcla de representaciones que tienen
la misma anchura pero distinto significado. También prueban un grupo textual
cuyo diccionario ocupa unos 101 kB y que se expandiría a unos 20 MB al convertir
todas sus cadenas a la vez. La nueva lectura conserva el diccionario y entrega
un registro cada vez.

La cobertura por sentencias y ramas, la complejidad de Radon y el índice CRAP
se conservan junto a las mediciones. CRAP utiliza la cobertura por sentencias
de cada función, con la expresión `C² × (1 − cobertura)³ + C`. Son indicadores
para revisar lógica poco comprobada, no una garantía de ausencia de errores.

Los resultados completos están en
[el registro de calidad](../resources/streamed-materialization-quality.json).
El [script de medición](../analysis/benchmark_streamed_modalities.py) utiliza
las decisiones admisibles del activo AA en la edición preparada. Las huellas
identifican datos, modelos y transformaciones. Los experimentos iniciales con
otras cargas simultáneas se conservaron separados y no se usaron en la tabla.

No se ha completado aquí la campaña científica del corpus ni la recuperación
de publicaciones contables chinas. La marca de preparación completa no sustituye
las etiquetas, la congelación de la población ni la evaluación de cada modelo.
