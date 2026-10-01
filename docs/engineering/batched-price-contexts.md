# Ventanas de precios calculadas por bloques

La transformación mantiene el logaritmo del precio respecto al primer cierre
de la ventana y el logaritmo del volumen relativo a su media. Agrupa como
máximo 256 ventanas, sin cambiar el orden del lector ni materializar todas las
ventanas del corpus. El resultado sigue siendo `float32`.

El perfilado previo localizó 1,086 de 3,179 segundos instrumentados en
`price_features`, al leer 49.152 filas. La reducción de llamadas y operaciones
repetidas utiliza NumPy. No requiere un kernel nuevo ni cambios de precisión.
El tiempo instrumentado sirve para localizar trabajo, no para estimar el
entrenamiento.

## Paridad

Se compararon las dos rutas sobre las 1.548.307 filas de entrenamiento y las
261.069 de validación de la edición estadounidense completa. La igualdad
numérica comprende las cinco entradas, etiquetas, fechas, pesos, orden y
cursor. La diferencia máxima del bloque de precios es cero. El contraste
usó la versión anterior completa del lector, identificada por su huella,
sin abrir la reserva final.

La asignación por bloques valida primero los índices y después los convierte
al entero nativo. Esto evita que la mezcla de índices `uint64` e `int64`
genere índices flotantes. Solo se leen las ventanas seleccionadas. Copiar la
fila de salida evita retener el bloque completo en una muestra aislada.

Para el máximo admitido de 256 ventanas de 512 sesiones, la matriz intermedia
de precios `float64` ocupa 5 MiB y el resultado 2,5 MiB, además de temporales
acotados. Es memoria adicional deliberada para reducir trabajo repetido. No
se modifica el presupuesto de 64 MiB de cada grupo Parquet ni se amplía el
tamaño máximo del corpus en memoria.

## Medición antes y después

La misma [sonda reproducible](../../reports/analysis/benchmark_full_corpus.py)
usa cuatro modalidades, macro, 128 unidades, dos capas y dropout 0,2. Cada
combinación tiene tres repeticiones de 24 lotes tras el calentamiento. Los
valores son medianas de muestras por segundo.

| Modelo | Lote | Referencia | Bloques | Relación de caudal |
| --- | ---: | ---: | ---: | ---: |
| RNN | 256 | 14.354 | 15.874 | 1,106 |
| RNN | 512 | 15.155 | 18.249 | 1,204 |
| LSTM | 256 | 10.959 | 12.819 | 1,170 |
| LSTM | 512 | 11.950 | 13.536 | 1,133 |
| GRU | 256 | 12.025 | 14.109 | 1,173 |
| GRU | 512 | 12.751 | 14.444 | 1,133 |
| DLinear | 256 | 16.082 | 18.841 | 1,172 |
| DLinear | 512 | 17.754 | 21.759 | 1,226 |

El pico asignado por PyTorch es idéntico en cada configuración. La GPU estaba
compartida y la caché del sistema operativo no estaba controlada. El tiempo
total de los procesos pasó de 37,261 a 28,800 segundos, pero la inicialización
también cambió. No se atribuye toda esa diferencia al código. La comparación
del caudal separa el calentamiento y la comprobación inicial del origen.

La [evidencia detallada](../../reports/resources/batched-price-contexts.json)
conserva repeticiones, versiones, cobertura, CRAP y paridad. Pasan 1.321 pruebas
locales y 18 casos específicos de esta transformación. Se detectaron cuatro
mutaciones dirigidas. La primera versión de la prueba de alineación compartía
parte del recorrido y dejaba escapar una selección de contexto incorrecta.
Un cálculo esperado independiente corrige esa limitación de la prueba.

El cambio no acredita mayor precisión predictiva, rendimiento exclusivo de
la GPU ni una comparación con código C++ propio. Tampoco permite reutilizar
un checkpoint que pertenezca a otra identidad de código. Las campañas nuevas
registran la huella de esta versión.
