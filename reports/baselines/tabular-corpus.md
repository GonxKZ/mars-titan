# Controles tabulares de la edición verificada

Se han ajustado tres regularizaciones de Ridge y un control HistGradientBoosting
sobre las mismas 160 muestras de entrenamiento y 135 de validación que la campaña
neuronal del 22 de septiembre. Una comprobación independiente contrastó igualdad
exacta de claves, fechas y objetivos entre ambas rutas. El test final sigue cerrado.

Cada entrada contiene 1.681 características de las cuatro modalidades y el
contexto macro. Ridge utiliza float64 en CUDA. HistGradientBoosting se ejecuta en
CPU, con cuatro hilos y sin parada temprana aleatoria.

| Referencia | Dispositivo de ajuste | Tiempo de ajuste | MAE de validación | MSE de validación |
| --- | --- | ---: | ---: | ---: |
| Cero | Sin ajuste | No corresponde | 0,014093 | 0,000390 |
| Ridge, alfa 0,1 | CUDA | 0,499 s | 0,026556 | 0,001238 |
| Ridge, alfa 1 | CUDA | 0,376 s | 0,026422 | 0,001227 |
| Ridge, alfa 10 | CUDA | 0,376 s | 0,025319 | 0,001132 |
| HistGradientBoosting | CPU | 0,482 s | 0,015172 | 0,000412 |

Los errores se expresan en unidades de retorno residual, no en porcentaje de
aciertos. Los cuatro ajustes empeoran MAE y MSE frente a cero. Se conservan esos
resultados. La cobertura editorial es reducida y no permite generalizar el orden
de modelos al mercado ni atribuir rentabilidad a una diferencia de error.

El tiempo completo de cada llamada estuvo entre 0,895 y 1,273 segundos. Incluye
validación de datos, ajuste, restauración y predicciones. Son medidas con otras
comprobaciones locales y verificación editorial concurrentes, sin controlar la
caché. No describen el coste del corpus completo.

## Memoria y comprobaciones

La matriz de ajuste estimada ocupa 2.152.960 bytes, incluidas sus etiquetas.
HistGradientBoosting cabe en el presupuesto de 256 MiB de esta edición. Ese
presupuesto no equivale al RSS del proceso ni justifica recortar filas si una
edición posterior lo supera. Ridge trabaja por bloques y guarda sus estadísticas
en función de la dimensión, no del número total de filas.

Pasan 1.032 pruebas locales, sin omisiones. Las siete específicas verifican
población, etiquetas, orden de modalidades, separación de validación, presupuesto
y restauración. Tres mutaciones dirigidas fueron detectadas. El [recibo de calidad
y resultados](../resources/tabular-corpus-quality.json) conserva huellas, versiones,
dispositivos, cobertura, CRAP y recursos observados.

La integración tabular no reanuda todavía estadísticas o árboles a mitad del
ajuste. Las ejecuciones interrumpidas se conservan y se repiten en otra salida.
La ampliación global de fuentes y las referencias de memoria externa continúan
fuera de este resultado.
