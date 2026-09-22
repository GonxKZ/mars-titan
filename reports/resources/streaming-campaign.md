# Ejecutor por bloques y tamaño de lote

El ejecutor reutiliza etiquetas confirmadas y no repite la preparación residual
en cada configuración. Mantiene una época completa por pasada, incluida la
última tanda incompleta. Las pruebas comparan la continuación desde una pausa
con la ejecución continua y conservan exactamente pesos y predicciones.

La revisión corrigió tres riesgos de procedencia. La campaña y los casos comparten
identidad científica, la transformación de precios queda incluida en sus huellas
y una continuación exige el checkpoint final exacto de su padre. No puede atribuir
a ese archivo los pesos recuperados de otra época.

## Medición con datos verificados

Se ejecutaron 60 ensayos de coste sobre las 160 muestras de ajuste y 135 de
validación disponibles. Cuatro familias, cinco tamaños solicitados y tres
repeticiones. Cada ensayo completó dos épocas, comprobó la población y guardó
checkpoints y predicciones. El orden de los ensayos varía de forma reproducible
entre repeticiones.

| Modelo | Lote 16 | Lote 64 | Lote 128 | Solicitud 256 | Solicitud 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RNN | 1,463 s | 1,370 s | 1,542 s | 1,518 s | 1,521 s |
| LSTM | 1,497 s | 1,480 s | 1,486 s | 1,483 s | 1,454 s |
| GRU | 1,464 s | 1,465 s | 1,407 s | 1,651 s | 1,657 s |
| DLinear | 1,394 s | 1,590 s | 1,932 s | 1,468 s | 1,439 s |

Las cifras son medias del tiempo de llamada completo. Incluyen validación de
entradas, ajuste, validación, checkpoints y escritura de predicciones. No incluyen
el arranque del intérprete. Las solicitudes de 256 y 512 solo produjeron un lote
de 160 muestras en esta edición. No miden lotes reales de esos tamaños.

El mayor pico asignado por PyTorch fue de 115.492.864 bytes. Esa cifra no incluye
toda la memoria del controlador y del contexto CUDA. Había otra campaña,
verificación editorial y pruebas locales concurrentes. La caché del sistema no
se controló. Se conservan las tres observaciones y su desviación típica en el
[recibo completo](streaming-campaign-quality.json).

No se observa una reducción uniforme del tiempo al solicitar lotes mayores.
La receta inicial mantiene 16. Cambiar el lote modifica el número de pasos y la
trayectoria de AdamW, por lo que estas medidas no prueban paridad de calidad
predictiva entre tamaños. No se elige una configuración por el MAE de validación.

## Comprobaciones y alcance pendiente

La batería completa pasó con 1.025 pruebas sin omisiones. Las 21 pruebas
específicas incluyen parada, continuación, cambio de entorno, corrupción del
padre, unión de mercados y protección del test final. Tres mutaciones dirigidas
fueron detectadas. Cobertura y CRAP se conservan como diagnóstico.

El motor dispone de brazos estadounidense, chino y conjunto, pero no convierte
datos pendientes en muestras admitidas. La campaña completa, las variantes de
ponderación por sesión y activo, las referencias tabulares adaptadas al corpus
y la evaluación estadística final siguen siendo trabajos separados.
