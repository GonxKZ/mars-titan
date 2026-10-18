# Comprobación de la representación geométrica

La revisión `2c4817eaaf3f06323173cb934f667c58145840c9` incorpora
`candle_geometry_linear64_v1`, desactivada por defecto. La variante produce 512 valores FP32,
2.048 bytes por ventana y cero parámetros aprendidos. La [especificación](../../../docs/data/candle-geometry.md)
describe la fórmula y su integración. No se han entrenado modelos, modificado la edición v3.1
ni medido calidad predictiva.

## Pruebas y diagnóstico

Pasan 164 pruebas CPU de extracción, materialización, ventanas, máscaras históricas, caché,
recuperación y comparación. Ruff y `check_repository.py` pasan. No se ejecutó la batería
completa del proyecto ni la ruta CUDA de este benchmark.

El módulo nuevo cubre 74/74 sentencias y 18/18 ramas con coverage.py 7.16.1. La función de
conexión `_chart_input` cubre sus cinco sentencias y dos ramas. El [informe de calidad](quality.json)
usa Radon 6.0.1 y CRAP = C² × (1 − cobertura de sentencias)³ + C. El máximo del módulo nuevo
es 16, correspondiente a la validación de la ventana, con cobertura completa. El diagnóstico
incluye también los archivos existentes de materialización, cuyas funciones no quedan todas
ejercitadas por esta selección. Esa cobertura no representa una verificación del corpus completo.

Las [nueve mutaciones dirigidas](mutations.json) fueron detectadas: ventana equivocada,
ausencias marcadas como observadas, ausencia del suelo del rango, hash sin precios, aceptación
de desorden OHLC, reparación de mechas, identidad modificable, rasterización accidental y
omisión de cambios macro en el cotejo. Se ejecutaron en copias aisladas. No son una exploración
exhaustiva ni una demostración de ausencia de errores.

La comparación con huecos encontró una limitación del comparador de v3 frente a v3.1: supone
64 filas observadas consecutivas en cada ventana común. El cotejo nuevo no lo modifica.
Comprueba preparación, contratos y todas las columnas salvo `charts` y `chart_hash` por lotes.
El fixture conserva cinco filas admisibles con el mismo hueco de calendario en ambas ediciones.
Las pruebas del extractor comprueban por separado que un sufijo futuro no afecta a una muestra.

## Medidas CPU

Se utilizó un AMD Ryzen 9 8945HS, Python 3.12.14, NumPy 2.5.3 y Pillow 12.3.0. Cada caso usa
las mismas ventanas en ambas representaciones, semilla 665, una pasada de referencia, dos de
calentamiento y nueve repeticiones con orden alterno. Los fixtures contienen velas con rango,
rango nulo y huecos. El caso de una ventana tiene rango nulo. Había otras comprobaciones en
el equipo y no se aisló el procesador.

| Ventanas por recorrido | Geometría y hash, media por ventana | PNG y tinta, media por ventana | Datos de entrada | Pico RSS del proceso |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0,1207 ms | 8,2331 ms | 2.112 bytes | 122.912.768 bytes |
| 64 | 0,0812 ms | 7,5916 ms | 135.168 bytes | 125.411.328 bytes |
| 256 | 0,0638 ms | 6,5857 ms | 540.672 bytes | 126.787.584 bytes |

Los recibos de [una ventana](cpu-1.json), [64 ventanas](cpu-64.json) y [256 ventanas](cpu-256.json)
conservan tiempos individuales, percentiles de lote, caudal, versiones, huellas y límites.
En 64 ventanas, p50/p95/p99 del recorrido geométrico fueron 5,254/6,462/6,562 ms y los del
recorrido PNG con tinta, 475,391/545,009/546,565 ms. Son percentiles de nueve repeticiones del
lote completo, no colas de latencia de cada fila ni de una aplicación en producción.

Los [procesos completos](processes.json), incluido el arranque y todas las pasadas, duraron
0,75, 7,08 y 22,92 segundos, respectivamente. Su RSS máximo incluye la lectura del módulo,
las entradas y ambas representaciones. Puede superar el valor registrado dentro de `measure`.

Son costes de preparación de representaciones distintas. No se exige paridad entre sus
vectores y no se presenta una aceleración de cálculo equivalente. Tampoco son latencias del
predictor que consume vectores precalculados. El recorrido no incluye toda la lectura de
Parquet, las demás modalidades, el ajuste ni la confirmación de una edición. No se midieron
energía, coste monetario o transferencias CPU/GPU.

En aquella revisión, `benchmarks.candle_geometry measure --device cuda` estaba preparado
sin una medida CUDA. La ejecución posterior se describe al final de este informe. La
comprobación CPU del fallo de admisión sigue siendo una prueba del rechazo de la petición,
sin ejecutar CUDA.

## Derivación desde una edición codificada

La ampliación añade `candle_geometry_edition`, que copia todas las columnas salvo las dos
de velas, valida sus índices contra el calendario y publica otro destino. No carga modelos
ni recodifica texto. La evidencia anterior se conserva con sus huellas originales.

La selección ampliada pasa 200 pruebas CPU. Incluye interrupción durante Parquet y después
de confirmar un activo, recuperación, origen modificado, restauración de los mismos bytes,
vinculación del índice al cierre, cambio de un sufijo futuro, metadatos Arrow, factores,
activos sin filas y límites de lote. Ruff y el control del repositorio pasan.

El [diagnóstico ampliado](derivation-quality.json) registra 77/77 sentencias y 18/18 ramas
en el extractor. El módulo de derivación cubre 308/326 sentencias y 102/118 ramas. Sus máximos
de complejidad y CRAP son 28 y 31,63, en el comparador reutilizado. Las ramas sin cubrir
incluyen rechazos adicionales de manifiestos y límites, y la carrera de publicación ante
un destino creado por otra ejecución. Los límites del diagnóstico no equivalen a garantías
de corrección ni al resultado de toda la batería del proyecto.

Se detectaron las [16 mutaciones dirigidas](derivation-mutations.json), incluidas las nueve
anteriores en el código actual. Los siete casos nuevos eliminan la vinculación de sesión,
usan una ventana futura, ignoran la máscara, aceptan un origen sin verificar, omiten la
verificación final, ignoran una reescritura con los mismos bytes o pierden metadatos Arrow.

El [benchmark de derivación](../../../benchmarks/candle_geometry_derivation.py) crea un
fixture temporal, con un activo, 140 indicadores y 26 conceptos contables ausentes. Conserva
un hueco en el calendario y no genera observaciones para esos campos. Usa una pasada de
calentamiento y cinco repeticiones, con lotes de 128 filas. Se ejecutó en el mismo procesador,
con PyArrow 25.0.1, sin aislarlo de otras comprobaciones locales.

```bash
uv run --no-sync python -m benchmarks.candle_geometry_derivation \
  --samples 128 --repeats 5 --batch-rows 128 \
  --output data/derived/candle-geometry/derivation-cpu-128.json
uv run --no-sync python -m benchmarks.candle_geometry_derivation \
  --samples 512 --repeats 5 --batch-rows 128 \
  --output data/derived/candle-geometry/derivation-cpu-512.json
```

| Filas | Lectura | Preparación de precios | Geometría y empaquetado | Escritura | Cotejo | Total de la derivación |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 6,82 ms | 4,99 ms | 16,51 ms | 18,05 ms | 11,92 ms | 95,07 ms |
| 512 | 10,61 ms | 4,47 ms | 35,39 ms | 21,78 ms | 17,01 ms | 116,07 ms |

Son medias por derivación completa. La verificación final del origen añadió 2,59 y 1,69 ms,
respectivamente. El resto del tiempo total incluye los controles iniciales, manifiestos,
aperturas y publicación. La escritura se mide restando del intervalo de confirmación
Parquet la lectura y el cálculo consumidos por su generador. No incluye la copia de factores,
que figura por separado y vale cero en este fixture.

Los recibos de [128 filas](derivation-cpu-128.json) y [512 filas](derivation-cpu-512.json)
conservan cada repetición y las huellas del código. Sus p50/p95/p99 totales son
92,63/111,51/115,05 ms y 117,84/120,05/120,32 ms. Con cinco repeticiones describen solo
la dispersión de esta medición. Los picos RSS del proceso fueron 227.180.544 y 229.679.104
bytes, incluida la construcción del fixture. El proceso completo duró 3,14 y 4,20 segundos.

Las muestras escritas ocupan 22.211 y 72.713 bytes. Las modalidades ausentes comprimen más
que un corpus con noticias y fundamentales presentes. Estas medidas no estiman todavía
el coste de v3.1, no miden inferencia y no comparan la derivación con recodificar modelos.

## Selección explícita y reserva de disco

El commit `597ba65987d331eafc82da1682a8a74e9e9f3f73` añade selección de candidatos de un
origen completo y estimación previa mediante metadatos. La salida de una selección conserva
todas sus filas, se identifica como `development_snapshot` y declara completitud solo dentro
de esa selección. La reserva de disco se comprueba antes de escribir y al avanzar por activos.
Los confirmados no se presupuestan de nuevo al recuperar una ejecución interrumpida.

La revisión posterior pasa 217 pruebas CPU. El [diagnóstico de selección](selection-quality.json)
registra 77/77 sentencias y 18/18 ramas para el extractor, 333/352 sentencias y 111/128 ramas
para la derivación y 85/93 sentencias y 28/36 ramas para selección y presupuesto. La función
que coordina la derivación tiene complejidad 37 y CRAP 37,31. Ese tamaño y los rechazos todavía
sin cubrir quedan visibles en el diagnóstico, sin convertir la cobertura en una garantía.

Las [23 mutaciones dirigidas](selection-mutations.json) fueron detectadas. Los casos nuevos
comprueban rechazo temprano del origen incompleto, duplicados y activos ajenos, estado de
la selección, vínculo con el manifiesto original, reserva adicional al espacio estimado y
ausencia de lectura de filas durante la estimación. Ruff y el control del repositorio pasan.

Una prueba adicional conserva el caso adversarial en el que todos los máximos y mínimos
valen 10 y un cierre difiere en 5 × 10⁻⁹. Con la tolerancia de orden admitida y el suelo
relativo de 10⁻⁹, ese cierre normalizado queda cerca de 0,5 y su mecha superior es negativa.
El contrato documenta ese efecto sin recortar ni reparar el precio. No supone una garantía
de precisión ni un resultado predictivo.

Las medidas de preparación y derivación publicadas arriba conservan sus propias versiones
y huellas. La selección y el presupuesto de disco son posteriores a esas medidas.

## Datos reales y extracción con ResNet en CUDA

La [derivación de 64 activos](cohort64.json) contiene 235.482 filas. Tardó 55,59 segundos
en la llamada y alcanzó 238.948.352 bytes de RSS. Sus 260 archivos ocupan 265.926.624 bytes.
El cotejo conserva exactamente las columnas restantes y sus metadatos. La selección tiene
`cohort_complete=false`, `selection_complete=true` y `training_ready=false`. No acredita
que la edición histórica completa esté lista para reanudar la campaña.

El [cotejo de rasgos](cohort64-feature-audit.json) conserva 15.070.848 posiciones de precios
observadas, sin huecos en esta selección US. No implica presencia completa de las demás
modalidades. Las pequeñas salidas de rango y mechas negativas no superan 1,625 × 10⁻¹⁴
en valor absoluto. El caso adversarial de amplificación cerca de rango nulo sigue siendo
válido y no se ha recortado para mejorar las cifras.

Se midieron cuatro ventanas repartidas a lo largo de la historia de cada activo. Son 256
ventanas entre el 3 de abril de 2000 y el 29 de diciembre de 2023. Su preparación verifica
258 huellas de fuentes, no lee objetivos y conserva un NPZ de 541.188 bytes. Las mismas
entradas se usan en geometría, tinta y ResNet. El [recibo de procedencia](real-verification.json)
vincula la selección, las ventanas y las vistas supervisadas.

La [medida final](real-windows-cuda.json), realizada sobre el benchmark del commit
`60be0893f`, usa cinco calentamientos y treinta repeticiones con orden alterno. ResNet se
ejecuta en la RTX 4070 Laptop, con FP32 sin TF32 y lotes de ocho imágenes. La geometría y
el control de tinta usan CPU. Había otras comprobaciones CPU en marcha.

| Representación | p50 del lote | p95 del lote | p99 del lote | Ventanas por segundo según p50 |
| --- | ---: | ---: | ---: | ---: |
| Geometría | 20,076 ms | 22,531 ms | 39,236 ms | 12.751,67 |
| PNG y tinta | 1.990,040 ms | 2.062,506 ms | 2.501,770 ms | 128,64 |
| PNG y ResNet | 1.529,460 ms | 1.592,660 ms | 1.597,114 ms | 167,38 |

Son representaciones distintas. La medida acredita un menor coste de extracción de la
geometría en esta carga, sin demostrar paridad de sus vectores, menor MAE ni la misma
aceleración cuando el predictor consume vectores precalculados. El pico de tensores CUDA
es 193.570.304 bytes, con 260.046.848 reservados. El RSS de todo el proceso es
1.835.184.128 bytes e incluye codificadores residentes que la geometría no utiliza.
No se atribuyen esos picos a una única variante. La llamada completa duró 136,53 segundos.

El primer intento rechazó las discrepancias OHLC ya admitidas porque no transmitía la
tolerancia del corpus. El segundo no fijaba FP32 estricto antes de cargar el codificador.
Ambos errores se corrigieron y se conservaron sus registros. Los datos y los contratos de
la edición no se relajaron. La medida final registra pesos, versiones, precisión y huellas
de implementación. Otra ejecución previa tuvo tiempos diferentes bajo carga concurrente,
por lo que no se interpreta esa diferencia como un efecto del cambio de metadatos.

Las vistas emparejadas superan 41 pruebas y 11 mutaciones dirigidas. Su
[cobertura y CRAP](views-quality.json) registran 118/119 sentencias, 39/40 ramas y CRAP
máximo 28. Reutilizan los archivos originales de etiquetas y sus fingerprints, sin
recalcular objetivos. Las dos ventanas conservan los mismos recuentos y cortes.

El benchmark y su integración superan 38 pruebas CPU. Se detectaron las
[cinco mutaciones dirigidas](benchmark-mutations.json) de tolerancia, precisión, sobrescritura
y cambio de fuente. El [diagnóstico del benchmark](benchmark-quality.json) utiliza
coverage.py 7.16.1 y Radon 6.0.1. La cobertura procede de pruebas CPU y no suma los recorridos
CUDA ejecutados sin instrumentación. Los grupos de pruebas de este informe se solapan.
La comparación predictiva queda pendiente de su estudio identificado.

## Ampliación a 256 activos

La [selección ampliada](cohort256.json) contiene 943.780 filas y conserva los 64 activos
anteriores. Los Parquet de muestras de esos 64 activos tienen exactamente los mismos
hashes en ambas selecciones. Las 256 representaciones geométricas ocupan 1.008.762.744 bytes
frente a los 3.169.913.544 bytes de los Parquet de muestras originales seleccionados.
Son tamaños de muestras, sin sumar etiquetas ni toda la edición histórica.

La derivación completa tardó 176,02 segundos de proceso y alcanzó 254.722.048 bytes de RSS.
Se usaron bloques de 128 filas, sin cargar un modelo ni utilizar CUDA. Las vistas emparejadas
reutilizan las etiquetas existentes y conservan 63.679 filas de evaluación en 2023.
No se añaden filas ficticias para completar 256 activos por sesión. Los indicadores de
edición completa y disponibilidad para la campaña siguen siendo falsos.

La comparación entre tamaños debe publicar también el error sobre los 64 activos comunes.
El promedio de 256 activos describe otra población y no permite atribuir por sí solo un
cambio de MAE a disponer de más datos de entrenamiento. Esta preparación no contiene
resultados de aprendizaje ni habilita el test final.
