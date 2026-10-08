# Representación episódica fija

`memory/episodic_codec.py` convierte una `CPUDecisionBatch` verificada en claves
de entrada y valores de 64 coordenadas FP32. No recibe etiquetas, fechas de
maduración ni parámetros del predictor. La normalización final de las claves
corresponde al banco nativo. Este módulo no admite episodios ni publica sesiones.

Las entradas ya llevan las transformaciones del corpus. El lector expresa los
OHLC respecto al primer cierre de la ventana y transforma el volumen con la
media de esa misma ventana. Los contextos contables y macroeconómicos contienen
valores con `signed_log1p`, máscaras y edades con `log1p`. Los embeddings de
noticias y gráficos llegan preparados. El codec no vuelve a aplicar logaritmos
ni ajusta estadísticas con otras observaciones.

Cada modalidad se aplana y se divide por su norma L2, calculada en FP64. Un
bloque nulo queda nulo. Se aplica un peso positivo fijo por modalidad. Las cinco
marcas de presencia forman otro bloque, dividido por `sqrt(5)` y con un peso
propio. Así, ausencia y cero observado siguen siendo entradas diferentes.

La receta `frozen_modal_projection_v1` usa signos Rademacher generados por un
`PCG64` local. La proyección de claves tiene 63 filas, escaladas por
`1 / sqrt(63)`, y la de valores tiene 64, escaladas por `1 / sqrt(64)`. Una
coordenada constante positiva completa `key_inputs`. La reducción se realiza
en FP64, por modalidad y en orden fijo, antes de convertir a FP32. El resultado
es independiente del orden de las filas y de su división en lotes físicos
dentro del entorno identificado.

La identidad conserva el contrato completo de entradas, dimensiones, catálogos,
semilla, pesos, constante, receta, precisión, versión de NumPy y SHA-256 de la
proyección. La reconstrucción exige coincidencia canónica de los tipos JSON.
Las matrices y los resultados usan buffers respaldados por bytes inmutables.
`verify()` comprueba geometría y contenido antes de su consumo.

Los diagnósticos incluyen la norma de cada modalidad antes y después de
normalizar, y la norma de su contribución a claves y valores. No son porcentajes
de importancia del pronóstico. La suma contiene términos cruzados y la
proyección puede identificar observaciones diferentes. La futura retención M
usará distancia euclídea entre las claves normalizadas almacenadas, que induce
una pseudométrica sobre episodios con la misma representación. `1 - cosine`
no se presenta como esa métrica.

El presupuesto cubre la proyección residente y una estimación conservadora de
los buffers propios, con un máximo de 64 MiB. Un exceso se rechaza antes del
cálculo. No es un límite del RSS del proceso ni incluye los datos del llamante
o las bibliotecas cargadas.

Las comprobaciones focales usan 33 casos CPU, incluida una referencia escalar
independiente, extremos FP32, máscaras, permutaciones, reconstrucción, cambios
de contenido y presupuesto. Ocho mutaciones dirigidas se detectaron. Con
coverage.py 7.16.2 se ejecutaron 162 de 165 sentencias y 35 de 38 ramas. El CCN
máximo de radon 6.0.1 fue 16. Estas medidas describen las pruebas ejecutadas.

Una medición con fixtures de dimensiones `64×5`, `384`, `512`, `78` y `420`,
NumPy 2.5.3 y dos hilos obtuvo una mediana de 27,58 ms para 256 observaciones,
tras un calentamiento y cinco repeticiones. La proyección ocupaba 1.746.504
bytes y el pico rastreado adicional fue de 2.403.720 bytes, frente a una
estimación de 5.615.789 bytes. El tramo incluye verificación, proyección y
diagnósticos. La creación de la vista, el banco, la persistencia y el predictor
quedaron fuera de esa medición.

La integración inicial prevista conserva el contrato nativo de claves y valores
de 64 coordenadas y capacidad máxima de 1.024 episodios. Tiene identidad propia
y no sustituye al candidato de claves 128 y rasgos 256. Siguen pendientes la
admisión temporal, la lectura posterior a MAC y la publicación conjunta del
banco con el estado rápido y las predicciones pendientes.
