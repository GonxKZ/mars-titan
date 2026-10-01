# Auditoría de población, particiones y resultados

La edición estadounidense revisada contiene 1.548.307 muestras de entrenamiento y 261.069 de validación. La auditoría no encontró discrepancias en los recuentos, el orden temporal ni las disponibilidades declaradas de las entradas. El recálculo de los 159 archivos de predicciones de validación confirmados también coincidió con sus informes. Estos resultados no acreditan por sí solos la procedencia externa de cada dato ni una mejora predictiva.

El corte de resultados es el 26 de septiembre de 2026 a las 18:04:32 UTC. Incluye 80 ejecuciones neuronales, 17 tabulares y 62 adaptadores. Los agregados y las huellas de evidencia están en el [informe JSON](../../reports/data/population-and-results-audit.json). El procedimiento de recálculo se describe en la [revisión incremental de predicciones](../engineering/prediction-review.md).

## Población y divisiones temporales

La preparación estadounidense recorre 4.784 candidatos. Hay 2.639 activos preparados y 2.145 sin todas las fuentes necesarias. De los preparados, 413 no producen muestras temporalmente admitidas. Los 2.226 restantes aportan 1.816.369 muestras antes de crear las etiquetas. Se excluyen 6.993 filas: 5.777 por historia insuficiente, 811 porque su objetivo cruza la frontera entre particiones y 405 por falta de la sesión siguiente.

| Partición | Predicciones observadas, UTC | Disponibilidad del objetivo, UTC | Filas | Sesiones | Activos |
| --- | --- | --- | ---: | ---: | ---: |
| Entrenamiento | 24/07/2009 20:05 a 29/12/2022 21:05 | 27/07/2009 20:05 a 30/12/2022 21:05 | 1.548.307 | 3.335 | 2.055 |
| Validación | 03/01/2023 21:05 a 22/12/2023 21:05 | 04/01/2023 21:05 a 26/12/2023 21:05 | 261.069 | 246 | 2.168 |

Las particiones comparten 1.999 activos y su unión contiene 2.224. Es una separación temporal, no una separación por empresas. El entrenamiento usa decisiones anteriores a 2023 y la validación usa 2023. El periodo desde 2024 sigue reservado como test. Esta edición solo materializa entrenamiento y validación, por lo que no hay un resultado de test ni una comparación ejecutada mediante múltiples ventanas walk-forward.

Los dos Parquet ordenados suman 7.442.027.072 bytes. Sus hashes coinciden con el manifiesto. No se detectaron claves repetidas o fuera de orden, objetivos no finitos, entradas disponibles después de la predicción ni etiquetas que cruzaran la frontera de su partición. Estas comprobaciones contrastan los campos registrados de disponibilidad. No reconstruyen la historia de cada proveedor.

## Modalidades, cobertura y procedencia

Las muestras conservan precios, noticias, gráficos y fundamentales, junto con contexto macroeconómico. Las ventanas de precios y gráficos contienen 64 sesiones y las noticias se admiten con una ventana de cinco sesiones. Los vectores tienen las siguientes dimensiones en ambas particiones.

| Entrada | Dimensión | Alcance de la comprobación |
| --- | ---: | --- |
| Precios | 320 | Ventana numérica con dimensión constante y valores finitos. |
| Noticias | 384 | Representación de las noticias admitidas y disponibilidad anterior o igual a la decisión. |
| Gráficos | 512 | Representación de velas generadas con las ventanas OHLC de precios. |
| Fundamentales | 45 | Quince conceptos con valor, máscara y antigüedad. |
| Macro | 420 | Ciento cuarenta indicadores con valor, máscara y antigüedad. |

No se encontraron vectores ausentes, dimensiones incorrectas, valores no finitos ni vectores enteramente nulos. Eso no significa que todos los conceptos estén observados. Las máscaras distinguen ausencia de un cero observado. Los valores y las antigüedades de las posiciones ausentes son cero y no se encontraron máscaras inválidas.

Aparecen 14 de los 15 conceptos fundamentales al menos una vez. Por fila hay entre 2 y 14 conceptos en entrenamiento y entre 3 y 14 en validación. La antigüedad máxima aproximada alcanza 4.914 días en entrenamiento y 5.272 en validación. Una fecha anterior a la decisión acredita el orden temporal declarado, pero no que el dato siga siendo reciente.

De los 140 indicadores macro del catálogo, 125 aparecen al menos una vez y 15 permanecen ausentes. Por fila se observan entre 91 y 125 en entrenamiento, con una media de 117,24, y entre 123 y 125 en validación, con una media de 124,96. Los 15 ausentes tienen causas registradas: cuatro indicadores de petróleo o sus derivados fallan por intervalos temporales inválidos al ingerir ALFRED, seis series chinas se excluyen por falta de versiones históricas acreditadas, dos indicadores de presión de suministro carecen de fecha de publicación verificada y tres de condiciones o estrés financiero se excluyen por la política `model_vintages_only`.

Todas las filas admitidas contienen noticias clasificadas como `article_candidate`. Ninguna contiene `verified_full_article`. La auditoría comprueba su tipo y disponibilidad registrados, sin verificar que cada texto sea un artículo íntegro ni contrastarlo de nuevo con la página original. Los gráficos son una transformación de OHLC y no añaden una fuente independiente de precios. La representación registra `historical_simulation=false`, por lo que no se presenta como una reproducción de los codificadores disponibles en cada fecha histórica. Tampoco se ha reconstruido aquí el efecto de los ajustes retrospectivos de precios.

## Métricas y estados seleccionados

`mae` divide la suma de errores absolutos entre las filas. `session_mae` calcula el MAE de cada par de mercado e instante y después da el mismo peso a cada sesión. Las tablas usan este segundo valor para comparar validación. El MAE de entrenamiento y validación publicado al final corresponde al mismo estado congelado. El MAE acumulado durante una época de entrenamiento, en cambio, combina predicciones de pesos que van cambiando.

Las 80 ejecuciones neuronales incluyen 48 configuraciones de búsqueda, 8 finalistas adicionales y 24 continuaciones. La siguiente tabla usa las semillas 42, 43 y 44 de la configuración seleccionada por familia. La desviación estándar es muestral, calculada entre semillas, y no es un intervalo de confianza. Ridge tiene una sola ejecución seleccionada.

| Familia | MAE train por filas | MAE validación por filas | MAE validación por sesión | Desviación entre semillas |
| --- | ---: | ---: | ---: | ---: |
| rnn | 0.01434517 | 0.01505531 | 0.01498065 | 0.00000347 |
| lstm | 0.01435407 | 0.01505625 | 0.01498190 | 0.00000512 |
| gru | 0.01435117 | 0.01505621 | 0.01498171 | 0.00000231 |
| dlinear | 0.01435226 | 0.01505599 | 0.01498158 | 0.00000228 |
| ridge | 0.01434211 | 0.01576844 | 0.01567666 | No disponible |
| xgboost | 0.01427608 | 0.01512135 | 0.01504729 | 0.00000000 |

El control de predicción cero tiene un MAE de validación por sesión de 0,0149903752. Las medias neuronales lo reducen entre un 0,0565 % y un 0,0649 %. No se han calculado intervalos ni contrastes por sesiones que permitan atribuir significación estadística a esas diferencias. Las tres semillas de XGBoost coinciden en las métricas publicadas.

En el orden de semillas 42, 43 y 44, las épocas seleccionadas son RNN 4/4/6, LSTM 7/1/1, GRU 3/4/4 y DLinear 3/3/6. Los doce finalistas terminaron cinco épocas después de su mejor validación. El JSON conserva los resultados individuales por semilla y las medias de cada grupo completo.

## Continuaciones y postajustes

Las 24 continuaciones neuronales MAE/MSE ejecutan cinco épocas y publican la última, sin selección ni parada temprana. Todas terminan con peor validación que el padre de su misma semilla. En 20 de 24 baja el MAE de entrenamiento del estado final. Esta combinación es compatible con sobreajuste a entrenamiento.

| Familia | Pérdida | MAE train por filas | MAE validación por sesión | Cambio frente al padre |
| --- | --- | ---: | ---: | ---: |
| rnn | mae | 0.01428864 | 0.01499359 | +0.00001293 |
| rnn | mse | 0.01431861 | 0.01501449 | +0.00003384 |
| lstm | mae | 0.01431597 | 0.01501810 | +0.00003619 |
| lstm | mse | 0.01433946 | 0.01507973 | +0.00009783 |
| gru | mae | 0.01430283 | 0.01499982 | +0.00001811 |
| gru | mse | 0.01433703 | 0.01504153 | +0.00005982 |
| dlinear | mae | 0.01427294 | 0.01501574 | +0.00003416 |
| dlinear | mse | 0.01431472 | 0.01506909 | +0.00008751 |

Las 24 curvas contienen una época anterior mejor que la publicada. En siete, una época previa incluso mejoró al padre. Es una observación retrospectiva. No se cambia el resultado confirmado ni se selecciona otro checkpoint después de observarlo.

Los 62 adaptadores comprenden 36 casos RNN y 26 LSTM. Cada método utiliza tres semillas del adaptador sobre un único padre de semilla 42. No son tres padres entrenados de forma independiente. RNN tiene doce grupos completos. En LSTM están completos REINFORCE y `expected`, mientras que MAE y los nueve grupos KLPO solo tienen las semillas 42 y 43 al corte. La tabla omite medias de estos grupos incompletos.

La selección del adaptador usa la mediana de su distribución discreta, que difiere del centro continuo y del padre continuo. Los padres tienen MAE de validación por sesión de 0,0149777758 para RNN y 0,0149774172 para LSTM. Un cambio positivo en la tabla indica empeoramiento de la mediana respecto al padre.

| Padre | Método | Beta | MAE validación por sesión | Desviación entre semillas | Cambio frente al padre |
| --- | --- | ---: | ---: | ---: | ---: |
| rnn | reinforce | No aplica | 0.01504405 | 0.00002708 | +0.443 % |
| rnn | expected | No aplica | 0.01504718 | 0.00003349 | +0.463 % |
| rnn | mae | No aplica | 0.01504012 | 0.00004462 | +0.416 % |
| rnn | klpo_full | 0.03 | 0.01698406 | 0.00058594 | +13.395 % |
| rnn | klpo_full | 0.1 | 0.01586276 | 0.00018250 | +5.909 % |
| rnn | klpo_full | 0.3 | 0.01528501 | 0.00002424 | +2.051 % |
| rnn | klpo_mc | 0.03 | 0.01697763 | 0.00057044 | +13.352 % |
| rnn | klpo_mc | 0.1 | 0.01585946 | 0.00017861 | +5.887 % |
| rnn | klpo_mc | 0.3 | 0.01528440 | 0.00003507 | +2.047 % |
| rnn | klpo_exact | 0.03 | 0.01641847 | 0.00043594 | +9.619 % |
| rnn | klpo_exact | 0.1 | 0.01548495 | 0.00007369 | +3.386 % |
| rnn | klpo_exact | 0.3 | 0.01520299 | 0.00009208 | +1.504 % |
| lstm | reinforce | No aplica | 0.01505968 | 0.00003915 | +0.549 % |
| lstm | expected | No aplica | 0.01505308 | 0.00003121 | +0.505 % |

Ninguno de los 62 estados seleccionados mejora la validación del padre, ni con la mediana discreta ni con el centro continuo. Los 62 empeoran asimismo el MAE de entrenamiento. Por eso no se puede atribuir toda su degradación únicamente al sobreajuste. El error esperado de una acción muestreada y los objetivos de entrenamiento describen cantidades diferentes del MAE de la mediana.

En 54 de 62, la última época tiene peor validación que la seleccionada. Se conservan 28 estados de época 1, siete de época 2, siete de época 3, doce de época 4 y ocho de época 5. Todos los casos confirmados completan cinco épocas y 30.245 actualizaciones.

El estado inicial no participa en la selección actual. El adaptador comienza con corrección cero, pero el selector solo compara las validaciones posteriores a las épocas 1 a 5. Su centro inicial coincide con el padre, mientras que la mediana discreta inicial no se ha evaluado en esta revisión. Una continuación futura deberá declarar cómo compite ese estado inicial con los posteriores usando el mismo criterio. La campaña activa conserva su protocolo.

## Comprobaciones ejecutadas y límites

Los 159 informes coinciden con los SHA-256 de sus resúmenes. Se recalcularon las 159 particiones de validación desde Parquet, con cero discrepancias y tolerancias absoluta de 10⁻¹² y relativa de 10⁻¹⁰. Se leyeron 1.955.042.570 bytes para sus hashes. El tiempo interno fue de 74,1347 segundos y el tiempo total del proceso, de 74,36 segundos. El pico de memoria residente fue de 140.890.112 bytes. La ejecución usó CPU y no cargó modelos ni abrió el test.

Esas medidas corresponden a la versión inicial del evaluador identificada en el JSON. El barrido posterior, con las defensas adicionales de rutas y población, terminó con 320 particiones verificadas de 160 ejecuciones, incluyendo entrenamiento y validación. No encontró discrepancias. Tardó 777,68 segundos internos y 778,04 segundos de proceso, con un pico RSS de 194.408.448 bytes. Las tablas conservan el corte original de 62 adaptadores. El recálculo posterior también respalda sus métricas de entrenamiento.

La auditoría temporal de las dos particiones terminó en 64,21 segundos, con un pico de memoria residente de 205.742.080 bytes y lotes de 512 filas. Se comprobaron los hashes completos de los Parquet ordenados. Los archivos por activo se conciliaron mediante metadatos, columnas temporales y etiquetas, sin recalcular todas sus huellas binarias.

La auditoría no ejecuta búsquedas, entrenamientos ni pruebas sobre el test reservado. Los resultados revisados pertenecen a la validación utilizada para seleccionar modelos y épocas. La cobertura temporal, la coincidencia de hashes y la consistencia numérica no sustituyen una evaluación final ni verifican por sí solas la procedencia externa de cada fuente.
