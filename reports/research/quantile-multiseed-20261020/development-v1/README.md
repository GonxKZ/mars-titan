# Rutas de cuantiles con cinco semillas

Se completaron los 25 ajustes del [protocolo v2](../../../../docs/experiments/quantile-routing-v2.md), con cinco configuraciones y semillas 42–46. La combinación reduce el MAE medio un 21,30 % frente a la referencia, pero queda un 7,32 % por encima del residual cero. Aislar solo la señal de dispersión hacia el codificador obtiene menor MAE que la combinación. No se promueve ninguna variante ni se acredita una mejora frente al estado del arte.

## Población y contraste

Se conservan 64 activos estadounidenses, fold-018 y todas sus filas admisibles desde 2000 con máscaras. Hay 203.350 filas de ajuste, 8.000 de validación, 3.968 de calibración y 15.936 de desarrollo. Este último tramo corresponde a las mismas 249 sesiones de 2023 en todos los casos. Cada modelo ejecuta una época y 685 actualizaciones, con la receta `lr1e-3`, K=1, proyecciones del artículo, disposición temporal contigua y banco episódico desactivado. Los estados se reinician en cada fase y se calientan solo con entradas anteriores permitidas.

El error se calcula primero por sesión en cada semilla y después se promedian sus cinco errores. Las cifras siguientes no proceden de promediar predicciones ni de un ensemble. La desviación típica es muestral, con divisor cuatro. El [resultado completo](results.json) conserva también cada semilla y sus métricas de dirección, rangos y cuantiles. El residual cero tiene MAE 0,01584945 y no depende de la semilla.

| Configuración | MAE medio por sesión | Desviación entre semillas | Diferencia frente al original | Más error que cero |
| --- | ---: | ---: | ---: | ---: |
| Original | 0,02161406 | 0,00484225 | 0 | 36,37 % |
| Ambas vías originales, control | 0,02161406 | 0,00484225 | 0 | 36,37 % |
| Ubicación aislada de las colas | 0,02738391 | 0,00272702 | +0,00576985 | 72,78 % |
| Dispersión aislada hacia el codificador | 0,01661728 | 0,00072115 | −0,00499678 | 4,84 % |
| Ambas vías aisladas | 0,01700958 | 0,00086600 | −0,00460448 | 7,32 % |

El contraste principal sigue siendo combinación menos original, como se fijó antes de ajustar. Su diferencia es −0,00460448 y el intervalo temporal del 95 % es [−0,00527537, −0,00400560]. Usa 2.000 réplicas circulares de bloques de 16 sesiones y semilla 20261020. Las sensibilidades predefinidas a bloques de 8 y 32 conservan el signo. Estos intervalos mantienen fijas las cinco semillas observadas. No estiman una población de nuevas inicializaciones ni corrigen la reutilización acumulada del desarrollo.

La combinación reduce el error en cuatro semillas y lo aumenta en la 46, en +0,00002283. La diferencia más favorable es −0,01252732 en la 45. La desviación muestral de las cinco diferencias es 0,00483293. La ventaja media no es una mejora consistente en todas las inicializaciones.

Los doce secundarios conservan intervalos simultáneos. La diferencia combinación menos cero es +0,00116013, con intervalo [0,00102634, 0,00129391]. La del aislamiento hacia el codificador menos cero es +0,00076783, con intervalo [0,00067528, 0,00086038]. Ambas pierden frente al control fuerte. La combinación añade +0,00039230 de error frente al aislamiento hacia el codificador, con intervalo [0,00020076, 0,00058384]. No se sustituye retrospectivamente el candidato principal por esa configuración.

La interacción en la escala de MAE, combinación menos cada componente más original, es −0,00537756, con intervalo simultáneo [−0,00679258, −0,00396254]. Ese signo depende de que aislar la ubicación por sí sola empeora mucho. No significa que combinar los factores supere a ambos componentes, ni identifica un mecanismo económico o cerebral.

## Qué se ha comprobado

Los [cinco controles unitarios](unity-controls.json) conservan exactamente los 45 tensores compartidos, la inicialización y las predicciones de validación, calibración y desarrollo dentro de cada semilla. Se verificaron también los bytes de tensores y de los Parquet, con tolerancia cero. La [repetición de 42](seed-42-parity.json) reproduce los cinco modelos de v1 con esos mismos criterios. Es una repetición de la misma ventana y no otra observación temporal.

La motivación de [separar vías de aprendizaje](../../../../docs/research/quantile-routing.md) utiliza una analogía limitada con compartimentos dendríticos. El mecanismo implementa gradientes sustitutos sobre la cabeza común. No reproduce neuronas ni demuestra novedad por utilizar stop-gradient. El diagnóstico anterior de una semilla es compatible con acercarse al predictor cero y no acredita más información útil. Esta ronda no contiene un nuevo diagnóstico causal de esa reducción. Una época tampoco permite decidir el comportamiento de convergencia o de un ajuste más largo.

La [verificación técnica focalizada](../verification.json) conserva 108 casos distintos y 16 mutantes dirigidos detectados, con cobertura y CRAP separados de los datos reales. No es toda la batería. Ejecutar el analizador sobre los artefactos reales no se suma a la cobertura de las pruebas unitarias.

## Coste y recuperación

Los [26 intentos nuevos](cost-summary.json) suman 3.559,714 segundos externos, unos 59,33 minutos. Incluyen 25 ajustes completos y un fallo de admisión antes de producir recibo interno. Los ajustes completos suman 17.125 actualizaciones y 3.447,141 segundos internos, ya incluidos en el tiempo externo. Los 64 intentos acumulados de estas rondas suman 8.478,750 segundos, dentro del máximo declarado de 14.400. El análisis añade 5,428 segundos de proceso. Estos totales no incluyen las pruebas técnicas, la investigación ni las pausas entre lanzamientos.

El intento fallido de `original-43` consumió 7,266 segundos. El supervisor detectó otra carga CUDA durante el relevo con comprobaciones técnicas y lo cerró antes de generar el recibo interno. Su log y recibo se conservan. Tiempo interno, pasos y memoria de ese intento permanecen como desconocidos, sin ceros inventados. Tras terminar las comprobaciones técnicas se repitió el trabajo, sin sobrescribir el fallo ni repetir los cinco ajustes ya completos de la semilla 42.

El RSS máximo de los ajustes completados fue de 2.096.893.952 bytes. PyTorch registró máximos de 206.327.808 bytes asignados y 236.978.176 reservados en CUDA. Se utilizó una RTX 4070 Laptop de 8 GiB, PyTorch 2.14.0+cu130 y FP32 estricto, con TF32 desactivado. Había escritorio gráfico y tareas CPU técnicas concurrentes. Los tiempos son costes reales, pero no una comparación aislada de aceleración. No se midió energía ni la VRAM completa del proceso a partir de estos contadores.

No cambia el recorrido de inferencia de la ronda anterior. Su [medida técnica CUDA](../../quantile-routing-20261020/README.md#coste-cuda-con-entradas-precodificadas) no mostró ahorro persistente ni aceleración de la combinación. No se presenta el número de variantes o la disponibilidad de GPU como una mejora de rendimiento.

El [manifiesto preparado](prepared.json) fija datos, configuración y fuentes. El [manifiesto del análisis](analysis-manifest.json) distingue el productor `828d7a9d40f5696d53536b0a6455655d167ad788` del consumidor `270ff4bcd65406819e07aaaa8de65aace938e16d`, cuya corrección solo afecta a costes sin recibo interno. Se verificó la preparación sin relajar las guardias. El [inventario](artifacts.json) identifica predicciones por muestra, métricas por sesión, checkpoints, manifiestos, logs y recibos conservados fuera de Git.

El [cierre](closure.json) confirma todos los intentos y autorizaciones locales cerrados y la protección global intacta. No queda un ajuste de esta ronda ejecutándose en segundo plano. La campaña completa, el test final de 2024, el contraste escalar frente a cuantiles y la aceptación de finalistas de [#22](https://github.com/GonxKZ/mars-titan/issues/22) siguen pendientes de sus condiciones propias.
