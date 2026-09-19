# Entrenamiento sobre el conjunto completo con memoria acotada

MARS-TITAN se prepara para recorrer toda la copia de FinMultiTime por bloques. La comparación principal se plantea inicialmente con hasta 128 activos estadounidenses, precedida por un piloto de hasta 64, para equilibrar repetición experimental y tiempo de entrega. El recorrido completo de entrenamiento queda como extensión de escalabilidad cuando el coste medido lo permita. No es una condición para ejecutar cada ablación.

«Todo el dataset» no significa entrenar con el test, aceptar información futura ni cargar todos los archivos en RAM. La cobertura debe informarse mediante un inventario de admisiones y exclusiones. Los datos sin disponibilidad defendible se conservan en el inventario y se excluyen del contraste estricto o se estudian en una variante identificada.

## Lo que se ha comprobado en el equipo

La [preparación ejecutada](../data/preparation.md) y el [presupuesto de almacenamiento](../../reports/resources/storage-budget.md) actualizan esta planificación. Se han generado Parquet de los paneles técnicos y del piloto de verificación, no de todo el corpus. El lector usa lotes de 256 filas y construye las ventanas al consumirlas. Las cifras ilustrativas siguientes no deben sustituir las mediciones de esos informes.

El [registro de recursos](../../reports/resource-inventory.json) contiene las observaciones y las órdenes de comprobación. El equipo tiene 32 GB de RAM según su configuración declarada, de los que el sistema informa 30,09 GiB utilizables. Dispone de ocho núcleos físicos, 16 hilos lógicos y una RTX 4070 Max-Q de 8 GB, identificada por NVIDIA como Laptop GPU con 8.188 MiB. El volumen dispone de unos 396 GiB libres en el momento de la inspección.

La copia tiene 108,21 GiB de tamaño aparente. El valor anterior de unos 109 GiB corresponde al espacio ocupado redondeado. Las tablas concentran aproximadamente 84,59 GiB, frente a 2,17 GiB de series. Se han contado 4.213 CSV de precios estadounidenses y 810 chinos, con 20.593.172 líneas físicas entre todos. Si cada archivo contiene una cabecera, resultan 20.588.149 filas potenciales. Esa resta no sustituye la auditoría de filas, duplicados y disponibilidad.

La diferencia entre volumen bruto y tensor de entrenamiento es importante. Parte de las tablas puede repetir hechos entre presentaciones, el texto se transforma en representaciones y las imágenes necesitan ventanas válidas. No se presupone una tasa concreta de compresión ni de eliminación de duplicados.

## Dimensionamiento transparente

Los siguientes valores son cálculos aritméticos, no consumos medidos de una implementación.

| Representación ilustrativa | Tamaño de los valores |
| --- | ---: |
| 20.588.149 filas × 64 variables × float32 | 4,91 GiB |
| Las mismas filas × 256 variables × float32 | 19,63 GiB |
| Ventanas materializadas de 64 pasos y 256 variables | 1.256,60 GiB |
| Estado de 5.023 activos × 2 capas × 128 valores × bfloat16 | 2,45 MiB |
| Una matriz 64 × 64 en float32 por cada uno de 5.023 activos | 78,48 MiB |
| Una matriz 256 × 256 en float32 por activo | 1,23 GiB |
| 8.192 episodios × 128 valores × bfloat16 | 2 MiB |

Se añadirán índices, máscaras, metadatos, claves, gradientes, optimizadores, copias y fragmentación. Una memoria neuronal entrenable por activo puede multiplicar varias veces su tamaño de pesos. Por eso se propone compartir parámetros y mantener estados pequeños por activo. Las ventanas se construyen al leer o se representan mediante índices. No se guardan millones de copias solapadas.

## Recorrido completo y preparación

La canalización prevista tiene cinco pasos, cada uno con memoria limitada:

1. Inventariar todos los archivos y fijar su origen. Separar mercado, instrumento, periodo, modalidad y evidencia temporal. Leer JSON y JSONL de forma incremental sin deserializar los 85 GiB de tablas en una lista de objetos.
2. Normalizar identificadores y hechos contables conservando periodo, unidad, presentación, revisión y disponibilidad. Deduplicar solo hechos equivalentes. Una revisión con otro valor es otro evento.
3. Escribir bloques columnares con esquemas estables. Elegir particiones por mercado y periodos suficientemente grandes. Evitar un archivo por activo y día, así como un único archivo que impida reanudar el trabajo.
4. Extraer representaciones de texto e imágenes por lotes, con modelos congelados y revisión registrada. La caché depende de contenido, versión del codificador y regla temporal. Cada imagen regenerada termina en el corte permitido.
5. Construir índices de muestras y bloques temporales de cada fold. Conservar normalización, selección y calibración separadas del test. Registrar todos los ejemplos elegibles procesados en una pasada completa.

Se propone comenzar con bloques de 64 a 256 MiB como punto de medición, no como tamaño óptimo declarado. El proyecto debe poder reanudar una preparación interrumpida a partir del manifiesto, sin reescribir los originales ni volver a calcular todos los embeddings.

La capacidad libre de disco también limita el diseño. Antes de una transformación grande se estimará el máximo simultáneo de originales, derivados, cachés, checkpoints y archivos temporales. El espacio observado hoy no autoriza materializar ventanas de más de un TiB.

## Cómo se entrena cada familia con todos los datos elegibles

| Familia | Estrategia de recorrido completo | Coste y límite que se debe declarar |
| --- | --- | --- |
| Cero y media histórica | Acumulación por bloques con corte temporal. | No necesitan una matriz de características completa. |
| Ridge | Acumular estadísticas suficientes del tramo permitido y resolver el sistema regularizado. | Mantiene una matriz de tamaño proporcional a F². Requiere normalización y control de condicionamiento, con referencia numérica de precisión suficiente. |
| SGD regularizado | Actualizaciones incrementales sobre bloques. | Es otra referencia, no una solución idéntica a Ridge. Registrar orden, épocas y convergencia. |
| XGBoost con memoria externa | Usar una matriz externa diseñada para leer bloques de características. | La caché, las etiquetas y otras estructuras siguen consumiendo recursos. Medir construcción, RAM, VRAM, disco e intercambio por PCIe. |
| Random Forest o HistGradientBoosting convencionales | Mantenerlos como controles del piloto si su implementación requiere materializar el conjunto. | No atribuirles una comparación sobre el conjunto completo si realmente usan una muestra. No agregarlos al ranking principal completo bajo otro presupuesto sin explicarlo. |
| GRU, TCN y DLinear compactos | Lotes o fragmentos temporales con generación de ventanas al vuelo. | Distinguir entrenamiento por ventanas independientes de entrenamiento con estado persistente. |
| SSM, regla delta y atención compacta | Fragmentos y algoritmos por bloques compatibles con cada arquitectura. | Medir el kernel disponible y la forma del problema. Una complejidad lineal no garantiza menor tiempo real. |
| MARS-TITAN | Compartir parámetros, consultar una instantánea común por cohorte y actualizar estados autorizados. | El estado global impide barajar sin más todas las fechas. Una época nueva no hereda el estado final de la anterior si ese estado contiene futuro respecto a su comienzo. |

La [guía de scikit-learn](https://scikit-learn.org/stable/computing/scaling_strategies.html) documenta el aprendizaje incremental. La [guía de XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html) distingue una matriz externa de una matriz cuantizada que termina concatenada en memoria. Se fijará la versión al implementar y no se asumirá compatibilidad sin prueba.

Los entrenamientos neuronales y las operaciones compatibles se programarán para `cuda:0`. Una referencia que se ejecute deliberadamente en CPU tendrá esa elección documentada y su coste medido. Un fallo de CUDA nunca cambiará silenciosamente el dispositivo.

## Paralelismo sin alterar el reloj

El paralelismo principal se encuentra entre lectura de bloques, preparación, transferencia y cálculo, además del lote de activos que comparten un corte. El eje temporal de una memoria dependiente del estado anterior no se paraleliza como si fueran muestras independientes.

La configuración inicial que se medirá usa de dos a cuatro trabajadores de datos, prelectura de uno o dos lotes y un límite explícito para memoria fijada. Se evita que cada trabajador lance a su vez 16 hilos de BLAS. Los valores se compararán con uno y ocho trabajadores para elegir según caudal, latencia y memoria total. No se maximiza el número de procesos como sustituto de medir rendimiento.

La [documentación de PyTorch](https://docs.pytorch.org/docs/2.14/data.html) permite concretar el reparto de iteradores. Cada registro debe tener un propietario para no duplicarse entre trabajadores. Las cohortes se forman por disponibilidad en UTC y los estados globales se publican solo según la política fijada. Las colas son acotadas y aplican contrapresión cuando el consumidor se retrasa.

## Dos mercados y dos calendarios

Estados Unidos y China se evaluarán primero por separado. Una fecha natural no alinea sus cierres. Un entrenamiento compartido deberá conservar el orden real de información y comunicar los pesos de muestreo, para que el mercado con más filas no determine por sí solo el resultado global.

La etiqueta y la simulación deben atender al instrumento y a las reglas que estaban vigentes en la fecha estudiada. El esquema estadounidense apertura-cierre no se traslada automáticamente a acciones chinas con restricciones de venta en el día de compra. La evaluación económica china requerirá calendario, disponibilidad de posiciones cortas, límites y versiones normativas históricas verificadas. La [revisión macrofinanciera](../references/macro-review.md) documenta las fuentes y sus límites.

## Qué permitirá declarar la cobertura completa

Se deberá registrar, por mercado y fold, número de archivos inventariados, registros leídos, aceptados, excluidos y finalmente usados, con motivos de exclusión. Una pasada deberá alcanzar el 100 % de los registros elegibles asignados a ese tramo, sin recurrir al test para completar entrenamiento.

El piloto valida que el recorrido cabe en recursos y conserva el tiempo. Una ejecución que se presente como completa deberá repetirlo sobre toda la cobertura elegible del universo declarado. La campaña principal puede cerrarse sobre la selección de 128 activos, con sus límites explícitos. Si una familia no cabe o no termina dentro del presupuesto fijado, se informa como límite de esa solución. No se oculta mediante una muestra distinta ni se atribuye el fallo a todo el proyecto.
