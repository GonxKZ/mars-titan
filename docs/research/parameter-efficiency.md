# Capacidad predictiva y coste por parámetro

Este documento diseña futuros contrastes de representaciones compactas, reutilización de parámetros, destilación y selección de memoria. Amplía una parte de [MT-028](https://github.com/GonxKZ/mars-titan/issues/28). No implementa el candidato MARS-TITAN, no inicia entrenamientos y no sustituye la matriz completa de ablaciones y falsaciones pendiente de esa tarea.

El diseño concreta el presupuesto de memoria de O3, los controles comparables de O4 y los límites de interpretación de O6. Se aplica junto al [protocolo](protocol.md), la [arquitectura candidata](candidate-architecture.md), el [registro experimental](experiment-matrix.md) y los [criterios de los objetivos](roadmap.md). Las cuatro modalidades de FinMultiTime y el contexto macro se conservan en todas las condiciones.

## Qué se contará

Un bloque compartido que se aplica cuatro veces almacena una sola copia de sus parámetros, pero realiza cuatro pasos de cálculo. Un banco de episodios puede ocupar más memoria sin añadir parámetros entrenables. La matriz de pesos rápidos de la referencia delta es estado adaptable dentro del tramo evaluado, distinto de los parámetros compartidos congelados. Estos objetos se contabilizarán por separado.

| Medida | Definición para el contraste |
| --- | --- |
| Calidad predictiva | MAE residual como medida principal, agregado por sesión y mercado según el protocolo. Rank IC, MSE y calibración como medidas secundarias, con las mismas muestras y reglas de exclusión. |
| Retención útil | Diferencia de pérdida al consultar recuerdos elegibles frente a controles de lectura anulada y memoria uniforme, por antigüedad y contexto definidos antes de evaluar. Conservar más episodios no demuestra retener señal útil. |
| Parámetros únicos almacenados | Número de escalares distintos en los pesos del predictor, separando entrenables y congelados. Un tensor compartido se cuenta una vez. Se informan el total con codificadores y el subtotal del componente que cambia. |
| Bytes de pesos y estado | Bytes efectivos por tipo numérico de pesos, memoria adaptable, episodios únicos, índices, colas de etiquetas y metadatos. Se añaden por separado gradientes, optimizador, RNG, cachés y copias necesarias para recuperar una ejecución. |
| RAM y VRAM | Picos observados durante preparación, ajuste, inferencia y guardado, con RSS del proceso y memoria CUDA asignada y reservada diferenciadas. El tamaño del checkpoint en disco se informa aparte. |
| Latencia y caudal | Tiempo por cohorte de decisión y muestras por segundo, incluidos preparación, lectura, transferencias, predicción y escritura. p50, p95 y p99 requieren suficientes decisiones y el número observado acompaña cada percentil. |
| Tiempo total | Duración de proceso para preparar, ajustar, validar y recuperar, incluyendo codificación, reconstrucción de memoria y maestro cuando proceda. La inferencia aislada se informa como otro recorrido. |

Se registrarán tanto los parámetros lógicos únicos como sus bytes residentes y serializados. Una copia temporal, un doble búfer o un checkpoint con pesos repetidos ocupa bytes aunque no cambie el número lógico de parámetros. El estado por activo se dimensionará con el mismo universo en ambas condiciones. Ni los codificadores congelados ni la memoria externa quedarán fuera del coste total por no recibir gradiente.

La comparación buscará configuraciones que reduzcan recursos con una pérdida predictiva aceptable o mejoren la predicción dentro de un presupuesto fijado. El margen de pérdida aceptable se registrará con el piloto de desarrollo antes de seleccionar los contrastes confirmatorios. Sin ese margen y su incertidumbre, un menor número de parámetros no bastará para declarar una mejora. No se estimará «información por neurona» a partir de estas medidas ni se trasladarán resultados de modelos de lenguaje a la predicción financiera.

## Hipótesis y controles

Los cuatro contrastes parten de una misma referencia y cambian un mecanismo cada vez. No se ejecutará el producto cartesiano de todas las opciones. Las combinaciones solo pasarán al registro confirmatorio si los contrastes individuales justifican su coste durante desarrollo.

| Contraste | Hipótesis que se pondrá a prueba | Modificación y controles |
| --- | --- | --- |
| Representación compacta | Una representación de menor dimensión puede conservar calidad predictiva y reducir pesos, estado o tiempo. | Reducir la dimensión después de los mismos codificadores congelados. Comparar con la representación de partida y con una proyección aleatoria fija de la misma dimensión, con semilla y escala declaradas. Conservar modalidades, muestras, familia de cabeza, política de memoria y número de pasos. |
| Parámetros compartidos | Reutilizar el bloque de refinamiento puede ofrecer una combinación de calidad y coste mejor que almacenar bloques distintos. | Comparar K = 1, 2 y 4 con pesos compartidos. Para cada K mayor que uno, añadir un control con bloques distintos de igual anchura y número de aplicaciones. Separar después los controles con igual número de parámetros o igual tiempo total. |
| Destilación | Un estudiante de K = 1 puede aprovechar las salidas de un maestro de K = 4 con menor coste de inferencia. | Comparar el estudiante con K = 1 ajustado directamente sobre los mismos datos y con un control que recibe el mismo número de actualizaciones adicionales sin objetivos del maestro. Incluir el maestro de K = 4 como referencia de calidad y coste. |
| Selección de memoria | La admisión selectiva puede conservar recuerdos más útiles que la selección uniforme con un presupuesto comparable. | Comparar la política candidata con reservorio uniforme, retención reciente y memoria sin escritura durante la evaluación. Mantener representación, lectura y K. Ablacionar error maduro, anomalía y relevancia por separado. La diversidad es una extensión identificada. |

### Representación compacta

Las dimensiones se elegirán con el piloto, dentro del presupuesto de hardware, y se fijarán antes del test. El dimensionamiento inicial de la arquitectura propone rasgos base de 256 y claves de 128. Esos valores no son resultados ni límites que hayan demostrado conservar señal.

Una proyección aprendida, su normalización y la selección de variables se ajustarán exclusivamente con entrenamiento. La validación real seleccionará la configuración, sin reajustar esas transformaciones con sus filas. Se informará el coste de aprender la proyección y el espacio ahorrado en cada componente. Las capas que deban adaptar su entrada a la nueva dimensión formarán parte del cambio contabilizado. Cambiar la dimensión no autoriza retirar fundamentales, gráficos, noticias, precios ni contexto macro.

Cada ventana congela la revisión de representación. Cambiar una revisión exige reconstruir o migrar la memoria compatible y comprobarla antes de usarla. El tiempo, las copias y los bytes de esa operación forman parte del contraste. Si una mejora procede de reducir la capacidad de memoria o las muestras procesadas, se identificará como otro cambio y no como efecto aislado de la representación.

### Reutilización de parámetros

Cada K cuenta lecturas y aplicaciones del bloque, como en la arquitectura candidata. Las iteraciones consultan la misma instantánea de memoria y no escriben ni reciben etiquetas entre pasos. El orden de activos y el fraccionamiento en microlotes no deben cambiar esa instantánea.

El control de bloques distintos con igual anchura aísla la compartición bajo una estructura de cálculo comparable, aunque almacena más parámetros. Igualar parámetros puede exigir cambiar anchura o profundidad. Igualar tiempo puede exigir cambiar el número de aplicaciones. Esos controles responden a preguntas de presupuesto diferentes y se registrarán como tales. No se afirmará haber igualado parámetros, capacidad de estado y cómputo mediante una única comparación que cambie varias cosas.

Se medirán el coste del desenrollado durante ajuste, las activaciones guardadas, los límites de norma, las salidas no finitas y las oscilaciones del estado. K mayor que uno no demuestra por sí mismo más retención, capacidad de razonamiento ni convergencia. El primer contraste usa K fijo y no añade una puerta adaptativa sin una pregunta separada.

### Destilación

Maestro y estudiante solo aprenderán con el tramo de entrenamiento autorizado. El maestro se congelará antes de generar sus objetivos para el estudiante. Se registrarán su configuración, pesos, corte y coste completo. El estudiante conservará las mismas modalidades y etiquetas reales, y su calibración usará el tramo separado del protocolo.

Las salidas del maestro en entrenamiento pueden ser objetivos de ajuste, pero no se presentarán como predicciones históricas fuera de muestra. Tampoco sustituirán el error de una predicción emitida para decidir qué habría escrito la memoria en el pasado. Si una variante necesita esas señales históricas, las generará con cortes internos cronológicos y solo con información disponible en cada corte.

La mezcla de pérdida supervisada y objetivo de destilación, así como su presupuesto de búsqueda, se fijará en desarrollo. La comparación distinguirá el coste inicial de maestro y estudiante del coste posterior de servir solo al estudiante. Una reducción de latencia de este último no acreditará un ahorro total si el ajuste adicional lo supera para la carga declarada.

### Selección de memoria y retención

La capacidad inicial de 8.192 referencias y hasta ocho episodios únicos recuperados por consulta procede del diseño candidato, no de una comprobación ejecutada. Se registrarán por separado referencias de índices, episodios únicos, bytes y candidatos examinados. Un mismo evento presente en varios índices comparte un registro y se deduplica antes de leerlo.

Los controles reciben el mismo flujo de candidatos maduros. Se igualarán el límite de bytes, el número máximo de episodios únicos recuperables y el presupuesto de escritura. Cuando no puedan igualarse simultáneamente, se fijará un límite común de bytes y se informarán las diferencias de ocupación y trabajo. Examinar todos los candidatos para rechazar la mayoría también tiene coste. La mezcla de reservorio, selección y recencia se comparará con cada control sin convertir tres actualizaciones de índice en una sola escritura equivalente.

El análisis de retención usará predicciones pareadas sobre sesiones posteriores a la admisión de los recuerdos. La antigüedad y los contextos se definirán con información disponible y reglas fijadas en entrenamiento. Para cada instante se evaluará una copia de la misma instantánea con lectura normal y otra con lectura anulada, sin cambiar pesos ni incorporar las etiquetas de las sesiones que se están prediciendo. Las copias de diagnóstico no alterarán la trayectoria principal. El control de memoria uniforme seguirá además su propia trayectoria causal completa. Se informará `MAE_control − MAE_variante` junto al número de sesiones y recuerdos elegibles, de modo que un valor positivo indique menor error de la variante.

El efecto de anular la lectura puede incluir una distribución de entradas distinta de la usada al ajustar. Por ello no bastará ese diagnóstico para atribuir una mejora al selector. Harán falta el control uniforme entrenado con la misma política de lectura y la comparación con memoria congelada. La recuperación de identificadores en mundos de mecanismo conocido será un diagnóstico adicional, separado del error sobre validación financiera real. El olvido de contextos anteriores y la adaptación a contextos nuevos se informarán por separado.

## Condiciones comunes y falsaciones

Cada par utilizará el mismo manifiesto, universo admisible, objetivo residual, disponibilidad de datos, cortes, calibración y política de evaluación. Las cuatro modalidades y macro se comprobarán en cada muestra. No se reducirá la población para favorecer una configuración. Los mundos sintéticos y el remuestreo solo podrán ampliar entrenamiento. La selección y el criterio predictivo principal se calcularán sobre validación real. El test final permanecerá cerrado.

Se conservarán las semillas 42, 43 y 44 para los finalistas, con la misma correspondencia entre condiciones. El cribado seguirá el presupuesto de búsqueda registrado y guardará los fallos y descartes. Los pares compartirán inicialización de los componentes compatibles y orden de datos cuando el contraste lo permita. Las semillas no se tratarán como tres mercados independientes. Las diferencias se estimarán por bloques de sesiones que mantengan juntos los activos, con longitud de bloque y sensibilidad fijadas en desarrollo.

Los parámetros compartidos, representaciones, umbrales y política de lectura quedarán congelados dentro de la evaluación. Solo el estado declarado podrá actualizarse con la regla registrada. Cada sesión emitirá todas sus predicciones desde el estado previo, conservará esas salidas y después procesará etiquetas maduras en orden canónico. Cada ventana reiniciará la memoria y su optimizador interno, con calentamiento exclusivamente anterior al corte.

| Falsación | Comprobación exigida antes de interpretar el contraste |
| --- | --- |
| Sufijo futuro | Cambiar entradas y etiquetas posteriores a un corte no altera predicciones ni estado confirmado hasta ese corte. |
| Etiquetas inmaduras | Cambiar o retirar una etiqueta pendiente no cambia ninguna decisión anterior a su maduración ni la puntuación registrada antes de que sea elegible. |
| Permutación de activos | Permutar el orden de carga conserva las predicciones de la cohorte, el estado posterior y las decisiones siguientes mediante el orden canónico del protocolo. |
| Permutación de etiquetas | Un control negativo permuta etiquetas solo dentro de entrenamiento con una semilla registrada. Se vuelve a ajustar cada condición con el mismo presupuesto y se mide en validación real intacta. Una ventaja parecida a la original exige investigar selección, fuga o señal no destruida por esa permutación. No se exige que todas las métricas sean exactamente cero. |
| Memoria vacía o congelada | La ausencia de recuerdos devuelve la máscara prevista y una salida válida. El control congelado no incorpora eventos posteriores al calentamiento. Los estados y recuentos deben acreditar ambas condiciones. |
| Interrupción y recuperación | Una ejecución continua y otra interrumpida recuperan las mismas predicciones y el mismo estado bajo la política numérica declarada. Se incluyen pesos, memoria, índices, colas, salidas emitidas, RNG, optimizador, cursor y contadores confirmados. |

La tolerancia numérica se fijará antes de comparar recuperaciones. Se exigirá identidad exacta cuando la configuración determinista la admita y se documentará cualquier tolerancia necesaria. La corrupción de un checkpoint o una revisión incompatible deben impedir su uso o recuperar un estado anterior íntegro según el [contrato de recuperación](../engineering/checkpoint-recovery.md). Una falsación temporal fallida invalida el contraste aunque mejore el MAE.

## Registro y decisión

Cada contraste se registrará antes de ejecutarse con una única pregunta, referencia, modificación, población, presupuesto de búsqueda, criterio principal y regla de interpretación. Se conservarán configuraciones, versiones, semillas, huellas de datos y código, conteos de parámetros, tipos numéricos y política de memoria. Los informes separarán medidas observadas de presupuestos configurados.

Las medidas de coste usarán la misma carga declarada, hardware, precisión, límites de hilos y política de concurrencia. Se distinguirán arranque, calentamiento, régimen estable y proceso completo. Se repetirán las medidas y se publicará su dispersión, junto con tamaños de entrada, bytes leídos y transferidos, asignaciones y cargas concurrentes. Las mediciones CUDA sincronizarán el tramo que corresponda sin confundir tiempo interno con tiempo total. Energía y coste monetario solo se informarán si pueden medirse con un procedimiento reproducible.

Una variante pasará a la comparación confirmatoria si conserva la calidad dentro del margen registrado y reduce un recurso relevante, o mejora la calidad dentro del presupuesto acordado. Las compensaciones entre latencia, memoria y tiempo de ajuste se mantendrán visibles. Un ahorro sin paridad predictiva, una ventaja explicada por más datos o una mejora que desaparece con controles comparables no sostendrán la hipótesis inicial.

La ausencia de mejora también se registrará. Este documento no acredita resultados de O3, O4 ni O6 y no cierra MT-028. Quedan pendientes la matriz completa, la implementación del candidato, las falsaciones ejecutadas y las campañas comparables.
