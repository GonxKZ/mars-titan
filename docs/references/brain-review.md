# Memoria biológica, aprendizaje continuo y recurrencia interna

Autor del proyecto: Gonzalo García Lama. Revisión documental: 18 de septiembre de 2026.

La evidencia revisada permite formular mecanismos concretos para MARS-TITAN: conservar experiencias seleccionadas, aprender a distintas velocidades, limitar interferencias y asignar un número acotado de iteraciones a cada predicción. No demuestra que su combinación elimine el olvido, reduzca todo el ruido o mejore los retornos residuales. La precisión y la latencia deben evaluarse conjuntamente, porque compartir pesos ahorra parámetros pero no elimina el trabajo de cada iteración.

Esta revisión amplía el [estado del arte neuronal](neural-review.md) con 24 entradas nuevas, registradas en [brain-sources.json](brain-sources.json) y [brain.bib](brain.bib). Se han contrastado fichas primarias, resúmenes y pasajes pertinentes de métodos y discusión. Esto no acredita una lectura integral de todos los artículos ni una revisión sistemática exhaustiva. Las barreras de acceso y las versiones de consulta constan en el registro. La preparación documental no constituye implementación ni experimento ejecutado. El [protocolo](../research/protocol.md) y la [hoja de ruta](../research/roadmap.md) delimitan la comparación.

## Qué puede tomarse del cerebro

El marco de sistemas de aprendizaje complementarios separa la incorporación rápida de episodios y la adquisición gradual de regularidades compartidas. Es una explicación computacional de una división funcional, no una especificación completa del cerebro. En una red financiera, una memoria de eventos y unos parámetros de cambio lento podrían cumplir funciones diferentes sin representar literalmente hipocampo y neocórtex. Esa correspondencia es una hipótesis de diseño que requiere ablaciones (McClelland et al., 1995). [Artículo de los autores](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf).

El replay tampoco es una copia exacta de un conjunto de entrenamiento. Wilson y McNaughton observaron reactivación de conjuntos neuronales en ratas durante sueño. Schuck y Niv encontraron secuencialidad compatible con replay de estados no espaciales en señales fMRI humanas durante reposo. Son evidencias de distinta especie, método y resolución temporal. Ninguna valida por sí sola una política de muestreo de noticias, una consolidación sin coste o un sistema financiero que deba permanecer actualizándose continuamente. [Wilson y McNaughton (1994)](https://doi.org/10.1126/science.8036517), [Schuck y Niv (2019)](https://doi.org/10.1126/science.aaw5181).

| Mecanismo | Evidencia y límite de observación | Traducción algorítmica que se podría probar | Qué no avala | Ablación informativa |
| --- | --- | --- | --- | --- |
| Aprendizaje complementario | Modelo computacional y síntesis neuropsicológica de McClelland et al. (1995). | Estado rápido y parámetros lentos con velocidades fijadas. | Que dos módulos reproduzcan dos regiones cerebrales. | Una velocidad frente a dos, igualando capacidad y actualizaciones. |
| Reactivación y consolidación | Registros en tres ratas de Wilson y McNaughton (1994). fMRI humano de Schuck y Niv (2019). | Buffer histórico limitado y replay de ejemplos ya disponibles. | Que repetir muestras garantice generalización o sea gratuito. | Sin replay, replay uniforme y replay selectivo con igual número de ejemplos. |
| Separación de patrones | Bakker et al. (2008), fMRI humano en CA3/DG. La resolución no separa todos los mecanismos celulares. | Representaciones o claves que reduzcan colisiones entre eventos. | Que dividir la memoria por activos sea necesariamente mejor. | Memoria compartida frente a particionada, con igual capacidad total y medición de interferencia. |
| Control de recuperación | Gagnepain et al. (2014), supresión de recuerdos visuales y cambios en priming humano. | Atenuar o retirar experiencias según una regla explícita. | Borrado exacto, olvido siempre beneficioso o aplicación clínica. | FIFO, muestreo uniforme y retención selectiva con el mismo tamaño. |
| Error de predicción de recompensa | Schultz et al. (1997), actividad dopaminérgica en primates no humanos y modelo computacional. | Usar un error observado para modular una actualización. | Equivalencia entre dopamina, novedad textual y beneficio futuro. | Error maduro frente a anomalía de entrada y criterio combinado. |
| Incertidumbre y cambio | Nassar et al. (2012), tarea humana y diámetro pupilar como señal indirecta de activación. | Distinguir una observación ruidosa de evidencia repetida de cambio. | Que todo error grande merezca más escritura o más iteraciones. | Sorpresa cruda frente a sorpresa normalizada, con límites de actualización. |

Las conexiones de la tabla son propuestas de MARS-TITAN, no resultados obtenidos por los experimentos biológicos. En particular, la pérdida de acceso a un recuerdo humano no equivale a borrar un registro de un buffer. Las señales de pupila tampoco permiten atribuir de manera exclusiva un cálculo a una sustancia neuromoduladora. La analogía debe terminar donde comienza una afirmación que las medidas no identifican.

## Retener conocimiento y mantener capacidad de aprender

Hay tres situaciones distintas. El olvido de una tarea anterior es una degradación al evaluarla después de aprender otras. La pérdida de plasticidad es una reducción de la capacidad de aprender experiencias nuevas. La obsolescencia aparece cuando una relación antigua deja de ser útil en el entorno actual. Proteger todos los pesos puede ayudar a la primera y perjudicar a las otras dos. Dohare et al. estudian directamente la segunda situación, mientras que EWC y SI se centran en proteger conocimientos anteriores. [Dohare et al. (2024)](https://www.nature.com/articles/s41586-024-07711-7), [Kirkpatrick et al. (2017)](https://doi.org/10.1073/pnas.1611835114), [Zenke et al. (2017)](https://proceedings.mlr.press/v70/zenke17a.html).

En finanzas, un error mayor sobre fechas antiguas no basta para diagnosticar un fallo. Puede reflejar una adaptación a otra distribución. La propuesta es conservar bloques de diagnóstico del periodo de desarrollo que no participen en replay, evaluar las mismas observaciones con instantáneas sucesivas del modelo y registrar también el error prequential sobre observaciones nuevas. Esta evaluación histórica sirve para estudiar retención. La elección principal sigue basándose en predicción futura bajo el protocolo temporal, sin consultar el conjunto final para decidir cuánto conservar.

| Familia | Mecanismo y coste adicional | Límite relevante para MARS-TITAN |
| --- | --- | --- |
| EWC | Penaliza mover parámetros de alta importancia aproximada mediante información de Fisher. Necesita guardar esa importancia y referencias de parámetros. | La aproximación diagonal y la distribución usada para calcular importancia condicionan la protección. No existe una garantía global de no olvidar. |
| SI | Acumula contribuciones durante la trayectoria de optimización y consolida importancia al finalizar tareas. | Un flujo bursátil no proporciona fronteras de tarea verdaderas. Una consolidación periódica sería una adaptación propia. |
| GEM | Proyecta gradientes usando restricciones calculadas sobre memoria episódica. | Las restricciones son locales y dependen del buffer. Calcular gradientes y resolver la proyección añade tiempo. |
| Experience replay | Mezcla observaciones nuevas y ejemplos históricos reales. | Necesita almacenamiento y una política de muestreo. Es una referencia necesaria antes de justificar mayor complejidad. |
| DER | Conserva entradas y salidas del modelo para regularizar predicciones posteriores. | Fue formulado sobre logits de clasificación. Adaptarlo a retornos exige especificar la pérdida y evitar perpetuar errores antiguos. |
| Replay generativo | Entrena un generador y usa sus muestras como aproximación al pasado. | Añade otro modelo y puede distorsionar episodios raros. Generar datos no aumenta la información histórica realmente observada. |
| Renovación de unidades | Continual backpropagation sustituye una fracción de unidades poco utilizadas para mantener plasticidad. | Puede alterar representaciones que la memoria aún utiliza. Requiere estudiar estabilidad de claves y no debe añadirse sin un problema medido. |

Las fuentes de esta tabla son [EWC](https://doi.org/10.1073/pnas.1611835114), [SI](https://proceedings.mlr.press/v70/zenke17a.html), [GEM](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html), [replay pequeño](https://arxiv.org/abs/1902.10486v4), [DER](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html), [replay generativo](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html) y [plasticidad](https://www.nature.com/articles/s41586-024-07711-7). Sus experimentos no constituyen validación financiera de estas adaptaciones.

Un estado de tamaño finito no puede conservar con precisión arbitraria todas las secuencias posibles de un flujo ilimitado. MARS-TITAN deberá decidir qué comprime, qué conserva y qué descarta. Una formulación evaluable es «reducir interferencia y conservar episodios útiles bajo un presupuesto». También debe quedar registrada la tasa de escrituras rechazadas, las sustituciones y la antigüedad de los eventos retenidos. Un buffer externo puede ampliar capacidad, pero consume almacenamiento y tiempo de búsqueda.

## Recurrencia interna y latencia

La memoria persistente y el cálculo recurrente resuelven problemas diferentes. La primera transporta información entre decisiones. El segundo transforma varias veces la información de una decisión antes de emitir una salida. Ningún mecanismo implica conciencia, una cadena de razonamiento interpretable ni acceso a datos nuevos. Una recurrencia que reescriba repetidamente el mismo evento también puede amplificarlo artificialmente.

| Antecedente | Operación principal | Consecuencia de coste y comparación necesaria |
| --- | --- | --- |
| ACT, Graves (2016) | Aprende un número de pasos por entrada y penaliza el tiempo de cómputo. | Comparar con números de pasos fijos. La penalización y el límite de pasos afectan precisión y coste. |
| PonderNet, Banino et al. (2021) | Aprende una distribución de parada con regularización. | El número esperado de pasos no garantiza una latencia máxima. La parada estocástica necesita una política reproducible. |
| Universal Transformer, Dehghani et al. (2019) | Repite un bloque compartido de atención, con posible parada por posición. | Reutilizar pesos mantiene parámetros, pero repite atención y transformaciones. La divergencia entre posiciones puede reducir la eficiencia del lote. |
| DEQ, Bai et al. (2019) | Busca un punto fijo y diferencia implícitamente. | Ahorra activaciones respecto a profundidad explícita. Hay que contar iteraciones del solver, tolerancia y fallos de convergencia. |
| Coconut, Hao et al. (2025) | Realimenta estados latentes como entradas continuas. | La reducción de tokens no elimina pasadas por la red. Su currículo supervisado no está disponible automáticamente para retornos financieros. |
| Profundidad recurrente, Geiping et al. (2025) | Escala cómputo en inferencia repitiendo un bloque de un modelo de lenguaje. | Demuestra una dirección de escalado en otras tareas. El entrenamiento de 3.500 millones de parámetros y 800.000 millones de tokens queda fuera del presupuesto local. |
| HRM, Wang et al. (2025) | Combina dos estados y módulos a distintas frecuencias, supervisión por segmentos y parada aprendida. | Una pasada contiene varias evaluaciones internas. La aproximación del gradiente y el límite de segmentos importan tanto como los parámetros. |
| TRM, Jolicoeur-Martineau (2025) | Usa una red pequeña para actualizar un estado latente y una respuesta provisional. | Sus ablaciones separan recurrencia y supervisión profunda. Un modelo pequeño puede ejecutar muchas operaciones por ejemplo. |

Fuentes: [ACT](https://arxiv.org/abs/1603.08983v6), [PonderNet](https://arxiv.org/abs/2107.05407v2), [Universal Transformer](https://openreview.net/forum?id=HyzdRiR9Y7), [DEQ](https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html), [Coconut](https://openreview.net/forum?id=Itxz7S4Ip3), [profundidad recurrente](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html), [HRM](https://arxiv.org/abs/2506.21734v3) y [TRM](https://arxiv.org/abs/2510.04871v1).

HRM reconoce que sus dos niveles no corresponden directamente a frecuencias neuronales específicas y que la necesidad causal de la jerarquía queda abierta. Su apartado de memoria jerárquica la describe como posible trabajo futuro. TRM cuestiona la justificación de punto fijo de HRM y propone una simplificación con ablaciones. Esto aconseja comparar primero un bloque compartido sencillo y explicitar el método de entrenamiento. No basta con atribuir la mejora a una semejanza con el cerebro. [HRM, método y discusión](https://arxiv.org/html/2506.21734v3), [TRM, apartados 3 y 4](https://arxiv.org/html/2510.04871v1).

Una formulación de estudio, todavía sin implementar, sería mantener fija la memoria previa de la sesión $M_t$ mientras se refina un estado de trabajo:

$$
z^{(0)}_{i,t}=E_\theta(x_{i,t}),\qquad
z^{(k+1)}_{i,t}=F_\theta\left(z^{(k)}_{i,t},x_{i,t},R(M_t,x_{i,t})\right),\qquad
\hat y_{i,t}=G_\theta(z^{(K_{i,t})}_{i,t}).
$$

Aquí $x_{i,t}$ contiene únicamente información disponible, $R$ lee memoria, $F$ es un bloque compartido y $z$ se reinicia para cada decisión. El índice $k$ cuenta pasos internos, no sesiones de mercado. El conjunto de candidatos $K\in\{1,2,4\}$ sería un presupuesto exploratorio, no un óptimo tomado de la literatura. Una parada adaptativa solo se estudiaría después de comprobar que esos pasos fijos aportan valor. Una variación pequeña entre dos salidas puede ser un indicador de estabilidad numérica, pero no prueba que la predicción sea correcta.

La memoria se actualizaría después de emitir todas las predicciones de la sesión, mediante una única política registrada y etiquetas maduras, como exige el protocolo. Las iteraciones internas no reciben varias copias del mismo objetivo ni permiten consultar el retorno futuro para decidir cuándo parar. Comparar métodos con distinta memoria inicial o con varias escrituras por evento confundiría el efecto del cálculo con el de la exposición a datos.

El coste orientativo por decisión es $C_E+C_R+K C_F+C_G$. Esta expresión cuenta operaciones aproximadas y no sustituye un cronometraje. Si la consulta depende del estado de trabajo, el término de lectura también se repite. Al entrenar con retropropagación por todas las iteraciones aumentan las activaciones retenidas. Truncar gradientes, hacer checkpointing o usar diferenciación implícita cambia ese coste y, en los casos aproximados, también el estimador del gradiente. Debe declararse la variante concreta.

Para un problema diario, la utilidad de minimizar microsegundos no se da por supuesta. El piloto deberá medir el tiempo de procesar una sesión completa y el coste de actualizar memoria. Se registrarán latencias mediana, p95 y p99, tiempo total de entrenamiento, número de evaluaciones del bloque y pico de VRAM. Las medidas distinguirán extracción textual, inferencia con representaciones preparadas y adaptación. Se usarán el mismo hardware, precisión, tamaño de lote y calentamiento, con sincronización CUDA al cronometrar. Un promedio bajo no compensa una cola de latencia que supere el presupuesto fijado antes de consultar el conjunto final.

## Antecedentes financieros y límites de novedad

SFM ya propuso una memoria recurrente que separa componentes de distinta frecuencia para predecir precios bursátiles. DoubleAdapt estudió la adaptación incremental de datos y modelo mediante metaaprendizaje bajo cambios de distribución. Sus tareas y protocolos no coinciden exactamente con retornos residuales multimodales. Aun así, descartan presentar «memoria para bolsa», «varias escalas» o «adaptación continua» como contribuciones nuevas por sí solas. [Zhang et al. (2017)](https://www.kdd.org/kdd2017/papers/view/stock-price-prediction-via-discovering-multi-frequency-trading-patterns), [Zhao et al. (2023)](https://doi.org/10.1145/3580305.3599315).

La contribución prevista consiste en determinar si una memoria compacta y una escritura selectiva aportan valor bajo disponibilidad temporal verificable, y si una recurrencia pequeña añade una mejora que compense su coste. No localizar una combinación idéntica en esta selección de fuentes no acredita prioridad mundial. La revisión de novedad deberá actualizarse antes de redactar el informe final.

## Hipótesis falsables y orden de contraste

Las siguientes hipótesis desarrollan preguntas de O3, O4 y O6. No sustituyen la métrica primaria ni amplían automáticamente el conjunto de modelos del protocolo. Se recomienda elegir una variante por pregunta y conservar los resultados negativos.

| Hipótesis propuesta | Contraste con presupuesto controlado | Evidencia que la debilitaría |
| --- | --- | --- |
| B1. Replay reduce interferencia útilmente | Memoria sin replay frente a replay uniforme, mismo número de actualizaciones y muestras totales procesadas. Error futuro y cambio en bloques históricos de diagnóstico. | Solo mejora sobre el propio buffer o empeora de forma relevante el error futuro. |
| B2. La selectividad supera a una política simple | Escritura propuesta frente a FIFO y muestreo uniforme, misma capacidad, frecuencia de escritura y consultas. | El efecto desaparece al igualar actualizaciones o depende de información posterior. |
| B3. Dos velocidades equilibran retención y adaptación | Estado rápido con y sin consolidación lenta, frente a una sola velocidad. | El sistema conserva relaciones obsoletas y tarda más en adaptarse sin compensación predictiva. |
| B4. La recurrencia compacta mejora la relación error/coste | $K=1,2,4$, mismo bloque y datos. Añadir una referencia sin pesos compartidos de coste aproximado comparable si el piloto lo permite. | Más pasos no reducen MAE de manera estable o una referencia sencilla domina en error y latencia. |
| B5. La parada adaptativa ahorra cómputo | Política adaptativa frente a cada $K$ fijo, con umbrales y techo fijados en desarrollo. | El sobrecoste del controlador anula el ahorro o empeora p95/p99, calibración o error en cambios de distribución. |
| B6. Proteger parámetros conserva plasticidad suficiente | Una regularización de referencia frente a replay sencillo. Medir aprendizaje sobre datos nuevos y retención por separado. | Reduce olvido histórico, pero bloquea sistemáticamente la adaptación nueva. |

Las diferencias predictivas se estimarán de forma pareada por sesiones y ventanas, con intervalos por bloques y el registro de comparaciones del protocolo. Igualar parámetros y contabilizar cómputo son controles complementarios. Cuando no sea posible igualar ambos, se mostrarán las dos cantidades y la frontera de alternativas no dominadas en error, latencia y memoria. Una ausencia de diferencia estadísticamente clara no demuestra equivalencia. El margen de relevancia práctica debe fijarse durante desarrollo, no después de observar el conjunto final.

La secuencia de menor coste consiste en mantener las referencias actuales, estudiar un replay real pequeño y comparar una recurrencia de pocos pasos. EWC o SI pueden aportar un contraste de protección de parámetros. GEM, replay generativo, DEQ y jerarquías completas quedan como alternativas documentales hasta que exista una necesidad experimental concreta y un presupuesto medido. No es necesario implementar todas las familias para responder la pregunta principal.

## Notas de las 24 fuentes

Cada nota contiene menos de 100 palabras. Los identificadores coinciden con los registros bibliográficos. Las fuentes neuronales ya existentes, como Titans, TTT, MIRAS y Nested Learning, conservan sus entradas canónicas en [neural-sources.json](neural-sources.json).

### 1. `mcclelland1995cls`

**Estado:** Psychological Review, 1995, revisado por pares. Desarrolla sistemas complementarios para incorporar episodios sin destruir regularidades previas. Es una fuente teórica y computacional, apoyada en evidencia neuropsicológica, no una demostración de que un par de redes replique el cerebro. Motiva separar velocidades y probar interferencia. Su utilidad para MARS-TITAN reside en la pregunta que formula, no en transferir resultados clínicos a datos financieros. [PDF del autor](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf).

### 2. `wilson1994replay`

**Estado:** Science, 1994, revisado por pares. Registra conjuntos de células de lugar en tres ratas y observa mayor coactivación posterior durante sueño entre células activas conjuntamente durante conducta. Fundamenta la reactivación como mecanismo candidato de consolidación. No mide retorno financiero, generalización de una red artificial ni capacidad infinita. Permite motivar una comparación de replay, especificando claramente la especie estudiada. [PDF institucional](https://courses.csail.mit.edu/6.803/pdf/sleep.pdf).

### 3. `bakker2008separation`

**Estado:** Science, 2008, revisado por pares. Examina mediante fMRI de alta resolución respuestas a estímulos nuevos, repetidos y similares. Encuentra un patrón compatible con separación en CA3/DG humanos. La agrupación anatómica y la señal BOLD no identifican directamente la regla celular responsable. Puede motivar claves que limiten interferencia, pero no demuestra que más compartimentos mejoren una memoria financiera. El manuscrito está en PMC. No se confirmó descarga directa del PDF. [Texto y ficha](https://pmc.ncbi.nlm.nih.gov/articles/PMC2829853/).

### 4. `schuck2019replay`

**Estado:** Science, 2019, revisado por pares. Estudia señales fMRI humanas en reposo tras una tarea de decisión. La secuencia de estados decodificados refleja la estructura de la tarea y se relaciona con representaciones orbitofrontales. Aporta evidencia de replay no espacial, con las limitaciones de resolución e inferencia de fMRI. No es una intervención que pruebe mejora por consolidación. El PDF del laboratorio contiene el artículo, no solo el resumen editorial. [PDF del autor](https://schucklab.gitlab.io/docs/papers/Schuck_Niv_2019_Science.pdf).

### 5. `gagnepain2014suppression`

**Estado:** PNAS, 2014, revisado por pares. Combina supresión de recuperación, percepción de objetos y fMRI en participantes humanos. La supresión reduce influencias posteriores de recuerdos visuales y el análisis propone modulación frontal de representaciones corticales. Motiva distinguir retención útil y acceso selectivo. No acredita eliminación total de una memoria, beneficio en toda tarea ni una intervención terapéutica. La regla de descarte de MARS-TITAN necesitará su propia evaluación. [PDF del laboratorio](https://memorycontrol.net/2014Gagnepain.pdf).

### 6. `nassar2012arousal`

**Estado:** Nature Neuroscience, 2012, revisado por pares. Relaciona cambios pupilares humanos con incertidumbre, posibles cambios del entorno y peso de nueva información en una tarea predictiva. Ayuda a separar una observación inesperada de evidencia de cambio persistente. La pupila es un marcador indirecto y no demuestra que el locus coeruleus implemente la ecuación del modelo. Para MARS-TITAN inspira normalizar sorpresa y limitar escrituras. PDF abierto directo no confirmado. [Texto y ficha](https://pmc.ncbi.nlm.nih.gov/articles/PMC3386464/).

### 7. `schultz1997reward`

**Estado:** Science, 1997, revisado por pares. Vincula observaciones dopaminérgicas en primates no humanos con un modelo de error de predicción de recompensa. Aclara que error, recompensa y novedad no son términos intercambiables. Un retorno residual observado puede definir un error supervisado cuando madure, pero no representa una señal dopaminérgica ni mide por sí mismo relevancia causal de noticias. [PDF del autor](https://www.gatsby.ucl.ac.uk/~dayan/papers/sdm97.pdf).

### 8. `kirkpatrick2017ewc`

**Estado:** PNAS, 2017, revisado por pares. EWC protege parámetros importantes para tareas anteriores mediante una penalización cuadrática. Los resultados se obtienen en clasificación y juegos, no en mercados. Sirve como referencia de estabilidad, pero depende de aproximaciones sobre importancia y de cómo se organiza la secuencia de tareas. En un flujo financiero debe medirse si esa protección impide aprender relaciones nuevas. [Artículo](https://doi.org/10.1073/pnas.1611835114), [preprint de consulta](https://arxiv.org/abs/1612.00796v2).

### 9. `zenke2017si`

**Estado:** ICML, 2017, revisado por pares. SI estima contribuciones de parámetros al descenso de pérdida durante el aprendizaje y utiliza la importancia acumulada para consolidarlos. Su coste y mecanismo difieren de calcular una importancia al final de una tarea. La consolidación original emplea transiciones de tarea, por lo que un calendario bursátil necesitaría una definición adicional. Es una referencia de comparación, no un componente obligatorio. [Actas y PDF](https://proceedings.mlr.press/v70/zenke17a.html).

### 10. `lopezpaz2017gem`

**Estado:** NeurIPS, 2017, revisado por pares. GEM utiliza recuerdos de tareas anteriores para restringir gradientes y estudia transferencia hacia tareas previas y posteriores. Su protección depende de ejemplos retenidos y de una aproximación local. No impide cualquier aumento de pérdida tras pasos finitos. Aporta métricas y una alternativa más costosa al replay simple, cuya proyección debe cronometrarse antes de considerarla viable. [Actas](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html).

### 11. `chaudhry2019replay`

**Estado:** preprint arXiv v4, 2019. Analiza búferes pequeños y entrenamiento conjunto sobre datos nuevos y retenidos. Muestra que una referencia sencilla puede competir con mecanismos especializados en las pruebas estudiadas. Es especialmente útil para no atribuir al criterio de sorpresa una mejora explicada por volver a ver datos. La adaptación financiera debe conservar marcas de disponibilidad y comparar políticas con igual presupuesto de muestras. [Preprint](https://arxiv.org/abs/1902.10486v4).

### 12. `buzzega2020der`

**Estado:** NeurIPS, 2020, revisado por pares. DER combina replay y consistencia con logits antiguos. DER++ añade supervisión sobre etiquetas guardadas. Los autores estudian aprendizaje continuo con fronteras de tarea poco definidas. La destilación de salidas pasadas ofrece un contraste para estabilidad, pero podría conservar errores y relaciones obsoletas. Una versión para regresión no es idéntica al método publicado y debe nombrarse como adaptación. [Actas](https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html).

### 13. `shin2017generativereplay`

**Estado:** NeurIPS, 2017, revisado por pares. Un generador aproxima datos de tareas previas y los mezcla con experiencias nuevas para entrenar un predictor. Permite estudiar consolidación sin guardar todos los ejemplos originales. Su generador también necesita entrenamiento y puede perder detalles relevantes de la distribución. Para MARS-TITAN es una alternativa de discusión, con una dificultad adicional para preservar colas y episodios raros. [Actas](https://proceedings.neurips.cc/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html).

### 14. `dohare2024plasticity`

**Estado:** Nature, 2024, revisado por pares y abierto. Documenta pérdida de capacidad para aprender en configuraciones de aprendizaje continuo y estudia una renovación limitada de unidades. Obliga a evaluar adaptación nueva además de recuerdo antiguo. Sus resultados no prueban plasticidad ilimitada en toda arquitectura. La versión publicada añade a Qingfeng Lan respecto al preprint consultado y cambia su título, por lo que se cita la versión editorial. [Artículo](https://www.nature.com/articles/s41586-024-07711-7).

### 15. `graves2016act`

**Estado:** preprint de 2016, versión v6 de 2017. ACT aprende cuántas transformaciones internas ejecutar por entrada mediante una señal de parada y un coste de cómputo. Estudia tareas sintéticas y modelado de caracteres, donde no todas las tareas presentan grandes mejoras. Motiva una comparación de pasos fijos frente a adaptativos. No garantiza que dedicar más pasos a una observación financiera reduzca su incertidumbre. [Preprint](https://arxiv.org/abs/1603.08983v6).

### 16. `banino2021pondernet`

**Estado:** taller AutoML de ICML 2021, PDF arXiv v2. Reformula la parada como un modelo probabilístico y regulariza su distribución. Distingue el coste esperado de una ejecución y el procedimiento de inferencia. Es útil para analizar estabilidad y reproducibilidad de una política de cómputo variable. Una distribución con baja media aún necesita un máximo de pasos y medición de colas de latencia. No es una publicación de las actas principales de ICML. [Artículo de taller](https://arxiv.org/abs/2107.05407v2).

### 17. `dehghani2019universal`

**Estado:** ICLR, 2019, revisado por pares. Universal Transformer combina atención y repetición de un bloque compartido, con posible parada por posición. Sus tareas de lenguaje y algoritmos muestran el valor de ese sesgo recurrente en contextos concretos. Permite fundamentar refinamiento interno sin incrementar parámetros proporcionalmente a la profundidad. No implica que tiempo y activaciones sean constantes ni que exista memoria entre decisiones distintas. [Publicación](https://openreview.net/forum?id=HyzdRiR9Y7).

### 18. `bai2019deq`

**Estado:** NeurIPS, 2019, revisado por pares. DEQ sustituye una pila explícita por un punto fijo resuelto numéricamente y usa diferenciación implícita. El ahorro de memoria se refiere a la profundidad efectiva bajo la formulación propuesta. No elimina memoria de estados, solver, parámetros o entradas, ni garantiza convergencia rápida. Su complejidad requiere una justificación frente a pocos pasos explícitos antes de incorporarse a MARS-TITAN. [Actas](https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html).

### 19. `hao2025coconut`

**Estado:** COLM, 2025, revisado por pares. Coconut realimenta estados ocultos como entradas continuas y utiliza un currículo que sustituye gradualmente pasos textuales. La síntesis técnica se refiere a arXiv v3. Sus mejoras se evalúan en tareas de razonamiento y no prueban que cualquier espacio latente desarrolle ese comportamiento. Tampoco convierte en supervisión disponible una cadena de causas económicas no observada. Existe una revisión v4 de agosto de 2026, no empleada para esta síntesis. [Publicación](https://openreview.net/forum?id=Itxz7S4Ip3).

### 20. `geiping2025recurrent`

**Estado:** NeurIPS, 2025, revisado por pares. Estudia un bloque recurrente de profundidad variable en un modelo de lenguaje entrenado a gran escala. Los experimentos muestran mejoras al ampliar cómputo en determinadas tareas y estrategias para reducir parte del coste. Su escala queda fuera del presupuesto local. La idea transferible es comparar profundidad con pesos compartidos, manteniendo separadas precisión, evaluaciones internas y tiempo observado. [Actas](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html).

### 21. `wang2025hrm`

**Estado:** preprint arXiv v3, 2025. HRM utiliza módulos a distintas frecuencias y supervisión por segmentos para tareas como puzles. La jerarquía propuesta es una abstracción algorítmica y su interpretación biológica no queda causalmente establecida. La recurrencia durante una solución no proporciona por sí sola memoria de experiencias entre sesiones. La prioridad para MARS-TITAN sería aislar si dos módulos aportan algo respecto a uno, con cómputo controlado. [Preprint](https://arxiv.org/abs/2506.21734v3).

### 22. `jolicoeur2025trm`

**Estado:** preprint arXiv v1, 2025. TRM simplifica HRM mediante una red pequeña que actualiza estado latente y respuesta provisional. Incluye ablaciones sobre profundidad, supervisión y entrenamiento. Cuestiona que la justificación de punto fijo de HRM baste para sus pasos finitos. Sus resultados en puzles apoyan investigar simplificaciones, no trasladar exactitudes a predicción financiera. El número de parámetros no sustituye el número de evaluaciones ni la latencia. [Preprint](https://arxiv.org/abs/2510.04871v1).

### 23. `zhang2017sfm`

**Estado:** KDD, 2017, revisado por pares. SFM separa estados de memoria en componentes de frecuencia para predecir precios a distintos horizontes. Constituye un antecedente financiero directo de memoria recurrente multiescala. Se ha verificado la ficha y el resumen de la conferencia, pero no un PDF abierto del autor o la editorial, por lo que no se atribuye una auditoría completa de su evaluación. [Ficha primaria](https://www.kdd.org/kdd2017/papers/view/stock-price-prediction-via-discovering-multi-frequency-trading-patterns).

### 24. `zhao2023doubleadapt`

**Estado:** KDD, 2023, revisado por pares. DoubleAdapt aprende a adaptar datos y parámetros para actualización incremental en predicción bursátil. Es un antecedente concreto de adaptación ante cambios de distribución. Su implementación oficial documenta restricciones entre intervalo de actualización y horizonte, relevantes para etiquetas retrasadas. Compararlo requeriría armonizar objetivo, datos, fechas y presupuesto, sin trasladar sus cifras de rendimiento. [Artículo](https://doi.org/10.1145/3580305.3599315), [código de los autores](https://github.com/SJTU-DMTai/DoubleAdapt).

## Evidencia pendiente

No se ha medido precisión, olvido, plasticidad, latencia ni consumo CUDA de estos mecanismos en MARS-TITAN. La siguiente fase científica deberá partir de las referencias y del contrato temporal, registrar configuraciones, semillas y fallos, y comprobar el presupuesto real de GPU. Esta revisión fundamenta el estudio de la memoria adaptativa, sus ablaciones y sus límites de interpretación. No cierra los objetivos experimentales.
