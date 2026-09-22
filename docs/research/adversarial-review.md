# Revisión crítica adversarial

Autor del proyecto: Gonzalo García Lama. Revisión documental: 18 de septiembre de 2026.

Estado: revisión de la arquitectura candidata y de sus correcciones de diseño. No se han ejecutado los experimentos propuestos aquí. Los contraejemplos son situaciones construidas para poner a prueba las hipótesis, no fallos observados de una implementación.

La revisión contrasta el [protocolo](protocol.md), la [arquitectura](../engineering/architecture.md) y la [matriz experimental](experiment-matrix.md) con antecedentes primarios. Su propósito es precisar qué podría mejorar MARS-TITAN, qué comparación permitiría atribuir esa mejora y qué resultado obligaría a revisar la propuesta. Las fuentes respaldan afirmaciones delimitadas. Su consulta no equivale a una lectura exhaustiva de toda la bibliografía ni a una búsqueda que garantice originalidad.

## Dictamen provisional

Es razonable investigar una memoria episódica acotada, consolidación histórica y un número adaptativo de pasos de lectura. Sus componentes tienen antecedentes directos. El interés del trabajo puede residir en su adaptación a eventos financieros con disponibilidad temporal verificable y en medir su utilidad bajo restricciones de recursos. No hay evidencia propia que permita anunciar olvido nulo, eliminación total del ruido, predicción perfecta, superioridad general o un nuevo estado del arte.

La candidatura revisada mantiene parámetros y codificador congelados durante cada tramo de evaluación principal, adapta únicamente los estados permitidos y separa la consolidación paramétrica de esa evaluación. Esta delimitación elimina varias vías de contaminación por diseño. Su cumplimiento tendrá que comprobarse mediante registros y pruebas. Una variante con adaptación paramétrica durante evaluación sería otro experimento.

La severidad «crítica» indica que el defecto invalidaría la interpretación temporal o la afirmación científica. «Alta» indica una amenaza seria a la atribución de la mejora o a la viabilidad. «Media» identifica un coste o una incertidumbre que requiere medición.

## Objeciones y pruebas de falsación

### R01. Retener lo útil no equivale a recordarlo todo

**Severidad: alta.** Titans incorpora una regla de olvido y MIRAS interpreta la retención como regularización. Ambos reconocen restricciones de memoria. No establecen conservación exacta de una historia arbitraria. La sorpresa asociativa de Titans tampoco coincide automáticamente con el error de un retorno financiero. [Titans, §3.1](https://arxiv.org/html/2501.00663v1#S3.SS1), [MIRAS, §§3–4](https://arxiv.org/html/2504.13173v1).

**Objeción y contraejemplo.** Una memoria de B bits tiene como máximo 2^B estados. Si recibe N bits independientes con N > B, no puede distinguir todas sus historias. Es un argumento de capacidad finita, no una afirmación de que sea imposible conservar una representación suficiente para una tarea concreta. Además, lo que parece irrelevante hoy puede importar cuando regrese un régimen antiguo. Un selector basado solo en error grande puede conservar errores impredecibles y expulsar sucesos raros pero informativos.

**Prueba propuesta.** Comparar selección por error maduro, selección con diversidad, muestreo uniforme y muestreo aleatorio con igual número de escrituras y bytes. Medir recuperación de episodios, error en revisitas de régimen y velocidad de adaptación tras un cambio. Una mejor recuperación que no reduzca error futuro no confirma utilidad predictiva. La selección propuesta pierde apoyo si no mejora frente al muestreo aleatorio en condiciones equiparables.

**Coste y concesión.** Conservar más episodios reduce descartes, pero aumenta lectura, almacenamiento y posibilidades de recuperar información obsoleta. El compromiso defendible es retención útil medida bajo un presupuesto.

### R02. Memoria rápida, consolidación lenta y replay tienen antecedentes

**Severidad: alta para la afirmación de novedad.** McClelland y colaboradores ya plantearon aprendizaje rápido de episodios e integración gradual mediante reactivación. Rolnick y colaboradores mostraron que el replay reduce olvido en tareas de aprendizaje por refuerzo, incluso con buffers limitados. Nested Learning describe niveles con distintas frecuencias de actualización y un sistema de memoria multiescala. Estas aportaciones no validan por sí mismas la aplicación financiera. [McClelland et al.](https://pubmed.ncbi.nlm.nih.gov/7624455/), [Rolnick et al.](https://arxiv.org/abs/1811.11682v2), [Nested Learning](https://arxiv.org/html/2512.24695v1).

**Objeción y contraejemplo.** La analogía cerebral orienta el diseño, pero no identifica una contribución nueva. Una consolidación extensa podría mantener patrones antiguos a costa de aprender el presente. La conservación y la plasticidad son propiedades distintas, como muestra el estudio de pérdida de plasticidad de Dohare y colaboradores. [Dohare et al.](https://doi.org/10.1038/s41586-024-07711-7).

**Prueba propuesta.** Contrastar memoria sola, replay solo y su combinación. Igualar el presupuesto total de ajuste y registrar el cómputo de consolidación. Examinar errores en regímenes nuevos y recurrentes por separado. Si la combinación solo vence porque realiza más actualizaciones, no queda aislado su mecanismo. Los regímenes del contraste se definen con pasado o mediante secuencias sintéticas conocidas, sin escoger retrospectivamente los tramos más favorables.

**Coste y concesión.** El replay puede reducir interferencia a cambio de trabajo adicional y de conservar ejemplos. Las frecuencias rápida y lenta deben expresarse en eventos o sesiones, con sus políticas fijadas antes de evaluar.

### R03. Versionar vectores no garantiza que sigan siendo compatibles

**Severidad: crítica si se mezclan representaciones incompatibles.** Latent Replay guarda activaciones intermedias y ralentiza el aprendizaje de las capas inferiores para limitar la obsolescencia de esas activaciones. Es un antecedente especialmente cercano al almacenamiento de episodios compactos. [Pellegrini et al.](https://arxiv.org/abs/1912.01100v2).

**Objeción y contraejemplo.** Si cambia el codificador, una consulta nueva y una clave antigua pueden tener igual dimensión y distinto significado. Una actualización atómica del identificador no repara esa incompatibilidad. El mismo problema aparece al cambiar un normalizador o al reutilizar estados de activos generados con otros pesos.

**Prueba propuesta.** El núcleo usa codificador y normalizadores congelados dentro de cada tramo. Cada episodio conserva su `encoder_revision`. Antes de publicar una revisión incompatible se requiere reiniciar y reconstruir el estado con historia autorizada, o aplicar una migración explícita comprobada. Una prueba de mezcla deliberada debe rechazar el estado incompatible. Para una futura variante adaptativa, comparar reconstrucción, representación congelada y migración con el mismo presupuesto.

**Coste y concesión.** Reconstruir requiere entradas históricas y tiempo. Congelar la representación simplifica la validez de la memoria, pero puede limitar adaptación. La candidatura acepta inicialmente esta segunda opción.

### R04. La etiqueta madura no legitima cualquier reconstrucción histórica

**Severidad: crítica.** El [protocolo](protocol.md) ya exige lectura antes de actualización y estados reproducibles. La revisión añade que el error usado para seleccionar un evento debe proceder de la predicción conservada que realmente se emitió, con su modelo, estado y reloj. Recalcularlo con el modelo actual mide otra cosa.

**Objeción y contraejemplo.** Una predicción antigua se recalcula después de entrenar con su etiqueta y parece poco sorprendente. El selector elimina precisamente el episodio que antes falló. Otro caso es repetir una época sobre el pasado arrastrando el estado del final de esa misma época. Las entradas parecen ordenadas, pero la memoria ya conoce sucesos posteriores.

**Prueba propuesta.** Cambiar todas las etiquetas aún no maduras debe conservar las predicciones actuales, las escrituras autorizadas y la política de recurrencia. Repetir el procesamiento desde un mismo estado inicial debe reconstruir el resultado dentro de la tolerancia numérica declarada. Cada pasada temporal debe reiniciar o reconstruir el estado según una regla explícita. Permutar la carga de activos no debe alterar una cohorte de predicciones emitidas con la misma instantánea.

**Coste y concesión.** Conservar predicciones y eventos ocupa espacio, pero permite distinguir adaptación predefinida de reconstrucción retrospectiva. El entrenamiento compartido entre activos no autoriza a usar cierres de otro mercado que aún no han ocurrido.

### R05. Un catálogo macroeconómico amplio amplía también las revisiones y ausencias

**Severidad: crítica.** ALFRED conserva versiones históricas, pero su fecha de publicación puede proceder de la fuente, de un proveedor o, si falta, de la primera disponibilidad en FRED. Sus periodos de tiempo real son fechas. La consulta por defecto refleja lo conocido hoy. No debe atribuirse a ese registro una hora histórica que no proporciona. [Ayuda de ALFRED](https://alfred.stlouisfed.org/help), [periodos de tiempo real](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

**Objeción y contraejemplo.** Una revisión del PIB recibida hoy modifica un trimestre antiguo. Si se sustituye su valor dentro de todas las ventanas pasadas, la memoria adquiere conocimiento futuro aunque cada fila conserve una fecha antigua. Tampoco puede considerarse sorpresa de publicación una diferencia respecto a un consenso reconstruido a posteriori.

**Prueba propuesta.** Registrar por observación periodo económico, versión, fecha de publicación, disponibilidad efectiva y procedencia. Procesar una revisión como un evento nuevo. Añadir revisiones futuras no debe cambiar ninguna predicción ni estado anterior. Comparar primera publicación y versión revisada solo en un análisis explícito de sensibilidad. Las series sin hora verificable siguen un retraso conservador documentado. El alta en un catálogo de candidatos no implica admisión como variable.

**Coste y concesión.** Más series exigen más controles, selección y tratamiento de ausencias. Las versiones añaden almacenamiento, pero no requieren mantener toda su historia en VRAM. Las familias macro se evalúan mediante ablaciones y búsqueda registrada.

### R06. Repetir una lectura puede mejorar el cálculo, pero no añade una observación

**Severidad: alta.** Geiping y colaboradores muestran beneficios de profundidad recurrente en modelos de lenguaje. RD-VLA ya combina una cabeza con pesos compartidos, refinamiento latente y parada adaptativa por convergencia en robótica. Son antecedentes de recurrencia y asignación variable de cómputo, no pruebas de mejora en retornos. [Geiping et al.](https://arxiv.org/html/2502.05171v1), [Tur et al.](https://arxiv.org/abs/2602.07845v1).

**Objeción y contraejemplo.** Con las mismas entradas, un paso adicional puede aproximar mejor una función o recuperar otra parte del contexto. Sería incorrecto descartarlo simplemente porque no recibe datos nuevos. También puede converger a una predicción equivocada o aumentar confianza sin reducir error. Una publicación con mucho ruido puede provocar más iteraciones justo donde el cálculo adicional aporta menos.

**Prueba propuesta.** Congelar la memoria durante cada bucle y contrastar K = 1, 2 y 4, puerta por presupuesto, puerta por incertidumbre y asignación aleatoria con igual cómputo medio. Medir diferencia de error por paso, calibración, frecuencia de cada K y latencias p50, p95 y p99. La puerta propuesta pierde apoyo si no supera reglas sencillas bajo las mismas restricciones. La estabilidad del vector latente no se usa como sustituto de cobertura o error observado.

**Coste y concesión.** Compartir pesos ahorra parámetros, pero repite operaciones. El entrenamiento puede conservar activaciones adicionales. La comparación requiere tanto igual cómputo como una frontera de error frente a latencia, ya que ambas condiciones no siempre se igualan simultáneamente.

### R07. La destilación puede heredar contaminación y errores del profesor

**Severidad: crítica para contaminación y alta para atribución.** La transferencia de un modelo costoso a otro más barato es un antecedente establecido por la destilación. Su existencia permite plantear un contraste, pero no garantiza preservar precisión o calibración en otro dominio. [Hinton et al.](https://arxiv.org/abs/1503.02531).

**Objeción y contraejemplo.** Un profesor K = 4 que ha visto el test puede transmitir esa información al estudiante K = 1. Una vía más sutil aparece al usar predicciones históricas del profesor como si hubieran sido emitidas antes de disponer de sus pesos o su memoria. También puede copiar excesiva seguridad o errores sistemáticos del profesor.

**Prueba propuesta.** El profesor y su selección usan exclusivamente el tramo autorizado. La destilación dentro del entrenamiento puede usar un profesor ajustado en ese mismo tramo, pero esas salidas no se presentan como predicciones históricas fuera de muestra. Las salidas usadas para aprender utilidad de la puerta, errores de escritura o reglas evaluadas de forma secuencial deben generarse con cortes históricos válidos. Comparar K = 1 directo, K = 1 destilado y K = 4 con igual información, incluyendo calibración independiente del estudiante.

**Coste y concesión.** Publicar un estudiante rápido desplaza cómputo al entrenamiento. Deben contarse profesor, generación de objetivos, búsqueda y recalibración. Una inferencia más barata no demuestra menor coste total.

### R08. Separar consolidación e inferencia no elimina la contención

**Severidad: alta.** Es una deducción del diseño del sistema, no un resultado de un benchmark local. Un segundo proceso o flujo CUDA comparte recursos físicos. El doble buffer puede mantener simultáneamente pesos, estados y copias adicionales. El optimizador y las activaciones de actualización incrementan el pico por encima del almacenamiento de dos instantáneas.

**Objeción y contraejemplo.** Durante una ráfaga de noticias, la inferencia aislada cumple su p95, pero consolidación, transferencias y publicación elevan p99 o fuerzan estados más antiguos. Una instantánea puede ser temporalmente válida y estar desactualizada. Esas dos propiedades deben medirse por separado.

**Prueba propuesta.** Medir la ruta completa con lecturas, colas y transferencias, en reposo y con actualización concurrente. Registrar `prediction_at`, corte de información, versión leída, eventos pendientes, antigüedad del estado y momento de publicación. Probar saturación, recuperación y rechazo de una publicación incompleta. El bundle publicado incluye codificador, normalizadores, pesos, memoria, versiones macro, calibrador y política K. Una cohorte no mezcla versiones.

**Coste y concesión.** Las colas acotadas requieren una política declarada de retraso, compactación o descarte. Usar K = 1 o abstenerse puede proteger un presupuesto, pero su efecto sobre cobertura y error forma parte del resultado. El núcleo documental no necesita implementar concurrencia antes de demostrar utilidad de la memoria.

### R09. Suavizar no identifica toda la señal ni autoriza ventanas futuras

**Severidad: crítica si utiliza futuro y alta si altera el objetivo.** Hamilton muestra que el filtro de Hodrick–Prescott puede introducir relaciones dinámicas espurias y problemas en los extremos. Es una advertencia concreta sobre un filtro, no una refutación universal de cualquier reducción de ruido. [Hamilton](https://www.nber.org/papers/w23429).

**Objeción y contraejemplo.** Sin supuestos adicionales, una observación x = s + e admite distintas descomposiciones entre señal s y ruido e. Un suavizado puede eliminar un salto informativo. Si utiliza una ventana centrada, también puede incorporar precios posteriores. Regenerar un gráfico con esos valores conservaría la misma fuga.

**Prueba propuesta.** Comparar datos sin suavizar con transformaciones unilaterales ajustadas solo con pasado. En datos sintéticos con señal y ruido conocidos, medir distorsión de saltos y recuperación, sin trasladar esa identidad al mercado real. Cambiar el sufijo futuro no debe modificar ninguna característica del prefijo. El retorno objetivo permanece fijo entre variantes. Una aparente mejora obtenida solo al suavizar la etiqueta cambia la pregunta experimental.

**Coste y concesión.** Un filtro causal puede reducir varianza introduciendo retraso. Se informa ese compromiso. Los estados de régimen empleados como entrada deben ser filtrados, no reconstruidos con toda la serie.

### R10. Más datos y dos mercados no garantizan más información útil

**Severidad: alta.** La [ficha local](../data/finmultitime-card.md) distingue aproximadamente 109 GiB de espacio ocupado de cobertura y calidad todavía no auditadas. El inventario contiene duplicados potenciales, hechos revisados y noticias cuya relevancia para el activo requiere revisión. Recorrer toda la copia no obliga a convertir todos sus bytes en entradas elegibles.

**Objeción y contraejemplo.** Un entrenamiento conjunto puede mejorar el promedio por el mercado con más muestras y empeorar el otro. Emparejar China y Estados Unidos por fecha natural puede usar un cierre estadounidense posterior a la decisión china. Un catálogo retrospectivo también puede introducir supervivencia aunque se lean todos sus archivos.

**Prueba propuesta.** Usar calendarios y disponibilidad en UTC, con evaluación separada por mercado. Comparar entrenamiento separado y compartido, informando pesos de muestreo y cobertura. Registrar admisión y exclusión de datos por motivo, sin seleccionar por rendimiento futuro. El conjunto completo elegible de entrenamiento pertenece al fold correspondiente. Test y calibración mantienen sus funciones aunque se haya inventariado físicamente toda la copia.

**Coste y concesión.** La lectura por lotes acotados permite trabajar con un corpus mayor que la RAM, pero su rendimiento debe medirse. Más activos elevan el estado total, el coste de cohortes y el tiempo de búsqueda. No se deduce una duración de entrenamiento a partir de los GiB ocupados.

### R11. Predecir mejor no implica rentabilidad segura ni causalidad económica

**Severidad: crítica para esas afirmaciones.** McLean y Pontiff estudian deterioro fuera de muestra y tras publicación de predictores de retornos. Su evidencia no demuestra que toda predicción sea imposible. Sí impide tratar un efecto retrospectivo como rentabilidad estable asegurada. Pearl distingue inferencia causal de asociaciones observacionales y explicita la necesidad de supuestos de identificación. [McLean y Pontiff](https://doi.org/10.1111/jofi.12365), [Pearl](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf).

**Objeción y contraejemplo.** Una menor MAE puede coexistir con pérdidas netas si los errores se concentran en operaciones costosas. Una variable macro puede anticipar retornos y estar asociada a factores omitidos. Residualizar mercado o sector no convierte esa relación en el efecto de una intervención.

Como contraejemplo a una garantía universal, si el próximo retorno fuera un choque independiente de +a o −a con igual probabilidad, ninguna función del pasado podría acertarlo siempre. Su error absoluto esperado mínimo sería a. Este ejemplo matemático no afirma que los retornos reales sigan esa distribución. Muestra qué supuesto adicional haría falta para prometer exactitud, a saber, que no exista incertidumbre futura relevante para el objetivo.

**Prueba propuesta.** Conservar métrica predictiva primaria y simulación económica separadas. Informar costes completos, periodos adversos, coberturas y exposiciones. Comparar una política que no opera. Una ablación permite estimar la contribución predictiva de un módulo bajo el protocolo, pero no identifica una causa económica. Las conclusiones deben limitarse al universo, fechas y condiciones efectivamente evaluados.

**Coste y concesión.** Abstenerse puede reducir exposición y error a costa de menos oportunidades. La cobertura se informa sin renormalizar el riesgo de forma implícita. Un resultado negativo conserva valor comparativo.

### R12. La búsqueda repetida puede producir una arquitectura aparentemente especial

**Severidad: crítica.** White formaliza el problema de reutilizar los mismos datos para buscar y evaluar modelos. La selección entre muchas alternativas puede producir un ganador aparente por azar. Registrar solo la mejor configuración no permite evaluar ese efecto. [White](https://doi.org/10.1111/1468-0262.00152).

**Objeción y contraejemplo.** Se prueban memorias, series macro, regímenes, puertas y semillas hasta encontrar una combinación favorable. Después se presenta su comparación aislada como confirmatoria. Que la última búsqueda use un optimizador automático no elimina los intentos manuales anteriores.

**Prueba propuesta.** Registrar todas las decisiones y ejecuciones, cerrar familias confirmatorias y presupuesto antes del test, y mantener ese test reservado. Las diferencias se calculan por sesiones emparejadas y se remuestrean bloques temporales que mantengan juntos los activos. Las semillas no cuentan como mercados independientes. Un intervalo que no excluye efectos de ambos signos deja una conclusión inconclusa, no acredita equivalencia.

**Coste y concesión.** Una comparación más amplia exige corrección y suficiente evidencia. La prioridad es una hipótesis específica con contrastes interpretables. El catálogo amplio de ideas no se convierte automáticamente en una cuadrícula de entrenamiento completa.

## Antecedentes y prueba de aportación específica

| Parte del candidato | Antecedente próximo | Diferencia que todavía hay que justificar |
| --- | --- | --- |
| Memoria por sorpresa y retención | Titans y MIRAS. | Selección por error financiero previamente emitido y maduro, con presupuesto de escrituras y tratamiento de disponibilidad. |
| Episodios y consolidación lenta | Sistemas complementarios, replay y Nested Learning. | Utilidad de la política de retención y consolidación en cambios y revisitas de régimen financiero. |
| Replay de representaciones compactas | Latent Replay. | Compatibilidad entre versiones, economía de almacenamiento y coste medido de reconstrucción. |
| Recurrencia con pesos compartidos y parada | Geiping et al. y RD-VLA. | Utilidad de una puerta condicionada por información económica disponible e incertidumbre evaluada, frente a puertas simples. |
| Cabeza rápida destilada | Hinton et al. | Transferencia de un refinamiento financiero a un predictor rápido sin contaminación ni pérdida inaceptable de calibración. |
| Versiones macro y disponibilidad | ALFRED y contrato temporal del proyecto. | Integración auditable con memoria y predicciones, distinguiendo fecha económica, publicación y revisión. |

La hipótesis más concreta consiste en comprobar si una política de retención con diversidad y una puerta de recurrencia condicionada por eventos disponibles mejoran el error y la calibración en revisitas de régimen bajo restricciones de memoria y latencia. Sigue siendo una hipótesis. La disponibilidad de una publicación macro no demuestra que merezca más cómputo.

La prueba debe separar ambos componentes. Un contraste factorial pequeño con retención básica o propuesta y puerta básica o propuesta permite comprobar si existe una interacción útil. Se conservan entradas, capacidad, particiones y reglas de selección. La diferencia de diferencias se estima con incertidumbre temporal, sin interpretar interacción predictiva como mecanismo económico causal. Si los efectos no se sostienen o las reglas simples dominan en error y coste, la combinación propuesta no recibe respaldo. La revisión bibliográfica permite describir antecedentes y diferencias, pero no certificar que nadie haya publicado una combinación semejante.

## Correcciones incorporadas a la candidatura

| Objeción | Ajuste de diseño aceptado | Evidencia pendiente |
| --- | --- | --- |
| Mezcla de representaciones y estados coadaptados. | Codificador, normalizadores y pesos congelados durante cada tramo principal. Episodios ligados a su revisión y reconstrucción antes de cambios incompatibles. | Pruebas de compatibilidad y reconstrucción de estado. |
| Consolidación que aprende del periodo evaluado sin delimitarlo. | Replay paramétrico limitado inicialmente a entrenamiento y reajustes programados con pasado. Adaptación paramétrica online tratada como variante distinta. | Manifiestos de muestras, cortes y actualizaciones. |
| Revisión macro que reescribe el pasado. | Las revisiones son eventos nuevos. Las instantáneas y predicciones anteriores permanecen inmutables. | Prueba de invariancia frente a revisiones futuras y auditoría de disponibilidad. |
| Lecturas que cambian de memoria dentro de un bucle. | Instantánea fija durante K pasos y dentro de la cohorte. | Trazas de versión y equivalencia bajo permutación de carga. |
| Calibrador que deja de corresponder al modelo publicado. | Bundle que incluye transformaciones, pesos, memoria, macro, calibrador y política K. No se presume cobertura tras una mutación. | Calibración y cobertura de la política completa, incluida adaptación de memoria y selección de K. |
| Latencia medida con GPU libre y una única copia. | Presupuesto del doble buffer, actualizaciones y contención. Registro de antigüedad del estado, colas y degradación a K = 1 o abstención. | p50/p95/p99, memoria pico y evaluación bajo carga concurrente. |
| Interpretar todo el dataset como entrenamiento indiscriminado. | Inventario completo y uso de todo el entrenamiento elegible de cada fold. Exclusiones justificadas y test protegido. | Auditoría de cobertura y ejecución del recorrido completo. |

Estos ajustes resuelven objeciones en la especificación. No certifican una implementación ni completan los objetivos experimentales de la [hoja de ruta](roadmap.md).

## Fuentes primarias consultadas

Las referencias siguientes corresponden a las afirmaciones delimitadas del documento. En los trabajos extensos se han contrastado el resumen y las secciones citadas, no se declara una lectura integral. Las pruebas de falsación son propuestas para MARS-TITAN.

1. Behrouz, A., Zhong, P., & Mirrokni, V. (2024). *Titans: Learning to Memorize at Test Time*. arXiv, versión 1 depositada el 31 de diciembre de 2024. [Texto y mecanismo de memoria](https://arxiv.org/html/2501.00663v1).
2. Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*. arXiv, versión 1. [Marco MIRAS](https://arxiv.org/html/2504.13173v1).
3. Behrouz, A., Razaviyayn, M., Zhong, P., & Mirrokni, V. (2025). *Nested Learning: The Illusion of Deep Learning Architecture*. arXiv, versión 1. [Niveles de actualización y memoria multiescala](https://arxiv.org/html/2512.24695v1).
4. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review, 102*(3), 419–457. [Resumen original y DOI](https://pubmed.ncbi.nlm.nih.gov/7624455/).
5. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). *Experience Replay for Continual Learning*. NeurIPS 2019. [Versión de los autores](https://arxiv.org/abs/1811.11682v2).
6. Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., & Sutton, R. S. (2024). Loss of plasticity in deep continual learning. *Nature, 632*, 768–774. [Publicación](https://doi.org/10.1038/s41586-024-07711-7).
7. Pellegrini, L., Graffieti, G., Lomonaco, V., & Maltoni, D. (2020). *Latent Replay for Real-Time Continual Learning*. IROS 2020. [Versión de los autores](https://arxiv.org/abs/1912.01100v2).
8. Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B. R., Kailkhura, B., Bhatele, A., & Goldstein, T. (2025). *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*. arXiv. [Arquitectura y evaluación](https://arxiv.org/html/2502.05171v1).
9. Tur, Y., Naghiyev, J., Fang, H., Tsai, W.-C., Duan, J., Fox, D., & Krishna, R. (2026). *Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning*. arXiv. [Preprint](https://arxiv.org/abs/2602.07845v1).
10. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. arXiv. [Preprint](https://arxiv.org/abs/1503.02531).
11. Hamilton, J. D. (2018). Why you should never use the Hodrick-Prescott filter. *The Review of Economics and Statistics, 100*(5), 831–843. [Documento de trabajo original de 2017 y referencia de publicación](https://www.nber.org/papers/w23429).
12. Federal Reserve Bank of St. Louis. (s. f.). *ALFRED Help* y *Real-Time Periods*. Documentación oficial consultada el 18 de septiembre de 2026. [Procedencia y fechas](https://alfred.stlouisfed.org/help), [semántica de periodos](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).
13. Pearl, J. (2009). Causal inference in statistics: An overview. *Statistics Surveys, 3*, 96–146. [Artículo en el sitio del autor](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf).
14. McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? *The Journal of Finance, 71*(1), 5–32. [Publicación y resumen](https://doi.org/10.1111/jofi.12365).
15. White, H. (2000). A reality check for data snooping. *Econometrica, 68*(5), 1097–1126. [Publicación y resumen](https://doi.org/10.1111/1468-0262.00152).
