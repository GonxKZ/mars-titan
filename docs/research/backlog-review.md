# Revisión del catálogo de tareas

Autor del proyecto: Gonzalo García Lama. Fecha: 18 de septiembre de 2026.

Registro de la consolidación inicial. La revisión posterior mantiene las cinco retiradas y añade dos trabajos distintos de preparación, el observatorio MT-068 y las guías MT-069. El [tablero actual](task-board.md) contiene 64 tareas con guías operativas y 183 dependencias. Los recuentos siguientes documentan la consolidación previa, no sustituyen ese estado actual.

Estado: consolidación aplicada y verificada. Se han auditado las 67 tareas originales, sus criterios, dependencias, prioridades y etiquetas. El [catálogo activo](../../.github/planning/issues.json) contiene 62 tareas y el [registro de retiradas](../../.github/planning/retired-issues.json) conserva las cinco consolidadas.

La comprobación posterior confirma 67 issues conservadas en el repositorio y 62 elementos en el tablero activo. MT-001 permanece cerrada y en Hecho, 61 tareas siguen abiertas y en Pendiente y cinco están cerradas como no planificadas por consolidación. Los números remotos de las nuevas tareas no coinciden con su identificador estable. Por ejemplo, MT-065 es la issue 67. El [mapa remoto](../../.github/planning/remote-map.json) conserva esa correspondencia.

## Resultado de la revisión

Se han aplicado cinco consolidaciones. Quedan 62 tareas canónicas, de las que 61 siguen pendientes y MT-001 conserva su cierre. No se recomienda reducir el catálogo mediante fusiones que mezclen implementación, evaluación y entrega académica. Tampoco se justifica conservar tareas que repiten la misma decisión y los mismos resultados esperados.

Además de las fusiones, se han delimitado responsabilidades y corregido el circuito lógico entre selección de muestra y presupuesto. El catálogo y el grafo remoto contienen 177 dependencias, sin ciclos. La actualización añadió 10 relaciones y retiró 28, incluidas las relaciones de las tareas consolidadas.

## Criterios para evitar duplicidades

- Cada tarea debe producir una decisión, componente, experimento o entrega que pueda revisarse por separado. Compartir una tecnología o una carpeta no convierte dos tareas en duplicadas.
- Si dos tareas persiguen el mismo resultado, la más específica aporta sus criterios a una tarea canónica. No se conservan ambas solo porque se redactaron en momentos distintos.
- Un componente se implementa una vez. Las tareas de integración o evaluación lo utilizan y comprueban su función, sin volver a atribuirse su desarrollo.
- Los controles normales de una implementación forman parte de sus criterios de aceptación. Solo merecen tarea propia cuando verifican un riesgo transversal o constituyen una comparación independiente.
- Las ampliaciones a otros mercados, tamaños y familias se activan con una decisión de continuidad. No se convierten en requisitos indirectos del núcleo.
- Una consolidación preserva identificadores, enlaces e historial. Su eventual cierre significa que el trabajo se gestiona en otra tarea, no que se haya ejecutado un experimento.

## Consolidaciones aplicadas

| Tarea consolidada | Tarea canónica | Motivo y contenido que debe conservarse |
| --- | --- | --- |
| MT-047 | MT-005 | Ambas proponen inventariar la misma copia y registrar su cobertura y limitaciones. MT-005 incorpora el recorrido completo, los estados inventariado, inspeccionado y validado, y la reconciliación de archivos. La auditoría detallada continúa en MT-006 a MT-011 y su resumen en MT-012. No se añade todo ese trabajo a MT-005. |
| MT-049 | MT-064 | Ambas deciden si ampliar la población y validan el coste de esa ampliación. MT-064 incorpora cobertura comparable de las referencias y registro de ejecuciones ampliadas. Los solucionadores de Ridge y boosting siguen siendo responsabilidad de MT-024 y MT-025. |
| MT-051 | MT-019 | La selección por error maduro, relevancia y diversidad es una ampliación de la política de escritura de MT-019. Sus controles uniforme y aleatorio se conservan allí. La capacidad en bytes, índices y metadatos se concreta en MT-018, que ya contempla capacidad, lectura y sustitución. Se evita planificar una segunda implementación de la misma memoria. |
| MT-061 | MT-037 | Ruido, duplicación, entradas corruptas y abstención son sensibilidades del mismo predictor. MT-037 incorpora estas perturbaciones acotadas y su efecto sobre estado, error y cobertura. Los rechazos de datos imposibles se prueban dentro de MT-007, MT-008 y MT-011. No se traslada a robustez la implementación de toda la canalización de datos. |
| MT-062 | MT-029 | Las falsaciones del candidato implementado forman parte de las ablaciones ejecutadas en desarrollo. MT-029 conserva el registro de objeciones, resultados negativos y correspondencia con mecanismos reales. Los antecedentes se documentan en MT-003 y las pruebas se diseñan antes en MT-028. |

Las cinco issues conservan un encabezado con la tarea canónica y su contenido anterior como historial. Están cerradas con motivo `not_planned` y etiqueta `duplicate`, y sus elementos se han retirado del proyecto activo. No se han borrado ni renumerado issues y el cierre no acredita trabajo científico.

## Responsabilidades delimitadas

| Tareas | Límite de responsabilidad que evita repetir trabajo |
| --- | --- |
| MT-005, MT-006 a MT-011 y MT-012 | Inventario de la copia, auditoría de cada modalidad, contrato temporal y definición de la muestra son resultados distintos. El inventario no acredita calidad de todas las filas. |
| MT-024, MT-025 y MT-064 | Las dos primeras desarrollan y validan referencias lineales y boosting. MT-064 decide y ejecuta una ampliación usando esas implementaciones. La memoria externa de XGBoost no es obligatoria si la muestra principal no la necesita. |
| MT-026 y MT-050 | MT-026 mantiene la GRU obligatoria y la decisión sobre DLinear. MT-050 es una comparación opcional de SSM, delta-rule y atención compacta. Cada familia debe tener una pregunta concreta, sin combinarlas todas. |
| MT-018, MT-019 y MT-020 | Almacenamiento y recuperación, selección de escrituras y enrutamiento por regímenes son mecanismos diferentes. Deben compartir interfaces y presupuesto, sin crear otra memoria en cada tarea. |
| MT-021 y MT-059 | MT-021 verifica orden temporal, etiquetas maduras y aislamiento entre ventanas. MT-059 comprueba compatibilidad del codificador, normalizadores, estado, calibrador y política de inferencia. No deben volver a implementar el mismo formato de instantánea. |
| MT-065 y MT-031 | MT-065 desarrolla recuperación íntegra tras una interrupción. MT-031 integra esa recuperación en la ejecución y el registro. No implementa una segunda serialización de optimizador, memoria o cursor. |
| MT-059 y MT-063 | MT-059 define qué versión completa puede publicarse. MT-063 estudia, de forma opcional, la concurrencia, las colas y su efecto sobre latencia y antigüedad. La compatibilidad no exige concurrencia. |
| MT-052 y MT-037 | MT-052 mide retención, interferencia y reaprendizaje con un protocolo propio. MT-037 reutiliza esos resultados al discutir robustez. Se ha eliminado de MT-037 la obligación de repetir el estudio de retención. |
| MT-060 y MT-038 | MT-060 mide el piloto y fija un presupuesto antes de seleccionar. MT-038 consolida las mediciones de los experimentos finales. Comparten instrumentación. No se escribe un segundo perfilador ni se repite una medición sin cambio de configuración o una duda concreta. |
| MT-056 y MT-057 | MT-056 valida fuentes macro y sus versiones históricas. MT-057 implementa su uso temporal y prueba revisiones. La adquisición de una muestra para auditar no equivale a una canalización periódica de actualización. |
| MT-056, MT-066 y MT-067 | La primera estudia el contenido macroeconómico. MT-066 selecciona fuentes complementarias de distintas modalidades y comprueba condiciones de uso. MT-067 implementa su actualización opcional con instantáneas. Las fuentes no macro no deben esperar a terminar toda la auditoría macro. |
| MT-041, MT-042, MT-043 y MT-045 | Son entregas académicas diferentes, con documentación, devolución y cierre propios. No deben fusionarse en una tarea genérica de redacción. |

## Dependencias actualizadas

No debe hacerse una sustitución mecánica que copie todas las dependencias de una tarea retirada a la canónica. En particular, trasladar las dependencias de MT-047 a MT-005 crearía un ciclo, porque las auditorías ya dependen del inventario.

| Tarea | Dependencias actuales | Razón |
| --- | --- | --- |
| MT-005 | Ninguna | Conserva su función de entrada al estudio. Las auditorías no se convierten en requisitos del inventario. |
| MT-019 | MT-015, MT-018, MT-021, MT-028 | La política de escritura ampliada usa etiquetas maduras, una memoria existente, semántica temporal y controles ya definidos. |
| MT-028 | MT-013, MT-017, MT-030, MT-003 | Los antecedentes y las diferencias que se quieren comprobar se identifican antes de diseñar ablaciones. |
| MT-029 | Mantener las actuales | Ya depende del candidato, la matriz y la referencia de memoria. Sus dependencias cubren el trabajo incorporado de MT-062. |
| MT-034 | Añadida MT-029 a las anteriores | Los finalistas se fijan después de las ablaciones de desarrollo relevantes, antes de abrir el test. |
| MT-037 | Mantener las actuales | MT-035 ya aporta las dependencias de modelos, métricas y calibración. Las perturbaciones se registran antes en MT-028, aunque su evaluación final corresponda a MT-037. |
| MT-048 | MT-004, MT-005, MT-011, MT-012 | Sustituye MT-047 por inventario y manifiesto piloto, sin exigir otra auditoría duplicada. |
| MT-052 | MT-024, MT-030, MT-019 | La escritura que antes aportaba MT-051 pasa a MT-019. |
| MT-053 | MT-019, MT-052, MT-059 | Mismo cambio de dependencia para consolidación y replay. |
| MT-064 | MT-030, MT-048, MT-060, MT-024, MT-025 | Incorpora las referencias escalables de MT-049. MT-058 se exige si se activa China, sin bloquear una ampliación exclusivamente estadounidense. |
| MT-066 | MT-005 | Seleccionar fuentes no macro no exige completar MT-056. La disponibilidad se revisa con el contrato común. |
| MT-067 | MT-011, MT-066 | La ingesta requiere contrato y fuentes admitidas. El adaptador macro añade sus comprobaciones cuando se active, sin hacerlas obligatorias para toda descarga. |

Las demás dependencias permanecen, salvo las referencias a tareas retiradas que se sustituyen por la canónica indicada. Esta transformación se ha comprobado sobre los 67 identificadores: quedan 62 nodos y 177 aristas, sin dependencias inexistentes, autorreferencias ni ciclos.

Se ha resuelto el circuito lógico entre muestra y presupuesto. MT-012 cierra la regla de selección y el manifiesto piloto. MT-060 decide el presupuesto observado y MT-034 congela la versión definitiva de la muestra principal junto con los finalistas. MT-012 no depende de MT-060.

## Redacción y etiquetas

Los títulos deben nombrar el resultado concreto, sin acumular tecnologías ni términos genéricos. En los cuerpos basta un contexto breve, el trabajo propio de la tarea y criterios observables. Las cautelas comunes y las rutas del proyecto pueden enlazarse una vez en la documentación común. No hace falta repetir en cada issue varios párrafos idénticos sobre ausencia de resultados o archivos grandes.

Se mantienen términos técnicos útiles como Ridge, CUDA, ETag y checkpoint. En prosa se recomienda «codificador» frente a encoder, «ejecutor» frente a runner, «lote precargado» frente a prefetched batch, «instantánea» frente a snapshot y «mantener o descartar» cuando no sea necesario el término go/no-go. Los nombres de campos y clases conservan su sintaxis. Las listas siguen siendo listas Markdown con guiones y separación en blanco.

Se han aplicado entre dos y cinco etiquetas por tarea activa, con al menos una de dominio y otra de función. `opcional` indica una extensión prescindible y `bloqueado` solo un impedimento real, no una dependencia pendiente. No se añaden etiquetas para repetir la prioridad o el hito, que ya son campos del tablero.

| Etiqueta incorporada | Uso acotado | Motivo |
| --- | --- | --- |
| `macroeconomia` | MT-056 y MT-057 | Distingue indicadores y revisiones macro de precios, noticias y fundamentales empresariales. |
| `aprendizaje-continuo` | MT-052 y MT-053 | Identifica retención, plasticidad, reaprendizaje y replay, distintos del simple almacenamiento de estado. |
| `checkpoints` | MT-065 y su integración en MT-031 | Permite localizar recuperación y reanudación sin agrupar toda la reproducibilidad. |
| `fuentes-externas` | MT-066 y MT-067 | Separa adquisición complementaria de la copia original de FinMultiTime. |

Estas cuatro etiquetas se han sumado a las 16 del catálogo. No se propone eliminar etiquetas ajenas al catálogo que ya existan en el repositorio. MT-001 conserva sus etiquetas y metadatos históricos. Las tareas retiradas tampoco necesitan una reclasificación extensa para explicar su cierre por consolidación.

## Auditoría de las 67 tareas

«Conservar y delimitar» implica revisar el cuerpo para respetar el límite indicado, no crear otro entregable. Las retiradas conservan sus etiquetas anteriores y añaden `duplicate`, que ya existía fuera del catálogo activo. La tabla registra los títulos aplicados y el motivo de conservar o consolidar cada tarea.

| ID | Decisión y unidad de trabajo | Motivo o límite principal | Etiquetas aplicadas |
| --- | --- | --- | --- |
| MT-001 | Conservada sin cambios | Preparación documental ya cerrada y verificada. | `transversal`, `redaccion`, `integridad-academica` |
| MT-002 | Conservada. Confirmar requisitos y calendario académico | Decisiones y documentos externos que no acredita ninguna implementación. | `transversal`, `redaccion`, `integridad-academica` |
| MT-003 | Conservada. Completar el estado del arte y las referencias | Justifica antecedentes. Las falsaciones ejecutadas quedan en MT-029. | `bibliografia`, `redaccion` |
| MT-004 | Conservada. Verificar el entorno con uv y CUDA | Verifica disponibilidad y versiones, sin asumir los perfiles experimentales de MT-060. | `transversal`, `reproducibilidad`, `rendimiento` |
| MT-005 | Conservada y ampliada con MT-047. Inventariar FinMultiTime | Recorre toda la fuente y registra permisos y estados de auditoría. | `datos`, `reproducibilidad`, `integridad-academica` |
| MT-006 | Conservada. Auditar precios y universo histórico | Ajustes, bajas e identificadores tienen controles propios. | `datos`, `metodologia` |
| MT-007 | Conservada. Normalizar noticias y su disponibilidad | Fecha, idioma, relevancia y duplicación de texto. | `datos`, `metodologia` |
| MT-008 | Conservada. Reconstruir fundamentales por publicación | Revisiones contables y fechas de cada hecho. | `datos`, `metodologia` |
| MT-009 | Conservada. Generar gráficos con historia disponible | Produce una representación visual reproducible, sin imágenes futuras. | `datos`, `modelos` |
| MT-010 | Conservada. Auditar y versionar codificadores congelados | Procedencia del modelo y cachés de representación. | `datos`, `modelos`, `reproducibilidad` |
| MT-011 | Conservada. Construir el contrato temporal de datos | Infraestructura común. Los adaptadores de modalidad reutilizan sus uniones. | `datos`, `metodologia`, `reproducibilidad` |
| MT-012 | Conservada y delimitada. Definir la selección y publicar el piloto | Cierra reglas y piloto, sin esperar un presupuesto que depende de ese piloto. | `datos`, `metodologia`, `reproducibilidad` |
| MT-013 | Conservada. Definir objetivo y reloj de decisión | Especifica la tarea predictiva antes de implementarla. | `metodologia`, `datos` |
| MT-014 | Conservada. Versionar factores de mercado y sector | Suministra factores para etiquetas, no todo el catálogo macro de entrada. | `datos`, `metodologia` |
| MT-015 | Conservada. Calcular residuales y maduración de etiquetas | Implementación verificable del objetivo. | `datos`, `metodologia`, `reproducibilidad` |
| MT-016 | Conservada. Diagnosticar cobertura y estabilidad del residual | Examina el comportamiento del objetivo en desarrollo. | `evaluacion`, `metodologia` |
| MT-017 | Conservada. Implementar codificación y fusión de eventos | Entrada común que permite retirar la memoria sin cambiar lo demás. | `modelos`, `datos` |
| MT-018 | Conservada y delimitada. Implementar memoria global acotada | Capacidad, claves, valores y recuperación. Incorpora límites de bytes de MT-051. | `memoria`, `modelos`, `reproducibilidad` |
| MT-019 | Conservada y ampliada con MT-051. Comparar políticas de escritura | Error maduro, relevancia, diversidad y controles uniformes con igual presupuesto. | `memoria`, `metodologia`, `comparativa` |
| MT-020 | Conservada. Comparar enrutamiento por regímenes | Hipótesis distinta de seleccionar eventos. No introduce otra memoria base. | `memoria`, `comparativa` |
| MT-021 | Conservada y delimitada. Verificar orden temporal y aislamiento | Cohortes, etiquetas maduras y reinicios entre ventanas. | `memoria`, `reproducibilidad`, `evaluacion` |
| MT-022 | Conservada. Implementar cuantiles y calibración | Produce la distribución y su calibrador, no solo calcula métricas. | `incertidumbre`, `modelos`, `evaluacion` |
| MT-023 | Conservada. Integrar el modelo y ejecutar un ensayo mínimo | Verifica la interacción de componentes antes de comparar. | `modelos`, `memoria`, `reproducibilidad` |
| MT-024 | Conservada. Validar cero, Ridge y el cálculo por bloques | Separa equivalencia numérica y alternativa SGD. | `comparativa`, `modelos` |
| MT-025 | Conservada y delimitada. Evaluar la referencia de boosting | Escoge una ruta adecuada a la muestra. Memoria externa solo cuando sea necesaria. | `comparativa`, `modelos`, `rendimiento` |
| MT-026 | Conservada. Evaluar GRU y decidir sobre DLinear | Referencia obligatoria y alternativa sencilla acotada. | `comparativa`, `modelos`, `rendimiento` |
| MT-027 | Conservada. Evaluar una referencia de memoria identificable | Distingue fidelidad a la publicación y adaptación propia. | `comparativa`, `memoria`, `bibliografia` |
| MT-028 | Conservada y delimitada. Diseñar ablaciones y falsaciones | Registra pruebas y perturbaciones antes de ver resultados finales. | `comparativa`, `metodologia` |
| MT-029 | Conservada y ampliada con MT-062. Ejecutar ablaciones y falsaciones | Produce resultados de desarrollo y el registro crítico del candidato. | `comparativa`, `evaluacion`, `bibliografia` |
| MT-030 | Conservada. Fijar el protocolo y la reserva final | Particiones, selección y condiciones de ampliación. | `metodologia`, `evaluacion`, `reproducibilidad` |
| MT-031 | Conservada y delimitada. Implementar ejecución y trazabilidad | Integra checkpoints de MT-065 sin volver a desarrollarlos. | `evaluacion`, `reproducibilidad`, `checkpoints` |
| MT-032 | Conservada. Verificar métricas y agregaciones | Cálculos con casos conocidos, separados de entrenar un calibrador. | `evaluacion`, `incertidumbre` |
| MT-033 | Conservada. Implementar la simulación con costes | Transforma predicciones en operaciones bajo reglas explícitas. | `evaluacion`, `metodologia` |
| MT-034 | Conservada y delimitada. Fijar muestra principal y finalistas | Usa presupuesto y ablaciones de desarrollo antes de abrir el test. | `comparativa`, `evaluacion` |
| MT-035 | Conservada. Ejecutar la comparación fuera de muestra | Produce la evidencia confirmatoria, sin selección posterior. | `evaluacion`, `comparativa`, `reproducibilidad` |
| MT-036 | Conservada. Estimar incertidumbre y efecto de la selección | Inferencia sobre diferencias y dependencia temporal. | `evaluacion`, `metodologia` |
| MT-037 | Conservada y ampliada con MT-061. Evaluar robustez y abstención | Sensibilidades y corrupción controlada. Reutiliza la retención de MT-052. | `evaluacion`, `comparativa`, `incertidumbre` |
| MT-038 | Conservada y delimitada. Informar recursos de los experimentos finales | Reutiliza instrumentación de MT-060 con los finalistas efectivamente ejecutados. | `rendimiento`, `evaluacion`, `reproducibilidad` |
| MT-039 | Conservada. Decidir sobre optimización nativa | Solo después de un perfil y con referencia numérica. | `opcional`, `rendimiento` |
| MT-040 | Conservada. Redactar conclusiones de los seis objetivos | Integra evidencia, sin crear otra campaña de experimentos. | `redaccion`, `evaluacion` |
| MT-041 | Conservada. Preparar y entregar el primer borrador | Entrega académica con contenido y devolución propios. | `redaccion`, `integridad-academica` |
| MT-042 | Conservada. Preparar y entregar el segundo borrador | Avance y respuesta a observaciones anteriores. | `redaccion`, `integridad-academica` |
| MT-043 | Conservada. Completar y entregar el tercer borrador | Documento completo antes del cierre definitivo. | `redaccion`, `integridad-academica` |
| MT-044 | Conservada. Reproducir figuras y revisar las evidencias | Comprueba el paquete final y su trazabilidad. | `reproducibilidad`, `redaccion` |
| MT-045 | Conservada. Cerrar predepósito y tramitar la autorización | Acto académico distinto de redactar un borrador. | `redaccion`, `integridad-academica` |
| MT-046 | Conservada. Preparar y realizar la defensa | Exige realización y dominio, no solo diapositivas. | `defensa`, `integridad-academica` |
| MT-047 | Consolidada en MT-005 | Duplica inventario. La auditoría de modalidades conserva sus tareas propias. | Históricas y `duplicate` |
| MT-048 | Conservada. Preparar datos por bloques y ventanas bajo demanda | Implementación de acceso eficiente, distinta de inventariar o validar fechas. | `datos`, `reproducibilidad`, `rendimiento` |
| MT-049 | Consolidada en MT-064 | Repite decisión y evaluación de ampliación. No desarrolla otros solucionadores. | Históricas y `duplicate` |
| MT-050 | Conservada. Contrastar familias secuenciales adicionales | Extensión separada de la referencia GRU obligatoria. | `comparativa`, `modelos`, `opcional` |
| MT-051 | Consolidada en MT-019 | Selección de eventos repetida. El límite físico pertenece a MT-018. | Históricas y `duplicate` |
| MT-052 | Conservada. Medir retención e interferencia | Protocolo propio de memoria y capacidad de reaprendizaje. | `memoria`, `evaluacion`, `comparativa`, `aprendizaje-continuo` |
| MT-053 | Conservada. Evaluar consolidación y replay | Cambia el aprendizaje paramétrico y requiere comparación propia. | `memoria`, `modelos`, `aprendizaje-continuo`, `opcional` |
| MT-054 | Conservada. Evaluar pasos recurrentes y parada | Hipótesis de asignación de cómputo, con salida válida en un solo paso. | `comparativa`, `modelos`, `rendimiento` |
| MT-055 | Conservada. Evaluar destilación temporalmente válida | Transferencia profesor-estudiante y coste total distintos del refinamiento. | `comparativa`, `modelos`, `opcional` |
| MT-056 | Conservada y delimitada. Auditar indicadores macro y sus versiones | Semántica y disponibilidad histórica, sin implementar un descargador general. | `datos`, `macroeconomia`, `bibliografia`, `reproducibilidad` |
| MT-057 | Conservada. Integrar macro con versiones y disponibilidad | Adaptador temporal de una modalidad, sobre el contrato común. | `datos`, `macroeconomia`, `metodologia`, `reproducibilidad` |
| MT-058 | Conservada. Versionar calendarios y reglas de mercado | Normas históricas y decisión de viabilidad operativa de China. | `datos`, `metodologia`, `evaluacion` |
| MT-059 | Conservada y delimitada. Comprobar compatibilidad de las versiones | Codificador, estado, calibrador y política coherentes al publicar. | `memoria`, `reproducibilidad`, `incertidumbre` |
| MT-060 | Conservada y delimitada. Medir el piloto y fijar el presupuesto | Decide alcance antes del test, con tiempos sostenidos y costes de preparación. | `rendimiento`, `reproducibilidad`, `evaluacion` |
| MT-061 | Consolidada en MT-037 | Es una familia de pruebas de robustez, con guardas en los módulos de datos. | Históricas y `duplicate` |
| MT-062 | Consolidada en MT-029 | Las falsaciones ejecutadas son ablaciones de desarrollo, justificadas por MT-003/028. | Históricas y `duplicate` |
| MT-063 | Conservada. Medir concurrencia y antigüedad del estado | Extensión de sistemas que requiere carga, colas y doble buffer reales. | `memoria`, `rendimiento`, `opcional` |
| MT-064 | Conservada y ampliada con MT-049. Evaluar ampliaciones del universo | Una decisión y sus pruebas por ampliación, manteniendo referencias comparables. | `comparativa`, `rendimiento`, `opcional` |
| MT-065 | Conservada. Verificar checkpoints y reanudación | Recuperación transaccional y equivalencia tras interrupciones. | `reproducibilidad`, `checkpoints`, `memoria`, `rendimiento` |
| MT-066 | Conservada. Verificar fuentes públicas complementarias | Selección, licencia y utilidad de fuentes externas a la copia original. | `datos`, `fuentes-externas`, `integridad-academica` |
| MT-067 | Conservada. Implementar actualizaciones con instantáneas inmutables | Descarga incremental, idempotencia y separación del benchmark congelado. | `datos`, `fuentes-externas`, `reproducibilidad`, `opcional` |

## Verificación de la consolidación

Antes de escribir se comprobaron los estados remotos y no había trabajo en curso registrado. Se trasladaron primero los criterios a las tareas canónicas, después se actualizaron relaciones y por último se cerraron las duplicadas y se retiraron sus elementos del proyecto. MT-001 conservó texto, etiquetas, estado y cierre.

La lectura posterior comprobó 62 elementos activos, 61 tareas pendientes, MT-001 en Hecho, cinco cierres `not_planned` y 177 dependencias nativas iguales al catálogo. Se revisó el HTML generado por GitHub en dos tareas fusionadas, incluidos apartados, listas, casillas pendientes y enlaces. El mapa remoto conserva fecha, huella y resultados de esa comprobación.

El número de tareas es una consecuencia de sus responsabilidades y evidencias, no una medida de avance científico.
