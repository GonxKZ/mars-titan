# Referencias Transformer, Titans-MAC y ampliaciones de MARS-TITAN

La comparación conserva el candidato GRU con banco episódico y añade una referencia Transformer compacta, una adaptación financiera identificable de Titans-MAC y MARS-TITAN con ampliaciones desactivables sobre ese núcleo. Cada arquitectura tendrá identidad, configuración y estado propios. Cambiar el codificador y añadir memoria son intervenciones distintas. No se presupone que el Transformer o Titans reduzcan el error.

La referencia bibliográfica es [Titans: Learning to Memorize at Test Time, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf), de Ali Behrouz, Peilin Zhong y Vahab Mirrokni. La adaptación al objetivo residual y a FinMultiTime pertenece a este proyecto. No constituye una reproducción de los resultados del artículo.

El [recibo de fuentes](../../reports/research/titans-mac-source-audit-20261008.json) fija versiones, hashes de los PDF, ecuaciones y discrepancias comprobadas. La inspección de una implementación no oficial no equivale a incorporarla ni ejecutarla.

## Arquitecturas y estado comprobado

| Referencia | Función | Estado revisado el 9 de octubre de 2026 |
| --- | --- | --- |
| GRU con banco episódico | Conservar el candidato previo y sus controles de escritura | [Componente C++20 integrado](../../native/candidate.md) y [adaptador histórico con máscaras](../../native/candidate_historical.md), con pruebas CPU/CUDA, gradientes y recuperación del módulo. Pendientes la conexión cronológica y el banco 128×256 de esta variante. No entrenado. |
| Transformer compacto | Aislar el cambio de codificador con proyecciones, fusión y cabeza comunes | [Referencia integrada](../engineering/compact-transformer-reference.md), con pruebas CPU/CUDA y paridad de los modos anteriores. Sin resultados predictivos propios. |
| Titans-MAC adaptado | Atención cercana, memoria neuronal actualizable y memoria persistente aprendida | [Núcleo](../engineering/titans-memory-core.md), [adaptador](../engineering/titans-financial-adapter.md) y [consumidor cronológico](../engineering/financial-session-v2.md) integrados, con pruebas técnicas CPU/CUDA. Sin entrenamiento ni trayectoria histórica completa ejecutada. |
| MARS-TITAN sobre Titans-MAC | Incorporar las modificaciones acordadas mediante componentes desactivables | El consumidor conecta banco, lectura y estados con M0/M1. K repite la lectura sin multiplicar actualizaciones de MAC. M2/M3 y la mezcla de índices de la especificación continúan pendientes. |
| CM-v1 | Contrastar C y M sobre una referencia concreta, sin sustituirla | [C local de MAC](../experiments/mars_titan_cm_v1/mac_local_control.md), selector M y codec comprobados. El consumidor integra retenciones configurables. Pendientes la composición factorial completa y el estudio científico. Desactivada por defecto. |

Estos nombres describen brazos del estudio. No renombran checkpoints ni convierten una referencia anterior en Titans. El valor de `baseline_id` de cada contraste debe señalar una configuración ejecutable y fijada. La elección de una nueva arquitectura para B crea otro contraste B, B+C, B+M y B+C+M, conservando los manifiestos anteriores.

El consumidor congelado requiere `fastpath=False` explícito y guarda ese ajuste en su identidad. Su banco v2 separa la semilla de los hashes completos del corpus. La guía registra el fallo FP64 de la ruta fusionada, la dependencia futura del RNG v1 y sus correcciones, sin reinterpretar las comprobaciones históricas ni atribuir mejoras predictivas.

## Correspondencia con Titans-MAC

La memoria neuronal representa una función parametrizada por pesos rápidos, no una lista de episodios. Para una entrada observada se obtienen claves y valores mediante proyecciones. La pérdida asociativa y la actualización corresponden a las ecuaciones (2) y (3):

$$\ell(M_{t-1};x_t)=\|M_{t-1}(k_t)-v_t\|_2^2,$$
$$S_t=\eta_t S_{t-1}-\theta_t\nabla_M\ell(M_{t-1};x_t),\qquad M_t=(1-\alpha_t)M_{t-1}+S_t.$$

La implementación debe declarar las formas y la aplicación de las puertas dependientes de la entrada. No basta una tasa fija o una regla delta sin momentum para identificar este núcleo. Las proyecciones y la atención pertenecen al ajuste externo. La actualización interna modifica los pesos rápidos y su momentum con el objetivo asociativo.

MAC recupera información desde el estado previo, la incorpora con los parámetros persistentes al contexto de atención, actualiza memoria con la salida de atención y combina ambas salidas. La sección 3.1 y las ecuaciones (7–10) especifican este recorrido. Los parámetros persistentes de la ecuación (6) se aprenden durante el ajuste externo y permanecen fijos durante inferencia.

Hay una discrepancia de orden que debe quedar visible: la ecuación (7) de las actas escribe `[P; S; h]`, mientras que la figura 4a presenta `[P; h; S]`, como la ecuación (22) del [preprint v1](https://arxiv.org/abs/2501.00663v1). La convención de implementación propuesta es `[P; h; S]`. Debe conservarla en configuración y pruebas, sin mezclar las dos versiones.

La máscara necesita una comprobación adicional. Si `h_j` se obtiene consultando con `S_j`, ya contiene una dependencia de esa entrada. Una máscara triangular sobre la secuencia concatenada no impide por sí sola que `S_i` lea un `h_j` construido desde una entrada posterior. Las salidas por token necesitan una máscara cruzada compatible con esos índices. Una salida emitida únicamente al terminar el segmento debe declarar esa granularidad y no presentarse como causal para prefijos incompletos. La prueba perturbando entradas futuras distingue ambos contratos.

## Estados y disponibilidad

| Estado | Contenido | Política de actualización |
| --- | --- | --- |
| Parámetros compartidos | Codificadores, proyecciones, atención, fusión y cabeza | Ajuste externo autorizado. Congelados durante cada tramo de evaluación. |
| Pesos rápidos | Parámetros de la red de memoria neuronal y momentum asociativo | Actualización interna explícita con entradas disponibles. No usa la etiqueta financiera pendiente. |
| Memoria persistente de Titans | Parámetros aprendidos independientes de la entrada | Ajuste externo. Fijos durante inferencia. No son episodios recuperables. |
| Banco episódico | Episodios reales, identidad, representación, disponibilidad y predicción emitida | Admisión y retención separadas. El error financiero solo se incorpora al madurar el resultado. |
| Estado de trabajo | Activaciones, consulta y refinamientos de una predicción | Se descarta al terminar esa predicción. |
| Cola pendiente y cursor | Predicciones emitidas, etiquetas aún no utilizables y posición confirmada | Confirmación temporal recuperable. No participa en recuperación ni consolidación antes de ser elegible. |

Cada recorrido debe declarar cómo se inicializan, reinician, arrastran y guardan los pesos rápidos. El estado por activo o flujo no puede depender del orden de los activos en el lote. Todas las predicciones de una misma sesión parten del snapshot publicado para esa sesión. Una adaptación local a entradas observadas debe ser explícita y no propagar efectos entre activos por el orden de ejecución.

La sorpresa asociativa es una cantidad del objetivo de memoria. El error financiero maduro se calcula contra la predicción realmente emitida. No son intercambiables ni comparten automáticamente escalas o reglas de escritura. K = 1, 2 y 4 cuenta los refinamientos de MARS-TITAN sobre el contexto definido. No cuenta las actualizaciones internas de la memoria de Titans.

La serialización debe conservar parámetros compartidos, pesos rápidos, momentum, memoria persistente, banco episódico, cola, cursor, representación, RNG y configuración. Los grafos de autograd del pasado no forman parte del estado persistido. La política de diferenciación distingue el ajuste externo de la actualización interna y debe verificarse con gradientes, no mediante una desconexión indiscriminada.

## Modificaciones y controles

Las [decisiones de integración](system-integration.md) y la [revisión de ampliaciones](neuroarchitecture-review.md) conservan las propuestas anteriores. Su incorporación requiere un punto de inserción y una comprobación concreta. La [referencia episódica](candidate-architecture.md) conserva las reglas del candidato previo.

| Componente | Inserción o responsabilidad | Control necesario |
| --- | --- | --- |
| Cuatro modalidades y macro | Representaciones comunes, máscaras explícitas y fusión con información disponible | Mismas filas, catálogos, objetivo residual y cortes en todos los brazos de la edición. |
| Codificador Transformer | Reemplazo identificado del codificador temporal, con la fusión y cabeza comparables | GRU previa y Transformer sin memoria neuronal. |
| Memoria neuronal | Lectura y actualización interna de Titans-MAC | Misma atención y parametrización pertinente con actualización congelada, y control sin memoria identificado. |
| Banco episódico | Lectura adicional de episodios, separada de los pesos rápidos | Banco desactivado y políticas original, uniforme y selectiva con capacidad comparable. |
| Error maduro y sorpresa económica | Admisión posterior a la predicción y a la disponibilidad del resultado | Error solo frente a combinación completa, sin recalcular la predicción histórica. |
| Régimen e incertidumbre | Estado filtrado y salidas ya previstas por sus contratos | Opciones independientes y calibración separada de selección. Su inclusión no se da por implementada. |
| Refinamientos K | Estado temporal de una predicción | K=1 principal, K=2 y K=4 con número de lecturas y presupuesto emparejados. |
| Replay, especialistas y destilación | Extensiones con padres y datos permitidos identificados | Componentes desactivados y comparación propia antes de incorporarlos al predictor. |
| C y M | Operador real de la dinámica y consolidación del banco episódico | Factorial independiente y paridad con ambos desactivados. |

Los identificadores M0–M3 de las políticas de escritura se conservan. En cualquier adaptación que ya tenga memoria neuronal debe indicarse expresamente qué memoria desactiva una ablación. No se puede presentar la retirada del banco episódico como retirada de toda la memoria de Titans.

Una mejora aparente puede proceder del codificador, del número de parámetros, de la exposición a datos o del tiempo adicional. Se registran esos costes por separado y se distinguen igualdad de actualizaciones e igualdad de cómputo. No se amplía retrospectivamente una búsqueda solo para favorecer un brazo.

## CM-v1 y revisión matemática

C debe analizar la actualización completa del estado pertinente. En Titans, los pesos rápidos y el momentum evolucionan juntos. Restringir la matriz que transforma una clave no controla automáticamente esa evolución, las puertas ni el acoplamiento con la atención. Una estimación local del Jacobiano sigue siendo un diagnóstico. Las cotas de un operador fijo no se trasladan a productos variables sin una condición adicional.

M selecciona representantes de episodios elegibles del banco externo. No comprime ni sustituye implícitamente la memoria paramétrica de Titans. Las heurísticas de medoids conservan su nombre y sus límites. La geometría de los episodios y el MAE predictivo son objetivos distintos.

Cada modificación propia debe declarar una hipótesis, un mecanismo, sus supuestos, un coste medible y un control que permita descartarla. La revisión adversarial buscará fuga temporal, dependencia del orden, pérdida de episodios raros, olvido por contracción, incompatibilidad de representaciones y ventajas debidas a más datos o ajuste. Una derivación propia se identifica como tal y no se convierte en un resultado del artículo.

Las pruebas técnicas deben cubrir ecuaciones, gradientes, inmutabilidad del estado de entrada, aislamiento entre recorridos, perturbación del futuro, recuperación de la siguiente predicción y paridad al desactivar ampliaciones. El núcleo MAC, el Transformer y el estimador C ya tienen comprobaciones CUDA acotadas y documentadas. Los caminos de integración nuevos necesitan sus propias verificaciones. No se atribuye aceleración a C++ o CUDA sin una comparación medida.

Continúa la preparación de todos los datos utilizables desde 2000, con [ausencias explícitas](../data/historical-input-masks.md), y la comparación estricta permanece separada. El bloqueo vigente impide entrenamientos, postentrenamientos, pilotos y evaluaciones científicas. La implementación y las pruebas técnicas no levantan ese bloqueo ni acreditan una mejora predictiva.
