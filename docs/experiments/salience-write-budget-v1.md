# Escrituras E3 con escala histórica por activo

`salience_write_budget_v1` implementa un selector de candidatos maduros y su estado recuperable. La [declaración del núcleo](../../configs/experiments/salience-write-budget-v1.json) conserva tres controles opcionales. Su [integración por ventanas](../../configs/experiments/salience-correction-window-v1.json) conecta E3 con las emisiones reales del recorrido de corrección y dispone de comprobaciones CPU con fixtures. La sesión durable `FinancialSession` también admite E3 desde el 28 de octubre, y desde el 30 de octubre su cupo es por mercado e instante de publicación, como en las ventanas. No se modifican los parámetros compartidos del predictor ni los índices M0–M3. No hay una comparación predictiva ejecutada.

El cupo del núcleo es por mercado y cohorte completa de publicación. Las ventanas agrupan los resultados de un instante en una cohorte. La sesión durable puede recibirlos en varias generaciones, que comparten el límite de escrituras de ese instante. Cuando llegan repartidos, la selección es voraz y puede diferir de la que elegiría la cohorte completa. Que un instante equivalga a una sesión del calendario depende de cómo se publiquen los eventos. Dividir un evento en varios cortes para renovar el cupo cambiaría el experimento.

## Error, escala y selección

Cada `MatureWrite` contiene un activo, la fecha de decisión, la disponibilidad del resultado, el punto emitido y la etiqueta recibida. Su identidad es `(flow_id, prediction_at)`. El consumidor debe enlazarlos con el registro real de emisiones y comprobar la maduración. Este selector solo comprueba el contrato de los valores que recibe. No autentica su procedencia ni permite deducirla de un checksum.

Para el resultado maduro se calcula \(e_i=y_i-\widehat y_i^{emitida}\). El selector utiliza la predicción final, no el error del núcleo sin corrector ni la sorpresa asociativa de Titans. Antes de recibir la cohorte \(t\), cada activo \(a\) conserva una cuenta y un RMS de sus errores anteriores:

\[
r_{a,<t}=\sqrt{\frac{1}{n_{a,<t}}\sum_{j<t}e_{a,j}^2},
\qquad s_i=\frac{|e_i|}{\max(r_{a,<t},\epsilon)}.
\]

Durante el arranque, con menos de cuatro resultados anteriores, se usa la escala inicial 0,01. El suelo es \(10^{-8}\). El cupo inicial es 64. Son valores de desarrollo declarados, sin ajuste ni validación predictiva. La API permite otros valores acotados y conserva su identidad.

Cada cohorte recibida se puntúa con el mismo prefijo previo. Después, todos sus errores actualizan las estadísticas, incluidos los candidatos descartados. Varias filas del mismo activo en una cohorte tampoco adelantan su escala de puntuación. En la sesión, cada generación recibe su propia cohorte. Una generación posterior del mismo instante ve las escalas actualizadas por la anterior, aunque comparte su cupo. La actualización numérica evita formar cuadrados potencialmente desbordantes:

\[
r_{n+1}=\operatorname{hypot}\!\left(r_n\sqrt{\frac{n}{n+1}},\frac{e_{n+1}}{\sqrt{n+1}}\right).
\]

Esta expresión es algebraicamente equivalente al RMS incremental. Se evalúa en FP64 ordinario y no es aritmética certificada. Los valores no finitos y el desbordamiento de una diferencia o puntuación producen un error. No se recortan para obtener una selección aparentemente válida.

Por mercado se eligen `min(writes_per_market, candidatos)` identidades reales. Los empates se resuelven por activo y decisión. El control `absolute_error` conserva el mismo cupo y usa \(|e_i|\). El control `uniform` ordena prioridades SHA-256 con dominio propio, semilla, corte e identidad, sin consumir el RNG del modelo. Conserva los 256 bits de prioridad. No se atribuye a esta construcción una garantía universal de aleatoriedad criptográfica o utilidad financiera.

Dividir todos los errores por una misma constante positiva no altera su orden. Por eso el mecanismo usa escalas por activo y necesita el control de error absoluto. El RMS incluye sesgo, ruido y cambios de dinámica, y no permite identificarlos por separado. Puede restar prioridad a activos con errores persistentes o dar prioridad excesiva a otros con un historial muy estable. Un error grande también puede deberse a un dato incorrecto.

La motivación procede de la distinción entre ruido de observación y cambios latentes en Piray y Daw, *Computational processes of simultaneous learning of stochasticity and volatility in humans*, Nature Communications 15, 9073, versión publicada el 21 de octubre de 2024, [DOI:10.1038/s41467-024-53459-z](https://www.nature.com/articles/s41467-024-53459-z). Sus experimentos y su filtro de partículas no son este algoritmo. El trabajo motiva separar esas explicaciones, sin acreditar que un RMS financiero las distinga. La fórmula y el selector E3 son una propuesta del proyecto. Se ha inspeccionado la página primaria, sin reproducir sus análisis o ejecutar su código.

## Estado, límites y recuperación

`MatureWriteBudget.select` devuelve un `WriteSelection` y otro `MatureWriteBudget`. No cambia el estado anterior. Se comprueban todas las disponibilidades antes de acceder al primer valor financiero. Una publicación no vacía debe avanzar el corte. Las decisiones repetidas, las regresiones por activo y las cohortes de tamaño desconocido se rechazan. Una llamada vacía no avanza el estado ni consume cupo.

El límite es de 8.192 registros por cohorte y 8.192 activos en el estado. Alcanzarlo no expulsa estadísticas antiguas automáticamente. La ordenación canónica cuesta \(O(N\log N)\), la selección con heap cuesta \(O(N\log q)\) y copiar y validar los agregados cuesta \(O(F)\), con \(F\) activos previos. El espacio transitorio es \(O(N+F)\). No se construyen matrices de distancias ni se conserva la historia de etiquetas.

El esquema JSON 1 conserva configuración, cuenta, RMS y últimas fechas por activo, cursor, publicaciones y escrituras seleccionadas. `restore` valida esos campos y la huella antes de aceptar el estado. El consumidor durable aún debe publicar conjuntamente la memoria escrita y el selector. La API del núcleo no implementa por sí sola esa transacción ni la cola de etiquetas pendientes.

`SalienceBudgetConfig(**declaracion["arms"][nombre])` carga los campos de cada control. La declaración no es una receta de `run_correction_window` y no puede ejecutarse como si ya lo fuera. El error usado para priorizar escrituras también debe permanecer separado del valor que escriba el backend. Una corrección madura puede escribir `etiqueta − núcleo` aunque la selección use `etiqueta − emisión final`.

## Comprobaciones ejecutables

Desde el repositorio, `uv run pytest tests/memory/test_salience_budget.py tests/benchmarks/test_salience_budget_cost.py` comprueba las fórmulas y sus límites con valores literales. No ajusta parámetros. La batería incluye escalas anteriores, arranque, varios mercados, cupo cero, inversión del orden de activos, recuperación, ausencia de acceso temprano a valores y rechazo de NaN, Inf y desbordamientos.

`PYTHONPATH=src:. uv run python benchmarks/salience_budget_cost.py` mide selección, exportación, serialización JSON y recuperación con 64, 256, 1.024 y 8.192 candidatos construidos a mano. Cada tamaño compara los tres controles al mismo cupo. El proceso informa muestras, calentamiento, mediana, extremos, bytes serializados, un ciclo separado con `tracemalloc` y RSS absoluto. El benchmark no lee datos financieros, ejecuta modelos ni mide la sesión completa. Su coste CPU de metadatos no demuestra el rendimiento de una futura integración CUDA.

La [verificación del 27 de octubre](../../reports/engineering/salience-budget-20261027/verification.json) registra 127 pruebas CPU, 68 propias y 59 regresiones o guardas, sin omisiones ni pasos de optimizador. Se detectan [cinco mutaciones dirigidas](../../reports/engineering/salience-budget-20261027/mutations.json), con referencias verdes antes y después. La [cobertura](../../reports/engineering/salience-budget-20261027/quality.json) de los dos módulos nuevos es de 222/229 sentencias y 69/76 ramas, con CRAP máximo 20,4. El diagnóstico utiliza coverage.py 7.16.1 y Radon 6.0.1. CRAP usa cobertura de sentencias y no demuestra ausencia de errores. No se ha ejecutado la suite completa.

La revisión adversarial corrigió la aceptación de `True` como equivalente a `1` al recuperar una configuración y el error sin tratar ante enteros financieros demasiado grandes. Las pruebas conservaron esos fallos antes de la corrección. La recuperación final los rechaza.

Los [costes registrados](../../reports/engineering/salience-budget-20261027/cost.json) utilizan cupo 32 por mercado e historial mínimo de dos resultados, con dos publicaciones previas, distintos de los valores iniciales de la declaración experimental. Cada perfil tiene dos calentamientos, siete ciclos medidos y uno adicional instrumentado. Son 84 ciclos medidos entre los doce perfiles. La tabla contiene medianas del recorrido de selección y recuperación, en milisegundos:

| Candidatos | Escala previa por activo | Error absoluto | Uniforme |
| ---: | ---: | ---: | ---: |
| 64 | 1,213 | 1,147 | 1,509 |
| 256 | 3,626 | 3,631 | 3,805 |
| 1.024 | 11,029 | 11,675 | 14,569 |
| 8.192 | 93,283 | 110,091 | 148,394 |

El proceso interno dura 8,404 s y registra 53.252 KiB de RSS absoluto. El estado serializado con 8.192 activos ocupa de 362.735 a 362.746 bytes, según la longitud del identificador de regla. Las selecciones y los estados recuperados coinciden en todas las repeticiones. No se interpreta la diferencia entre reglas como una aceleración de MARS-TITAN ni se estima una latencia p95/p99 estable con siete muestras. El coste de acumular y verificar las emisiones reales permanece fuera de esta medida.

La integración durable y sus comprobaciones de etiquetas parciales o tardías, publicación conjunta y recuperación se detallan más adelante. Sigue pendiente acreditar la correspondencia con sesiones del calendario para cada protocolo. La comparación científica debe mantener filas, presupuesto y controles, declarar el protocolo antes de ejecutarse y contabilizar errores y coste. La implementación no acredita reducir el sobreajuste o mejorar la predicción.

## Integración en la inferencia y las ventanas

`SalienceCorrectionInference` extiende el recorrido de B6. Guarda el punto final de `run.pending` antes de que la resolución lo retire. Tras emitir todas las decisiones del evento, selecciona entre los resultados maduros. El conjunto elegido limita las filas de `_staged`, pero conserva sus valores `etiqueta − núcleo`. No sustituye ese valor por el error de la emisión final. Las escalas se actualizan con todos los candidatos, aunque no se escriban.

La comprobación CPU cubre delta, proximal, B1, B2 y tres composiciones E1. La selección también limita las actualizaciones acopladas a `memory.write`, incluidos mapas de B2 y pesos de mezcla de E1. No se presenta como una intervención exclusiva sobre una sola matriz. Los controles de un mismo contraste deben compartir ese backend y cupo. Los componentes emitidos de E1 que no se eligen se retiran de su cola madura.

`run_correction_window` acepta `salience=None` por defecto. Ese caso sigue construyendo `CorrectionInference`. Con un `SalienceBudgetConfig` válido utiliza la extensión y registra configuración y fuentes en la petición y la identidad. La variante completa obtiene una huella distinta del corrector base. `selected.json` conserva el factor. El traslado `carry_correction` lo restaura del ancla y rechaza contradicciones de identidad, configuración o fuentes antes de abrir los tramos.

La [declaración de ventanas](../../configs/experiments/salience-correction-window-v1.json) contiene argumentos comunes y los cuatro controles. Cada valor no nulo de `arms` se convierte mediante `SalienceBudgetConfig(**valores)` y se entrega al argumento `salience` del mismo lanzador. No modifica las recetas históricas, sus casos de búsqueda ni el padre seleccionado. La preparación de ese perfil no significa que se haya ejecutado su comparación científica.

Cada fase reinicia el corrector y las escalas. La recuperación comprueba las fases confirmadas y repite desde el inicio una fase interrumpida, como hace B6. La sesión durable, descrita en el apartado siguiente, guarda en cambio el estado E3 en cada generación. La fuente de ventanas agrupa el índice completo por instante antes de decodificar sus bloques y comprueba la disponibilidad de cada etiqueta contra ese índice. Este contrato de publicación no acredita todavía equivalencia con una sesión del calendario para toda posible llegada parcial o tardía.

Las fuentes de cada ejecución permanecen fijadas por huellas. Los recibos anteriores requieren su versión de código. No se rebajan esas comprobaciones para reanudar con otra implementación.

La [verificación de ventanas](../../reports/engineering/salience-window-20261027/verification.json) registra 160 pruebas CPU distintas, 59 propias y 101 regresiones, sin omisiones. La prueba de regeneración se reforzó y repitió después, sin contarla como otro caso distinto. Comprueba también el estado y los contadores E3 porque un fixture homogéneo puede producir los mismos pronósticos aunque se ignore el factor. La [cobertura del adaptador](../../reports/engineering/salience-window-20261027/quality.json) es de 62/65 sentencias y 19/22 ramas, con CRAP máximo 9. El soporte y el módulo reutilizado tienen diagnósticos separados. Seis casos CUDA están colectados y pendientes de ejecución por falta de admisión GPU. No se ha ejecutado la suite completa.

Se detectan [doce mutaciones dirigidas](../../reports/engineering/salience-window-20261027/mutations.json), con controles verdes antes y después y sin intentos de pasos de optimizador. Cubren emisión final, cupo, escalas por fase, limpieza de E1, identidad, fuentes, regeneración y bloqueo. El arnés detecta también reutilizar resultados para medir latencia, perder muestras confirmadas e ignorar el formato físico solicitado. El recibo conserva el primer superviviente de regeneración y la prueba reforzada que lo detecta. No es una enumeración exhaustiva de todos los fallos posibles.

La [medida completa](../../reports/engineering/salience-window-20261027/cost.json) recorre 32 ventanas de fixtures, con tres fases cada una, cuatro controles, un calentamiento y tres repeticiones medidas por tamaño. Incluye lectura, carga del padre, MAC, codec, escrituras, métricas y publicación. La preparación del fixture y las comprobaciones de población se cuentan en el tiempo de proceso, fuera del intervalo de cada ventana. El padre permanece sin cambios. Sus parámetros son literales, con dimensión 32, FP64, contexto de precios de 64 sesiones y anchuras 5/384/512/3/6. No se ha entrenado.

| Control | Mediana con 64 activos (s) | Mediana con 256 activos (s) |
| --- | ---: | ---: |
| Sin E3 | 3,107 | 17,702 |
| Escala previa por activo | 2,427 | 17,343 |
| Error absoluto | 2,797 | 17,489 |
| Uniforme | 3,113 | 17,779 |

Cada ventana produce 384 filas con etiqueta para 64 activos y 1.536 para 256. Los cuatro controles conservan sus identidades, fechas y etiquetas. Con 64 activos el cupo permite escribir todos los candidatos. La dispersión temporal y tres repeticiones no permiten atribuir una aceleración ni estimar colas p95/p99 estables. El proceso completo tarda 331,61 s y registra 1.382.980 KiB de RSS absoluto. No se mide VRAM, energía ni incremento de RAM exclusivo de E3.

El [intento anterior](../../reports/engineering/salience-window-20261027/cost-incomplete-v2.json) conserva las 16 ventanas completadas de 64 activos y una medida parcial del bucle de fase. Terminó por límite de tiempo antes de completar 256. El [perfil de preparación](../../reports/engineering/salience-window-20261027/preparation-profile.json) mostró el coste de leer precios del fixture con grupos Parquet de tres filas. La medida final usa grupos de 8.192, conservando exactamente los mismos precios, muestras y etiquetas. La disposición anterior sigue por defecto en los otros fixtures. No se comparan esos dos formatos como un efecto del modelo.

El arnés publica una muestra confirmada antes de seguir y conserva un recibo parcial ante un fallo posterior. Rechaza una salida existente para evitar cronometrar una relectura de informes como si fuera una inferencia nueva. La orden explícita es `PYTHONPATH=src:. uv run pytest -p tests.conftest benchmarks/salience_window_cost_check.py -q --junitxml=/tmp/mars-titan-salience-window-cost.xml`. El JSON de coste queda en una propiedad del XML. Solo utiliza fixtures propios y pesos fijados a mano. Las protecciones globales de entrenamiento y test final no se modifican.

## Publicación durable en la sesión financiera

`FinancialSession` acepta `salience=None` por defecto y entonces conserva su contrato, su identidad y sus artefactos. Con un `SalienceBudgetConfig` exige una corrección madura delta o proximal y registra la configuración en el contrato, en la identidad de integración y en la comprobación de cada paso. Cambiar el cupo durante la sesión o al reanudarla se rechaza.

El cupo se aplica por mercado e instante de publicación. `CommittedWriteBudget` reutiliza la puntuación y las escalas de `MatureWriteBudget` y añade la generación, el corte y las escrituras ya elegidas en ese corte. Los candidatos son los resultados maduros que el ejecutor nativo acepta para predicciones pendientes ya emitidas. El ejecutor garantiza que cada resultado se aplica una sola vez, así que el estado E3 no guarda el historial de identidades. Varias generaciones pueden compartir corte, por ejemplo una decisión y una liquidación en el mismo instante, y entonces comparten también su cupo. Cada una elige entre sus resultados con lo que no gastaron las anteriores de ese instante, y un corte nuevo lo renueva. Una etiqueta tardía también puede pertenecer a una decisión anterior a otra ya resuelta. Cada generación avanza su contador aunque no tenga resultados y la siguiente debe ser exactamente la anterior más uno, de modo que un reintento no renueva el cupo.

Si todas las etiquetas de un instante llegan en una sola generación, la selección y las escalas coinciden con las de la ventana, aunque haya generaciones vacías con el mismo corte antes o después. Si llegan repartidas, ningún mercado supera el cupo del instante, pero la selección es voraz. La primera generación elige sin ver los resultados que llegarán después, y la siguiente puntúa con escalas que ya incluyen los resultados anteriores del mismo instante. No puede reproducir la elección de la ventana, porque las escrituras confirmadas no se deshacen. Hasta el 30 de octubre cada generación renovaba el cupo, de modo que dos publicaciones con el mismo corte podían escribir el doble que la ventana. El estado pasa a `salience_committed_publication_v2` y rechaza los del esquema anterior.

La prioridad usa el error de la emisión final y el valor escrito sigue siendo etiqueta menos núcleo, como en las ventanas. Todos los resultados actualizan las escalas, pero solo los elegidos escriben la memoria A, incluidas las partes de E1 asociadas. Los diagnósticos B4 siguen observando todos los resultados. El calentamiento, las exclusiones por prefijo y las predicciones que se cierran sin etiqueta no crean errores.

El estado E3 es un artefacto propio de cada generación, junto al banco, la memoria rápida, la cola pendiente y la memoria A, y se poda con la misma retención. Al verificar una generación se comprueba que las escrituras elegidas coinciden con las de A, que los resultados observados coinciden con los aplicados por el ejecutor y que la última selección pertenece a los resultados de ese evento. Al reanudar, A se concilia con las escrituras elegidas por E3 y no con todas las etiquetas aplicadas.

La [verificación del 28 de octubre](../../reports/engineering/salience-session-20261028/verification.json) registra 114 pruebas CPU sin omisiones ni pasos de optimizador, 34 propias y 80 regresiones, con el enlace nativo compilado existente. Con cupo completo, E3 reproduce los puntos y la memoria de la referencia con la regla delta, con la mezcla E1 y con E1 junto a las áreas y el puente de consultas. Las pruebas cubren también cupo cero, prioridad por la emisión final, publicaciones parciales con el mismo corte, etiquetas tardías, cierres sin etiqueta, recuperación antes y después de confirmar, orden de activos, RNG y diagnósticos B4.

Se detectan [ocho mutaciones dirigidas](../../reports/engineering/salience-session-20261028/mutations.json) con referencias verdes antes y después: escribir todos los resultados pese al cupo, puntuar con la etiqueta en vez de la emisión, exigir orden de decisiones entre publicaciones, admitir generaciones saltadas, conciliar A con todas las etiquetas, dos variantes de quitar la configuración del contrato y no avanzar la generación en eventos vacíos. Una de las dos variantes del contrato fallaba en el primer paso por una incoherencia interna, así que solo demostraba que ambas partes debían coincidir. La segunda la detecta la prueba porque el rechazo llega desde la restauración con otro mensaje. La [cobertura](../../reports/engineering/salience-session-20261028/quality.json) de `salience_session` y `salience_budget` es de 239/250 sentencias y 75/86 ramas, con CRAP máximo 23,8. La rama de error de `verify_confirmed` no se recorre. Es una comprobación defensiva frente a un estado E3 con huella válida pero incoherente con el registro nativo, que ninguna prueba fabrica. El [diagnóstico de `financial_session`](../../reports/engineering/salience-session-20261028/quality-support.json) se limita a esta batería y no describe todo el módulo.

La [medida de coste](../../reports/engineering/salience-session-20261028/cost.json) recorre cinco controles con 64 y 256 activos ficticios del mercado US, con corrección proximal, un calentamiento y tres repeticiones. Cada caso prepara tres eventos, cierra y reabre la sesión, cronometra el evento 128 con todas las etiquetas del anterior, vuelve a cerrar y reabrir y ejecuta la transición siguiente. El cupo completo reproduce exactamente el estado de la referencia y cada caso repite su estado y su transición siguiente en todas las repeticiones. Los tres selectores usan cupo uno por mercado, así que escriben 3 de 192 resultados con 64 activos y 3 de 768 con 256.

| Control | Mediana con 64 activos (s) | Mediana con 256 activos (s) |
| --- | ---: | ---: |
| Sin E3 | 1,612 | 5,076 |
| Cupo completo | 1,779 | 5,293 |
| Escala previa por activo | 1,449 | 4,955 |
| Error absoluto | 1,569 | 6,462 |
| Uniforme | 1,803 | 5,950 |

La carga media del equipo estuvo entre 27 y 38 durante la medida, con otras pruebas en marcha. Con tres repeticiones y esa dispersión no se puede atribuir una diferencia de coste a ningún control. El proceso tardó 1.190 s y su RSS máximo fue de 822.976 KiB, incluida la preparación de los fixtures. Respecto a la referencia, el estado persistido crece entre 12 y 16 kB con 64 activos y entre 31 y 50 kB con 256. No se mide VRAM ni energía.

La [verificación del cupo compartido por instante, del 30 de octubre](../../reports/engineering/salience-instant-quota-20261030/verification.json), registra 105 casos CPU dirigidos aprobados sin omisiones. Comprueba las tres reglas de selección, el cupo que comparten varias generaciones del mismo corte, su renovación al avanzar, publicaciones vacías, selección parcial voraz, paridad con la ventana al llegar la cohorte completa, recuperación y rechazo de estados incoherentes. Las 15 mutaciones dirigidas se detectan con controles verdes antes y después y sus dos fuentes coinciden con la integración. No hubo intentos de optimizador.

La regresión de importadores registra 1.986 aprobados y 28 omitidos en `tests/training`, y 4.149 aprobados y 83 omitidos en el resto, sin fallos ni errores. Sus 111 omisiones se conservan: 72 están vinculadas al bloqueo y 39 a dependencias auxiliares ausentes. La selección dirigida y las regresiones se solapan y no se suman como casos distintos. Los módulos de presupuesto y sesión cubren 252/262 sentencias y 82/92 ramas. Se midieron las seis funciones cambiadas, con CRAP máximo 27,20. Ruff, formato y el comprobador del repositorio pasan. Los costes del apartado anterior corresponden a la revisión del 28 de octubre y no se presentan como una nueva medida de este esquema.

Siguen pendientes las comprobaciones CUDA y la comparación predictiva con datos reales. Si las etiquetas de una sesión de un mercado se publican en varios instantes, por ejemplo con eventos del otro mercado en medio, cada instante tiene su cupo en los dos recorridos, y el contraste debe declararlo. El aprendizaje y el test final permanecen bloqueados. Nada de esto acredita que E3 reduzca el error.
