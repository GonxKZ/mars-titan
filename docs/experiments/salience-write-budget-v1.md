# Escrituras E3 con escala histórica por activo

`salience_write_budget_v1` implementa un selector de candidatos maduros y su estado recuperable. La [declaración del núcleo](../../configs/experiments/salience-write-budget-v1.json) conserva tres controles opcionales. Su [integración por ventanas](../../configs/experiments/salience-correction-window-v1.json) conecta E3 con las emisiones reales del recorrido de corrección y dispone de comprobaciones CPU con fixtures. La sesión durable por evento sigue pendiente. No se modifican los parámetros compartidos del predictor ni los índices M0–M3. No hay una comparación predictiva ejecutada.

El cupo del núcleo es por mercado y cohorte completa de publicación, no todavía por sesión de su calendario. Quien lo integre debe demostrar esa correspondencia o declarar un contraste distinto. Dividir un evento en varios cortes para renovar el cupo cambiaría el experimento.

## Error, escala y selección

Cada `MatureWrite` contiene un activo, la fecha de decisión, la disponibilidad del resultado, el punto emitido y la etiqueta recibida. Su identidad es `(flow_id, prediction_at)`. El consumidor debe enlazarlos con el registro real de emisiones y comprobar la maduración. Este selector solo comprueba el contrato de los valores que recibe. No autentica su procedencia ni permite deducirla de un checksum.

Para el resultado maduro se calcula \(e_i=y_i-\widehat y_i^{emitida}\). El selector utiliza la predicción final, no el error del núcleo sin corrector ni la sorpresa asociativa de Titans. Antes de recibir la cohorte \(t\), cada activo \(a\) conserva una cuenta y un RMS de sus errores anteriores:

\[
r_{a,<t}=\sqrt{\frac{1}{n_{a,<t}}\sum_{j<t}e_{a,j}^2},
\qquad s_i=\frac{|e_i|}{\max(r_{a,<t},\epsilon)}.
\]

Durante el arranque, con menos de cuatro resultados anteriores, se usa la escala inicial 0,01. El suelo es \(10^{-8}\). El cupo inicial es 64. Son valores de desarrollo declarados, sin ajuste ni validación predictiva. La API permite otros valores acotados y conserva su identidad.

La cohorte completa se puntúa con el mismo prefijo previo. Después, todos sus errores actualizan las estadísticas, incluidos los candidatos descartados. Varias filas del mismo activo en una cohorte tampoco adelantan su escala de puntuación. La actualización numérica evita formar cuadrados potencialmente desbordantes:

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

La integración durable posterior deberá comprobar correspondencia con las sesiones del calendario, etiquetas parciales o tardías, publicación conjunta y recuperación de la siguiente transición. La comparación científica debe mantener filas, presupuesto y controles, declarar el protocolo antes de ejecutarse y contabilizar errores y coste. La implementación no acredita reducir el sobreajuste o mejorar la predicción.

## Integración en la inferencia y las ventanas

`SalienceCorrectionInference` extiende el recorrido de B6. Guarda el punto final de `run.pending` antes de que la resolución lo retire. Tras emitir todas las decisiones del evento, selecciona entre los resultados maduros. El conjunto elegido limita las filas de `_staged`, pero conserva sus valores `etiqueta − núcleo`. No sustituye ese valor por el error de la emisión final. Las escalas se actualizan con todos los candidatos, aunque no se escriban.

La comprobación CPU cubre delta, proximal, B1, B2 y tres composiciones E1. La selección también limita las actualizaciones acopladas a `memory.write`, incluidos mapas de B2 y pesos de mezcla de E1. No se presenta como una intervención exclusiva sobre una sola matriz. Los controles de un mismo contraste deben compartir ese backend y cupo. Los componentes emitidos de E1 que no se eligen se retiran de su cola madura.

`run_correction_window` acepta `salience=None` por defecto. Ese caso sigue construyendo `CorrectionInference`. Con un `SalienceBudgetConfig` válido utiliza la extensión y registra configuración y fuentes en la petición y la identidad. La variante completa obtiene una huella distinta del corrector base. `selected.json` conserva el factor. El traslado `carry_correction` lo restaura del ancla y rechaza contradicciones de identidad, configuración o fuentes antes de abrir los tramos.

La [declaración de ventanas](../../configs/experiments/salience-correction-window-v1.json) contiene argumentos comunes y los cuatro controles. Cada valor no nulo de `arms` se convierte mediante `SalienceBudgetConfig(**valores)` y se entrega al argumento `salience` del mismo lanzador. No modifica las recetas históricas, sus casos de búsqueda ni el padre seleccionado. La preparación de ese perfil no significa que se haya ejecutado su comparación científica.

Cada fase reinicia el corrector y las escalas. La recuperación comprueba las fases confirmadas y repite desde el inicio una fase interrumpida, como hace B6. No hay un checkpoint E3 por decisión en `FinancialSession`. La fuente de ventanas agrupa el índice completo por instante antes de decodificar sus bloques y comprueba la disponibilidad de cada etiqueta contra ese índice. Este contrato de publicación no acredita todavía equivalencia con una sesión del calendario para toda posible llegada parcial o tardía.

Las fuentes de cada ejecución permanecen fijadas por huellas. Los recibos anteriores requieren su versión de código. No se rebajan esas comprobaciones para reanudar con otra implementación.
