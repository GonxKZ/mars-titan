# Propuestas finitas desde errores maduros

Los módulos `mars_titan.proposals.error_rule`, `error_source` y `error_proposer` añaden controles programados al [registro de propuestas](mature-proposals.md). Leen emisiones archivadas, comparan el error del padre con el retorno residual cero y proponen configuraciones ya registradas. No construyen candidatas, entrenan un proponente, seleccionan un modelo ni cambian el consumidor activo.

La pérdida de una acción alternativa del padre no evalúa otra arquitectura, programa regional, semilla o número de iteraciones J. Por eso la utilidad de cada receta propuesta permanece desconocida. Su comparación requiere ejecutar esa receta en un tramo posterior con controles propios.

## Plan y fuente

`ErrorProposalPlan` identifica el contrato de datos, el consumidor padre, la partición de generación, el universo ordenado de flujos y los pesos de mercado. También fija soporte mínimo, margen, candidatas en orden y límites. La configuración basal se resuelve mediante `ProposalRegistry`. Las candidatas deben ser distintas de esa referencia y pertenecer al registro, con un máximo de siete.

Los intervalos de generación, selección y evaluación deben ser sucesivos y disjuntos. Su duración conjunta no puede superar 366 días. Este contrato admite experimentos de meses o de hasta un año, sin recorrer todo el histórico. La generación solo admite `train` o `validation`. La declaración del plan y la selección previa del padre deben preceder a su comienzo. Todos los cortes quedan antes del test reservado, con el inicio de 2024 como frontera exclusiva para las observaciones. Las fechas declaradas son un contrato local, no una certificación externa de cuándo se decidió el protocolo.

`prepare_error_review(session, *, registry, plan)` recibe una `FinancialSession` existente, cerrada y con recogida nativa de alternativas. No reconstruye ni ejecuta un predictor. Contrasta el contrato, la fase, el consumidor efectivo y la receta basal completa, incluidos precisión, acción, lector y catálogo. El registro inicial excluye selectores aprendidos, calibradores y controles locales que no identifica.

La lectura recorre todas las generaciones de la fase, incluidas las de calentamiento y liquidación. Las decisiones no vacías aportan todas sus emisiones. No admite una lista de generaciones escogida por el llamante. El marco conserva sus IDs, las generaciones de decisión, la cabeza nativa y las huellas del archivo y de la revisión. Los flujos declarados sin emisiones se informan como ausentes. No se inventan filas para ellos.

Se permiten como máximo 256 generaciones y 64 MiB de bytes lógicos de fuente. Después de contabilizar los eventos, el lector de colecciones recibe únicamente el presupuesto restante. El límite incluye los eventos JSON y los bytes de la ventana histórica. No representa lecturas físicas, transferencias ni pico de RAM. Se mantiene el límite de 1 MiB para la revisión y otro de 1 MiB para los metadatos de evidencia. Una fase que no quepa falla sin recortar su marco.

`ErrorProposalEvidence` conserva la revisión y una copia sellada de los metadatos. Tanto el constructor como `record()` comprueban el esquema cerrado y la coherencia entre padre, contrato, fase, cabeza, generaciones, filas y huella del marco. Esa validación no autentica un archivo completo falsificado de forma coherente. La procedencia sigue dependiendo del lector nativo verificado y de la custodia local de sus anclajes.

## Estadística y controles

`parent_zero_statistics(review, *, plan)` usa la mediana realmente emitida y su etiqueta financiera madura. Calcula `abs(prediction - target)` para el padre y `abs(target)` para el residual cero, sobre los mismos IDs. Primero promedia activos de cada mercado y sesión, después sesiones de cada mercado y finalmente los pesos de mercado declarados. La normalización no elimina mercados sin soporte.

Todas las emisiones del marco necesitan una etiqueta madura. Una fila pendiente, excluida o censurada causa abstención, con sus recuentos explícitos. También se abstiene si falta un mercado declarado o no alcanza el mínimo de sesiones. En esos casos las métricas quedan como `None`, sin puntuar un subconjunto favorable.

`excess_mae = parent_mae - zero_mae` conserva su signo. El disparador exige `excess_mae > min_excess_mae`. El empate y los excesos negativos no lo activan. La regla no afirma que la candidata vaya a corregir ese error.

| Control | Comportamiento con soporte completo |
| --- | --- |
| `none` | Ninguno |
| `fixed` | Primera candidata declarada |
| `enumerate` | Todas las candidatas, una vez y en orden |
| `mature_parent_vs_zero` | Intenta la primera candidata y solo la propone si se activa el disparador |

Los tres controles activos aplican la misma barrera de soporte. Una candidata intentada sin soporte termina como `rejected` y conserva su cargo. Ningún control estima su utilidad a partir de las doce pérdidas condicionales del catálogo.

## Reserva y recuperación

El coordinador prepara la evidencia y construye `ProposalLedger` con `generation_plan_id=plan.plan_id`. El vínculo usa esquema 2 y aparece en la identidad y los recibos. Omitir el argumento, o pasar `None`, conserva exactamente el esquema 1 anterior, sus campos e identidades.

`propose_from_mature_errors(ledger, *, plan, evidence)` contrasta plan, registro, revisión, fase, universo y presupuestos antes de reservar. Después persiste el cargo y calcula la estadística. Un fallo técnico posterior deja un único intento `reserved` y cargado. Repetir la operación recupera la misma petición y no devuelve su cargo.

Este recorrido se ejecuta tras cerrar el tramo fuente. No añade un bucle al forward de cada activo. Medir la lectura y publicación de propuestas tampoco mide la latencia de predicción del modelo.

La cuota no se renueva automáticamente. Una enumeración que la agote informa de las candidatas no intentadas y de que está incompleta. Cambiar margen, cortes, orden o universo cambia la identidad del plan y no permite reabrir la misma ejecución como si no hubiera consumido intentos. `proposed` sigue sin significar selección o despliegue.

## Alcance de las comprobaciones

Las pruebas de `tests/proposals/` usan contratos financieros reales y fixtures identificadas. La fuente completa se comprueba con archivo nativo, consumidor FP64 en `cuda:0`, dos cohortes, ocho emisiones y recepciones maduras o censuradas. Las pruebas estadísticas con ambos mercados usan cuerpos sintéticos completos y referencias aritméticas independientes. No son resultados sobre un corpus financiero.

La revisión adversarial incluye metadatos incoherentes con una revisión intacta, vínculos entre plan y revisión, cortes, cuotas, recuperación y paridad del formato anterior. La vigilancia de pureza observa llamadas sin sustituir métodos protegidos del consumidor congelado. La [evidencia técnica](../../reports/engineering/error-proposals-20261023/README.md) distingue pruebas, cobertura, CRAP y mutaciones dirigidas.

La referencia [Self-Adapting Language Models, versión 2](https://arxiv.org/html/2506.10943v2), utiliza adaptación interna y evaluación posterior en su bucle exterior. Este control financiero conserva pendiente esa evaluación de la candidata. No reproduce su algoritmo ni acredita novedad o una mejora predictiva.
