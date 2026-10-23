# Revisión de errores maduros y propuestas registradas

`mars_titan.proposals.mature_review` construye una señal sobre predicciones ya emitidas y resultados recibidos en un prefijo nativo verificado. `mars_titan.proposals.ledger` registra propuestas limitadas a un [registro finito de configuraciones](proposal-registry.md), con cuota y recuperación. Ninguno construye predictores, ejecuta nuevas predicciones, ajusta parámetros o sustituye consumidores.

Este incremento técnico de [#711](https://github.com/GonxKZ/mars-titan/issues/711) conserva evidencia para una revisión posterior. El ciclo exterior completo, el selector aprendido y la confirmación de un cambio de consumidor siguen sin implementar.

Los [controles finitos desde errores maduros](error-proposals.md) amplían este contrato con una fuente cerrada completa, un plan temporal y generación programada. El vínculo opcional `generation_plan_id` del ledger usa esquema 2. Omitirlo conserva el formato anterior descrito aquí. Los controles no completan el aprendizaje, selección o despliegue del ciclo exterior.

## API y procedencia de la señal

La función de entrada es `build_mature_review(window, decision_events, *, review_at, parent_consumer_id, expected_window_sha256, expected_events_sha256)`. Devuelve un `MatureReview` con identidad `review_id`. Su método `record()` comprueba el sello y la coherencia interna y devuelve una copia sin alias mutables. El constructor público `MatureReview(body)` también valida el esquema cerrado, los tipos, los enlaces temporales, las pérdidas y el recuento de filas maduras.

El coordinador obtiene `window` mediante `FinancialSession.mechanism_collections_at`, con `before=review_at`, y los eventos de las decisiones seleccionadas mediante `FinancialSession.mechanism_event`. Son los [lectores históricos existentes](mechanism-collection-asof.md). La nueva API no abre otro lector ni verifica por sí sola los archivos originales. El coordinador debe conservar las dos huellas después de la lectura verificada:

- `expected_window_sha256` identifica las colecciones, la supervisión y la procedencia del prefijo según el contrato de la ventana.
- `expected_events_sha256` cubre el contenido canónico completo de la lista de eventos de decisión, incluidos sus cuantiles. La huella se calcula con `fingerprint(decision_events, MAX_SOURCE_BYTES)` de `mature_review` sobre los eventos verificados, antes de entregarlos a la revisión.

La revisión compara ambas huellas esperadas con el contenido recibido. Una huella de ventana correcta no cubre por sí sola todos los valores de `decision_events`. Tampoco `native_record_sha256` sustituye la huella de esos eventos. Además de las huellas, se contrastan generaciones, cortes, modelo, consumidor padre, muestras y el ancla de cada colección mediante `NativeMechanismCollections.check_anchor`.

El vínculo con el padre exige que `parent_consumer_id` coincida con el consumidor de la observación y del evento. Cada fila conserva el snapshot y `bank_entries` de la observación. El valor canónico `snapshot_sha256=None` se admite cuando `bank_entries=0`. Si el banco no está vacío, se exige una huella. La señal conserva esas referencias y no copia los episodios ni los pesos del padre.

## Recepción nativa y significado de las pérdidas

La recepción de un resultado se obtiene enlazando `resolutions.generation` con `as_of.native_records[].generation`, su huella y su `cutoff`. Este último se conserva como `native_received_at`. En las vistas as-of, `MechanismSupervision.clock` es un watermark de disponibilidad reconstruido por `_Collection.receive`. No es el corte de recepción nativa ni permite adelantar una recepción tardía.

La ventana debe corresponder exactamente al corte exclusivo `review_at`. Para utilizar una etiqueta, la decisión, la madurez, la disponibilidad y la recepción nativa deben preceder estrictamente a ese corte. La disponibilidad tampoco puede ser posterior a la recepción. El prefijo no admite registros en el propio corte de revisión. `review_at` no puede superar `FINAL_TEST_US`, correspondiente al inicio de 2024, y todos los registros utilizados quedan antes de ese límite. La revisión no abre el test final reservado.

La pérdida de la emisión usa el punto exacto del evento nativo, que debe ser la mediana `quantiles[2]` de sus cinco cuantiles finitos y ordenados. No usa como sustituto el cuantil de una alternativa que solo coincida dentro de la tolerancia del ancla. Para una etiqueta `y`, se conserva:

```text
emitted_point_loss = abs(emitted_quantiles[2] - y)
emitted_point_gain = reference_point_loss - emitted_point_loss
```

Las pérdidas y ganancias de alternativas se calculan solo cuando constan como observadas en la colección. Son evidencia condicional sobre las mismas entradas y el snapshot de la decisión emitida. No representan la trayectoria que habría seguido otro consumidor durante toda la fase. La acción solicitada y la realmente emitida permanecen separadas, también cuando hay una alternativa de respaldo.

Las filas pendientes, excluidas o censuradas conservan pérdidas y ganancias desconocidas como `None`. Una alternativa no observada tampoco recibe una pérdida cero. Si falta la pérdida de referencia, su ganancia permanece desconocida. Las ganancias negativas se conservan con su signo. `mature_rows` cuenta únicamente las filas con etiqueta madura elegible y debe coincidir con el contenido de la señal.

## Reserva y cuota finita

`ProposalLedger(directory, *, registry, review, limits, resume=False)` vincula una ejecución al registro, su configuración de referencia, el padre declarado, la revisión, los dos anclajes y los límites. Solo admite `ProposalRegistry` y `MatureReview` validados. El registro identifica recetas, no los pesos de un modelo entrenado ni una trayectoria experimental. Esa identidad no demuestra por sí sola que dos ejecuciones sean comparables.

`limits` debe contener exactamente estas cinco claves, con enteros positivos y sin valores booleanos. Los límites pueden reducirse, pero no superar los máximos:

| Clave | Máximo |
| --- | ---: |
| `max_candidates` | 8 configuraciones registradas |
| `max_attempts` | 32 intentos |
| `max_review_rows` | 8192 filas |
| `max_review_bytes` | 1 MiB de JSON de la revisión |
| `max_state_bytes` | 1 MiB del estado JSON persistido |

Los límites se aplican de forma conjunta. El máximo de filas no garantiza que una revisión con ese número de filas quepa en los límites de bytes. El estado incluye la revisión y sus recibos. El constructor comprueba que el estado inicial cabe antes de crear la salida. La señal mantiene además los límites de entrada de 256 colecciones y registros del prefijo, y 64 MiB de fuente. Estas cotas no son una medición del pico de RAM ni del coste del lector histórico.

`reserve(configuration_id, *, request_id)` solo acepta una configuración del registro. Persiste un recibo `reserved` y aumenta `charged_attempts` antes de devolverlo. Solo puede existir un intento pendiente. Repetir el mismo `request_id` con la misma configuración devuelve su recibo actual sin un nuevo cargo. Reutilizarlo con otra configuración produce `ValueError`.

`finish(proposal_id, *, disposition, reason)` termina el intento como `proposed`, `rejected` o `abandoned`. `proposed` exige al menos una fila madura elegible. No exige ganancia positiva ni selecciona o despliega un consumidor. Tanto la revisión como los recibos conservan `selection_performed=False` y `deployment_performed=False`.

Repetir una finalización con el mismo resultado y motivo es idempotente. Intentar cambiar un resultado terminal falla. Cada reserva conserva un `proposal_id` estable, mientras que la huella del recibo refleja su estado actual. `request_id` debe ser legible y ocupar entre 1 y 128 bytes UTF-8. El motivo terminal ocupa entre 1 y 256 bytes UTF-8.

La cuota no se renueva automáticamente. Rechazar, abandonar o reabrir un intento no devuelve su cargo. Una revisión sin etiquetas puede conservar intentos, pero no finalizarlos como `proposed`. El agotamiento de la cuota o del límite de bytes impide añadir nuevos intentos. No hay compactación ni campaña ilimitada implícitas en este contrato.

## Persistencia, recuperación y confianza

`snapshot()` relee el estado confirmado bajo bloqueo exclusivo y devuelve una copia. Para reabrirlo se utiliza el mismo constructor con `resume=True`, el mismo registro, revisión y límites. La recuperación contrasta la identidad, el sello de estado, el número de cargos, los ordinales y la cadena de recibos. No ejecuta de nuevo predicciones o ajustes. Si se interrumpe la finalización después de una reserva confirmada, el intento sigue pendiente y cargado hasta que se finalice de forma explícita.

La ejecución utiliza `state.json` y un archivo `.lock`. Publica cada cambio mediante `data.storage.atomic_json`, con sustitución atómica y sincronización del archivo y del directorio. Solo admite como residuo recuperable un temporal acotado de esa escritura, que retira tras validar el estado confirmado. No acumula un archivo por intento.

Se rechazan estados dañados, claves JSON duplicadas, archivos sobredimensionados o no regulares y rutas simbólicas. La apertura de `state.json` combina `O_NOFOLLOW` y `O_NONBLOCK` con la comprobación de archivo regular mediante `fstat`. Así, una FIFO no bloquea antes de comprobar su tipo. El bloqueo y estas restricciones utilizan las API locales de POSIX.

Los sellos detectan cambios incoherentes entre el contenido, sus contadores y sus enlaces. No autentican al coordinador ni impiden falsificar coherentemente todas las entradas, incluidas las huellas esperadas. Tampoco detectan por sí solos la restauración coherente de una versión anterior de todo el estado local. La lectura nativa verificada y la custodia de los anclajes por el coordinador forman parte de la frontera de confianza. El constructor de `MatureReview` valida coherencia interna, no sustituye esa procedencia.

Las [pruebas del incremento](../../tests/proposals/test_mature_proposals.py) comprueban recepción tardía, corte estricto, mediana nativa, alteración conjunta de punto y cuantil, desconocidos, ganancias negativas, pureza, cuota, idempotencia, reapertura y rechazo de una FIFO mediante un proceso de metadatos separado. La verificación conjunta de propuestas pasó 80 casos sin omisiones, incluyendo las pruebas previas del registro. Las fixtures utilizan el archivo nativo y CUDA para producir las emisiones sintéticas. Esto no acredita resultados predictivos ni completa la generación, comparación posterior, selección o despliegue del ciclo exterior.
