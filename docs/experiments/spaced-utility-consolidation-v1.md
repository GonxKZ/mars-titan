# Evidencia de utilidad espaciada para la consolidación E2

El registro `SpacedUtilityLedger`, los productores estándar y regional K1 y la retención sobre el banco nativo están implementados y comprobados con fixtures CPU. Conservan pronósticos emitidos, calculan evidencia al madurar y proponen qué episodios mantener. La sesión financiera conecta ambos productores con publicación durable en M1 y C desactivado. El adaptador `SpacedInference`, el cargador, el controlador y la CLI conservan por ahora el lector estándar. Quedan sus rutas regionales, CUDA y la comparación científica. E2 no cambia las referencias por defecto.

E2 estudia si conviene conservar episodios que han ayudado en decisiones separadas. No es una técnica de regularización de los parámetros del predictor y no demuestra una reducción del sobreajuste. La comparación predictiva permanece sin ejecutar.

## Contraste y agregación

Para una predicción original emitida \(p_i\), una predicción diagnóstica \(p_{i,-e}\) emitida sin el episodio \(e\) y su resultado maduro \(y_i\), se define:

\[
g_{i,e}=|y_i-p_{i,-e}|-|y_i-p_i|.
\]

Un valor positivo indica que el pronóstico original tuvo menor error. El contraste para K = 1 elimina el episodio del soporte ya seleccionado, renormaliza sus pesos y no introduce otro vecino. No equivale a quitarlo de todo el banco y repetir la búsqueda. El productor utiliza una sola preparación del padre y una sola selección de vecinos. El registro recibe sus números, sin repetir la inferencia cuando llega la etiqueta. La sesión y el adaptador de ventanas cotejan esos números con los puntos emitidos. El registro aislado no acredita esa procedencia.

Dentro de cada mercado e instante se promedian las filas que utilizaron ese episodio. Después se actualiza la media de sus sesiones con evidencia. Varias acciones de una sesión no crean varias confirmaciones. Esta media condicional al soporte utilizado no es el MAE global del proyecto. Los episodios no utilizados carecen de evidencia y no reciben un cero inventado.

La calificación exige una media superior a `positive_gain_margin` y ganancias positivas en dos días UTC separados al menos por `min_gap_days`. Se guardan los extremos de esos días, de modo que un resultado antiguo que llegue tarde conserve su fecha de decisión. Dos mercados del mismo día no acreditan dos días distintos. No se afirma independencia estadística. La implementación admite exactamente este criterio de dos días extremos, no una regla de tres o más confirmaciones.

Una nueva ganancia negativa puede retirar la calificación. La media no tiene decaimiento y puede conservar evidencia obsoleta cuando cambia el mercado. El margen cero y un día de separación son elecciones iniciales del núcleo técnico, no valores validados ni constantes de la neurociencia.

La utilidad por supresión no es aditiva. Si dos episodios equivalentes permiten acertar indistintamente, quitar cualquiera de ellos puede dar ganancia cero y quitar ambos puede empeorar. Tampoco mide una contribución de Shapley ni una causa económica. El diagnóstico congelado permite contrastarla con retención original, uniforme, reciente y utilidad permutada, a igual capacidad y calendario. Esa comparación aún no se ha ejecutado con datos financieros.

## Orden temporal y estado

`begin(cutoff)` abre un evento posterior al último publicado. `issue(snapshot, rows)` registra una cohorte completa de un mercado antes de recibir resultados. Todas las cohortes del evento comparten el mismo `EpisodeSnapshot`, con codec, corte, huella y episodios maduros comprobados. `maturity_at` es la maduración financiera del episodio. `available_at` describe la disponibilidad de sus entradas y no la sustituye.

Cada fila contiene flujo, pronóstico original, pares de episodio y pronóstico diagnóstico y `label_not_before`. No admite un objetivo financiero en la emisión. `receive` exige `decision_at < label_not_before <= available_at <= cutoff`. Si el calendario todavía no acredita una maduración, `label_not_before=None` solo permite censurar la fila. Una censura no aporta una etiqueta.

`publish()` espera a resolver o censurar todas las filas de cada cohorte. Prepara las actualizaciones antes de confirmarlas y publica primero por fecha de decisión y después por mercado. Las filas y referencias se procesan con orden canónico. La cola pendiente queda separada de las estadísticas recuperables. La llegada de resultados de un episodio descartado genera su recibo, pero no lo vuelve a insertar en el registro.

`scores()` devuelve copias de las estadísticas. `export()` solo guarda eventos publicados. `restore()` comprueba versión, identidad, límites, maduraciones, contadores y huella. Rechaza estructuras profundas en campos escalares antes de serializarlas. La huella detecta modificaciones del contenido, pero no autentica a su productor ni demuestra que los pronósticos sean reales.

Los límites predeterminados son 1.024 episodios, 40.960 filas pendientes, 512 cohortes y 262.144 referencias. Los máximos configurables son 40.960 cohortes y 524.288 referencias. Una cohorte admite hasta 8.192 filas y cada fila hasta 64 supresiones. La capacidad pendiente se comprueba antes de copiar otro snapshot. No se guarda el grafo de autograd ni se consume RNG. El registro usa escalares CPU y no necesita un backend CUDA.

Las validaciones de emisión y publicación recorren las referencias pendientes y las referencias de las cohortes que cierran, respectivamente. Su coste está acotado por los límites declarados, aunque no es constante. La verificación del snapshot incluye sus tensores y la serialización ocupa espacio transitorio adicional. No se mide solo el tamaño de las estadísticas finales.

## Productor del lector estándar K1

`apply_spaced_utility_readout` consume `PreparedDecisions`, la cabeza del padre, el lector y su snapshot. Devuelve la predicción ordinaria y un diagnóstico opcional. Con `enabled=False` delega la ruta anterior, incluidas las opciones diferenciables y K = 2 o K = 4. La activación exige el lector estándar `bank`, K = 1, módulos en `eval` y ninguna diferenciación. Esta función rechaza lectores regionales y refinamientos superiores. El productor regional independiente se describe a continuación. No se modifica la identidad ni el checkpoint del modelo.

El productor reutiliza claves, valores proyectados e IDs del soporte elegido. Hace una sola selección y una sola proyección de valores. Repite una proyección pequeña de la consulta para reconstruir los logits del soporte, evitando añadir una caché de logits a las rutas históricas. Para cada episodio enmascara su logit y calcula el softmax sobre los restantes. No utiliza la resta de contribuciones dividida por `1 - peso`, que puede fallar por cancelación o cuando el peso redondea a uno. Las pruebas incluyen ese caso saturado en FP32 y FP64.

Cada lectura suprimida pasa por el refinador completo y la misma cabeza. Si solo había un vecino, la lectura queda ausente y a cero, pero se aplica también el refinador sin banco, con su sesgo y su entrada de presencia. No se sustituye arbitrariamente por la predicción del padre. Con memoria vacía no se generan supresiones. La cabeza cuantílica existente conserva sus cinco salidas y su mediana. El diagnóstico solo guarda puntos para el criterio de error, sin añadir cabezas.

Antes de leer se comprueba el presupuesto conjunto mediante la cota del lector para `1 + min(episodios, vecinos)` refinamientos. Es conservadora porque algunas estructuras se reutilizan. El cálculo mantiene el orden de selección y acota las supresiones a los ocho vecinos admitidos por el lector. Se rechazan cabezas con otra precisión o dispositivo y autocast sin un contraste independiente. No se cambia la política numérica global para obtener cifras más rápidas.

`SpacedUtilityDiagnostics` conserva flujos, corte, huella del snapshot, IDs y puntos. `ledger_rows` exige maduraciones previstas alineadas por flujo antes de transferir los tensores. Convierte IDs y puntos por bloques a CPU y comprueba finitud e identidades sin construir una segunda copia normalizada de todas las filas. El registro mantiene su validación independiente al recibirlas. No introduce resultados financieros. Su huella identifica el snapshot declarado, sin autenticar al productor. La verificación fuerte de bytes corresponde a la frontera de publicación y recuperación de la instantánea, como en el lector anterior. Las comprobaciones rápidas por consulta no prometen detectar una modificación externa mediante `.data` después de esa frontera.

## Productor del lector regional K1

`memory.spaced_regional_readout.apply_regional_spaced_utility_readout` mantiene la emisión original del lector regional y suprime cada episodio de todas sus vistas a la vez. Las regiones utilizan sus propios pesos sobre el mismo soporte de IDs. No se crean episodios adicionales ni otra historia persistente. Los tensores seleccionados y proyectados son buffers transitorios de cálculo.

La predicción ordinaria utiliza `apply_episodic_readout` sin cambios. El diagnóstico reutiliza sus posiciones y grafos, sin otra búsqueda ni otra preparación MAC. Recalcula una consulta y una proyección de los valores seleccionados para compartirlas entre las supresiones. Con dos o más vecinos hay dos proyecciones del pool en total, una ordinaria y otra diagnóstica. El lector anterior no expone su pool, y esta decisión evita modificar sus fuentes e identidades canónicas. No se presenta como reutilización de una única proyección total.

Para cada columna del soporte, se enmascara su logit en todas las regiones y se calcula de nuevo el softmax. No se divide por uno menos el peso original, que puede redondear a cero. Después se ejecutan la transformación regional, la comunicación y el refinador completo. Con un único episodio, su supresión genera lectura ausente y cero, conservando el sesgo y la entrada de presencia del refinador. Con banco vacío solo existe el punto original.

El productor reutiliza las funciones originales de mezcla lineal, recurrencia J = 1, 2 o 4, parada adaptativa y jerarquía. Cada supresión aplica su propio criterio de parada, en vez de forzar el número de pasos de la lectura original. Suprimir evidencia no desactiva una región ni elimina una arista. Esas son otras intervenciones. La cota de trabajo y la cuota de actualizaciones se comprueban para el conjunto de `1 + vecinos` lecturas antes de empezar, sin renovar la cuota para cada diagnóstico. Todas parten del mismo estado de trabajo. No forman una cadena de refinamientos K = 9.

La activación requiere `bank`, K = 1, evaluación, precisión común FP32 o FP64 y ausencia de autocast y diferenciación. La ruta apagada conserva las salidas y gradientes del lector anterior, incluidos K = 2 y K = 4. Los diagnósticos reutilizan `SpacedUtilityDiagnostics` y su contrato de emisión y maduración. Las tres fuentes canónicas del puente, recurrencia y jerarquía permanecen intactas, junto con las 34 fuentes del predictor base.

La [verificación regional](../../reports/engineering/spaced-regional-20261027/verification.json) registra 365 pruebas CPU aprobadas sin omisiones, incluidas 142 del productor nuevo y 223 regresiones. La selección excluye 89 casos CUDA, tres de ellos nuevos y todavía pendientes. La referencia independiente filtra un snapshot pequeño por fila solo en pruebas y ejecuta el lector original completo. Comprueba el soporte sin relleno, todas las vistas, condiciones vacías, regiones inactivas, atención saturada, cuantiles, presupuestos, orden por activo, RNG y recuperación. Incluye pesos densos y consultas distintas para una y ocho regiones, además de los casos estructurados. Las pruebas con gradientes no ejecutan pasos de optimizador ni ajustan etiquetas.

Las nueve [mutaciones dirigidas](../../reports/engineering/spaced-regional-20261027/mutations.json) se detectan con controles verdes antes y después. Incluyen suprimir solo en una región, omitir mensajes jerárquicos, forzar todos los pasos adaptativos, ignorar regiones inactivas, conservar lectura al retirar el último episodio, saltarse la cuota conjunta, repetir la búsqueda, admitir diferenciación activa y transponer los pesos receptores. El módulo nuevo cubre 79/79 sentencias y 14/14 ramas, con CRAP máximo 16 según [coverage.py 7.16.1 y Radon 6.0.1](../../reports/engineering/spaced-regional-20261027/quality.json). Los predicados compuestos no instrumentan cada condición como una rama y estas cifras no prueban calidad global.

La [medida de coste](../../reports/engineering/spaced-regional-20261027/cost.json) conserva dos pasadas de ocho combinaciones de tamaño y lector, con dos calentamientos y diez muestras por modo. Cada pasada mantiene exactamente el punto original en sus 96 pares. Incluye lectura, cabeza, ocho supresiones, conversión de filas y JSON. La preparación MAC queda fuera del tiempo por llamada. Son fixtures CPU FP32 de dimensión 32, con anchuras 5/2/2/3/3 y evidencia de 64 coordenadas. No representan las formas del corpus completo.

| Lector | Filas | Original, medianas de las pasadas 1 / 2 en ms | E2, medianas de las pasadas 1 / 2 en ms |
| --- | ---: | ---: | ---: |
| Anillo lineal | 64 | 1,157 / 3,493 | 4,330 / 14,490 |
| Recurrencia J4 | 64 | 1,934 / 5,204 | 10,576 / 28,479 |
| J4 con parada compacta | 64 | 2,893 / 7,020 | 18,280 / 44,175 |
| Jerarquía de árbol | 64 | 2,661 / 7,529 | 17,018 / 46,522 |
| Anillo lineal | 256 | 2,659 / 13,226 | 10,301 / 49,041 |
| Recurrencia J4 | 256 | 4,400 / 11,616 | 23,530 / 62,485 |
| J4 con parada compacta | 256 | 5,987 / 16,610 | 39,275 / 106,466 |
| Jerarquía de árbol | 256 | 7,250 / 18,566 | 46,252 / 118,752 |

Los procesos completos tardan 5,27 y 14,24 segundos y alcanzan 754.852 y 736.568 KiB de RSS absoluto. El productor conserva los mismos bytes. La segunda versión del script añade la creación de la carpeta de salida antes de escribir, fuera del tiempo por llamada. El recibo incluye un patch para reconstruir la primera versión. La variación afecta a ambos modos, no se midió una causa y no se atribuye a ese cambio ni al modelo. No se descarta ninguna pasada ni se presenta una latencia estable.

El recibo conserva las muestras y las mediciones separadas de asignación Python. Excluye ledger, consolidación, persistencia durable, disco, CUDA y transferencias GPU. No mide energía ni una distribución fiable de p95/p99. El diagnóstico añade coste y no hay mejora predictiva medida.

```bash
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_regional_readout.py tests/memory/test_spaced_utility_readout.py tests/models/titans/test_evidence_bridge.py tests/models/titans/test_bridge_intervention.py tests/models/titans/test_bridge_recurrence.py tests/models/titans/test_bridge_hierarchy.py -m 'not cuda' -q
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-regional-20261027/cost_probe.py "$HOME/.cache/mars-titan/spaced-regional-cost.json"
```

La medida requiere un destino nuevo. El productor está integrado también en `FinancialSession`, como se describe más abajo. El cargador seleccionado y el estudio de ventanas regionales siguen pendientes. El adaptador estándar los rechaza explícitamente antes de ejecutar. Esta comprobación técnica no levanta el bloqueo de aprendizaje ni abre el test final.

## Retención sobre el banco nativo

`SpacedUtilityBank` utiliza `EpisodicMemory` causal v2 del enlace C++ existente. `propose` recibe registros maduros ya admitidos y un ledger publicado del mismo corte. Devuelve `SpacedRetentionProposal`, con un banco nuevo y su evidencia sincronizada. No modifica ninguno de los estados anteriores. La publicación conjunta corresponde a la sesión financiera descrita más abajo, no a esta propuesta en memoria.

El primer encaje corresponde a M1. No sustituye los tres índices de M2/M3 ni cambia sus condiciones de admisión. Los consumidores actuales siguen rechazando configuraciones de retención que no admiten. No se presenta este backend como integración de E2 en todas las políticas.

Cada banco exige codec, predictor, mundo, partición y ventana. Su huella incluye configuración, fuentes, NumPy y versión y bytes del enlace nativo. El ledger asociado lleva `scope_id` igual a esa huella. Se rechazan evidencias de otro modelo, política o ventana antes de consultar sus datos. El modo sin ámbito conserva el formato anterior del ledger y no se acepta como evidencia de este banco. `usage` permite adelantar el presupuesto de copias y `synchronize` sigue los IDs retenidos sin puntuar nuevos episodios ni resolver las colas pendientes.

Los modos son:

- `utility`: reserva los últimos `min(frontier, nuevos)` IDs y protege el resto de plazas con episodios antiguos calificados, por ganancia media descendente y después ID. Completa por recencia de admisión.
- `permuted_utility`: aplica la misma regla y reserva, pero permuta la asignación de las estadísticas entre los IDs antiguos con un PCG64 propio. No reescribe la evidencia real ni las etiquetas. Conserva la distribución de utilidades usada por ese control.
- `uniform_frontier`: conserva la misma reserva y sortea sin reemplazo el resto entre los candidatos actuales. Es una selección condicional al conjunto disponible, no una muestra uniforme de todo el stream histórico.
- `recent`: conserva los mayores IDs de admisión, con la misma reserva.
- `reservoir`: conserva el reservorio nativo original, sin imponer la reserva. Es un control interpretativo. La comparación con las otras políticas cambia también esa condición y debe distinguirse del contraste con utilidad permutada.

Los representantes son registros reales del banco previo o del lote entrante. Se preservan sus valores y etiquetas. La validación nativa normaliza las claves y la escritura recibe los originales para evitar normalizar una clave ya normalizada. No se invoca k-median ni se calcula un objetivo geométrico para atribuir utilidad predictiva.

El snapshot del banco conserva identidad, archivo nativo, RNG del control, recibo y checksum. La recuperación comprueba esquema, tipos, cursores, pertenencia, reserva, ranking y presupuesto antes de aceptar el estado. Después debe conciliarse con el ledger recuperado mediante `validate_evidence`. El checksum no autentica la historia ni demuestra que las ganancias registradas sean verdaderas. La procedencia de la predicción emitida y la publicación atómica de los dos estados siguen siendo requisitos de la integración financiera.

La estimación de trabajo incluye copias nativas, exportación del ledger, filas y referencias pendientes. Se contrasta con el límite configurado antes de copiar. Es conservadora y no certifica un pico exacto del asignador de Python. No se forma una matriz de distancias N por N. El coste de ordenar utilidades y convertir estados debe contarse, junto con la recuperación y la continuación.

La [verificación del banco](../../reports/engineering/spaced-retention-20261027/verification.json) y su [benchmark](../../reports/engineering/spaced-retention-20261027/cost.json) utilizan el enlace nativo real, con fixtures y sin aprendizaje. El script permite repetir las quince combinaciones de capacidad 64/256/1.024 y las cinco políticas:

```bash
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_retention.py tests/memory/test_spaced_utility.py tests/memory/test_spaced_utility_readout.py tests/memory/test_retention_bank.py -m 'not cuda' -q
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-retention-20261027/cost_probe.py /tmp/mars-titan-e2-retention-cost.json
```

Ambas órdenes requieren `MARS_TITAN_EPISODIC_NATIVE` con la ruta del enlace compilado compatible. No compilan ni descargan un binario. El benchmark incluye propuesta, snapshot, recuperación, conciliación y dos continuaciones emparejadas. Excluye predictor, almacenamiento durable y GPU. No es una comparación de velocidad con el `RetentionBank` que además calcula una distorsión geométrica.

La batería del banco aprueba 232 casos CPU, incluidos 34 de retención E2, 57 del registro, 60 del productor y 81 de retención histórica. Excluye los dos casos CUDA del productor. Las ocho mutaciones dirigidas finales se detectan con referencias verdes antes y después. La comprobación de cuatro módulos registra 566/567 sentencias y 91/92 ramas, con CRAP máximo 19. Una primera prueba de ámbitos solo comprobaba el rechazo final, que también hacía la recuperación. Se amplió para exigir el rechazo antes de consultar la evidencia ajena. La paridad del ledger sin ámbito conserva cinco payloads completos frente a `f2a6265da`, incluida una recuperación con etiquetas parciales. Ninguna cifra se suma a las baterías anteriores.

La medida usa dos hilos, un calentamiento y cinco repeticiones por caso. Cada ciclo incluye una propuesta, la recuperación y dos continuaciones, comparadas entre sí. Conserva una cohorte pendiente con una referencia por episodio repartida entre filas de hasta 64 supresiones literales. Los datos son fixtures y no proceden del corpus financiero.

| Política | 64 plazas, mediana en ms | 256 plazas, mediana en ms | 1.024 plazas, mediana en ms |
| --- | --- | --- | --- |
| Utilidad | 12,991 | 33,367 | 104,284 |
| Utilidad permutada | 13,122 | 35,632 | 106,366 |
| Uniforme con reserva | 12,970 | 31,086 | 89,642 |
| Recencia | 12,026 | 31,520 | 108,293 |
| Reservorio | 12,466 | 30,317 | 89,526 |

El proceso completo tarda 13,03 segundos y su pico RSS externo absoluto es 738.416 KiB. El mayor pico Python medido en pasadas separadas es 6.953.963 bytes y excluye asignaciones nativas. El payload lógico con 1.024 plazas ronda los 779–801 kB y no incluye el framing del almacenamiento. No se midió energía. Las diferencias de tiempo entre políticas en estos fixtures no son un criterio para elegir la de menor error predictivo.

## Integración financiera durable

`FinancialSession` reconoce `SpacedRetentionConfig` en su argumento de retención. `RetentionConfig` conserva el recorrido de referencia. El enlace E2 es un tipo cerrado, reutiliza el mismo bucle de bloques y solo admite M1, lector estándar K1 y C desactivado. No introduce callbacks predictivos arbitrarios ni combina implícitamente B6, B4 o trabajo regional.

Cada bloque prepara el predictor una sola vez. El productor usa ese estado y conserva el orden físico del bloque, incluso si se solicita explícitamente el modo de varios mercados. Después se agrupan los diagnósticos por mercado para registrar las cohortes. La fecha prevista se obtiene con `PrefixTargetVerifier.expected_maturity`, que consulta el calendario sin leer precios futuros. Una fecha desconocida no se transforma en una etiqueta.

El ledger abierto solo vive dentro de la transacción en curso. Tras emitir, el ejecutor entrega los resultados maduros, exclusiones y finalizaciones. Sus puntos se comparan con las predicciones originales, se cierran las cohortes completas y se propone la retención. El ledger cerrado y su último evento se guardan en el mismo bundle que banco, memoria rápida, pendientes y evidencias. La poda conserva las referencias de las dos generaciones recuperables. `spaced_state()` y `spaced_event()` permiten inspeccionar el estado y el último evento. No constituyen por sí solos un archivo científico completo de todos los eventos anteriores.

La liquidación posterior a una decisión puede compartir su instante, tal como admite el ejecutor nativo. `begin(..., feedback_only=True)` permite ese feedback sin autorizar nuevas predicciones en ese corte. El formato anterior y el avance estricto permanecen como opción predeterminada del ledger.

La sesión deriva los límites de los 40.960 pendientes transitorios admitidos por el ejecutor: hasta ese número de cohortes y `40.960 × vecinos` referencias. Los máximos configurables del registro pasan a 40.960 cohortes y 524.288 referencias. Sus defaults de 512 y 262.144 no cambian. Son límites de almacenamiento, no una relajación de precisión o disponibilidad del corpus. Los artefactos mantienen su límite de tamaño.

Esta revisión cambia las fuentes del coordinador y su identidad de integración. E2 declara la receta `frozen_financial_spaced_session_v3`. El predictor, lector canónico, parámetros y sus 34 fuentes permanecen intactos. Los contratos de integración anteriores requieren su runtime fijado. No se eluden las comprobaciones para reanudar un checkpoint bajo otro código ni se sobrescriben resultados históricos. La comparación debe identificar tanto el consumidor como el contrato de sesión.

La retención y su reserva no garantizan una mejora. Si una cohorte nueva llena toda la capacidad y el relleno por recencia expulsa episodios antes de recibir varias confirmaciones, E2 puede no llegar a protegerlos. Capacidad, rotación y horizonte de maduración condicionan esa posibilidad. No se cambia la política después de observar resultados para ocultar este caso.

La verificación de la integración, sus mutaciones y costes está en [su recibo](../../reports/engineering/spaced-session-20261027/verification.json). Las pruebas usan fuentes sintéticas con etiquetas manuales, no ajustan parámetros ni ejecutan campañas científicas.


La batería de integración y regresión pasó 222 casos CPU y dejó fuera dos casos CUDA del productor. Una ejecución posterior de mecanismos y E2 pasó 175 casos. Se solapan y no se suman. Los 19 casos GRU que inicialmente se omitieron por el enlace elegido se ejecutaron con otro binario compatible ya compilado, con 19 aprobados y ninguna omisión. Se detectaron seis mutaciones dirigidas con referencias verdes antes y después. El módulo de integración nuevo cubre 140/146 sentencias y 30/34 ramas, con CRAP máximo 23. La cobertura de los módulos compartidos es parcial en esta selección. No se ha ejecutado la suite completa.

Las pruebas cubren las cinco políticas, el cierre y reapertura, la siguiente predicción y actualización, las siete fronteras de fallo expuestas por el ejecutor, el orden de activos y el cotejo de puntos pendientes con las predicciones nativas. La preparación de la salida conserva los tensores y registros del reservorio de referencia. Son comprobaciones de fixtures y no resultados de generalización.

El benchmark durable alterna ocho pares de recorridos, con un calentamiento y tres muestras por modo y tamaño. Cada recorrido incluye cuatro decisiones, cierre, recuperación, liquidación y la siguiente decisión, con almacenamiento y comprobaciones. Las modalidades usan anchuras compactas 5/2/1/3/3 y presencia de precios y gráficos. Las etiquetas son manuales y la fuente es sintética.

| Activos ficticios | Sin E2, mediana en segundos | E2 con reservorio, mediana en segundos |
| --- | --- | --- |
| 64 | 4,323 | 6,378 |
| 256 | 15,667 | 19,842 |

Los estados comunes coinciden en cada par, incluidos puntos, registros y tensores rápidos. Estas cifras incluyen el coste de gestionar diagnósticos y recuperación, no solo la consulta. No se utilizan para afirmar una mejora predictiva ni para elegir una política de utilidad. Tres muestras no permiten estimar colas de latencia con precisión. GPU y energía quedan sin medir.

```bash
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_session.py -q
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-session-20261027/cost_probe.py /tmp/mars-titan-spaced-session-cost
```

El benchmark requiere un directorio nuevo y `MARS_TITAN_EPISODIC_NATIVE` con el enlace compatible. Escribe allí las fuentes de fixture y los recorridos. El informe ligero está en [cost.json](../../reports/engineering/spaced-session-20261027/cost.json). No ejecuta entrenamientos ni abre el test final.

## Sesión durable con lector regional

`SpacedTitansBinding` elige el productor regional cuando el consumidor declara un puente de evidencia. `SpacedRetentionConfig` activa E2 y `RetentionConfig` conserva la ruta anterior. La combinación admite M1, K = 1 y C desactivado. No activa el programa regional anterior a MAC ni lo mezcla implícitamente con B6 o B4. El coordinador `FinancialSession`, su consumidor de referencia y las 34 fuentes canónicas permanecen intactos.

Antes de MAC, el enlace comprueba la cuota conjunta de la lectura original y las supresiones. Suma la estimación de la traza ordinaria retenida de toda la cohorte, los puntos y filas diagnósticas y el trabajo del bloque. No basta con que cada parte quepa por separado. La estimación es conservadora y no certifica el pico de RAM. El control guarda refinamientos diagnósticos, proyecciones adicionales y una cota superior de actualizaciones regionales. Esta última no se presenta como profundidad adaptativa observada. La traza ordinaria mantiene sus propios contadores y profundidades reales.

Banco, pesos rápidos, pendientes y evidencia se publican juntos mediante el bucle transaccional existente. Las comprobaciones de artefactos concilian la traza ordinaria, las emisiones, el soporte, los bloques y los contadores. La confirmación comprueba también el tipo de evento del recibo nativo. Un valor nulo, falso o vacío no se sustituye por una inferencia del tipo. Si se rechaza un evento, el ejecutor se cierra y la recuperación vuelve a la generación confirmada, sin publicar estados parciales.

La firma regional añade el descriptor `regional_spaced_session_v1` a los campos existentes. Se fijan las fuentes del enlace, el productor y el helper regional. La edición del código E2 cambia sus contratos de ejecución, incluso para su lector estándar. Los contratos anteriores necesitan el runtime fijado en su manifiesto. No se omite esa comprobación para recuperar ni se sobrescriben sus artefactos. Los pesos y fuentes del predictor conservan su identidad.

La [verificación durable regional](../../reports/engineering/spaced-regional-session-20261027/verification.json) registra 426 pruebas CPU correctas sin omisiones, incluidas 43 nuevas y 383 regresiones. La selección excluye siete casos CUDA, dos de ellos nuevos y aún pendientes. Comprueba las cinco políticas sobre lectores J4, adaptativo, jerárquico y con regiones desactivadas, además de la paridad con reservorio, cuantiles, calentamiento, cierre, orden temporal, recuperación y siete fronteras de fallo nativas. La batería no ajusta parámetros ni calcula etiquetas residuales.

Se detectan nueve [mutaciones dirigidas](../../reports/engineering/spaced-regional-session-20261027/mutations.json) con controles verdes antes y después. Cubren el despacho, las cuotas antes de MAC, los controles durables, los tipos de contadores y de eventos nativos, la traza ordinaria y el rechazo del adaptador estándar. La [cobertura y el CRAP](../../reports/engineering/spaced-regional-session-20261027/quality.json) recogen 345/347 sentencias y 73/74 ramas de los tres módulos afectados, con CRAP máximo 23. El helper nuevo cubre 61/61 sentencias y 8/8 ramas. Se usan coverage.py 7.16.1 y Radon 6.0.1, con sentencias para CRAP. No se declara cobertura de cada condición ni de la suite completa.

La [comparación con el enlace anterior](../../reports/engineering/spaced-regional-session-20261027/legacy-parity.json) ejecuta la ruta sin E2 con el código de `babe9f780` y el actual sobre las mismas fuentes y parámetros de fixture. Conserva exactamente 16 predicciones, episodios retenidos, snapshot completo e identidad del contrato. Esta comparación no acredita recuperar una sesión E2 antigua bajo el código nuevo.

```bash
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_regional_session.py tests/memory/test_spaced_session.py tests/memory/test_spaced_inference.py tests/memory/test_spaced_utility_readout.py tests/memory/test_spaced_regional_readout.py tests/memory/test_spaced_retention.py tests/memory/test_retention_bank.py tests/training/test_learning_hold_inventory.py tests/training/test_learning_hold_graph.py -m 'not cuda' -q
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-regional-session-20261027/compare_legacy.py "$HOME/.cache/mars-titan/spaced-session-parity"
```

La comparación necesita un destino nuevo y el enlace nativo compatible. El coste durable de este enlace aún está pendiente de medida. Los costes del productor aislado no lo sustituyen. No hay resultados de generalización ni acceso al test final. La adaptación regional de fases, el cargador y el controlador se mantienen pendientes.

## Adaptador para fases de ventanas congeladas

`training.spaced_inference.SpacedInference` consume el padre y el lector de un `FrozenFinancialConsumer`, la receta del lector, `PrefixTargetVerifier` y la retención E2. Reutiliza preparación, codificación, emisión, filas y métricas de `MarsTitanInference`. Su método público `evaluate` recibe `FinancialObservationSource` y comprueba el bloqueo de aprendizaje antes de acceder a la fuente. Solo admite M1, lector estándar K1 y C desactivado.

Cada evento emite todas sus predicciones con el mismo snapshot. Después registra las supresiones por mercado, recibe los resultados maduros, publica la evidencia y propone el banco sincronizado. Las fechas previstas proceden del calendario de los prefijos, sin consultar precios futuros. El adaptador conserva temporalmente la referencia a `next_state` que la ruta común quita antes del lector. La utiliza para las identidades de los flujos durante ese bloque, sin copiar ni acumular pesos rápidos.

El banco utiliza el mismo ámbito de mundo, objetivo residual y horizonte uno que la sesión financiera. La evidencia no se traslada entre fases. Cada recorrido reinicia banco, ledger y memoria rápida y exige un cierre explícito. Una interrupción descarta el estado parcial y la fase se repite desde el principio. La recuperación por evento sigue siendo responsabilidad de la sesión durable. `spaced_state` copia la evidencia final y `spaced_snapshot` permite comparar también el banco, con `resumable=False`. Ninguno es un checkpoint completo del predictor.

Este recorrido compara políticas de retención sobre unos mismos pesos ya elegidos. No sustituye al contraste de entrenar cada variante bajo condiciones emparejadas. La carga del estado seleccionado se comprueba por separado. El controlador externo, la CLI y la agregación estadística del estudio siguen pendientes. Las pruebas usan etiquetas manuales y no autorizan abrir una ventana científica.

La [verificación del adaptador](../../reports/engineering/spaced-window-20261027/verification.json) registra 106 pruebas CPU distintas aprobadas, incluidas las 26 del módulo nuevo y 17 del inventario y grafo del bloqueo. La batería de regresión de 87 casos y las ejecuciones posteriores se solapan. Las cinco políticas y las capacidades de 4 y 16 plazas conservan exactamente las emisiones, la evidencia y el banco final de la sesión durable. Se comprueban también dos bloques físicos por evento, filas cuantílicas frente al reservorio original, cierre, interrupción, RNG y lectura de un índice de observaciones construido con fixtures. No hay pasos de optimizador ni ejecución de la suite completa.

Las [cuatro mutaciones dirigidas](../../reports/engineering/spaced-window-20261027/mutations.json) detectan descartar resultados maduros, omitir censuras al cerrar, aceptar un punto diagnóstico distinto del emitido y saltar el bloqueo científico. Cada una tiene controles verdes antes y después y cero omisiones. La [cobertura](../../reports/engineering/spaced-window-20261027/quality.json) del módulo nuevo es 119/119 sentencias y 22/22 ramas, con CRAP máximo 6. Se ha usado coverage.py 7.16.1 y Radon 6.0.1, con cobertura de sentencias para CRAP. Las 34 fuentes canónicas y el recorrido histórico permanecen intactos. Estos resultados no prueban ausencia de errores ni generalización.

La [medida de coste](../../reports/engineering/spaced-window-20261027/cost.json) alterna doce pares, con un calentamiento y cinco muestras medidas por modo y tamaño. Cada fase incluye cinco decisiones, etiquetas maduras, una liquidación intermedia y cierre. Ambos modos usan el reservorio original para aislar el coste de añadir diagnósticos y evidencia. Conservan exactamente los puntos y las métricas comunes en cada par.

| Activos ficticios | Fase original, mediana en segundos | Fase E2, mediana en segundos |
| --- | --- | --- |
| 64 | 0,077 | 0,142 |
| 256 | 0,230 | 0,444 |

Se usan dos hilos CPU, FP64, estado de 32 unidades, cuatro vecinos y capacidades de 256 y 1.024 episodios. Los fixtures tienen anchuras 5/2/1/3/3 y solo precios y gráficos presentes. La medida incluye las comprobaciones de integridad y excluye construcción, lectura del dataset y persistencia durable. El proceso externo completo tarda 17,66 s y su pico RSS absoluto es 797.400 KiB. No se ha medido el incremento RAM por modo, CUDA ni energía. Cinco muestras no permiten caracterizar colas de latencia con precisión.

El recibo conserva también las dos pasadas anteriores. La primera tenía una etiqueta de exclusión de disco demasiado amplia y la siguiente precisó esa descripción. La última registra las huellas de las fuentes ejecutadas. Las tres conservan los mismos puntos y métricas, con medianas de E2 entre 0,142 y 0,160 s para 64 activos y entre 0,444 y 0,528 s para 256. No hubo una optimización del cálculo entre pasadas y no se atribuye esa variación a una aceleración del modelo.

```bash
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_inference.py tests/memory/test_spaced_session.py tests/memory/test_mars_titan_session_parity.py tests/memory/test_spaced_retention.py -q
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-window-20261027/cost_probe.py "$HOME/.cache/mars-titan/spaced-window-cost"
```

El benchmark exige un destino nuevo y el mismo enlace nativo compatible. La orden mide fixtures sin aprendizaje y solo cubre el adaptador de fases, sin el controlador completo.

## Carga de los parámetros elegidos

`training.spaced_selected.load_selected_reader` recibe la carpeta del modelo seleccionado, su vista original y la especificación de entrada del padre. Reutiliza los constructores y cargadores de `mars_titan_walk_forward` y la carga restringida de checkpoints. Devuelve un `SelectedSpacedReader` con consumidor congelado, codec, receta y procedencia. No ejecuta predicciones ni restaura optimizador, RNG, memoria rápida, banco o etiquetas pendientes. El payload de recuperación se libera tras cargar y verificar los parámetros.

La selección exige la variante estándar M1 K1 sin puente y el padre `mac_online` de la misma vista y semilla, sin control C. Comprueba `run.json`, `selected.json`, los informes de ajuste y sus mejores checkpoints. Los parámetros deben conservar forma, precisión y finitud. La selección usa `best` aunque exista un archivo posterior, y rechaza archivos fuera del presupuesto de 512 MiB antes de calcular su checksum. Las referencias deben conservar rutas e identificadores compatibles. Al terminar vuelve a comprobar los archivos de procedencia.

El índice de entrada debe ser el original del padre. El controlador puede obtener su especificación desde el informe mediante `_parent_specification`, sin abrir el índice de entrenamiento. Cargar otra vista o remapear otro índice requiere un contrato temporal separado. El cargador no activa esa transferencia. Los checksums detectan cambios, pero no autentican que se haya entrenado un modelo ni confirman el contenido científico de un informe.

La [verificación de carga](../../reports/engineering/spaced-selected-20261027/verification.json) recoge 34 pruebas nuevas correctas y cinco [mutaciones dirigidas](../../reports/engineering/spaced-selected-20261027/mutations.json), con controles verdes antes y después y cero intentos de optimizador. Las pruebas usan checkpoints reales con pesos fijados a mano e identificados como fixtures sin entrenamiento. Cubren cabezas escalar y cuantílica, RNG, selección del mejor estado, metadatos con checksums recalculados, pesos no finitos, forma y precisión incorrectas, enlaces simbólicos, límites de lectura y cambios durante la carga. Una fase E2 completa ejecuta el consumidor cargado y reproduce su estado al repetirla desde cero.

La batería del cargador y las regresiones de checkpoints reúnen 47 pruebas distintas correctas. Un caso antiguo que necesita un paso de AdamW queda omitido por el bloqueo y un caso CUDA se excluye de la selección CPU. Los recuentos se solapan con las pruebas nuevas y no se suman. La [cobertura](../../reports/engineering/spaced-selected-20261027/quality.json) es de 97/97 sentencias, con CRAP máximo 30. Coverage.py 7.16.1 no instrumenta ramas en estos predicados booleanos encadenados, y el informe registra cero ramas. No equivale a comprobar cada condición. La complejidad se calcula con Radon 6.0.1. Las 34 fuentes canónicas permanecen intactas.

La [medida de carga](../../reports/engineering/spaced-selected-20261027/cost.json) registra medianas CPU de 21,502, 24,414 y 45,744 ms para estados de 32, 64 y 128 unidades, respectivamente. Usa FP64, dos hilos, dos calentamientos y diez muestras por tamaño. Comprueba metadatos y checkpoints, construye los módulos, congela y revalida. Los fixtures no contienen archivos de predicciones que auditar. No estima la carga de una campaña completa, inferencia, CUDA ni energía. El proceso completo tarda 4,36 s y su pico RSS externo absoluto es 763.128 KiB, sin atribuir un incremento por tamaño.

```bash
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_selected.py -q
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-selected-20261027/cost_probe.py "$HOME/.cache/mars-titan/spaced-selected-cost"
```

El benchmark requiere un destino nuevo. La prueba de integración con E2 necesita además el enlace nativo compatible. Estas órdenes solo comprueban artefactos técnicos y no levantan el bloqueo de aprendizaje.

## Controlador del diagnóstico congelado

`training.spaced_window.run_spaced_window` y `scripts/run_spaced_window.py` consumen una vista temporal, un lector seleccionado, el manifiesto original de datos y la [configuración independiente](../../configs/experiments/spaced-utility-study-v1.json). Esta declara `baseline`, `reservoir`, `utility`, `permuted_utility`, `uniform_frontier` y `recent`. Cada salida corresponde a una sola variante. Todas conservan los parámetros elegidos, las filas y las fases del padre. El diagnóstico modifica la retención sin reentrenar y no sustituye la comparación principal de aprendizaje. La memoria neuronal mantiene sus actualizaciones asociativas previstas. El término congelado se refiere a los parámetros compartidos, no a inmovilizar todos los estados.

El controlador exige la misma vista e índice de entrada del modelo seleccionado. Comprueba que las fases originales coincidan y que la capacidad y semilla del banco sean las del lector. Fija hashes de configuración, fuentes, checkpoint e índices en `run.json`. La identidad indica la política realmente utilizada y si existe evidencia E2. `baseline` conserva `MarsTitanInference` y su reservorio, sin diagnósticos. El control `reservoir` añade los diagnósticos E2 con la misma retención original.

Cada fase empieza con memoria rápida, banco y evidencia vacíos. Publica predicciones por muestra, un Parquet de MAE, MSE y RMSE por mercado e instante y, cuando corresponde, el ledger final de evidencia. Recalcula las métricas por sesión desde las filas y las coteja con las del consumidor. La media de activos precede a la agregación de sesiones. El ledger no es un checkpoint completo por evento y no contiene un banco desde el que continuar a mitad de fase.

La confirmación de una fase es atómica mediante su registro en `run.json`. La reanudación verifica los checksums y no repite las fases confirmadas. Una fase interrumpida se vuelve a ejecutar desde el principio. Los fallos de preparación posteriores a adquirir una salida propia dejan estado, error y tiempo consumido. Las salidas ajenas y los archivos confirmados que hayan cambiado se rechazan. Los checksums no autentican la procedencia científica de esos artefactos.

El presupuesto predeterminado admite 900 segundos acumulados y hasta un millón de filas por fase. Una fase demasiado grande rechaza la ventana completa, sin quitar observaciones. El tiempo incluye los intentos anteriores y la preparación del actual. La pausa es cooperativa entre bloques, no un límite duro que interrumpa cualquier operación en curso. Las tablas y sus comprobaciones permanecen en memoria dentro del límite de filas. No se declara procesamiento ilimitado del corpus.

La CLI comprueba el bloqueo antes de leer fuentes o reservar recursos. Usa `cuda:0` por defecto y el controlador mantiene una reserva `GpuLease`. CPU requiere `--device cpu` explícito. El test final y la activación por defecto siguen cerrados. La ayuda y las pruebas técnicas se pueden comprobar con:

```bash
PYTHONPATH=src:. uv run --no-sync python scripts/run_spaced_window.py --help
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_study.py tests/memory/test_spaced_window.py tests/memory/test_spaced_selected.py tests/training/test_learning_hold_inventory.py tests/training/test_learning_hold_graph.py -q
```

La [verificación del controlador](../../reports/engineering/spaced-study-20261027/verification.json) registra 94 casos CPU aprobados sin omisiones, incluidos 43 de configuración, CLI y ventanas, 34 regresiones del cargador y 17 del bloqueo. Usa etiquetas manuales y checkpoints con pesos fijados, sin entrenamiento. La prueba de opciones CUDA utiliza un doble del controlador y no acredita una ejecución GPU. Las diez [mutaciones dirigidas](../../reports/engineering/spaced-study-20261027/mutations.json) se detectan con controles verdes antes y después. Comprueban activación indebida, pérdida del baseline, omisión de guardas, presupuesto, repetición de fases, checksums y esquema de recuperación.

La [cobertura y el CRAP](../../reports/engineering/spaced-study-20261027/quality.json) abarcan los tres módulos nuevos, con 238/241 sentencias y 43/46 ramas recorridas. Coverage.py 7.16.1 mide los recorridos y Radon 6.0.1 la complejidad. El CRAP máximo es 27. Las cifras no cubren todos los predicados ni prueban la ausencia de errores. La ruta CUDA y la evaluación científica permanecen pendientes.

La [medida del controlador completo](../../reports/engineering/spaced-study-20261027/cost.json) recorre 48 ventanas de fixtures, con seis controles, dos tamaños, un calentamiento y tres muestras medidas por control y tamaño. Usa CPU FP64 y dos hilos. Cada ventana de 64 activos emite 384 filas y la de 256 emite 1.536. Incluye carga del estado seleccionado, fuentes, las tres fases, auditorías y escrituras de predicciones, métricas y evidencia. Las exclusiones de preparación del fixture y firmas posteriores se refieren al tiempo por ventana. El tiempo total del proceso sí las incluye.

| Configuración | 64 activos, mediana en segundos | 256 activos, mediana en segundos |
| --- | ---: | ---: |
| `baseline` | 10,810 | 43,058 |
| `reservoir` | 10,551 | 42,755 |
| `utility` | 11,119 | 43,108 |
| `permuted_utility` | 11,090 | 42,917 |
| `uniform_frontier` | 11,282 | 42,956 |
| `recent` | 10,667 | 42,676 |

Los ocho pares de referencia y reservorio E2 conservan exactamente las predicciones. Los seis controles mantienen las mismas filas en cada ronda. El recibo conserva todas las muestras, incluida una de `utility` con 256 activos de 55,313 segundos. Tres muestras no permiten caracterizar colas de latencia. Las diferencias pequeñas de mediana no acreditan una aceleración.

El proceso completo tarda 24 min y 34,97 s y alcanza 1.025.028 KiB de RSS, sin atribuir un incremento a cada variante. Los precios del fixture usan grupos Parquet de tres filas, propios de las pruebas de integridad, aunque las anchuras de noticias y gráficos son 384 y 512. Esta disposición no representa la del corpus real. No se midieron energía, CUDA ni transferencias GPU. No hubo pasos de optimizador y se mantuvieron las aplicaciones ajenas y el perfil de energía. Para repetir la medida se necesita un destino nuevo y el enlace nativo compatible:

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-study-20261027/cost_probe.py "$HOME/.cache/mars-titan/spaced-study-cost"
```

## Fuentes y alcance

Mattar y Daw, *Prioritized memory access explains planning and hippocampal replay*, Nature Neuroscience 21, 1609–1617 (2018), proponen ordenar el acceso a recuerdos por su utilidad para decisiones en una tarea de navegación. El resumen describe el equilibrio entre necesidad y ganancia. E2 toma la pregunta sobre utilidad, pero su diferencia de errores y su criterio temporal son una propuesta propia. No implementa el algoritmo de planificación de ese artículo. [Artículo y DOI 10.1038/s41593-018-0232-z](https://www.nature.com/articles/s41593-018-0232-z).

Glas, Hübener, Bonhoeffer y Goltstein, *Spaced training enhances memory and prefrontal ensemble stability in mice*, Current Biology 31(18), 4052–4061.e6 (2021), observaron efectos del espaciado sobre memoria y similitud de actividad prefrontal en una tarea con ratones. El intervalo de 60 minutos descrito no se traslada a sesiones bursátiles. [Metadatos y resumen, DOI 10.1016/j.cub.2021.06.085](https://pubmed.ncbi.nlm.nih.gov/34324833/).

La consulta del 27 de octubre de 2026 contrastó metadatos y resúmenes de ambos artículos. No se declara lectura íntegra, reproducción de sus experimentos, comprobación Lean ni garantía formal de este diseño. Las pruebas sintéticas validan el contrato del registro, sin resultados sobre FinMultiTime.

## Comprobaciones reproducibles

Desde la raíz de esta rama, con su entorno uv preparado:

```bash
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_utility.py tests/memory/test_lesion_protocol.py tests/memory/test_lesion_ledger.py -q
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-utility-20261027/cost_probe.py /tmp/mars-titan-e2-cost-20261027.json
```

El productor estándar y sus regresiones se comprueban por separado:

```bash
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_utility_readout.py tests/memory/test_spaced_utility.py tests/models/titans/test_episodic_readout.py tests/models/titans/test_episodic_readout_fixed_episodes.py -m 'not cuda' -q
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-readout-20261027/cost_probe.py /tmp/mars-titan-e2-readout-cost.json
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-readout-20261027/compare_rows.py /tmp/mars-titan-e2-rows-comparison.json
```

La batería del productor aprueba 207 casos CPU, incluidos 60 del módulo nuevo. Excluye expresamente cuatro casos CUDA, dos del lector anterior y dos del productor. Los del productor aún no se han ejecutado en GPU. Las ocho mutaciones dirigidas de esta versión se detectan con referencias verdes antes y después. El módulo nuevo registra 88/88 sentencias y 10/10 ramas, con CRAP máximo 17. No se ha ejecutado la suite completa y estas cifras se solapan con las del registro, por lo que no se suman.

Los 34 archivos que componen la huella canónica de `FrozenFinancialPredictor` para su configuración base conservan los mismos bytes que en `404b91649`. El archivo del lector histórico permanece intacto. La adaptación de la salida está aislada en el módulo experimental y reutiliza la misma instancia de cabeza y la mediana existente. Catorce casos contrastan salidas y gradientes de las referencias frente a ese commit. No se presenta esta comprobación como recuperación de todas las campañas ni como validación CUDA.

La [evidencia del productor](../../reports/engineering/spaced-readout-20261027/verification.json), la [comparación de conversión](../../reports/engineering/spaced-readout-20261027/rows-comparison.json) y el [coste del lector con diagnósticos](../../reports/engineering/spaced-readout-20261027/cost.json) identifican fuentes y medidas. La comparación reconstruye solo el conversor anterior en una carpeta temporal, conserva el resto del productor y alterna ambos modos sobre las mismas filas. Las aplicaciones ajenas y el perfil de energía no se modificaron. El criterio técnico exige al menos un 5 % menos de mediana con 1.024 filas en ambas pasadas y ninguna regresión mayor del 5 % con 64 o 256. No es un criterio de mejora financiera.

La comparación final reduce la mediana un 12,36 % y un 10,33 % con 1.024 filas en sus dos pasadas y también reduce las medianas de 64 y 256 filas. Conserva exactamente filas y tamaño del JSON. El pico Python medido por separado pasa de 2.085.085 y 2.094.685 bytes a 1.807.336 bytes en ambas pasadas. El comparador conserva las muestras y el patch permite reconstruir la referencia. No se interpreta esta reducción como ahorro de toda la RAM ni como aceleración del predictor completo.

La medida final de lectura, supresiones, filas y JSON utiliza FP32, dos hilos, dos calentamientos por modo y diez muestras por tamaño. El padre MAC se prepara una vez por tamaño, fuera del tiempo por llamada. Los episodios tienen 64 coordenadas y el lector ocho vecinos y dimensión 32. Son fixtures compactos con anchuras 5/2/2/3/3 y entradas repetidas por flujo con IDs distintos. No reproducen las formas ni la distribución del corpus.

| Filas por llamada | Lectura ordinaria y JSON, mediana en ms | Lectura, ocho supresiones y JSON, mediana en ms |
| --- | --- | --- |
| 64 | 0,626 | 1,942 |
| 256 | 1,865 | 5,633 |
| 1.024 | 15,874 | 30,865 |

El proceso completo tarda 3,10 segundos y su pico RSS absoluto externo es 751.668 KiB. El mayor JSON diagnóstico ocupa 334.763 bytes. La medida excluye ledger, consolidación, persistencia durable, GPU y transferencias. No mide energía. Los costes absolutos proceden de una pasada independiente de la comparación intercalada y no se usan para calcular su reducción porcentual.

El archivo del benchmark se escribe en el destino indicado por el último argumento. El [recibo técnico](../../reports/engineering/spaced-utility-20261027/verification.json) recoge fuentes, pruebas, mutaciones, cobertura, CRAP y límites. El [informe de coste](../../reports/engineering/spaced-utility-20261027/cost.json) conserva las muestras individuales de dos sesiones, con JSON, recuperación y publicación. Excluye predictor, generación de supresiones, disco y GPU. No acredita latencia del modelo ni mejora de generalización.

La batería del registro y las regresiones seleccionadas aprobaron 102 casos sin omisiones, incluidos 51 del registro. Otra batería aprobó 155 casos de los puntos de entrada protegidos, con ajustes sustituidos por dobles de prueba. Se detectaron ocho mutaciones concretas con referencias verdes antes y después. Coverage.py 7.16.1 y Radon 6.0.1 registran 320/320 sentencias y 60/60 ramas en los dos módulos nuevos, con CRAP máximo de 19. Esta evidencia no se extrapola a toda la canalización ni a todos los fallos posibles.

La confirmación de coste usa un Ryzen 9 8945HS, dos hilos de cómputo y diez muestras tras dos calentamientos. Para dos sesiones de 64, 256 y 1.024 filas, las medianas son 8,043, 29,923 y 122,094 ms. Los JSON pendientes máximos ocupan 20.211, 79.923 y 320.584 bytes. El mayor pico de asignación Python observado con tracemalloc es 4.964.924 bytes en una pasada separada. Excluye las asignaciones nativas. El tiempo total externo es 5,81 segundos y su pico RSS absoluto, 715.296 KiB, incluye el proceso completo. No se midieron energía ni incremento RAM atribuible al módulo. La primera medida solapó mutaciones y se conserva aparte. La confirmación no solapó otras pruebas propias, pero mantuvo activas las aplicaciones ajenas y el perfil de energía existente.

Quedan pendientes la adaptación regional de fases, el cargador y el controlador, CUDA y la evaluación autorizada. El diagnóstico congelado no habilita entrenamientos, no cambia el baseline y no abre el test final.
