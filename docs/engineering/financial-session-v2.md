# Ciclo financiero con parámetros compartidos congelados

`FinancialSession` conecta `FinancialPredictor`, `EpisodicReadout`, el codec
fijo, el banco nativo y el puente de estados. El ejecutor nativo publica una
sola generación por evento. La implementación admite comprobaciones técnicas
con fixtures. No levanta el bloqueo de aprendizaje ni acredita una ejecución
sobre el corpus histórico completo.

La sesión delega lo que depende del consumidor en un enlace cerrado. `TitansBinding`
conserva el recorrido de Titans descrito aquí y `CandidateBinding` conecta la
[GRU episódica](gru-financial-sessions.md) con su codec 128×256, banco tensorial y
cursores sin estado neural. Fases, prefijos, pendientes y publicación son comunes.

`FrozenFinancialConsumer` exige parámetros y buffers sin gradiente y todos los
submódulos en `eval`. Su identidad incluye configuración, bytes de parámetros,
código, precisión, dispositivo y opciones numéricas. Rechaza hooks y métodos
sustituidos. Las huellas completas se comprueban en fronteras de sesión y
recuperación. Dentro de cada bloque se revisan versiones, formas y modos sin
copiar los parámetros a CPU. El contador de versiones no detecta cualquier
cambio mediante `.data`, por lo que no sustituye la verificación de bytes.
La identidad serializa también los atributos numéricos de las capas usadas,
como epsilon y forma de LayerNorm, cabezas y disposición de atención, orden de
normalización y activación. Las direcciones de objetos solo intervienen en la
firma interna del proceso. Cambiar esos atributos o sustituir la activación
callable invalida la sesión antes de publicar otra generación.

Antes de construir este consumidor se declara
`torch.backends.mha.set_fastpath_enabled(False)`. El ajuste debe permanecer
durante toda la ejecución y se registra en la identidad recuperable. El
constructor rechaza `True` y no modifica el flag. Los controles, incluido el
baseline de cualquier factorial nuevo, usan el mismo ajuste. La ruta del
Transformer independiente conserva su API y configuración anteriores.

En PyTorch 2.14.0+cu130 se observó una diferencia CPU/CUDA de 2,50e-6 en la
primera emisión FP64 con la ruta fusionada activa. Las entradas y los pesos
eran iguales. La diferencia apareció en el bloque fusionado, mientras que las
etapas manuales de atención y FFN coincidieron hasta 1,11e-15. Con la ruta
desactivada la emisión difirió 1,25e-16. Los recibos del fallo se conservan y las
tolerancias no se ampliaron. Cuando `local_control` existe, también en modo
`disabled`, el contexto SDPA Math de `FinancialPredictor.prepare` abarca el
codificador de precios y MAC. El baseline correspondiente usa ese mismo contexto.
El diagnóstico C sigue midiendo la transición rápida de MAC anterior al lector.

Cada nueva observación produce un token multimodal y una preparación del núcleo.
El lector utiliza `working_state` después de MAC y la cabeza existente. K = 1, 2
y 4 modifica las lecturas, no el número de escrituras asociativas. La instantánea
completa del banco se prepara una vez por evento, copia sus tensores al dispositivo
y conserva IDs crecientes.
C sigue midiendo la transición rápida anterior al lector. Su selección se fija
sobre el grupo lógico antes de dividirlo en bloques.

M0 exige `readout=None` o `mode="no_bank"` y no admite episodios. `None` conserva
la predicción del núcleo. `no_bank` ejecuta el mismo refinador con lectura cero.
M1 admite los resultados maduros y aplica la retención configurada.
[M2](mature-error-write-policy.md) usa un reservorio, un índice selectivo por error
maduro de la emisión y un índice reciente, con un único presupuesto 50/25/25.
Sus pruebas de composición incluyen CPU y una comprobación CUDA focal en
FP32/FP64, K=1, B_mem=4 y C apagado. [M3](m3-write-policy.md) usa los mismos
índices con una puntuación de error, anomalía y relevancia. La sesión calcula
anomalía y relevancia al madurar con los inputs retenidos de cada decisión.
El factor de retención M permanece separado de estas reglas y está desactivado
en M2 y M3.

La sesión utiliza `memory_contract="causal_v2"`. El reservorio inicializa
`mt19937_64` mediante `seed_seq` con las dos palabras de 32 bits de la semilla
explícita. Las identidades de corpus, codec, brazo, partición y fold se guardan
para comprobar la recuperación, pero no intervienen en el RNG. Con la misma
semilla, política, capacidad y secuencia de episodios, los controles emparejados
realizan los mismos sorteos. La retención uniforme conserva su `PCG64` propio y
la política anclada ordena sus candidatos por semilla, contador e ID numérico.
Los IDs numéricos siguen el orden de admisión madura y los desempates usan esos
IDs, sin derivarlos de un hash del corpus.

El contrato nativo v1 conserva su inicialización histórica, que mezcla el ámbito
y la representación con la semilla. Esa representación puede incluir hashes de
datos futuros. Por tanto, las comprobaciones anteriores de v1 no acreditan
invariancia frente a cambios del sufijo del corpus. V1 y v2 tienen archivos
incompatibles entre sí. Recuperar una sesión Python histórica exige su runtime
fijado, incluidas las fuentes y el binario originales. No se ha medido una mejora
predictiva ni se han ejecutado experimentos científicos con este cambio.

## Observaciones y resoluciones

`FinancialPhase` declara calentamiento, intervalo de decisiones, partición y
cierre. El calentamiento avanza el estado rápido sin emitir ni crear pendientes.
Una decisión conserva la emisión aunque falte su etiqueta. Un `settlement`
resuelve pendientes sin ejecutar el núcleo ni el lector.

Cada fase nueva empieza sin episodios ni flujos confirmados. Cada flujo arranca
mediante `initial_state` del predictor. El calentamiento no incorpora etiquetas
al banco y la API no traslada estado entre particiones de forma implícita.

`PrefixTargetVerifier` acredita las exclusiones usando precios y factor efectivos
del corpus, el calendario del manifiesto materializado original y sus hashes.
Cuenta los pares disponibles de las últimas 252 sesiones y conserva el mínimo
de 126 y el umbral `numpy.var(x) <= numpy.finfo(float).eps`. No calcula alpha,
beta ni el objetivo siguiente. Una sustitución del factor necesita otra edición
de origen o una revisión identificada. La evidencia se vuelve a comprobar contra
los registros, no contra una bandera del llamante.

El predictor no recibe esa evidencia ni `reason`. La exclusión se aplica después
de emitir toda la cohorte. Los pendientes sin resolución causal permanecen en
la cola. El cierre administrativo se ejecuta solo en la fecha predeclarada y
no produce label, error financiero, episodio ni `known_at`. No hay expiraciones
por antigüedad.

`CorpusDataset.observation_batches` reutiliza la decodificación de modalidades
para leer todas las observaciones históricas del intervalo sin seleccionar por
labels. `prepare_observation_index` ordena posiciones y fechas mediante el mismo
ordenador externo del corpus. El índice no duplica las modalidades ni calcula
etiquetas. `FinancialObservationSource` recupera los inputs por posición y separa
las maduraciones. `run_observation_source` reúne las observaciones y resultados
del mismo instante antes de llamar una vez a la sesión. Si solo hay resultados,
genera un `settlement`. La especificación de entrada debe vincular su
`view_sha256` al índice. Las fuentes US/CN conservan sus instantes de calendario.

El índice conserva las particiones de supervisión existentes. No convierte
calibración en selección. La etiqueta financiera y su purga siguen procediendo
de la edición supervisada, mientras que el rol de una observación se deriva de
las fechas declaradas. Recuperar un cursor evita volver a decodificar las
modalidades del prefijo confirmado.

## Persistencia y límites

Los estados rápidos se guardan en bloques CPU sin grafo. Cada referencia enlaza
contenido, fila, configuración, parámetros y cursores. Los flujos nuevos necesitan
un `initial_state` explícito. Los ausentes de una cohorte siguen vivos.

La compactación CPU copia filas vivas sin avanzar contadores ni alterar pesos o
momentum. Solo se aplica al superar los cupos declarados de fragmentación. Los
inputs pendientes también pueden compactarse entre fechas. Antes de publicar
se verifican los artefactos propuestos. La poda conserva las referencias anidadas
de las dos generaciones, incluidos estados rápidos e inputs.

El grupo lógico admite 8192 flujos y los bloques físicos hasta 256. La cola
confirmada admite 32.768 pendientes. Las resoluciones de un evento pueden liberar
capacidad antes de comprobar la cola resultante. Los artefactos ocupan como
máximo 2 GiB en 1024 archivos y cada archivo admite hasta 64 MiB. Las copias
transitorias de compactación cuentan contra ese espacio. El registro nativo v2
declara su límite, con 16 GiB por defecto. Los modos anteriores mantienen sus
propios límites.

El índice usa dos hilos, 256 MiB de memoria de DuckDB, hasta 32 GiB de spill y
4 GiB por archivo de posiciones. Los límites de buffers y archivos no equivalen
al RSS completo del proceso. Si una fase no cabe, falla de forma explícita.
La suficiencia de cola y disco para cada fase del censo real sigue siendo una
aceptación separada. Esta implementación no declara materializada ni verificada
una edición histórica completa.

## Comprobaciones técnicas

La [suite CPU](../../reports/engineering/financial-session-quality-20261008.json)
comprende 492 casos con cobertura de ramas y ocho mutaciones Python detectadas.
Parte de esa batería prepara etiquetas de fixtures mediante `prepare_corpus_targets`,
que estima coeficientes residuales. No ejecuta pasos del optimizador del predictor,
pero tampoco se describe como ausencia de todo ajuste auxiliar. Las nuevas
[pruebas de composición](../../reports/engineering/frozen-financial-controls-manual-labels-20261009.json)
usan etiquetas manuales y bloquean llamadas al residualizador.
El [recibo del RNG](../../reports/engineering/financial-session-rng-20261008.json)
separa la recuperación de v1 de los contrafactuales de v2. La
[matriz CUDA](../../reports/engineering/financial-session-cuda-20261008.json)
comprueba 18 configuraciones con parámetros compartidos congelados, paridad
CPU/GPU y recuperación exacta. La guarda posterior de atributos numéricos tiene
sus [regresiones propias](../../reports/engineering/financial-execution-attributes-20261008.json).
La [revisión independiente](../../reports/engineering/financial-session-review-20261008.json)
separa el ciclo nativo, el consumidor y los cambios de ejecución numérica.
El [recibo de cierre técnico](../../reports/engineering/financial-session-completion-20261008.json)
añade la sonda CUDA focal posterior de atributos y los 77 contratos de datos
comprobados al conciliar la rama con `develop`. Esa sonda conserva recuperación
exacta y rechaza una mutación de epsilon sin publicar.

Un recorrido técnico con 32 activos sintéticos y 128 observaciones produjo
64 emisiones, 32 resoluciones maduras y 32 cierres administrativos. Incluyó
lectura, MAC, lector, retención y publicación. Con dos hilos CPU y FP32, las
tres repeticiones tardaron entre 2,43 y 2,50 segundos con bloques de 8 filas y
entre 1,79 y 1,82 segundos con bloques de 32. El pico del proceso fue de
694.005.760 bytes. El fixture usa dimensiones pequeñas de modalidades y no
permite extrapolar ese caudal al corpus real.

La [cota de registro](../../reports/engineering/financial-session-storage-bound-20261008.json)
utiliza los IDs y recuentos del censo de ventanas de precios. Para 17.076.024
emisiones y como máximo una resolución por emisión, reserva 15.128.713.996 bytes,
incluidas cabeceras de hasta 100.000 eventos. Es una cota de serialización por
recorrido y brazo. El pool de artefactos, dos checkpoints protegidos, propuestas
transitorias y el índice se presupuestan aparte. No demuestra que la cola de una
fase real quepa en 32.768 pendientes.
