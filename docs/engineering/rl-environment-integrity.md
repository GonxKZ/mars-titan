# Integridad de los entornos de refuerzo

Un algoritmo de refuerzo optimiza lo que el entorno permite, incluidos sus errores. Antes de levantar el bloqueo de aprendizaje he revisado los entornos y controladores de MARS-TITAN con una pregunta concreta: ¿puede una política, o un arnés descuidado, obtener recompensa sin una señal legítima? Esta auditoría corresponde a la issue #365 y parte de la revisión `48860dab` de `develop`.

El método combina lectura del código, agentes guionizados deterministas, perturbaciones del futuro y mutación dirigida. Ningún agente aprende. Las pruebas nuevas no ejecutan pasos de optimizador ni ajustan políticas, rejillas o normalizadores. El [recibo técnico](../../reports/engineering/rl-environment-integrity-20261009.json) registra cada intento, su resultado, las mutaciones y los límites de la comprobación.

Superar esta batería no demuestra que no exista ninguna trampa. Muestra que los vectores enumerados aquí fallan o están acotados en los entornos actuales, con datos sintéticos y en CPU.

## Entornos inventariados

| Entorno | Observación | Acción | Recompensa | Final | Estado recuperable |
| --- | --- | --- | --- | --- | --- |
| [`CausalPredictionEnv`](causal-prediction-environment.md), con `ParquetCohortSource` y `EpisodePredictionEnv` | Cinco modalidades de la cohorte actual y máscara de activos presentes | Índice de 21 retornos por activo | Media de `−\|a − y\| / s` de los créditos que maduran en ese paso | Termina al consumir la fuente y resolver las etiquetas pendientes. Sin truncamiento | Cursor, reloj, decisiones pendientes sin etiqueta, orígenes, reinicios abandonados y RNG |
| [`FinancialEnv`](persistent-simulation.md) con `MarketTape` y `Portfolio` o `NativePortfolio` | Por activo: predicción, peso, último retorno, volumen, orden pendiente y máscara. Efectivo y derechos de cobro | Seis niveles de exposición | Cambio del logaritmo del patrimonio neto de costes. Ruina con −20 | Ruina termina. Cierre ausente y fin de cinta truncan | Cartera completa, cursor, pausa y RNG |
| [`FinancialBatch` y `FinancialSession`](batched-rl-environments.md) | Las mismas seis características por activo, con contexto fechado opcional | Las mismas seis decisiones por entorno | La misma, con máscara de validez | Igual que la referencia Python | Carteras, cursores e identificadores de contexto |
| [PPO nativo](native-ppo.md), [esquema 2](adaptive-rl.md) y Double DQN | Lote anterior | Muestreo categórico en ajuste y `argmax` en evaluación | GAE por entorno en FP64 | Reinicio solo tras terminación o truncamiento | Pesos, Adam, RNG, rollout parcial, cursores y selección |
| [Colector KLPO terminal](terminal-klpo.md) | Lote anterior | Pasos forzados en calentamiento y muestreados después | Retorno descontado del episodio completo | Ola completa | Prefijo de decisiones reproducible |
| `FinancialTrainer` Python | `FinancialEnv` | PPO o ε-greedy | GAE o objetivo Double DQN | Reinicio tras terminación o truncamiento | Redes, optimizador, replay o rollout, RNG y entorno |

Los entornos se ejecutan siempre en CPU. La GPU solo interviene en la inferencia y el ajuste de las políticas, que esta tarea no modifica.

## Vectores revisados

La columna de estado combina cinco situaciones. «Corregido» indica un defecto confirmado con prueba de regresión en esta rama. «Comprobado» significa que el intento de trampa falla y queda cubierto por una prueba. «Limitación» describe un comportamiento conocido que no permite obtener recompensa en las rutas actuales, pero sesga o restringe el estudio. «Pendiente» depende de datos o predicciones reales. «Fuera del entorno» pertenece a otra capa.

| Vector | Entorno | Estado | Evidencia |
| --- | --- | --- | --- |
| Etiquetas pendientes en `snapshot()` | Predictivo | Corregido | El estado de versión 2 no contiene objetivos. `restore` los relee de la cohorte de origen y comprueba su huella |
| Etiquetas inyectadas o maduración alterada en un checkpoint | Predictivo | Corregido | La versión 1 aceptaba cualquier objetivo finito. Ahora se rechazan campos extra, huellas distintas y maduraciones cambiadas |
| Fuente de validación etiquetada como entrenamiento | Predictivo | Corregido | Los meses de validación walk-forward anteriores a 2023 pasaban el corte fijo. La fuente declara su partición y debe coincidir |
| Reinicio que descarta créditos pendientes | Predictivo | Corregido | El reinicio no emite recompensa y ahora contabiliza episodios abandonados y créditos perdidos en el estado |
| Observación, `info` o render con el futuro | Predictivo | Comprobado | Un agente oráculo no encuentra objetivos en observaciones, `info` ni snapshots. `render` no está implementado |
| Sufijos futuros y etiquetas pendientes alterados | Predictivo y financiero | Comprobado | Observaciones, recompensas, créditos, operaciones y órdenes emitidas no cambian |
| Ejecución en la barra que generó la señal | Financiero | Comprobado | La orden se dimensiona con el cierre de decisión y se ejecuta a la apertura siguiente. Las cantidades no cambian si se alteran el cierre, el volumen o las predicciones de la barra de ejecución |
| Costes nulos o asimétricos | Financiero | Comprobado | El coste por defecto es de 10 pb en compras y ventas. Todas las configuraciones de ajuste usan 10 pb y cero solo aparece en rejillas de evaluación |
| Apalancamiento y cortos | Financiero | Comprobado | Efectivo y posiciones no negativos y exposición acotada por el patrimonio en secuencias aleatorias |
| Acciones NaN, flotantes, booleanas o fuera de rango | Predictivo y financiero | Corregido y comprobado | Fallan antes de la transición. Un entero mayor que int64 lanzaba `OverflowError` y ahora lanza `ValueError` |
| Recompensa por terminar o reiniciar | Financiero | Comprobado | La suma de recompensas coincide con el logaritmo del patrimonio final entre el inicial y un reinicio no arrastra estado |
| Coste de salida no pagado al final | Financiero, Python y nativo | Corregido con métrica declarada | La evaluación Python añade una estimación separada. La selección nativa admite `ruin_count_then_mean_liquidated_log_growth`. La métrica anterior se conserva para las configuraciones existentes |
| Cierre ausente de una posición | Financiero, Python y nativo | Corregido y pendiente | Las fuentes de ajuste con cierres ausentes se rechazan en ambos entrenadores. La evaluación lo marca como incompleto y la validación nativa falla. Las bajas reales necesitan su retorno de salida |
| Acción corporativa antes de la primera decisión | Financiero | Corregido | Python la ignoraba y la sesión nativa la aplicaba. La cinta la rechaza |
| Predicciones dentro de muestra en cintas históricas | Financiero | Corregido y pendiente | Una cinta real exige el fin de ajuste de cada sesión, anterior a su decisión |
| Orden de activos e índice del lote | Todos | Comprobado | Los activos se ordenan por identificador. Las permutaciones conservan resultados y el índice del lote no entra en la observación |
| RNG compartido entre ajuste y evaluación | Python y nativo | Corregido | Los entornos no usan RNG y la evaluación es `argmax`. El entrenador Python deriva su flujo de `SeedSequence(seed).spawn` y ya no repite el del generador del mundo |
| Solapamiento de episodios de ajuste y validación | Cintas sintéticas | Comprobado por lectura | Las semillas, huellas y particiones separadas se exigen en los ejecutores. `generate_world` solo usa la semilla |
| Supervivencia | Real | Sesgo medido y pendiente | De 4.202 activos US con precios solo 2 terminan antes de diciembre de 2023, y los 810 chinos llegan al 29-12-2023. La población está formada casi solo por supervivientes |
| Selección con episodios fallidos | Python y nativo | Corregido | Nativo falla ante una validación incompleta y sus medias quedan como NaN. La campaña Python publica `failed` y cuenta las valoraciones incompletas |
| Pasos forzados de calentamiento | PPO, DQN y KLPO | Comprobado por lectura | PPO y DQN enmascaran su recompensa. KLPO la suma al retorno terminal, pero desde efectivo vale cero |
| Conocimiento posterior en codificadores congelados | Observación | Fuera del entorno | Un codificador preentrenado después de la decisión puede contener información futura. El entorno no puede detectarlo |
| Activos omitidos por capacidad | Cohortes y cintas | Corregido | La sesión más poblada tiene 4.200 activos US. Cohortes, cintas y carteras Python admiten 8.192 con presupuesto de memoria explícito |
| Reglas del mercado chino ausentes | Financiero, Python | Corregido parcialmente | Lotes, resto impar, mínimo de STAR, bandas diarias y timbre por fechas, con [fuentes primarias](china-market-rules.md). ST y salidas a bolsa siguen pendientes |

## Correcciones

### Etiquetas selladas en el entorno predictivo

La versión 1 de `snapshot()` exportaba las decisiones pendientes con su objetivo. El checkpoint de una ejecución contenía, por tanto, etiquetas que todavía no habían madurado. Además, `restore` aceptaba cualquier objetivo finito, de modo que un checkpoint modificado podía cambiar recompensas futuras sin tocar la fuente.

El esquema 2 guarda acción, activo, instante, maduración y posición de origen. Añade la huella de cada cohorte de origen todavía referenciada. Al restaurar se vuelve a leer esa cohorte, se comprueba su huella y se recuperan los objetivos. Los créditos maduros siguen apareciendo en `info["matured"]` con las mismas claves. La trayectoria de un entorno con acciones válidas no cambia, como comprueba la comparación con la revisión base.

La huella de cohorte incluye los bytes del objetivo. En teoría una búsqueda exhaustiva podría comprobar valores candidatos si los objetivos fueran discretos y las entradas conocidas. Con entradas de alta entropía y objetivos continuos ese canal no es práctico. Aun así, el snapshot es un artefacto del arnés y no debe entregarse a la política.

### Partición declarada por la fuente

`CausalPredictionEnv` clasifica las cohortes con los cortes fijos de 2023. La edición histórica usa un walk-forward cuya primera validación empieza el 1 de diciembre de 2022. Una fuente de esa validación pasaba como entrenamiento porque sus fechas y maduraciones eran anteriores a 2023. Ahora una fuente que declara `partition` debe coincidir con el entorno. `ParquetCohortSource`, `EpisodeView` y `SyntheticWorld` la declaran.

El entorno sigue sin poder recorrer esa validación de diciembre, porque sus cortes fijos la rechazan. Es un fallo cerrado, no una fuga. Para usarla habrá que tomar los límites de la supervisión mediante `ordered_bounds`, con una identidad nueva del entorno.

### Reinicios contabilizados

Un reinicio con decisiones tomadas descarta sus créditos pendientes. Gymnasium necesita poder reiniciar en cualquier momento, así que el entorno no lo impide. Registra en el estado los episodios abandonados y los créditos perdidos. Un informe que oculte esos reinicios queda en evidencia al comparar el contador con los episodios publicados.

### Procedencia de las predicciones históricas

Cuando una cinta sintética usa un padre entrenado, lo aplica a mundos ficticios que nunca vio, así que sus puntuaciones están fuera de muestra. Una cinta real de entrenamiento podría, en cambio, llevar las predicciones del modelo final ajustado con toda la partición. La política aprendería a confiar en un ajuste que no existía en cada fecha.

Una cinta de dominio real exige ahora `audit["prediction_fit_ends"]`, con un entero por sesión. Cada valor indica la última maduración de etiqueta usada para ajustar el modelo que produjo esas puntuaciones y no puede ser posterior al instante de la predicción. Este contrato admite el walk-forward expansivo y rechaza las predicciones dentro de muestra. El dato es una declaración del productor, como el resto de la auditoría de procedencia. Debe derivarse de los recibos de cada ventana walk-forward y no escribirse a mano.

### Acciones corporativas en la primera apertura

El episodio empieza en el primer cierre. La sesión nativa aplicaba en el arranque las acciones con fecha en la primera apertura y la referencia Python no. Con una baja en esa fecha, Python podía comprar un activo que la sesión nativa ya había retirado. `MarketTape` rechaza ahora esas acciones. Ningún productor actual las genera. La sesión nativa conserva su comportamiento y debería adoptar la misma regla cuando se revise su lector.

### Coste de salida informado

La recompensa valora la cartera al cierre sin pagar su liquidación. Terminar invertido evita el coste de venta, hasta 10 pb del patrimonio con exposición completa. El retorno telescópico no tiene otra bonificación terminal, pero esa asimetría favorece a las políticas que acaban invertidas. La evaluación Python añade `terminal_liquidation`, con el coste estimado al último cierre y el retorno neto correspondiente. No cambia recompensas, retornos publicados ni identidades. Desde la cuarta fase esa venta paga también el timbre chino en ambos motores.

## Paridad e identidades

Las trayectorias válidas no cambian. Con un mundo sintético de seis activos y sesenta sesiones, la revisión base y esta rama producen la misma huella de cinta, las mismas observaciones, recompensas e `info` en los motores Python y nativo, las mismas métricas de las tres referencias fijas y la misma trayectoria predictiva. El recibo conserva las huellas comparadas.

Cambian el esquema del snapshot predictivo, la admisión de cintas reales y de acciones en la primera apertura, el tipo de error de una acción fuera de int64 y el informe de evaluación, que gana un campo. El entorno predictivo no tiene resultados previos y no hay cintas reales construidas. Las huellas de código de `environment.py`, `market.py` y `evaluation.py` forman parte de las identidades de `FinancialTrainer` y de la campaña Python. Una ejecución pausada con el código anterior no puede reanudarse con el nuevo, como exige su contrato. En la primera fase el código nativo no cambió.

## Segunda fase

Los puntos abiertos de la primera revisión tienen ahora una corrección o una política explícita. Cada cambio de comportamiento tiene identidad propia y las trayectorias válidas anteriores se conservan, salvo las huellas que dependen del código modificado. El [recibo de la segunda fase](../../reports/engineering/rl-environment-integrity-2-20261009.json) registra medidas, pruebas, mutaciones y paridad.

**Capacidad medida.** Con DuckDB, en lectura sobre los `prices.parquet` de la población preparada, la sesión más poblada tiene 4.200 activos US (6 de noviembre de 2023). Con ventana completa de 64 sesiones son 4.195. En entrenamiento, hasta 2022, el máximo es 4.189. China llega a 810 activos. Hay 4.202 activos US distintos entre 2000 y 2023. `MAX_COHORT_ASSETS` pasa de 4.096 a 8.192 en cohortes, rejillas de acciones y huellas del padre. Con las formas de la edición histórica cada activo ocupa 6.725 bytes, 28 MB la sesión más poblada y 55 MB el máximo, dentro de los 64 MiB por defecto. Las cintas y carteras Python admiten 8.192 activos y 2.097.152 celdas, 96 MiB de precios y predicciones en el máximo, y un año de los 4.202 activos cabe en una cinta. La biblioteca C++ conserva 4.096 activos y el motor nativo rechaza explícitamente una cinta mayor.

**Evaluación nativa incompleta.** `evaluate_policy` ya no devuelve un cero con apariencia de válido. Las medias de una evaluación pausada, vacía o incompleta quedan como NaN y la agregación por bloques omite los incompletos.

**Selección con venta final.** Las sesiones publican un retorno liquidado y la evaluación una media liquidada. La métrica declarada `ruin_count_then_mean_liquidated_log_growth` selecciona con ella. Las configuraciones existentes conservan la métrica anterior y producen los mismos registros. Una campaña nueva debe declarar cuál usa.

**Valoraciones incompletas en la campaña Python.** Cada evaluación toma su estado de `financial_validation.completed` y la campaña cuenta las incompletas.

**Flujos aleatorios del entrenador Python.** La exploración, el replay y el barajado proceden de un hijo de `SeedSequence(seed)`, con el esquema `seed_sequence_spawn_v1` en la identidad.

**Cierres ausentes.** La política es rechazar en origen las fuentes de ajuste con algún cierre ausente, en `FinancialTrainer` y en `load_ppo_input`. Una transición censurada no puede llegar al objetivo y la exposición no puede usarse para esquivar una pérdida. Las aperturas ausentes siguen siendo órdenes pendientes. La evaluación y la validación ya trataban el caso como incompleto. Con datos reales esta política impedirá entrenar mientras las bajas y suspensiones no lleguen como acciones acreditadas.

**Reglas chinas.** La [revisión de reglas](china-market-rules.md) recoge fuentes, artículos y vigencias. La simulación Python aplica lotes, resto impar, mínimo de STAR, bandas por tablero y fecha con redondeo por la mitad hacia arriba sobre el precio de referencia exderecho, y el timbre por fechas. T+1 se cumple por construcción. ST, salidas a bolsa, ampliaciones y topes por orden quedan documentados como pendientes.

**Paridad de la segunda fase.** Frente a `d03229b2`, un mundo sintético de 6 activos y 60 sesiones produce las mismas observaciones, recompensas, `info`, instantáneas contables y métricas de referencia en los motores Python y nativo, y la misma trayectoria predictiva. Cambian las huellas de cinta y de entorno, porque `SyntheticWorld` incluye en su identidad la huella de `environments/cohorts.py`, que contiene el contrato de capacidad. También cambian las identidades de `FinancialTrainer`, de la campaña Python y del PPO nativo, que dependen de su código. Una ejecución pausada con el código anterior no se reanuda con este.

## Tercera fase: cinta real reconstruida y cortes walk-forward

La [edición de precios negociados reconstruidos](../data/unadjusted-prices.md) de #379 no satisface por sí sola `MarketTape`. Esta fase la conecta con los entornos sin relajar el contrato anterior y lleva el entorno predictivo a las ventanas walk-forward de 2000. Nada se entrena ni se evalúa. Las pruebas avanzan los entornos con acciones fijas o aleatorias declaradas y no ejecutan pasos de optimizador. El [recibo de la tercera fase](../../reports/engineering/rl-environment-integrity-3-20261009.json) conserva medidas, intentos, mutaciones y paridad.

### Cinta desde la edición reconstruida

`simulation/reconstructed_tape.py` construye una cinta de un mercado con `build_reconstructed_tape(edition, windows, predictions, market=..., partition=..., dividend_payment_lag_sessions=...)`. Antes de leer un activo comprueba la identidad del manifiesto (`edition_id` recalculada), la base `unadjusted_reconstructed`, el corte de 2023 y la huella de cada Parquet, que se interpreta desde los mismos bytes que se han comprobado. Las reglas son estas:

- **Filas verificadas.** Un activo entra solo si todas sus filas dentro de la cinta están verificadas, tiene un cierre negociado verificado no posterior a la primera sesión, sin filas dudosas entre ambos, y su serie no termina dentro de la cinta. Si no, se excluye con uno de estos motivos: `no_verified_rows`, `unverified_rows_in_tape`, `row_outside_calendar`, `no_verified_traded_close_at_start`, `series_ends_in_tape`, `event_outside_calendar` o `invalid_event`. Ninguna fila se rellena con precios inventados. La última sesión sin fila de una serie que continúa después se trata como cualquier otra sesión sin negociación ([cuarta fase](#sesión-final-sin-fila)).
- **Sesiones sin negociación.** Una fila con volumen cero y una sesión del calendario sin fila no tienen apertura ejecutable. El volumen queda en cero o ausente y el cierre conserva el último cierre negociado verificado. El cierre que el proveedor mueve en algunas suspensiones se ignora.
- **Rejilla de cotización.** Los precios que la edición sitúa en la rejilla se llevan a su múltiplo exacto (0,01, 1/256 o 0,0001 según mercado, fecha y precio) con la tolerancia relativa de la edición, 2e-6. Sin ese paso, en la prueba de la edición sintética, una apertura guardada como 10,9999945 sobre un cierre anterior guardado como 10,000004 quedaba por debajo del límite de 11,00 y la simulación compraba 900 acciones en una sesión bloqueada al alza. Una apertura fuera de la rejilla no es ejecutable.
- **Acciones corporativas.** Splits y dividendos proceden de los eventos del proveedor posteriores a la primera sesión. El dividendo usa el importe por acción negociada de la edición. Cuando split y dividendo coinciden en la misma sesión, la edición no permite saber si ese importe se refiere a la acción anterior o a la posterior al split. Se abona el menor de los dos importes y esa apertura no es ejecutable, de modo que ni el pago ni el precio de referencia del límite diario pueden aprovecharse. El dividendo se aplica antes que el split, como en el precio de referencia de `Portfolio`.
- **Pago de dividendos.** La edición no tiene fechas de pago. `dividend_payment_lag_sessions` es un supuesto obligatorio y declarado. El cobro llega en la apertura de esa sesión posterior, o queda como derecho pendiente si cae fuera de la cinta.
- **Calendario.** Las aperturas son las oficiales de XNYS o XSHG (`MarketClock.opens`) y las decisiones son las del proyecto, cinco minutos después del cierre. Así las predicciones emitidas en la decisión cumplen `prediction_times <= close_times`.
- **Predicciones.** Cada ventana aporta sus predicciones, que deben coincidir con la huella del recibo y caer en una decisión de su tramo. Las de activos excluidos se descartan y se cuentan.

`MarketTape` admite esta base solo con el tratamiento fijo `RECONSTRUCTED_CONTRACT` en su auditoría. Una cinta que declare acciones completas, retornos de salida, otra valoración o otro mercado se rechaza. Además exige un cierre valorado en cada sesión y activo, deriva `prediction_fit_ends` de los tramos walk-forward declarados y rechaza sesiones fuera de esos tramos o tramos que alcancen 2024. `FinancialEnv` rechaza una cinta china reconstruida sin las reglas de acciones A de `market_rules`. Al cerrar esta fase el motor nativo no las aplicaba. Ahora lo hace con paridad exacta, como recoge la [revisión de reglas chinas](china-market-rules.md#motor-nativo).

| Garantía de `MarketTape` | Cinta reconstruida | Cómo se cumple o se declara |
| --- | --- | --- |
| Precios negociados | Parcial | Precios reconstruidos y verificados por tramos, no capturas de la época. Con precios altos la rejilla apenas discrimina |
| Acciones corporativas completas | No | `corporate_actions_complete=false`. Escisiones registradas como splits fraccionarios o ausentes. Las ampliaciones chinas no publicadas suelen dejar sin verificar las filas anteriores. Si su fecha cae dentro de un tramo verificado, la caída exderecho aparece sin el derecho que la compensa, lo que penaliza mantener la posición pero no permite ganar |
| Retornos de salida | No | `exit_returns="unavailable"` y `population="listed_through_2025_03"`. Ningún activo puede quedar sin valorar, así que una serie que termina dentro de la cinta se excluye |
| Calendario de aperturas | Sí | Calendarios de `exchange_calendars` y decisión del proyecto |
| Predicciones fuera de muestra | Sí, según el recibo | El último dato de ajuste de cada sesión es `labels_used_until` de su ventana, anterior a la evaluación |
| Ninguna fila de 2024 | Sí | Tramos acotados por el test reservado y rechazo de filas posteriores al corte de la edición |
| Sin ejecución en suspensiones | Sí | Apertura ausente y volumen cero o ausente en la sesión |
| Elegibilidad sin información futura | No | La verificación de cada fila usa una constante ajustada al final de 2023 y los tramos posteriores. Exigir que la serie continúe después de la cinta condiciona además el universo a sobrevivir al tramo |

### Recibo de ventana walk-forward

`environments/walk_forward_receipt.py` define el contrato que leerán los entornos. El orquestador de la [campaña con máscaras](../research/training-campaign-2000.md#ejecución-y-recuperación) escribe un recibo por brazo, semilla, ventana y mercado con las funciones de este módulo. Los entornos no dependen de su código.

| Campo | Contenido | Comprobación |
| --- | --- | --- |
| `kind`, `schema_version` | `walk_forward_window_receipt`, 1 | Exactos |
| `protocol` | Configuración completa del protocolo | Versión 2, purga por intervalo y test reservado desde 2024-01-01. `build_folds` la valida |
| `fold` | Ventana con sus cuatro tramos | Debe ser una de las ventanas que produce el protocolo. Unos límites escritos a mano no se aceptan |
| `parent` | `id` y `sha256` del predictor ajustado en la ventana | Identificador acotado y huella hexadecimal |
| `labels_used_until` | Última maduración de etiqueta usada en ajuste, selección y calibración, en microsegundos UTC | Anterior al inicio de la evaluación |
| `predictions` | Por tramo, `rows` y `sha256` de las predicciones emitidas | Huella canónica de `prediction_fingerprint` |

La huella ordena por instante y activo, rechaza claves repetidas y puntuaciones no finitas, y resume los instantes `int64`, los identificadores y las puntuaciones `float64`. No depende del formato en que el productor guarde las predicciones. Las predicciones de validación o calibración de un predictor seleccionado con esas etiquetas no pueden alimentar una cinta, porque su sesión sería anterior a `labels_used_until`.

`CausalPredictionEnv(..., window=...)` toma los cortes del recibo y admite los cuatro tramos. Cada decisión debe caer dentro del tramo y cada etiqueta debe madurar antes de su final. Si la fuente declara límites por mercado deben coincidir. La ventana entra en la identidad, así que un estado de otra ventana no se restaura. Sin `window`, el entorno conserva sus cortes fijos y su identidad.

### Costes y ejecución con datos reconstruidos

| Elemento | Situación | Fuente o supuesto |
| --- | --- | --- |
| Comisión y diferencial | `cost_bps`, 10 pb por defecto en compras y ventas, configurable y en la identidad del entorno | Supuesto sin fuente. No varía por mercado ni por fecha |
| Deslizamiento e impacto | Sin modelo propio más allá de `cost_bps` y del límite de participación | Supuesto declarado |
| Límite de volumen | 1 % del volumen de la sesión de decisión, en acciones negociadas (`V/K` en la edición) | Supuesto configurable. Una decisión tomada en una sesión sin volumen no tiene capacidad |
| Suspensiones y filas ausentes | Sin ejecución ni precio nuevo | Edición y calendario |
| Lotes y resto impar | 100 acciones en el tablero principal y ChiNext, mínimo de 200 en STAR | [Reglas chinas](china-market-rules.md), obligatorias en cintas chinas reconstruidas |
| Límites diarios | Banda por tablero y fecha sobre precios en la rejilla, con referencia exderecho | Reglas chinas. Sin estado ST ni días posteriores a la salida a bolsa |
| T+1 | Una ejecución por sesión, en la apertura siguiente a la decisión | Por construcción |
| Impuesto de timbre | Por fechas | Reglas chinas |
| Retenciones sobre dividendos | No se modelan. Se abona el bruto | Regla china de 2015 revisada, con su efecto declarado en las [reglas chinas](china-market-rules.md#lo-que-falta). EE. UU. depende del tipo de inversor y no se ha revisado |
| EE. UU. | Sin límites diarios ni interrupciones intradía | No representables con barras diarias |

### Medidas sobre la edición real

Todas las lecturas fueron de solo lectura, con dos hilos y sin GPU. Un primer recorrido de los 5.012 activos (55 s) contó en filas verificadas 315.342 sesiones US y 97.556 chinas con volumen cero, de las que 36.050 y 627 tienen un cierre distinto del anterior. Dentro de los tramos verificados faltan 57.188 sesiones US y 4.868 chinas del calendario, y no hay filas fuera del calendario. Split y dividendo coinciden en 105 eventos US de 65 activos y en 1.286 eventos chinos de 498 activos. Excluir los activos afectados habría retirado entre 31 y 116 casos por año en las cintas chinas medidas y habría condicionado el universo de la cinta a un evento posterior a su inicio. Por eso se usa la regla del menor importe sin ejecución.

Después construí cintas completas con puntuaciones sintéticas constantes, que no proceden de ningún modelo:

| Cinta | Activos | Sin filas verificadas | Filas sin verificar | Sin cierre al empezar | Sin última sesión | Sesiones sin volumen | Filas ausentes | Aperturas fuera de rejilla | Eventos ambiguos | Tiempo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| US 2023 | 3.581 de 4.202 | 547 | 70 | 1 | 3 | 4.610 | 216 | 3.067 | 0 | 36,6 s |
| CN 2023 | 799 de 810 | 8 | 3 | 0 | 0 | 96 | 11 | 0 | 31 | 7,8 s |
| US 2008 | 2.026 de 4.202 | 547 | 24 | 1.595 | 10 | 14.571 | 2.861 | 49 | 4 | 42,6 s |
| CN 2011 | 532 de 810 | 8 | 69 | 200 | 1 | 6.160 | 289 | 0 | 116 | 9,0 s |

La memoria residente máxima fue de 379 MB. En 2008 y 2011 la mayoría de los activos sin cierre al empezar todavía no cotizaba. Los tres activos US de 2023 sin última sesión son GEM, IBDO y PREF, cuya serie termina entre el 21 y el 26 de diciembre. Son exactamente los casos en los que haría falta un retorno de salida.

### Pruebas, mutación y paridad

- `tests/simulation/test_reconstructed_tape.py` usa una edición sintética con el formato real: filas verificadas y sin verificar, volumen cero con cierre movido, filas ausentes, splits, dividendos, evento en la primera sesión, split y dividendo el mismo día, apertura fuera de la rejilla y aperturas en los límites diarios con el residuo de la reconstrucción. Comprueba exclusiones, identidad, ausencia de 2024, causalidad frente a filas y predicciones futuras, integridad de la edición y las huellas, y políticas guionizadas que intentan operar en suspensiones, filas ausentes, aperturas fuera de rejilla, sesiones ambiguas y límites diarios, o capturar un dividendo. La variante nativa coincide con la de Python en observaciones, recompensas e `info` sobre una cinta US reconstruida.
- `tests/environments/test_walk_forward_receipt.py` y `test_prediction_walk_forward.py` cubren el contrato del recibo, la huella y los cuatro tramos del entorno predictivo.
- `tests/simulation/test_reconstructed_tape_edition.py` es una prueba de humo con la edición real, cinco activos por mercado, 2023 y acciones fijas. Solo se ejecuta con `MARS_TITAN_UNADJUSTED_EDITION`.
- Mutación dirigida: 29 mutantes sobre la cinta, el contrato, el recibo, el entorno predictivo y la exigencia de reglas chinas. Todos fallan alguna prueba. El que acepta un protocolo v1 sobrevivía porque el caso de prueba también fallaba por campos ausentes, así que añadí una prueba con el protocolo v1 versionado.
- Paridad frente a `b02bcb9a`: la cinta sintética, las trayectorias Python y nativa, la identidad del entorno, una trayectoria con reglas chinas y la trayectoria y el estado del entorno predictivo sin ventana producen las mismas huellas.

### Lo que sigue sin cubrir

- La elegibilidad depende de la verificación retrospectiva y de la presencia al final de la cinta. Es un sesgo de selección declarado, no una fuga de precios, pero un resultado sobre estas cintas está condicionado a ello.
- Retornos de salida, estado ST, días posteriores a la salida a bolsa, ampliaciones de capital, retenciones y fechas reales de pago siguen sin datos.
- La regla del menor importe subestima el cobro cuando el importe de la edición se refiere a la acción posterior al split. En 600239.SS, en junio de 2015, 0,046154 por 1,3 da 0,06, lo que sugiere ese caso. No lo he contrastado con el anuncio de la empresa.
- No hay ruta CUDA en estos entornos. Las reglas chinas en el motor nativo se añadieron después, con su [paridad](china-market-rules.md#motor-nativo).
- No existen todavía recibos reales. El orquestador de la campaña con máscaras debe escribirlos con este contrato.

```bash
MARS_TITAN_UNADJUSTED_EDITION=~/.local/state/mars-titan/unadjusted-prices-20261009/edition-v1 \
  CUDA_VISIBLE_DEVICES=-1 uv run pytest tests/simulation/test_reconstructed_tape_edition.py
```

## Cuarta fase: realismo de la etapa de políticas e informe financiero

Esta fase (issues #137 y #365, rama `fix/rl-realism-reporting`) cierra los huecos de realismo y de informe que dejó la auditoría de la etapa de políticas por ventana. Nada se entrena ni se evalúa científicamente. Las pruebas usan fixtures sintéticos identificados, la etapa reducida con ejecutores guionizados y, solo como prueba de humo de lectura, la edición real de precios. Ninguna ejecuta pasos de optimizador.

### Sesión final sin fila

La regla anterior excluía un activo sin fila en la última sesión de la cinta (`missing_last_session`). Como el universo se fija con ajuste y validación, esa exclusión anulaba la evaluación de la ventana para todos los predictores. DVN no tiene fila el 31 de diciembre de 2009 y vuelve a negociar el 4 de enero de 2010, así que la ventana US `fold-004` quedaba sin evaluación por una sola fila.

La regla nueva, `missing_row_valued_at_last_traded_close_when_series_continues_v1`, entra en la identidad de la cinta (`source.final_session`). Si la serie tiene alguna fila posterior a la cinta, la última sesión sin fila se trata como cualquier otra sesión sin negociación: sin apertura ejecutable, sin volumen y valorada con el último cierre negociado verificado. No se crea ningún precio ni ninguna ejecución, y el recuento `final_sessions_without_row` la declara. Si la serie termina dentro de la cinta, el activo se excluye con `series_ends_in_tape`, porque haría falta un retorno de salida. Saber que la serie continúa usa información posterior para la elegibilidad, igual que la regla anterior, y queda declarado en la tabla de garantías.

En la edición real, entre 0 y 16 activos US por ventana (144 casos en 19 ventanas) y entre 0 y 3 chinos (9 en 13) tienen la última sesión sin fila y la serie continúa. Solo tres series admisibles terminan dentro de una evaluación, todas en EE. UU.: EAI en 2017, WSTL en 2020 e IBDO en 2023. Las pruebas cubren un caso sintético, en el que la sesión C++ coincide paso a paso con Python, y la ventana real de DVN, que ya se construye y da el mismo patrimonio en los motores Python y nativo.

Ninguna sesión sin fila, sea la última o una ausencia marcada por máscara en la edición, crea un precio, una ejecución o una recompensa. Una prueba con dos sesiones sin fila de ningún activo comprueba en los dos motores que la cinta deja apertura, máximo, mínimo y volumen ausentes y repite el último cierre negociado, que las órdenes de esas sesiones quedan sin ejecutar con `missing_open`, sin coste, que la recompensa de esos pasos es exactamente cero y que las órdenes no ejecutadas no se arrastran a la primera apertura real. Una mutación que inventa la apertura de esas sesiones con el último cierre falla la prueba.

### Liquidación final con timbre

La venta hipotética al último cierre paga `cost_bps` sobre el valor de la cartera y el impuesto de venta vigente en la fecha de ese cierre, en `evaluate` y en `liquidated_nav` de la sesión C++. Ambos motores suman igual, primero el coste sobre el total y después el timbre, y coinciden en la prueba que cruza el cambio de tipo de 2023. La base se llama ahora `final_close_minus_cost_bps_and_sell_taxes`.

### Límite de etiquetas derivado de lo leído

`labels_used_until` ya no es el microsegundo anterior a la evaluación. El orquestador lo deriva con `training/label_maturity.py`: es la mayor maduración (`target_available_at`) de las etiquetas aceptadas de ajuste, validación y calibración de la vista en la que se fijó el predictor elegido (la propia ventana o, en un traslado, su ancla), y de las de calibración de la propia ventana. Cada archivo de etiquetas se comprueba con su huella y se lee por columnas.

Una prueba adversaria adelanta en una copia de las vistas la maduración de una etiqueta de ajuste hasta el inicio de la evaluación y recalcula todas las huellas. El lector del corpus la rechaza por sí mismo. Con un ejecutor que no validase sus etiquetas, la campaña no publica el recibo de esa ventana (`read_window_receipt` rechaza el límite) y, sin recibo, no hay cinta. Si un recibo o una cinta se manipulasen después, `MarketTape` y el lector C++ ya rechazan un corte de ajuste posterior a la decisión. Tres mutaciones dirigidas (omitir el ajuste, volver a la constante y omitir la calibración de un traslado) fallan las pruebas. La etapa de adaptadores y la GRU candidata siguen escribiendo la constante y deberían adoptar la misma derivación.

### Referencias sin predicción y costes declarados

La versión 2 de `historical-masked-rl-policies.json` declara antes de evaluar cinco referencias, los costes de evaluación de 0, 5, 10 y 20 pb, la sección `report` del informe y la sensibilidad `survival_sensitivity`. Los binarios aceptan esos costes con `--evaluation-cost` y lo declaran con la capacidad `native_policy_equity_and_costs`. La etapa exige esa capacidad y comprueba que la evaluación usó exactamente los costes declarados.

- `equal_weight_monthly` reparte la exposición completa por igual entre todos los activos valorados de la cinta (`allocation="valid_assets_equal_weight"`) y la reequilibra cada 21 sesiones, con los mismos costes, participación, lotes, bandas y timbre que las demás políticas. No usa predicciones: su patrimonio es idéntico con cualquier predictor sobre la misma cinta, como comprueban las pruebas.
- `market_index` compra en la primera apertura y mantiene SPY en una cinta propia de un activo, construida con el mismo recibo y el mismo calendario que la evaluación. Los dividendos se cobran con la contabilidad del motor. En las 19 ventanas US reales, SPY se admite siempre, cobra al menos tres dividendos por año y los motores Python y nativo coinciden.
- China no tiene en la edición ningún fondo que replique el CSI 300. Su `market_index` no se ejecuta en el motor y el informe lo calcula con niveles diarios del índice (`simulation/index_benchmark.py`). Compra en la primera apertura con nivel posterior a la decisión, paga `cost_bps` al entrar y al salir y no cobra dividendos, porque es un índice de precios. Los niveles se leen solo después de evaluar y se identifican por su huella. Los previstos son la apertura y el cierre del [factor CSI 300](../data/csi300-market-factor.md), tomados de los boletines mensuales de SSE, que declara `point_in_time_verified=false` y `financial_simulation_ready=false`. Por eso esta referencia solo valora una compra hipotética del índice, no simula una operación ejecutable y no entra en ninguna decisión. Si no se pasan los niveles, el informe lo declara en `missing_benchmarks` y forma las familias chinas sin esa referencia. Una ventana sin niveles suficientes queda fuera de su familia con el motivo `index_levels_unavailable`.

Los límites de la etapa siguen al nuevo número de referencias: 1.368 ajustes y 1.221 evaluaciones en A, y 456 ajustes con 2.133 evaluaciones en B, con 10.356 episodios en cada variante. El diseño conjunto de la campaña A (#363) volverá a cambiar estos recuentos.

### Patrimonio por sesión e informe financiero

Cada episodio de evaluación publica su patrimonio a cada cierre (`equity`). La etapa exige que empiece en el capital, que no tenga deuda, que solo el último valor pueda ser cero en una ruina y que el patrimonio final concilie con el retorno publicado con una tolerancia relativa de 1e-9. El motor nativo lo reconstruye con sus recompensas logarítmicas.

`simulation/stage_report.py` (`run_masked_campaign.py rl-report`) agrega una o varias salidas de la etapa. Lee `stage.json`, `summary.json` y cada recibo confirmado, y exige que las métricas del resumen salgan de esos recibos. Forma familias pareadas por ámbito, mercado, predictor y coste, con KLPO y todos sus controles disponibles, y solo usa las ventanas en que todos los brazos terminaron con los mismos cierres. Las reglas, declaradas antes de ver resultados, son estas:

- Cada ventana termina con su liquidación y las ventanas se encadenan. Si la serie encadenada se arruina, sus métricas terminan en la ruina.
- Un brazo con tres semillas reparte el capital entre ellas. Sus métricas por semilla se publican aparte.
- Las métricas de `evaluation/financial_metrics.py` son rentabilidad acumulada y anualizada (también en porcentaje), volatilidad, Sharpe y Sortino con tipo sin riesgo cero, drawdown máximo, giro y costes sobre el capital. Se calculan en `evaluation/financial_conventions.py`, el mismo módulo que usa la [cartera larga y corta](../research/long-short-portfolio.md) de la comparación predictiva (#32, `evaluation/long_short.py`), así que una misma serie da las mismas cifras en los dos informes. Cada familia se anualiza con las sesiones de su mercado, 252 en US y 243 en CN. Las dos remuestrean con los mismos índices de una semilla y usan `family_intervals` para los intervalos simultáneos. La cartera da intervalos también para el drawdown, que necesita recorrer cada réplica en orden. Este informe publica el drawdown solo como estimación puntual, porque ese recorrido multiplicaría su coste.
- La incertidumbre sale del bootstrap circular por bloques de sesiones, con bloque de 21, sensibilidad de 5 y 63, 2.000 réplicas, semilla 20261009 y confianza del 95 %. Todas las series de una familia usan las mismas réplicas.
- Los contrastes son KLPO menos cada control, con intervalos simultáneos por máximo estudentizado dentro de cada familia y métrica. No se corrige entre predictores, costes ni métricas. El coste principal declarado es 10 pb.

El informe escribe `report.json`, `metrics.csv`, `contrasts.csv` y `equity.parquet`. Las pruebas lo recorren sobre la etapa reducida y comprueban que el retorno encadenado coincide con el producto de los retornos liquidados de cada ventana, que el patrimonio de KLPO es la media de sus semillas, que el contraste es KLPO menos control y que un resumen o una serie manipulados se rechazan. Seis mutaciones dirigidas fallan las pruebas.

### Supervivencia

El sesgo principal está en la población, no en las cintas. La edición contiene empresas que seguían cotizando en marzo de 2025 (`population="listed_through_2025_03"`). La auditoría privada de bajas, con las listas de bajas de Shanghái y Shenzhen y la serie de empresas cotizadas del Banco Mundial, cuenta 101 bajas de acciones A en Shanghái y 129 en Shenzhen entre 2000 y 2023, de cuyos códigos solo 7 aparecen en el conjunto de datos. En 2004 el Banco Mundial registra 5.226 empresas nacionales cotizadas en EE. UU. y la edición tiene 1.650 activos. Cualquier resultado de la etapa está condicionado a la supervivencia hasta 2025 y probablemente sobrestima la rentabilidad de comprar y mantener, de la cartera 1/N y de las políticas. Ninguna sensibilidad sobre las cintas puede corregirlo, porque los activos dados de baja no están en los datos.

Dentro de las cintas, una serie que termina en la evaluación deja la ventana sin evaluar para todos los brazos. La sensibilidad declarada, con retornos de salida de 0, −30 % y −100 %, es secundaria y el informe lista en `survival` las ventanas afectadas. En la edición actual no habrá ninguna en la campaña A: las tres series que terminan dentro de una evaluación ocupan los puestos 3.101, 3.418 y 1.700 por efectivo negociado mediano en su validación, y el universo admite 128 activos con cortes de entre 233 y 453 millones de dólares diarios. Por eso no se ha llevado a los motores la baja con retorno de salida, que exigiría cambiar el contrato de la cinta reconstruida en Python y en C++. Si una edición con bajas o un universo afectado lo necesitan, el informe lo marca como `secondary_evaluation_pending`. Una evaluación que excluye todo su universo se registra ahora como fallo de la ventana en lugar de detener la etapa. Cinco mutaciones dirigidas (listar cualquier motivo de exclusión, aceptar cualquier tipo de fallo, marcar siempre la sensibilidad como pendiente, volver a detener la etapa y perder los motivos de la exclusión) fallan las pruebas. Las dos primeras sobrevivieron a las pruebas sobre la etapa reducida y motivaron una prueba unitaria del filtro.

### Pendientes declarados

- **Inicios aleatorios de episodio.** No se implementan. Cambiarían la distribución de ajuste de los tres algoritmos, la definición de oleada completa de KLPO, el estado recuperable de ambos motores y las pruebas de paridad, y se solaparían con el trabajo de rendimiento de la recogida nativa. Cada episodio sigue recorriendo su cinta anual desde el principio, con el riesgo de memorización ya declarado.
- **Paciencia.** La selección guarda el mejor estado en validación con mejora mínima de 0,0001 y presupuesto fijo, sin parada temprana. Parar cada brazo por su cuenta rompería la igualdad de presupuesto emparejada, porque KLPO consume oleadas completas y PPO y Double DQN un número fijo de transiciones, y una parada conjunta exigiría coordinar trabajos que se ejecutan por separado. La paciencia declarada se conserva en la configuración y no se usa para parar.
- **Semillas del predictor.** Las cintas usan las predicciones de la semilla 42 de cada predictor. La campaña A repite con 43 y 44 solo el caso seleccionado, de modo que la variabilidad por semilla del predictor se mide en la comparación predictiva y la etapa de políticas mide la de sus tres semillas propias.
- **Reglas chinas.** Las que no tienen fuente con fechas quedan como simplificaciones con su efecto esperado en la [revisión de reglas](china-market-rules.md#lo-que-falta).

## Quinta fase: solo datos reales, ventanas de ajuste y predictor de la cadena

El usuario fijó dos condiciones para la etapa de políticas de la campaña A: aprender solo con datos reales del conjunto, con un corte walk-forward de toda la serie, y no usar nada sintético ni en los entornos de refuerzo ni en los postentrenamientos. El diseño por etapas (#437) añade que cada cinta lleve las predicciones del predictor de la cadena de su ventana. Esta fase lo hace comprobable sin entrenar ni ejecutar pasos de optimizador.

### Solo cintas reales de la edición declarada

- **Política de datos en el plan.** `historical-masked-rl-policies.json` declara `data = {policy: real_edition_only, edition: unadjusted_price_edition, edition_id}` con la identidad de la edición verificada. `policy_plan` rechaza otra política, otro tipo de edición, una identidad que no sea una huella o un campo añadido.
- **Edición antes de leer nada.** `run_stage` recalcula `edition_id` desde el manifiesto de la edición y lo compara con el declarado antes de abrir la campaña base o crear la salida.
- **Cada cinta antes de un ejecutor.** `window_tapes.require_real_tape` exige dominio `real`, origen `unadjusted_edition_tape`, base reconstruida y la misma identidad de edición. Se aplica a las cintas de ajuste, validación y evaluación, a las del índice, a las de admisión del universo y a las que ya estaban en disco al reanudar. Una cinta sintética escrita en la carpeta de una cinta confirmada detiene la etapa sin llamar a ningún ejecutor.
- **Ninguna ruta hacia los mundos sintéticos.** `test_rl_real_data_only.py` recorre el grafo de importaciones del proyecto desde `campaign_stage` y `stage_report`, también las importaciones dentro de funciones. Comprueba que no aparecen `mars_titan.episodes`, `adaptation_scenarios` ni `posttraining.preparation`, ni los nombres de sus generadores fuera de `market.py`, que define `MarketTape` junto a `from_world`. Otra prueba ejecuta `run_masked_campaign.py rl check` en un proceso aparte y comprueba que no carga ninguno de esos módulos ni la etapa de posentrenamiento, de la que solo lee el contrato de la cadena (`posttraining/staged_chain.py`). Para eso el despachador de la orden importa cada subcomando solo cuando se pide. Una tercera recorre la etapa reducida con `MarketTape.from_world` sustituido por un error y registra el dominio de cada cinta creada: todas son reales.
- **Predicciones compactadas por la retención.** Las puntuaciones de las cintas se leen con el lector de la comparación, que resuelve con `data/prediction_files.py` los archivos que la retención v2 compacta o libera. Una prueba comprueba con tres formatos de escritor que el archivo original y el compactado dan a las cintas las mismas puntuaciones bit a bit, y que uno liberado detiene la etapa con `PredictionsReleased` hasta que se regenere.
- **Sonda de reglas chinas sin cinta.** La capacidad `native_cn_a_share_rules` se comprobaba con una cinta mínima de precios inventados. Ahora `probe_cn_rules` pregunta a la biblioteca nativa las bandas de cada tablero y periodo de `PRICE_LIMITS` para cinco precios de referencia y las compara con las de Python, sin construir ninguna cinta. Una biblioteca con otros límites pierde la capacidad.

El código sintético histórico (mundos de episodios y `scripts/generate_episode_worlds.py`, escenarios de adaptación, la preparación del postentrenamiento emparejado y `MarketTape.from_world`) se conserva por trazabilidad de los experimentos anteriores. Queda excluido de la etapa por construcción, no por convención.

### Ventanas de ajuste: regla fija y sensibilidad en expansión

La regla `expanding_prior_evaluations_v1` ajusta la política de la ventana k con las cintas de evaluación de todas las ventanas anteriores a k−1, desde la primera del protocolo. Fue la regla principal hasta el 10 de octubre de 2026 y ahora es una sensibilidad secundaria, declarada en `window_sensitivity` con su propia identidad y desactivada. La regla principal es `fixed_prior_evaluations_v1`, con las tres evaluaciones más recientes. El motivo es el universo. Exigir admisión en todas las cintas de ajuste deja en la expansión unos 1.580 candidatos de EE. UU. en todas las ventanas mientras la población de los predictores crece de 2.205 a 4.192 activos, y 533 chinos frente a 810. El universo de 2023 solo comparte 92 de 128 activos con el de la regla fija en EE. UU. y 85 en China. Las cifras por ventana y mercado, medidas con la población real de los predictores, están en la [campaña](../research/training-campaign-2000.md#etapa-de-políticas-por-ventana) y en el [informe de medida](../../reports/engineering/rl-universe-rules-20261010.json).

El máximo de la sensibilidad sale del motor. KLPO asigna a cada entorno una cinta fija (`lane % n`) y rechaza más cintas que entornos. PPO y Double DQN, en el esquema 4, admiten hasta 512 fuentes y rotan por ellas al terminar cada episodio. Las tres familias deben ver las mismas cintas, así que `policy_plan` rechaza un máximo superior al presupuesto de entornos y la sensibilidad declara 16. Con la regla fija, KLPO da seis entornos a la cinta más antigua y cinco a las otras dos.

La sensibilidad no se puede lanzar mientras esté desactivada. `run` y `rl-report` solo la aceptan con `--sensitivity`, y la identidad de su salida incluye la sensibilidad, de modo que la etapa principal y su informe rechazan esa salida y al revés. `rl check` informa de las cintas por predictor de las dos reglas: 75 en EE. UU. y 45 en China con la fija, y 179 y 81 con la expansión.

La admisión con todos los activos de cada tramo se guarda por ventana y recibo. Así cada ventana se monta una vez con unos 4.200 activos en EE. UU. aunque varias anclas la lean.

### Predictor de la cadena

`predictor.source` elige de dónde salen las predicciones de las cintas. Con `posttraining_chain_v1`, la regla de la campaña A, `chain_source` lee por ventana `windows/<ámbito>/<ventana>/<brazo>__chain/seed-<s>/selection.json`, que el posentrenamiento escribe la última. Sin ella la ventana no está confirmada y no hay cinta. La selección se lee con `read_selection` de `posttraining/staged_chain.py`, el mismo lector con el que la etapa de adaptadores confirma lo que escribe, así que las dos etapas comparten un único contrato. Ese lector exige:

- Los campos exactos del documento, con tipo, versión, regla `chain_validation_score_v1`, ámbito, ventana, brazo y semilla iguales a los pedidos.
- En la ventana sin padre, el estado elegido de la base sin padre, candidatos ni filas nuevas. En las demás, el candidato que da la regla entre el padre congelado y los demás casos, de modo que el padre solo se sustituye con una mejora estricta.
- La huella de cada recibo de mercado igual a la que fija `markets`, con `parent` igual al trabajo elegido y la huella de su recibo, y el mismo `labels_used_until` que la selección.

`chain_source` añade lo que ese lector no puede comprobar sin la campaña base ni las vistas:

- La huella del recibo confirmado del trabajo elegido, que se lee de la campaña base en la ventana 0 y de la salida del posentrenamiento en las demás, igual a la de la selección, y las filas y la huella de sus predicciones de evaluación iguales a las del recibo del mercado.
- `labels_used_until` igual a la mayor maduración de las etiquetas de ajuste, validación y calibración de la vista de la ventana y de la anterior, recalculada con `training/label_maturity.py`.

La regla `base_campaign_selected_v1` conserva la lectura de la campaña base. Ahora también recalcula la maduración de las etiquetas que leyó el predictor elegido y la compara por igualdad con la del recibo. Cada trabajo declara en `depends` las selecciones de la cadena de todas las cintas que lee, también las del predictor del universo. En un traslado, el primer requisito sigue siendo el ajuste de su ancla. Los identificadores (`<ámbito>/<ventana>/<brazo>__chain/select-s<semilla>`) salen de `chain_job_id` de `posttraining/staged_chain.py`, sin otra definición en la etapa de políticas.

Antes de crear la salida y de llamar a ningún ejecutor, `run_stage` lee con `read_selection` todas las selecciones que necesita el plan (`predictor_reads` de `simulation/policy_plan.py`, la misma lista que forma `depends`) y, si falta alguna, se detiene con un único error que las enumera todas. Sin esa comprobación, una selección ausente solo aparecería al montar la cinta de su trabajo, quizá con otros trabajos ya confirmados. La lectura de cada cinta mantiene su propia comprobación. Cada predictor que leen las políticas de A (las cinco redes, Ridge, XGBoost y los cuatro brazos de Titans-MAC) tiene su cadena en la [etapa de adaptadores](masked-posttraining.md#etapa-por-ventana-de-la-campaña), y una prueba comprueba que cada dependencia de la etapa de políticas es una selección planificada allí. Ridge y XGBoost usan la [cadena trivial](masked-posttraining.md#cadena-trivial-de-ridge-xgboost-y-b6), que solo tiene el padre congelado. La etapa de políticas de B también declara la cadena, pero la etapa de adaptadores de B no se ejecuta y no la publica, así que B se detendría en esta comprobación. Con la declaración ampliada de la campaña, las políticas leerían además la cadena de la GRU candidata, de MARS-TITAN y de CM-v1, que la etapa de adaptadores aún no planifica.

Las fuentes de `campaign_source` y `chain_source` devuelven el recibo y un lector diferido de las predicciones. `_Tapes` solo lo llama al montar una cinta o un universo que no están en disco. Así la etapa se reanuda sobre sus cintas confirmadas aunque la [retención v2](../research/training-campaign-2000.md#retención-v2-ventana-a-ventana) haya liberado después las filas, lo que esta solo hace cuando cada trabajo de políticas que las lee tiene su recibo. Una prueba completa la etapa reducida, libera las evaluaciones elegidas de la base y la vuelve a recorrer sin llamar a ningún ejecutor y con las mismas métricas.

Catorce manipulaciones de una cadena publicada con el contrato detienen la etapa antes de llamar a un ejecutor: otra ventana, otra semilla, otro tipo de documento, un estado `base` con padre, un estado desconocido, un padre congelado que empata con el candidato elegido, otra huella del recibo de mercado, otra huella del recibo del trabajo elegido, la huella del punto de control como padre, otras predicciones y cuatro límites de etiquetas (anterior o posterior en recibo y selección a la vez, o distinto solo en el recibo o solo en la selección). Una ventana sin `selection.json` tampoco se acepta, y la comprobación previa enumera todas las que faltan.

### Pruebas y mutación de la quinta fase

`test_rl_real_data_only.py` (4 pruebas), `test_window_tapes.py` (20), `test_policy_plan.py` (99) y `test_campaign_stage.py` (60) cubren estas reglas sin aprender. La mutación dirigida aplicó de uno en uno 37 defectos en una copia aislada del código: regla y máximo de la ventana, tope de entornos, política de datos, predictor declarado, cada condición de `require_real_tape`, su llamada en las cintas de disco y del universo, identidad de la edición, maduración en la campaña base, caché de admisión, sonda china, despachador perezoso, cada comprobación de la lectura de la cadena y las dependencias del plan. En la primera pasada sobrevivieron tres: un máximo no entero, una cinta real reetiquetada solo como sintética y la cinta de admisión del universo sin comprobar. Se añadieron sus pruebas y los 37 fallan ahora. Dos mutantes se consideran equivalentes y no se cuentan. Quitar el recibo de la clave de la caché de admisión no cambia nada en una ejecución, donde cada ventana tiene un solo recibo del predictor del universo. Leer solo la vista de la ventana en la maduración de la cadena tampoco, porque las vistas están anidadas y la de la ventana anterior nunca madura después. Al pasar a la regla fija se aplicaron otros 12 a la sensibilidad (activación sin comprobar, identidad y informe sin la sensibilidad, regla igual a la principal, papel, lanzamiento, tipo de la activación, identificador, tope de entornos, la etapa derivada con las ventanas principales y dos errores en el recuento de cintas). Los 12 fallaron en la primera pasada. Al pasar la lectura de la selección a `read_selection` se repitieron seis sobre lo que añade `chain_source`: la huella del recibo del trabajo, las huellas de evaluación, la maduración recalculada, la selección ausente, la raíz de la ventana 0 y la maduración de una sola vista. Los cinco primeros fallan y el último es el mutante equivalente ya descrito.

## Hallazgos sobre la población preparada

**Supervivencia.** De los 4.202 activos US con precios, solo 2 terminan antes de diciembre de 2023 (el primero el 13 de marzo de 2017). Los 810 activos chinos tienen datos hasta el 29 de diciembre de 2023. La población crece de 1.343 activos US en 2000 a 4.200 en 2023 sin bajas apreciables. Es una población de empresas que existían al final del periodo. Cualquier resultado sobre ella, predictivo o financiero, hereda un sesgo de supervivencia que los entornos no pueden corregir. Hace falta una lista histórica de cotizadas y bajas, con retornos de salida, o declarar el resultado como condicionado a la supervivencia hasta 2023. La cuarta fase lo cuantifica con fuentes públicas y declara el resultado de la etapa como condicionado ([supervivencia](#supervivencia)).

**Base de precios.** Con una tolerancia de 0,05 céntimos, el 84 % de los cierres chinos y el 69 % de los estadounidenses no caen en un múltiplo de céntimo, y en 2023 siguen fuera el 75 % y el 62 %. Es coherente con precios ajustados por acciones corporativas posteriores. La cinta real exige precios sin ajustar, así que estos Parquet no la satisfacen.

## Limitaciones que siguen abiertas

**Semilla del mundo sin partición.** Con la misma semilla, `generate_world` produce la misma trayectoria de precios en entrenamiento y validación, desplazada en el tiempo. Los ejecutores rechazan esa coincidencia, pero la función no. Incluir la partición en la semilla cambiaría todos los mundos y requiere otra versión del generador.

**Validación optimista.** Las cifras de validación del mejor checkpoint son las mismas que lo seleccionaron. Solo la auditoría separada del esquema 2 estima su rendimiento sin ese sesgo.

**Confianza en el arnés.** Un checkpoint coherente pero inventado, con más efectivo, pasa las comprobaciones de la cartera. `FinancialBatch::step_checked` permite descartar un paso tras ver su resultado. Los ejecutores actuales no lo hacen, pero la garantía depende de ellos. Las trazas de decisiones confirmadas permiten auditar una ejecución reproduciendo sus acciones.

**Memorización de cintas.** Cada episodio recorre la misma cinta desde el principio. Repetir una cinta no crea trayectorias independientes y una política puede memorizarla. Solo las fuentes de validación separadas lo controlan.

**Diferencias entre simulación y mercado.** Los mundos de adaptación abren al cierre anterior, sin salto nocturno. No hay deslizamiento ni impacto de precio más allá del límite del 1 % del volumen de decisión. El coste en puntos básicos no varía por mercado ni por fecha. Las reglas chinas existen en ambos motores, con los pendientes de la [revisión de reglas](china-market-rules.md#lo-que-falta).

**Dependencias de capacidad fuera de estos módulos.** La preparación del corpus causal (`corpus_source._partition`) sigue rechazando cohortes de más de 4.096 filas y debe usar `MAX_COHORT_ASSETS`. Los padres del postentrenamiento y los lotes de las referencias aceptan 4.096 filas por llamada. Si un padre no es independiente por fila, trocear una cohorte cambiaría sus predicciones. El modo clásico del banco episódico mantiene 4.096 activos. Esos módulos pertenecen a otras tareas.

**Predicciones dentro de muestra en el postentrenamiento.** El [postentrenamiento predictivo](../research/predictive-adaptation.md) usa predicciones del padre sobre su propia partición de entrenamiento. Está documentado y queda fuera de estos entornos.

## Comprobaciones que esperan datos reales

| Comprobación | Motivo | Propuesta o estado |
| --- | --- | --- |
| Precios sin ajustar, splits, dividendos y bajas | Una cinta con precios ajustados y splits declarados crearía saltos falsos de patrimonio | La tercera fase construye la cinta desde la edición reconstruida, solo con filas verificadas y acciones del proveedor declaradas como incompletas. Las bajas siguen sin datos |
| Retornos de salida de activos dados de baja | Sin ellos, las bajas aparecen como cierres ausentes y las fuentes de ajuste se rechazan | Pendiente. La cinta reconstruida lo declara y excluye las series que terminan dentro de ella |
| Calendario de aperturas | La ejecución depende de `open_times` reales | Cubierto en la tercera fase con las aperturas oficiales de XNYS y XSHG |
| Predicciones fuera de muestra | La cinta real exige `prediction_fit_ends` | Contrato del recibo definido en la tercera fase. Desde la cuarta, el orquestador deriva `labels_used_until` de las etiquetas que leyó el predictor, y desde la quinta la etapa lo recalcula y lee el predictor de la cadena con su selección confirmada. Faltan los recibos reales, porque ni la campaña ni el posentrenamiento se han ejecutado |
| Supervivencia | La población preparada apenas contiene bajas | Los resultados se declaran condicionados a seguir cotizando en 2025 y la [cuarta fase](#supervivencia) mide el alcance. Corregirlo exige una población con bajas y retornos de salida |
| Estado ST y salidas a bolsa | Cambian la banda diaria de un activo | Incorporar el historial de advertencias de riesgo y fechas de admisión con su fuente |
| Codificadores congelados | Su preentrenamiento puede ser posterior a la decisión | Registrar la fecha de corte de cada codificador y contrastar con la modalidad enmascarada |

## Comprobaciones CUDA

Ningún entorno se ejecuta en GPU y esta rama no modifica código CUDA ni políticas. Las pruebas CUDA existentes de las políticas ejecutan Adam y siguen bloqueadas. Queda pendiente, con la GPU libre y sin pasos de optimizador, comprobar que la evaluación `argmax` de una política congelada en `cuda:0` elige las mismas acciones que en CPU sobre las cintas de estas pruebas y publica el mismo patrimonio por sesión con los costes declarados. Esa prueba no existe todavía y debe escribirse sin llamar a `optimizer.step()`, con una política de pesos iniciales evaluada por `--audit-run` con `--device cpu` y con `--device cuda:0` sobre los binarios del preset `native-ppo-release`, que activa LibTorch CUDA:

```bash
cd native
cmake --preset native-ppo-release
cmake --build --preset native-ppo-release --target mars-titan-ppo mars-titan-klpo mars_titan_simulation
```

Las referencias, el informe y la cinta del índice no usan la GPU.

## Reproducción

La biblioteca de simulación se compila desde `native` y las pruebas se ejecutan desde la raíz:

```bash
cd native
cmake --preset native-release -DMARS_TITAN_BUILD_RUNNER=OFF
cmake --build --preset native-release --target mars_titan_simulation
cd ..
CUDA_VISIBLE_DEVICES=-1 uv run pytest tests/environments/test_prediction_adversarial.py \
  tests/simulation/test_financial_adversarial.py tests/environments/test_cohort_capacity.py \
  tests/simulation/test_tape_capacity.py tests/simulation/test_trainer_streams.py \
  tests/simulation/test_china_market_rules.py
```

Las pruebas nativas de la segunda fase se compilan con el preset de PPO sin CUDA y no ejecutan Adam:

```bash
cd native
cmake --preset native-ppo-release -DMARS_TITAN_LIBTORCH_ENABLE_CUDA=OFF -DMARS_TITAN_BUILD_FINANCIAL=ON
cmake --build --preset native-ppo-release --target ppo_evaluation_tests ppo_inputs_tests
cd ../build/native/native-ppo-release
CUDA_VISIBLE_DEVICES=-1 ctest -R '^(ppo_evaluation|ppo_inputs|financial_session|financial_batch)$'
```

Ambos archivos usan fixtures sintéticos identificados como tales. No cargan pesos ni ejecutan optimizadores.

Las pruebas de la cuarta fase usan la biblioteca y los binarios del mismo preset para la paridad nativa:

```bash
MARS_TITAN_NATIVE_LIBRARY=build/native/native-ppo-release/libmars_titan_simulation.so \
MARS_TITAN_SIM_EXECUTABLE=build/native/native-ppo-release/mars-titan-sim \
CUDA_VISIBLE_DEVICES=-1 uv run pytest tests/simulation/test_reconstructed_tape.py \
  tests/simulation/test_reference_portfolios.py tests/simulation/test_stage_report.py \
  tests/simulation/test_policy_plan.py tests/simulation/test_campaign_stage.py \
  tests/simulation/test_window_tapes.py tests/simulation/test_rl_real_data_only.py \
  tests/evaluation/test_financial_metrics.py tests/training/test_masked_campaign.py
```
