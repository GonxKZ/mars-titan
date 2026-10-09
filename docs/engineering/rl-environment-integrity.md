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

La recompensa valora la cartera al cierre sin pagar su liquidación. Terminar invertido evita el coste de venta, hasta 10 pb del patrimonio con exposición completa. El retorno telescópico no tiene otra bonificación terminal, pero esa asimetría favorece a las políticas que acaban invertidas. La evaluación Python añade `terminal_liquidation`, con el coste estimado al último cierre y el retorno neto correspondiente. No cambia recompensas, retornos publicados ni identidades.

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

## Hallazgos sobre la población preparada

**Supervivencia.** De los 4.202 activos US con precios, solo 2 terminan antes de diciembre de 2023 (el primero el 13 de marzo de 2017). Los 810 activos chinos tienen datos hasta el 29 de diciembre de 2023. La población crece de 1.343 activos US en 2000 a 4.200 en 2023 sin bajas apreciables. Es una población de empresas que existían al final del periodo. Cualquier resultado sobre ella, predictivo o financiero, hereda un sesgo de supervivencia que los entornos no pueden corregir. Hace falta una lista histórica de cotizadas y bajas, con retornos de salida, o declarar el resultado como condicionado a la supervivencia hasta 2023.

**Base de precios.** Con una tolerancia de 0,05 céntimos, el 84 % de los cierres chinos y el 69 % de los estadounidenses no caen en un múltiplo de céntimo, y en 2023 siguen fuera el 75 % y el 62 %. Es coherente con precios ajustados por acciones corporativas posteriores. La cinta real exige precios sin ajustar, así que estos Parquet no la satisfacen.

## Limitaciones que siguen abiertas

**Semilla del mundo sin partición.** Con la misma semilla, `generate_world` produce la misma trayectoria de precios en entrenamiento y validación, desplazada en el tiempo. Los ejecutores rechazan esa coincidencia, pero la función no. Incluir la partición en la semilla cambiaría todos los mundos y requiere otra versión del generador.

**Validación optimista.** Las cifras de validación del mejor checkpoint son las mismas que lo seleccionaron. Solo la auditoría separada del esquema 2 estima su rendimiento sin ese sesgo.

**Confianza en el arnés.** Un checkpoint coherente pero inventado, con más efectivo, pasa las comprobaciones de la cartera. `FinancialBatch::step_checked` permite descartar un paso tras ver su resultado. Los ejecutores actuales no lo hacen, pero la garantía depende de ellos. Las trazas de decisiones confirmadas permiten auditar una ejecución reproduciendo sus acciones.

**Memorización de cintas.** Cada episodio recorre la misma cinta desde el principio. Repetir una cinta no crea trayectorias independientes y una política puede memorizarla. Solo las fuentes de validación separadas lo controlan.

**Diferencias entre simulación y mercado.** Los mundos de adaptación abren al cierre anterior, sin salto nocturno. No hay deslizamiento ni impacto de precio más allá del límite del 1 % del volumen de decisión. El coste en puntos básicos no varía por mercado ni por fecha. Las reglas chinas solo existen en el motor Python.

**Dependencias de capacidad fuera de estos módulos.** La preparación del corpus causal (`corpus_source._partition`) sigue rechazando cohortes de más de 4.096 filas y debe usar `MAX_COHORT_ASSETS`. Los padres del postentrenamiento y los lotes de las referencias aceptan 4.096 filas por llamada. Si un padre no es independiente por fila, trocear una cohorte cambiaría sus predicciones. El modo clásico del banco episódico mantiene 4.096 activos. Esos módulos pertenecen a otras tareas.

**Predicciones dentro de muestra en el postentrenamiento.** El [postentrenamiento predictivo](../research/predictive-adaptation.md) usa predicciones del padre sobre su propia partición de entrenamiento. Está documentado y queda fuera de estos entornos.

## Comprobaciones que esperan datos reales

| Comprobación | Motivo | Propuesta |
| --- | --- | --- |
| Precios sin ajustar, splits, dividendos y bajas | Una cinta con precios ajustados y splits declarados crearía saltos falsos de patrimonio | Conciliar precios brutos y ajustados del proveedor con las acciones acreditadas, y rechazar la cinta si un split explica un salto ya ajustado |
| Retornos de salida de activos dados de baja | Sin ellos, las bajas aparecen como cierres ausentes y las fuentes de ajuste se rechazan | Convertir cada baja en una acción acreditada con su valor de recuperación, también cero |
| Calendario de aperturas | La ejecución depende de `open_times` reales | Derivarlo del calendario de cada bolsa y comprobar que cada apertura sigue al cierre de decisión |
| Predicciones fuera de muestra | La cinta real exige `prediction_fit_ends` | Generar las puntuaciones con las ventanas walk-forward y copiar cada corte de sus recibos |
| Supervivencia | La población preparada apenas contiene bajas | Incorporar listas históricas de cotizadas y bajas con retornos de salida, o declarar los resultados como condicionados a sobrevivir hasta 2023 |
| Estado ST y salidas a bolsa | Cambian la banda diaria de un activo | Incorporar el historial de advertencias de riesgo y fechas de admisión con su fuente |
| Codificadores congelados | Su preentrenamiento puede ser posterior a la decisión | Registrar la fecha de corte de cada codificador y contrastar con la modalidad enmascarada |

## Comprobaciones CUDA

Ningún entorno se ejecuta en GPU y esta rama no modifica código CUDA ni políticas. Las pruebas CUDA existentes de las políticas ejecutan Adam y siguen bloqueadas. Queda pendiente, con la GPU libre y sin pasos de optimizador, comprobar que la evaluación `argmax` de una política congelada en `cuda:0` elige las mismas acciones que en CPU sobre las cintas de estas pruebas. Esa prueba no existe todavía y debe escribirse sin llamar a `optimizer.step()`.

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
