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
| Coste de salida no pagado al final | Financiero, Python y nativo | Limitación, informada | La valoración final no liquida. La evaluación Python añade una estimación separada. La selección nativa sigue sin ella |
| Cierre ausente de una posición | Financiero, Python y nativo | Limitación y pendiente | La transición es inválida y se excluye del objetivo. La validación nativa falla si ocurre. Las cintas sintéticas no tienen cierres ausentes |
| Acción corporativa antes de la primera decisión | Financiero | Corregido | Python la ignoraba y la sesión nativa la aplicaba. La cinta la rechaza |
| Predicciones dentro de muestra en cintas históricas | Financiero | Corregido y pendiente | Una cinta real exige el fin de ajuste de cada sesión, anterior a su decisión |
| Orden de activos e índice del lote | Todos | Comprobado | Los activos se ordenan por identificador. Las permutaciones conservan resultados y el índice del lote no entra en la observación |
| RNG compartido entre ajuste y evaluación | Python y nativo | Limitación | Los entornos no usan RNG. La evaluación es `argmax`. El RNG del entrenador Python reproduce el flujo del generador del mundo cuando coinciden las semillas |
| Solapamiento de episodios de ajuste y validación | Cintas sintéticas | Comprobado por lectura | Las semillas, huellas y particiones separadas se exigen en los ejecutores. `generate_world` solo usa la semilla |
| Supervivencia | Real | Pendiente | Depende de que el corpus conserve activos dados de baja y sus retornos de salida |
| Selección con episodios fallidos | Python y nativo | Limitación | Nativo falla ante una validación incompleta. La campaña Python marca `status="completed"` aunque la valoración sea incompleta |
| Pasos forzados de calentamiento | PPO, DQN y KLPO | Comprobado por lectura | PPO y DQN enmascaran su recompensa. KLPO la suma al retorno terminal, pero desde efectivo vale cero |
| Conocimiento posterior en codificadores congelados | Observación | Fuera del entorno | Un codificador preentrenado después de la decisión puede contener información futura. El entorno no puede detectarlo |

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

Cambian el esquema del snapshot predictivo, la admisión de cintas reales y de acciones en la primera apertura, el tipo de error de una acción fuera de int64 y el informe de evaluación, que gana un campo. El entorno predictivo no tiene resultados previos y no hay cintas reales construidas. Las huellas de código de `environment.py`, `market.py` y `evaluation.py` forman parte de las identidades de `FinancialTrainer` y de la campaña Python. Una ejecución pausada con el código anterior no puede reanudarse con el nuevo, como exige su contrato. El código nativo no cambia.

## Limitaciones que siguen abiertas

**Selección nativa sin coste de salida.** `ruin_count_then_mean_log_growth` usa el logaritmo del patrimonio final sin liquidar. La diferencia puede llegar a 0,001 con 10 pb y exposición completa, diez veces el `min_delta` de 0,0001 configurado. Propongo una métrica nueva con valoración liquidada y su propia identidad, sin sustituir la actual en las ejecuciones ya registradas.

**Transiciones censuradas por cierres ausentes.** Python y nativo excluyen del objetivo la transición con un cierre ausente de una posición. Como la validez depende de la exposición elegida, una pérdida previa a una suspensión o baja sin acción acreditada no entra en el ajuste. Las cintas sintéticas actuales no tienen cierres ausentes y la validación nativa falla si aparecen. Con datos reales, cada baja debe llegar como acción acreditada con su retorno de salida, y el informe debe publicar cuántas transiciones se censuran.

**Puntuación cero de una evaluación incompleta.** `evaluate_policy` devuelve `mean_log_growth = 0` cuando hay episodios incompletos. Su único llamador lo rechaza antes de seleccionar, pero un consumidor nuevo podría tomarlo como válido. Conviene devolver un valor ausente con un estado explícito.

**RNG del entrenador Python.** `FinancialTrainer` crea `default_rng(seed)` y `generate_world` usa `default_rng(config.seed)`. Los mundos de entrenamiento y las semillas de los comparadores usan 42, 43 y 44, así que la exploración puede consumir el mismo flujo que generó el mundo. No he encontrado un alineamiento temporal explotable, pero la independencia no está garantizada. La solución es derivar flujos con `SeedSequence.spawn` y una identidad nueva del entrenador.

**Semilla del mundo sin partición.** Con la misma semilla, `generate_world` produce la misma trayectoria de precios en entrenamiento y validación, desplazada en el tiempo. Los ejecutores rechazan esa coincidencia, pero la función no. Incluir la partición en la semilla cambiaría todos los mundos y requiere otra versión del generador.

**Estado de la campaña Python.** `run_campaign` publica `status="completed"` aunque `financial_validation.completed` sea falso. El observatorio exporta este último campo. Los agregados deben contar los episodios incompletos con ese campo y no con el estado del proceso.

**Validación optimista.** Las cifras de validación del mejor checkpoint son las mismas que lo seleccionaron. Solo la auditoría separada del esquema 2 estima su rendimiento sin ese sesgo.

**Confianza en el arnés.** Un checkpoint coherente pero inventado, con más efectivo, pasa las comprobaciones de la cartera. `FinancialBatch::step_checked` permite descartar un paso tras ver su resultado. Los ejecutores actuales no lo hacen, pero la garantía depende de ellos. Las trazas de decisiones confirmadas permiten auditar una ejecución reproduciendo sus acciones.

**Memorización de cintas.** Cada episodio recorre la misma cinta desde el principio. Repetir una cinta no crea trayectorias independientes y una política puede memorizarla. Solo las fuentes de validación separadas lo controlan.

**Diferencias entre simulación y mercado.** Los mundos de adaptación abren al cierre anterior, sin salto nocturno. No hay deslizamiento ni impacto de precio más allá del límite del 1 % del volumen de decisión. El coste es constante en ambos mercados. No se modelan el impuesto de timbre chino en ventas, los lotes de cien acciones, la regla T+1, los límites diarios de precio ni las suspensiones salvo como precio ausente. Estas diferencias no permiten obtener recompensa en los datos sintéticos actuales, pero sobrestiman la ejecutabilidad en China.

**Capacidad.** El censo histórico registra 4.200 activos estadounidenses con archivos de precios durante 2023. El máximo por sesión será igual o menor y todavía no está medido. Las cohortes y las cintas admiten 4.096 activos y `MarketTape` un máximo de 1.048.576 celdas, unas 249 sesiones con 4.200 activos. Superar el límite hace fallar la preparación. No autoriza omitir empresas.

**Predicciones dentro de muestra en el postentrenamiento.** El [postentrenamiento predictivo](../research/predictive-adaptation.md) usa predicciones del padre sobre su propia partición de entrenamiento. Está documentado y queda fuera de estos entornos.

## Comprobaciones que esperan datos reales

| Comprobación | Motivo | Propuesta |
| --- | --- | --- |
| Precios sin ajustar, splits, dividendos y bajas | Una cinta con precios ajustados y splits declarados crearía saltos falsos de patrimonio | Conciliar precios brutos y ajustados del proveedor con las acciones acreditadas, y rechazar la cinta si un split explica un salto ya ajustado |
| Retornos de salida de activos dados de baja | Sin ellos, las bajas aparecen como cierres ausentes y se censuran | Convertir cada baja en una acción acreditada con su valor de recuperación, también cero |
| Calendario de aperturas | La ejecución depende de `open_times` reales | Derivarlo del calendario de cada bolsa y comprobar que cada apertura sigue al cierre de decisión |
| Predicciones fuera de muestra | La cinta real exige `prediction_fit_ends` | Generar las puntuaciones con las ventanas walk-forward y copiar cada corte de sus recibos |
| Supervivencia | Las cohortes deben incluir activos que luego desaparecen | Comparar el censo por instante con listas históricas de constituyentes y bajas |
| Tamaño máximo de cohorte | El censo anual roza el límite de 4.096 | Medir el máximo por `prediction_at` y mercado y, si lo supera, ampliar el límite con medidas de memoria |
| Reglas del mercado chino | T+1, lotes, límites diarios y timbre cambian lo ejecutable | Añadirlas como identidad nueva del simulador con pruebas de contabilidad conocida |
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
  tests/simulation/test_financial_adversarial.py
```

Ambos archivos usan fixtures sintéticos identificados como tales. No cargan pesos ni ejecutan optimizadores.
