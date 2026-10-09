# Adaptación y memoria en escenarios financieros sintéticos

El esquema 2 de `mars-titan-ppo` compara políticas con memoria, recurrencia y contexto de régimen sobre mundos de mecanismo conocido. El predictor de entrada permanece congelado. Este módulo amplía el [ejecutor PPO nativo](native-ppo.md), pero no implementa el candidato MARS-TITAN ni sustituye el objetivo principal de predicción multimodal.

La configuración y las pruebas descritas aquí son técnicas. Los precios, avisos y variables macro son ficticios. El generador declara `analysis_domain=technical`, `real_corpus_compatible=false` y `modality_contract=technical_context_only`. Simula tres conceptos macro de los 140 del catálogo: el tipo estadounidense a dos años, el de diez años y su diferencia. Los otros 137 están ausentes. La dimensión de un tensor no acredita cobertura macro ni cuatro modalidades reales verificadas.

## Mundos y separación de fuentes

[adaptation-scenarios.json](../../configs/simulation/adaptation-scenarios.json) fija 16 activos, 256 sesiones y 64 sesiones iniciales sin operaciones. Cada una de las ocho familias contiene 32 mundos de entrenamiento, 16 de validación y 64 reservados para auditoría. Son 896 mundos con semillas distintas: 256 de entrenamiento, 128 de selección y 512 de auditoría.

| Familia | Mecanismo controlado |
| --- | --- |
| `no_signal` | Las señales observables no aportan un retorno esperado. |
| `known_signal` | La señal disponible al cierre modifica el retorno de la sesión siguiente. |
| `delayed_cue` | Un único aviso durante el calentamiento informa sobre un tramo posterior. La señal deja de estar visible antes de operar. |
| `signal_reversal` | Cambia el signo de la relación entre señal y retorno. |
| `regime_recurrence` | La relación recorre A→B→A dentro del mismo mundo. |
| `source_conflict` | Dos fuentes emiten señales contradictorias con fiabilidades declaradas de 0,9 y 0,1. |
| `volatility_shift` | La desviación del ruido común pasa de 0,002 a 0,009 y la proporción idiosincrásica de 0,25 a 1,5. El segundo régimen añade saltos comunes de ±0,04 con probabilidad 0,02. |
| `execution_friction` | Disminuye el volumen disponible. Un split 2:1 y un dividendo ficticio con pago diferido modifican precios y contabilidad de forma coherente. |

La secuencia de precios se genera en orden temporal. Ampliar el horizonte conserva el prefijo ya generado. La verdad del mecanismo y las etiquetas con su fecha de maduración se guardan en `evaluator/truth.parquet`. La política no lee ese archivo. Los saltos también se registran solo allí, mientras que el contexto recibe sus efectos observados sobre retornos y volatilidad.

Cada mundo tiene `manifest.json`, `market.parquet`, `context.json`, `context.parquet` y `events.parquet`. Los eventos conservan identificador, fuente, ámbito, unidad y disponibilidad. Los manifiestos enlazan los archivos mediante SHA256. Las predicciones del padre analítico son positivas e iguales para todos los activos, sin utilizar la verdad reservada.

Los once campos de contexto son comunes a todas las familias. Incluyen las dos señales, sus fiabilidades, retorno, volatilidad, dispersión, los tres valores macro y `trading_enabled` en la última posición. Antes de la sesión 64 se fuerza la acción de efectivo y se excluye la decisión del aprendizaje. La política recurrente observa esas sesiones, pero no puede usarlas para acumular posiciones u órdenes que almacenen indirectamente el aviso.

El entrenador mantiene 16 entornos activos y rota por el catálogo de entrenamiento cuando termina un episodio. Guarda el índice de fuente de cada entorno y el siguiente índice del catálogo. Esta secuencia de exposición no incorpora mundos de validación o auditoría. El calentamiento suma `observed_transitions`, pero no consume el presupuesto de transiciones de aprendizaje. El lector admite hasta 512 fuentes por partición y comprueba el presupuesto de memoria antes de cargar sus matrices.

## Variantes implementadas

| Variante | Estado o actualización que se contrasta |
| --- | --- |
| `ppo` | Política sin recurrencia con dos capas de 64 unidades. |
| `ppo_window` | Ventana de 16 observaciones. Ajusta el ancho de la MLP para quedar a un 5 % o menos de los parámetros de la GRU comparable. |
| `ppo_gru` | Estado recurrente GRU de 64 unidades y actualización PPO por secuencias de hasta 16 pasos. |
| `ppo_episodic` | Recuperación de hasta cuatro recuerdos maduros mediante claves de proyección fija. |
| `ppo_hmm` | Contexto con probabilidades filtradas de un HMM de dos estados. |
| `ppo_episodic_hmm` | Combina recuperación episódica y filtrado HMM. |
| `ppo_recent_aux` | Añade a la combinación anterior una tarea auxiliar sobre recompensas maduras recientes. |
| `ppo_replay_aux` | Usa la misma tarea auxiliar con selección por reservorio de experiencias maduras. |
| `double_dqn` | Control independiente con replay, red objetivo y acciones ε-greedy. No utiliza el objetivo PPO. |

La memoria episódica conserva hasta 1024 registros por entorno y episodio. Sus claves y valores tienen 64 componentes. La proyección usa la semilla fija 1729. La consulta respeta mundo, partición, fold, representación, entorno y fecha de maduración. Los valores recuperados se combinan por media uniforme, sin interpretar la similitud como confianza financiera. Un cambio de fuente reinicia el ámbito de memoria.

La [caché FP64 de consulta](../../reports/resources/episodic-query-20261004.md) evita convertir toda la matriz antes de cada GEMV. La fila provisional se restaura al salir y los snapshots siguen siendo FP32. El informe cuantifica el aumento de memoria residente, la reducción de asignaciones temporales y los tiempos del recorrido CUDA, con paridad entre versiones.

La GRU conserva el prefijo del episodio y reconstruye sus estados con los parámetros actuales antes de actualizar cada secuencia. Los reinicios separan episodios y el relleno no contribuye a la pérdida. Un estado oculto previo a una actualización no se reutiliza como si perteneciera a los pesos nuevos.

El HMM se ajusta una vez con `hmmlearn==0.3.3`, usando solo las secuencias de entrenamiento y las variables observadas de retorno, volatilidad y dispersión. Exporta prior, transiciones, medias y varianzas para `MarkovFilter` en C++. Su manifiesto identifica todas las fuentes del ajuste. La inferencia utiliza filtrado hacia delante, sin Viterbi ni suavizado con observaciones futuras. El ajuste es offline sobre entrenamiento, como declara `training_context_point_in_time=false`, y no demuestra que esos estados tengan una interpretación económica real.

Las variantes auxiliares conservan hasta 8192 experiencias y comparan selección reciente con reservorio. Su objetivo MAE utiliza la recompensa financiera ya observada, nunca las etiquetas privadas del generador. El optimizador y el RNG auxiliares están separados. El cuerpo compartido puede cambiar con la consolidación, aunque la cabeza del actor no reciba directamente esa pérdida. Las experiencias antiguas no entran en el objetivo PPO recortado.

Double DQN utiliza 4096 experiencias, actualización de la red objetivo cada 256 transiciones y calentamiento de replay de 256 transiciones. ε desciende de 1 a 0,05 durante la primera mitad del presupuesto. La acción siguiente la selecciona la red online y su valor se obtiene de la red objetivo. La traza registra la distribución ε-greedy utilizada, no un softmax inventado de los valores Q.

## Selección, presupuesto y recuperación

La [configuración base](../../configs/simulation/adaptive-ppo.json) declara 262144 transiciones, recorridos de 1024, un trabajador y evaluaciones cada 16384 transiciones de aprendizaje. Siempre evalúa θ₀ y el estado final. Solo una validación completa puede cambiar el mejor checkpoint. El criterio prioriza menos ruinas y, a igualdad de ruinas, mayor crecimiento logarítmico medio neto de costes, con `min_delta` declarado. También registra retorno, caída máxima, costes, rotación y pasos por mundo, ligados al SHA256 de su manifiesto.

La comparación conserva un presupuesto fijo y permite que θ₀ siga siendo el mejor. La parada temprana está desactivada por defecto. Si se activa en otro protocolo, sus presupuestos ya no deben presentarse como una comparación de igual número de actualizaciones. PPO y Double DQN tampoco realizan el mismo número de pasos de optimizador por transición. Ese recuento se conserva por separado.

La segmentación y el relleno de la GRU también pueden cambiar el número de minibatches respecto a la MLP. En la comprobación de 8192 transiciones se observaron 544 actualizaciones recurrentes y 520 de la MLP, con cuatro épocas y minibatches nominales de 64. El presupuesto común controla interacciones, no garantiza idéntico trabajo de optimización. Los controles auxiliares comparan además sus contadores de pasos y ejemplos realmente utilizados.

El [coordinador](../../scripts/run_adaptive_campaign.py) declara un máximo de 168 horas activas del proceso: 12 para el piloto, 108 para la comparación principal, 36 para auxiliares y 12 de reserva. El piloto usa 8192 transiciones por caso y elige un presupuesto común entre 65536, 262144 y 524288 según tiempos observados y un margen declarado. No utiliza puntuaciones de validación para elegir esa duración. Las semillas de política son 42, 43 y 44.

`initial_reserve_seconds` permite imputar las comprobaciones previas a la reserva y al total antes de iniciar la campaña. Esa reserva se muestra separada de los segundos observados. Los intentos fallidos consumen presupuesto. La espera de una GPU disponible no se cuenta como ejecución. Un intervalo perdido sin un supervisor acreditado queda bloqueado, sin inventar su duración.

La fase principal tiene siete variantes. Las dos variantes auxiliares solo se admiten si, agregando los tres pares de semillas de validación, la combinación de memoria y HMM mejora al HMM en al menos 0,001 de crecimiento logarítmico medio, no aumenta las ruinas y no empeora la caída máxima media en más de 0,02. Esta regla está en [adaptive-campaign.json](../../configs/simulation/adaptive-campaign.json). La auditoría no interviene en esa decisión. Horas activas del proceso no equivalen a horas de kernels GPU ni a una medición de energía.

El checkpoint del esquema 2 añade estado oculto, prefijos, contexto, memorias episódicas, filtro HMM, replay, generadores y contadores observados al estado financiero, modelo, Adam, RNG y rollout. Los índices de fuente y la rotación del catálogo se guardan explícitamente. Se mantienen dos estados recientes y el mejor si es distinto. Un fallo durante una actualización parcial deja al entrenador inutilizable para nuevos snapshots hasta restaurar un checkpoint confirmado.

La restauración comprueba que el número de episodios coincida con los reinicios del contexto y los que quedan pendientes. También contrasta la observación del contexto con la cartera recuperada. El presupuesto del rollout recurrente incluye historia, prefijo, máscaras, longitudes y estados ocultos, junto con las copias necesarias para actualizar o restaurar. Se valida antes de reservar esos tensores. Los archivos serializados y la política tienen límites independientes.

SIGINT, SIGTERM y `--stop-after` solicitan una pausa recuperable. El retorno del proceso es 2 cuando queda pausado. El lanzador mantiene la admisión exclusiva de `cuda:0`, comprueba memoria disponible y no cambia silenciosamente a CPU. El diagnóstico CPU debe declararse y limita el aprendizaje a 32 transiciones.

En Linux, el lanzador comunica su PID al ejecutable mediante `--parent-pid`. El proceso nativo configura `PR_SET_PDEATHSIG` con SIGKILL y comprueba que su padre siga siendo el esperado antes de abrir datos o usar CUDA. Si el vigilante muere de forma abrupta, el proceso nativo también termina y la continuación parte del último checkpoint ya confirmado. Esa muerte no equivale a la pausa normal ni produce un checkpoint nuevo.

El [servicio de usuario](../../configs/systemd/mars-titan-adaptive-rl.service) reanuda una salida inicializada, con todo el grupo de procesos bajo su control y 90 segundos para detenerse. No reinicia automáticamente un error. Su archivo local `adaptive-rl.env` debe identificar el runtime congelado, catálogo, binario, configuración y salida mediante las variables `MARS_TITAN_RUNTIME`, `MARS_TITAN_SCENARIOS`, `MARS_TITAN_BINARY`, `MARS_TITAN_CAMPAIGN_CONFIG` y `MARS_TITAN_OUTPUT`. El entorno Python debe resolver los módulos desde ese runtime.

`registry.json` publica la cola y los contadores confirmados en cada cambio de estado. No incluye métricas, pesos ni motivos privados de fallo. El observatorio usa ese registro para identificar etapa, variante, semilla y configuración antes de que exista un recibo nativo. Un presupuesto previsto no se presenta como trabajo realizado. El resumen del coordinador no duplica las ejecuciones que coordina.

## Trazas, explicaciones y auditoría final

El observador registra decisiones antes de confirmar el paso financiero. Cada fila contiene IDs, mundo, contexto, entorno, episodio, cursor, actualización, fecha de decisión, acción, modo, distribución de acciones, crítico y recuerdos consultados. `outcome_at` identifica el resultado posterior, con recompensa, máscaras y costes. Los resultados posteriores no se describen como información disponible al decidir.

`TraceWriter` agrupa hasta 128 filas por archivo Parquet inmutable. Admite como máximo 512 MiB y reserva espacio antes de aceptar un lote. El recuento de bytes se actualiza incrementalmente y el índice completo se publica al vaciar la traza antes del checkpoint. La falta de capacidad provoca una pausa, sin expulsar filas silenciosamente. El recibo expone solo el cursor confirmado. Al reanudar, retira la cola propia que haya quedado adelantada, incluidas versiones huérfanas cuya identidad y SHA256 pueda comprobar.

[explain_rl_decisions.py](../../scripts/explain_rl_decisions.py) lee únicamente decisiones hasta ese cursor. Produce texto en español mediante plantillas, sin un LLM adicional ni otro muestreo de la política. Distingue calentamiento, acción muestreada y acción por máximo. Las probabilidades describen la política, no una probabilidad de rentabilidad.

La auditoría tiene una entrada separada, `--audit-run`, y solo carga el mejor checkpoint de una selección terminada. Comprueba configuración, código, datos y semilla antes de abrir sus mundos reservados. El coordinador congela las selecciones antes de iniciar esa fase. Evalúa costes de 0, 10 y 25 puntos básicos en grupos de hasta 16 mundos. Una pausa repite desde el principio solo el grupo que no llegó a confirmarse.

Durante la auditoría puede efectuar una segunda inferencia ocultando los recuerdos en la misma observación y con el mismo estado oculto previo. Registra ambas distribuciones, su distancia L1 y si cambia la acción por máximo. Es sensibilidad local a esa entrada de memoria. No identifica causas económicas ni explica todo el razonamiento de una red recurrente.

`audit-state.json` conserva el progreso confirmado. `audit.json` contiene métricas por mundo y coste, identidad del checkpoint elegido y referencia a la traza. `run.json` publica solo estado y contadores para el observatorio, con `partition=audit`, sin las métricas reservadas. El test final de datos reales continúa cerrado.

## Reproducción y evidencia disponible

La compilación usa los [presets PPO existentes](native-ppo.md#compilación-y-ejecución). Para preparar un catálogo nuevo con el HMM, desde la raíz del repositorio:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync --with hmmlearn==0.3.3 python scripts/prepare_adaptation_scenarios.py \
  --config configs/simulation/adaptation-scenarios.json \
  --output data/interim/adaptation-scenarios-v1 \
  --fit-hmm
```

La salida debe ser nueva. El generador no sobrescribe un catálogo previo. Las comprobaciones pequeñas se ejecutan con el binario nativo preparado:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MARS_TITAN_PPO_EXECUTABLE=build/native/native-ppo-release/mars-titan-ppo \
uv run --no-sync --with hmmlearn==0.3.3 pytest \
  tests/simulation/test_adaptation_scenarios.py \
  tests/simulation/test_decision_explanations.py \
  tests/simulation/test_native_ppo_runner.py -q
```

El siguiente comando inicia únicamente el piloto del coordinador y requiere la GPU admitida:

```bash
uv run --no-sync python scripts/run_adaptive_campaign.py \
  --scenarios data/interim/adaptation-scenarios-v1/index.json \
  --binary build/native/native-ppo-release/mars-titan-ppo \
  --config configs/simulation/adaptive-campaign.json \
  --output artifacts/native/adaptive-campaign-v1 \
  --pilot-only
```

`--resume` reutiliza esa salida e identidad. Omitir `--pilot-only` permite continuar las fases restantes dentro de sus presupuestos. Los datos sintéticos y los pesos quedan fuera de Git.

El benchmark de proceso completo reutiliza el mismo catálogo y mantiene una sola carga GPU:

```bash
uv run --no-sync python scripts/benchmark_adaptive_rl.py \
  --scenarios data/interim/adaptation-scenarios-v1/index.json \
  --binary build/native/native-ppo-release/mars-titan-ppo \
  --output artifacts/native/adaptive-benchmark-v1 \
  --variants ppo ppo_episodic_hmm --workers 1 2 4 8 \
  --transitions 8192 --repetitions 5
```

Admite `--reference-binary` y `--reference-launcher` para un contraste A/B y B/A. Conserva el calentamiento separado, las huellas, el orden de fuentes y las medidas por repetición. Detiene el contraste si cambia una entrada o falla un proceso. El [informe de verificación del coordinador y del benchmark](../../reports/resources/adaptive-campaign-verification.md) separa las pruebas con procesos controlados de las mediciones reales.

Cuando la auditoría de la combinación de memoria y HMM esté terminada, sus decisiones confirmadas se pueden consultar sin cargar la política:

```bash
uv run --no-sync python scripts/explain_rl_decisions.py \
  --run artifacts/native/adaptive-campaign-v1/audit/ppo_episodic_hmm-42/audit.json \
  --output artifacts/native/adaptive-campaign-v1/explanations-ppo_episodic_hmm-42.txt \
  --limit 20
```

La [verificación de escenarios, trazas y ejecutor](../../reports/resources/adaptive-scenarios-verification.json) registra 39 pruebas Python, 41 integraciones CPU del ejecutor y cinco mutaciones dirigidas detectadas. La revisión final de `TraceWriter` pasa diez casos con ASan, UBSan y detección de fugas, además de clang-tidy y Clang Static Analyzer. Comprueban bloques sin publicar, cortes antes y después de `flush`, rechazo de cursores imposibles, límites de bytes y recuperación sin borrar archivos ajenos.

El informe conserva la huella del binario de las 41 integraciones anteriores al cambio del índice. La cobertura y CRAP pertenecen a los dos módulos Python identificados en esa medición, no al escritor C++. Las proyecciones de bytes lógicos del índice no son latencias ni medidas físicas de disco. Estas comprobaciones no aportan resultados de calidad comparativa. La utilidad de memoria, HMM y consolidación necesita la comparación prevista y puede resultar nula o negativa.

La [comprobación de recuperación CUDA](../../reports/resources/adaptive-cuda-recovery-20260929.json) recorre las nueve variantes con 1024 transiciones cada una, pausa en 512 y recuperación. Los parámetros y las filas de traza coinciden exactamente con las ejecuciones continuas en esa prueba. El informe conserva el binario y los tiempos medidos. Ese presupuesto pequeño no equivale al piloto de la campaña ni demuestra una mejora de aprendizaje.

## Etapa de políticas de la campaña con máscaras

La [etapa de políticas por ventana](../research/training-campaign-2000.md#etapa-de-políticas-por-ventana) compara variantes de PPO, KLPO y Double DQN sobre cintas reconstruidas de la edición desde 2000, con las predicciones fuera de muestra de cada ventana walk-forward. Reutiliza los identificadores de objetivo de las [variantes PPO](ppo-objective-variants.md), el criterio de cartera del ejecutor y el [controlador KLPO](terminal-klpo-updates.md), pero no los escenarios sintéticos de este documento ni sus resultados.

El ejecutor de este documento no basta para esa etapa. `financial_parquet.cpp` solo carga fuentes de dominio sintético, así que `mars-titan-ppo` no puede leer una cinta reconstruida con su auditoría walk-forward, y `KlpoLearningController` no tiene orden ejecutable. La etapa declara esas piezas como capacidades pendientes (`native_policy_reconstructed_tapes` y `native_klpo_financial_runner`) y se detiene antes de crear la salida mientras falten. Las cintas chinas necesitan además `native_cn_a_share_rules`, que se comprueba con el motor instalado. Cuando existan, el ejecutor nativo tendrá que recibir las cintas de ajuste, validación y evaluación de cada trabajo, ajustar con el presupuesto declarado, seleccionar con el criterio de cartera sobre la validación y devolver un episodio por coste, también si falla, con la huella de la política elegida.
