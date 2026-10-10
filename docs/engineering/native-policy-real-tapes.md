# Políticas nativas sobre cintas reconstruidas

Este documento describe cómo los motores nativos leen las cintas reconstruidas de la edición desde 2000 que escribe la [etapa de políticas por ventana](../research/training-campaign-2000.md#etapa-de-políticas-por-ventana) y cómo la etapa los lanza. Afecta a `mars-titan-sim`, a `mars-titan-ppo` con el esquema 4 y a la orden nueva `mars-titan-klpo`. Todo está implementado y comprobado sin aprendizaje. Ninguna prueba ejecuta pasos de Adam, `update_ready` ni `terminal_step`, y el bloqueo de aprendizaje sigue vigente. Por eso no hay todavía políticas ajustadas ni resultados financieros que comparar.

## Admisión de una cinta real

Una cinta de dominio real solo se admite si su manifiesto declara la auditoría `unadjusted_reconstructed_walk_forward_v1` con sus tramos walk-forward, el recibo de cada tramo, el plazo de pago de dividendos y la base histórica `mercado/edición/lag-N`. `MarketTape::validate` exige además, sesión a sesión, lo mismo que el lector Python:

- Un corte de ajuste por sesión, que sale del recibo de su tramo y nunca es posterior a la decisión, y cierres anteriores al inicio del test final (1 de enero de 2024).
- Todos los cierres valorados y la primera apertura sin acción ejecutable.
- Aperturas ejecutables sobre la rejilla de precios de su mercado y fecha. En EE. UU. son fracciones de dólar antes del 28 de agosto de 2000, fracciones o céntimos hasta el 9 de abril de 2001 y céntimos después, con diezmilésimas por debajo de un dólar. En China son céntimos.
- Dividendos y splits del proveedor con su activo, su apertura y su decisión, sin eventos ambiguos.
- En China, el manifiesto de versión 2 con las reglas acreditadas de cada tablero (principal, ChiNext y STAR, con sus bandas y timbres hasta el 31 de diciembre de 2023), comparadas con la tabla nativa `china_a_share_rules`.

Las cintas sintéticas conservan sus bytes y su identidad. Una cinta sintética que declare cortes o la marca de auditoría se rechaza, en lugar de tratarse como real. Las fuentes de una misma política se emparejan por su origen: el padre en las sintéticas y la base histórica en las reales, porque cada reajuste de una ventana cambia el padre del predictor (`same_policy_origin`).

`mars-titan-sim` recorre estas cintas con las referencias fijas y puede publicar una traza por paso con `--trace`. `tests/simulation/test_native_real_tapes.py` compara esa traza con el entorno Python en ediciones sintéticas con el formato real y, si se declara `MARS_TITAN_UNADJUSTED_EDITION`, con cinco activos de EE. UU. y cinco de China de la edición real y tres planes de acciones fijas.

## Cintas de una política

`policy_tapes.cpp` reúne las comprobaciones comunes a PPO, Double DQN y KLPO. Cada cinta debe tener el papel que se le pide (ajuste, validación o evaluación) y nunca puede abrir el test. Las cintas de una política empiezan después de que termine la anterior, comparten activos, moneda, reglas de mercado y base histórica, y no admiten contexto observable. Cada cinta tiene como máximo 256 sesiones, la historia recurrente acotada de la política, suficiente para un tramo anual. La evaluación solo acepta cintas posteriores a la validación de la selección y con sus mismos activos, reglas y base.

La evaluación congelada (`run_frozen_evaluation`) recorre la política elegida con `argmax`, sin muestreo, una vez por cinta y por coste de 0, 10 y 25 pb. Escribe `identity.json` y `evaluation.json` sellados con su huella. Al reanudar devuelve el registro completado sin repetirlo, y una pausa repite los costes pendientes desde el principio.

## PPO y Double DQN con el esquema 4

El esquema 4 de `mars-titan-ppo` ajusta PPO o Double DQN sobre cintas reconstruidas, sin contexto, máscara ni HMM. Los 16 entornos recorren en ciclo las cintas de ajuste (`entorno mod cintas`) y la validación usa exactamente una cinta. La selección mantiene el criterio de cartera declarado sobre la validación. `--audit-run` evalúa después el estado elegido en cintas posteriores, solo cuando la selección está cerrada, y su identidad incluye la huella del archivo de la política, los pasos del optimizador, las transiciones, las cintas, los costes, la semilla, el dispositivo y las huellas nativas. `run.json` añade `selected_policy_sha256` y las fuentes.

La orden comparte el análisis de argumentos con `mars-titan-klpo` en `policy_command.cpp`. `--capabilities`, como único argumento, imprime el binario, sus capacidades y su identidad de compilación sin leer datos. Al llevar PPO a las cintas reconstruidas apareció un fallo latente: reanudar una ejecución seleccionada con la métrica liquidada rechazaba su propio mejor candidato por el campo de puntuación adicional. Está corregido y cubierto por la reanudación de las pruebas.

## KLPO terminal con `mars-titan-klpo`

`mars-titan-klpo` ejecuta el [colector terminal](terminal-klpo.md) y el [controlador](terminal-klpo-updates.md) sobre las mismas cintas y con la misma interfaz que `mars-titan-ppo`: configuración, semilla, presupuesto, pausa, reanudación y evaluación separada con `--audit-run`. La configuración (`kind = "native_klpo_terminal"`) declara objetivo, controlador, Adam, retorno terminal con KL de peso uno y sin descuento, actualizaciones confirmadas por referencia, bloque de gradiente, entorno, evaluación y selección. La parada temprana no está implementada y la configuración debe declararla falsa.

Cada entorno forma un episodio por oleada sobre una cinta de ajuste, que recorre en ciclo. Una oleada tiene tantas transiciones como la suma de las sesiones menos una de las cintas de sus entornos, y el controlador solo consume oleadas completas. La regla `complete_waves_within_declared_transitions` usa las oleadas enteras que caben en el presupuesto, así que KLPO nunca supera el presupuesto de PPO y puede quedarse por debajo en menos de una oleada. Un presupuesto que no admite ni una oleada se rechaza.

El actor se evalúa con `argmax` en validación antes de la primera oleada, cada `max(1, evaluation_transitions / oleada)` oleadas y tras la última. El mejor actor se guarda como `best-actor-<huella>.pt` y solo cambia si mejora el criterio de cartera. `selection.json` y `experiment.json` van sellados. Los puntos de control de `PpoCheckpointStore` guardan actor, estado de Adam, referencia, generadores, cursores de cartera y contadores de oleadas, con dos estados recientes. Una pausa durante la recogida o con la oleada completa se confirma antes de `update_ready`, nunca dentro. La evaluación exige una selección cerrada y comprueba la huella, los parámetros y los pasos del optimizador del actor elegido.

## Bloqueo de aprendizaje y etapa

Los dos binarios leen la protección local en C++ antes de abrir cintas o crear salidas, tanto al ajustar como al evaluar sobre cintas reales. El lanzador `scripts/run_native_ppo.py`, que comparten, comprueba lo mismo antes de arrancarlos, y `rl run` se detiene antes de consultar capacidades, abrir fuentes o lanzar binarios. Con la protección real del equipo, `rl run` termina con `LearningHoldError` sin crear la salida.

La etapa usa `simulation.native_policy_runs`. Un ajuste escribe su configuración junto al trabajo, la compara al reanudar, lanza el binario a través del lanzador (admisión GPU, carga única, pausa y vigilancia del padre), lee la selección en validación y evalúa la política elegida en la cinta de evaluación con los tres costes. Un traslado evalúa la política de su ancla con la configuración de ese ajuste. Los códigos de salida 2, 3 y 4 del lanzador dejan el trabajo pausado. Las cintas chinas se escriben con las reglas de acciones A que exige el lector nativo. La comprobación del informe de KLPO acepta las oleadas que caben en el presupuesto, no el número exacto de transiciones de PPO.

Las capacidades `native_policy_reconstructed_tapes` y `native_klpo_financial_runner` se comprueban preguntando a cada binario con `--capabilities`. El informe de `rl check` conserva la ruta, la huella del binario y la identidad nativa. Con los binarios del preset `native-ppo-release` compilados, `rl check` declara disponibles las cuatro capacidades de la etapa A.

## Comprobaciones

Las pruebas lanzan los binarios en el diagnóstico CPU de 32 transiciones. `tests/simulation/test_native_policy_tapes.py` (13 pruebas) pausa PPO antes de su primer recorrido completo en EE. UU. y China y rechaza evaluar esa selección abierta, completa Double DQN por debajo de su calentamiento de 256 pasos, evalúa el estado elegido y rechaza una cinta de evaluación anterior a la selección, recoge una oleada KLPO y se pausa en la fase `ready` sin actualizar, rechaza un presupuesto KLPO sin ninguna oleada completa, rechaza fuentes en otro orden, con otro papel, otro plazo de pago, otro mercado, sintéticas o chinas sin reglas y rechaza máscaras, HMM y variantes de memoria en el esquema 4. También recorre los ejecutores de la etapa con ajuste, traslado, fallo, pausa, reanudación y cambio de configuración, rechaza un ancla con otra huella o fuera de la salida de la etapa y comprueba que la etapa escribe las cintas chinas con sus reglas. La evaluación de KLPO se recorre con una selección cerrada fabricada en la prueba con el actor inicial. `test_native_real_tapes.py` (61) cubre la admisión y la paridad de las cintas reales en `mars-titan-sim`, `test_campaign_stage.py` (32) la etapa con las sondas por binario y `test_native_ppo_launcher.py` (47) el lanzador con el esquema 4 y KLPO.

CTest pasa en Release y con ASan y UBSan (25 de 28 pruebas). Las tres excluidas, `ppo_policy`, `ppo_training` y `ppo_gru_packing`, ejecutan pasos de Adam y no se lanzan bajo el bloqueo. Las 74 pruebas de `test_native_policy_tapes.py` y `test_native_real_tapes.py`, con la prueba de humo de la edición real, pasan también con los binarios instrumentados y sin avisos de los sanitizadores. clang-tidy con la configuración del proyecto y Clang Static Analyzer no dan avisos en las 16 fuentes nativas modificadas, después de corregir los hallazgos iniciales de clang-tidy en `reconstructed_tape.cpp` y `klpo_experiment.cpp`.

La mutación dirigida aplicó de uno en uno 27 defectos a la lógica nueva. Quince estaban en Python: regla de oleadas de KLPO y su cota, capacidad exigida a cada binario, fórmula de la oleada, costes del motor, cinta y política de cada episodio evaluado, reanudación, cambio de configuración, carpeta del ancla, códigos de pausa, reglas chinas al escribir la cinta y bloqueo, nombre y catálogo del lanzador. Doce estaban en C++: origen de las fuentes, corte posterior a la decisión, orden de las cintas, evaluación anterior a la selección, oleadas redondeadas hacia arriba, evaluación inicial de KLPO, selección abierta en KLPO y en PPO, bloqueo de las dos evaluaciones, coste no aplicado y reanudación con la métrica liquidada. En la primera pasada sobrevivieron nueve. Se añadieron las pruebas que faltaban y se corrigió la comprobación de la carpeta del ancla, que con una ruta poco profunda fallaba con un `IndexError` en lugar de explicar el error. Con un mutante más, el presupuesto KLPO sin ninguna oleada, los 28 fallan ahora. No se aplicó el mutante que retira la pausa antes de `update_ready`, porque ejecutaría pasos de Adam.

## Caudal del paso nativo con cintas reales

La medida compara `mars-titan-sim --compare --workers 1` (tres referencias fijas por tres costes, nueve episodios, sin red ni aprendizaje) sobre una cinta reconstruida de 2023 de la edición real y sobre una cinta sintética de la misma forma, ambas con puntuaciones sintéticas. Se usó el binario Release, diez repeticiones tras una de calentamiento, en un AMD Ryzen 9 8945HS con el perfil de ahorro de energía y otros procesos activos (carga media de 7,7 en 16 hilos). El caudal es la suma de pasos dividida por la suma del tiempo interno de cada episodio.

| Mercado y activos | Cinta | Transiciones | Pasos por segundo, mediana (mínimo a máximo) | Lectura (ms) |
| --- | --- | --- | --- | --- |
| EE. UU., 32 | Real | 2.241 | 83.362 (78.767 a 86.008) | 5,4 |
| EE. UU., 32 | Sintética | 2.241 | 85.292 (81.636 a 88.514) | 4,7 |
| China, 32 | Real | 2.169 | 79.290 (59.915 a 84.836) | 5,4 |
| China, 32 | Sintética | 2.169 | 80.550 (73.675 a 85.216) | 5,6 |
| EE. UU., 111 | Real | 2.241 | 49.337 (42.768 a 53.188) | 11,3 |
| EE. UU., 111 | Sintética | 2.241 | 49.857 (48.066 a 54.483) | 8,8 |
| China, 128 | Real | 2.169 | 47.650 (41.599 a 52.095) | 11,5 |
| China, 128 | Sintética | 2.169 | 49.718 (48.332 a 50.409) | 14,7 |

En EE. UU. se pidieron los 128 primeros símbolos de la edición por orden alfabético y la reconstrucción conservó 111. Las medianas con cintas reales quedan entre un 1 % y un 4 % por debajo de las sintéticas y los intervalos se solapan en los cuatro casos, así que la medida no muestra un coste propio de las cintas reales en el paso. La lectura de la cinta real tarda algo más en EE. UU. y menos en China con 128 activos, siempre por debajo de 15 ms. El proceso completo tarda entre 41 y 81 ms. El pico de unos 0,6 GiB anotado en esta medida no es del proceso C++. Su `VmHWM` es de unos 40 MB con una cinta de 128 activos, mientras que el `ru_maxrss` que devuelve `wait4` conserva en Linux el máximo de la imagen anterior a `exec`, es decir, el del proceso Python que lanza el binario. Con un padre de 15 MB y otro de 686 MB, `ru_maxrss` vale 40 MB y 686 MB y `VmHWM` vale 40 MB en ambos casos ([informe de la etapa RL nativa](../../reports/engineering/rl-stage-native-20261009/README.md#memoria)). No hay, por tanto, memoria que reducir en `mars-titan-sim`. Con estas cifras no hay un cuello de botella que justifique optimizar el lector o el paso, y la medida no dice nada del coste de la red, del optimizador ni de la GPU.

## Recogida de las políticas en CUDA y CPU

El preset `native-ppo-release` compila también con `-DMARS_TITAN_LIBTORCH_ENABLE_CUDA=ON`, y CTest pasa en esa compilación sin las tres pruebas que ejecutan Adam. `mars-titan-policy-benchmark` mide sin pasos de optimizador el recorrido de las políticas sobre cintas reconstruidas de 128 activos: paso del entorno, inferencia con sus copias, GAE, minilote PPO con forward y backward, recogida de `PpoTrainer`, oleada KLPO con su objetivo y evaluación. El [informe de la etapa RL nativa](../../reports/engineering/rl-stage-native-20261009/README.md) recoge las medidas, la paridad y la proyección de horas.

Dos cambios del colector reducen el coste sin alterar ninguna huella. KLPO valida en cada paso solo los pasos añadidos de cada episodio (la oleada pasa de 0,55 s a unos 0,2 s en CPU) y PPO muestrea de la salida del bootstrap validado en lugar de repetir el forward (el paso de `PpoTrainer` en `cuda:0` baja de 647 a 526 µs en EE. UU.). Para estas MLP pequeñas, un proceso en CPU es más rápido que uno en `cuda:0` en casi todo el recorrido, porque la GPU queda limitada por lanzamientos y sincronizaciones. Varios procesos CUDA solo escalan bien con MPS. La etapa sigue exigiendo `cuda:0` y la elección de dispositivo queda pendiente del protocolo, porque cambia las secuencias aleatorias y la identidad de las ejecuciones.

## Pendiente

Mientras dure el bloqueo no se han recorrido una actualización de Adam de KLPO ni su recuperación posterior, un ajuste PPO completo sobre cintas reales ni las tres pruebas CTest que ejecutan pasos de Adam. El informe deja las órdenes para medir el coste de la actualización de cada motor cuando se levante, antes de fijar el presupuesto de transiciones.
