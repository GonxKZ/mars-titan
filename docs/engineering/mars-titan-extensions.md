# MARS-TITAN con ampliaciones sobre Titans-MAC

Este documento describe la implementación de la variante MARS-TITAN con ampliaciones: el constructor de combinaciones desde su [declaración](../../configs/titans/mars-titan-extensions.json), la corrección asociativa B6 dentro de `FinancialSession` y en las ventanas walk-forward, el modo de K con episodios fijos, el entrenador cronológico del lector episódico, la entrada por ventana walk-forward con su traslado y el registro en la campaña con máscaras. Nada de esto se ha ejecutado con datos. El bloqueo de aprendizaje sigue vigente y las pruebas recorren los bucles hasta el paso del optimizador con un registrador que no modifica pesos.

La especificación, los niveles y la correspondencia con el documento de integración están en la [arquitectura Titans-MAC](../research/titans-mac-architecture.md#variante-mars-titan-con-ampliaciones). Aquí se fijan contratos y evidencias.

## Constructor de la variante

`memory/mars_titan_variant.py` lee la declaración y exige que los componentes marcados como conectados coincidan con los que conoce el código: `episodic_bank`, `refinements`, `refinement_episodes` y `associative_memory`. Una deriva entre declaración y código hace fallar la carga.

`select_variant(declaration, components, base=...)` valida una combinación y devuelve `MarsTitanVariant`. La base es la identidad del núcleo que devuelve `core_identity(predictor, recipe)`: arquitectura, configuración de `FinancialPredictor`, precisión, huella de parámetros y receta del núcleo. La receta entra como parámetro, así que la variante parte de cualquier receta de Titans-MAC `mac_online`, incluida la de campaña con `memory_residual_layer_norm` y `gate_bias`.

Reglas de la combinación:

- Un valor apagado se expresa omitiendo el componente. Así una misma combinación no tiene dos identidades.
- `refinements` necesita `episodic_bank` activo. `refinement_episodes="first_read"` necesita K mayor que 1, porque con K = 1 los dos modos coinciden bit a bit.
- `associative_memory` declara `rule`, `key`, `rate` y `forgetting`. Se rechaza junto al banco episódico, porque el error del banco y la escritura de A mezclarían dos memorias en una misma emisión.
- `episodic_bank="m3"` usa la [escritura M3](m3-write-policy.md). La carga exige que su bloque en la declaración coincida con `write_scores.declaration()`, y `session_options` necesita las escalas `scalers` del tramo de entrenamiento.
- Los seis componentes declarados sin conexión se rechazan con el texto `pending` de la declaración.

Con todos los componentes apagados, `identity()` devuelve la base sin cambios. La combinación no crea otro brazo: es el Titans-MAC `mac_online` elegido, con `readout=None`, admisión M0 y sin control local. Cada combinación activa tiene identidad propia, el JSON canónico de la base y de los componentes normalizados. Las 23 combinaciones válidas de los componentes conectados producen 23 huellas distintas, y el orden de las claves o escribir 1 frente a 1.0 no cambia la huella.

`consumer(predictor, readout)` exige que el predictor sea exactamente el núcleo de la base y que el lector tenga el modo, K y selección de la combinación. `session_options(capacity, seed)` devuelve admisión, retención y corrección B6 para `FinancialSession`.

La paridad completa está comprobada en `tests/memory/test_mars_titan_variant.py::test_all_disabled_session_is_exactly_the_mac_online_session`: la sesión construida desde la declaración con todo apagado y la sesión `mac_online` escrita a mano tienen el mismo contrato y modelo, las mismas emisiones en seis eventos con labels y liquidación, el mismo estado de banco, la misma cola pendiente y los mismos pesos rápidos bit a bit, sin cambiar el RNG global. La combinación M1 también reproduce bit a bit la sesión M1 escrita a mano.

## K con episodios fijos

`EpisodicReadoutConfig(episode_selection="first_read")` añade un modo con identidad propia. El primer paso selecciona los vecinos igual que antes. Los pasos siguientes recalculan los pesos softmax con la consulta del estado refinado sobre esas mismas posiciones, sin volver a buscar. Así K = 2 o 4 examina los mismos episodios únicos que K = 1 y separa más cálculo de más evidencia. El modo anterior (`per_step`) conserva literalmente su identidad y su huella.

Las pruebas de `tests/models/titans/test_episodic_readout_fixed_episodes.py` comprueban la paridad bit a bit de K = 1 en los dos modos, en FP32 y FP64 y con o sin grafo, un control en el que `per_step` cambia de episodio mientras `first_read` no lo hace, la ecuación de los pesos recalculados, gradcheck FP64 respecto al estado y a la matriz de consulta, y la copia de parámetros entre modos.

## Corrección asociativa B6 en la sesión

`FinancialSession(..., associative=MatureCorrection(...))` conecta la memoria asociativa como control B6 sobre Titans-MAC sin banco (admisión M0). En cada evento con corte t:

```text
emitida_i = núcleo_i + k_iᵀ A_(g−1)
```

donde `A_(g−1)` es la matriz confirmada en la generación anterior, común a todo el evento, y `k_i` la entrada del codec de la decisión normalizada en L2 y FP64. Con `key="constant"` la única clave es 1 y A se reduce a un corrector de sesgo, que es el control que permite descartar la dependencia de la clave.

Al resolver los resultados maduros del evento, cada etiqueta aplicada escribe A con el valor `etiqueta − núcleo`, nunca con la predicción emitida. Los identificadores siguen el contador de escrituras en el orden canónico en el que el ejecutor nativo entrega los resultados, así que la escritura no depende del orden de carga. La generación guarda A y la predicción del núcleo de cada pendiente, alineadas con la cola. La verificación exige que el número de escrituras de A coincida con los labels aplicados. Sin corrección, el contrato y la generación no cambian.

`tests/memory/test_financial_session_associative.py` reproduce la recurrencia con un oráculo independiente que parte de la sesión sin corrección: emisiones a 10⁻¹⁵, matriz bit a bit en cada generación, cola del núcleo y pesos rápidos iguales a los de Titans-MAC, causalidad (perturbar las etiquetas de un evento solo cambia emisiones posteriores), control de clave constante, corte antes de publicar con recuperación exacta y rechazos con banco o fuera de Titans-MAC. Siete mutaciones dirigidas (signo de la corrección, escritura con la emisión, poda de la cola, admisión con banco, contrato, ausencia de escritura y normalización de clave) hicieron fallar las pruebas.

La regla `kalman` (PT3) usa el mismo contrato con sus propias varianzas en lugar de η y λ, y se declara como ablación A12 frente a la proximal ([documento](kalman-associative-memory.md)). El recorrido por ventanas de la campaña emite la corrección con las reglas delta y proximal en [su propia ventana](#corrección-b6-en-las-ventanas-walk-forward). La regla `kalman` todavía no tiene ventana ni brazo. La elección de η y λ, y de las varianzas de kalman, en desarrollo sigue pendiente.

## Entrenador del lector episódico

`training/mars_titan_run.py` ajusta solo `EpisodicReadout` sobre el padre `mac_online` seleccionado, que llega congelado, en evaluación y sin gradientes. La memoria rápida del padre avanza con la regla asociativa de Titans en cada observación, igual que en el brazo Titans-MAC, y no recibe información de etiquetas. La receta es `configs/titans/episodic-readout-historical-masked.json`, con el mismo número de casos de búsqueda que Titans-MAC (tasas 1e-4 y 1e-3), 30 épocas, presupuesto fijo, selección por `session_mae` de validación, recorte 1,0 y `update_instants` igual al truncamiento de la receta de Titans.

Orden dentro de cada evento del ajuste:

1. Resolver las etiquetas maduras contra la predicción emitida y guardarlas para la admisión.
2. Si el tramo tiene `update_instants` instantes, actualizar con las etiquetas ya maduras.
3. Predecir las entradas del instante con la instantánea del banco confirmada antes de esas etiquetas.
4. Publicar en el banco las etiquetas del evento, con IDs en orden canónico de decisión y flujo.

Las predicciones del tramo no guardan su grafo. Al actualizar se repite el lector sobre cada bloque emitido con su misma instantánea y estado de trabajo, se exige que la repetición reproduzca exactamente lo emitido y se acumula la pérdida media del tramo. La pérdida es la pinball de los cinco niveles con `quantile_head_v1` y L1 con la cabeza escalar. No hay BPTT entre instantes. En M0 el lector usa el control `no_bank` y las proyecciones de lectura quedan fuera del optimizador porque no reciben información. M1 usa la retención original y M2 sus tres índices con el error maduro de la predicción emitida. M3 usa los mismos índices con la puntuación compuesta. Calcula anomalía y relevancia al emitir, las guarda con la predicción pendiente y acumula por pasada admitidos, rechazados y expulsados del selectivo y candidatos sin relevancia conocida. `ReadoutTrainer` rechaza unas escalas M3 que no procedan de su tramo de entrenamiento.

El checkpoint de recuperación guarda parámetros del lector, optimizador, RNG, cursor, selección, historial, pesos rápidos del padre por flujo, cola pendiente, admisión del evento pendiente, banco nativo empaquetado como bytes y errores de sesión. Con M3 añade los rasgos de cada predicción pendiente y los componentes y contadores del banco, y las escalas forman parte de la identidad del ajuste. El mejor estado se guarda aparte y solo se sustituye al mejorar el criterio declarado. `run()` llama siempre a `require_learning_allowed`.

## Ventana walk-forward y traslado

`training/mars_titan_walk_forward.py` sigue el contrato de `titans_walk_forward`. `run_mars_titan_window(view, parent, recipe, components=..., seed, output, search_case)` recibe la carpeta de la ventana Titans-MAC `mac_online` completada en la misma vista y semilla. Comprueba el padre y sus predicciones, valida la combinación antes de abrir fuentes, exige las mismas fases que el padre, carga su mejor estado exacto y ajusta el lector. Con M3 estima antes las escalas con el tramo de entrenamiento de la ventana, o reutiliza las de su `fit/run.json` al reanudar. Escribe `selected.json` con el checkpoint del padre y el mejor lector, que es el estado elegido que firma el recibo de la campaña, y un Parquet por tramo con las mismas filas y objetivos que el padre.

`carry_mars_titan` aplica en una ventana posterior el padre y el lector elegidos en el ancla, sin ajustar nada. Solo admite el cambio de huella de la vista en el padre y en el codec del lector. Cada tramo empieza con la memoria rápida inicial, el banco vacío y su calentamiento de entradas, el mismo contrato que en el ancla. M3 aplica las escalas del ancla y su huella queda en el recibo.

La ventana y el traslado son comunes con el [factorial CM-v1](../experiments/mars_titan_cm_v1/factorial.md). `ReadoutFamily` reúne lo que distingue a cada familia: el control C que debe declarar el padre, la variante, la retención del banco y la identidad. MARS-TITAN exige un padre sin control C.

## Corrección B6 en las ventanas walk-forward

`training/mars_titan_correction.py` lleva la corrección B6 al recorrido por ventanas. `run_correction_window(view, parent, recipe, components=..., seed, output, search_case)` recibe la misma carpeta de la ventana Titans-MAC `mac_online` que el lector. Comprueba el padre, valida la combinación y el caso antes de abrir fuentes, exige las mismas fases que el padre y carga su mejor estado exacto. No construye ni lee el índice de entrenamiento. La especificación de entrada del padre se reconstruye con la huella de ese índice que guarda su informe y se exige compatible con la del tramo medido, así que un padre de otra vista o de otra política de entradas se rechaza.

`CorrectionInference` hereda de `ChronologicalInference` y conserva sin cambios el cálculo del núcleo. Dentro de cada evento con corte t:

1. Resolver las etiquetas que maduran en t contra la predicción del núcleo guardada al emitir y preparar su escritura con el valor `etiqueta − núcleo`.
2. Predecir con el núcleo las entradas del evento y sumar a cada predicción y a sus cinco cuantiles la lectura kᵀA, con la A confirmada antes del evento.
3. Escribir en A las etiquetas del paso 1 en orden canónico de decisión y flujo, con identificadores consecutivos al contador de escrituras.

Es el orden de `FinancialSession`: ninguna emisión ve las etiquetas que maduran en su propio evento. A empieza en cero en cada tramo medido, igual que la memoria rápida del padre, y solo recibe etiquetas maduras de ese tramo. Los eventos de calentamiento avanzan la memoria rápida sin emitir ni corregir. Como la corrección desplaza por igual el punto y los cuantiles, no cambia la anchura de los intervalos antes de la calibración común. La ventana escribe `run.json`, `selected.json` con el padre y la corrección, y un Parquet por tramo con las columnas de `titans_walk_forward`. Un tramo interrumpido se repite desde su inicio, porque A y la memoria rápida se reinician en cada recorrido. `carry_correction` aplica en una ventana posterior el padre y el η y el λ elegidos en el ancla, sin seleccionar nada. `mars_titan_fit` envía a esta ventana los brazos con `associative_memory` y `mars_titan_carry` elige el traslado según el tipo del ancla, así que los ejecutores de la campaña son los mismos para todos los brazos de MARS-TITAN.

La receta [`mature-correction-historical-masked.json`](../../configs/titans/mature-correction-historical-masked.json) fija λ = 0,01 y declara dos casos de búsqueda, η = 0,05 y η = 0,25, como Titans-MAC y las referencias neuronales. Se fijaron antes de ejecutar a partir de la memoria efectiva que implican, sin mirar datos. Con la regla proximal y la clave constante, la media de una cohorte madura pesa η/(1+η) en cada evento (0,048 y 0,2), y con retención 1 − λ = 0,99 una dirección que deja de recibir etiquetas se reduce a la mitad en unas 69 sesiones. La campaña elige el caso con el MAE por sesión de validación. B6 no tiene parámetros entrenables, épocas ni optimizador, pero elige η con la validación, así que la ventana también comprueba el bloqueo de aprendizaje antes de leer ninguna fuente.

Los brazos de la campaña usan la regla proximal, invariante al orden dentro de una cohorte. La regla delta escribe fila a fila y, con cohortes de miles de etiquetas, su resultado depende del orden canónico de los activos salvo que η sea del orden del inverso de la cohorte. Queda conectada en la ventana y comprobada contra la sesión, sin brazo declarado. `mars_titan_b6` usa la clave del codec y `mars_titan_b6_bias` la clave constante, que reduce A a un corrector de sesgo. La comparación declara dos familias: `mature_correction` contrasta Titans-MAC con los dos brazos y `correction_key` contrasta el corrector de sesgo con la clave del codec. Si B6 no mejora al corrector de sesgo con las mismas etiquetas, la dependencia de la clave queda descartada.

`tests/memory/test_mars_titan_correction_parity.py` entrega los seis eventos de cuatro flujos a `FinancialSession` con la corrección y, como eventos de observación equivalentes, a `CorrectionInference`. Las 20 emisiones coinciden bit a bit con la regla delta, con la proximal y con la clave constante, y la matriz A, el cursor y la huella finales coinciden con la última generación de la sesión. Con η = 0 las filas coinciden con las de `ChronologicalInference`. Alterar las etiquetas que maduran en el tercer instante solo cambia emisiones posteriores, y alterar las entradas del último instante no cambia ninguna anterior. Una cohorte con etiquetas de dos instantes de decisión, entregadas en orden inverso, se escribe en orden de decisión y flujo. Las predicciones del núcleo, sus parámetros y sus buffers no cambian y no reciben gradientes.

`tests/training/test_mars_titan_correction.py` recorre ventanas del corpus técnico en CPU. Con η = 0 la ventana reproduce bit a bit las predicciones del padre recargado desde su checkpoint y, con tolerancia relativa 10⁻¹⁴, las del Parquet del padre. Esa diferencia ya existe sin B6: 2 de las 110 filas de evaluación difieren en un ulp entre el padre que queda en memoria al terminar su ajuste y el padre recargado, y una `ChronologicalInference` sin corrección sobre el padre recargado repite la misma diferencia. Las pruebas comprueban también que la corrección desplaza por igual punto y cuantiles y deja intactas las primeras decisiones, la identidad y los índices de los tramos medidos, la reanudación tras una parada con las mismas huellas, el bloqueo, los rechazos previos a crear la salida, la cota η ≤ 2 − λ de la regla delta, el esquema de la receta, el padre (variante, semilla, vista, control C y estado) y una petición cambiada al reanudar.

## Registro en la campaña

`campaign_plan` acepta una sección opcional `mars_titan` con la receta del lector, la combinación de cada brazo, `pending_arms` con el motivo de los brazos sin definición, el brazo padre y la semilla de búsqueda. El padre debe ser el brazo `mac_online` de la sección `titans_mac`, con la misma semilla de búsqueda y semillas que cubran las de cada brazo. Cada búsqueda depende de las búsquedas del padre en su ventana y cada finalista, además de sus búsquedas, del finalista del padre con su semilla. `masked_campaign` resuelve ese padre y lo entrega al ejecutor en `JobRun.parent`, y la identidad del trabajo incluye su recibo. El finalista elige ganador solo entre sus propias búsquedas.

La sección admite además `correction_recipe`, la receta de B6, que se declara si y solo si algún brazo usa `associative_memory`. Un brazo B6 declara solo la regla y la clave, nunca banco ni K, y la receta debe tener tantos casos de η como Titans-MAC. Sus búsquedas y finalistas dependen del padre igual que los del lector.

Las configuraciones A y B no declaran todavía la sección, igual que la GRU episódica. La campaña elegida, la A, la incluye en su declaración ampliada. Con los nueve brazos (M0, M1, M2, M3, M1 con K = 2 y K = 4, M1 con K = 4 y episodios de la primera lectura, B6 con la clave del codec y B6 con la clave constante) añade 1.620 ajustes en A y 612 ajustes con 756 traslados en B. Cada brazo nuevo suma 180 trabajos en A. Los de B6 no ajustan parámetros y solo predicen los tramos medidos. La [declaración preparada](../../configs/baselines/historical-masked-campaign-extensions.json) incluye los nueve brazos, y la medida de caudal estima las escalas de M3 con la regla de la campaña sobre el tramo de entrenamiento de la ventana medida. Estos recuentos corresponden a la configuración A ampliada. La campaña A v2 conjunta (#431) copió la sección `mars_titan` con los seis brazos anteriores a esta integración y todavía no declara los otros tres.

`tests/training/test_mars_titan_campaign.py` comprueba el plan sobre la comparación declarada y ejecuta una campaña B reducida sobre US con `titans_mac_online`, `mars_titan_m1` y `mars_titan_m3` con los ejecutores reales en CPU hasta la tercera ventana: un ajuste por caso y dos traslados, con las mismas filas que Titans-MAC, el padre elegido en la identidad y los recibos de ventana publicados. M3 estima sus escalas en la ventana ajustada y traslada las del ancla. La misma campaña reducida ajusta y traslada `mars_titan_b6` con los ejecutores reales, y el traslado repite bit a bit sin ajustar. Con las cuatro secciones declaradas, los 27 brazos de la comparación tienen productor, salvo el control en línea, cuyos trabajos declara la campaña A por etapas.

## Comprobaciones

| Archivo | Qué comprueba |
| --- | --- |
| `tests/memory/test_mars_titan_variant.py` | Declaración, identidades, rechazos con motivo, consumidor y paridad de sesión |
| `tests/memory/test_financial_session_associative.py` | Corrección B6 frente a un oráculo, causalidad, control constante y recuperación |
| `tests/models/titans/test_episodic_readout_fixed_episodes.py` | Modo `first_read`, paridad K = 1 y gradcheck FP64 |
| `tests/training/test_mars_titan_run.py` | Receta, roles, rechazos, bucle hasta el paso sin cambiar pesos, orden del evento, causalidad, aislamiento del estado rápido del padre, reanudación y protección |
| `tests/training/test_mars_titan_walk_forward.py` | Ventana sobre el padre, estado compuesto, filas, reanudación y rechazos previos a abrir fuentes |
| `tests/training/test_mars_titan_campaign.py` | Plan, dependencias, pendientes con motivo, receta de B6 y campaña reducida con ejecutores reales |
| `tests/memory/test_mars_titan_correction_parity.py` | Ventana B6 frente a `FinancialSession`, η = 0, orden canónico de escritura, causalidad y aislamiento del padre |
| `tests/training/test_mars_titan_correction.py` | Ventana B6 sobre el corpus técnico: paridad con el padre, cuantiles, identidad, reanudación, bloqueo, rechazos y frontera macro |
| `tests/training/test_campaign_throughput.py` | Medida de B6 sin lector ni pasos y horas de un trabajo que solo predice |
| `tests/memory/test_write_scores.py`, `tests/memory/test_write_policy_m3.py` y `tests/memory/test_financial_session_m3.py` | [Escritura M3](m3-write-policy.md): componentes, escalas, banco y sesión |

El aislamiento se comprueba registrando cada llamada a `prepare` del padre: con M0, M1, M2 y M3 la secuencia de predicciones del núcleo y de estados rápidos exportados coincide bit a bit con la de `ChronologicalInference` de Titans-MAC. `tests/memory/test_mars_titan_session_parity.py` entrega los mismos eventos a `FinancialSession` y al recorrido cronológico del lector, con un banco de capacidad 4 que recibe 16 etiquetas: las 20 emisiones coinciden bit a bit en M1, M2, M3 y K = 2 con sus dos modos de selección, igual que los IDs admitidos. En M3 los flujos tienen noticias, fundamentales y últimos precios distintos, así que los tres componentes varían.

Se aplicaron 16 mutaciones dirigidas en copias aisladas, comprobando que las pruebas importaban la copia. Catorce hicieron fallar las pruebas: admitir antes de predecir, M0 con lectura, el orden de admisión, el peso de cada bloque en la pérdida del tramo, la identidad con todo apagado, B6 con banco, la selección del consumidor, la semilla y el lector elegidos del padre, el codec trasladado, el ganador del finalista, el padre ausente, las dependencias de búsqueda y el motivo pendiente. El peso de bloque solo se detectó tras añadir la prueba que compara bloques de una fila con bloques de todo el instante. Las otras dos eran equivalentes. Invertir el signo del error M2 no cambia nada observable porque el banco solo usa su valor absoluto, y la comprobación explícita de K con episodios fijos repetía la regla `requires` de la declaración, así que se retiró.

La ventana B6 y su registro recibieron 13 mutaciones más, aplicadas de una en una. La pasada final se repitió en una copia aislada, comprobando que las pruebas importaban la copia: escribir A antes de emitir, escribir con la emisión en vez del núcleo, ordenar la cohorte por flujo antes que por decisión, no desplazar los cuantiles, no reiniciar A en cada tramo, anular la lectura en los eventos que traen etiquetas, corregir el calentamiento, aceptar campos de más en el brazo y en el plan, admitir un lector con B6, aceptar brazos B6 sin receta, omitir la validación en las horas de B6 y trasladar B6 con el traslado del lector. En la primera pasada sobrevivió el orden de la cohorte, porque en la fixture cada evento solo maduraba un instante de decisión. La prueba añadida retrasa las etiquetas de un evento al siguiente y lo detecta. Otra mutación inicial, que sustituía la comprobación del calentamiento por la de predicciones pendientes, era equivalente y se cambió por retirar la comprobación, que se detecta. En la pasada final fallan las 13, todas por una aserción o por el error que la mutación provoca en la ventana.

## Comprobaciones CUDA

`tests/training/cuda_mars_titan_run_check.py` compara en CPU y `cuda:0` un ajuste completo sin pasos y su validación, en FP32 y FP64, con M1 y K = 1, K = 4 por paso y K = 4 con episodios fijos, y con M3 y K = 1 y K = 4 con episodios fijos. Los casos M3 se describen en la [guía de M3](m3-write-policy.md#comprobaciones-cuda). Sus diez casos pasan en `cuda:0` desde el 9 de octubre ([recibo](../../reports/engineering/cuda-checks-20261009/mars-titan-readout-cuda.json)), una vez inicializada CUDA antes de reiniciar el pico de memoria. La orden, desde la raíz del repositorio:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  MARS_TITAN_EPISODIC_NATIVE=<enlace episódico nativo compilado> \
  MARS_TITAN_MARS_RUN_CHECK_REPORT=$PWD/mars-titan-readout-cuda.json \
  UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA> \
  uv run --no-sync python -m pytest -q tests/training/cuda_mars_titan_run_check.py
```

`tests/models/titans/cuda_episodic_readout_check.py` cubre los dos modos de selección desde el 10 de octubre. En `cuda:0`, con FP32 y FP64 y K = 1, 2 y 4, `first_read` lee en todos los pasos los episodios del primero, con K = 1 coincide bit a bit con `per_step` y, en el control que fuerza otra búsqueda, conserva el episodio que `per_step` cambia ([recibo](../../reports/engineering/cuda-adapter-checks-20261010/episodic-readout-cuda.json)). La sesión con B6 no tiene comprobación CUDA porque A vive en CPU por diseño. En la ventana B6 el padre sí puede estar en `cuda:0`, así que `tests/memory/cuda_mars_titan_correction_check.py` recorre los seis eventos de la fixture de cuatro flujos con el mismo padre en CPU y `cuda:0`, en FP32 y FP64, con las reglas delta y proximal y con la clave constante. Pasó en `cuda:0` el 9 de octubre en 3,5 s, con un pico de 66,6 MiB del asignador ([recibo](../../reports/engineering/mars-titan-integration-components-20261010.json)). En esta fixture las emisiones y la matriz coinciden exactamente con CPU en los seis casos y el segundo recorrido repite el primero bit a bit. Una diferencia nula en una fixture pequeña no garantiza la misma igualdad a escala, así que la comprobación conserva sus tolerancias.

## Coste medido de los brazos nuevos

Los brazos nuevos se midieron en `cuda:0` el 10 de octubre con la función de la orden de caudal (`measure_mars_titan`), sin pasos de optimizador y con el padre `mac_online` en sus pesos iniciales ([recibo](../../reports/engineering/mars-titan-integration-components-20261010.json)). M1 y M1 con K = 4 se midieron en la misma ejecución como referencia. La ventana es la US+CN `fold-012`, la más poblada y la única vista conjunta que seguía en disco, con hasta 4.089 entradas por evento. Cada brazo recorre 64 eventos tras 8 de calentamiento, en FP32 estricto sin TF32. Índices y observaciones se prepararon sin la plaza de GPU (13 min) y la fase de GPU (13 min) se ejecutó con el candado compartido, sin otra carga de cómputo en CUDA. El proceso llegó a 4,4 GB de RAM.

| Brazo | Ajuste (filas/s) | Inferencia (filas/s) | Inferencia frente al núcleo | Pico de VRAM | Horas en A | Horas por trabajo | Horas del ámbito US+CN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Núcleo `mac_online` sin corrección | | 2.607 | 1,00 | 1,29 GiB | | | |
| `mars_titan_b6` | Sin ajuste | 1.152 | 2,26 | 1,29 GiB | 122 | 0,68 | 55 |
| `mars_titan_b6_bias` | Sin ajuste | 2.398 | 1,09 | 1,29 GiB | 59 | 0,33 | 27 |
| `mars_titan_m1_k4_first_read` | 1.033 | 1.534 | 1,70 | 1,43 GiB | 8.649 | 48,1 | 4.089 |
| `mars_titan_m1_k4` (referencia) | 1.074 | 1.449 | 1,80 | 1,43 GiB | 8.413 | 46,7 | 3.976 |
| `mars_titan_m1` (referencia) | 1.123 | 1.465 | 1,78 | 1,43 GiB | 8.083 | 44,9 | 3.819 |

Las horas son de GPU y aplican el caudal de `fold-012` a los 180 trabajos de cada brazo en la campaña A de `develop` (52 en el ámbito US+CN), con los recuentos verificados de las vistas, 12 meses de calentamiento y 30 épocas, la cota superior de la regla de presupuesto fijo. Las de un lector no incluyen el ajuste de su padre Titans-MAC. Un trabajo B6 solo predice validación, calibración y evaluación con su calentamiento. Hay una sola medida por brazo y sin repeticiones, así que las diferencias de pocos puntos porcentuales entre lectores no son concluyentes, y el caudal de ventanas con cohortes más pequeñas puede ser distinto. Durante la medida había otros procesos de CPU en el equipo, sin registrar su carga, así que los caudales pueden quedar por debajo de los de una ejecución sin competencia. La medida usó el código de la rama antes de fusionarla con develop y una vista v3 de la edición. Las proyecciones de la sección 4.4 (#474) y PT1 (#477) quedan desactivadas con la configuración de la campaña, pero dos cambios posteriores sí afectan a la tabla. #446 reorganizó el recorrido del lector para los adaptadores, y #482 cambió la receta de Titans-MAC (bloques de 1.024 flujos en lugar de 128, `max_batch` de 1.024 y precisión `fp32_strict`) y el codificador del Transformer compacto, que ahora solo calcula el último token. B6 predice con ese núcleo y recorre los eventos en bloques de la receta del padre, así que sus cifras y las del núcleo describen también el código anterior. El 10 de octubre se intentó repetir la medida de B6 sobre la misma `fold-012`, pero la vista ya no abre (`Falta un artefacto regular dentro del origen declarado`), porque la sustitución de la edición v3 ha empezado. Todas las cifras de la tabla quedan pendientes de volver a medir sobre las vistas de la edición v3.1 cuando existan. Hasta entonces, la tabla sirve para ordenar los brazos por coste y no para fijar el calendario.

El brazo B6 con la clave del codec cuesta alrededor del 1,5 % de un lector y el de clave constante, la mitad. Como la regla es la misma en los dos brazos, la diferencia procede de la clave. Con la del codec la inferencia tarda 2,26 veces lo que el núcleo y con la constante solo 1,09 veces. Son 63 horas en A, así que optimizar ahora esa codificación no compensa frente al coste de los lectores. `first_read` cuesta lo mismo que un lector con K = 4.

Con estas medidas, los dos brazos B6 responden a la segunda vía de memoria con un control fuerte por 181 horas de GPU en A, y se proponen para la campaña. `mars_titan_m1_k4_first_read` (A10) añadiría 8.649 horas en A, o 4.089 si solo cuenta el ámbito US+CN, y su entrada queda pendiente de decidir, con ese coste a la vista, al declarar los brazos de integración en la campaña A v2 (#431). La medida muestra además que el coste de MARS-TITAN lo dominan los lectores, de 45 a 48 horas de GPU por trabajo de media con el lector anterior a #446 y la receta anterior a #482, y no los componentes nuevos.

### Recursos en la declaración de ejecución

La declaración de ejecución de #469 no tiene entradas por brazo para MARS-TITAN, así que los nueve brazos heredan las del modelo: 3.072 MiB de VRAM y 9.216 MiB de RAM (5.120 en CN). Según #482, esa VRAM cubre la memoria del proceso medida en M0, M1, M3 y K = 4 con bloques de 512, de hasta 2.808 MiB. `mars_titan_m1_k4_first_read` usa la misma receta del lector que K = 4 y no se ha medido por separado, igual que M2 y K = 2. B6 y B6 con clave constante no tienen lector ni receta propia de precisión: predicen con el núcleo congelado y la receta de Titans-MAC del padre, cuya precisión `fp32_strict` aplica la inferencia y registra la identidad de la ventana como en los lectores. Su memoria con los bloques de 1.024 de #482 no se ha medido, por el mismo motivo que su caudal, y se medirá con las vistas v3.1 antes de lanzar.

## Pendiente

- Ejecutar ajustes y comparaciones en la campaña A. La edición histórica desde 2000 y sus objetivos ya están verificados, pero el bloqueo de aprendizaje sigue activo.
- Declarar `mars_titan_m1_k4_first_read`, `mars_titan_b6` y `mars_titan_b6_bias` en la campaña A v2, en su comparación conjunta y en sus etapas de adaptadores y de políticas, con los límites que den `count_stage` y `check` (#437). La etapa de adaptadores de la v2 ya recorre la cadena de los lectores de MARS-TITAN, así que `first_read`, que tiene lector, entraría sin código nuevo.
- Publicar la cadena de B6. La etapa de adaptadores deja B6 en espera porque no tiene puntos de adaptación, y sin su cadena la comprobación previa de las políticas echaría en falta sus selecciones. Sería trivial como la de Ridge y XGBoost (#483): el ajuste elegido en k-1 trasladado a k con `carry_correction`, que también tendría que predecir la validación de k.
- Decidir si entra `mars_titan_m1_k4_first_read`, que cuesta lo mismo que un lector con K = 4, y repetir la medida de caudal y de memoria de los lectores, de B6 y del núcleo con el código actual sobre las vistas v3.1, en varias ventanas y con repeticiones, antes de fijar el calendario.
- Los seis componentes declarados sin conexión quedan [fuera de la campaña A](../research/titans-mac-architecture.md#componentes-que-no-entran-en-la-campaña-a), cada uno con su motivo y su control de descarte.
