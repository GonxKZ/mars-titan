# MARS-TITAN con ampliaciones sobre Titans-MAC

Este documento describe la implementación de la variante MARS-TITAN con ampliaciones: el constructor de combinaciones desde su [declaración](../../configs/titans/mars-titan-extensions.json), la corrección asociativa B6 dentro de `FinancialSession`, el modo de K con episodios fijos, el entrenador cronológico del lector episódico, la entrada por ventana walk-forward con su traslado y el registro en la campaña con máscaras. Nada de esto se ha ejecutado con datos. El bloqueo de aprendizaje sigue vigente y las pruebas recorren los bucles hasta el paso del optimizador con un registrador que no modifica pesos.

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

El recorrido cronológico por ventanas todavía no emite B6. La elección de η y λ en desarrollo sigue pendiente.

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

## Registro en la campaña

`campaign_plan` acepta una sección opcional `mars_titan` con la receta del lector, la combinación de cada brazo, `pending_arms` con el motivo de los brazos sin definición, el brazo padre y la semilla de búsqueda. El padre debe ser el brazo `mac_online` de la sección `titans_mac`, con la misma semilla de búsqueda y semillas que cubran las de cada brazo. Cada búsqueda depende de las búsquedas del padre en su ventana y cada finalista, además de sus búsquedas, del finalista del padre con su semilla. `masked_campaign` resuelve ese padre y lo entrega al ejecutor en `JobRun.parent`, y la identidad del trabajo incluye su recibo. El finalista elige ganador solo entre sus propias búsquedas.

Las configuraciones A y B no declaran todavía la sección, igual que la GRU episódica. La campaña elegida, la A, la incluye en su declaración ampliada. `mars_titan_m3` ya tiene productor. Con los seis brazos (M0, M1, M2, M3, M1 con K = 2 y M1 con K = 4) añadiría 1.080 ajustes en A y 408 ajustes con 504 traslados en B. La [declaración preparada](../../configs/baselines/historical-masked-campaign-extensions.json) incluye los seis brazos, y la medida de caudal estima las escalas de M3 con la regla de la campaña sobre el tramo de entrenamiento de la ventana medida. Antes hay que medir memoria y caudal del lector en `cuda:0`.

`tests/training/test_mars_titan_campaign.py` comprueba el plan sobre la comparación declarada y ejecuta una campaña B reducida sobre US con `titans_mac_online`, `mars_titan_m1` y `mars_titan_m3` con los ejecutores reales en CPU hasta la tercera ventana: un ajuste por caso y dos traslados, con las mismas filas que Titans-MAC, el padre elegido en la identidad y los recibos de ventana publicados. M3 estima sus escalas en la ventana ajustada y traslada las del ancla. Con las cuatro secciones declaradas, los 23 brazos de la comparación tienen productor.

## Comprobaciones

| Archivo | Qué comprueba |
| --- | --- |
| `tests/memory/test_mars_titan_variant.py` | Declaración, identidades, rechazos con motivo, consumidor y paridad de sesión |
| `tests/memory/test_financial_session_associative.py` | Corrección B6 frente a un oráculo, causalidad, control constante y recuperación |
| `tests/models/titans/test_episodic_readout_fixed_episodes.py` | Modo `first_read`, paridad K = 1 y gradcheck FP64 |
| `tests/training/test_mars_titan_run.py` | Receta, roles, rechazos, bucle hasta el paso sin cambiar pesos, orden del evento, causalidad, aislamiento del estado rápido del padre, reanudación y protección |
| `tests/training/test_mars_titan_walk_forward.py` | Ventana sobre el padre, estado compuesto, filas, reanudación y rechazos previos a abrir fuentes |
| `tests/training/test_mars_titan_campaign.py` | Plan, dependencias, pendientes con motivo y campaña reducida con ejecutores reales |
| `tests/memory/test_write_scores.py`, `tests/memory/test_write_policy_m3.py` y `tests/memory/test_financial_session_m3.py` | [Escritura M3](m3-write-policy.md): componentes, escalas, banco y sesión |

El aislamiento se comprueba registrando cada llamada a `prepare` del padre: con M0, M1, M2 y M3 la secuencia de predicciones del núcleo y de estados rápidos exportados coincide bit a bit con la de `ChronologicalInference` de Titans-MAC. `tests/memory/test_mars_titan_session_parity.py` entrega los mismos eventos a `FinancialSession` y al recorrido cronológico del lector, con un banco de capacidad 4 que recibe 16 etiquetas: las 20 emisiones coinciden bit a bit en M1, M2, M3 y K = 2 con sus dos modos de selección, igual que los IDs admitidos. En M3 los flujos tienen noticias, fundamentales y últimos precios distintos, así que los tres componentes varían.

Se aplicaron 16 mutaciones dirigidas en copias aisladas, comprobando que las pruebas importaban la copia. Catorce hicieron fallar las pruebas: admitir antes de predecir, M0 con lectura, el orden de admisión, el peso de cada bloque en la pérdida del tramo, la identidad con todo apagado, B6 con banco, la selección del consumidor, la semilla y el lector elegidos del padre, el codec trasladado, el ganador del finalista, el padre ausente, las dependencias de búsqueda y el motivo pendiente. El peso de bloque solo se detectó tras añadir la prueba que compara bloques de una fila con bloques de todo el instante. Las otras dos eran equivalentes. Invertir el signo del error M2 no cambia nada observable porque el banco solo usa su valor absoluto, y la comprobación explícita de K con episodios fijos repetía la regla `requires` de la declaración, así que se retiró.

## Comprobaciones CUDA

`tests/training/cuda_mars_titan_run_check.py` compara en CPU y `cuda:0` un ajuste completo sin pasos y su validación, en FP32 y FP64, con M1 y K = 1, K = 4 por paso y K = 4 con episodios fijos, y con M3 y K = 1 y K = 4 con episodios fijos. Los casos M3 se describen en la [guía de M3](m3-write-policy.md#comprobaciones-cuda). Sus diez casos pasan en `cuda:0` desde el 9 de octubre ([recibo](../../reports/engineering/cuda-checks-20261009/mars-titan-readout-cuda.json)), una vez inicializada CUDA antes de reiniciar el pico de memoria. La orden, desde la raíz del repositorio:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  MARS_TITAN_EPISODIC_NATIVE=<enlace episódico nativo compilado> \
  MARS_TITAN_MARS_RUN_CHECK_REPORT=$PWD/mars-titan-readout-cuda.json \
  UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA> \
  uv run --no-sync python -m pytest -q tests/training/cuda_mars_titan_run_check.py
```

`tests/models/titans/cuda_episodic_readout_check.py` no cubre todavía el modo `first_read`, y la sesión con B6 no tiene comprobación CUDA porque A vive en CPU por diseño.

## Pendiente

- Ejecutar ajustes y comparaciones en la campaña A. La edición histórica desde 2000 y sus objetivos ya están verificados, pero el bloqueo de aprendizaje sigue activo.
- Copiar la sección `mars_titan` a la configuración de A. El caudal del lector en `cuda:0` sigue sin medir.
- Emitir B6 en el recorrido por ventanas y elegir η y λ en desarrollo.
- Conectar los seis componentes declarados sin conexión, cada uno con su control de descarte.
