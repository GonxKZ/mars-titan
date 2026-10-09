# Entrenador cronológico de la GRU candidata

`training/candidate_run.py` ajusta la [GRU candidata con banco episódico](gru-financial-sessions.md) recorriendo en orden los instantes de decisión de la edición desde 2000. Es un brazo independiente de la comparación, con identidad propia, y no sustituye a Titans-MAC ni a MARS-TITAN. Está implementado y comprobado técnicamente en CPU sin pasos de optimizador. No se ha ejecutado ningún entrenamiento, piloto ni evaluación científica: el bloqueo de aprendizaje sigue vigente y el test de 2024 permanece cerrado.

## Cómo se ajusta el módulo nativo

El cálculo de la candidata vive en `Candidate` (C++ con LibTorch). Se compararon tres formas de ajustarlo:

| Opción | Qué exige | Decisión |
| --- | --- | --- |
| (a) Optimizador de PyTorch sobre el módulo nativo a través del enlace | Que el enlace devuelva tensores de la misma instalación de PyTorch conectados al grafo | Elegida |
| (b) Réplica PyTorch del mismo cálculo con paridad frente al nativo | Duplicar GRU, lectura top-k estable, normalización escalada, refinador y cabeza, y mantener la paridad en cada cambio | Descartada |
| (c) Bucle nativo con `torch::optim` | Llevar a C++ lector de corpus, banco, cola, checkpoints y selección, que hoy están en Python | Descartada |

La opción (a) funciona sin cambiar el código nativo. `_episodic_native` se compila contra la misma instalación de PyTorch, `named_parameters()` devuelve en cada llamada los mismos objetos Python de los 25 tensores registrados y `forward` conserva el grafo cuando el modo con gradiente está activo. Una sonda con las dimensiones reales (precios 64×5, noticias 384, gráficos 512, fundamentales 45 y macro 420), 128 filas, FP32 y 16 episodios obtuvo gradientes finitos y no nulos en los 25 tensores, sin aplicar ningún paso. Las salvedades del C++ también se respetan: el top-k queda fuera de autograd y con un único vecino el gradiente de la consulta es cero.

La opción (b) añadiría una segunda implementación del brazo cuya identidad habría que demostrar equivalente en cada cambio. Las pruebas nativas ya contrastan la GRU con sus ecuaciones explícitas y la consulta con diferencias finitas, así que la réplica no aporta una comprobación que falte. La opción (c) movería a C++ todo el recorrido sin un beneficio medido. Además, sus pasos no pasarían por el gancho global de `torch.optim` y necesitarían el lector nativo de la protección. El coste por fila está en la GRU de ATen y en su backward, que ya son nativos en (a).

Los parámetros solo cambian mediante el optimizador. Las proyecciones fijas del codec (rasgos 256 y claves 128) no son parámetros, así que el contenido del banco no depende de lo aprendido. Solo cambia cómo se consulta y cómo se proyectan sus valores.

## Recorrido y orden de cada evento

El entrenador reutiliza el índice de [observaciones financieras](financial-session-v2.md) y su lector por bloques. Dentro de un evento el orden es fijo:

1. Se toma la instantánea del banco confirmado antes de las etiquetas del evento.
2. Se resuelven las etiquetas que maduran en ese instante contra la mediana realmente emitida. Su error entra en `SessionErrors`, su grafo entra en la pérdida del tramo y, con M1, se prepara su admisión.
3. Si el tramo ya contiene `update_instants` instantes de decisión, o el evento cierra la fase, se calcula la pérdida, se recorta el gradiente y se llama al optimizador. Después se descartan todos los grafos.
4. Se predicen las entradas del instante con los parámetros vigentes y la instantánea del paso 1.
5. Se publican en el banco las admisiones preparadas en el paso 2.

Es el mismo orden que el entrenador de Titans-MAC para pérdida y actualización, y el mismo que `FinancialSession` para el banco: las predicciones de un instante no ven las etiquetas que maduran en ese instante. Las admisiones siguen el orden canónico del ejecutor nativo (maduración, decisión y flujo) y reciben IDs consecutivos desde `seen + 1`. Las claves y valores admitidos son los bytes del codec fijo CPU de la fila emitida y la etiqueta se guarda en FP64.

La GRU reinicia su estado en cada ventana, así que no hay BPTT entre instantes. `update_instants` solo fija la cadencia de actualización y coincide con `truncation` de Titans-MAC para que ambos brazos actualicen en los mismos eventos. Una etiqueta cuya predicción pertenece a un tramo ya actualizado no tiene grafo y se cuenta en `labels_without_graph`. Con etiquetas de la sesión siguiente vale cero en las pruebas.

El calentamiento no predice, porque la GRU no conserva estado. El banco empieza vacío en cada pasada de ajuste, validación, calibración y evaluación, como pide el [protocolo walk-forward v2](../research/walk-forward-2000.md#estado-y-calentamiento-por-ventana). El índice de observaciones solo contiene etiquetas de decisiones del tramo medido, por lo que hoy no existe un calentamiento episódico con etiquetas anteriores al tramo. Se registra como pendiente.

## Banco, refinamientos y política

| Política | Lectura | Escritura | Parámetros sin gradiente |
| --- | --- | --- | --- |
| M0 | Memoria vacía, el refinador se conserva | Ninguna | `query_*` y `value_*`, fuera del optimizador |
| M1 | Instantánea del banco confirmado | Reservorio causal v2 con etiquetas maduras | Ninguno una vez admitidos dos episodios |

K = 1 es la configuración principal. K = 2 y K = 4 repiten lectura y refinamiento sobre la misma instantánea y no escriben en el banco ni cuentan como actualizaciones de memoria. M2 y M3 no están definidos sobre la geometría 128×256 de este banco y siguen fuera.

## Pérdida y cabeza

La cabeza es la de `Candidate::quantiles`, que la PR #381 reprodujo como `quantile_head_v1` con paridad exacta de salidas y gradientes en FP32 y FP64. La parametrización coincide (mediana libre e incrementos softplus acumulados hacia cada cola), así que el entrenador no cambia la cabeza común. La pérdida es `pinball_loss` de `models/quantile_head.py` sobre los cinco cuantiles y la selección usa el MAE por sesión de la mediana emitida.

La variante `m1_k1_l1_median` declara la salida escalar de retroceso de [#22](https://github.com/GonxKZ/mars-titan/issues/22): pérdida L1 sobre la mediana. Las pruebas comprueban que con ella solo la fila de la mediana de `head_weight` recibe gradiente.

## Roles de parámetros

| Rol | Parámetros |
| --- | --- |
| `encoder` | GRU de precios, proyecciones de noticias, gráficos, fundamentales y macro, fusión y estado inicial |
| `episodic_read` | Consulta y proyección de valores leídos (`query_*`, `value_*`) |
| `refiner` | `update_*` y `step_logit` |
| `head` | `head_weight` y `head_bias` |

El optimizador debe cubrir exactamente los roles de la política. Con M0, la lectura episódica queda fuera porque nunca recibe gradiente.

## Selección y presupuesto

`load_recipe` lee [`configs/candidate/chronological-training.json`](../../configs/candidate/chronological-training.json) y el protocolo que referencia, y rechaza la receta si épocas, paciencia, mejora mínima o modo de parada difieren de `stopping_rule(protocol)`. La receta propuesta recorre 30 épocas con `fixed_budget`, paciencia 5 como diagnóstico y `min_delta` 1e-5, y conserva el mejor estado. Tasa de aprendizaje, recorte, lote de 128 activos y cadencia de 8 instantes coinciden con la receta de Titans-MAC. Las semillas son las del protocolo. Las de rasgos y claves del codec quedan fijas (43 y 44) para que el banco tenga la misma geometría en todas las semillas.

La receta se identifica con el brazo `gru_episodic` de la [comparación walk-forward declarada](../../configs/evaluation/historical-masked-2000-comparison.json), que corresponde a la variante principal `m1_k1`. Las variantes `m0_k1`, `m1_k2`, `m1_k4` y `m1_k1_l1_median` son controles propios de este brazo y esa comparación todavía no las declara.

La receta de Titans-MAC declara hoy `min_delta` 0, distinto del protocolo v2. No se ha cambiado aquí, pero la comparación emparejada exige igualarlo antes de entrenar ([#363](https://github.com/GonxKZ/mars-titan/issues/363)).

## Checkpoints y reanudación

`save_training_state` conserva los dos estados de recuperación más recientes y el mejor estado aparte. Un estado de recuperación incluye parámetros, huella de parámetros, optimizador, RNG, cursor, selección, historial y, dentro de un ajuste, la cola pendiente con sus claves y valores del codec, el banco con su RNG de reservorio, la admisión preparada del evento, los errores y los contadores.

La barrera está justo después de una actualización, sin grafos vivos. En ese punto ya se han resuelto las etiquetas del evento, pero sus predicciones y su admisión siguen pendientes. Por eso el estado guarda el banco previo y la admisión preparada por separado, y al reanudar se predice con el banco previo antes de publicar la admisión. La validación no se reanuda a mitad, porque es determinista y se repite desde su inicio. Los parámetros recuperados se copian en el módulo vigente, de modo que el optimizador conserva sus referencias, y se comprueba su huella.

`run` llama a `require_learning_allowed()` antes de abrir fuentes o crear salidas, con cualquier optimizador, y se detiene con `LearningHoldError` mientras la protección local no declare `training_allowed` verdadero. El entrenador no usa `torch::optim` en C++, así que no necesita el lector nativo de la protección. `evaluate` y `restore_selected` son inferencia congelada y no se bloquean.

## Predicciones

Tras la última época se carga el mejor estado y se escriben `validation-predictions.parquet` y, si se declaran sus fases, `calibration-predictions.parquet` y `evaluation-predictions.parquet`. El esquema es el de las referencias neuronales (`sample_id`, `asset_id`, `market`, `prediction_at`, `target`, `prediction` y `zero`) más `QUANTILE_COLUMNS`. `prediction` y `quantile_0500` tienen los mismos bits y `ForecastPanel.from_arrow` lee la tabla con sus cinco niveles. Solo se escriben las filas cuya etiqueta madura dentro de la fase.

`restore_selected` carga el mejor estado de una ejecución completa en un adaptador con el mismo contrato y comprueba la huella. Ese adaptador, congelado, sirve a `FrozenCandidateConsumer`.

## Paridad con la inferencia congelada

La GRU de ATen en CPU no produce los mismos bits cuando sus pesos requieren gradiente, aunque el modo sin gradiente esté activo. La diferencia medida fue de hasta 2,2e-16 en FP64 sobre el corpus técnico. Por eso la evaluación desactiva temporalmente `requires_grad` en los parámetros y lo restaura al terminar. Con ello la validación del entrenador coincide bit a bit con una repetición independiente que usa `FrozenCandidateConsumer`, su codec y un `CandidateEpisodeBank` propio, en M0 y M1 con K = 1 y 2. Las pruebas existentes de `FinancialSession` ya comprueban que la sesión emite lo mismo que ese consumidor con la instantánea previa a las etiquetas. No se ejecutó la sesión completa sobre el corpus técnico porque su verificador de prefijos necesita una edición materializada cuyo fixture estima objetivos residuales.

`FrozenCandidateConsumer`, `FinancialSession`, el banco y el código nativo no se han modificado.

## Coste y memoria medidos

Una sonda en CPU con dos hilos, FP32, dimensiones reales, 128 filas, 16 episodios y K = 1 midió 0,055 s de forward con grafo, de 0,074 a 0,079 s de backward y 264 KB de tensores guardados por fila (almacenamientos únicos sin parámetros). Con la receta de #389 cada predicción conserva su grafo hasta el paso del tramo, que llega tras `update_instants` instantes. El pico crece con los activos de cada instante multiplicados por esos 8 instantes.

## Memoria del tramo: acumulación y recomputación

La receta declara dos opciones, desactivadas por defecto para conservar la paridad exacta con #389:

| Opción | Qué hace | Identidad y gradiente |
| --- | --- | --- |
| `accumulation_rows` | Al madurar las etiquetas de un evento calcula el backward de su pérdida sumada, por grupos de bloques de predicción completos de hasta ese número de filas (como mínimo `block_rows`). El paso del tramo divide el gradiente por el total de etiquetas, recorta y llama una vez al optimizador | Cambia `loss_reduction`. Solo cambian el orden de las sumas y la división final |
| `recompute` | Ejecuta cada bloque de ajuste con `torch.utils.checkpoint` sin reentrada. Guarda sus entradas y repite el forward durante el backward con los mismos parámetros, la misma instantánea y el mismo K | Cambia la receta. En las comprobaciones el gradiente coincide bit a bit con #389 |

En este entrenador el forward ocurre al predecir y la pérdida solo existe cuando madura la etiqueta, en un evento posterior. Por eso la acumulación agrupa al madurar los bloques formados al predecir, y el forward por bloques de `block_rows` filas no cambia. Cada grupo es una ponderación de sus filas sobre el total del tramo, que solo se conoce en el paso, así que la división se aplica una vez al gradiente acumulado. Un grupo nunca reparte un bloque, para no recorrer su grafo dos veces. Las predicciones, la instantánea de cada instante y la admisión al final del evento no cambian, y las pruebas lo comprueban sobre el registro de auditoría.

Un bloque solo libera su grafo cuando han madurado todas sus filas. Mientras tanto se conserva con `retain_graph`, de modo que una etiqueta que madure más tarde sigue entrando en la pérdida. La contrapartida es que una fila sin etiqueta (varianza de mercado nula, sesión siguiente ausente o historia insuficiente) retiene su bloque completo hasta el paso del tramo. En el corpus técnico una sola fila sin etiqueta mantuvo vivo un bloque de 128 filas. Liberar antes el grafo exigiría saber al predecir si la etiqueta llegará, y eso depende de datos posteriores. La recomputación evita el problema, porque un bloque retenido solo conserva sus entradas.

### Medidas en CPU

[`candidate_memory_check.py`](../../tests/training/candidate_memory_check.py) recorre una época completa sin pasos sobre el corpus técnico sintético con dimensiones reales (precios 64×5, noticias 384, gráficos 512, 45 fundamentales y 420 macro), FP32, `update_instants` 8, `block_rows` 128 y un banco de 1.024 episodios. El pico es el máximo neto del asignador de ATen en CPU durante la ejecución, leído de los eventos de memoria del perfilador de PyTorch. Incluye activaciones, entradas, recomputaciones, gradientes y temporales, pero no la memoria de Arrow ni de NumPy.

| Configuración | 128 activos | 256 activos | Pendiente por activo |
| --- | --- | --- | --- |
| Tramo completo (#389) | 293,5 MB | 564,5 MB | 2,12 MB |
| Acumulación con 128 o 1.024 filas | 88,8 MB | 122,8 MB | 0,27 MB |
| Recomputación | 74,1 MB | 82,4 MB | 65 KB |
| Acumulación y recomputación | 59,3 MB | 60,4 MB | 8 KB |

Los picos de la acumulación incluyen el bloque retenido por la fila sin etiqueta del corpus. El tamaño de grupo no cambia el pico, porque lo determinan los bloques vivos del instante. Frente al tramo completo, la mayor diferencia absoluta del gradiente fue 6,0e-8 con la acumulación, sobre un gradiente máximo de 0,1. Con la recomputación los gradientes coincidieron bit a bit en ambos tamaños.

El coste de la recomputación se midió sobre un bloque de 128 filas, con la mediana de cinco repeticiones: 0,055 s de forward y backward sin recomputar frente a 0,095 s con ella, y 0,021 s de forward sin grafo. Una medición anterior dio 0,068 s frente a 0,093 s. Ambas se hicieron con otros procesos en curso, así que el sobrecoste del ajuste queda entre un 36 % y un 74 %. Los tiempos de la época completa (de 16 a 40 s según el tamaño) incluyen validación, predicciones y lectura, y su dispersión no permite atribuir diferencias.

### Extrapolación a la GPU

Con N filas por evento, las medidas anteriores dan aproximadamente:

| Configuración | Memoria en CPU | N = 5.676 |
| --- | --- | --- |
| Tramo completo | 23 MB + 2,12 MB × N | 12,0 GB |
| Acumulación sin bloques retenidos | 21 MB + 264 KB × N | 1,5 GB |
| Recomputación | 66 MB + 65 KB × N | 0,43 GB |
| Acumulación y recomputación | 58 MB + 8 KB × N | 0,11 GB |

N = 5.676 es una cota superior, porque cada evento contiene un solo mercado. Es una estimación hasta medir en `cuda:0`. La GRU de cuDNN guarda sus propios búferes, el contexto CUDA y los espacios de trabajo de cuBLAS y cuDNN no se cuentan como tensores y el asignador con caché fragmenta la memoria. Con esas salvedades, el tramo completo no cabe en los 8 GB de la RTX 4070 Max-Q, la acumulación cabe si pocos bloques quedan retenidos y la recomputación cabe con margen cualquiera que sea el patrón de etiquetas.

## Comprobaciones

Las pruebas de `tests/training/test_candidate_run.py` usan el corpus técnico cronológico con etiquetas declaradas, sin estimar objetivos, y un optimizador que solo registra llamadas y gradientes. Cubren:

- Gradientes finitos en cada parámetro de la política, con algún gradiente no nulo por parámetro, para M1 con K = 1, 2 y 4 y para M0, donde la lectura no recibe gradiente.
- Paridad bit a bit de la validación con la repetición mediante `FrozenCandidateConsumer`, incluidos los cuantiles escritos en Parquet.
- Etiquetas solo tras madurar, cada predicción leyendo exactamente las admisiones anteriores a su instante, orden e IDs de admisión del ejecutor y ningún grafo perdido con etiquetas de la sesión siguiente.
- Predicciones, admisiones y gradientes pasados iguales cuando se alteran entradas, precios y etiquetas posteriores a un corte.
- Reanudación con la misma secuencia de predicciones, admisiones, actualizaciones, gradientes, historial y Parquet que una ejecución continua, y recuperación de un estado construido a mano con banco ya sustituido, admisión preparada y cola.
- Copia de los parámetros guardados sobre valores escritos a mano, recorte de gradiente y contrato del Parquet con `ForecastPanel`.
- Gradiente de la cabeza con pinball en las cinco filas de `head_weight` y solo en la de la mediana con la variante L1, y descarte contado de una etiqueta cuyo grafo ya no existe.
- Escritura de las predicciones de validación, calibración y evaluación con el fixture temporal histórico, cada una dentro de su partición.
- Rechazo con la protección temporal bloqueada antes de crear salidas, con el optimizador de registro y con AdamW, y rechazo de políticas estrictas, vistas de otro corpus, particiones mal declaradas, parámetros congelados, ámbitos vacíos y cambios de receta al reanudar.

`tests/training/test_candidate_accumulation.py` comprueba las dos opciones de memoria:

- Gradientes de la acumulación con 2, 4 y 1.024 filas iguales a los del tramo completo en FP64 (rtol 1e-12, atol 1e-15, con una diferencia máxima observada de 4,2e-17) y en FP32 (rtol 1e-5, atol 1e-7, con 7,5e-9), y gradientes de la recomputación iguales bit a bit. Las predicciones, etiquetas, admisiones y pasos del registro coinciden en todos los casos.
- Un bloque cuyas filas maduran en eventos distintos conserva su grafo hasta la última, con y sin recomputación, y los grupos nunca reparten un bloque.
- Las filas de un bloque no ven las demás: alterar otras filas no cambia ni la salida ni el gradiente de la primera, con la misma instantánea para todas.
- La acumulación reduce el pico de tensores guardados y la recomputación deja fuera de los bloques solo la pila y la pérdida.
- Reanudación de una ejecución con acumulación, con y sin recomputación, igual a la continua.

El recuento de pruebas, las mutaciones dirigidas y las versiones están en el [recibo técnico](../../reports/engineering/candidate-chronological-trainer-20261009.json) y en el [recibo de memoria](../../reports/engineering/candidate-trainer-memory-20261009.json).

## Pendiente

- CUDA. La comprobación [`cuda_candidate_run_check.py`](../../tests/training/cuda_candidate_run_check.py) compara en CPU y `cuda:0` un ajuste completo sin pasos y su validación, en FP32 y FP64 con K = 1 y 4. Cuando la GPU quede libre, desde `native/`:

  ```bash
  cmake --preset native-candidate-cuda -B ../build/native/candidate-trainer-cuda \
    -DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON -DMARS_TITAN_BUILD_SIMULATION=ON \
    -DMARS_TITAN_TORCH_ENVIRONMENT=<entorno uv con PyTorch CUDA>
  cmake --build ../build/native/candidate-trainer-cuda -j 2
  cd .. && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
    MARS_TITAN_EPISODIC_NATIVE=$PWD/build/native/candidate-trainer-cuda/_episodic_native.cpython-312-x86_64-linux-gnu.so \
    MARS_TITAN_CANDIDATE_RUN_CHECK_REPORT=$PWD/candidate-trainer-cuda.json \
    UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA> \
    uv run --no-sync python -m pytest -q tests/training/cuda_candidate_run_check.py
  ```

  Después hay que perfilar en `cuda:0` el recorrido completo, incluidas las sincronizaciones de las comprobaciones de finitud del nativo y la memoria de los grafos del tramo.
- Calentamiento episódico con etiquetas anteriores al tramo medido, que necesita indexarlas en las observaciones financieras.
- Predicciones de calibración y evaluación sobre vistas temporales reales. Solo se han comprobado con el fixture temporal histórico.
- Presupuesto definitivo y alternativa de reentrenamiento por ventana en [#363](https://github.com/GonxKZ/mars-titan/issues/363), con el caudal medido.
- Memoria en `cuda:0`. Con el enlace CUDA compilado como arriba, desde la raíz:

  ```bash
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
    MARS_TITAN_EPISODIC_NATIVE=$PWD/build/native/candidate-trainer-cuda/_episodic_native.cpython-312-x86_64-linux-gnu.so \
    MARS_TITAN_CANDIDATE_MEMORY_CHECK_DEVICE=cuda:0 \
    MARS_TITAN_CANDIDATE_MEMORY_CHECK_ASSETS=128,256,512 \
    MARS_TITAN_CANDIDATE_MEMORY_CHECK_REPORT=$PWD/candidate-memory-cuda.json \
    UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA> \
    uv run --no-sync python -m pytest -q tests/training/candidate_memory_check.py
  ```

  Con esa medida hay que decidir qué opción usa la campaña. La recomputación es la que acota la memoria con cualquier patrón de etiquetas sin cambiar el gradiente en las comprobaciones de CPU.
