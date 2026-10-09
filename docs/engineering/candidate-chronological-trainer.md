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

La opción (b) añadiría una segunda implementación del brazo cuya identidad habría que demostrar equivalente en cada cambio. Las pruebas nativas ya contrastan la GRU con sus ecuaciones explícitas y la consulta con diferencias finitas, así que la réplica no aporta una comprobación que falte. La opción (c) movería a C++ todo el recorrido sin un beneficio medido, y sus pasos no pasarían por el gancho de pruebas ni por `learning_blocked()`. El coste por fila está en la GRU de ATen y en su backward, que ya son nativos en (a).

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

Con un optimizador real de PyTorch, `run` se niega a empezar antes de crear salidas mientras la protección local declare `training_allowed=false`. Usa `learning_blocked()`, porque la función que lanza el error común (#382) todavía no está en `develop`.

## Predicciones

Tras la última época se carga el mejor estado y se escriben `validation-predictions.parquet` y, si se declaran sus fases, `calibration-predictions.parquet` y `evaluation-predictions.parquet`. El esquema es el de las referencias neuronales (`sample_id`, `asset_id`, `market`, `prediction_at`, `target`, `prediction` y `zero`) más `QUANTILE_COLUMNS`. `prediction` y `quantile_0500` tienen los mismos bits y `ForecastPanel.from_arrow` lee la tabla con sus cinco niveles. Solo se escriben las filas cuya etiqueta madura dentro de la fase.

`restore_selected` carga el mejor estado de una ejecución completa en un adaptador con el mismo contrato y comprueba la huella. Ese adaptador, congelado, sirve a `FrozenCandidateConsumer`.

## Paridad con la inferencia congelada

La GRU de ATen en CPU no produce los mismos bits cuando sus pesos requieren gradiente, aunque el modo sin gradiente esté activo. La diferencia medida fue de hasta 2,2e-16 en FP64 sobre el corpus técnico. Por eso la evaluación desactiva temporalmente `requires_grad` en los parámetros y lo restaura al terminar. Con ello la validación del entrenador coincide bit a bit con una repetición independiente que usa `FrozenCandidateConsumer`, su codec y un `CandidateEpisodeBank` propio, en M0 y M1 con K = 1 y 2. Las pruebas existentes de `FinancialSession` ya comprueban que la sesión emite lo mismo que ese consumidor con la instantánea previa a las etiquetas. No se ejecutó la sesión completa sobre el corpus técnico porque su verificador de prefijos necesita una edición materializada cuyo fixture estima objetivos residuales.

`FrozenCandidateConsumer`, `FinancialSession`, el banco y el código nativo no se han modificado.

## Coste y memoria medidos

Medidas en CPU con dos hilos, FP32, dimensiones reales, 128 filas, 16 episodios y K = 1, con la codificación histórica y otros procesos en curso:

| Medida | Valor |
| --- | --- |
| Forward con grafo | 0,055 s |
| Backward | 0,074 a 0,079 s |
| Tensores guardados para el backward, almacenamientos únicos sin parámetros | 264 KB por fila |

Con 128 activos y 8 instantes por tramo, los grafos vivos ocupan unos 270 MB. Con el universo completo (unos 5.000 activos) serían unos 10,6 GB, que no caben en los 8 GB de la GPU. Antes de escalar habría que reducir la cadencia o recalcular cada predicción al madurar su etiqueta. Son cifras de una sonda, no un perfil del recorrido completo.

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
- Rechazo con el bloqueo vigente antes de crear salidas, con políticas estrictas, vistas de otro corpus, particiones mal declaradas, parámetros congelados, ámbitos vacíos y cambios de receta al reanudar.

El recuento de pruebas, las mutaciones dirigidas y las versiones están en el [recibo técnico](../../reports/engineering/candidate-chronological-trainer-20261009.json).

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
- Sustituir la consulta de `learning_blocked()` por la función común de #382 cuando se integre.
