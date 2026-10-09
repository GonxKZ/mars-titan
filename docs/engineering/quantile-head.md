# Cabeza común de cuantiles `quantile_head_v1`

Implementación de la opción B adoptada el 9 de octubre de 2026 en [#22](https://github.com/GonxKZ/mars-titan/issues/22), con la justificación de la [propuesta](../research/quantile-head-decision.md). Este documento describe el código y sus comprobaciones técnicas. No se ha entrenado ni evaluado ningún modelo con esta cabeza y el bloqueo de aprendizaje sigue vigente.

## Especificación

El módulo `src/mars_titan/models/quantile_head.py` contiene la única implementación PyTorch.

| Elemento | Definición |
| --- | --- |
| Niveles | 0,025, 0,1, 0,5, 0,9 y 0,975 |
| Parametrización | `nn.Linear(D, 5)` produce `r`. La mediana es `r_2`. Los incrementos son `softplus(r_j)` con `beta = 1` y umbral 20 |
| Cuantiles | `q_2 = r_2`, `q_1 = q_2 − sp(r_1)`, `q_0 = q_1 − sp(r_0)`, `q_3 = q_2 + sp(r_3)`, `q_4 = q_3 + sp(r_4)` |
| Pérdida | Media de `max(τ u, (τ − 1) u)` sobre los cinco niveles y las filas, con `u = y − q_τ` y pesos iguales |
| Predicción puntual | La mediana `q_2` |
| Columnas de las tablas | `quantile_0025`, `quantile_0100`, `quantile_0500`, `quantile_0900` y `quantile_0975` |

La parametrización es la de `Candidate::quantiles` en `native/src/candidate.cpp`, con el mismo orden de operaciones. El orden no decreciente se cumple por construcción y también en coma flotante, porque restar o sumar un valor no negativo con redondeo al más cercano nunca deja el resultado al otro lado del operando. Un incremento muy negativo puede producir cuantiles iguales, pero no cruzados.

`QuantileHead` hereda de `nn.Linear`. Conserva `weight`, `bias`, su inicialización y los nombres del estado (`head.weight` y `head.bias`), que corresponden a `head_weight` y `head_bias` del candidato nativo.

En un residuo nulo la pérdida usa el subgradiente medio `τ − 1/2`, que es el que da `torch.maximum` al repartir el empate. Con esa convención el término de la mediana vale exactamente `|u| / 2` y su gradiente es la mitad del de `l1_loss`, también en el cero. Como la media incluye cinco niveles, la mediana recibe una décima parte del gradiente que tendría con una pérdida L1 escalar, como anticipaba la propuesta. Con AdamW esa escala común se compensa en gran parte, pero el reparto entre la mediana y las colas sobre el tronco compartido sí cambia y es lo que mide el control de la cabeza.

## Integración

**Referencias neuronales.** `MultimodalReference(..., head="quantile_head_v1")` devuelve `[lote, 5]`. La cabeza se construye la última, así que con la misma semilla el tronco recibe los mismos pesos iniciales que la variante escalar. La configuración solo añade `head` en la variante de cuantiles.

**Runner de referencias.** Un caso con `head: "quantile_head_v1"` debe declarar `loss: "pinball"` y una arquitectura científica. Conserva `huber_delta` para compartir el esquema del caso. La identidad añade el campo `output_head` con el contrato completo y la huella de `models/quantile_head.py`. El error, la selección por MAE de sesión y el resumen de ajuste usan la mediana. Las tablas por fila añaden las cinco columnas, y la mediana se guarda con los mismos bits en `prediction` y en `quantile_0500`. `ForecastPanel.from_arrow(table, quantile_columns=QUANTILE_COLUMNS, levels=LEVELS)` las acepta y `score_sessions` informa `point_equals_median = True`. Un padre escalar no puede inicializar una continuación de cuantiles.

**Titans-MAC.** `FinancialConfig(head="quantile_head_v1")` cambia `output` por `quantile_head_v1` y añade `output_head` a la identidad. El predictor construye la referencia escalar y la fusión con máscaras como antes, y crea la cabeza de cuantiles al final. Así las cuatro variantes conservan el tronco de la variante escalar con la misma semilla. `prepare` devuelve la mediana en `point_predictions` y los cinco niveles en `quantiles`. `copy_paired_parameters` y la carga de estados rechazan mezclar cabezas.

**Entrenador cronológico.** La receta admite `loss: "pinball"` solo con un predictor que tenga la cabeza de cuantiles, y la cabeza solo con esa pérdida. El grafo de cada etiqueta pendiente guarda los cinco niveles y el tramo aplica la pinball media sobre las etiquetas maduras. El error emitido, la validación y la selección siguen usando la mediana. La receta declarada `configs/titans/chronological-training-quantile.json` solo difiere de la escalar en la cabeza y la pérdida, y conserva el estado `propuesta_sin_ejecutar`.

**Lectura episódica y consumidor congelado.** `apply_episodic_readout` acepta `QuantileHead`, comprueba la forma `[flujos, 5]` y la finitud, y devuelve la mediana con los cuantiles en `EpisodicPrediction.quantiles`. La ruta sin ampliación reenvía los cuantiles preparados por el núcleo. Una cabeza que no corresponde a la salida del núcleo se rechaza, igual que una `nn.Linear(D, 5)` sin ordenar. `FrozenFinancialConsumer` admite el módulo de la cabeza en su firma de ejecución y expone `quantiles` en `FrozenPreparation`.

Las rutas escalares no cambian. En el mismo proceso, el código de `origin/develop` y el de esta rama producen los mismos bytes de estado, salidas y configuración en 20 combinaciones de `MultimodalReference` (cinco familias, dos fusiones y una o dos capas) y la misma identidad, huella de parámetros, predicción, estado de trabajo y estado adicional en las cuatro variantes de `FinancialPredictor` con y sin máscaras. Las huellas de código de las identidades sí cambian, como con cualquier modificación de esos archivos.

## Control de la cabeza

`configs/baselines/quantile-head-control-us.json` declara el control de la decisión sin ejecutarlo. `reference_design.head_control_cases` lo convierte en doce casos del Transformer compacto: los índices de diseño 0 y 10 y las semillas 42, 43 y 44 de la búsqueda histórica de US, cada uno con salida escalar L1 y con `quantile_head_v1`. Los dos brazos de cada par comparten arquitectura, tasa de aprendizaje, semilla, presupuesto fijo de 30 épocas y selección por MAE de sesión. Solo cambian la cabeza y la pérdida.

El contraste es `delta(scalar_l1, quantile_head_v1)` del MAE por sesión de la mediana en la partición de validación, con un intervalo simultáneo del 95 % por bootstrap circular por bloques y un contraste por índice de diseño. Si algún intervalo excluye el cero en contra de la cabeza, la comparación principal pasa a la opción A. Usar validación y no evaluación es una concreción de esta rama: el factor se fija antes de la comparación principal y no debe elegirse con los años que esta informa. La concreción queda pendiente de revisión. La función rechaza cualquier cambio del contraste, la parada, la política de entrada o el estado declarado.

## Comprobaciones

Las pruebas usan CPU con `CUDA_VISIBLE_DEVICES=-1` y no aplican pasos de optimizador. Los recorridos de los runners usan sustitutos que registran gradientes sin modificar pesos.

| Archivo | Qué comprueba |
| --- | --- |
| `tests/models/test_quantile_head.py` | Fórmula del candidato bit a bit en FP32 y FP64, ausencia de cruces con valores libres hasta 1e30, efecto de cada valor libre, pinball calculada a mano y frente a scikit-learn y al panel, gradiente analítico con empates, Jacobiano de softplus, `gradcheck`, relación exacta con L1, invariancias de traslación, escala y permutación |
| `tests/models/test_quantile_head_native.py` | Salidas y gradientes idénticos (`rtol = atol = 0`) a `Candidate::quantiles` con los mismos pesos en FP32 y FP64, y la cabeza aplicada al estado refinado del candidato completo |
| `tests/models/test_quantile_reference.py` | Tronco idéntico al escalar en cinco familias y dos fusiones, mediana igual a la salida escalar con los mismos pesos, ruta escalar por defecto idéntica a la explícita, gradientes y rechazo de cabezas desconocidas |
| `tests/training/test_quantile_reference_run.py` | Identidad nueva, gradiente registrado igual al de la pinball sobre el primer lote, tablas aceptadas por el panel, identidad y columnas escalares sin cambios y fallos tempranos |
| `tests/models/titans/test_financial_quantiles.py` | Tronco y estados de Titans iguales a los escalares, identidad, gradientes, lectura episódica, consumidor congelado y rechazos |
| `tests/training/test_financial_run_quantiles.py` | Pinball sobre `[etiquetas, 5]` en cada tramo, mediana emitida igual a la del consumidor congelado, emparejamiento de cabeza y pérdida y receta declarada |
| `tests/training/test_quantile_head_control.py` | Pares que solo difieren en salida y pérdida, coherencia con la búsqueda histórica, validación del runner y rechazo de cambios del contraste |

La paridad nativa usa el enlace `_episodic_native` compilado en CPU con el candidato, según [la entrada histórica de la GRU](../../native/candidate_historical.md), y se ejecuta con `MARS_TITAN_EPISODIC_NATIVE` apuntando a ese archivo. Sin la variable la prueba se omite.

La mutación dirigida introdujo 26 defectos, uno cada vez, y ejecutó las pruebas del área afectada. Incluyen cambiar el índice o la función de un incremento, mover la mediana, invertir el residuo, sumar los niveles, usar el subgradiente `τ` en el cero, ajustar el runner con L1 sobre la mediana, invertir las columnas, perder la huella o el contrato de la identidad, crear la cabeza antes del tronco, perder los cuantiles en la lectura o en el consumidor y entrenar Titans con la mediana. Un primer mutante era equivalente, porque adelantaba la cabeza al `nn.Sequential` cuando las capas de fusión ya estaban creadas, y se sustituyó por uno que consume el generador antes de esas capas. La omisión de la comprobación de cabeza del padre sobrevivió porque la huella de código también la rechazaba con otro mensaje. La prueba se reforzó para exigir el rechazo por la cabeza. Con esos dos cambios mueren los 26 mutantes. Es una selección dirigida, no una campaña completa de mutación.

## Pendiente

- La calibración común al estilo CQR está implementada en `src/mars_titan/calibration/conformal_quantiles.py` y la aplica la [evaluación walk-forward](../research/metrics.md#calibración-común-de-intervalos), ajustada en el tramo de calibración de cada ventana y congelada antes de evaluar. No se ha aplicado a predicciones reales.
- Ejecutar el control de la cabeza cuando se levante el bloqueo y repetir la búsqueda de tasas de aprendizaje.
- Integrar los cuantiles en los consumidores de sesión de `memory/` y en los padres de `posttraining/`, que hoy solo leen la predicción puntual y construyen referencias escalares.
- Comprobaciones CUDA de las rutas nuevas. Las órdenes están en la descripción de la PR.
- Medir el coste de la cabeza en la integración. Se espera pequeño frente a los codificadores, pero no se ha medido.
