# Cabezas auxiliares de horizontes de 5 y 20 sesiones

Este documento describe F5 del estudio de variantes de [#742](https://github.com/GonxKZ/mars-titan/issues/742), que se implementa en [#749](https://github.com/GonxKZ/mars-titan/issues/749). `auxiliary_horizons_v1` es un componente opcional de `FinancialConfig` (`auxiliary_horizons`) que añade cabezas lineales con objetivos residuales acumulados a 5 y 20 sesiones. Solo cambia el ajuste: el objetivo principal, la predicción emitida y la evaluación no cambian.

Está integrada en el predictor y en el recorrido cronológico, con su registro de etiquetas recuperable desde checkpoint. Se ha comprobado con pruebas en CPU sobre fixtures. La comprobación CUDA está pendiente. No se ha entrenado ni evaluado con datos financieros y nada de lo que sigue es una mejora observada.

## Motivación

El objetivo principal es el residuo de la sesión siguiente, una señal con mucho ruido. Un objetivo acumulado a varias sesiones promedia parte de ese ruido y puede orientar a codificadores y fusión hacia rasgos más persistentes. Es la idea del aprendizaje multitarea con objetivos auxiliares de [Caruana (1997)](https://doi.org/10.1023/A:1007379606734). También puede no aportar nada o empeorar el objetivo principal si los horizontes compiten. El control con peso cero permite descartarla.

## Etiquetas auxiliares

Sea o(t) el ordinal de la decisión t entre las decisiones de su mercado en el recorrido. Cada evento con entradas avanza en uno el ordinal de los mercados que aparecen en él. La etiqueta auxiliar de horizonte h de la decisión del flujo f con ordinal s es

$$y^{(h)}_{f,s}=\sum_{j=0}^{h-1}y_{f,s+j},$$

donde y es la etiqueta principal, el residuo de la sesión siguiente con sus α̂ y β̂ estimados hasta su propia decisión. La derivación reutiliza la residualización del objetivo principal y no lee ni escribe columnas nuevas, así que no toca `dataset/` ni los objetivos de la edición v3.1. La suma se calcula con `math.fsum` en orden de ordinal.

La etiqueta auxiliar existe solo si:

- el flujo emitió una decisión en los h ordinales consecutivos. Un hueco del activo rompe todas las ventanas que lo cruzan.
- las h etiquetas principales maduran dentro del tramo. El índice cronológico ya excluye las etiquetas que maduran después del final del tramo, y el registro se reinicia en cada recorrido, así que una ventana nunca cruza la frontera entre ajuste y validación.
- su último sumando madura cuando el ordinal del mercado no supera s + h − 1 + `maturity_slack`. La declaración usa una holgura de 2 decisiones.

Madura con su último sumando, en el mismo evento en que madura la etiqueta principal correspondiente. Antes no existe. La decisión de si existe solo depende de la disponibilidad de las etiquetas y del calendario de decisiones, nunca de sus valores.

## Pérdida y gradiente

Cada cabeza es una proyección lineal del token fusionado x, antes de MAC: a⁽ʰ⁾ = W_h x + c_h. Con A las etiquetas auxiliares entregadas desde la actualización anterior, el ajuste suma

$$\mathcal L_{\text{aux}}=w\,\frac1{\lvert A\rvert}\sum_{(f,s,h)\in A}\frac{\lvert a^{(h)}_{f,s}-y^{(h)}_{f,s}\rvert}{\sqrt h}$$

al objetivo principal del tramo, sin cambiarlo. El divisor √h iguala aproximadamente la escala de horizontes cuyos sumandos fueran independientes. La declaración usa w = 0,5, un valor fijado antes de ejecutar y no ajustado.

Cuando madura una etiqueta auxiliar, el grafo de su decisión ya no existe: el truncamiento lo corta cada pocos instantes y los parámetros han cambiado. Por eso el registro guarda las entradas CPU de cada decisión emitida y `ChronologicalTrainer._auxiliary` vuelve a calcular su token con los parámetros vigentes mediante `FinancialPredictor.fused_tokens`, que coincide bit a bit con el token de la emisión para las mismas entradas y parámetros. El gradiente llega a codificadores, fusión y cabezas auxiliares, pero no a la memoria ni a la atención de MAC, ni a la cabeza principal. La prueba del recorrido lo comprueba con los gradientes registrados.

Las etiquetas de una decisión se entregan juntas cuando ya no puede completarse ninguna más: al madurar la ventana del horizonte más largo, que contiene a las demás, al pasar su plazo o al cerrar el tramo. Así su token se calcula una sola vez para todos sus horizontes. La primera versión calculaba un token por etiqueta, dos por decisión con dos horizontes. A cambio, una etiqueta de 5 sesiones espera a la de 20 de su decisión o, si esta no se completa, al plazo de 22 decisiones de mercado si el flujo sigue decidiendo, o al cierre del tramo si deja de hacerlo. Siempre entra después de madurar.

El término se acumula por bloques de hasta `accumulation_rows` decisiones, o del lote del predictor, y cada bloque divide su suma por |A|. Con w = 0 no se calcula nada y el gradiente es exactamente el de la referencia. F5 no añade pasos: si un tramo no tiene etiquetas principales, sus etiquetas auxiliares se descartan y se cuentan, igual que los términos de C.

## Estado y recuperación

`training/horizon_labels.py` guarda por flujo como mucho 20 − 1 + 2 + 1 = 22 decisiones con sus entradas, su etiqueta principal y las etiquetas auxiliares completadas. Las decisiones que ya entregaron sus etiquetas conservan solo la etiqueta, necesaria para otras ventanas. Los ordinales por mercado, las decisiones pendientes y las entradas guardadas forman parte del estado del recorrido que se exporta en la barrera posterior a cada paso, con tensores CPU compatibles con `weights_only`. La prueba de reanudación interrumpe un ajuste y comprueba que la continuación reproduce bit a bit el registro de auditoría y los gradientes de la ejecución continua.

Sin declarar `auxiliary_horizons`, `FinancialConfig.identity()`, el estado exportado y los contadores no cambian. Con las cabezas, los parámetros de la referencia conservan sus valores con la misma semilla: las cabezas usan su propio generador derivado de la semilla, como la rama ancha de PT6. Se añaden `auxiliary_heads.weight` y `auxiliary_heads.bias`, 130 parámetros con H = 64 y dos horizontes, en el papel `shared`.

## Desviaciones y límites

- La suma de residuos diarios no es un residuo de h sesiones con α̂ y β̂ reestimados para ese horizonte.
- El token se recalcula al madurar con los parámetros vigentes, no con los de la emisión. Es la única forma de obtener gradiente sin conservar grafos durante 20 sesiones.
- El término no ajusta MAC. Unas cabezas sobre la salida de MAC necesitarían conservar o reconstruir el estado rápido de cada flujo en la emisión.
- Con fusión regional, programa regional o sus variantes C1 y C6 de [#751](https://github.com/GonxKZ/mars-titan/issues/751), el token depende del bloque emitido y no se calcula fila a fila. Esas combinaciones se rechazan, igual que el postentrenamiento, que no tiene todavía una pérdida declarada para las cabezas, y el ajuste Dust inicial. También se rechazan los rangos por sesión y el objetivo de IC de F4, porque usan todas las filas emitidas de la sesión y el cálculo fila a fila no las reproduce. Los rangos necesitarían guardar las entradas ya transformadas.
- Las entradas guardadas ocupan unos 7,1 kB por decisión en FP32 con las dimensiones de v3.1. El tamaño de checkpoint y la memoria crecen con el número de activos (tabla de coste).

## Correspondencia entre ecuaciones, módulo y prueba

| Ecuación o regla | Módulo | Prueba |
| --- | --- | --- |
| y⁽ʰ⁾ = suma de h etiquetas consecutivas, madura con la última | `training/horizon_labels.py`, `HorizonLabels.mature` | `test_labels_wait_for_the_longest_window_and_share_one_delivery`, `test_labels_arriving_out_of_order_complete_each_window_once` |
| Un hueco rompe la ventana y la holgura limita la maduración | `HorizonLabels.mature`, `remember` | `test_a_missing_decision_breaks_every_window_that_crosses_it`, `test_a_late_component_beyond_the_slack_is_not_used` |
| Entrega al pasar el plazo y memoria acotada por flujo | `HorizonLabels.remember` | `test_a_decision_past_its_deadline_delivers_what_it_completed`, `test_stored_decisions_stay_bounded_per_flow` |
| Exportación y restauración del registro | `HorizonLabels.export`, `restore` | `test_export_and_restore_continue_with_the_same_windows` |
| \|a⁽ʰ⁾ − y⁽ʰ⁾\|/√h | `models/titans/auxiliary_horizons.py`, `auxiliary_errors` | `test_errors_follow_the_declared_formula_and_gradients`, con `gradcheck` en FP64 |
| Token por filas igual al emitido | `FinancialPredictor.fused_tokens` | `test_row_tokens_are_the_tokens_of_the_emission`, con y sin PT6 |
| Objetivos iguales a una derivación independiente desde el corpus | `ChronologicalTrainer._auxiliary` | `test_targets_are_sums_of_matured_labels_derived_independently` |
| Peso cero igual a la referencia | `_auxiliary` | `test_weight_zero_control_keeps_the_reference_gradients` |
| El término no cambia MAC ni la cabeza principal | `_auxiliary` | `test_matured_horizons_train_heads_and_token_but_not_the_rest` |

La derivación independiente reconstruye las ventanas con las etiquetas del corpus, los instantes emitidos y los instantes de maduración del registro de auditoría, sin usar el registro de F5. Las demás pruebas del recorrido comprueban que la repetición por bloques reproduce los gradientes, que alterar el sufijo futuro del corpus no cambia los gradientes anteriores al corte y que la reanudación conserva el registro.

## Controles

La [declaración](../../configs/experiments/auxiliary-horizons-v1.json) parte de la receta histórica con máscaras. El control son las mismas cabezas con w = 0: tiene la misma identidad estructural, el mismo registro y las mismas etiquetas contadas, pero no calcula el término y sus gradientes coinciden con los de la referencia. Las dos configuraciones son `mac_online`, están desactivadas por defecto y no acceden al test final. La variante se descarta si no mejora al control en el contraste principal de la criba, el IC medio por sesión, o si empeora su MAE calibrado.

## Coste medido

La medida CPU del 28 de octubre usa las formas de la edición v3.1 descritas en [F1](laplace-scale-head-v1.md#coste-medido), con 256 y 1.024 activos, ocho instantes, H = 64, cuatro hilos, FP32 estricto y SDPA MATH. El término auxiliar se mide con el mismo método del entrenador sobre una actualización estable: ocho instantes de decisiones que entregan a la vez sus dos horizontes. Se repitió siete veces con 256 activos y cinco con 1.024. El equipo tenía otras cargas, con una media de carga de un minuto entre 5 y 7 al empezar. Los recibos [de 256](../../reports/engineering/scale-head-horizons-20261027/cost-256.json) y [de 1.024 activos](../../reports/engineering/scale-head-horizons-20261027/cost-1024.json) conservan muestras y huellas.

| Activos | Modelo | Decisiones y etiquetas auxiliares | Término auxiliar | Actualización completa | Relación con la referencia |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | Referencia L1 | 0 | 0 | 0,673 s | 1,00 |
| 256 | Control w = 0 | 2.048 y 4.096 | 0,0004 s | 0,676 s | 1,00 |
| 256 | F5 | 2.048 y 4.096 | 0,347 s | 1,020 s | 1,52 |
| 1.024 | Referencia L1 | 0 | 0 | 3,330 s | 1,00 |
| 1.024 | Control w = 0 | 8.192 y 16.384 | 0,001 s | 3,327 s | 1,00 |
| 1.024 | F5 | 8.192 y 16.384 | 1,120 s | 4,548 s | 1,37 |

Los tiempos son medianas. Para cada decisión entregada, el término auxiliar vuelve a calcular los codificadores y la fusión, sin MAC, y añade entre un 37 % y un 52 % a cada actualización con estas formas. Calcular un token por etiqueta habría duplicado ese término.

El registro, medido aparte en régimen estable, tarda una mediana de 2,3 ms por evento con 256 activos y de 11,5 ms con 1.024, con máximos de 2,8 ms y 83,9 ms. Guarda 7.117 bytes por decisión pendiente, 36,4 MB y 145,8 MB en total, que forman parte del checkpoint de recuperación. Exportarlo tardó entre 0,03 y 0,11 s y entre 0,17 y 0,27 s. El pico de RAM del proceso fue de 1,45 GB y 3,41 GB, incluidos los demás modelos medidos. No se han medido VRAM ni energía, y la medida no describe un entrenamiento ni su utilidad predictiva.

## Comprobación y trabajo pendiente

La comprobación del 27 y 28 de octubre se hizo en CPU, sin pasos de optimizador y con la GPU ocupada por otra carga.

- Las 91 pruebas dirigidas de F1 y F5 pasan con medida de cobertura. Tras integrar los últimos cambios de develop, esas pruebas y las de los consumidores congelados, Dust y variantes regionales pasan con 281 casos.
- La regresión de los módulos que recorren estos cambios pasa con 385 casos y 7 omitidos por el bloqueo o por el enlace nativo. Falla `test_core_without_bank_trains_only_the_core` del adaptador del lector, que también falla en develop desde [#772](https://github.com/GonxKZ/mars-titan/pull/772) y no pasa por F1 ni F5.
- `make test-fast` pasa 11.220 pruebas y omite 381. La guarda del bloqueo de aprendizaje pasa con 54 casos y una omisión prevista.
- Nueve mutaciones dirigidas de F5, dentro de las quince de la familia, se detectan con los controles en verde antes y después. Se ejecutaron antes de integrar desde develop C1, C6 y el registro de sus fuentes, sobre el mismo código de F5. Quitar C1 y C6 del rechazo de la configuración o de `fused_tokens` hace fallar la prueba de operadores que mezclan filas.
- Con coverage 7.16.1 y radon 6.0.1, `auxiliary_horizons.py` cubre 49/49 sentencias y 10/10 ramas, con CRAP máximo 14, y `horizon_labels.py` cubre 106/106 sentencias y 42/42 ramas, con CRAP máximo 19.
- El 28 de octubre, tras pasar F1 al marco de objetivos de salida de #760, la regresión de los 22 módulos que recorren estos cambios pasó 588 casos con una omisión de la guarda. Los cuatro fallos eran pruebas del recorrido de F1 que esperaban la interfaz anterior, y sus 9 casos pasan tras actualizarlas.

El [recibo de verificación](../../reports/engineering/scale-head-horizons-20261027/verification.json) conserva los recuentos y las huellas de los informes JUnit y de las mutaciones.

Queda pendiente:

- La comprobación en `cuda:0`, que espera a que la GPU quede libre.
- La combinación con F4, la fusión regional, C1, C6, Dust y el postentrenamiento, que necesitan su propio contraste.
- La entrada en la criba de [#746](https://github.com/GonxKZ/mars-titan/issues/746) y cualquier evaluación predictiva, que sigue bloqueada.
