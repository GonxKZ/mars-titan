# Bucles anidados frente a un bucle plano de igual profundidad

`nested_equal_depth_v1` es la variante D1 del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#751](https://github.com/GonxKZ/mars-titan/issues/751), relacionada con [#709](https://github.com/GonxKZ/mars-titan/issues/709)). Contrasta si anidar J pasos del puente de evidencia dentro de K lecturas del banco aporta algo frente a un bucle plano con el mismo número de pasos, los mismos parámetros y la misma supervisión. El tipo de supervisión, final o por paso, es el segundo factor. La implementación añade los estados de cada refinamiento al resultado del lector y una opción de la receta de su ajuste. No se ha entrenado ni evaluado y no hay ninguna mejora medida.

## Motivación

El análisis independiente de HRM publicado por [ARC Prize](https://arcprize.org/blog/hrm-analysis) encontró que un Transformer de tamaño parecido, en la misma canalización, queda a pocos puntos del modelo jerárquico y que variar los pasos internos y externos respecto a la configuración base empeora el resultado. La mayor parte de la ganancia venía del bucle exterior de refinamiento durante el entrenamiento. Pasar de uno a dos bucles mejoró unos 13 puntos, y entrenar con hasta 16 bucles mejoró en más de 15 puntos la inferencia de una sola pasada. [TRM (Jolicoeur-Martineau, 2025)](https://arxiv.org/abs/2510.04871) cita ese análisis, sustituye las dos redes de HRM por una sola red de dos capas y conserva la supervisión en hasta 16 pasos exteriores.

MARS-TITAN ya tiene dos bucles anidados en el lector episódico. Cada uno de los K refinamientos (K ∈ {1, 2, 4}) consulta el banco con el estado de trabajo actual, ejecuta J pasos del puente recurrente (J ∈ {1, 2, 4}) y actualiza el estado. La pregunta de D1 es si ese anidamiento importa o si lo que importa es la profundidad total y cómo se supervisa.

## Configuraciones y contrastes

Todas las configuraciones ejecutan cuatro pasos del puente por predicción (K·J = 4), usan M1, tres vistas en anillo y la misma semilla, y tienen exactamente los mismos parámetros. J reutiliza los pesos de cada región en todos sus pasos y K reutiliza el refinador, así que ninguno de los dos añade parámetros. El trabajo del puente también coincide, 12 actualizaciones de nodo por fila con tres vistas.

| Configuración | K | J | Refinamientos supervisados | Profundidades supervisadas |
| --- | ---: | ---: | --- | --- |
| `nested_k2_j2_final` | 2 | 2 | 2 | 4 |
| `nested_k2_j2_per_step` | 2 | 2 | 1 y 2 | 2 y 4 |
| `flat_k4_j1_final` | 4 | 1 | 4 | 4 |
| `flat_k4_j1_per_step` | 4 | 1 | 2 y 4 | 2 y 4 |
| `flat_k1_j4_final` | 1 | 4 | 1 | 4 |

La [declaración](../../configs/experiments/nested-equal-depth-v1.json) fija cinco contrastes. El anidamiento se mide entre `nested_k2_j2` y `flat_k4_j1` con la misma supervisión, final o en las profundidades 2 y 4. La supervisión se mide dentro de cada estructura. El último contraste compara el anidado con una única lectura de cuatro pasos (`flat_k1_j4_final`), que separa el efecto de volver a consultar el banco. Esa configuración solo admite pérdida final, porque sus pasos intermedios ocurren dentro del puente y no producen un estado de trabajo.

El plano exterior (`flat_k4_j1`) consulta el banco cuatro veces y el anidado dos. Esa diferencia forma parte de la estructura que se contrasta, no es un desequilibrio oculto, y su coste se mide más abajo. D1 se descarta si las configuraciones anidadas no mejoran a las planas de igual supervisión con el contraste y la corrección que fije la criba ([#746](https://github.com/GonxKZ/mars-titan/issues/746)). La supervisión por paso se descarta si no mejora a la final dentro de alguna de las dos estructuras.

## Pérdida supervisada

Sea $s_k$ el estado de trabajo tras el refinamiento k y h la cabeza congelada del padre. Sin declaración, el ajuste usa la pérdida de la predicción emitida, $\ell(h(s_K), y)$, igual que antes. Con `supervised_refinements` = $(k_1, \dots, k_m)$, creciente y con $k_m = K$,

$$
L=\frac{1}{m}\sum_{i=1}^{m}\ell\bigl(h(s_{k_i}),\,y\bigr).
$$

ℓ es pinball de cinco niveles con `quantile_head_v1` o L1 con la cabeza escalar. Las filas y su peso dentro del tramo son los mismos que en la pérdida final. El último término reutiliza la predicción emitida y los anteriores aplican la misma cabeza a estados intermedios. La predicción emitida, la comprobación bit a bit de la repetición y la selección por MAE de sesión no cambian.

El gradiente recorre todos los refinamientos dentro de la misma predicción. TRM separa el estado del grafo entre pasos de supervisión para acotar la memoria con hasta 16 pasos. Aquí K ≤ 4 y todos los pasos pertenecen a una misma emisión, así que se conserva la propagación completa. Es una desviación declarada.

## Correspondencia entre ecuaciones, módulo y pruebas

Las pruebas están en `tests/models/titans/test_nested_equal_depth.py`.

| Elemento | Código | Prueba |
| --- | --- | --- |
| $s_k$ tras cada refinamiento, con $s_K$ igual al estado final | `EpisodicReadout._forward`, `ReadoutResult.states` y `BridgedReadoutResult.states` | `test_each_refinement_keeps_its_state_and_the_last_one_is_the_result`, con y sin puente |
| Pérdida sin declaración igual a la anterior | `readout_loss` | `test_without_declared_supervision_the_loss_is_the_emitted_one`, con L1 y pinball, sin tolerancia |
| L como media de las pérdidas declaradas | `readout_loss` y `head_predictions` | `test_declared_supervision_averages_the_loss_after_each_refinement` |
| Gradiente adicional del estado intermedio | `readout_loss` | `test_intermediate_supervision_reaches_the_reader_through_the_earlier_state` |
| La lista acaba en K | `readout_loss` y `MarsTitanInference.__init__` | `test_supervision_must_end_at_the_reader_k` y el caso `supervision_before_k` de `tests/training/test_mars_titan_run.py` |
| Receta válida e identidad anterior sin declaración | `ReadoutRecipe` | `test_the_recipe_declares_supervision_without_changing_the_previous_identity` y `test_the_recipe_rejects_invalid_supervision` |
| Iguales parámetros, pesos y trabajo del puente | `EpisodicReadout` y `BridgeRecurrenceConfig.node_updates` | `test_equal_depth_arms_share_parameters_and_bridge_work` |
| Configuraciones válidas y supervisión emparejada | declaración y `check_components` | `test_declared_arms_are_valid_variants_of_equal_depth_and_supervision` |
| Recorrido real del ajuste | `ReadoutTrainer._loss` y `_update` | `test_supervising_only_the_last_refinement_reproduces_the_previous_gradients`, con el enlace episódico nativo: supervisar solo K reproduce bit a bit los gradientes anteriores y añadir el primer refinamiento los cambia |

## Límites

D1 no implementa la jerarquía de tres ritmos ni el selector aprendido de #709, que siguen pendientes allí. Tampoco cambia el puente, el refinador ni el número de transiciones MAC. Repetir refinamientos no escribe episodios ni aplica dos veces el mismo feedback. La supervisión por paso es una opción del ajuste del lector, no del predictor congelado, y por eso vive en la receta y no en la identidad del modelo.

## Coste medido

`benchmarks/nested_equal_depth_cost.py` mide en CPU, con FP32 y cuatro hilos, una llamada completa del ajuste sin optimizador: lectura con puente recurrente sobre un banco ficticio de 256 episodios, cabeza escalar, pérdida L1 y propagación hacia atrás hasta los parámetros del lector. Un gancho de `torch.optim` rechaza cualquier paso. Cada caso tiene tres llamadas de calentamiento y 20 medidas en cada una de dos pasadas con el orden de las configuraciones invertido. El [recibo](../../reports/engineering/nested-equal-depth-20261027/cpu-cost.json) guarda las muestras. La máquina tenía una carga media de 28 durante toda la medida, así que los percentiles son descriptivos y no cotas superiores. Todas las configuraciones tienen 57.665 parámetros.

| Configuración | p50 con 64 filas | p95 con 64 filas | p50 con 256 filas | p95 con 256 filas |
| --- | ---: | ---: | ---: | ---: |
| `nested_k2_j2_final` | 13,8 ms | 28,9 ms | 33,1 ms | 59,2 ms |
| `nested_k2_j2_per_step` | 12,4 ms | 20,3 ms | 34,5 ms | 50,1 ms |
| `flat_k4_j1_final` | 24,3 ms | 51,2 ms | 58,1 ms | 79,8 ms |
| `flat_k4_j1_per_step` | 29,5 ms | 54,3 ms | 49,3 ms | 75,5 ms |
| `flat_k1_j4_final` | 9,8 ms | 12,1 ms | 17,7 ms | 37,3 ms |

Con 256 filas, el plano exterior cuesta unas 1,7 veces el anidado, porque consulta el banco y aplica el refinador cuatro veces en lugar de dos. La lectura única con cuatro pasos es la más barata. La supervisión por paso solo añade una pasada de la cabeza por refinamiento supervisado, y su diferencia con la pérdida final queda dentro de la dispersión que produce la carga. El coste en `cuda:0` no se ha medido porque la GPU sigue ocupada por otras aplicaciones. El benchmark ya admite `--device cuda:0` sin TF32.
