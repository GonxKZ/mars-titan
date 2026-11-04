# Espacio de trabajo compartido en el operador regional

`shared_workspace_v1` es la variante C6 del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#751](https://github.com/GonxKZ/mars-titan/issues/751)). Sustituye el resumen de `regional_fusion_v1` por unas pocas ranuras compartidas. Las regiones compiten por escribir en ellas y después cada región lee lo escrito. Está implementada en `src/mars_titan/models/titans/regional_variants.py` y comprobada con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y expectativa previa

La idea de una conciencia que une todas las áreas tiene una traducción publicada en aprendizaje profundo, el espacio de trabajo compartido de [Goyal et al. (2022)](https://arxiv.org/abs/2103.01197). En lugar de que todos los módulos se comuniquen por pares, compiten por escribir en un espacio de capacidad limitada cuyo contenido se difunde después a todos. Los autores argumentan que ese límite favorece la especialización y la coordinación entre módulos.

En MARS-TITAN ese papel se parece mucho al de la memoria persistente de Titans-MAC y al resumen del operador regional. La expectativa previa es la redundancia. La variante se incluye porque es la forma más directa de contrastar esa idea, y su resultado esperado más probable es que no supere a sus controles.

## Recurrencia ejecutada

Los pasos locales son los de la [fusión regional](regional-fusion-v1.md), con atención de una cabeza sobre los nodos, la misma máscara y la misma ganancia η. Lo que cambia es el resumen. Cada fila empieza con las ranuras aprendidas $S^0\in\mathbb R^{s\times H}$, comunes a todas las filas. Al acabar los pasos locales del ciclo c, con R los estados de los nodos:

$$
E=\frac{\operatorname{LN}_s(S)\,W_{wq}\,\bigl(R\,W_{wk}\bigr)^{\mathsf T}}{\sqrt H}\in\mathbb R^{s\times 5}.
$$

Con competición, $F_j$ son las k regiones presentes de mayor $E_{j\cdot}$ para la ranura j. La elección usa las puntuaciones sin gradiente, y el softmax sobre las elegidas sí lo recibe. Con escritura densa, $F_j$ son todas las regiones presentes. Después,

$$
\omega_{j\cdot}=\operatorname{softmax}_{F_j}(E_{j\cdot}),\qquad
S\leftarrow S+\gamma\,\operatorname{MLP}\bigl(\operatorname{LN}_s(S)+\omega\,R\,W_{wv}\bigr).
$$

Una fila sin regiones presentes conserva sus ranuras. Cada nodo lee las ranuras actualizadas:

$$
C=\operatorname{softmax}\!\left(\frac{R\,W_{rq}\,(S\,W_{rk})^{\mathsf T}}{\sqrt H}\right) S\,W_{ro}\in\mathbb R^{5\times H}.
$$

C ocupa el lugar de la difusión U del operador original, pero cada nodo recibe su propio contexto en vez de uno común. Entra en la normalización de los pasos locales del ciclo siguiente y en la lectura final $Y=X+P\odot\bigl((R-R^0+C)W_r^{\mathsf T}\bigr)$. Las ranuras y el contexto se reinician en cada predicción y solo persisten entre los ciclos de una misma llamada.

## Desviaciones respecto al artículo

| Artículo | C6 | Motivo |
| --- | --- | --- |
| El espacio persiste entre pasos temporales o capas, y el apéndice D compara una versión sin persistencia | Se reinicia en cada predicción | El estado entre sesiones pertenece a MAC y al banco episódico, con sus propias reglas de recuperación |
| Actualización con la compuerta de RMC y atención sobre [M; A] | MLP residual con ganancia γ, sin que las ranuras se atiendan a sí mismas | Repetir la actualización del resumen original y aislar el efecto de la competición |
| Top-k sobre los valores previos al softmax, y en RIMs+SW selección global por relevancia de la entrada | Top-k por ranura entre las regiones presentes | Con cinco regiones, cada ranura puede recoger una combinación distinta |
| La difusión suma $\sum_j s_{kj}\hat v_j$ al estado del especialista | El contexto entra en la normalización y en la lectura final | Mantener el estado local y la lectura del operador original |
| Entre 1 y 10 ranuras según el experimento | 2 por defecto, de 1 a 4 | Con cinco regiones, más de cuatro ranuras dejaría de ser un cuello de botella |

La selección top-k no es diferenciable. Las puntuaciones que la deciden sí reciben gradiente a través del softmax de las regiones elegidas, como en la selección híbrida que describen los autores.

## Controles

El control propio es `write="dense"`. Escribe con atención cruzada sobre todas las regiones presentes y tiene exactamente los mismos parámetros. Corresponde a la competición blanda (TR+SSW) de los experimentos del artículo. Ambas construcciones derivan sus pesos de la misma semilla de familia y empiezan con los mismos parámetros. Cuando hay k regiones presentes o menos, las dos escriben igual.

El segundo control es el predictor `mac_online` con `regional_fusion_v1` y el mismo calendario, que ya tiene la memoria persistente de MAC y el resumen común. C6 almacena 15H² + (s + 4)H parámetros, 61.824 para H = 64 y dos ranuras, frente a los 41.216 de `regional_fusion_v1`. Esa comparación no está emparejada en parámetros.

La variante se descarta si no mejora a la escritura densa, porque entonces la competición no aporta nada. Si ninguna de las dos mejora al operador original, las ranuras no añaden nada al resumen común ni a la memoria persistente de MAC. El contraste y la corrección los fija el protocolo de la criba ([#746](https://github.com/GonxKZ/mars-titan/issues/746)).

## Correspondencia entre ecuaciones, módulo y pruebas

Las pruebas están en `tests/models/titans/test_regional_variants.py` y `tests/models/titans/test_financial_regional_variants.py`.

| Elemento | Código | Prueba |
| --- | --- | --- |
| Calendario común igual al operador original | `_RegionalVariant.forward` | `test_the_common_schedule_reproduces_the_original_operator_bit_for_bit` |
| E, $F_j$ y ω | `SharedWorkspace._write_weights` | `test_competition_writes_only_with_the_best_regions_of_each_slot` |
| Mismos parámetros en la variante y su control | `_RegionalVariant.__init__` | `test_written_and_dense_workspaces_start_from_the_same_parameters` |
| Actualización de S y lectura C | `SharedWorkspace._summary` | `test_executed_recurrence_gradients_match_finite_differences` |
| Nodos ausentes y filas vacías | `forward`, `_write_weights` | `test_missing_regions_send_nothing_and_keep_their_blocks` |
| Estado de trabajo por fila y por llamada | `forward` | `test_rows_are_independent_and_the_working_state_resets` |
| Semilla de familia sin tocar el RNG global | `_RegionalVariant.__init__` | `test_construction_and_calls_leave_the_global_generator_untouched` |
| Checkpoints, declaraciones y entradas | `load_state_dict`, `parse_regional_variant`, `_checked_presence` | `test_checkpoints_restore_the_same_variant_and_reject_another_contract`, `test_incomplete_or_invalid_declarations_are_rejected` y `test_inputs_outside_the_contract_are_rejected` |
| Integración en el predictor | `FinancialConfig.regional_variant` | `test_the_variant_changes_only_the_fusion_and_advances_mac_once` y `test_a_saved_variant_restores_only_into_the_same_declaration` |
| Igualdad CPU y CUDA | todo el módulo | `test_cuda_matches_cpu`, pendiente de ejecutar con la GPU libre |

## Integración y límites

La integración es la misma que la de [C1](region-axis-mixing-v1.md). La variante exige `summary=true` en el calendario, porque sustituye al resumen, y solo se declara sobre `mac_online`. El consumidor congelado la rechaza hasta que se registre su módulo.

El entrenador fija la fuente de C6 en su identidad cuando está activo. El [contrato de fuentes](../engineering/regional-source-provenance.md) conserva el rechazo de padres congelados antes de construir el modelo y distingue esa frontera de una comprobación matemática o predictiva.

Las ranuras no son memoria a largo plazo, ni pesos rápidos de Titans, ni episodios del banco. Se aprenden como parámetros compartidos del ajuste externo y su contenido de trabajo desaparece al terminar cada predicción. La variante no prueba ninguna teoría de la conciencia. Solo contrasta si un cuello de botella con competición mejora la comunicación entre fuentes.

## Coste

La medida de [`benchmarks/regional_variants_cost.py`](../../benchmarks/regional_variants_cost.py) llama solo al operador regional con bloques aleatorios [B,5,64] en FP32 y las cinco fuentes presentes, con el anillo, dos ciclos, dos pasos locales y resumen. `forward` es la inferencia sin gradiente. `backward` añade la propagación hasta los parámetros del operador, sin optimizador, y un gancho de `torch.optim` rechaza cualquier paso. Hubo tres calentamientos y veinte repeticiones por condición en dos pasadas de orden inverso. Se ejecutó en CPU con 8 hilos, PyTorch 2.14.0 y Python 3.12.14, con otras cargas del equipo activas y una media de carga en torno a 11. Las cifras son medianas en milisegundos por llamada. El [JSON](../../reports/engineering/regional-variants-20261027/cpu-cost.json) conserva el p95 y las muestras.

| Operador | Parámetros | 64 filas, forward | 64 filas, backward | 256 filas, forward | 256 filas, backward |
| --- | ---: | ---: | ---: | ---: | ---: |
| `regional_fusion_v1` | 41.216 | 7,32 | 15,75 | 15,14 | 30,61 |
| C6 con escritura densa | 61.824 | 8,63 | 16,96 | 16,37 | 40,67 |
| C6 con competición | 61.824 | 8,76 | 18,15 | 16,66 | 40,18 |

La competición y su control cuestan prácticamente lo mismo, como se espera de una selección top-k sobre cinco regiones. Frente al operador original, C6 añade un 50 % de parámetros y, con 256 filas, cerca de un tercio más de tiempo en la propagación hacia atrás. La revisión del estudio estimaba entre 1 y 1,2 U para el predictor completo, y esta medida del operador no lo sustituye. Falta la medida en `cuda:0`, que se hará con la GPU libre.
