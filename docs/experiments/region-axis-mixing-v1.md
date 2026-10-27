# Mezcla entre cabezas del operador regional

`region_axis_mixing_v1` es la variante C1 del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#751](https://github.com/GonxKZ/mars-titan/issues/751)). Sustituye a `regional_fusion_v1` dentro de `mac_online` con el mismo calendario de pasos locales y resumen, y añade a la atención local la mezcla de talking-heads. Está implementada en `src/mars_titan/models/titans/regional_variants.py` y comprobada con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y alcance

La revisión del estudio formaliza la «atención 3D por regiones» con un tensor (lote, región, token, canal). Si cada región atiende solo a sus tokens con sus propios pesos, el resultado es una atención multicabeza ordinaria, y solo aparece una operación distinta cuando la atención mezcla el eje de región. La forma más barata de esa mezcla es talking-heads ([Shazeer et al., 2020](https://arxiv.org/abs/2003.02436)), que proyecta los logits a lo largo del eje de cabezas antes del softmax y los pesos después.

En el operador regional cada fuente llega como un único vector de dimensión H, así que no hay un eje de tokens dentro de cada región. C1 aplica la mezcla a cinco cabezas que atienden sobre los cinco nodos. El número de cabezas coincide con el de regiones, pero ninguna cabeza queda atada a una fuente. Una versión con tokens propios de cada región tendría que actuar dentro de los codificadores y queda fuera de esta variante.

## Recurrencia ejecutada

Para una fila, Z es la entrada normalizada y enmascarada del paso local, igual que en la [fusión regional](regional-fusion-v1.md). Con d = ⌊H/5⌋ y cinco cabezas h = 1, …, 5:

$$
(Q_h, K_h, V_h) = Z\,W^{h}_{qkv},\qquad
J_h = Q_h K_h^{\mathsf T}/\sqrt d .
$$

El control calcula $A_h=\operatorname{softmax}(J_h+\mathcal M)$. Con mezcla,

$$
L_g=\sum_{h=1}^{5}P_{gh}\,\bigl(J_h\odot \mathbb 1_{\mathcal M}\bigr),\qquad
W_g=\operatorname{softmax}(L_g+\mathcal M),\qquad
A_g=\sum_{h=1}^{5}Q_{gh}\,W_h .
$$

M es la misma máscara de aristas y presencia del operador original, con la arista propia de las consultas ausentes. $\mathbb 1_{\mathcal M}$ pone a cero los logits prohibidos antes de mezclar. Como todas las cabezas comparten la máscara, la mezcla de una posición prohibida solo afecta a esa posición, que vuelve a prohibirse. Por eso el cero no cambia la salida, pero evita que un logit no finito en una posición prohibida contamine el gradiente de P. Los pesos mezclados valen cero fuera de las aristas y no se renormalizan, igual que en la sección 4 del artículo. Tras ajustar P y Q, el mensaje puede dejar de ser una combinación convexa.

El mensaje concatena $A_g V_g$ de las cinco cabezas, lo proyecta de 5d a H y actualiza los nodos presentes con la ganancia η del operador. El resumen, su difusión y la lectura final son los de `regional_fusion_v1`.

## Desviaciones respecto al artículo

| Artículo | C1 | Motivo |
| --- | --- | --- |
| $h_k$, h y $h_v$ pueden diferir | Las tres valen 5 | Igualar el control sin mezcla, que tiene cinco cabezas |
| Atención sobre secuencias de tokens | Atención sobre los cinco nodos de una fila | El operador recibe un vector por fuente |
| La sección 4 no fija la inicialización de $P_l$ y $P_w$ | P y Q empiezan en la identidad | La variante y su control calculan lo mismo al construirse |
| El pseudocódigo no trata máscaras | Logits prohibidos a cero antes de mezclar y prohibidos de nuevo después | Conservar aristas y presencia del operador |
| Cabezas de dimensión $d_k$ libre | d = ⌊H/5⌋, con 60 canales internos para H = 64 | Reparto habitual de la atención multicabeza |

C1 calcula logits y pesos con multiplicaciones explícitas, porque la mezcla necesita los logits antes del softmax. No usa `scaled_dot_product_attention` ni ninguno de sus backends.

## Controles

El control propio es `mixing="none"`. Tiene las mismas cabezas, proyecciones y resumen, sin P ni Q. Ambas construcciones derivan sus pesos de la misma semilla de familia, así que comparten todos los parámetros comunes y empiezan con salidas idénticas. La diferencia entre ellas al terminar un ajuste solo puede venir de la mezcla y de lo que el optimizador haga con ella.

`regional_fusion_v1` con el mismo calendario es la referencia del operador anterior, con una cabeza de dimensión H. Para H = 64 almacena 41.216 parámetros, el control de C1 40.192 y C1 40.242. El bloque denso con iguales FLOPs que proponía la revisión del estudio no está implementado. El coste medido figura más abajo, y la comparación con el operador anterior no está emparejada en parámetros.

La variante se descarta si no mejora a su control sin mezcla con el contraste y la corrección que fije el protocolo de la criba ([#746](https://github.com/GonxKZ/mars-titan/issues/746)). En ese caso el intercambio entre cabezas no aporta nada que no consigan cinco cabezas independientes.

## Correspondencia entre ecuaciones, módulo y pruebas

Las pruebas están en `tests/models/titans/test_regional_variants.py` y `tests/models/titans/test_financial_regional_variants.py`.

| Elemento | Código | Prueba |
| --- | --- | --- |
| Calendario común igual al operador original | `_RegionalVariant.forward` | `test_the_common_schedule_reproduces_the_original_operator_bit_for_bit`, con tres topologías, resumen activado o no y tres calendarios |
| $L_g$, $W_g$ y $A_g$ | `RegionAxisMixing._messages` | `test_mixing_starts_equal_to_its_control_and_only_the_mixing_separates_them` (FP32 y FP64) |
| Pesos mezclados dentro de las aristas | `_messages` | `test_mixed_weights_keep_each_head_inside_its_edges` |
| Nodos ausentes sin mensajes y con su bloque | `forward` | `test_missing_regions_send_nothing_and_keep_their_blocks` |
| Estado de trabajo por fila y por llamada | `forward` | `test_rows_are_independent_and_the_working_state_resets` |
| Gradientes de la recurrencia ejecutada | todo el módulo | `test_executed_recurrence_gradients_match_finite_differences` |
| Semilla de familia sin tocar el RNG global | `_RegionalVariant.__init__` | `test_construction_and_calls_leave_the_global_generator_untouched` |
| Checkpoints, declaraciones y entradas | `load_state_dict`, `parse_regional_variant`, `_checked_presence` | `test_checkpoints_restore_the_same_variant_and_reject_another_contract`, `test_incomplete_or_invalid_declarations_are_rejected`, `test_inputs_outside_the_contract_are_rejected` y `test_autocast_does_not_relax_the_declared_precision` |
| Integración en el predictor | `FinancialConfig.regional_variant` | `test_the_variant_changes_only_the_fusion_and_advances_mac_once` y `test_mixing_predictor_starts_equal_to_its_control` |
| Igualdad CPU y CUDA | todo el módulo | `test_cuda_matches_cpu`, pendiente de ejecutar con la GPU libre |

## Integración y límites

`FinancialConfig.regional_variant` acepta la configuración solo con `mac_online` y rechaza combinarla con `regional_fusion`, `regional_execution` o PT6. La receta la declara como un cambio del predictor (`PREDICTOR_CHANGES`). Con `None` el predictor conserva su identidad y sus parámetros anteriores. Con la variante, los parámetros del tronco no cambian y solo se añaden los de `regional_variant.*`. MAC sigue avanzando una vez por observación.

El consumidor congelado rechaza la variante, porque su módulo no está entre los componentes que conoce. Registrarla allí exige ampliar su huella de código y se hará cuando la criba la necesite. `regional_fusion.py` no se modifica, porque su fuente forma parte de la identidad de los consumidores congelados y de los estudios ya preparados.

## Coste

La medida de [`benchmarks/regional_variants_cost.py`](../../benchmarks/regional_variants_cost.py) llama solo al operador regional con bloques aleatorios [B,5,64] en FP32 y las cinco fuentes presentes, con el anillo, dos ciclos, dos pasos locales y resumen. `forward` es la inferencia sin gradiente. `backward` añade la propagación hasta los parámetros del operador, sin optimizador, y un gancho de `torch.optim` rechaza cualquier paso. Hubo tres calentamientos y veinte repeticiones por condición en dos pasadas de orden inverso. Se ejecutó en CPU con 8 hilos, PyTorch 2.14.0 y Python 3.12.14, con otras cargas del equipo activas y una media de carga en torno a 11. Las cifras son medianas en milisegundos por llamada. El [JSON](../../reports/engineering/regional-variants-20261027/cpu-cost.json) conserva el p95 y las muestras.

| Operador | Parámetros | 64 filas, forward | 64 filas, backward | 256 filas, forward | 256 filas, backward |
| --- | ---: | ---: | ---: | ---: | ---: |
| `regional_fusion_v1` | 41.216 | 7,32 | 15,75 | 15,14 | 30,61 |
| C1 sin mezcla | 40.192 | 8,86 | 17,57 | 16,44 | 32,77 |
| C1 con mezcla | 40.242 | 10,95 | 22,24 | 19,10 | 47,52 |

Frente a su control, la mezcla encarece el operador entre un 16 % y un 45 % según la forma y el recorrido, aunque solo añade unas 1.250 multiplicaciones por fila y paso local a las 79.800 del control. A este tamaño pesan sobre todo las operaciones adicionales (dos `einsum`, la anulación de logits y su máscara), no la aritmética. La revisión del estudio estimaba entre 1 y 1,05 U para el predictor completo. El operador es solo una parte de ese recorrido y esta medida no lo sustituye. Falta la medida en `cuda:0`, que se hará con la GPU libre. No se ha optimizado nada, porque todavía no hay un perfil del ajuste completo que muestre aquí un cuello de botella.
