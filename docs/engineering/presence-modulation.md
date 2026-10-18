# Fusión condicionada por la presencia de modalidades (PT6)

Estado a 18 de octubre de 2026: implementada y comprobada. [#458](https://github.com/GonxKZ/mars-titan/issues/458) cerró el diseño y la declaración del contraste, no su resultado experimental. La motivación, la alternativa trivial y la predicción están en las [propuestas posteriores a Titans](../research/post-titans-proposals.md#pt6-fusión-condicionada-por-la-presencia-de-modalidades-línea-posterior). No entra en la campaña A v2.

El [ensayo combinado del 18 de octubre](../../reports/research/combined-memory-20261018/README.md) ejecutó FiLM en CUDA con 64 instrumentos, dos épocas y evaluación en 2023. La celda con PT6 solo obtiene un MAE medio por sesión de 0,01422116 frente a 0,01482282 del núcleo, sobre tres semillas y todas las filas. La rama ancha quedó sin ejecutar. Esa media no es el contraste principal de esta tarea, que exige comparar FiLM con la rama ancha en filas con ausencias. La combinación PT5+PT6 reduce el error observado frente al núcleo, pero ninguna celda supera al residual cero.

La combinación PT1+PT5+PT6 presenta [sensibilidad numérica al congelar los parámetros en FP32](../../reports/research/combined-memory-20261018/numerics.json), seguida en [#662](https://github.com/GonxKZ/mars-titan/issues/662). La identidad local de FiLM con todas las modalidades presentes no demuestra equivalencia entre las trayectorias numéricas completas.

## Qué hace

La [fusión común](architecture.md#codificación-y-fusión-de-eventos) anula la proyección de cada modalidad ausente y concatena los cinco bits de presencia antes de la primera capa. Eso permite desplazar la salida según el patrón de ausencias, pero no que la ausencia de las noticias, por ejemplo, cambie el peso que la fusión da a los fundamentales. PT6 añade una modulación FiLM (Perez et al., 2018) de los cinco bloques cuya escala y desplazamiento solo dependen del vector de ausencias.

`FinancialConfig.presence_modulation` admite dos valores, solo con la política `historical_masked_2000_v1`, que es la única con bits de presencia:

- `film_v1`, la propuesta.
- `wide_fusion_v1`, la alternativa trivial: el mismo orden de parámetros añadido a la fusión sin interacción multiplicativa con la presencia.

Sin declararla, la identidad y el cálculo son los anteriores. El código está en `src/mars_titan/models/titans/presence_modulation.py`, con su punto de inserción en `fuse_modalities` (`src/mars_titan/models/baselines/multimodal.py`), que llama `FinancialPredictor.prepare` antes de Titans-MAC.

## Ecuaciones

Con presencia p ∈ {0, 1}⁵, ausencias a = 1 − p y el bloque z_m ∈ ℝᴴ de cada modalidad m (precios, noticias, gráficos, fundamentales y macro), ya anulado si falta:

$$z'_m = p_m\left(z_m + (\Gamma_m a) \odot z_m + \mathrm{B}_m a\right), \qquad \Gamma_m, \mathrm{B}_m \in \mathbb{R}^{H \times 5}.$$

La entrada de la fusión pasa a ser $[z'_1; \dots; z'_5; p]$. Tres propiedades se siguen de la fórmula y se comprueban en las pruebas:

1. **Identidad exacta con las cinco modalidades.** Si a = 0, la fila no pasa por la modulación (`torch.where`), así que conserva sus bits sean cuales sean Γ y Β.
2. **Las ausentes siguen anuladas.** El factor p_m las deja en cero, como en la fusión común.
3. **Arranque en el control.** Γ y Β empiezan en cero y no consumen el generador, así que con la misma semilla PT6 emite al principio exactamente lo mismo que su control.

La alternativa trivial suma a la salida de la fusión una rama residual de k unidades ocultas sobre la misma entrada x:

$$y = \mathrm{fusión}(x) + D\,\mathrm{SiLU}(E x), \qquad E \in \mathbb{R}^{k \times (5H+5)},\ D \in \mathbb{R}^{H \times k}.$$

D empieza en cero, de modo que también arranca en el control. E se sortea con el reparto de `nn.Linear` y un generador propio derivado de la semilla, sin tocar el generador global. k es el entero que deja la cuenta k(6H + 5) más cerca de los 50H parámetros de FiLM. Con H = 64 son 8 unidades y 3.112 parámetros frente a 3.200. Las dos cuentas quedan en la identidad.

| Pieza | Fuente o derivación | Módulo | Pruebas |
| --- | --- | --- | --- |
| Escala y desplazamiento lineales en las condiciones | FiLM (Perez et al., 2018), aplicado aquí al vector de ausencias | `PresenceFiLM` | `test_film_rescales_present_blocks_by_the_absence_of_others_and_keeps_absent_ones_zero` |
| Identidad con todas las modalidades y arranque en el control | Derivación propia, por construcción | `PresenceFiLM.forward`, inicialización a cero | `test_film_is_the_identity_on_complete_rows_whatever_its_weights`, `test_film_keeps_complete_rows_bit_for_bit_including_signed_zeros`, `test_both_options_start_exactly_at_the_control_with_the_same_seed` |
| Gradiente solo desde filas con ausencias y solo en las columnas de modalidades que faltan | Consecuencia de la fórmula | `PresenceFiLM` | `test_film_gradient_comes_only_from_rows_with_some_absence` |
| Rama ancha con salida inicial nula y cuenta de parámetros | Alternativa trivial declarada | `WideFusionBranch`, `wide_units` | `test_wide_branch_learns_through_its_output_and_starts_without_effect`, `test_wide_branch_draws_its_expansion_from_its_own_seed_and_applies_silu`, `test_parameter_counts_are_recorded_and_the_wide_branch_matches_film_within_five_percent` |
| Identidad, estado guardado y política | Contrato del predictor | `FinancialConfig.identity` | `test_identity_records_the_option_and_its_parameters_only_when_declared`, `test_unknown_options_and_the_strict_policy_are_rejected` |
| Comparación declarada sin ejecutar | Regla escrita de antemano | `configs/evaluation/presence-modulation-comparison.json` | `test_comparison_is_declared_with_its_control_rule_and_budget_but_not_executed` |

Las pruebas están en `tests/models/titans/test_presence_modulation.py`. Calculan salidas y gradientes con lotes técnicos y cargan pesos fijos con `load_state_dict` cuando necesitan Γ, Β o D distintos de cero. No hay optimizador ni pasos.

## Entradas, orden y estado

PT6 solo lee los bits de presencia de la propia fila, que el lector fija con la disponibilidad de cada modalidad en el corte de la decisión. No lee etiquetas ni otras filas. Actúa dentro de la fusión, antes de la memoria, así que no añade estado por flujo ni cambia lo que Titans-MAC guarda entre decisiones. Sus parámetros son parámetros compartidos más del predictor y se ajustarían con el resto.

## Control con el que se descarta

La [declaración](../../configs/evaluation/presence-modulation-comparison.json) fija tres modelos sobre la receta de 2000 con `mac_online`: el control, FiLM y la rama ancha. Los tres comparten filas, ventanas, semillas, pesos iniciales del tronco, casos de búsqueda, actualizaciones del optimizador, regla de parada y truncación, y la declaración registra el presupuesto declarado (6 h de GPU) frente al consumido (cero). La métrica principal es la diferencia relativa de MAE por sesión entre FiLM y la rama ancha en las filas con alguna modalidad ausente, emparejada y con bootstrap circular por bloques de 16 sesiones. FiLM se conserva si el límite superior del intervalo del 95 % queda por debajo de −MID, la banda entre semillas del [control de varianza](../research/protocol.md#dispersión-entre-semillas-en-los-pilotos-de-memoria) calculada en esas filas (regla `pt6`), y si en todas las filas no empeora al control más de un 0,2 % con la MID de esas filas por debajo de ese margen (regla `pt6_control`). Si la rama ancha la iguala, la mejora vendría de añadir capacidad y PT6 se abandona. También se abandona si el límite superior frente al control pasa de +0,5 % con la MID por debajo de ese límite. La predicción registrada es que no habrá diferencia detectable.

## Coste medido sin entrenar

[`cost.json`](../../reports/engineering/presence-modulation-20261010/cost.json) mide en CPU, en FP32 y con dos hilos, los tres modelos de la declaración sobre entradas reales de la vista de la campaña sin objetivos: 128 flujos de EE. UU. y 8 instantes, 1.024 filas de las que 950 tienen alguna modalidad ausente (928 sin noticias y 600 sin fundamentales). Cada repetición recorre los 8 instantes con grafo, calcula una pinball frente a ceros y ejecuta el backward, sin optimizador, y después repite el recorrido sin grafo. Las 15 repeticiones se alternan entre modelos con orden rotatorio.

| Modelo | Parámetros añadidos | Forward con grafo | Backward | Inferencia |
| --- | ---: | ---: | ---: | ---: |
| Control | 0 | 0,246 s | 0,111 s | 0,214 s |
| FiLM | 3.200 | 0,227 s (0,92) | 0,113 s (1,02) | 0,238 s (1,11) |
| Rama ancha | 3.112 | 0,250 s (1,01) | 0,120 s (1,08) | 0,232 s (1,08) |

Son medianas, con el cociente frente al control entre paréntesis. La máquina tenía otras cargas (carga media entre 10 y 15) y los intervalos entre mínimo y máximo de los tres modelos se solapan, así que a esta escala PT6 no tiene un coste distinguible del control. Ninguna de las dos opciones cambia el estado por flujo (65.552 bytes). Esta medida corresponde solo a CPU. El ensayo combinado registra el recorrido de FiLM en CUDA, con otras comprobaciones CPU concurrentes y sin ejecutar la rama ancha. Sigue sin haber una comparación de coste aislada entre las dos opciones en GPU.

## Desviaciones y límites

- La propuesta hablaba de una capa de fusión más ancha. Con una sola capa de fusión, su salida es el token de Titans y su anchura está fijada por la memoria, así que la anchura extra se añade como rama residual de k unidades. Un entero no siempre iguala la cuenta de FiLM y la diferencia queda registrada.
- La modulación también actúa sobre precios y gráficos, aunque el lector los exige presentes. Sus columnas de Γ y Β nunca reciben gradiente, porque esas ausencias son siempre cero.
- Las referencias no reciben PT6. Es una variante del núcleo de Titans que se compara con su propio control, no con otras familias.

## Pendiente sin medir

- La comparación con la rama ancha en filas con ausencias, el piloto de 2018 a 2020 y su regla de decisión. El ensayo combinado no completa ese contraste y la campaña sigue bloqueada.
- La proporción de filas de validación de 2018 a 2020 con alguna ausencia en los activos del piloto, que decide la potencia de la regla. Debe contarse sin leer objetivos antes de lanzar el piloto.

## Referencias

- Perez, E., Strub, F., de Vries, H., Dumoulin, V. y Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1). https://doi.org/10.1609/aaai.v32i1.11671
