# Memoria multiescala sobre los pesos rápidos (PT5)

Estado a 18 de octubre de 2026: implementada y comprobada. [#457](https://github.com/GonxKZ/mars-titan/issues/457) cerró el diseño y la declaración del contraste, no su resultado experimental. La motivación, la alternativa trivial y la predicción están en las [propuestas posteriores a Titans](../research/post-titans-proposals.md#pt5-memoria-multiescala-línea-posterior). No entra en la campaña A v2.

El [ensayo combinado del 18 de octubre](../../reports/research/combined-memory-20261018/README.md) ejecutó la cascada en CUDA con 64 instrumentos, dos épocas y evaluación en 2023. La celda con PT5 solo obtiene un MAE medio por sesión de 0,01369191 frente a 0,01482282 del núcleo, sobre tres semillas. El banco EMA quedó sin ejecutar, por lo que el ensayo no permite atribuir una ventaja al acoplamiento entre niveles ni aplicar la regla de esta tarea. Ninguna celda del factorial supera al residual cero.

La combinación PT1+PT5+PT6 presenta [sensibilidad numérica al congelar los parámetros en FP32](../../reports/research/combined-memory-20261018/numerics.json), seguida en [#662](https://github.com/GonxKZ/mars-titan/issues/662). La contracción de la parte homogénea derivada abajo no garantiza estabilidad de toda la trayectoria de MAC ni igualdad entre rutas numéricas.

## Qué hace

La memoria de Titans olvida con una sola tasa α por fila de salida, que elige una escala de tiempo en cada paso. Los datos tienen varias: eventos de días, ciclos de meses y regímenes de años. PT5 añade a cada matriz de pesos rápidos dos copias ocultas, de modo que el estado por flujo pasa de dos matrices por capa (pesos y momentum) a cuatro. Hay dos formas, que se declaran con `MemoryConfig.multiscale` y, en el predictor, con `FinancialConfig.memory_multiscale`:

- `cascade`, la propuesta: los tres niveles forman la cadena de Benna y Fusi (2016), integrada con Euler y paso 1 como en Kaplanis et al. (2018). La memoria lee el primer nivel.
- `ema_bank`, la alternativa trivial: tres integradores independientes que reciben la misma escritura, cada uno con su olvido. La memoria lee su media.

Sin declararla, la identidad, el estado y el cálculo son los anteriores. El código está en `MemoryMultiscale` (`src/mars_titan/models/titans/config.py`) y en `NeuralMemory.readable_weights` y `NeuralMemory._advance_levels` (`neural_memory.py`). El estado lleva los niveles en `NeuralMemoryState.levels`, que exportan, restauran y reúnen por bloques `mac.py`, `financial_blocks.py` y `transition_jacobian.py`.

## Ecuaciones

Para cada capa, con el olvido α_t ∈ [0, 1] y el momentum S_t = η_t S_{t−1} − θ_t ∇ℓ de Titans, sin cambios, y con g = `coupling`:

**Cascada.** Capacidades C = (1, 2, 4) y conductancias g_{1,2} = g, g_{2,3} = g/2 y g_{3,4} = g/4, con fuga del último nivel hacia cero:

$$\begin{aligned}
u_1' &= (1-\alpha_t)\,u_1 + S_t + g_{1,2}(u_2 - u_1)\\
u_2' &= u_2 + \tfrac{1}{2}\left[g_{1,2}(u_1 - u_2) + g_{2,3}(u_3 - u_2)\right]\\
u_3' &= u_3 + \tfrac{1}{4}\left[g_{2,3}(u_2 - u_3) - g_{3,4}\,u_3\right]
\end{aligned}$$

La memoria lee y deriva su pérdida asociativa en u_1. Las escalas C_k/g_{k,k+1} son 1/g, 4/g y 16/g. Con g = 1/16 son 16, 64 y 256 observaciones del flujo, un mes, un trimestre y un año de sesiones.

**Banco EMA.** Mismo α en el primer nivel y, en los otros dos, olvidos fijos con las escalas ocultas de la cascada, λ_2 = g/4 y λ_3 = g/16:

$$z_1' = (1-\alpha_t)\,z_1 + S_t,\qquad z_2' = (1-\lambda_2)\,z_2 + S_t,\qquad z_3' = (1-\lambda_3)\,z_3 + S_t,\qquad W = \tfrac{1}{3}(z_1 + z_2 + z_3).$$

La memoria lee W y deriva la pérdida en W. Como cada nivel recibe S_t entero, la escritura instantánea de la lectura es S_t, la misma que en el núcleo y en la cascada.

Los niveles ocultos empiezan como copias de los pesos iniciales, así que la primera lectura coincide con la del núcleo en las dos formas. En la cascada la primera escritura también coincide, porque u_2 − u_1 = 0.

| Pieza | Fuente o derivación | Módulo | Pruebas |
| --- | --- | --- | --- |
| Cadena con C_k = 2^{k−1}, g_{k,k+1} ∝ 2^{−k−2}, fuga final y Euler de paso 1 | Ecuaciones 2 y 3 de Kaplanis et al. (2018), que resumen a Benna y Fusi (2016) | `_advance_levels`, `MemoryMultiscale.conductances` | `test_cascade_follows_the_euler_equations_with_titans_forgetting` |
| Olvido α de Titans en el primer nivel y escritura S_t como Δw | Adaptación propia | `_advance_levels`, `readable_weights` | `test_cascade_first_write_equals_the_control_because_levels_start_equal`, `test_the_cascade_reads_only_its_first_level` |
| Banco con las mismas escalas, gradiente en la media | Alternativa trivial declarada | `readable_weights`, `bank_decays` | `test_bank_reads_the_mean_takes_the_gradient_there_and_forgets_each_level_alone`, `test_scales_of_the_bank_are_the_hidden_scales_of_the_cascade` |
| Arranque en el núcleo y apagado sin niveles | Por construcción | `initial_state`, `MemoryConfig.identity` | `test_both_forms_start_reading_the_control_weights_with_the_same_draws`, `test_disabled_memory_keeps_no_levels_and_the_previous_identity` |
| Condición de estabilidad con las puertas de PT1 | Derivación propia (abajo) | `MemoryMultiscale.contraction` | `test_transition_contracts_uniformly_inside_the_pt1_box`, `test_without_a_forgetting_floor_the_cascade_has_no_uniform_contraction`, `test_state_stays_inside_the_bound_with_pt1_gates_and_clip` |
| Jacobiano completo con los niveles | `transition_jacobian.py` | `fast_state_jacobian` | `test_full_jacobian_contains_the_exact_hidden_blocks_of_the_cascade` |
| Causalidad, aislamiento, recuperación y presupuesto | Contrato del estado | `NeuralMemoryState.levels`, `export_state`, `_check_size` | `test_future_tokens_and_other_flows_do_not_change_a_flow_state`, `test_recovery_continues_bit_for_bit_and_rejects_another_contract`, `test_budget_counts_the_hidden_levels` |
| Predictor, bloques y gradientes | `FinancialConfig.memory_multiscale` | `financial.py`, `financial_blocks.py` | `test_predictor_carries_the_levels_through_blocks_and_gradients` |
| Comparación declarada sin ejecutar | Regla escrita de antemano | `configs/evaluation/multiscale-memory-comparison.json` | `test_comparison_is_declared_with_its_control_rule_and_budget_but_not_executed` |

Las pruebas están en `tests/models/titans/test_multiscale_memory.py`. Calculan estados, lecturas, Jacobianos y gradientes con tokens técnicos y no ejecutan ningún paso de optimizador.

## Condición de estabilidad con las puertas de PT1

Cada elemento de una matriz evoluciona con su propio vector x = (u_1, u_2, u_3), porque α es común a una fila de salida. La parte homogénea es x' = A(α) x con

$$A(\alpha) = \begin{pmatrix} 1-\alpha-g_{1,2} & g_{1,2} & 0\\ g_{1,2}/2 & 1-(g_{1,2}+g_{2,3})/2 & g_{2,3}/2\\ 0 & g_{2,3}/4 & 1-(g_{2,3}+g_{3,4})/4\end{pmatrix}.$$

Sea D = diag(d_1, 1, d_3) con d_1 = 1 + α_lo/(2 g_{1,2}) y d_3 = 1 + g_{3,4}/(2 g_{2,3}). Si α ∈ [α_lo, 1] y 0 < g ≤ 1/4, la norma ∞ ponderada ‖D A D⁻¹‖_∞ es menor que 1:

- Fila 1. Si α + g_{1,2} ≤ 1 vale 1 − α + (d_1 − 1) g_{1,2} ≤ 1 − α_lo/2. Si no, vale α + g_{1,2}(1 + d_1) − 1 ≤ 2g + α_lo/2 < 1.
- Fila 2. Vale 1 − (g_{1,2} + g_{2,3})/2 + g_{1,2}/(2 d_1) + g_{2,3}/(2 d_3), menor que 1 porque d_1 y d_3 son mayores que 1.
- Fila 3. Vale 1 − (g_{2,3} + g_{3,4})/4 + d_3 g_{2,3}/4, menor que 1 porque d_3 < 1 + g_{3,4}/g_{2,3}.

Como D no depende de α, la norma inducida acota cualquier producto de transiciones: ‖A(α_T) ⋯ A(α_1)‖_D ≤ ρ^T para cualquier secuencia de puertas dentro de la caja, con ρ el máximo de las filas. Solo la primera fila depende de α, como |1 − α − g_{1,2}| más una constante, que es convexa, así que `contraction` la evalúa en los extremos α_lo y 1. Con α_lo = 1/500 y g = 1/16, ρ = 1 − α_lo/2 = 0,999. En el banco la transición es diagonal y ρ = max(1 − α_lo, 1 − λ_3).

Con el recorte de PT1, cada fila del gradiente tiene norma a lo sumo G, de modo que |S_t| ≤ s̄ = θ_max G/(1 − η_hi) elemento a elemento. Entonces ‖x_t‖_D ≤ ρ‖x_{t−1}‖_D + d_1 s̄, y el estado queda acotado por max(‖x_0‖_D, d_1 s̄/(1 − ρ)) con cualquier profundidad. La prueba lo comprueba con 320 escrituras de tokens grandes, en profundidad 1 y 2 y en las dos formas.

Sin suelo de olvido no hay contracción uniforme: con α = 0 y sin la fuga final la cadena conserva C · x y tiene un autovalor 1. Es la misma situación que el núcleo sin PT1. La cota cubre la parte homogénea y el estado. No demuestra la contracción incremental de la memoria completa, en la que el gradiente depende de u_1. Para eso queda el diagnóstico con el Jacobiano completo, que ya incluye los niveles.

## Entradas, orden y estado

PT5 no lee nada nuevo: usa α y S_t, calculados por Titans con el token de la decisión. Los niveles avanzan una vez por observación de `mac_online`, después del momentum, con los valores del paso anterior. `mac_frozen` y `mac_disabled` no escriben memoria. Cada flujo tiene sus propios niveles y no se mezclan entre activos. Triplica las copias de los pesos rápidos y duplica el estado rápido por flujo: con H = 64, dos capas y FP32 pasa de 64 KiB a 128 KiB, sin contar las ventanas de la convolución, y el presupuesto de bytes lo cuenta.

## Control con el que se descarta

La [declaración](../../configs/evaluation/multiscale-memory-comparison.json) fija tres modelos sobre la receta de 2000 con `mac_online`: el control, la cascada y el banco, con g = 1/16 en los dos. Comparten filas, ventanas, semillas, pesos iniciales, casos de búsqueda, actualizaciones del optimizador, regla de parada, truncación y número de matrices por flujo, y la declaración registra el presupuesto declarado (9 h de GPU) frente al consumido (cero). La cascada se conserva solo si el límite superior del intervalo del 95 % de la diferencia relativa de MAE por sesión frente al banco queda por debajo de −MID, la banda entre semillas del [control de varianza](../research/protocol.md#dispersión-entre-semillas-en-los-pilotos-de-memoria) (regla `pt5`), que comparte el control del piloto común. En otro caso se abandona. La predicción registrada es que el banco igualará a la cascada.

## Coste medido sin entrenar

[`cost.json`](../../reports/engineering/multiscale-memory-20261010/cost.json) mide en CPU, en FP32 y con dos hilos, los tres modelos de la declaración con el mismo script y las mismas entradas reales sin objetivos que PT6: 128 flujos de EE. UU. y 8 instantes. Cada repetición recorre los instantes con grafo, calcula una pinball frente a ceros y ejecuta el backward, sin optimizador, y después repite el recorrido sin grafo. Las 15 repeticiones se alternan entre modelos con orden rotatorio.

| Modelo | Estado por flujo | Forward con grafo | Backward | Inferencia |
| --- | ---: | ---: | ---: | ---: |
| Control | 65.552 B | 0,288 s | 0,141 s | 0,292 s |
| Cascada | 131.088 B | 0,508 s (1,77) | 0,161 s (1,14) | 0,490 s (1,68) |
| Banco EMA | 131.088 B | 0,547 s (1,90) | 0,176 s (1,25) | 0,464 s (1,59) |

Son medianas, con el cociente frente al control entre paréntesis, y con otras cargas en la máquina (carga media entre 12 y 14). Una medida anterior con el mismo script dio cocientes parecidos (1,89 y 1,92 en el forward y 1,67 y 1,70 en la inferencia). Las dos formas casi duplican el forward y la inferencia porque cada escritura avanza tres copias de cada matriz, y el banco además calcula su media en cada lectura. No añaden parámetros. Esta medida corresponde solo a CPU. El ensayo combinado registra costes del recorrido en CUDA, con otras comprobaciones CPU concurrentes y sin ejecutar el banco EMA. No constituye una comparación de coste aislada entre las dos formas. Antes del piloto propio siguen pendientes el perfil de `_advance_levels`, las comprobaciones de finitud y un presupuesto basado en ese recorrido.

## Desviaciones y límites

- Kaplanis et al. (2018) no tienen olvido propio en el primer nivel. Aquí se conserva el α de Titans, que la dirección arquitectónica exige mantener, y los niveles ocultos añaden escalas más largas.
- Kaplanis et al. (2018) inician los niveles ocultos en cero. Aquí empiezan en los pesos iniciales para que la primera lectura coincida con la del núcleo y la cadena no arrastre la memoria hacia cero al principio.
- g = 1/16 se fija antes de ver datos por las escalas que produce. No se ajusta.
- La receta base no lleva PT1, así que la comparación declarada no tiene la garantía de contracción. Si PT1 se conserva en su comparación, los tres modelos se repiten con PT1.

## Pendiente sin medir

- La comparación con el banco EMA, el piloto de 2018 a 2020 y su regla de decisión. La estimación original de nueve horas debe recalcularse con una medida propia antes de ejecutar. El ensayo combinado no consume ni completa esa declaración, y la campaña sigue bloqueada.
- El exponente de Lyapunov de la memoria completa con niveles, que el Jacobiano permite estimar sobre trayectorias reales.

## Referencias

- Benna, M. K. y Fusi, S. (2016). Computational principles of synaptic memory consolidation. *Nature Neuroscience, 19*(12), 1697–1706. https://doi.org/10.1038/nn.4401
- Kaplanis, C., Shanahan, M. y Clopath, C. (2018). Continual reinforcement learning with complex synapses. *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 2497–2506. https://proceedings.mlr.press/v80/kaplanis18a.html
