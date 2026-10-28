# Fiabilidad madura como señal común

`mature_reliability_signal_v1` es la señal común de la segunda ola del estudio holográfico ([#782](https://github.com/GonxKZ/mars-titan/issues/782), [#791](https://github.com/GonxKZ/mars-titan/issues/791)). Mide el acierto de las predicciones realmente emitidas cuando su resultado madura, a tres velocidades, y modula con él las puertas de escritura y olvido de la memoria de Titans. La señal está en `src/mars_titan/memory/mature_reliability.py`, las puertas en `src/mars_titan/models/titans/reliability_gates.py` y la declaración, con sus tres controles, en [su configuración](../../configs/experiments/mature-reliability-signal-v1.json). Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y alcance

Titans escribe en su memoria con la sorpresa asociativa, una señal interna e inmediata que no sabe si la predicción financiera acertó. Ese resultado solo se conoce cuando madura la etiqueta, varias sesiones después de emitir la predicción. La issue pregunta si una única señal causal construida con esos resultados debe gobernar a la vez la escritura y el olvido de la memoria, el tercer factor de P1, la confianza en cada área, los árbitros y la abstención.

Este documento cubre la señal y sus primeros consumidores, las puertas θ y α de la memoria neuronal. Los demás consumidores todavía no existen y se conectarán con la misma señal en sus issues: P1 en [#784](https://github.com/GonxKZ/mars-titan/issues/784), la confianza por área en [#783](https://github.com/GonxKZ/mars-titan/issues/783), los árbitros en [#789](https://github.com/GonxKZ/mars-titan/issues/789) y la abstención en [#787](https://github.com/GonxKZ/mars-titan/issues/787). La sorpresa asociativa de Titans sigue siendo otra señal y solo sustituye a la madura en su control.

## Señal

Sea $\hat y$ una predicción emitida e $y$ su resultado cuando madura. El acierto direccional es

$$
h=\operatorname{sign}(\hat y)\,\operatorname{sign}(y),
$$

que vale 0 si alguno de los dos es nulo. Para el mercado m y el canal c, la puntuación de un evento es la media de los aciertos de las etiquetas que maduran en él, $s_{m,c}(t)=\frac1n\sum_i h_i$, y cada velocidad k la promedia con $\lambda_k=e^{-1/\tau_k}$:

$$
v_{m,c,k}(t)=\lambda_k\,v_{m,c,k}(t-1)+(1-\lambda_k)\,s_{m,c}(t),\qquad v=0\ \text{al empezar}.
$$

Las constantes declaradas son $\tau=(1,5,21)$, una sesión, una semana y un mes. Un evento sin etiquetas del mercado no cambia la señal, así que $\tau$ cuenta eventos de maduración y no sesiones del calendario. La media usa `math.fsum`, que redondea una sola vez, y el acumulador trabaja en FP64. Por eso la señal no depende del orden de las filas ni de su reparto en bloques.

El recorrido cronológico aplica primero las etiquetas de un instante y después emite sus decisiones, así que una decisión del instante t lee exactamente lo madurado hasta t. Todos los bloques de un evento reciben la misma señal. Como el estado rápido, la señal se reinicia en cada recorrido, de modo que cada fase de ajuste, validación o evaluación empieza neutral y la primera etiqueta madura llega un horizonte después de la primera decisión emitida.

La señal es del mercado y no de cada flujo. Con un horizonte de una sesión madura una etiqueta por flujo y evento, y la velocidad de la sesión de un flujo sería casi una moneda de ±1. La media de N aciertos de un mercado sin capacidad predictiva tiene desviación típica $1/\sqrt N$, que con 64 activos es 0,125.

## Puertas

La señal llega a `FinancialPredictor.prepare` como un tensor [filas, consumidores, velocidades] y se convierte en un desplazamiento de logit por consumidor:

$$
u_{c,k}=\operatorname{sign}(r_{c,k})\max(|r_{c,k}|-\mu,0),\qquad
\delta_c=b\tanh\Bigl(\sum_k w_{c,k}\,u_{c,k}\Bigr),
$$

$$
\theta_t=\theta_{\max}\,\sigma(z_\theta+\delta_{\text{escritura}}),\qquad
\alpha_t=\sigma(z_\alpha+\delta_{\text{olvido}}),
$$

con $z_\theta$ y $z_\alpha$ los logits de Titans para la entrada. El margen $\mu=0{,}05$ descarta las desviaciones pequeñas y la cota $b=1$ limita cada desplazamiento a un logit, así que las odds de cada puerta cambian como mucho en un factor $e$. Los dos valores se fijan antes de ejecutar. Los pesos $w$ empiezan en cero, de modo que $\delta=0$ y el predictor parte exactamente de su referencia. Con $w=0$ la derivada de $\delta_c$ respecto a $w_{c,k}$ es $b\,u_{c,k}$, nula mientras la señal no supere el margen. Con la caja de PT1, el desplazamiento se suma al logit antes de la caja y α conserva su suelo. η no se modula y la memoria compartida de PT4 tiene sus propias puertas, que tampoco cambian.

El margen equivale a unas 2,6 desviaciones típicas de la velocidad del mes y a 0,6 de la velocidad de la sesión con 64 activos sin capacidad predictiva, porque la varianza estacionaria de una media exponencial de puntuaciones independientes es $\frac{1-\lambda}{1+\lambda}\cdot\frac1N$. Las mezclas $w$ deciden cuánto cuenta cada velocidad.

## Controles

| Modelo | Cambio respecto a la señal compartida | Pregunta que contrasta |
| --- | --- | --- |
| `reference` | Sin el componente | Si la señal mejora el núcleo |
| `shuffled_in_time` | En la actualización n, la puntuación es la de una de las n + 1 actualizaciones vistas hasta entonces, elegida al azar con una semilla que deriva de la declarada, el mercado, el canal y n | Si importa el orden temporal o basta su distribución |
| `titans_surprise` | La puntuación sale de la pérdida asociativa de Titans en las filas emitidas, $s=(r-\bar\ell)/(r+\bar\ell)$ con $r$ su promedio a la velocidad más lenta. Se registra al emitir y la leen los eventos siguientes | Si el resultado maduro aporta algo frente a la sorpresa interna |
| `independent_signals` | Cada puerta tiene su canal y solo recibe las etiquetas de la mitad de los flujos que le asigna la semilla | Si compartir una señal ruidosa produce fallos correlacionados |

El control barajado es causal: nunca usa una puntuación posterior. Conserva la distribución de las puntuaciones y rompe su orden. El control independiente tiene la misma esperanza que la señal compartida con ruidos independientes, pero cada canal usa la mitad de las etiquetas y es más ruidoso. Cuando existan más consumidores, el reparto se extenderá a todos.

## Correspondencia entre ecuaciones, código y pruebas

| Ecuación o regla | Código | Prueba |
| --- | --- | --- |
| Acierto, media por evento y velocidades | `ReliabilitySignal.settle` y `_update` | `test_each_speed_averages_the_directional_hits_of_the_matured_labels` |
| Invariancia al orden y a los bloques | `math.fsum` en `settle` | `test_the_signal_does_not_depend_on_the_order_or_split_of_the_labels` |
| Mercados separados y eventos sin etiquetas | `_groups` y `values` | `test_markets_keep_separate_signals_and_events_without_labels_do_not_move_them` |
| Control barajado causal | rama `shuffled_in_time` de `_update` | `test_the_shuffled_control_draws_each_input_from_the_scores_seen_so_far` |
| Control de sorpresa | `ReliabilitySignal.observe` y `PreparedDecisions.surprise` | `test_the_surprise_control_compares_each_event_with_its_slowest_average`, `test_the_surprise_control_returns_the_associative_loss_of_each_row` |
| Reparto independiente | `ReliabilitySignal.channel` | `test_independent_signals_split_the_flows_between_the_consumers` |
| Margen, cota y desplazamiento | `ReliabilityGates.forward` | `test_each_shift_stays_within_the_bound_and_vanishes_inside_the_margin` |
| Desplazamiento de θ y α | `gate_shift` de `NeuralMemory.update` | `test_the_signal_shifts_the_write_and_forget_logits_by_the_declared_amount` |
| Paridad al empezar | mezclas nulas | `test_a_declared_signal_starts_exactly_at_the_reference`, `test_gradients_match_the_reference_and_reach_the_mixing_of_each_gate` |
| Lectura de lo madurado hasta la decisión | `_labels` y `_observe` del recorrido | `test_each_decision_reads_the_labels_matured_before_it_and_nothing_later`, `test_the_signal_changes_predictions_only_after_a_label_matures` |
| Recuperación | `export`, `restore` y el checkpoint del recorrido | `test_a_restored_signal_continues_exactly_like_the_original`, `test_resume_after_interruption_reproduces_the_continuous_run` |
| Repetición por bloques | señal guardada con cada lote del tramo | `test_replay_by_flow_blocks_keeps_the_signal_of_each_emission` |

## Paridad e identidad

Sin declarar el componente, la identidad del predictor y el cálculo no cambian. Al declararlo, las predicciones, la memoria y los gradientes de los parámetros compartidos coinciden bit a bit con la referencia mientras las mezclas valgan cero, con cualquier señal, y las mezclas reciben gradiente en cuanto la señal supera el margen. La identidad del predictor registra la fuente, las constantes, el margen, la cota, los consumidores, el reparto y la semilla. El recorrido añade a su identidad de código el módulo de la señal solo cuando el componente está declarado, y su checkpoint guarda los acumuladores de cada mercado con su configuración.

El componente solo existe con `mac_online`. Se rechaza con el control local C de CM-v1, que vuelve a calcular la escritura de MAC sin la señal, y en el consumidor congelado, que todavía no la recibe.

## Desviaciones y límites

- El acierto direccional ignora la magnitud del error. Es acotado y robusto a las colas, pero una predicción correcta por poco cuenta igual que una muy precisa.
- La señal es del mercado. Todos los flujos de un mercado reciben el mismo desplazamiento, como un neuromodulador global, y no hay una señal por activo.
- Las velocidades cuentan eventos de maduración. Tras un periodo sin etiquetas la señal no decae.
- Cada fase empieza neutral, así que las primeras decisiones de validación y evaluación no se modulan.
- El control de sorpresa cambia también el momento de la señal, porque la sorpresa se conoce al emitir. Es una diferencia buscada, ya que la madura no puede adelantarse.
- Solo están conectadas la escritura y el olvido. η, PT4, el consumidor congelado y CM-v1 C quedan fuera.
- La señal es una adaptación propia. Las puertas dependientes de la entrada son las de Titans (Behrouz, Zhong y Mirrokni, 2024) y la modulación de la plasticidad por un resultado conocido con retraso viene de las reglas de tres factores (Gerstner y otros, 2018). No reproduce ningún resultado publicado.

## Coste

Pendiente de la medida en CPU con `benchmarks/mature_reliability_cost.py`, sin pasos de optimizador. La medida CUDA queda para cuando la GPU esté libre.

## Comprobación

Pendiente de la verificación final.

## Pendiente

- Conectar P1, la confianza por área, los árbitros y la abstención cuando existan.
- Registrar la señal en el protocolo de la segunda ola antes de abrir S, con unas 4,4 h de GPU por modelo con cinco semillas.

## Referencias

- Behrouz, A., Zhong, P. y Mirrokni, V. (2024). *Titans: Learning to memorize at test time*. arXiv. https://arxiv.org/abs/2501.00663
- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D. y Brea, J. (2018). Eligibility traces and plasticity on behavioral time scales: Experimental support of neoHebbian three-factor learning rules. *Frontiers in Neural Circuits, 12*, 53. https://doi.org/10.3389/fncir.2018.00053
