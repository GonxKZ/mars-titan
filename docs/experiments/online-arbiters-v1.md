# Árbitros en línea con resultados maduros

`online_arbiters` reúne el núcleo de los árbitros en línea de la segunda ola ([#789](https://github.com/GonxKZ/mars-titan/issues/789), épica [#782](https://github.com/GonxKZ/mars-titan/issues/782)). `src/mars_titan/evaluation/online_arbiters.py` implementa AdaNormalHedge y su forma TV, y sobre ese núcleo el árbitro de versiones `version_arbiter_v1` y la combinación de modelos `model_combination_v1`, declarados en [su configuración](../../configs/experiments/online-arbiters-v1.json). Solo se han comprobado con secuencias construidas y paneles ficticios. Ninguno se ha ejecutado sobre predicciones de la criba y no hay ninguna mejora medida.

## Motivación y alcance

La [enmienda de selección](variant-screening-v1.md) elige una época común por grupo con el IC de validación y emite además, en calibración y en S, las versiones de las épocas 3, 6, 9 y 12 de cada ajuste. Una época fijada antes de S puede dejar de ser la mejor a lo largo del año. El árbitro de versiones pregunta si repartir la confianza entre esas versiones con sus resultados ya maduros ordena mejor los activos de cada sesión que la época fija. La idea es propia. Se inspira también en el arbitraje por incertidumbre de [Daw, Niv y Dayan (2005)](https://doi.org/10.1038/nn1560), donde en cada momento decide el sistema de control más fiable. Aquí la fiabilidad es el IC maduro de cada versión y no hay ningún modelo de incertidumbre.

La combinación de modelos aplica el mismo núcleo a todos los modelos comparados, a partir de la combinación de previsiones de [Bates y Granger (1969)](https://doi.org/10.1057/jors.1969.103). Se informa aparte y no se atribuye a MARS-TITAN. Su control obligatorio es la media simple, porque las combinaciones ponderadas suelen perder frente a ella con muestras cortas ([Stock y Watson, 2004](https://doi.org/10.1002/for.928), [Smith y Wallis, 2009](https://doi.org/10.1111/j.1468-0084.2008.00541.x)).

El árbitro fractal y el de herramientas registradas quedan pendientes, como se explica al final.

## Núcleo AdaNormalHedge

AdaNormalHedge ([Luo y Schapire, 2015](https://arxiv.org/abs/1502.05934)) no tiene parámetros que ajustar. Con N opciones, prior q, pérdidas $\ell_{t,i}\in[0,1]$ e indicadores de disponibilidad $I_{t,i}$, el algoritmo 1 del artículo usa el potencial y el peso de su ecuación (3),

$$
\Phi(R,C)=\exp\!\left(\frac{[R]_+^2}{3C}\right),\quad \Phi(0,0)=1,\qquad
w(R,C)=\tfrac12\bigl(\Phi(R+1,C+1)-\Phi(R-1,C+1)\bigr),
$$

y en cada ronda

$$
p_{t,i}\propto q_i\,I_{t,i}\,w(R_{t-1,i},C_{t-1,i}),\qquad
r_{t,i}=I_{t,i}\bigl(\langle p_t,\ell_t\rangle-\ell_{t,i}\bigr),\qquad
R_{t,i}=\sum_{s\le t}r_{s,i},\quad C_{t,i}=\sum_{s\le t}|r_{s,i}|.
$$

`weights` calcula w como $e^{b}\,\operatorname{expm1}(a-b)/2$, con a y b los exponentes de los dos potenciales, para no restar dos exponenciales casi iguales cuando C crece. Si todas las opciones despiertas tienen peso cero, lo que ocurre cuando todas tienen R ≤ −1, el artículo deja la mezcla libre (nota 1) y aquí se usa el prior de las despiertas. Una ronda sin opciones despiertas no mezcla nada ni cambia el estado.

La garantía es sobre el arrepentimiento de la pérdida lineal de la mezcla. El lema 2 acota $\sum_i q_i\Phi(R_{t,i},C_{t,i})\le B$, con $B=1+\tfrac32\sum_i q_i\bigl(1+\ln(1+C_{t,i})\bigr)$. Para cualquier mezcla u, el teorema 1 da

$$
R_t(u)\le\sqrt{3\,(u\cdot C_t)\,\bigl(\operatorname{RE}(u\,\Vert\,q)+\ln B+\ln(1+\ln N)\bigr)},
$$

y para una u uniforme sobre un subconjunto cambia el último término por 1. El teorema 3 extiende las dos cotas a opciones que duermen.

Para competir con una mejor opción que cambia en el tiempo se usa AdaNormalHedge.TV, la reducción de la sección 5.1. Cada ronda τ despierta una copia de cada opción con prior $q_i/\tau^2$, y la mezcla suma los pesos de las copias,

$$
p_{t,i}\propto q_i\,I_{t,i}\sum_{\tau\le t}\tau^{-2}\,w\bigl(R_{[\tau,t-1],i},\,C_{[\tau,t-1],i}\bigr),
$$

donde $R_{[\tau,t-1],i}$ y $C_{[\tau,t-1],i}$ suman desde la ronda τ. Así la cota vale para cualquier intervalo de rondas, y el horizonte completo es el caso de la mejor opción fija. Cada ronda cuesta O(N t). En una secuencia construida en la que la mejor opción cambia en la ronda 100, la forma TV supera un peso de 0,9 en la nueva opción en la ronda 143 y la forma fija no lo hace hasta la 217. Los dos árbitros usan la forma TV.

## Rondas maduras

Cada sesión es una ronda. La pérdida de la opción i en la sesión s es $\ell_{s,i}=(1-\mathrm{IC}_{s,i})/2$, con el IC de Pearson entre su predicción y el objetivo de los activos de la sesión. La issue pide el IC negativo. Restar el mismo valor a todas las opciones no cambia el arrepentimiento instantáneo, porque la mezcla suma uno, y el factor 1/2 lleva la pérdida al intervalo [0, 1] que exige el algoritmo. Puntúa una sesión con al menos tres activos y objetivo no constante. Una opción constante en una sesión que puntúa tiene IC 0.

Las etiquetas maduran días después de decidir. `mature_mixture` aplica cada sesión cuando madura, en orden de madurez y después de decisión, y antes de decidir una sesión aplica todas las que cumplen `matures_at <= decided_at`, la convención del resto de etapas. Cada sesión se actualiza con la mezcla vigente al aplicarla. Así el algoritmo es exactamente AdaNormalHedge sobre la secuencia de rondas maduras y la cota vale para esas mezclas. No cubre la mezcla emitida al decidir cada sesión, que se calculó con menos resultados. Los instantes de decisión de un mismo estado no se repiten, de modo que el resultado no depende del orden en que lleguen las filas.

## Árbitro de versiones

`online_mixture` mantiene un estado TV por mercado, con prior uniforme, porque las sesiones de US y CN pueden decidirse en el mismo instante y tienen calendarios distintos. Las predicciones de cada opción se tipifican dentro de cada sesión antes de mezclarlas, así que la mezcla compara ordenaciones y no amplitudes. La amplitud final la calibra el λ de cada mercado, elegido con la salida del árbitro en calibración y aplicado sin cambios en S, como en el resto de la criba. El estado recorre una sola vez la calibración y después S. Si la época común coincide con una de las versiones, cuenta como una sola opción para no duplicar su prior.

La fiabilidad de una opción al decidir es su IC medio en las últimas 21 sesiones maduras que puntúan, un mes de mercado. Si ninguna opción supera el margen 0, la sesión se abstiene y predice cero. También se abstiene mientras no haya ninguna sesión madura. La ventana y el margen están fijados en la declaración antes de abrir datos y no se ajustan. Para no medir el árbitro solo en las sesiones que elige, cada sesión abstenida cuenta como IC 0 y se informa la cobertura.

Los controles son tres.

- La época común emitida, sin árbitro.
- La media uniforme de las opciones tipificadas, sin abstención (`uniform_mixture`).
- El mismo árbitro con los resultados desordenados en el tiempo. Cada sesión recibe una clave aleatoria fija con la semilla 20261028. En cada decisión el estado se reconstruye desde cero con las sesiones ya maduras ordenadas por su clave, y la fiabilidad usa las últimas 21 de ese orden. El control pierde el orden temporal de los resultados y conserva cuáles se conocen en cada momento, así que tampoco usa ninguno sin madurar. La primera versión de este control desordenaba todas las sesiones del panel y dejaba entrar resultados futuros, y se rehízo antes de declararlo.

El árbitro frente a la época común mide si repartir la confianza mejora la ordenación. Frente a la media uniforme separa la ponderación por resultados de la simple combinación de versiones. Frente al control desordenado separa el seguimiento en el tiempo de la fiabilidad media.

## Combinación de modelos

`model_combination_v1` usa el mismo `online_mixture` sin margen, de modo que nunca se abstiene, con todos los modelos comparados del mismo origen y semilla. Su control es la media simple de los modelos tipificados. El resultado se informa aparte y no cuenta como una variante de MARS-TITAN.

## Declaración

`stage_options` lee la declaración antes de abrir ningún dato. Comprueba que usa AdaNormalHedge.TV, que está desactivada por defecto y sin acceso al test final, que la ventana tiene de 1 a 252 sesiones, que cada sesión necesita al menos tres activos y que cada etapa declara sus controles. El árbitro de versiones necesita un margen finito y la semilla de su control desordenado. La combinación de modelos no admite margen y su único control es la media simple.

## Correspondencia con el código y las pruebas

Las pruebas están en `tests/evaluation/test_online_arbiters.py`.

| Elemento | Código | Pruebas |
| --- | --- | --- |
| Φ y w, ecuación (3) | `potential`, `weights` | `test_the_weight_is_the_difference_of_potentials_of_the_article` |
| Algoritmo 1 | `AdaNormalHedge.distribution`, `update` | `test_the_rounds_match_a_reference_written_option_by_option` |
| Lema 2 | `AdaNormalHedge.potential_bound` | `test_the_potential_respects_the_bound_of_lemma_2_after_every_round` |
| Teorema 1, cotas (1) y (2) | estado tras la secuencia | `test_the_regret_to_any_fixed_mixture_respects_theorem_1` |
| Opciones que duermen y nota 1 | máscara `awake`, prior de las despiertas | `test_a_sleeping_option_gets_no_weight_and_keeps_its_state`, `test_a_round_without_options_mixes_nothing_and_changes_nothing`, `test_the_prior_decides_when_every_awake_weight_is_zero` |
| Sección 5.1 y teorema 3 | `AdaNormalHedgeTV` | `test_the_tracking_form_matches_the_interval_sums_of_the_article`, `test_the_tracking_form_respects_lemma_2_and_the_bound_on_every_interval`, `test_the_tracking_form_follows_a_change_that_the_fixed_one_misses` |
| Rondas maduras | `mature_mixture` | `test_only_mature_losses_reach_a_decision`, `test_the_mixtures_do_not_depend_on_the_order_of_the_sessions`, `test_a_session_without_lag_is_applied_before_the_next_decision` |
| Tipificación y pérdida | `standardized`, `session_losses` | `test_each_session_is_standardized_and_a_constant_one_is_zero`, `test_the_loss_is_half_of_one_minus_the_session_ic` |
| Fiabilidad y abstención | `_reliability`, `online_mixture` | `test_the_reliability_is_the_mean_ic_of_the_last_mature_sessions`, `test_the_arbiter_abstains_when_no_version_is_reliable_enough`, `test_without_a_margin_the_mixture_never_abstains` |
| Árbitro por mercado | `online_mixture` | `test_the_arbiter_moves_to_the_version_that_ranks_the_assets`, `test_each_market_has_its_own_state`, `test_no_future_target_reaches_a_decision` |
| Control desordenado | `_shuffled` | `test_the_shuffled_control_loses_the_order_but_not_the_known_outcomes`, `test_the_arbiter_follows_a_change_of_the_better_version_and_its_control_does_not` |
| Media uniforme | `uniform_mixture` | `test_the_uniform_control_is_the_mean_of_the_standardized_options` |
| Declaración | `stage_options` | `test_the_declared_stages_fix_their_options_before_any_data`, `test_a_declaration_that_breaks_its_contract_is_rejected` |

## Desviaciones respecto al artículo

| Luo y Schapire (2015) | Árbitros | Motivo |
| --- | --- | --- |
| La pérdida de cada ronda se conoce al terminarla | La sesión se aplica al madurar, con la mezcla vigente entonces | Las etiquetas maduran días después. La cota cubre esas mezclas y no las emitidas al decidir |
| Un único problema de opciones | Un estado por mercado | Instantes de decisión únicos y calendarios separados |
| Pérdida en [0, 1] | (1 − IC)/2 de la predicción tipificada por sesión | Equivale al IC negativo salvo un desplazamiento, que no cambia el arrepentimiento, y un factor 1/2 |
| Sin abstención | Abstención con margen fijo sobre el IC maduro de las 21 últimas sesiones | La pide la issue. Queda fuera de la garantía y por eso se puntúa como IC 0 |
| Prior 1/τ² de las copias en la forma TV | El mismo, normalizado sobre las rondas jugadas y la siguiente para comprobar el lema 2 | Es un prior fijo válido para ese horizonte |

## Comprobación técnica

La comprobación del 28 de octubre se hizo en CPU, sin pasos de optimizador y sin leer predicciones de la criba.

- Las 70 pruebas del núcleo y de los árbitros y las 3 de su medida de coste pasan con medida de cobertura. Con coverage 7.16.1 y radon 6.0.1, `online_arbiters.py` cubre 238/238 sentencias y 42/42 ramas, y el CRAP máximo es 19 (`stage_options`, que encadena las condiciones de la declaración). En la medida de coste solo quedan sin cubrir la entrada de línea de órdenes y la recogida del entorno.
- Las cotas se comprueban sin ajustar nada. El lema 2 se cumple después de cada ronda en secuencias aleatorias, con opciones que duermen, adversarias y constantes. El teorema 1 se cumple frente a cada opción y a veinte mezclas de Dirichlet, y en la forma TV se cumplen el lema 2 y la cota de cada intervalo. Las mezclas coinciden con una referencia escrita opción a opción con una tolerancia relativa de 1e-9, la que permite la cancelación de la referencia cerca de R = −1.
- Ninguna mezcla, fiabilidad, abstención ni predicción de una sesión cambia al redibujar los objetivos que todavía no han madurado al decidirla, ni en el árbitro ni en su control desordenado. Tampoco cambia al alterar las pérdidas no maduras, y las mezclas no dependen del orden de entrega de las sesiones.
- Dieciocho mutaciones dirigidas se detectan con los controles en verde antes y después. Cubren aplicar pérdidas no maduras, retrasar la sesión que madura justo al decidir, quitar la parte positiva del peso, acumular la magnitud con signo en las dos formas, sustituir el prior de la nota 1 por uno uniforme, acumular arrepentimiento en opciones dormidas, quitar el prior 1/τ², premiar el IC negativo, puntuar sesiones con dos activos, calcular la fiabilidad con todas las sesiones maduras, ignorar el margen, mezclar predicciones sin tipificar, compartir el estado entre mercados, que el control desordenado lea sesiones no maduras o conserve el orden temporal, y aceptar una declaración activada o una semilla negativa.
- La regresión de las pruebas de evaluación, de las medidas de coste y de las protecciones del bloqueo pasa sobre el árbol final, tras integrar C5 y la corrección de [la PR #795](https://github.com/GonxKZ/mars-titan/pull/795). Da 1.674 casos superados y 7 omitidos porque falta `scoringrules`, del grupo opcional `reference`.

El [recibo de verificación](../../reports/engineering/online-arbiters-20261028/verification.json) conserva los recuentos y las huellas de los informes JUnit, de la cobertura, de las mutaciones y de la medida de coste.

## Coste

La medida CPU del 28 de octubre llama a `online_mixture` y `uniform_mixture` en FP64 sobre paneles ficticios con dos mercados, 128 activos por sesión y la madurez de cada etiqueta cinco sesiones después de decidir. Con cinco opciones (la época común y cuatro versiones) mide el árbitro de versiones, su control desordenado y la media uniforme. Con 32 opciones mide la combinación de modelos y su media simple. Se usó un hilo de BLAS, un calentamiento y cinco repeticiones en cada uno de los dos órdenes de los modos. El equipo tenía otras cargas, entre ellas las mutaciones de la enmienda, con una media de carga de un minuto de 9,9 al empezar y 7,1 al terminar, así que los percentiles son descriptivos. La memoria es el pico de tracemalloc en una llamada aparte. El [recibo](../../reports/engineering/online-arbiters-20261028/cost-cpu.json) conserva todas las muestras.

| Sesiones por mercado | Opciones | Modo | Mediana | p95 | Pico de memoria |
| ---: | ---: | --- | ---: | ---: | ---: |
| 64 | 5 | Árbitro de versiones | 0,09 s | 0,10 s | 1,3 MB |
| 64 | 5 | Control desordenado | 0,44 s | 0,53 s | 1,3 MB |
| 64 | 5 | Media uniforme | 0,02 s | 0,03 s | 1,4 MB |
| 320 | 5 | Árbitro de versiones | 0,48 s | 0,55 s | 6,6 MB |
| 320 | 5 | Control desordenado | 11,61 s | 23,09 s | 6,6 MB |
| 320 | 5 | Media uniforme | 0,11 s | 0,13 s | 7,2 MB |
| 320 | 32 | Combinación de modelos | 2,68 s | 3,12 s | 42,0 MB |
| 320 | 32 | Media simple | 0,74 s | 0,87 s | 42,6 MB |

Las 320 sesiones corresponden a la calibración y un año de S. El control desordenado reconstruye el estado en cada decisión, y como cada ronda TV cuesta O(N t), su coste crece como O(N T³) con T sesiones por mercado. Con cinco modelos con versiones, cinco semillas y cuatro orígenes son 100 ajustes, unos 19 minutos de CPU con la mediana medida, frente a las unas 97 horas de GPU de la criba. No se ha optimizado. Cubrir los intervalos con una familia geométrica reduciría el número de copias, pero cambiaría el algoritmo declarado. En la combinación, la mayor parte del tiempo se va en el IC de cada sesión y opción, que reutiliza `scale_diagnostics.session_correlations` para medir igual que el resto de la criba. No se han medido la energía ni el coste en GPU, porque los árbitros solo usan CPU.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 uv run python -m benchmarks.online_arbiters_cost --run \
  --output "${MARS_ONLINE_ARBITERS_COST:?}"
```

## Pendiente

- La etapa del análisis que lee las versiones de la criba, con el λ de calibración, la abstención puntuada como IC 0 y la cobertura, y su entrada en el protocolo de la segunda ola, que #782 exige registrar antes de abrir S.
- `fractal_arbiter_v1`, sobre la ponderación por árboles de contexto ([Willems, Shtarkov y Tjalkens, 1995](https://doi.org/10.1109/18.382012)) y la mejor poda de un árbol ([Helmbold y Schapire, 1997](https://doi.org/10.1023/A:1007396710653)). Necesita los niveles de H1 de [#783](https://github.com/GonxKZ/mars-titan/issues/783).
- `tool_arbiter_v1`, a partir del enrutador TRA ([Lin y otros, 2021](https://doi.org/10.1145/3447548.3467358)). Necesita las predicciones de cada acción registrada.

El aprendizaje sigue bloqueado y los árbitros no se activan por defecto.
