# Rangos por sesión y pérdida de IC con cabeza de escala

Este documento describe F4 del estudio de variantes de [#742](https://github.com/GonxKZ/mars-titan/issues/742), que se implementa en [#749](https://github.com/GonxKZ/mars-titan/issues/749). Son dos componentes opcionales de `FinancialConfig` con identidad propia:

- `session_rank_inputs_v1` sustituye los precios y los fundamentales por las puntuaciones normales de sus rangos dentro de cada sesión.
- `session_ic_objective_v1` ajusta la ordenación con una pérdida de correlación por sesión y devuelve la predicción a unidades de retorno con una escala separada.

Los dos están integrados en el recorrido cronológico y se comprueban con fixtures en CPU. La comprobación CUDA está pendiente. No se han entrenado ni evaluado sobre datos financieros. Nada de lo que sigue es una mejora observada.

## Motivación

La [fase 0](../../reports/research/scale-diagnostics-20261023/README.md) reanalizó las predicciones guardadas de los pilotos de octubre. Con la familia declarada, ningún predictor de la población principal tiene un IC medio distinguible de cero. Las pendientes de calibración indican además que las predicciones tienen entre 65 y 165 veces la amplitud que les corresponde. F4 separa dos objetivos de ajuste: ordenar los activos de una sesión y acertar su amplitud. La ordenación usa una aproximación regularizada de la correlación y la amplitud, un parámetro ajustado por error absoluto. Esta separación puede perder información útil y necesita su contraste frente al control MAE.

## Entradas en rangos gaussianos por sesión

Sea x una columna numérica y n el número de filas de la sesión en que está presente. Con r el rango de cada fila, con empates promediados:

$$g_i=\Phi^{-1}\left(\frac{r_i-1/2}{n}\right).$$

La fórmula fija la transformación inversa normal de esta variante. La revisión de [Beasley, Erickson y Allison (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2921808/) advierte que las transformaciones inversas normales no garantizan mejor control del error ni más potencia estadística en sus aplicaciones genéticas. No aporta evidencia de mejora financiera. [Gu, Kelly y Xiu (2020), nota 29](https://academic.oup.com/rfs/article/33/5/2223/5758276) ordenan las características de cada mes en sección cruzada y llevan el rango a [−1, 1]. Aquí se usa una puntuación normal por sesión, con máscaras y sobre un objetivo residual diario.

Se transforman:

- los cinco canales de la ventana de precios, columna a columna para cada una de las 64 sesiones de la ventana, entre las filas cuyo bit de presencia vale uno en esa sesión.
- los valores de los fundamentales cuya máscara vale uno.

No cambian el bit de presencia de la ventana, las máscaras y edades de los fundamentales, los bits de presencia de las modalidades, los vectores de noticias y gráficos ni el contexto macro. El contexto macro es igual para todos los activos de la sesión y su rango sería constante. Los vectores de los codificadores congelados no tienen un orden transversal con significado. Una posición ausente sigue valiendo cero. La sesión es el par (mercado, instante de decisión), así que un lote con filas de dos mercados se transforma por separado en cada uno.

### Causalidad y sesión completa

Un `DecisionBatch` tiene por contrato un único instante de decisión y entradas con `input_available_at` no posterior a él. La transformación solo lee filas del lote, de modo que no puede usar datos de otra sesión ni del futuro. Las filas de un instante son las muestras con entradas en él. El lector no las filtra por la aceptación de su etiqueta, así que la población de la sesión no depende de la sesión siguiente.

Si el lote no contiene todas las filas del instante, los rangos serían los de un subconjunto. Por eso `FinancialPredictor.prepare` exige `session_complete=True` cuando la configuración declara una operación por sesión y rechaza la llamada en caso contrario. `ChronologicalInference._observe` y el lector de MARS-TITAN solo lo declaran cuando el evento cabe en un único bloque. La repetición de los adaptadores conserva todas las filas observadas, incluidas las que aún no tienen etiqueta. Rechaza un presupuesto de flujos que obligaría a fragmentarlas. El consumidor congelado admite la declaración explícita de sesión completa.

Los rangos de entrada se han conectado con M0, M1 y la repetición de adaptadores del núcleo. El objetivo IC puede consumirse congelado sin lector. Su combinación con el lector episódico sigue pendiente: el lector utiliza la cabeza escalar como retorno y no incorpora todavía la normalización y la pérdida del objetivo IC. Esa combinación se rechaza, también en postentrenamiento, para no interpretar una puntuación de ordenación como retorno financiero. Las pruebas verifican la recuperación del lector con entradas en rangos y el aislamiento de los parámetros del padre.

La alternativa de precomputar los estadísticos por sesión en la preparación de la vista no hace falta en el recorrido cronológico. El lector entrega todas las filas de un instante juntas y la receta histórica usa bloques de 1.024 filas. Sería necesaria si un consumidor tuviera que repartir una sesión entre varias llamadas. Entonces la vista tendría que guardar columnas derivadas con su huella, y la misma prueba de sufijo futuro tendría que comprobar que no cambian las filas anteriores.

## Pérdida de IC por sesión con escala separada

La cabeza escalar del predictor conserva sus pesos iniciales y pasa a emitir una puntuación s. Sea T el conjunto de sesiones del tramo con al menos `minimum_rows` etiquetas maduras y g(y) las puntuaciones normales de los rangos de sus objetivos. La correlación regularizada es $\tilde\rho_t=\operatorname{media}(z(s)z(g(y)))$, con la normalización z definida debajo:

$$\mathcal L=-\frac1{|T|}\sum_{t\in T}\tilde\rho_t\bigl(s,g(y)\bigr)+\frac{w}{N}\sum_i\lvert y_i-\hat y_i\rvert.$$

La predicción emitida es

$$\hat y_i=\operatorname{softplus}(\kappa)\,z_i,\qquad z_i=\frac{s_i-\bar s_t}{\sqrt{\operatorname{var}_t(s)+\varepsilon_t}},\qquad \varepsilon_t=10^{-10}\,\overline{s^2_t}+\operatorname{eps}(\mathrm{dtype})^2,$$

con s sin gradiente en z y estadísticos calculados con todas las filas de la sesión en la emisión. El primer término solo ajusta la puntuación y el tronco. El segundo solo ajusta κ. Como κ es común a todas las filas y softplus(κ) > 0, la predicción conserva el orden de s y el IC emitido es el de la puntuación.

El centrado retira un componente común de la sesión, por ejemplo uno que solo dependa del contexto macro. La división usa un suelo y no fija exactamente la varianza en uno. Su término relativo reduce la amplitud cuando la dispersión es muy pequeña frente al nivel de s. El término absoluto usa eps² para evitar que puntuaciones nulas produzcan gradientes cuya norma desborde en FP32. La implementación divide primero por una potencia de dos próxima al mayor valor absoluto, calcula las medias en esas unidades y reescala también el suelo. Así evita elevar números grandes al cuadrado y conserva la misma función matemática.

Esta pérdida no es la correlación de Pearson exacta. El suelo relativo depende del nivel de s y el absoluto rompe la invariancia exacta a su escala. Una sesión con objetivo constante aporta cero al término de ordenación. La evaluación sigue calculando su IC con las reglas del protocolo, por separado de esta pérdida. En aritmética exacta, una escala positiva y común conserva el orden dentro de la sesión. En coma flotante esa afirmación tiene los límites del redondeo y del rango representable.

κ empieza en softplus(κ₀) = `initial_scale` = 10⁻³, cerca de predecir cero. La escala que corresponde a la evaluación con MAE sin calibrar es la que aprende κ. La calibración común con λ de G se aplica después a la predicción emitida, igual que en cualquier otro modelo.

En el recorrido cronológico, todas las etiquetas de una sesión maduran en el mismo evento y se resuelven antes de la actualización de ese evento. Una sesión entra entera en una sola actualización, como comprueba la prueba del recorrido.

### Desviaciones y límites

- Es una derivación propia del contraste principal de la criba, no la reproducción de un artículo.
- La correlación usa las puntuaciones normales del objetivo y la puntuación continua del modelo, que es derivable. No es el IC de Spearman, que exigiría también los rangos de s.
- El gradiente de ρ respecto a s crece como 1/σ_t(s). El recorte global de la norma del gradiente lo acota, y con él también el gradiente de κ dentro del mismo paso.
- La escala es común a todos los mercados. Con varios mercados, la calibración por mercado de G corrige la amplitud de cada uno.
- Las sesiones con menos de `minimum_rows` etiquetas maduras no entran en el primer término. Con la población de la criba casi todas las sesiones superan el mínimo de 8.
- Con la memoria v1 sin residual ni LayerNorm, la salida de MAC se iguala entre activos a lo largo del recorrido ([escala de la salida de MAC](../engineering/titans-mac-output-scale.md)). En el fixture técnico la dispersión de s cae entonces por debajo del suelo y la predicción vale cero. Por eso la prueba del recorrido usa la memoria de la receta histórica.
- La pérdida no es una media de términos por fila. La acumulación por bloques exige repetir todos los flujos del tramo en un solo bloque.

## Correspondencia entre ecuaciones, módulo y prueba

| Ecuación o regla | Módulo | Prueba |
| --- | --- | --- |
| g = Φ⁻¹((r − 1/2)/n) con empates y ausencias | `models/titans/session_ranks.py`, `gaussian_ranks` | `test_gaussian_ranks_match_the_reference_with_ties_and_absent_rows`, contra `scipy.stats.rankdata` y `scipy.special.ndtri` |
| Invariancia al orden de las filas y a transformaciones monótonas | `gaussian_ranks` | `test_ranks_ignore_row_order_and_monotone_changes_of_a_column` |
| Precios por sesión de la ventana y fundamentales observados, por mercado | `rank_session_inputs` | `test_session_inputs_rank_price_steps_and_fundamental_values_of_each_market` |
| El predictor con rangos equivale a la referencia sobre entradas transformadas a mano | `FinancialPredictor._prepare`, `DecisionBatch.derived` | `test_ranked_predictor_equals_the_reference_on_inputs_ranked_by_hand` |
| ŷ = softplus(κ) z, conserva el orden | `models/titans/session_ic_objective.py`, `SessionScale` | `test_scale_head_keeps_the_order_and_standardizes_each_market` |
| L con ρ_t sobre puntuaciones normales y sesiones mínimas | `session_ic_loss`, `session_correlations` | `test_session_loss_matches_pearson_on_normal_scores_and_skips_small_sessions`, contra `scipy.stats.pearsonr` |
| ∂(−ρ/\|T\|)/∂s_i = −(ẑ_g,i − ρ ẑ_s,i)/(\|T\| n σ_s) y ∂L/∂ŷ_i = −w sign(y_i − ŷ_i)/N | `session_ic_loss` | `test_session_loss_gradients_pass_gradcheck_and_match_the_pearson_derivative`, con `gradcheck` en FP64 |
| El ranking llega al tronco y no a κ, el error absoluto solo a κ | `SessionScale.forward` | `test_ranking_reaches_the_trunk_and_the_absolute_error_only_the_scale` |

Las pruebas del recorrido están en `tests/training/test_financial_run_session_objective.py`. Comprueban con un optimizador que solo registra gradientes que cada actualización recibe sesiones enteras, que el gradiente llega al tronco y a κ, que un instante repartido en dos bloques se rechaza, que la repetición por bloques reproduce los gradientes directos y rechaza grupos parciales, y que alterar el sufijo futuro del corpus no cambia predicciones ni gradientes anteriores al corte.

## Paridad, estado y recuperación

Sin declarar los campos, `FinancialConfig.identity()` no añade ninguna clave, el predictor no crea módulos nuevos y sus predicciones coinciden bit a bit con las de la referencia. Con ellos, todos los parámetros de la referencia conservan sus valores iniciales con la misma semilla y solo se añade `output_objective.scale`, que no consume el generador global. κ es un parámetro compartido, así que viaja en el `state_dict` del checkpoint con el resto del modelo. No hay estado por flujo ni por sesión que guardar. La prueba de ida y vuelta reproduce la predicción tras cargar el estado y rechaza un estado con otra configuración.

## Controles

La [declaración](../../configs/experiments/session-rank-objective-v1.json) fija como control la misma receta con la cabeza escalar y la pérdida MAE, sin cambios del predictor. Las tres variantes son los rangos solos, la pérdida de IC sola y ambas juntas. Todas son `mac_online`, con la política con máscaras, desactivadas por defecto y sin acceso al test final. El entrenador rechaza el objetivo si la receta no usa `mae` o declara destilación, porque el control debe ser el error absoluto de la misma cabeza. Una variante se descarta si su IC medio por sesión no supera al del control con el contraste declarado de la criba, o si su MAE calibrado empeora sin una mejora de IC que lo compense.

## Coste medido

La medida CPU del 26 de octubre usa 256 activos, ocho instantes, 64 sesiones de precios, 384 componentes de noticias, 512 de gráficos, 26 conceptos fundamentales y 140 indicadores macro. El predictor tiene dimensión oculta 64. Se usaron dos hilos, FP32 estricto, SDPA MATH, un calentamiento y cinco repeticiones con orden rotatorio. Son entradas sintéticas y ningún parámetro compartido cambia. El [recibo](../../reports/engineering/session-objectives-20261026/cost-256.json) conserva los tiempos individuales, las huellas de código y la carga concurrente.

| Variante | Parámetros adicionales | Forward y pérdida | Backward y comprobación de finitud | Inferencia de los ocho instantes | Pico neto de ATen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control MAE | 0 | 0,850 s | 0,338 s | 0,822 s | 504,9 MB |
| Rangos de entrada | 0 | 1,014 s | 0,383 s | 1,039 s | 505,6 MB |
| Objetivo por sesión | 1 | 0,915 s | 0,425 s | 0,901 s | 505,0 MB |
| Ambos | 1 | 1,056 s | 0,417 s | 1,110 s | 505,6 MB |

Los tiempos son medianas. La ejecución completa, incluida preparación y perfilado, duró 63,0 s y alcanzó 1,93 GB de RAM. El pico neto de ATen no incluye todas las asignaciones del proceso y no equivale al pico de RAM. No se han medido VRAM ni energía. Esta medida describe el coste técnico de las variantes, no el tiempo de un entrenamiento ni su utilidad predictiva.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run python -m benchmarks.objective_variants_cost \
    --rows 256 --instants 8 --repetitions 5 --warmup 1 --threads 2 \
    --output "${MARS_OBJECTIVE_COST:?}"
```

## Pendiente

- La comprobación en `cuda:0`, que espera a que la GPU quede libre.
- La pérdida y normalización del objetivo IC en el lector episódico. Los rangos de entrada ya tienen ese recorrido.
- La entrada en la criba de [#746](https://github.com/GonxKZ/mars-titan/issues/746), con su protocolo y el contraste de IC frente al control.
- Cualquier evaluación predictiva, que sigue bloqueada.
