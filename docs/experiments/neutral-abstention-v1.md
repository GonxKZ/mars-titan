# Abstención neutral con el intervalo calibrado

`neutral_abstention_v1` es la variante A2 del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#748](https://github.com/GonxKZ/mars-titan/issues/748)). Es una etapa de salida. Actúa sobre las predicciones y los cuantiles que ya emitió un ajuste y no cambia pesos, recetas ni la identidad del predictor. Está implementada en `src/mars_titan/evaluation/neutral_abstention.py` y comprobada con pruebas técnicas sobre paneles escritos a mano o sintéticos. No se ha evaluado con datos y no hay ninguna mejora medida.

## Motivación

El resumen de la fase 0 publicado en #742 indica que, en los pilotos de octubre, el factor λ que minimiza el MAE por sesión queda entre 0 y 0,019. Las predicciones son mucho más amplias de lo que justifica su ordenación. La contracción λ·x de A1 reduce esa amplitud con un factor común. Esta variante prueba otra vía: emitir solo cuando el propio modelo, con sus cuantiles calibrados, sitúa el residuo lejos del cero, y emitir una observación neutra (cero) en el resto. Es la traducción funcional de la idea de leer el mismo dato como bueno o como malo y quedarse en neutro cuando no hay evidencia.

## Regla

Sea i una fila del mercado g con cuantiles emitidos $q_{0,1}$, $q_{0,5}$ y $q_{0,9}$. La corrección CQR de [Romano, Patterson y Candès (2019)](https://arxiv.org/abs/1905.03222) se ajusta solo con la partición de calibración (G):

$$
E_j=\max\left(q_{0,1,j}-y_j,\,y_j-q_{0,9,j}\right),\qquad
Q_g=E_{(\lceil (n_g+1)\cdot 0{,}8\rceil)}.
$$

$E_{(k)}$ es el estadístico de orden k de las $n_g$ puntuaciones del mercado. En la evaluación, sin mirar objetivos,

$$
l_i=q_{0,1,i}-Q_g,\qquad u_i=q_{0,9,i}+Q_g,\qquad
\hat y_i=\begin{cases}q_{0,5,i}&\text{si } l_i>0 \text{ o } u_i<0,\\0&\text{en otro caso.}\end{cases}
$$

La regla de orden de `conformal_quantiles` ensancha un extremo hasta la mediana si la corrección lo cruzaría, así que la mediana queda dentro de $[l_i,u_i]$ y la predicción emitida conserva el signo del intervalo. Un intervalo con un extremo exactamente en cero no excluye el cero y produce abstención. Es la misma afirmación de signo que `forecast_scores` ya cuenta como `commitments`. La cobertura nominal es configurable entre los intervalos que emite la cabeza de cinco cuantiles (0,8 y 0,95), de modo que G1 puede editarla.

## Controles con la misma cobertura

Cada control se abstiene en el mismo número de filas de cada sesión que la variante y emite la mediana sin cambios en las demás. La cobertura coincide sesión a sesión, no solo en total. Como el MAE por sesión da el mismo peso a cada sesión, un control que igualara solo la cobertura total podría concentrar sus abstenciones en otras sesiones y cambiar el resultado por esa vía.

| Control | Filas que deja en cero | Información que usa |
| --- | --- | --- |
| `random` | Al azar dentro de la sesión | Generador `SeedSequence([random_seed, semilla del modelo])` |
| `small_prediction` | Menor \|mediana\| | La mediana emitida |
| `past_volatility` | Mayor volatilidad pasada del residuo del activo | Raíz de la media de los cuadrados de sus 60 últimas etiquetas con madurez ≤ t, con un mínimo de 20 |
| `seed_disagreement` | Mayor desviación típica de la mediana entre semillas | Las medianas de todas las semillas, emitidas en el mismo instante |

Los empates siguen el orden canónico del panel. Una puntuación no definida (volatilidad sin etiquetas suficientes) queda detrás de todas las definidas y el registro cuenta esas filas. Además de estos controles, la criba compara con la mediana sin abstención y con predecir cero, que son controles comunes del estudio.

La variante se descarta si no mejora a alguno de los cuatro controles emparejados. En ese caso el intervalo calibrado no decide dónde abstenerse mejor que una regla trivial, y la mejora frente a la mediana se explicaría solo por la menor amplitud. El contraste y la familia de corrección los fija el protocolo de la criba ([#746](https://github.com/GonxKZ/mars-titan/issues/746)). Como propuesta para ese protocolo, esta etapa cambia sobre todo la amplitud, así que el contraste que puede descartarla es el MAE por sesión frente a cada control emparejado. El IC por sesión se informa también, sabiendo que los ceros añaden empates.

A diferencia de la predicción selectiva de [Geifman y El-Yaniv (2017)](https://arxiv.org/abs/1705.08500), abstenerse aquí no retira la fila de la evaluación. La fila emite cero y su error |y| sigue contando en el MAE. Por eso la cobertura no es gratis y la comparación con predecir cero sigue siendo necesaria.

## Correspondencia entre ecuaciones, módulo y pruebas

Las pruebas están en `tests/evaluation/test_neutral_abstention.py`.

| Elemento | Código | Prueba |
| --- | --- | --- |
| $Q_g$ por mercado con filas de G | `fit`, que llama a `fit_conformal_quantiles` | `test_the_stage_emits_the_median_only_when_the_calibrated_interval_excludes_zero` (Q = −4 calculado a mano) y `test_the_wider_declared_interval_uses_its_own_correction` (Q = −2) |
| $\hat y_i$ con desigualdades estrictas | `apply` | La misma prueba, con extremos que tocan el cero, y `test_all_or_no_calls_reproduce_the_median_or_zero_bit_for_bit` |
| Igualdad con `commitments` | `apply` | `test_the_calls_are_the_sign_commitments_of_the_calibrated_panel` |
| Misma cobertura por sesión | `matched_abstention` | `test_matched_controls_abstain_on_as_many_rows_of_each_session` y `test_each_control_ranks_rows_by_its_own_score` |
| Volatilidad con etiquetas maduras | `past_volatility` | `test_past_volatility_matches_a_direct_reference` y `test_past_volatility_only_reads_labels_matured_at_emission` |
| Desacuerdo entre semillas | `seed_disagreement` | `test_seed_disagreement_needs_the_same_rows_in_every_seed` |
| Control aleatorio uniforme y reproducible | `control_scores` | `test_the_random_control_is_reproducible_per_stream_and_uniform_within_sessions` |

## Causalidad, estado y paridad

- La CQR solo ve filas de G. El registro guarda la mayor madurez de esas etiquetas y `apply` rechaza un tramo con alguna emisión anterior. Una emisión en el mismo instante en que madura la última etiqueta se admite.
- `apply` no lee los objetivos de la evaluación. La prueba cambia todos los objetivos y obtiene las mismas salidas bit a bit.
- La volatilidad de una fila emitida en t solo usa etiquetas con madurez ≤ t. La prueba cambia una etiqueta todavía no madura y la puntuación no cambia. Una etiqueta que madura justo en t sí cuenta. Dos etiquetas del mismo activo con la misma madurez se rechazan, porque su orden sería arbitrario.
- Cualquier instante en 2024 se rechaza, también en el historial.
- El estado es un registro JSON congelado con la configuración, la CQR, su huella y la madurez del ajuste. `stage_sha256` lo identifica. La prueba de ida y vuelta por JSON reproduce las salidas bit a bit y un cambio en la corrección se rechaza por su huella.
- Sin etapa declarada, `checked_output_stages(None)` devuelve `None` y la salida es la del modelo. Aplicar la etapa no modifica las predicciones ni los cuantiles del panel. Si todos los intervalos excluyen el cero, la salida es la mediana bit a bit, y si ninguno lo hace, es cero.

## Desviaciones y límites

- **No es un `predictor_changes`.** Un campo nuevo en `FinancialConfig` cambiaría la huella del predictor y obligaría a repetir un ajuste idéntico solo para cambiar la salida. La etapa se registra en `evaluation/output_stages.py` (`OUTPUT_STAGES` y `checked_output_stages`). La criba combina cada trabajo (receta, `predictor_changes`, semilla y origen) con las etapas declaradas allí. Esa combinación está pendiente en #746.
- La garantía de cobertura de la CQR exige intercambiabilidad y la dependencia temporal no la asegura. El registro conserva `coverage_guaranteed=False`.
- La calibración es la CQR estática de G, no la calibración en línea de PT2 ([#454](https://github.com/GonxKZ/mars-titan/issues/454)). Así la decisión de abstenerse queda fijada en G, igual que λ en A1.
- La volatilidad pasada usa las etiquetas residuales maduras, no precios, y es la raíz de la media de los cuadrados, sin restar la media.
- Una mediana exactamente nula cuenta como abstención en cualquier salida. Con salidas continuas es un caso de medida nula, pero un control podría tener entonces un cero más que la variante en esa sesión.
- La lectura de los Parquet y la alineación de la madurez con la vista de cada ventana las hará la criba, con la misma comprobación de filas que `walk_forward_comparison`.

## Coste medido

Sin tensores ni parámetros entrenables no hay forward ni backward. `benchmarks/output_stage_cost.py` mide la etapa en CPU con datos sintéticos de semilla fija que solo fijan las formas, cinco semillas y dos hilos. El recibo está en `reports/engineering/output-stages-20261023/neutral-abstention-cost.json`. Son medianas de cinco repeticiones con otras sesiones compartiendo la máquina (carga media de 21,0 al empezar), así que son cotas superiores ruidosas.

| Caso | Filas evaluadas | Ajuste en G | Variante | Volatilidad pasada | Desacuerdo entre 5 semillas | Variante y cuatro controles | Pico de NumPy en la emisión |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 activos, 63 sesiones | 16.128 | 0,018 s | 0,013 s | 0,058 s | 0,005 s | 0,044 s | 2,6 MB |
| 256 activos, 252 sesiones | 64.512 | 0,016 s | 0,048 s | 0,121 s | 0,017 s | 0,211 s | 10,4 MB |
| 4.200 activos, 252 sesiones | 1.058.400 | 0,131 s | 1,399 s | 3,062 s | 1,008 s | 4,676 s | 170,4 MB |

La columna «Variante y cuatro controles» incluye la emisión de la variante y no incluye la volatilidad ni el desacuerdo, que se calculan una vez por ventana. La fracción de filas abstenidas de estos datos sintéticos depende de cómo se generaron y no dice nada de los pilotos.

En la población de la criba (256 activos) la etapa completa con sus cuatro controles cuesta décimas de segundo por año, frente a horas de ajuste por trabajo. No es un cuello de botella y no se ha optimizado. En el caso de 4.200 activos domina la ordenación por sesión. `matched_abstention` reutiliza la de `forecast_scores`, que con un millón de filas tardó 0,69-0,75 s frente a 1,00-1,06 s de `np.lexsort` con el mismo resultado.

No se midieron la GPU, la lectura de Parquet ni la energía.

## Mutaciones dirigidas

Se aplicaron a mano 18 mutaciones sobre la lógica crítica y se revirtieron después. Todas hacen fallar alguna prueba. Una de ellas (cambiar el signo de la puntuación de `small_prediction`) sobrevivía al principio y motivó `test_the_small_prediction_control_keeps_the_largest_medians`.

- `>` por `>=` en $l_i>0$ y `<` por `<=` en $u_i<0$.
- `side="right"` por `"left"` en la madurez de la volatilidad, y la ventana desplazada en una etiqueta.
- `>=` por `>` en el mínimo de etiquetas y en la comprobación de madurez del ajuste.
- Puntuaciones no definidas al principio en lugar de al final.
- Recuentos de abstención globales en lugar de por sesión.
- `ddof=1` en el desacuerdo entre semillas.
- Generador aleatorio sin la semilla del modelo.
- Controles con la abstención invertida y máscara de abstención devuelta al revés.
- Admitir una madurez igual al instante de predicción.
- Quitar la comprobación de que la predicción puntual es la mediana, la de madurez duplicada y la de huella del registro.
- Invertir el orden dentro de la sesión.
- Signo de la puntuación de `small_prediction`.

## Pendiente

- Aplicar la etapa en el análisis de la criba sobre los Parquet de cada trabajo (#746).
- Cualquier evaluación con datos, bloqueada mientras rija el bloqueo de aprendizaje.
