# Conformal ponderado por contextos parecidos

`regime_weighted_conformal_v1` es la variante A3 del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#748](https://github.com/GonxKZ/mars-titan/issues/748)). Es una etapa de salida. Recalibra los cuantiles que ya emitió un ajuste y no cambia pesos, recetas ni la identidad del predictor. Está implementada en `src/mars_titan/calibration/regime_weighted_conformal.py` y comprobada con pruebas técnicas sobre paneles sintéticos o escritos a mano. No se ha evaluado con datos y no hay ninguna mejora medida.

## Motivación y fuente

La CQR estática de A2 usa una sola corrección por mercado, ajustada en G. PT2 ([#454](https://github.com/GonxKZ/mars-titan/issues/454)) la sigue en el tiempo, pero cada error maduro pesa lo mismo. Si el error de calibración depende del régimen del mercado o de la propia incertidumbre que declara el modelo, los errores emitidos en situaciones parecidas a la actual deberían contar más.

La idea procede de HopCPT ([Auer, Gauch, Klotz y Hochreiter, NeurIPS 2023](https://arxiv.org/abs/2303.12783v2)). Su ecuación 6 asocia la fila actual con los errores pasados mediante una memoria de Hopfield moderna, un softmax sobre productos escalares de proyecciones aprendidas con temperatura β. Su ecuación 7 forma el intervalo con cuantiles ponderados por esa asociación, con pesos que suman uno y sin masa para el punto evaluado. Esta variante conserva esa forma de lectura, pero no entrena ninguna red. Las diferencias se enumeran en «Desviaciones».

## Regla

Para una fila i del mercado g emitida en t se forma su contexto con información disponible en t:

$$
c_i=\left(\log\left(q_{0,9,i}-q_{0,1,i}\right),\ \log \operatorname{RMS}\{y_k : k\in g,\ t-30\,\mathrm{d}<\tau_k\le t\}\right),
$$

donde $\tau_k$ es la madurez de la etiqueta k y $y_k$ es el objetivo residual. El primer rasgo es la incertidumbre que declara el modelo y el segundo el régimen reciente del mercado. Cada rasgo se estandariza con la media y la desviación típica de los contextos de G, $z_i=(c_i-\mu_G)/\sigma_G$. Un rasgo con $\sigma_G$ menor que $10^{-9}$ se rechaza, porque solo contendría redondeo.

El conjunto pasado $P_i$ reúne las filas j del mismo mercado, de G o de la evaluación, con madurez $\tau_j\le t$, emitidas como mucho 120 días antes y con contexto definido. Cada una aporta su puntuación CQR para cada intervalo central declarado:

$$
E_j=\max\left(q_{\mathrm{bajo},j}-y_j,\ y_j-q_{\mathrm{alto},j}\right),\qquad
p_{ij}=\frac{\exp\left(-\lVert z_i-z_j\rVert^2/2h^2\right)}{\sum_{k\in P_i}\exp\left(-\lVert z_i-z_k\rVert^2/2h^2\right)},
$$

$$
Q_i=\inf\Big\{e:\sum_{j\in P_i}p_{ij}\,\mathbb 1\{E_j\le e\}\ge 1-a\Big\},\qquad
[\,q_{\mathrm{bajo},i}-Q_i,\ q_{\mathrm{alto},i}+Q_i\,].
$$

Con $h=0{,}5$ en la configuración por defecto. Los intervalos de 0,8 y 0,95 se corrigen del más estrecho al más ancho con la regla de orden de la CQR (`conformal_quantiles.widen_in_order`), así que ningún extremo cruza la mediana ni el intervalo interior. La mediana no cambia. Se exigen al menos 1.000 filas pasadas por mercado y sesión, y una fila evaluada sin contexto definido se rechaza en lugar de calibrarse con otra regla.

Los cuantiles calibrados pueden pasar a la abstención de A2 con `neutral_abstention.emit_calibrated`, que es la misma decisión que usa `neutral_abstention.apply` con la CQR estática.

## Controles que pueden descartarla

Los tres modos comparten conjunto pasado, puntuaciones, regla de cuantil y regla de orden. Solo cambian los pesos:

| Modo | Pesos | Qué aísla |
| --- | --- | --- |
| `context` | Softmax de la distancia entre contextos | La variante |
| `uniform` | $1/\lvert P_i\rvert$ | Si ponderar aporta algo frente al conformal sin ponderar con la misma ventana |
| `shuffled` | Los de `context` tras permutar los contextos del conjunto pasado | Si aporta la relación entre contexto y error o solo la concentración de los pesos |

La permutación de `shuffled` es propia de cada mercado y sesión (`SeedSequence([semilla, mercado, sesión])`). Como solo reasigna los mismos contextos a otras puntuaciones, cada fila conserva el mismo conjunto de pesos y el mismo tamaño efectivo $1/\sum_j p_{ij}^2$. Además de estos controles, la criba dispone de la CQR estática de G, de PT2 y de su recalibración periódica (`calibration.periodic_conformal`) con las mismas filas.

Como propuesta para el protocolo de la criba ([#746](https://github.com/GonxKZ/mars-titan/issues/746)), la variante se descarta si no mejora a `uniform` y a `shuffled` con las métricas que ya usa PT2 en `evaluation/online_calibration.py`: la desviación absoluta de la cobertura respecto al nominal por año y mercado y el interval score por sesión con la comparación emparejada común. Si se usa para abstenerse, el contraste es el MAE por sesión frente a A2 con la CQR estática. La familia de corrección y el bloque del bootstrap los fija #746.

## Correspondencia entre ecuaciones, módulo y pruebas

Las pruebas están en `tests/calibration/test_regime_weighted_conformal.py`.

| Elemento | Código | Prueba |
| --- | --- | --- |
| Régimen con etiquetas maduras en $(t-30\,\mathrm{d},t]$ | `market_regime` | `test_the_market_regime_reads_only_labels_matured_in_the_window`, con los dos bordes y un valor de $10^{300}$ |
| $\mu_G$, $\sigma_G$ y madurez del ajuste | `fit` | `test_the_scale_uses_only_calibration_contexts_and_labels_already_matured` y `test_rows_without_a_defined_context_are_rejected_in_every_weighting` |
| $p_{ij}$, $Q_i$, $P_i$ y regla de orden, fila a fila | `replay`, `_weights`, `_sorted_quantile` | `test_the_replay_matches_a_row_by_row_reference`, contra una implementación independiente con bucles de Python, dos mercados y los bordes de madurez, ventana y mínimo de filas |
| $Q_i$ como cuantil ponderado | `weighted_quantile` | `test_the_weighted_quantile_matches_a_hand_computation` y `test_uniform_weights_give_the_exact_order_statistic` ($\lceil n(1-a)\rceil$ hasta 30.001 filas) |
| Regla de orden | `conformal_quantiles.widen_in_order` | `test_an_interval_too_wide_collapses_on_the_median_by_the_order_rule` y, en la CQR, `test_widening_counts_rows_whose_lower_or_upper_end_crosses_alone` |
| Controles | `replay(weighting=...)` | `test_identical_contexts_reproduce_the_uniform_weighting_bit_for_bit` y `test_the_three_weightings_share_past_rows_and_shuffling_keeps_each_weight_set` |
| Estabilidad numérica | `_weights` | `test_far_contexts_put_the_weight_on_the_nearest_past_row` y `test_evaluating_rows_in_blocks_does_not_change_the_result` |
| Abstención de A2 | `neutral_abstention.emit_calibrated` | `test_the_calibrated_interval_feeds_the_a2_abstention` |

## Causalidad, estado y paridad

- El contexto de una fila emitida en t solo lee etiquetas con madurez ≤ t, y el conjunto pasado solo contiene filas cuya etiqueta maduró como mucho en t. Una etiqueta que madura justo en t cuenta. La prueba de causalidad cambia todos los objetivos que maduran después de un corte y las salidas emitidas hasta ese corte no cambian en ningún modo.
- La escala del contexto solo ve contextos de G. El registro guarda la mayor madurez de G y `replay` rechaza un tramo con alguna emisión anterior. Quitar del historial las etiquetas posteriores a G produce el mismo registro.
- Cualquier instante en 2024 se rechaza, también en el historial.
- El estado es un registro JSON con la configuración, los niveles, la escala y la madurez del ajuste. `stage_sha256` lo identifica y la ida y vuelta por JSON reproduce las salidas bit a bit. La semilla de `shuffled` forma parte de la configuración.
- Sin etapa declarada, `checked_output_stages(None)` devuelve `None` y la salida es la del modelo. Con contextos idénticos, `context` reproduce `uniform` bit a bit.

## Desviaciones y límites

- **Sin proyecciones aprendidas.** HopCPT entrena las proyecciones y la temperatura. Aquí no hay entrenamiento: los rasgos se estandarizan en G y la anchura de banda h es fija. Con proyecciones identidad, $-\lVert z_i-z_j\rVert^2/2h^2$ difiere de un producto escalar con $\beta=1/h^2$ solo en el término $-\lVert z_j\rVert^2/2h^2$, que favorece las claves cercanas a la media de G.
- **Puntuación de la CQR.** HopCPT pondera residuos con signo de un predictor puntual y toma dos cuantiles. Esta variante pondera la puntuación simétrica de la CQR sobre los cuantiles del modelo, para conservar la regla de orden y la conexión con A2.
- **Sin garantía de cobertura.** Como en la ecuación 7 de HopCPT, no se reserva masa para el punto evaluado ni se usa la corrección $n+1$. Con dependencia temporal y pesos que dependen de los datos no se afirma ninguna garantía en muestra finita.
- **Rasgos propios.** Los dos rasgos del contexto son una elección del proyecto. Son una hipótesis que los controles pueden descartar, no una reproducción del artículo.
- **Escala fija en G.** La media y la desviación del contexto no se reajustan durante la evaluación, igual que λ en A1.
- **Mismo mercado.** El conjunto pasado no mezcla mercados, como la CQR por mercado de A2.
- La lectura de los Parquet y la alineación de la madurez con la vista de cada ventana las hará la criba, como en A2.

## Coste medido

Sin tensores ni parámetros entrenables no hay forward ni backward. `benchmarks/regime_weighted_conformal_cost.py` mide la etapa en CPU con datos sintéticos de semilla fija que solo fijan las formas: 30 sesiones previas para el régimen, 63 de calibración y el tramo evaluado. El recibo está en `reports/engineering/output-stages-20261023/regime-weighted-conformal-cost.json`. Se tomó antes de rebasar la rama sobre `develop`, así que su campo `commit` no aparece en la historia publicada. Las huellas SHA-256 de las tres fuentes que guarda coinciden con las de esta versión. Son medianas de tres repeticiones (una en el caso de 4.200 activos) con otras sesiones compartiendo la máquina, con una carga media de 8,5 al empezar y 4,6 al terminar. La carga concurrente limita la comparación y estas medidas no son cotas superiores.

| Caso | Filas evaluadas | Filas pasadas por sesión | Ajuste en G | `context` | `uniform` | `shuffled` | Pico de NumPy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 activos, 63 sesiones | 16.128 | 16.128-30.720 | 0,013 s | 30,3 s | 0,80 s | 31,6 s | 22,2 MB |
| 256 activos, 126 sesiones | 32.256 | 16.128-30.720 | 0,007 s | 100,0 s | 1,94 s | 80,1 s | 24,4 MB |
| 4.200 activos, 1 sesión | 4.200 | 264.600 | 0,154 s | 42,5 s | 0,33 s | 44,3 s | 50,1 MB |

El pico es el de `context`, el mayor de los tres modos. `context` y `shuffled` hacen el mismo trabajo, así que la diferencia entre ellos en el segundo caso (y su mínimo de 50,1 s frente a 97,5 s) es variación por la carga. Las sesiones sintéticas son días consecutivos y la ventana de 120 días reúne hasta 30.720 filas pasadas, más que las unas 21.000 del calendario real.

El coste crece con el producto de filas evaluadas y filas pasadas. Con 256 activos, una sesión cuesta entre 0,4 y 0,8 s. Con 4.200 activos, una sola sesión ya cuesta 42 s. Extrapolar a un año de esa población daría varias horas por modo, pero no se ha medido. La criba usa 256 activos. No se midieron la GPU, la lectura de Parquet ni la energía.

## Optimización medida

La primera versión calculaba los pesos con forma [filas evaluadas, filas pasadas] y la suma acumulada recorría el último eje. En una sesión con 256 filas evaluadas y 24.000 pasadas, esa suma tardaba 545 ms y la distancia con difusión de tres ejes 388 ms. Con las filas pasadas en el primer eje, la suma acumulada suma filas contiguas, el reordenamiento por puntuación es una selección de filas y el primer índice que alcanza el nivel se obtiene contando las posiciones que quedan por debajo, porque la suma de pesos no negativos no decrece. Las distancias coinciden bit a bit y los cuantiles seleccionados son los mismos.

El tamaño del bloque también importa. Mínimo de cinco repeticiones de una sesión en la misma ejecución, con carga media de 8,8:

| Filas evaluadas por bloque | Elementos | Disposición anterior | Disposición actual |
| ---: | ---: | ---: | ---: |
| 256 | 6,1 M | 3.234 ms | 1.679 ms |
| 64 | 1,5 M | 809 ms | 253 ms |
| 16 | 0,38 M | 635 ms | 383 ms |
| 4 | 0,10 M | 646 ms | 820 ms |

La versión anterior usaba bloques de 4,2 M de elementos. Una comparación intercalada de nueve repeticiones con 16.128, 24.000 y 30.720 filas pasadas situó el mejor bloque entre 0,5 M y 1 M de elementos, y se fijó `MAX_BLOCK = 524.288` (4 MiB por temporal en float64). La reproducción completa de un trimestre tardó 164 s con la versión anterior (carga media de 29) y entre 23,6 y 31,6 s con la actual en el recibo. Las cargas distintas impiden atribuir toda la diferencia al cambio, así que la comparación de una sesión es la evidencia más limpia.

Se midieron y descartaron dos alternativas. Reutilizar búferes para la selección de filas y la suma acumulada mejoró una sesión un 15 % (mediana de 508 a 431 ms), menos que la dispersión entre repeticiones con la carga presente. Repartir los bloques entre hilos competiría con el paralelismo por trabajos de la criba, que ya reparte semillas, orígenes y modos entre procesos.

## Mutaciones dirigidas

Se aplicaron a mano 45 mutaciones y se revirtieron después: 37 en el módulo, 4 en `widen_in_order` y 4 en `emit_calibrated`. Todas hacen fallar alguna prueba salvo dos equivalentes: sumar uno al tamaño de bloque, que solo cambia la memoria, y cambiar `<` por `<=` frente a $1-a-10^{-9}$, un valor que ninguna suma acumulada alcanza exactamente en las pruebas. Cuatro mutaciones sobrevivían al principio y motivaron pruebas nuevas:

- Calcular la anchura del contexto con el intervalo de 0,95. Los paneles sintéticos tenían colas proporcionales y la estandarización anulaba la diferencia. El caso de referencia usa ahora colas con forma propia en cada fila.
- Quitar el rechazo de filas evaluadas sin contexto definido, que con `uniform` no fallaba por otra vía.
- Quitar el mínimo de dos contextos definidos en G, que fallaba después con un mensaje engañoso.
- Contar solo los extremos inferiores ensanchados en la regla de orden de la CQR. Ninguna prueba previa ensanchaba solo el superior.

Una prueba escrita antes de las mutaciones encontró además un fallo real. Con anchura constante en G, $(m+s)-(m-s)$ no da exactamente $2s$ y la desviación del rasgo era un redondeo positivo que pasaba la comprobación `> 0`. La variante habría estandarizado ese redondeo. Desde entonces se exige $\sigma_G\ge 10^{-9}$.

## Pendiente

- Aplicar la etapa en el análisis de la criba sobre los Parquet de cada trabajo (#746), con los tres modos y las calibraciones de PT2 como referencias.
- Cualquier evaluación con datos, bloqueada mientras rija el bloqueo de aprendizaje.
