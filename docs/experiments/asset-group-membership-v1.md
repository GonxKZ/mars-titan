# Grupos de activos con la ventana pasada

`asset_group_membership_v1` es la variante H4 de la red de redes holográfica ([#782](https://github.com/GonxKZ/mars-titan/issues/782), [#783](https://github.com/GonxKZ/mars-titan/issues/783)). Después de MAC, cada flujo suma a su representación lo que aportan los grupos de activos de su sesión a los que pertenece. La pertenencia es fraccionaria y sale de la ventana de 64 sesiones que termina en la sesión de la decisión. Las medidas y las pertenencias están en `src/mars_titan/data/asset_groups.py`, el módulo en `src/mars_titan/models/titans/asset_groups.py` y la declaración, con sus controles, en [su configuración](../../configs/experiments/asset-group-membership-v1.json). Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y alcance

H1 lee cada fila con varias áreas a la vez. Su segundo nivel previsto son los grupos de activos de la sesión, y H4 aísla ese nivel para contrastarlo por separado. La pregunta es si una fila gana algo al leer lo que hacen en la misma sesión los activos que se le parecen por volatilidad, momentum, liquidez, beta o correlación de sus residuos, con grupos fijados por datos pasados y no aprendidos.

La pertenencia fraccionaria a hiperaristas viene de MaGNet ([Tan y otros, 2025](https://arxiv.org/abs/2511.00085v1)) y la combinación por ranuras de Soft MoE ([Puigcerver y otros, 2024](https://openreview.net/forum?id=jxpsAj7ltE)). En los dos trabajos las pertenencias se aprenden. En H4 salen de medidas de la ventana y no tienen parámetros, así que el modelo solo aprende qué hacer con cada grupo. C5 ([atención entre activos](cross-asset-attention-v1.md)) resuelve la misma pregunta aprendiendo con atención a qué activos leer, y por eso es un control externo de H4.

## Medidas de la ventana

Con c, h y l el logaritmo del cierre, del máximo y del mínimo respecto del ancla de la ventana, y $r_t=c_t-c_{t-1}$ cuando las dos sesiones están presentes, cada fila tiene cinco medidas.

| Familia | Medida | Fila válida |
| --- | --- | --- |
| Volatilidad | Desviación típica de r con ddof = 1 | Al menos 36 rendimientos |
| Momentum | c de la última sesión presente menos c de la primera | Al menos 36 rendimientos |
| Diferencial | $\frac1n\sum_t\sqrt{\max\{4(c_t-\eta_t)(c_t-\eta_{t+1}),0\}}$ con $\eta=(h+l)/2$, la versión CHL corregida por dos días de [Abdi y Ranaldo (2017)](https://doi.org/10.1093/rfs/hhx084) | Al menos 36 pares con las dos sesiones presentes y con rango |
| Beta | Pendiente de MCO de r frente a m, la media equiponderada de los rendimientos de las filas válidas de la sesión | Al menos 36 rendimientos y 36 pasos cubiertos, y un mercado que se aparte de su media más que el redondeo |
| Correlación | Grupos de c-medias difusas de los residuos e = (r − r̄) − β (m − m̄) | Beta válida y un residuo mayor que el redondeo |

El volumen de la ventana está dividido por su media en cada activo y no permite comparar activos entre sí, por eso la liquidez sale de los precios. Restar la media de 63 términos deja un error de redondeo que no supera unas 63 veces el épsilon $\varepsilon$ de la precisión, relativo a la escala de los términos. Por eso la beta exige $\sum(m-\bar m)^2>(63\varepsilon)^2\sum m^2$ y la correlación $\lVert e\rVert>63\varepsilon\lVert r\rVert$. Sin esa tolerancia, un mercado o una fila de rendimientos constantes daban una beta y un grupo de correlación hechos de redondeo. El mínimo de 36 aplica a los 63 pares de la ventana la proporción entre los 12 días aplicables que el artículo de Abdi y Ranaldo exige en un mes (páginas 20 y 21 de la versión de autor) y los meses de 21 sesiones de sus simulaciones (página 17).

## Pertenencias

**Cubos.** Las cuatro primeras medidas se convierten en el rango medio $u\in(0,1)$ entre las filas válidas de la sesión, con los empates en el mismo valor. Con Q cubos y $p=\operatorname{clamp}(uQ-\tfrac12,0,Q-1)$, la fila pertenece al cubo $\lfloor p\rfloor$ con $1-(p-\lfloor p\rfloor)$ y al siguiente con el resto. Es una partición difusa en el sentido de Ruspini (1969): las pertenencias de una fila válida suman uno y solo dos cubos vecinos las comparten. Una fila no válida no pertenece a ningún cubo de esa familia.

**Correlación.** Sea $z_i$ el residuo de la fila i normalizado. Como la v3.1 solo admite ausencias de todo el mercado, las filas de una sesión tienen las mismas sesiones presentes y $z_i\cdot z_j$ es la correlación de Pearson de sus residuos. Las K primeras componentes principales, los vectores propios de $Z^\top Z$ de mayor valor propio, dan la inmersión $s_i=V_K^\top z_i$, que se normaliza por filas como en el paso 4 de Ng, Jordan y Weiss (2001), $y_i=s_i/\lVert s_i\rVert$. Como $SS^\top=U_K\Sigma_K^2U_K^\top$ es la reconstrucción de rango K de la matriz de correlaciones,

$$
\lVert y_i-y_j\rVert^2=2\,(1-\hat\rho_{ij}),\qquad
\hat\rho_{ij}=\frac{(U_K\Sigma_K^2U_K^\top)_{ij}}{\sqrt{(U_K\Sigma_K^2U_K^\top)_{ii}\,(U_K\Sigma_K^2U_K^\top)_{jj}}},
$$

y dos filas quedan más lejos cuanto menos se parecen sus residuos reconstruidos. Esta igualdad es una derivación del proyecto y la comprueba una prueba. Las c-medias difusas de Bezdek (1981) con m = 2 reparten la inmersión en K grupos con un número fijo de iteraciones, desde los centros del recorrido más lejano de Gonzalez (1985) a partir de la media:

$$
u_{ik}=\frac{d_{ik}^{-2}}{\sum_l d_{il}^{-2}},\qquad
c_k=\frac{\sum_i u_{ik}^2\,y_i}{\sum_i u_{ik}^2}.
$$

Agrupar los residuos y no los rendimientos evita que el modo común del mercado decida los grupos. Agrupar la inmersión de dimensión K y no las 63 sesiones responde a que, en dimensión alta, los prototipos de las c-medias difusas acaban en el centro de gravedad de los datos, según el resumen de Winkler, Klawonn y Kruse (2011).

La pertenencia de una fila es la concatenación de las cinco familias, con $4Q+K$ columnas. Todo se calcula sin gradiente y en el dispositivo de la ventana. Los valores declarados, Q = 5, K = 8, 36 observaciones y 32 iteraciones, se fijaron antes de ejecutar y sin mirar datos.

## Ecuación ejecutada

Sea $e_i$ la representación de la fila i tras MAC, $M_{ig}$ su pertenencia al grupo g y $F_i$ el número de familias en que tiene alguna. Para las filas del bloque con el mismo mercado y el mismo instante de decisión, que son la sesión entera:

$$
n_i=\operatorname{LayerNorm}(e_i),\qquad
p_g=\frac{\sum_i M_{ig}\,n_i}{\sum_i M_{ig}},\qquad
s_g=W_2\operatorname{SiLU}(W_1p_g+b_1)+b_2,
$$

$$
e'_i=e_i+W_o\,\frac{\sum_g M_{ig}\,s_g}{F_i}.
$$

Un grupo sin miembros queda en cero y ninguna fila lo lee. Dividir por $F_i$ hace que la lectura sea la media de las familias de la fila y que su escala no dependa de cuántas tenga. Una fila sin ninguna medida válida conserva su representación exacta. $W_o$ empieza en cero, de modo que el predictor con H4 emite al construirse lo mismo que sin ella. Los pesos salen de una semilla derivada de la del predictor con SHA-256, sin consumir el generador global, así que H4 y sus dos controles parten de los mismos pesos.

## Controles

| Modelo | Cambio | Pregunta que contrasta |
| --- | --- | --- |
| `reference` | Sin H4 | Línea base |
| `shuffled_rows_control` | Las mismas pertenencias permutadas entre las filas de la sesión, con una permutación que sale de la semilla, el mercado y el instante | Si importa qué fila pertenece a qué grupo o basta con la mezcla por sesión |
| `asset_group_membership` | H4 | Si los grupos de la ventana pasada aportan algo |
| `current_sector_leak_control` | Una sola familia, el sector actual de cada activo según las fichas de FinMultiTime | Cuánto daría un reparto que puede adelantar reclasificaciones posteriores |
| C5 (externo) | Atención entre los activos de la sesión | Si fijar los grupos con datos pasados rinde como aprenderlos |

El control barajado conserva los tamaños de los grupos, la normalización y el cálculo, y rompe la relación entre cada fila y sus medidas. Los tres contrastes declarados son H4 frente al control barajado, el control barajado frente a la referencia y el control sectorial frente a H4. El sector de las fichas ([`SectorMap`](../../src/mars_titan/data/sector_map.py)) es una clasificación actual y no una serie histórica, así que el control sectorial no es un candidato. Si superara a H4, parte de su ventaja podría venir de información posterior a la decisión.

## Causalidad

La ventana de la v3.1 cubre las 64 sesiones del calendario que terminan en la sesión de la decisión y solo usa precios publicados hasta su cierre. Las pertenencias de una fila dependen de su ventana y de las ventanas de las demás filas de su sesión, todas del mismo instante. H4 no lee etiquetas, ni rendimientos posteriores, ni instantes de otras sesiones. La semilla del control barajado depende del mercado y del instante, no del resultado. El control sectorial es la única excepción declarada, porque lee las fichas actuales.

## Requisitos de ejecución

H4 necesita todas las filas de una sesión en el mismo bloque y el predictor la marca como operación por sesión (`session_level`), igual que C5. `prepare` rechaza un bloque declarado incompleto y el recorrido se detiene si `block_rows` parte un instante. También necesita la ventana de la v3.1 con el bit de presencia, y lee la ventana recibida, sin los rangos por sesión de F4.

H4 se rechaza junto a C5, porque son dos respuestas a la misma pregunta, y junto a las cabezas de horizontes auxiliares de F5, que necesitan otro contraste con operaciones por sesión. El consumidor congelado, Dust y los padres del recorrido de MARS-TITAN la rechazan hasta que la registren. El código de los dos módulos entra en la identidad del recorrido solo cuando H4 está declarada.

## Correspondencia entre ecuaciones, código y pruebas

| Ecuación o regla | Código | Prueba |
| --- | --- | --- |
| Cinco medidas y su validez | `window_measures` | `test_each_measure_matches_a_direct_calculation`, `test_rows_with_too_few_observations_are_not_valid`, `test_the_minimum_counts_exactly_its_observations`, `test_momentum_reads_the_first_and_last_present_sessions` |
| Residuo y tolerancia al redondeo | `window_measures` | `test_the_residual_removes_the_session_market_with_its_beta`, `test_a_market_without_variance_gives_no_beta_or_correlation`, `test_a_row_with_constant_returns_has_no_correlation_group` |
| Diferencial corregido por dos días | `window_measures` | `test_the_spread_sets_each_negative_two_day_estimate_to_zero_before_the_square_root` |
| Rangos medios y partición de Ruspini | `midranks`, `fuzzy_buckets` | `test_midranks_match_the_average_ranks_and_ignore_invalid_rows`, `test_fuzzy_buckets_are_a_ruspini_partition_between_neighbours`, `test_a_rank_on_a_centre_belongs_to_one_bucket_and_the_edges_saturate` |
| Inmersión y distancia de la correlación reconstruida | `correlation_embedding` | `test_the_embedding_distance_follows_the_reconstructed_residual_correlation`, `test_the_embedding_does_not_depend_on_the_sign_of_each_component`, `test_the_embedding_ignores_the_values_of_invalid_rows` |
| Recorrido más lejano y c-medias difusas | `farthest_first`, `fuzzy_clusters` | `test_farthest_first_starts_far_from_the_mean_and_skips_invalid_rows`, `test_farthest_first_follows_the_traversal_of_gonzalez`, `test_each_iteration_follows_the_fuzzy_c_means_update`, `test_fuzzy_clusters_recover_separated_sectors`, `test_fewer_valid_rows_than_clusters_put_each_row_in_its_own_group` |
| Pertenencias de la sesión | `session_memberships`, `family_slices` | `test_session_memberships_sum_to_one_in_each_family_of_a_valid_row`, `test_memberships_follow_a_permutation_of_the_rows`, `test_the_fill_of_an_absent_session_does_not_reach_the_memberships` |
| Agregación, estado de grupo y lectura | `AssetGroupMembership.forward` | `test_the_groups_match_a_row_by_row_reference`, `test_each_row_only_reads_rows_of_its_market_and_instant`, `test_a_row_with_fewer_families_reads_the_mean_of_its_own`, `test_a_row_without_history_keeps_its_representation_exactly` |
| Controles | `memberships` y `_permutation` | `test_the_controls_have_the_same_parameters_and_the_shuffle_permutes_whole_rows`, `test_the_sector_control_groups_by_the_current_sector` |
| Paridad, semilla y gradientes al empezar | $W_o=0$ y la semilla derivada | `test_h4_and_its_controls_start_as_the_reference`, `test_the_weights_come_from_a_derived_seed_without_the_global_generator`, `test_at_construction_only_the_output_projection_receives_gradient`, `test_gradients_reach_the_rows_of_the_session_and_never_the_window` |
| Sesión completa en el recorrido | `session_level` y el recorrido cronológico | `test_h4_needs_the_complete_session_in_one_block`, `test_each_update_reads_whole_sessions_with_their_presence_windows`, `test_an_instant_split_into_blocks_is_refused` |
| Identidad y rechazos | `FinancialConfig`, `_code` y los consumidores | `test_disabled_field_keeps_identity_parameters_and_predictions`, `test_the_sources_enter_the_trainer_identity_only_when_active`, `test_h4_needs_the_presence_window_and_does_not_combine_with_c5_or_f5`, `test_the_frozen_consumer_rejects_h4_until_it_registers_it`, `test_dust_refuses_h4`, `test_the_mars_titan_parents_refuse_h4` |

## Desviaciones y límites

- Abdi y Ranaldo estiman el diferencial por meses. H4 lo estima sobre la ventana de 64 sesiones y descarta los pares con una sesión ausente o sin rango, como en sus simulaciones (página 18 de la versión de autor), en lugar de arrastrar los precios del día anterior, como hace su aplicación empírica (página 20). En el proyecto no se imputa ningún precio.
- Ng, Jordan y Weiss toman los vectores propios de una afinidad gaussiana normalizada por el grado y agrupan con k-medias u otro algoritmo que minimice la distorsión. H4 los toma de la correlación de los residuos, que puede ser negativa, y agrupa con c-medias difusas.
- Las c-medias difusas usan un número fijo de iteraciones, sin criterio de convergencia, para que el coste sea constante. Las pertenencias pueden no estar en un punto fijo.
- En MaGNet y Soft MoE las pertenencias se aprenden. En H4 son fijas, se recalculan en cada decisión y no reciben gradiente.
- El beta se mide frente a la media equiponderada de la sesión del bloque, no frente a un índice de mercado.
- La adaptación es propia del proyecto y no reproduce ningún resultado publicado.

## Coste

La medida CPU del 29 de octubre (`benchmarks/asset_groups_cost.py`, commit `eb06ca91c`, [recibo](../../reports/engineering/asset-group-membership-20261029/cost-cpu.json)) llama solo al módulo con representaciones aleatorias en FP32, H = 64 y la configuración declarada. Las ventanas son sintéticas, con un factor de mercado, cuatro sectores y ruido propio, y solo fijan las formas. Se usaron dos hilos, tres calentamientos y veinte repeticiones en cada uno de los dos órdenes de los modos, sin optimizador. El equipo tenía otras baterías de pruebas en marcha, con una carga media de un minuto de 25,8 al empezar y 25,9 al terminar en 16 hilos, así que los percentiles son descriptivos. `memberships` mide solo las pertenencias, `forward` la inferencia sin gradiente y `backward` añade la propagación de una suma de cuadrados hasta los parámetros de H4.

| Filas | Sesiones | Pasada | H4 | Control barajado | Control sectorial |
| ---: | ---: | --- | ---: | ---: | ---: |
| 64 | 1 | Pertenencias | 12,05 ms (14,50) | 12,76 ms (16,68) | 0,16 ms (0,19) |
| 64 | 1 | Forward | 13,66 ms (14,55) | 13,66 ms (15,15) | 0,93 ms (1,00) |
| 64 | 1 | Backward | 14,04 ms (17,02) | 14,65 ms (19,83) | 2,35 ms (2,62) |
| 256 | 1 | Pertenencias | 17,89 ms (20,68) | 17,66 ms (21,88) | 0,43 ms (0,53) |
| 256 | 1 | Forward | 19,35 ms (21,27) | 20,14 ms (21,91) | 1,41 ms (1,57) |
| 256 | 1 | Backward | 20,18 ms (33,22) | 19,51 ms (22,85) | 3,37 ms (4,51) |
| 1.024 | 4 | Pertenencias | 70,41 ms (78,03) | 72,90 ms (93,32) | 2,01 ms (24,88) |
| 1.024 | 4 | Forward | 71,47 ms (85,15) | 77,25 ms (85,16) | 5,15 ms (6,30) |
| 1.024 | 4 | Backward | 79,09 ms (87,69) | 80,53 ms (84,80) | 10,10 ms (11,68) |

Cada celda da la mediana y, entre paréntesis, el percentil 95. Los tres modos tienen 12.544 parámetros. Las pertenencias se llevan casi todo el tiempo, y el coste crece con el número de sesiones más que con sus filas, porque cada sesión recorre sus 32 iteraciones y su descomposición propia por separado. El control barajado cuesta lo mismo que H4 y el sectorial es mucho más barato porque no mide la ventana. Con 256 activos, los 19 a 20 ms de H4 serían del orden del 6 al 7 % de los 0,31 s por instante del forward con grafo del predictor completo, medido también con dos hilos para la [fiabilidad madura](mature-reliability-signal-v1.md), aunque esa relación no se ha medido en un mismo proceso. La medida anterior, sobre `fa5da1608` y antes de la tolerancia al redondeo, dio valores del mismo orden. No se han medido la memoria, la VRAM ni la energía.

En CUDA, `torch.linalg.eigh` sincroniza el dispositivo con el anfitrión, según la documentación de PyTorch 2.14, así que H4 añadiría una sincronización por sesión y pasada. La medida CUDA dirá si compensa calcular las sesiones de un bloque en una sola llamada por lotes.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 uv run python -m benchmarks.asset_groups_cost --run \
  --output "${MARS_ASSET_GROUPS_COST:?}"
```

## Comprobación técnica

Pendiente de la verificación final.

## Pendiente

Faltan la medida CUDA, el registro en el consumidor congelado y la comparación predictiva con las mismas filas, cortes y presupuesto. Los niveles 2 y 3 de H1 pueden reutilizar estas pertenencias cuando se integren. El aprendizaje sigue bloqueado y la variante no se activa por defecto.

## Referencias

- Abdi, F. y Ranaldo, A. (2017). A simple estimation of bid-ask spreads from daily close, high, and low prices. *The Review of Financial Studies, 30*(12), 4437–4480. https://doi.org/10.1093/rfs/hhx084
- Bezdek, J. C. (1981). *Pattern recognition with fuzzy objective function algorithms*. Plenum Press. https://doi.org/10.1007/978-1-4757-0450-1
- Gonzalez, T. F. (1985). Clustering to minimize the maximum intercluster distance. *Theoretical Computer Science, 38*, 293–306. https://doi.org/10.1016/0304-3975(85)90224-5
- Ng, A. Y., Jordan, M. I. y Weiss, Y. (2001). On spectral clustering: Analysis and an algorithm. En T. Dietterich, S. Becker y Z. Ghahramani (Eds.), *Advances in Neural Information Processing Systems 14*. MIT Press. https://proceedings.neurips.cc/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html
- Puigcerver, J., Riquelme, C., Mustafa, B. y Houlsby, N. (2024). From sparse to soft mixtures of experts. En *International Conference on Learning Representations*. https://openreview.net/forum?id=jxpsAj7ltE
- Ruspini, E. H. (1969). A new approach to clustering. *Information and Control, 15*(1), 22–32. https://doi.org/10.1016/S0019-9958(69)90591-9
- Tan, P., Shi, C., Tu, D. y Xie, L. (2025). *MaGNet: A Mamba dual-hypergraph network for stock prediction via temporal-causal and global relational learning* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2511.00085
- Winkler, R., Klawonn, F. y Kruse, R. (2011). Fuzzy c-means in high dimensional spaces. *International Journal of Fuzzy System Applications, 1*(1), 1–16. https://doi.org/10.4018/ijfsa.2011010101
