# Regla kalman de B6 con ruido de cohorte correlacionado (PT3)

Estado a 18 de octubre de 2026: implementado, conectado a las ventanas walk-forward y a la campaña A v2, con pruebas del componente y de la ventana completa sobre fixtures aislados. La comparación de [#455](https://github.com/GonxKZ/mars-titan/issues/455) sigue sin resultados financieros. El [ensayo combinado de memoria](../../reports/research/combined-memory-20261018/README.md) no ejecuta la corrección Kalman ni sus controles de correlación y retraso. La motivación, el control y la regla de decisión están en las [propuestas posteriores a Titans](../research/post-titans-proposals.md#pt3-corrección-madura-bayesiana-con-ruido-de-cohorte-correlacionado).

## Qué hace

B6 es una matriz A de clave por valor que corrige la predicción de Titans-MAC con resultados maduros ([memoria asociativa madura](mature-associative-memory.md)). Sus reglas delta y proximal tratan cada etiqueta de una cohorte como evidencia independiente y usan un paso fijo η. La regla `kalman` trata A como el estado de un modelo lineal dinámico. Guarda además una covarianza P compartida por las columnas de valor, de modo que la ganancia de cada dirección de clave depende de la incertidumbre acumulada en esa dirección. Antes de cada cohorte A deriva con un paseo aleatorio, y los resultados de una misma cohorte comparten un componente común con correlación ϱ, así que una sesión con miles de activos no cuenta como miles de observaciones independientes.

La regla vive en `AssociativeMemory` (`src/mars_titan/memory/associative_memory.py`) y se declara con `AssociativeMemoryConfig("kalman", kalman=KalmanNoise(...))`. `MatureCorrection` y `FinancialSession` la usan sin cambios de contrato, y la variante MARS-TITAN la acepta como valor de `associative_memory` con la ablación A12 de la [declaración de ampliaciones](../../configs/titans/mars-titan-extensions.json). No tiene parámetros entrenables, no usa gradientes y trabaja en CPU y FP64 como las otras dos reglas.

## Flujo de datos y de estado

```mermaid
flowchart TD
    KEYS["Claves k_i del codec<br/>normalizadas en L2, FP64"]
    VALS["Resultados maduros del evento<br/>y_i = etiqueta − núcleo, con T_i ≤ corte"]
    ST0[("Estado B6 de la generación g − 1<br/>A, P, cursor")]
    RULE{"rule"}
    OLD["Reglas delta y proximal<br/>sin cambios"]
    RW["Paseo aleatorio<br/>P⁻ = P + qI = CCᵀ"]:::pt3
    RES["Residuo con el estado previo<br/>R = Y − KA"]:::pt3
    HUB{"huber_threshold"}
    WTS["Pesos de Huber de un paso<br/>w_i = min(1, c/z_i)"]:::pt3
    INFO["Información de la cohorte<br/>M = KᵀΣ_w⁻¹K, b = KᵀΣ_w⁻¹R<br/>g = ϱ/(1 − ϱ + nϱ)"]:::pt3
    UPD["Actualización sin inversas<br/>P⁺ = C(I + CᵀMC)⁻¹Cᵀ<br/>A⁺ = A + P⁺b"]:::pt3
    ST1[("Estado B6 de la generación g<br/>A⁺, P⁺, cursor")]
    READ["Lectura del evento siguiente<br/>kᵀA⁺ y varianza kᵀP⁺k + σ²"]

    ST0 --> RULE
    RULE -- delta o proximal --> OLD --> ST1
    RULE -- kalman --> RW
    KEYS --> RES
    VALS --> RES
    RW --> RES --> HUB
    HUB -- None --> INFO
    HUB -- c --> WTS --> INFO
    RW --> UPD
    INFO --> UPD --> ST1
    ST1 -. emisión siguiente .-> READ

    subgraph LEY["Leyenda"]
        L1["Pieza de PT3, solo con rule = kalman"]:::pt3
        L2["B6 existente sin cambios"]
    end

    classDef pt3 fill:#dcefdc,stroke:#2e7d32,color:#102a12
```

**Entradas permitidas.** Las mismas que B6: las claves estables del codec de cada decisión y los resultados ya maduros de un evento, con `decision_at < available_at ≤ cutoff`. El valor escrito es la etiqueta menos la predicción del núcleo, nunca la predicción emitida. La regla no recibe pesos declarados. Los de Huber se calculan dentro de la escritura con el estado previo.

**Orden temporal.** La emisión de un evento lee A de la generación anterior. Al resolver los resultados maduros del evento, la escritura aplica primero el paseo aleatorio y después la información de toda la cohorte de una vez. El cursor canónico (disponibilidad, decisión e identificador) rechaza entregas repetidas o fuera de orden igual que en las otras reglas.

**Estado que cambia.** A y P, el contador de resultados y el cursor, una vez por escritura con resultados. Una cohorte vacía devuelve la misma memoria y no aplica el paseo aleatorio. `export` añade P y su huella a los campos de siempre, y `restore` comprueba forma, simetría exacta, definición positiva y huella.

**Qué apaga el interruptor.** Las reglas delta y proximal recorren exactamente el código anterior y conservan su identidad, su exportación y su salida. Sin `associative_memory` en la variante no hay B6. Las dos cosas están comprobadas con huellas capturadas antes del cambio.

## Ecuaciones, fuentes, módulos y pruebas

La notación sigue la de la [memoria asociativa madura](mature-associative-memory.md) y, para A como memoria lineal, la del [núcleo de Titans](titans-memory-core.md): una cohorte de n resultados tiene claves K ∈ ℝ^{n×d} con filas k_iᵀ, valores Y ∈ ℝ^{n×m} y estado A ∈ ℝ^{d×m}. P ∈ ℝ^{d×d} es la covarianza de cada columna de A.

**(K1) Modelo lineal dinámico.** Filtro de Kalman (Kalman, 1960) y modelo lineal dinámico (West y Harrison, 1997), con un efecto temporal aleatorio común a la cohorte como en los datos de panel (Baltagi, 2021). La combinación para cohortes maduras de B6 es adaptación propia:

$$
A_g=A_{g-1}+\Omega_g,\quad \operatorname{vec}\Omega_g\sim\mathcal N(0,\,I_m\otimes qI_d),\qquad Y_g=K_gA_g+E_g,\quad E_g^{(j)}\sim\mathcal N(0,\Sigma),\qquad \Sigma=\sigma^2\big[(1-\varrho)I+\varrho\mathbf 1\mathbf 1^\top\big],
$$

con A_0 de columnas independientes N(0, p0 I). Módulos `KalmanNoise` y `AssociativeMemory`. Pruebas `test_kalman_is_a_new_identity_without_rate_or_forgetting` y `test_impossible_noise_is_rejected`.

**(K2) Inversa cerrada del ruido de cohorte.** Álgebra estándar (Sherman y Morrison) que hace innecesario formar Σ:

$$
\Sigma^{-1}=\frac{1}{\sigma^2(1-\varrho)}\big(I-g\,\mathbf 1\mathbf 1^\top\big),\qquad g=\frac{\varrho}{1-\varrho+n\varrho}.
$$

Prueba `test_closed_form_inverse_of_the_equicorrelated_noise`.

**(K3) Actualización en forma de información.** Forma estándar del filtro. La factorización sin inversas explícitas es una elección de implementación propia:

$$
P^-=P+qI=CC^\top,\quad M=K^\top\Sigma^{-1}K,\quad b=K^\top\Sigma^{-1}(Y-KA),\qquad P^+=C\,(I+C^\top MC)^{-1}C^\top,\quad A^+=A+P^+b .
$$

Con I + CᵀMC = LLᵀ y T = L⁻¹Cᵀ, P⁺ = TᵀT y A⁺ = A + Tᵀ(Tb). TᵀT se simetriza con su traspuesta, porque con d = 64 el producto de BLAS no sale simétrico bit a bit y la restauración exige simetría exacta. M y b cuestan O(nd²) a partir de KᵀK y Kᵀ1, y la factorización O(d³). Módulo `AssociativeMemory._kalman`. Prueba `test_information_update_matches_the_dense_covariance_form`, que compara con la forma de covarianza y ganancia de Kalman, G = P⁻Kᵀ(KP⁻Kᵀ + Σ)⁻¹, con Σ densa y la covarianza de Joseph, en FP64.

**(K4) Recorte de Huber ponderado.** Los pesos ψ(z)/z = min(1, c/|z|) proceden de Huber (1964). Aplicarlos en un solo paso con el residuo estandarizado por la varianza predictiva previa, y la forma cerrada ponderada, son derivación propia:

$$
z_i=\frac{\lVert y_i-A^\top k_i\rVert}{\sqrt{k_i^\top P^-k_i+\sigma^2}},\qquad \Sigma_w^{-1}=W^{1/2}\Sigma^{-1}W^{1/2}=\frac{1}{\sigma^2(1-\varrho)}\Big(W-g\,\sqrt w\sqrt w^{\top}\Big).
$$

Equivale a inflar la varianza de las filas atípicas, Σ_w = W^{−1/2}ΣW^{−1/2}, y conserva el coste de (K3). Pruebas `test_information_update_matches_the_dense_covariance_form` con una fila atípica y `test_a_large_huber_threshold_is_the_unweighted_rule_bit_for_bit`.

**(K5) Casos de referencia.** Con un resultado por cohorte, Σ⁻¹ se reduce a 1/σ² y la regla es el filtro de Kalman escalar para cualquier ϱ. Con q = 0 y ϱ = 0, dos cohortes seguidas dan la posterior de una regresión ridge con todas las filas, de precisión I/p0 + KᵀK/σ², es decir RLS por bloques. Pruebas `test_one_outcome_per_cohort_is_the_scalar_kalman_filter` y `test_zero_correlation_and_process_noise_is_block_rls_with_its_prior`.

**(K6) Tamaño efectivo de una cohorte.** Derivación propia a partir de (K2), igual al efecto de diseño de un muestreo por conglomerados. Con n claves iguales a k,

$$
M=\frac{n}{\sigma^2\,(1+(n-1)\varrho)}\,kk^\top\ \le\ \frac{1}{\sigma^2\varrho}\,kk^\top ,
$$

así que una sesión aporta como mucho 1/ϱ observaciones de su componente común. Prueba `test_cohort_correlation_limits_the_information_of_identical_keys` con n = 1, 4, 64 y 4.096.

**(K7) Crecimiento de la covarianza.** Derivación propia. Como P⁺ ⪯ P⁻ = P + qI en el orden de Loewner, tras t cohortes λ_max(P_t) ≤ p0 + qt. La cota se alcanza en una dirección que ninguna clave excita. La propuesta pedía acotar ese crecimiento o declararlo, y aquí se declara: con q = 10⁻⁵ σ² y unas 5.000 cohortes de 2000 a 2019, la varianza de una dirección no excitada sube 0,05 σ², la mitad de p0 = 0,1 σ². Prueba `test_covariance_grows_at_most_linearly_in_unexcited_directions`.

**(K8) Lectura y varianza predictiva.** Estándar: la corrección es kᵀA y su varianza para un resultado nuevo kᵀPk + σ². No suma el q del paseo que precederá a la siguiente cohorte. Módulos `AssociativeMemory.read` y `AssociativeMemory.variance`. Prueba `test_write_order_and_read_purity`.

## Por qué podría ayudar y por qué podría no hacerlo

En el mercado las etiquetas maduran por cohortes de miles de activos a la vez y los residuos de una sesión siguen correlacionados aunque el objetivo reste la beta. Con la regla proximal una sesión de 4.000 activos pesa como 4.000 observaciones independientes y puede arrastrar A hacia el shock común de ese día. Con ϱ = 0,05 la misma sesión cuenta como unas 20 observaciones de su componente común. La covarianza además da una ganancia mayor en direcciones de clave poco vistas y menor en las ya conocidas, que es lo que una regla de paso fijo no puede hacer, y la varianza predictiva puede servir para calibrar la corrección.

Puede no ayudar porque la señal es muy débil. Si el residuo no tiene estructura lineal estable en las claves del codec, cualquier corrección persigue ruido y lo mejor es no corregir. RLS con un η bien elegido puede ser casi equivalente a la regla proximal. Un q grande persigue ruido y un q pequeño no sigue los cambios. Las varianzas σ², ϱ y q se estiman con la ventana de entrenamiento y pueden no valer en la de evaluación.

## Control con el que se descarta

La [declaración del modelo](../../configs/evaluation/kalman-associative-comparison.json) fija, antes de ejecutar nada, cuatro modelos con las mismas predicciones, claves, cohortes y madurez: Titans-MAC `mac_online` sin corrección, B6 proximal como control directo, B6 kalman y la alternativa trivial con ϱ = 0. σ² y ϱ se estiman por momentos con los residuos de los últimos meses del ajuste de cada ventana, p0 = 0,1 σ² y q = κ σ² con κ elegido entre 10⁻⁶ y 10⁻⁵ en cada ventana con su validación, como η en B6 proximal. La regla conserva PT3 si el límite superior del 95 % de la diferencia relativa de MAE por sesión frente a B6 proximal es negativo, o si no supera +0,1 % y la cobertura del intervalo ±1,645 desviaciones predictivas de la corrección queda entre el 85 % y el 95 %, y además conserva al menos la mitad de la ganancia con una sesión más de retraso (D3). Se abandona si empeora más de un 0,1 % o si la ganancia desaparece con el retraso. Si ϱ = 0 iguala a PT3, la correlación de cohorte no aporta. Nada se ha ejecutado.

## Ventana walk-forward y campaña A v2

La regla kalman entra en la [ventana B6](mars-titan-extensions.md#corrección-b6-en-las-ventanas-walk-forward) con su propia receta, [`kalman-correction-historical-masked.json`](../../configs/titans/kalman-correction-historical-masked.json). Sus dos casos fijan κ = q/σ² en 10⁻⁶ y 10⁻⁵, y la receta fija p0/σ² = 0,1, el tope ϱ ≤ 0,95 y tres meses de estimación. La campaña elige κ en cada ventana con el MAE por sesión de validación, igual que η en la regla proximal, así que los dos modelos comparados disponen de la misma búsqueda.

```mermaid
flowchart LR
    PAR[("Padre Titans-MAC mac_online<br/>elegido en la ventana")]
    EST["Tramo de estimación<br/>tres últimos meses del ajuste<br/>con su calentamiento"]:::pt3
    RES["Residuos etiqueta − núcleo<br/>por mercado e instante de decisión"]:::pt3
    ANO["ANOVA de un factor con estratos<br/>σ² y ϱ (memory/cohort_noise.py)"]:::pt3
    VAR["q = κσ², p0 = 0,1σ²<br/>ϱ estimada o ϱ = 0"]:::pt3
    WIN["Validación, calibración y evaluación<br/>CorrectionInference con la regla kalman"]
    ID[("run.json<br/>identidad con cohort_noise")]
    CAR["Traslado, ablación y padre congelado<br/>con la estimación del ancla"]

    PAR --> EST --> RES --> ANO --> VAR --> WIN --> ID --> CAR
    PAR --> WIN

    subgraph LEY["Leyenda"]
        L1["Pieza de PT3 en la ventana"]:::pt3
        L2["Ventana B6 existente"]
    end

    classDef pt3 fill:#dcefdc,stroke:#2e7d32,color:#102a12
```

**Estimación de σ² y ϱ.** Antes de validar, `window_cohort_noise` recorre con el padre congelado y sin corrección los últimos meses del tramo de ajuste, desde el primer día del mes situado tres meses antes de su final. Observa antes el calentamiento de la receta sin etiquetas y no usa ninguna etiqueta que madure después del final del ajuste, que precede a la validación. Cada etiqueta madura da un residuo etiqueta menos núcleo con la predicción emitida antes de conocerla. `memory/cohort_noise.py` estima con esos residuos los componentes de la varianza del ANOVA de un factor, con la sesión (mercado e instante de decisión) como efecto aleatorio y el mercado como estrato fijo. Con g sesiones en M mercados, N residuos, n_i en la sesión i y N_m en el mercado m:

$$
\mathrm{MSB}=\frac{\sum_i n_i(\bar y_i-\bar y_{m(i)})^2}{g-M},\qquad \mathrm{MSW}=\frac{\sum_i\sum_j (y_{ij}-\bar y_i)^2}{N-g},\qquad n_0=\frac{N-\sum_m\sum_{i\in m}n_i^2/N_m}{g-M}.
$$

E[MSW] = σ_w² y E[MSB] = σ_w² + n₀σ_b², de donde σ_b² = max(0, (MSB − MSW)/n₀), σ² = σ_b² + MSW y ϱ = min(0,95, σ_b²/σ²). El estrato evita que un sesgo medio distinto en EE. UU. y en China parezca correlación dentro de la sesión. El estimador procede del modelo de un factor con efectos aleatorios (Baltagi, 2021). La esperanza de MSB con estratos y sesiones de tamaño distinto es una derivación propia del caso de un factor, comprobada contra un cálculo escrito con bucles.

**Varianzas de la regla.** `arm_components` fija con la estimación q = κσ², p0 = 0,1σ² y la ϱ estimada, o ϱ = 0 en la alternativa trivial. Como p0 y q escalan con σ² y la regla no usa Huber, P escala con σ² y la ganancia P⁺b no cambia: la media de la corrección solo depende de ϱ y κ, y σ² solo fija la varianza predictiva. Una prueba lo comprueba en la vista técnica con σ² = 1 y σ² = 7.

**Identidad y reanudación.** La estimación entra en la identidad de la ventana como `cohort_noise`, con su tramo, la huella de su índice y sus recuentos, y la variante elegida guarda las varianzas resultantes. Una ventana reanudada reutiliza la estimación de su identidad si coinciden tramo e índice, porque repetirla en GPU podría no dar los mismos bits. La estimación no atiende una parada: recorre un tramo acotado y la parada pedida se atiende al empezar la validación, ya con la estimación guardada. Las reglas delta y proximal conservan exactamente su identidad y su recorrido anteriores.

**Varianza predictiva y retraso de una sesión (D3).** Con la regla kalman, cada predicción de la ventana lleva dos columnas más. `predictive_variance` es kᵀPk + σ² con el estado confirmado antes del evento, la misma lectura que la media. `prediction_label_delay1` es la predicción de una segunda memoria con la misma configuración que recibe cada etiqueta una sesión más tarde: la retiene hasta la siguiente decisión de su mercado y la escribe al final de ese evento, después de emitir. Las dos memorias se recorren en la misma pasada, así que D3 no repite la inferencia del padre en GPU. Una prueba con eventos construidos a mano exige que esa predicción coincida bit a bit con la de una memoria normal que recibe cada etiqueta en la siguiente decisión. La sección `kalman_diagnostics` de la comparación conjunta, declarada el 14 de octubre de 2026 antes de ejecutar nada, lee las dos columnas sobre las mismas filas que puntúa la comparación. Da la cobertura de ±1,645 desviaciones predictivas en filas y la fracción de la ganancia sobre `titans_mac_online` que se conserva con el retraso, las dos en los años de 2014 a 2023 y con un intervalo por bloques circulares de días. Aplica la regla declarada: cobertura entre el 85 % y el 95 % y al menos la mitad de la ganancia.

**Traslado, ablación y adaptadores.** `carry_correction` aplica la estimación del ancla junto con su κ y no vuelve a estimar. La ablación de modalidades y el padre congelado de la cadena trivial usan ese traslado.

**Campaña A v2.** `mars_titan_b6_kalman` y `mars_titan_b6_kalman_rho0` usan la clave del codec sin enrutar, las semillas 42, 43 y 44 y la receta `kalman_correction_recipe` de la sección de MARS-TITAN. La comparación conjunta añade `kalman_correction`, con los dos modelos frente a `mars_titan_b6`, que es la regla proximal con la misma clave, y `kalman_cohort_correlation`, con ϱ estimada frente a ϱ = 0. En US+CN suman 152 trabajos de ajuste sin pasos de optimizador, 108 padres congelados y 114 selecciones de la cadena, 342 predicciones de ablación y 144 ajustes y 222 referencias de políticas, que pasan de 31 a 33 predictores. La proyección de horas suma a cada ajuste kalman las filas de calibración como medida de su tramo de estimación, porque los recuentos no separan el final del ajuste y la calibración dura los mismos tres meses con el mismo calentamiento. La proyección comprueba que sigan coincidiendo.

## Comprobaciones

| Propiedad | Prueba |
| --- | --- |
| Delta y proximal sin cambios: salida, estado exportado, identidad y variante | `test_delta_and_proximal_keep_their_outputs_states_and_identities`, con huellas capturadas en `42e7dbca` antes de añadir la regla |
| Identidad nueva sin η ni λ, y declaraciones imposibles rechazadas | `test_kalman_is_a_new_identity_without_rate_or_forgetting`, `test_invalid_rule_declarations_are_rejected` y `test_impossible_noise_is_rejected` |
| Inversa cerrada de Σ | `test_closed_form_inverse_of_the_equicorrelated_noise` |
| Forma de información igual a la forma densa de covarianza, con y sin Huber | `test_information_update_matches_the_dense_covariance_form` |
| Huber con umbral alto idéntico bit a bit a la regla sin recorte | `test_a_large_huber_threshold_is_the_unweighted_rule_bit_for_bit` |
| Filtro escalar y RLS por bloques | `test_one_outcome_per_cohort_is_the_scalar_kalman_filter` y `test_zero_correlation_and_process_noise_is_block_rls_with_its_prior` |
| Tamaño efectivo de la cohorte | `test_cohort_correlation_limits_the_information_of_identical_keys` |
| Covarianza simétrica, definida positiva y con el crecimiento declarado | `test_covariance_grows_at_most_linearly_in_unexcited_directions` |
| Orden de carga y lectura sin efectos | `test_write_order_and_read_purity` |
| Causalidad y madurez: inmaduras, repetidas o con pesos rechazadas sin cambiar el estado | `test_immature_repeated_or_weighted_cohorts_are_rejected_without_changing_the_state` y `test_labels_of_a_later_cohort_do_not_change_earlier_reads` |
| Recuperación exacta de A, P y cursor, y rechazo de estados alterados | `test_recovery_restores_matrix_covariance_and_cursor_exactly` |
| Corrección madura y variante con d = 64 y simetría exacta tras BLAS | `test_mature_correction_and_variant_accept_the_kalman_rule` |
| Modelo declarado sin ejecutar | `test_comparison_arm_is_declared_with_its_control_and_rule_but_not_executed` |
| Estimador de σ² y ϱ igual al ANOVA con bucles, con sesiones de tamaño distinto y dos mercados | `test_unbalanced_estimate_matches_the_loop_anova` en `tests/memory/test_cohort_noise.py` |
| Recuento de cohortes que mezclan sesiones o mercados y de sesiones repartidas, y rechazo de etiquetas desalineadas | `test_cohort_composition_counts_cohorts_that_are_not_one_session` y `test_cohort_composition_rejects_misaligned_labels` en `tests/memory/test_cohort_noise.py` |
| σ² y ϱ simulados recuperados, invariancia a un desplazamiento por mercado y a la escala, truncado a [0; tope] y rechazos | `test_balanced_sessions_use_their_size_and_recover_the_simulated_noise`, `test_a_market_offset_is_not_cohort_correlation`, `test_scale_changes_the_variance_and_keeps_the_correlation`, `test_negative_correlation_is_zero_and_a_large_one_is_capped` y `test_degenerate_panels_are_rejected` |
| Receta kalman: κ en los casos, valores fijos en la base y rechazos | `test_declared_recipe_searches_two_process_ratios_and_fixes_the_rest` y `test_kalman_recipe_rejects_impossible_or_misplaced_values` en `tests/training/test_kalman_correction.py` |
| q = κσ², p0 = 0,1σ² y ϱ estimada o nula, sin mezclar recetas ni reglas, y delta y proximal sin cambios | `test_kalman_arm_scales_prior_and_process_noise_with_the_estimated_variance`, `test_kalman_and_proximal_arms_only_take_their_own_values` y `test_proximal_arm_keeps_its_previous_combination` |
| Tramo de estimación al final del ajuste, con su calentamiento dentro de la ventana | `test_estimation_tramo_closes_the_fit_and_keeps_its_warmup_inside_the_window` |
| Residuos etiqueta menos núcleo por mercado e instante, solo del tramo, y rechazo de un tramo medido | `test_window_noise_uses_label_minus_core_residuals_of_each_market_session`, `test_residual_rows_keep_the_market_of_each_flow`, con EE. UU. y China en las mismas sesiones, y `test_window_noise_refuses_a_measured_tramo` |
| Identidad de la estimación y reutilización en la reanudación solo con el mismo tramo e índice | `test_window_records_its_estimation_tramo_and_a_resumed_window_reuses_it` |
| Media de la corrección invariante a σ², P proporcional a σ² y primeras decisiones sin corregir | `test_the_mean_correction_does_not_depend_on_the_estimated_variance` |
| ϱ cambia la corrección y la alternativa trivial equivale a ϱ = 0 | `test_cohort_correlation_changes_the_correction_and_zero_is_its_alternative` |
| Plan, contrastes, rechazos y proyección de horas en A v2 | `test_v2_declares_the_kalman_correction_with_its_own_recipe_and_contrasts`, `test_v2_rejects_kalman_models_without_their_recipe_or_correlation`, `test_the_projection_measures_the_kalman_estimation_with_the_calibration` y `test_uniform_rates_project_the_b6_correction_as_prediction_only` en `tests/training/test_campaign_a_joint.py`, y `test_correction_hours_predict_each_measured_tramo_once` y la medida de caudal con la regla kalman en `tests/training/test_campaign_throughput.py` |
| Ventana completa: una sola estimación, reanudación, alternativa ϱ = 0 y traslado bit a bit | `test_kalman_window_estimates_its_noise_once_and_its_carry_reuses_it` en `tests/training/test_mars_titan_correction.py`. Superada en CPU con fixtures aislados, según el [recibo del 18 de octubre](../../reports/engineering/omitted-tests-20261018/verification.json), que conserva el identificador, el resultado y la huella del registro |
| Sesión financiera con referencia, causalidad y corte antes de publicar | `test_kalman_rule_follows_the_same_session_contract_and_recovers_its_covariance` en `tests/memory/test_financial_session_associative.py`, que necesita `MARS_TITAN_EPISODIC_NATIVE` |

Las pruebas están en `tests/memory/test_kalman_associative_memory.py`. El aislamiento entre activos y sesiones lo da el contrato de B6: A es común a la variante por diseño y cada escritura devuelve una memoria nueva sin tocar la anterior. La prueba de la sesión financiera necesita el enlace nativo. El 13 de octubre de 2026 pasó, con las otras diez de su archivo, usando el enlace compilado de las mismas fuentes nativas.

La mutación dirigida aplicó 19 defectos de uno en uno en una copia aislada, tras comprobar que las pruebas importaban la copia: anular g, quitar n de g, quitar o escalar el paseo aleatorio, olvidar los pesos de Huber en la evidencia, tomar su raíz, estandarizar sin σ² o con P posterior, quitar 1 − ϱ de la escala, quitar el término de correlación de la información, no simetrizar P, invertir el signo de A, usar el residuo sin restar la lectura, restaurar sin huella o sin simetría, y alterar las identidades de kalman, delta y proximal o la varianza predictiva. Las pruebas detectan los 19 ([recibo](../../reports/engineering/kalman-associative-memory-mutations-20261010.json)). La mutación que quita la simetrización sobrevivió al principio, porque todas las pruebas usaban d ≤ 4 y ahí TᵀT sale simétrica. Una prueba con d = 64 la detecta. Es una selección dirigida, no una campaña completa de mutación.

La conexión con la ventana y la campaña tuvo su propia mutación dirigida el 13 de octubre, con 17 defectos aplicados de uno en uno: dividir mal las dos medias cuadráticas del ANOVA, no truncar la varianza común ni respetar el tope de ϱ, agrupar las sesiones sin su mercado o ignorar su concentración en n₀, atribuir cada residuo a EE. UU., usar la etiqueta sin restar el núcleo, adelantar el tramo de estimación, reutilizar al reanudar una estimación de otro tramo, no escalar q o p0 con σ², usar la ϱ estimada en la alternativa ϱ = 0, admitir modelos kalman sin receta o con una correlación no declarada, y quitar de la proyección de horas el tramo de estimación o su comprobación. Las pruebas detectan los 17 ([recibo](../../reports/engineering/kalman-window-mutations-20261013.json)). Los dos defectos de la proyección sobrevivían en la primera pasada porque solo se ejecutaron las pruebas de caudal. Los detectan las de la campaña A v2, que ahora forman parte de su selección.

## Coste medido sin entrenar

Medido el 10 de octubre de 2026 con `benchmarks/associative_memory_cost.py` dentro de `memslot light`, en un AMD Ryzen 9 8945HS con PyTorch 2.14.0+cu130 en CPU, dos hilos, d = 64 y m = 1, tres calentamientos y veinte repeticiones ([recibo](../../reports/engineering/kalman-associative-memory-cost-20261010.json)). La carga media del equipo era de 29,7 al empezar por otras sesiones, así que los percentiles 95 son ruidosos. Claves y valores sintéticos con semilla fija, solo para fijar formas.

| Resultados por evento | Delta p50 (ms) | Proximal p50 (ms) | Kalman p50 / p95 (ms) | Kalman con Huber p50 (ms) | Varianza kalman p50 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0,43 | 0,21 | 0,56 / 6,75 | 0,55 | 0,04 |
| 256 | 3,62 | 0,42 | 0,78 / 0,86 | 0,78 | 0,14 |
| 1.024 | 15,90 | 2,27 | 1,49 / 2,01 | 1,78 | 0,37 |
| 8.192 | 130,95 | 8,35 | 12,60 / 24,83 | 17,03 | 2,83 |

La escritura kalman cuesta lo mismo que la proximal en orden de magnitud y bastante menos que la delta en cohortes grandes. El estado exportado pasa de 512 bytes a 33.280 por la covarianza de 64 × 64. Con estas cifras no hay un cuello de botella que justifique optimizar. No se midieron energía ni GPU, porque B6 trabaja en CPU por diseño.

## Desviaciones y límites

- La medida de una cohorte se aplica al estado actual y no al de cada decisión. Con el paseo aleatorio, el error de esa aproximación está acotado por el ruido de proceso acumulado durante el retraso, como declaraba la propuesta.
- El paseo aleatorio se aplica una vez por escritura con resultados, no por sesión transcurrida. Una cohorte vacía no añade q.
- Los pesos de Huber se calculan en un solo paso con el residuo previo y no se iteran hasta converger.
- La propuesta daba q ∈ {10⁻⁶, 10⁻⁵} sin unidad. La escala del residuo solo se conoce al estimar σ², así que esos valores pasan a ser fracciones de σ², q = κ σ², igual que la tasa de PT2 pasó a ser relativa a su escala.
- La propuesta no fijaba p0. La declaración usa p0 = 0,1 σ² antes de ver datos y el modelo principal no usa Huber.
- σ² y ϱ se estiman con los residuos de los tres últimos meses del ajuste, no de todo el tramo. El recorrido cuesta lo mismo que una calibración y usa los residuos más próximos a la validación. Son residuos del tramo con el que se ajustó el padre, así que σ² puede quedar por debajo de la varianza fuera de muestra y ϱ puede bajar si el núcleo absorbió parte del componente común. La media de la corrección no depende de σ², pero sí de ϱ. No hay otro tramo anterior a la validación que no sea del ajuste.
- La regla aplica ϱ a la cohorte que se escribe junta, las etiquetas que maduran en el mismo evento, y la estimación agrupa por mercado e instante de decisión. Las dos agrupaciones solo coinciden si todas las etiquetas de una sesión maduran en el mismo evento y ninguna otra madura con ellas. Si una cohorte mezclara sesiones, ϱ sería la correlación dentro de la sesión aplicada a residuos que probablemente estén menos correlacionados entre sí. El 14 de octubre de 2026 `scripts/audit_kalman_cohorts.py` lo comprobó en las 19 vistas walk-forward de A v2 en US+CN. Usa las fases de la ventana B6 (tramo de estimación, validación, calibración y evaluación) y el mismo filtro que el índice de observaciones de `memory/financial_observations.py` (decisión dentro del tramo y madurez antes de su final), que pone cada etiqueta como evento en su instante de madurez. Solo lee etiquetas, sin entradas ni modelos, y deja 2024 sin abrir. Sumando ventanas y fases hay 32.439.451 etiquetas en 17.670 cohortes, y cada cohorte coincide con una sola sesión: ninguna mezcla sesiones ni mercados y ninguna sesión se reparte entre cohortes ([recibo](../../reports/engineering/kalman-cohort-composition-20261014.json)). En esta edición ϱ se aplica, por tanto, a la misma agrupación con la que se estima. Las vistas de un solo mercado todavía no están construidas. Sus etiquetas salen de la misma construcción, pero el recuento debe repetirse con ellas y siempre que cambie la regla de madurez. El recorrido tardó 2.110 s en un núcleo, con 0,93 GiB de memoria como máximo.
- La declaración inicial elegía κ una vez con las ventanas de 2005 a 2013. El 13 de octubre de 2026, antes de ejecutar nada, se cambió a una elección por ventana con la validación, la misma búsqueda que tiene B6 proximal, para que el contraste no dé a un modelo más selección que al otro.

## Pendiente sin medir

- El efecto predictivo, la cobertura de la varianza predictiva y la regla de decisión, bloqueados porque no hay predicciones de la campaña.
- La cobertura del intervalo de ±1,645 desviaciones predictivas y la ganancia con una sesión más de retraso (D3). La ventana las emite y la comparación las calcula desde el 14 de octubre de 2026, pero no hay predicciones de la campaña.
- La ventana completa con un padre aprendido sobre la cohorte financiera y su coste. La prueba con fixtures ya se ejecutó, pero no cubre ese resultado científico.

## Referencias

- Baltagi, B. H. (2021). *Econometric analysis of panel data* (6.ª ed.). Springer. https://doi.org/10.1007/978-3-030-53953-5
- Fentazi, M. R., Ameur, M. y Ksentini, A. (2026). *Training on the future: A delay-aware audit of test-time adaptation for time-series forecasting* (arXiv:2610.12232v1). https://arxiv.org/abs/2610.12232v1
- Huber, P. J. (1964). Robust estimation of a location parameter. *The Annals of Mathematical Statistics, 35*(1), 73–101. https://doi.org/10.1214/aoms/1177703732
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering, 82*(1), 35–45. https://doi.org/10.1115/1.3662552
- West, M. y Harrison, J. (1997). *Bayesian forecasting and dynamic models* (2.ª ed.). Springer. https://doi.org/10.1007/b98971
