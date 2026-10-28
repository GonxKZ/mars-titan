# Autoedición acotada de la salida

`posthoc_self_edit_v1` es la variante G1 del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#748](https://github.com/GonxKZ/mars-titan/issues/748)). Esta entrega cubre la gramática de salida: un ciclo propone, elige y contrasta reescrituras de las predicciones guardadas de un ajuste, sin tocar pesos. La gramática de arquitectura, que combinaría componentes registrados como `predictor_changes` y emitiría recetas, queda pendiente porque depende del motor de la criba de [#746](https://github.com/GonxKZ/mars-titan/issues/746). Todo lo descrito está implementado y comprobado con pruebas técnicas sobre paneles sintéticos. No se ha ejecutado ningún ciclo con datos y no hay ninguna mejora medida.

El [diagrama de G1](../diagrams/variants/g1-posthoc-self-edit.png) resume el recorrido.

## Motivación y fuente

La idea de partida es una autoedición acotada al estilo de SEAL ([Zweiger y colaboradores, versión 2](https://arxiv.org/html/2506.10943v2)). En SEAL un modelo de lenguaje genera sus propias ediciones, un bucle interior las aplica y un bucle exterior las refuerza según su rendimiento posterior. Aquí no hay modelo de lenguaje ni refuerzo. Las ediciones son configuraciones de una gramática cerrada, las propone un generador que solo ve la partición G y su utilidad se mide en bloques posteriores. La pregunta que responde el ciclo es si ese generador elige mejor que un placebo aleatorio con la misma gramática y el mismo número de candidatas.

## Gramática de salida

`src/mars_titan/evaluation/output_program.py` define cuatro campos con dominios finitos y tipados. Sea s la semilla del trabajo, S el número de semillas y q los cuantiles emitidos:

$$
\bar q=w\,q_s+(1-w)\frac{1}{S}\sum_k q_k,\qquad
A_i=\mathbf 1\{l_i>0\ \text{o}\ u_i<0\},\qquad
T_i=\mathbf 1\{|\bar m_i|\ge Q_\tau(|\bar m|\ \text{en G del mercado})\},
$$

$$
\hat y_i=\begin{cases}\lambda_g\,\bar m_i&\text{si } A_i \text{ y } T_i,\\0&\text{en otro caso.}\end{cases}
$$

| Campo | Dominio | Papel |
| --- | --- | --- |
| `seed_weight` (w) | 1, 0,75, 0,5, 0,25, 0 | Pesos de combinación entre la propia semilla y la media de semillas, nivel a nivel |
| `coverage` (c) | ninguna, 0,8, 0,95 | Abstención de A2 con el intervalo [l, u] de cobertura c, calibrado con la CQR de q̄ en G |
| `threshold` (τ) | 0, 0,25, 0,5, 0,75 | Cuantil lineal de \|m̄\| en G de cada mercado bajo el que no se emite |
| `shrinkage` (λ) | 1, 1/4, 1/16, 1/64, `fit_g` | Factor fijo o el λ por mercado de A1, elegido en G con el MAE por sesión |

m̄ es la mediana de q̄. La gramática tiene 300 configuraciones. Los valores fijos de λ bajan en potencias de cuatro hasta 1/64 porque la fase 0 encontró λ entre 0 y 0,019 en los pilotos. Una combinación convexa de cuantiles ordenados sigue ordenada en coma flotante, porque el redondeo es monótono. La configuración de referencia (w = 1, sin cobertura, λ = 1 y τ = 0) emite la mediana de la propia semilla bit a bit. Se registra como etapa de salida en `evaluation/output_stages.py`, igual que A2, y no como `predictor_changes`.

## Ciclo

`src/mars_titan/proposals/self_edit_cycles.py` lleva el registro y `src/mars_titan/proposals/output_self_edit.py` recorre el ciclo de salida. Cada etapa solo recibe los tramos que puede ver.

1. **Declaración con G.** El generador (`inner_validation_rank_v1`) ajusta cada configuración con la validación de G, la puntúa con el MAE medio por sesión en la calibración de G y propone las N = 8 mejores distintas de la vigente. El placebo muestrea sin reemplazo otras 8 de la misma gramática, también sin la vigente, con `SeedSequence([semilla del plan, ciclo])`. El registro fija las dos listas, la huella de la calibración y la mayor madurez de G antes de abrir S. N lo declara el plan y cualquier lista de otro tamaño se rechaza.
2. **Selección en un primer bloque de S.** Cada configuración se ajusta con la calibración de G y se aplica al bloque. Se elige la mejor de cada lista con empates por el orden declarado. La del generador solo se adopta si mejora estrictamente a la vigente. Las 16 candidatas quedan cobradas y las rechazadas conservan su motivo.
3. **Contraste en un bloque posterior de S.** La serie de MAE por sesión de la elegida del generador menos la del placebo pasa por `compare_series`, con bloque, réplicas, semilla y confianza fijados en el plan. El ciclo bate al placebo si el intervalo queda entero por debajo de cero. Si las dos listas eligen la misma configuración no hay contraste y el ciclo no bate al placebo.

El bloque E del estudio no interviene. Cada bloque debe empezar después de que maduren las etiquetas del anterior, y el contraste de un ciclo usa un bloque posterior al contraste del ciclo previo. Así ningún veredicto se apoya en filas que otro veredicto ya vio.

## Puerta entre ciclos

La decisión 13 de #742 dice que cada ciclo edita la salida y que el segundo ciclo solo se ejecuta si el primero bate al placebo. El registro lo implementa como una cadena. Un ciclo, de salida o de arquitectura, solo puede declararse si el anterior está contrastado y batió a su placebo. Esta lectura bloquea el segundo ciclo completo, sea de salida o de arquitectura, y cumple la decisión con las dos lecturas posibles de «segundo ciclo». Al restaurar un estado también se rechaza una secuencia que contenga un ciclo posterior a otro que no batió al placebo.

## Correspondencia entre ecuaciones, módulo y pruebas

| Elemento | Código | Prueba |
| --- | --- | --- |
| Gramática cerrada y tipada | `programs`, `checked_program` | `test_the_grammar_is_closed_typed_and_contains_the_reference` |
| q̄ nivel a nivel | `combine` | `test_the_combination_averages_quantiles_level_by_level` |
| A con la CQR de q̄ | `fit`, `apply` | `test_the_coverage_field_is_the_neutral_abstention_of_the_combined_quantiles` |
| T con el cuantil de G | `fit`, `apply` | `test_the_threshold_uses_the_declared_quantile_of_each_market_in_g` y `test_a_median_on_the_threshold_is_emitted` |
| λ fijo o de A1 | `fit`, `apply` | `test_shrinkage_is_fixed_or_the_market_lambda_of_g` |
| Paridad de la referencia | `combine`, `apply` | `test_the_reference_program_emits_the_own_median_bit_for_bit` y `test_the_reference_needs_no_other_seed_and_starts_when_g_matures` |
| Generador con G | `generate`, `score` | `test_the_generator_proposes_the_best_programs_of_g_only` y `test_the_score_gives_each_session_the_same_weight` |
| N fijo, cobro y placebo | `declare`, `select`, `placebo` | `tests/proposals/test_self_edit_cycles.py` |
| Veredicto | `assess` | `test_the_cycle_beats_the_placebo_only_with_a_favourable_interval` y `test_the_contrast_must_follow_the_declared_rule` |
| Puerta | `declare`, `restore` | `test_a_second_cycle_opens_only_after_the_first_beats_its_placebo` y `test_the_state_survives_export_and_resumes_a_cycle_halfway` |
| Ciclo completo | `declare_output_cycle`, `select_output_cycle`, `assess_output_cycle` | `test_a_full_output_cycle_declares_before_selecting_and_assesses_later` |

## Causalidad, estado y paridad

- La CQR, λ y los umbrales se ajustan solo con filas de G. `output_program.apply` rechaza un tramo con alguna emisión anterior a la madurez de esas etiquetas y no lee sus objetivos. La prueba cambia todos los objetivos y obtiene la misma salida bit a bit.
- El generador recibe solo la validación y la calibración de G. La selección y el contraste exigen ajustar con la misma calibración que vio el generador, comprobada por su huella.
- El registro rechaza bloques que empiezan antes de que maduren las etiquetas anteriores, incluidos los del ciclo previo. Los instantes de 2024 se rechazan.
- El registro se exporta en JSON con su huella y se restaura para continuar un ciclo a medias. La prueba reanuda un ciclo declarado y obtiene el mismo estado que sin interrupción. Un estado alterado se rechaza por su huella o por su secuencia.
- Sin etapa declarada la salida es la del modelo. La configuración de referencia también la reproduce bit a bit, con una sola semilla o con varias.

## Desviaciones y límites

- No hay modelo de lenguaje ni aprendizaje del generador. SEAL refuerza el generador con las ediciones que funcionan. Aquí el generador es una ordenación fija dentro de G y el ciclo solo comprueba si esa ordenación aporta algo frente al azar.
- El registro no reutiliza las clases de `proposals.ledger` (#711). Exigen una `MatureReview` de una `FinancialSession` y recetas `FinancialConfig`, que no existen sobre predicciones guardadas. Sí sigue su contrato de cuota fija, cargos sin devolución y estado sellado.
- La combinación solo mezcla semillas del mismo ajuste. No combina modelos distintos.
- λ con `fit_g` se elige con todas las filas de G, antes de la abstención y del umbral.
- El contraste es una familia de un solo contraste. Su corrección por multiplicidad entre ciclos o entre variantes la fija el protocolo de la criba.
- El generador puntúa las 299 configuraciones distintas de la vigente. Una configuración que no puede ajustarse (por ejemplo, un mercado con menos filas que el mínimo de la CQR) hace fallar el ciclo en lugar de desaparecer en silencio.

## Coste medido

Sin tensores ni parámetros entrenables no hay forward ni backward. `benchmarks/output_self_edit_cost.py` mide en CPU un ciclo completo con paneles sintéticos de semilla fija que solo fijan las formas: 256 activos de EE. UU., cinco semillas, dos hilos y cuatro tramos consecutivos del mismo tamaño. El contraste usa 2.000 réplicas y bloques de 16 días. El recibo está en `reports/engineering/output-stages-20261023/output-self-edit-cost.json`. Son medianas de tres repeticiones con otras sesiones compartiendo la máquina (carga media de 2,6 al empezar).

| Tramos | Filas por tramo | Generador (299 configuraciones) | Ciclo completo | Pico de NumPy del ciclo | Estado exportado |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tres meses | 16.128 | 1,53 s | 1,73 s | 3,9 MB | 3,6 KB |
| Seis meses | 32.256 | 3,49 s | 3,22 s | 7,8 MB | 3,6 KB |

El generador domina el ciclo. Con solo tres repeticiones y carga concurrente, en el caso de seis meses su mediana supera a la del ciclo completo, que también lo incluye. Un ciclo cuesta segundos y no se ha optimizado. No se midieron la GPU, la lectura de Parquet ni la energía.

## Mutaciones dirigidas

Se aplicaron a mano 28 mutaciones sobre la lógica crítica y se revirtieron después. Todas hacen fallar alguna prueba. Cinco sobrevivían al principio y motivaron pruebas nuevas: la igualdad con el umbral, el empate con la vigente, la madurez de G como la mayor de validación y calibración, el MAE medio y no mediano por sesión, y la calibración comprobada también en el contraste.

- En la gramática: la ruta exacta de w = 1, los pesos de la combinación invertidos, la abstención y la emisión invertidas, `>=` por `>` en el umbral y en la madurez, umbrales calculados con τ = 0, la madurez del ajuste tomada como mínimo y la comprobación de semillas eliminada.
- En el registro: la puerta entre ciclos y su comprobación al restaurar, el intervalo favorable, la mejora estricta, el cobro de las 2N, el placebo con la vigente o sin el ciclo en su semilla, el bloque que empieza justo al madurar, las listas repetidas o con la vigente, el mínimo por el máximo, la regla de contraste del plan y el bloque nuevo para cada veredicto.
- En el ciclo: las N peores en vez de las mejores, el contraste invertido, la madurez de G, la media por la mediana y la comprobación de la calibración en la selección y en el contraste.

## Pendiente

- La gramática de arquitectura sobre componentes registrados como `predictor_changes`, que emitirá recetas para el motor de #746 sin ejecutarlas.
- Aplicar los ciclos en el análisis de la criba sobre los Parquet de cada trabajo (#746).
- Cualquier ejecución con datos, que necesita la aprobación de los ciclos de autoedición prevista en #742 y sigue bloqueada mientras rija el bloqueo de aprendizaje.
