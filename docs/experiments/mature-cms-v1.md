# Memoria multifrecuencia con errores maduros

`mature_cms_v1` es la pieza R2 de la segunda ola del estudio holográfico ([#782](https://github.com/GonxKZ/mars-titan/issues/782), [#786](https://github.com/GonxKZ/mars-titan/issues/786)). Da a cada experto de la mezcla E1 su propio periodo de escritura: la sesión, la semana o el mes. Está en `src/mars_titan/memory/multiscale_mixture.py` como un campo opcional de `MultiscaleMixtureConfig`, con [su declaración](../../configs/experiments/mature-cms-v1.json). Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha ejecutado y no hay ninguna mejora medida.

## Motivación y alcance

El sistema de memoria continua de Nested Learning encadena bloques que se actualizan con frecuencias distintas. Cada bloque acumula los gradientes de su tramo y los aplica al cerrarlo, de modo que los bloques lentos cambian poco y conservan lo aprendido durante más tiempo. En E1 los expertos de la corrección madura ya se diferencian por su semivida, pero todos escriben en cada evento. R2 pregunta si escribir los mismos errores maduros con frecuencias distintas mejora la corrección. Su control son los mismos tres expertos escritos en cada sesión.

R2 está en la corrección B6 que ejecuta la ventana de correcciones, no en el núcleo de Titans. Es una adaptación propia. El artículo actualiza por gradiente los parámetros de bloques MLP encadenados, y aquí cada experto es un corrector lineal con la regla proximal.

## Regla

Cada experto e declara un periodo: `session`, `week` o `month`. El primero escribe en cada sesión, y los periodos van de rápido a lento. El periodo de un corte t en microsegundos UTC es la semana ISO, que empieza el lunes a las 00:00 UTC, o el mes de calendario UTC. El experto de sesión escribe cada evento como en E1.

Un experto lento guarda en orden canónico los resultados maduros de su periodo abierto. En la primera escritura con un corte de un periodo posterior, y después del decaimiento de E1 hasta ese corte, aplica un único paso proximal sobre la pérdida media de los eventos del periodo cerrado P:

$$
A^{+}=\arg\min_A\ \tfrac12\lVert A-\rho A_e\rVert^2+\frac{\eta}{2}\sum_{j\in P}\frac{1}{|P|}\sum_i w_{ji}\bigl(v_{ji}-k_{ji}^\top A\bigr)^2,
$$

es decir, $(I+\eta K^\top WK)A^{+}=\rho A_e+\eta K^\top Wv$ con $W=\operatorname{diag}(w_{ji}/|P|)$. Los pesos $w_{ji}$ de cada evento suman uno, así que cada evento pesa lo mismo tenga las filas que tenga. Después el evento que llega abre el periodo nuevo. La lectura usa la mezcla de E1 sobre los estados aplicados, y el periodo abierto no se lee nunca.

CMS suma los gradientes de su tramo. La regla proximal exige pesos que sumen uno, de modo que aquí el periodo cerrado es la media de las pérdidas de sus eventos. El paso es implícito y estable sea cual sea η. Con la regla delta, que escribe fila a fila, cerrar un periodo equivaldría a escribir sus filas con retraso, así que R2 solo admite la regla proximal.

## Madurez

Ninguna velocidad lee un error antes de que madure. Cada fila de un periodo abierto llegó en una escritura que ya comprobó que su etiqueta estaba disponible antes del corte. Un experto lento solo la aplica en una escritura posterior y la lectura solo usa los estados aplicados. La prueba cambia el último resultado ya maduro de la primera semana y comprueba que la sesión lo lee desde su escritura, la semana desde el lunes siguiente y el mes desde febrero.

La lectura del primer evento de un periodo nuevo todavía no incluye el periodo cerrado, porque se aplica en la escritura de ese evento.

## Control

| Modelo | Cambio | Pregunta que contrasta |
| --- | --- | --- |
| `same_frequency` | Tres copias con semivida de 91 días escritas en cada sesión | Si escribir con frecuencias distintas mejora la corrección |
| `mature_cms` | Las mismas copias escritas en cada sesión, al cerrar cada semana y al cerrar cada mes | |

`same_frequency` es literalmente el control `replicated_quarter` de E1, así que en la criba es el mismo trabajo y no se repite. Los dos usan pesos uniformes, la misma semivida y la misma receta, de modo que solo difiere la frecuencia de escritura. Con la misma frecuencia las tres copias coinciden y la mezcla equivale a un único corrector.

## Correspondencia entre reglas, código y pruebas

| Regla | Código | Prueba |
| --- | --- | --- |
| Semana ISO desde el lunes y mes UTC | `period_index` | `test_periods_follow_iso_weeks_from_monday_and_utc_months` |
| Paso proximal del periodo cerrado | `MultiscaleMemory._continuum_write`, `PendingPeriod.closed` | `test_a_slow_expert_closes_its_period_with_one_mean_proximal_step`, `test_each_event_of_a_closed_period_weighs_the_same_whatever_its_rows` |
| Ninguna velocidad lee errores sin madurar | lectura de los estados aplicados | `test_no_speed_reads_a_result_before_the_write_that_closes_its_period` |
| Estado y recuperación de los periodos abiertos | `export`, `restore`, `PendingPeriod` | `test_restore_keeps_the_open_periods_and_the_next_write`, `test_restore_rejects_a_corrupted_open_period`, `test_session_window_and_recovery_keep_the_open_periods` |
| Ventana completa con periodos cerrados | `CorrectionInference` | `test_window_matches_an_independent_recurrence_of_closed_periods` |
| E1 sin cambios al no declarar periodos | `identity`, `export`, `_normalized` | `test_without_periods_the_identity_and_state_of_e1_do_not_change`, `test_the_factor_is_normalized_by_the_common_window_builder` |
| Límite de filas y bytes | `PendingPeriod.extend`, `state_tensor_bytes` | `test_an_open_period_holds_at_most_the_cohort_limit`, `test_state_bytes_count_the_open_periods_and_their_bound` |
| Declaración y control | [declaración](../../configs/experiments/mature-cms-v1.json) | `test_the_control_is_the_replicated_quarter_of_e1_and_the_same_screening_job`, `test_declared_models_are_executable_and_only_r2_names_its_periods` |

La fixture de la sesión solo escribe del 6 al 9 de julio de 2021, dentro de una semana y de un mes. La prueba de sesión se ejecuta con el calendario real, en el que los expertos lentos solo esperan, y con un calendario comprimido de semanas de un día y meses de dos, en el que cierran periodos. La regla real del calendario la comprueban las pruebas unitarias.

## Paridad e identidad

Sin `write_periods`, la identidad de E1, su identidad normalizada, los campos de su estado y sus huellas no cambian. Las pruebas fijan las huellas calculadas con el código anterior. Un periodo de sesión en todos los expertos es E1 y se rechaza como declaración de R2, para que cada cálculo tenga una sola identidad. Con periodos, la identidad añade `write_periods`, `continuum`, el reloj de los periodos y las reglas del periodo cerrado y del abierto. El estado añade los periodos abiertos y su huella, que entra en la del estado. Las métricas de la ventana añaden las filas aplicadas y las que espera cada experto.

## Coste

Medido en CPU con `benchmarks/multiscale_mixture_cost.py --declaration configs/experiments/mature-cms-v1.json` sobre el commit a60780a8b, sin pasos de optimizador. Las dos pasadas usan 64, 256 y 1.024 filas, cinco ciclos de calentamiento y cuarenta medidas por caso, con dos hilos y FP64 en el corrector, en un AMD Ryzen 9 8945HS. El estado medido parte de una cohorte que la sesión ya aplicó y que los dos expertos lentos esperan. Cada ciclo incluye el codec, la lectura y la escritura madura del día 14, en la que el experto semanal cierra ese periodo con su paso proximal y el mensual lo amplía. La [primera pasada](../../reports/engineering/mature-cms-20261028/benchmark-1.json) y la [segunda](../../reports/engineering/mature-cms-20261028/benchmark-2.json) conservan todas las muestras, percentiles y huellas.

| Filas | Mediana de R2, pasada 1 (ms) | Mediana del control, pasada 1 (ms) | Cociente, pasada 1 | Cociente, pasada 2 |
| --- | --- | --- | --- | --- |
| 64 | 6,95 | 7,12 | 0,98 | 1,04 |
| 256 | 11,42 | 10,55 | 1,08 | 1,05 |
| 1.024 | 28,66 | 26,13 | 1,10 | 1,33 |

La CPU estaba compartida con otras baterías, con una carga media de 12 a 17, y la segunda pasada tardó casi el doble en los dos modelos. En la primera, R2 cuesta entre lo mismo que el control y un 10 % más. R2 escribe dos veces por ciclo, la sesión y el periodo cerrado, y el control escribe tres, pero R2 además copia y concatena las filas de sus periodos abiertos. Con esta dispersión no se puede atribuir una diferencia menor que la propia variación entre pasadas. La medida con la máquina libre queda pendiente.

El estado numérico pasa de 1.560 bytes en el control a 72.216, 284.184 y 1.132.056 bytes con R2, porque los expertos lentos guardan las filas de su periodo abierto a 552 bytes por fila, con tres enteros, la clave de 64 coordenadas, el valor y el peso. La recuperación desde el estado exportado fue idéntica bit a bit en todos los casos. Cada periodo abierto guarda como mucho 8.192 resultados, así que la cota para los dos expertos lentos es de unos 9 MB dentro del artefacto de 64 MiB de la sesión.

## Desviaciones y límites

- CMS encadena bloques MLP y los actualiza por gradiente. Aquí hay tres correctores lineales con la mezcla convexa de E1 y la regla proximal, sin cadena.
- El periodo cerrado aplica la media de las pérdidas de sus eventos, no su suma.
- Los periodos son semanas y meses de calendario UTC, no sesiones de mercado. Una semana con festivo tiene menos eventos, y cada uno pesa lo mismo.
- La lectura del primer evento de un periodo nuevo no incluye el periodo cerrado.
- Un periodo abierto admite como mucho 8.192 resultados. Un mes de 256 activos, unas 5.400 filas, cabe. Uno de 512 activos no.
- Los tres expertos tienen la misma semivida para que solo cambie la frecuencia.
- No admite áreas, roles ni la regla delta.
- No reproduce ningún resultado publicado.

## Referencias

- Behrouz, A., Razaviyayn, M., Zhong, P. y Mirrokni, V. (2025). Nested Learning: The illusion of deep learning architectures. En *Advances in Neural Information Processing Systems 38* (pp. 46968–47002). https://doi.org/10.52202/085713-1565
