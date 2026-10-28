# Puerta de familiaridad del lector episódico

`familiarity_gate_v1` es la pieza R3 de la segunda ola del estudio holográfico ([#782](https://github.com/GonxKZ/mars-titan/issues/782), [#786](https://github.com/GonxKZ/mars-titan/issues/786)). Hace que el refinamiento del lector episódico pese menos cuando los episodios recuperados se parecen poco al presente. Está en `src/mars_titan/models/titans/episodic_readout.py` como el campo opcional `familiarity_gate` de `EpisodicReadoutConfig` y de la receta del lector, con [su declaración](../../configs/experiments/familiarity-gate-v1.json). Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha ejecutado y no hay ninguna mejora medida.

## Motivación y alcance

El hipocampo se ha descrito como un comparador entre la experiencia pasada y la presente, que emite una señal de novedad cuando lo recuperado no coincide con lo que llega. Kumaran y Maguire (2007) encontraron en humanos una actividad hipocampal compatible con un detector de coincidencia y discrepancia asociativa. En los modelos de lenguaje con recuperación, Drozdov y otros (2022) observaron que el kNN-LM ayuda sobre todo cuando los elementos recuperados se parecen semánticamente a la consulta, y fijaron el coeficiente de interpolación según esa calidad de recuperación.

R3 traslada esa idea al lector episódico de MARS-TITAN, que refina el estado de trabajo de Titans-MAC con los episodios maduros del banco M1. Es una adaptación propia. No modela la fisiología del hipocampo ni reproduce el kNN-LM, cuya mezcla es entre dos distribuciones de palabras y no entre dos estados. La pregunta es si pesar el refinamiento con la familiaridad de su lectura mejora la predicción frente a la misma puerta sin esa dependencia.

## Regla

En cada refinamiento k, el lector normaliza la consulta q̂ y elige los ocho episodios de mayor coseno. Las claves del banco ya tienen norma uno, así que $s_j=\hat q^\top k_j$ es el coseno con el episodio j y la atención es $w=\operatorname{softmax}(s/T)$. La familiaridad de la lectura es el coseno medio con esos pesos, y la puerta escala el cambio del refinamiento:

$$
f=\sum_j w_j s_j,\qquad g=2\,\sigma(a f+b),\qquad z_{k+1}=z_k+\sigma(\theta)\,g\odot\tanh\bigl(U[z_k,z_0,r_k,p_k]\bigr).
$$

a y b son dos escalares que empiezan en cero, así que la puerta empieza en $2\sigma(0)=1$ y el lector activo empieza siendo el de referencia. El rango (0, 2) deja que la memoria pese también más que en la referencia cuando la lectura es familiar. El signo de a no se fija: si la hipótesis es cierta, el ajuste debería darle un valor positivo. Las filas sin episodios, al principio de cada pasada con el banco vacío, usan el paso de referencia.

f no lleva gradiente. La puerta aprende a usar la familiaridad, pero la consulta no recibe gradiente a través de ella, así que no se ajusta para fabricarla. Por eso `refinement_jacobians` rechaza un lector con R3, porque su Jacobiano omitiría la dependencia de f respecto del estado.

## Control

| Modelo | Puerta | Pregunta que contrasta |
| --- | --- | --- |
| `familiarity_gate` | $2\sigma(af+b)$ | |
| `constant_gate` | $2\sigma(b)$ en todas las filas con episodios | Si depender de la familiaridad mejora la predicción |
| `reader_reference` | Ninguna | Si una escala aprendida de la memoria cambia algo por sí sola |

El control es la misma puerta con la familiaridad sustituida por una constante aprendida, que hace el papel de su valor medio. Tiene la misma inicialización, la misma lectura y el mismo refinador, y un parámetro menos. `reader_reference` es el lector M1 con K = 1 de la criba anterior. Los tres usan la misma receta, el mismo padre congelado y el mismo presupuesto de actualizaciones.

## Correspondencia entre reglas, código y pruebas

| Regla | Código | Prueba |
| --- | --- | --- |
| Arranque exacto en la referencia, con los mismos gradientes compartidos | `EpisodicReadout.__init__`, `_step` | `test_a_new_gate_starts_exactly_at_the_reference_reader`, `test_a_new_gate_keeps_the_prediction_of_the_frozen_parent_and_the_reader` |
| Familiaridad, puerta y paso de cada fila | `_familiarity`, `_familiarity_gate`, `_step` | `test_the_gate_follows_its_equation_in_every_row` |
| Paso de referencia sin episodios | `_read_parts`, `_familiarity_gate` | `test_without_episodes_the_step_is_the_reference_one` |
| Familiaridad de cada paso con episodios fijados o en caché | `_read_parts` | `test_fixed_episodes_and_cached_reads_keep_the_familiarity_of_each_step` |
| Solo sobre el banco y sin puente regional | `EpisodicReadoutConfig.__post_init__` | `test_the_gate_needs_its_name_and_the_bank`, `test_the_gate_is_not_combined_with_the_regional_bridge` |
| Lecturas sin familiaridad rechazadas | `refine`, `_familiarity_gate` | `test_a_read_without_its_familiarity_is_refused_by_the_gate` |
| Estado, identidad y presupuesto de trabajo | `identity`, `state_dict`, `estimated_bytes` | `test_without_the_gate_the_identity_of_the_reader_does_not_change`, `test_the_identity_names_the_gate_and_its_control`, `test_the_gate_is_saved_with_the_reader_and_refused_by_another_contract`, `test_the_working_budget_counts_the_gate` |
| Jacobiano del refinamiento | `refinement_jacobians` | `test_the_transition_jacobian_refuses_the_gate` |
| Receta, ventana y criba | `ReadoutRecipe`, `_readout`, `ReadoutTrainer` | `test_the_recipe_names_the_gate_only_when_declared`, `test_the_window_builds_the_reader_with_the_gate_of_its_recipe`, `test_the_reader_run_needs_the_gate_of_its_recipe`, `test_the_declared_models_differ_only_in_the_gate_and_reach_the_screening_recipe` |

## Paridad e identidad

Sin `familiarity_gate`, la identidad del lector, sus parámetros y su receta no cambian. Las pruebas fijan las huellas de tres configuraciones calculadas con el código anterior. Con la puerta, la identidad añade el nombre de la puerta, la definición de la familiaridad, la forma de la puerta, su inicialización, lo que escala y el paso sin episodios. El estado añade `familiarity_bias` y, en R3, `familiarity_weight`. Los ceros no consumen el generador, así que las proyecciones y el refinador se sortean igual.

Con la puerta en uno, el producto por el cambio es exacto. El estado refinado y los gradientes de todos los parámetros compartidos coinciden bit a bit con los del lector de referencia en CPU, en FP32 y FP64 y con uno, dos o cuatro refinamientos. La misma prueba en CUDA queda pendiente de tener la GPU libre. La puerta escala el cambio antes del paso para que la reducción del gradiente de θ siga siendo la misma.

## Coste

Medido en CPU con `benchmarks/familiarity_gate_cost.py` sobre el commit 0c3c8fab5, sin pasos de optimizador. Las dos pasadas usan un bloque de 512 filas, 64 coordenadas, ocho vecinos y una instantánea de 1.024 episodios, con tres ciclos de calentamiento y treinta repeticiones por modelo en orden rotatorio, dos hilos y FP32 estricto, en un AMD Ryzen 9 8945HS. Cada repetición mide la lectura con grafo, su backward y la lectura sin grafo. Antes de medir, las dos puertas reprodujeron bit a bit el estado de la referencia, y ningún parámetro cambió durante la medida. La [primera pasada](../../reports/engineering/familiarity-gate-20261028/benchmark-1.json) y la [segunda](../../reports/engineering/familiarity-gate-20261028/benchmark-2.json) conservan todas las muestras y las huellas de las fuentes.

| Medida | Mediana de la referencia, pasada 1 (ms) | Cociente de `familiarity_gate`, pasadas 1 y 2 | Cociente de `constant_gate`, pasadas 1 y 2 |
| --- | --- | --- | --- |
| Lectura con grafo | 57,75 | 1,00 y 1,01 | 1,00 y 1,00 |
| Backward | 4,40 | 1,02 y 1,04 | 1,07 y 1,01 |
| Lectura sin grafo | 58,56 | 1,00 y 1,01 | 1,01 y 1,00 |

La CPU estaba compartida con otras baterías, con una carga media de 15 a 24 en sus 16 hilos. Las dos lecturas cuestan lo mismo que la referencia con un margen del 1 %, porque la consulta, la selección de vecinos y la proyección de los episodios dominan el tiempo. El backward de la referencia pasó de 4,4 a 7,0 ms entre pasadas, y las diferencias de las puertas en él no conservan su orden de una pasada a otra, así que no se atribuyen a R3. La medida en CUDA y con la máquina libre queda pendiente.

El presupuesto de trabajo estimado sube de 24.068.096 a 24.240.128 bytes sin grafo (un 0,7 %) y de 45.744.128 a 46.432.256 bytes con grafo (un 1,5 %). La puerta de familiaridad añade dos parámetros escalares y su control uno, sobre los 20.801 del lector. El pico de memoria residente del proceso fue de 687 y 715 MiB.

## Desviaciones y límites

- La señal de novedad del hipocampo se reduce a un coseno medio entre la consulta y los episodios leídos. No hay comparación entre una predicción y lo observado.
- La puerta no cambia la admisión ni la escritura del banco. Que la novedad favorezca codificar episodios nuevos queda fuera de R3.
- La familiaridad no lleva gradiente.
- No admite el puente regional, cuyas lecturas no tienen una sola familiaridad por fila, ni el control sin banco.
- No reproduce ningún resultado publicado.

## Referencias

- Drozdov, A., Wang, S., Rahimi, R., McCallum, A., Zamani, H. y Iyyer, M. (2022). You can't pick your neighbors, or can you? When and how to rely on retrieval in the kNN-LM. En *Findings of the Association for Computational Linguistics: EMNLP 2022* (pp. 2997–3007). https://aclanthology.org/2022.findings-emnlp.218/
- Kumaran, D. y Maguire, E. A. (2007). Match–mismatch processes underlie human hippocampal responses to associative novelty. *Journal of Neuroscience, 27*(32), 8517–8524. https://doi.org/10.1523/JNEUROSCI.1677-07.2007
