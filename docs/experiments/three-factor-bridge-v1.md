# Puente hebbiano de tres factores

`three_factor_bridge_v1` es P1 de la segunda ola del estudio holográfico ([#782](https://github.com/GonxKZ/mars-titan/issues/782), [#784](https://github.com/GonxKZ/mars-titan/issues/784)). Añade un peso rápido a cada arista entre dos áreas del puente de la [red holográfica H1](holographic-network-v1.md) y lo cambia durante el recorrido con el resultado maduro de las predicciones que usaron esa arista. La configuración está en `src/mars_titan/models/titans/plastic_bridge.py`, la lectura en `src/mars_titan/models/titans/holographic_network.py`, la regla en `src/mars_titan/memory/three_factor_bridge.py` y su paso por el recorrido en `src/mars_titan/training/financial_run.py`. La declaración, con sus dos fuentes y sus controles, está en [su configuración](../../configs/experiments/three-factor-bridge-v1.json). Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y alcance

H1 comunica sus áreas con atención sobre una topología fija, con los mismos parámetros en todos los mercados e instantes. Si una conexión entre dos áreas ayuda en un régimen y estorba en otro, solo el ajuste de los parámetros compartidos puede cambiarla, y lo hace despacio y para todos los mercados a la vez. P1 pregunta si conviene que la fuerza de cada conexión cambie durante el recorrido según hayan acertado o no las predicciones que la usaron, como una regla hebbiana modulada por un resultado que se conoce con retraso.

La forma de la regla viene de las reglas de tres factores que revisan Gerstner y otros (2018). Una coincidencia de actividad deja una traza de elegibilidad en la sinapsis (su ecuación 1, página 3 de la versión de arXiv), y el peso solo cambia cuando llega un tercer factor que multiplica esa traza (ecuación 2, página 6). El artículo define ese factor como desviación de su media móvil, por ejemplo la recompensa menos la esperada (página 7). P1 es la primera parte de #784. La consolidación de esquemas (P2), la reconsolidación (P3) y la repetición priorizada en el banco episódico quedan para después.

## Lectura del puente

Sean $E$ las aristas plásticas, los pares $(i,j)$ de áreas distintas que une la topología de H1, en orden de filas. Con el anillo de 19 áreas de la declaración hay 38. Cada mercado $m$ tiene un peso $w_{m,ij}$ por arista, que empieza en cero en cada recorrido. En cada paso del puente, el logit de la consulta del área $i$ sobre la clave $j$ de una fila del mercado es

$$
\ell_{ij}=\frac{q_i\cdot k_j}{\sqrt H}+w_{m,ij}\quad\text{si }(i,j)\in E,
$$

y el resto de pares permitidos por la máscara de H1 conserva su logit. La red recibe los pesos de cada fila como un tensor [filas, aristas] sin grafo y construye la máscara real `torch.where(máscara, peso, −∞)`. En las comprobaciones CPU, con pesos nulos esa máscara da en SDPA la misma atención que la booleana y el predictor emite H1 bit a bit en FP32 y en FP64. La paridad CUDA sigue pendiente.

La traza de elegibilidad de cada fila es la media de las probabilidades de atención realizadas en los $C$ ciclos y $L$ pasos locales,

$$
e_{ij}=\frac{1}{C\,L}\sum_{c,l}a^{(c,l)}_{ij},
$$

calculada sin grafo con la misma escala $1/\sqrt H$ que SDPA. Vale cero cuando falta una de las dos áreas. Un área ausente no envía mensajes y la consulta de un área ausente solo existe para que ninguna fila de la máscara quede vacía, porque su estado se descarta, así que la red anula también su traza.

## Regla

La traza de cada predicción emitida espera junto a ella hasta que madura su etiqueta y no decae mientras espera. Cuando madura, la regla calcula el tercer factor $\delta$ con una de las dos fuentes declaradas.

La brecha relativa del error (`relative_error_gap`) es la fuente de la issue. Con $\varepsilon=|\hat y-y|$ el error absoluto de la predicción realmente emitida y $E_m$ la expectativa del mercado antes del evento,

$$
\delta=\max\Bigl(\frac{E_m-\varepsilon}{E_m},-1\Bigr),
$$

que no pasa de 1 porque $\varepsilon\ge 0$ y es positivo cuando la predicción falló menos de lo esperado. En el primer evento del mercado, o con $E_m=0$, $\delta=0$. Después del evento, $E_m\leftarrow\lambda E_m+(1-\lambda)\,\bar\varepsilon$ con $\lambda=e^{-1/\tau}$, $\tau=21$ eventos de maduración y $\bar\varepsilon$ la media del evento, que es también el valor inicial.

La señal común (`mature_directional_hit`) convierte P1 en otro consumidor de la [fiabilidad madura](mature-reliability-signal-v1.md) de #791, como decidió la segunda ola. Con $h=\operatorname{sign}(\hat y)\operatorname{sign}(y)$ el acierto direccional de la etiqueta y $r_m$ la velocidad más lenta de la señal del mercado leída antes de que la señal reciba las etiquetas del evento,

$$
\delta=\frac{h-r_m}{2},
$$

que ya está en $[-1,1]$. Las dos fuentes son desviaciones de una expectativa, como en la definición del artículo.

Con las $n$ etiquetas del mercado que maduran en el evento, ordenadas por (flujo, decisión),

$$
w_m\leftarrow w_m+\eta\,\frac1n\sum_k\delta_k\,e_k .
$$

Después, para cada área destino $i$, si $\sum_j|w_{m,ij}|$ supera la cota $B$, sus aristas se escalan en proporción hasta $B$. La tasa $\eta=0{,}05$ hace que un evento con $\delta=1$ y traza máxima mueva cada peso como mucho 0,05 logits. La cota $B=1$ limita a un logit el cambio total de las aristas que llegan a un área, así que los pesos rápidos cambian las odds de atención como mucho en un factor $e$. Los tres valores se fijaron al declarar la variante, antes de abrir datos. Todas las etiquetas del evento usan la expectativa anterior al evento, el estado es FP64 y las sumas escalares usan `math.fsum`, de modo que el resultado no depende del orden de las etiquetas ni de su reparto en bloques.

## Orden temporal y estado

El recorrido aplica primero las etiquetas de un instante y después emite sus decisiones. Cada bloque lee los pesos de los mercados de sus filas como `bridge_bias`, las filas emitidas guardan su traza con su predicción pendiente y cada evento de etiquetas aplica la regla antes que la señal de #791, que es la que da el acierto esperado de la fuente común. Una decisión del instante $t$ lee así exactamente lo madurado hasta $t$. Ninguna etiqueta interviene antes de madurar y solo se compara con la predicción que se emitió.

Los pesos y las trazas se reinician en cada recorrido, como el estado rápido, así que el ajuste y cada validación empiezan sin pesos. El checkpoint guarda los pesos de cada mercado, sus contadores, la expectativa y las trazas pendientes. Al restaurarlo se exige que las trazas sean exactamente las de las predicciones pendientes. La repetición por bloques de flujos recorta los pesos con las filas de cada bloque, igual que la señal, de modo que cada lote se repite con los pesos de su emisión.

## Control y contrastes

El control `mean_third_factor_v1` usa el mismo puente y las mismas trazas, pero sustituye cada $\delta_k$ por la media de todos los $\delta$ calculados hasta el evento en el mercado. Conserva la escala y el signo medios del tercer factor y retira el mérito de cada predicción concreta. Ninguna de las configuraciones añade parámetros.

| Contraste | Variante | Control | Pregunta |
| --- | --- | --- | --- |
| `three_factor_bridge` | `error_gap` | `h1_reference` | ¿Aporta que las aristas cambien con el error maduro? |
| `credit_of_each_label` | `error_gap` | `error_gap_mean` | ¿Importa el mérito de cada predicción o basta la tendencia media? |
| `bridge_as_consumer_of_the_common_signal` | `common_signal` | `common_signal_reference` | ¿Aporta el puente como consumidor de la señal de #791? |
| `credit_of_each_label_with_the_common_signal` | `common_signal` | `common_signal_mean` | La misma pregunta del mérito con la fuente común |

Los dos primeros son los principales. Los otros dos pertenecen al hilo de la señal común y comparan con H1 más la señal de #791, para separar el efecto del puente del de las puertas que la señal ya modula. P1 se descarta si no mejora a su referencia y a su control de media con la regla de la criba. La declaración estima 2,6 h de GPU por configuración con tres semillas.

## Requisitos de ejecución

P1 se declara sobre H1 o H2, que tienen áreas del catálogo, y la ablación H3, sin áreas, lo rechaza. Las cabezas auxiliares de F5 recalculan los tokens fila a fila sin los pesos del recorrido, así que no se combinan con P1. El control local C de CM-v1 lo rechaza hasta tener su propio contraste. El consumidor congelado, Dust y el recorrido de MARS-TITAN con padre congelado rechazan H1 y, con ella, P1. C2 tampoco se combina con H1. El recorrido de Titans sí lo admite y guarda en su identidad la configuración y la regla. La fuente común exige la señal compartida de aciertos de #791, sin el reparto independiente ni los controles barajado y de sorpresa.

## Correspondencia con el código y las pruebas

| Elemento | Código | Prueba |
| --- | --- | --- |
| Aristas plásticas | `plastic_edges` | `test_the_plastic_edges_are_the_distinct_area_pairs_of_the_topology` |
| Peso en el logit y traza | `HolographicNetwork.forward` con `edge_bias` y `traces` | `test_the_bias_and_the_trace_match_a_reference_written_row_by_row`, `test_the_trace_is_a_probability_between_present_areas` |
| Paridad con pesos nulos | máscara real de ceros y −∞ | `test_zero_weights_keep_the_bridge_output_bit_for_bit`, `test_the_predictor_with_zero_weights_emits_h1_bit_for_bit` |
| Gradientes | pesos sin grafo | `test_gradients_cross_the_biased_bridge_and_never_reach_the_weights` |
| Brecha del error y expectativa | `BridgePlasticity.settle` y `_update` | `test_the_first_event_only_sets_the_expectation`, `test_a_later_event_follows_the_rule_written_by_hand`, `test_the_third_factor_is_clipped_and_zero_without_an_expectation` |
| Normalización homeostática | escalado en `_update` | `test_the_homeostasis_scales_the_edges_of_each_area_in_proportion` |
| Control de media | rama `mean_third_factor_v1` | `test_the_control_replaces_each_third_factor_by_its_running_mean` |
| Señal común | `_third_factor` y `ReliabilitySignal.expected_hits` | `test_the_common_signal_gives_half_the_gap_between_the_hit_and_its_expectation`, `test_the_expected_hit_is_the_slowest_speed_of_the_shared_channel`, `test_the_common_third_factor_needs_the_shared_signal_of_hits` |
| Referencia aleatoria y orden | FP64 en el orden de las claves | `test_random_events_match_the_reference_and_ignore_the_label_order`, `test_a_market_does_not_move_the_weights_of_another` |
| Atomicidad | validación previa en `hold` y `settle` | `test_an_invalid_event_is_refused_before_changing_anything`, `test_a_trace_is_held_once_and_atomically` |
| Lectura causal en el recorrido | `_observe`, `_emit` y `_labels` | `test_each_decision_reads_the_weights_of_the_labels_matured_before_it`, `test_altering_the_data_after_a_cut_changes_nothing_before_it`, `test_the_bridge_reproduces_its_reference_until_its_weights_move` |
| Recuperación | `export`, `restore` y el checkpoint del recorrido | `test_resume_from_an_export_continues_bit_for_bit`, `test_restore_refuses_corrupted_or_foreign_states`, `test_resume_after_interruption_reproduces_the_continuous_run` |
| Repetición por bloques | recorte de `bridge_bias` en `_replay` | `test_replay_by_flow_blocks_keeps_the_weights_of_each_emission` |
| Identidad y consumidores | `identity`, `_code` y el registro de cambios | `test_the_disabled_field_keeps_the_identity_and_each_kind_is_another_identity`, `test_the_registry_admits_it_and_the_consumers_refuse_it`, `test_the_checkpoint_and_the_code_identity_follow_the_declared_bridge` |

## Desviaciones respecto a las fuentes

- La traza no es el producto de una actividad presináptica por una función de la postsináptica, como en la ecuación 1 de Gerstner y otros, sino la probabilidad de atención realizada entre las dos áreas, que depende de la consulta y de la clave.
- La traza no decae mientras la predicción espera, el caso $\tau_e\to\infty$ que el artículo también contempla. El retraso hasta madurar es el horizonte de la etiqueta y no un tiempo biológico.
- El tercer factor es del mercado y llega a todas sus aristas, como la señal global sin índice de neurona del artículo. No hay un factor por área ni por activo.
- La brecha del error se divide por su expectativa para no depender de la escala de los rendimientos y se recorta a $[-1,1]$. La del acierto se divide por dos para quedar en el mismo intervalo.
- La normalización solo reduce. El escalado sináptico de Turrigiano (2008) sube o baja de forma multiplicativa todas las sinapsis de una neurona hacia una actividad objetivo. Aquí solo hace de cota cuando la suma de un área pasa de $B$.
- La regla es discreta, un paso por evento de maduración, y no continua en el tiempo.
- Los pesos no se consolidan entre recorridos. Eso corresponde a P2.
- No reproduce ningún resultado publicado.

La regla puede aumentar el error incluso con un tercer factor positivo. En una atención escalar de dos claves, con probabilidades $(4/5,1/5)$ y valores $(1,-1)$, la salida es $3/5$. Si el objetivo es $1/2$ y la expectativa de error es $1/5$, el error absoluto es $1/10$ y $\delta=1/2$. Con $\eta=1/20$, la primera conexión plástica recibe un sesgo $1/50$ y la segunda, fija, conserva cero. Las odds pasan de $4$ a $4\exp(1/50)$, así que la salida sube y el error aumenta. La suma de sesgos sigue por debajo de $B=1$ y la normalización no interviene. Es un contraejemplo algebraico a una garantía de descenso, no una evaluación predictiva de MARS-TITAN.

## Comprobación técnica

Pendiente de la verificación final.

## Coste

La medida CPU del 29 de octubre (`benchmarks/three_factor_bridge_cost.py`, commit `38f1363dc`) llama a `HolographicNetwork` con la H1 de la declaración, 19 áreas en anillo y 38 aristas plásticas, en FP32 y con H = 64, dos hilos, tres calentamientos y veinte repeticiones en cada uno de los dos órdenes. Las entradas son las de la medida de H1. Los pesos rápidos son fijos, aleatorios y dentro de la cota, y la red devuelve sus trazas como en el recorrido. `forward` es la inferencia sin gradiente y `backward` añade la propagación de una suma de cuadrados, sin optimizador. La carga media de la máquina estaba entre 12 y 17 en 16 hilos lógicos, así que los percentiles son descriptivos. El [recibo](../../reports/engineering/three-factor-bridge-20261029/cost-cpu.json) conserva todas las muestras.

| Forward | 64 filas | 256 filas | 1.024 filas |
| --- | ---: | ---: | ---: |
| H1 | 13,6 ms (16,3) | 42,9 ms (56,8) | 184,6 ms (211,6) |
| H1 con P1 | 15,4 ms (16,2) | 45,8 ms (66,2) | 231,6 ms (262,0) |

| Backward | 64 filas | 256 filas | 1.024 filas |
| --- | ---: | ---: | ---: |
| H1 | 34,9 ms (35,8) | 113,2 ms (146,4) | 460,0 ms (543,5) |
| H1 con P1 | 35,9 ms (37,8) | 114,6 ms (139,9) | 444,5 ms (516,2) |

Cada celda da la mediana y, entre paréntesis, el percentil 95 de las cuarenta muestras. P1 no añade parámetros a los 53.379 de la red. El backward no cambia de forma apreciable, porque la traza se calcula sin grafo. El forward cuesta más con 1.024 filas. Para saber de dónde sale, una medida aparte de cada operación con 1.024 filas, 23 nodos y los mismos dos hilos dio por paso del puente 17,6 ms para SDPA con la máscara real y 19,5 ms con la booleana, así que la máscara real no encarece la atención. La traza cuesta 7,4 ms por paso, sobre todo por volver a calcular $q\,k^\top$ (4,4 ms) y su softmax (1,2 ms), y la máscara real y la colocación de los pesos suman unos 3 ms por llamada. Con dos pasos son unos 18 ms, cerca del 10 % de la referencia. La diferencia de medianas de la tabla, 47 ms, es mayor que esa suma y queda dentro de la dispersión de una máquina cargada. Evitar el segundo producto exigiría calcular la atención a mano en lugar de con SDPA, y eso rompería la paridad bit a bit con H1 cuando los pesos son nulos, así que no se ha cambiado.

La regla, medida en CPU con el mismo recibo, tarda por evento entre 0,4 y 0,8 ms en guardar las trazas y aplicar las etiquetas de 64 predicciones de un mercado, entre 1,5 y 2,3 ms con 256 y entre 6,2 y 8,6 ms con 1.024, según la configuración. Leer los pesos de esas filas tarda entre 0,07 y 1,5 ms. El estado de cada mercado ocupa 304 bytes de pesos en FP64 y cada predicción pendiente añade otros 304 bytes de traza. No se han medido la VRAM ni la energía, y la medida CUDA queda pendiente hasta que la GPU esté libre.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 uv run python benchmarks/three_factor_bridge_cost.py --run \
  --output "${MARS_THREE_FACTOR_COST:?}"
```

## Pendiente

Faltan la medida CUDA, la entrada de las cuatro configuraciones en la criba y la comparación predictiva con las mismas filas, cortes y presupuesto. P2, P3 y la repetición priorizada siguen en #784. El aprendizaje sigue bloqueado y la variante no se activa por defecto.

## Referencias

- Gerstner, W., Lehmann, M., Liakoni, V., Corneil, D. y Brea, J. (2018). Eligibility traces and plasticity on behavioral time scales: Experimental support of neoHebbian three-factor learning rules. *Frontiers in Neural Circuits, 12*, 53. https://doi.org/10.3389/fncir.2018.00053
- Turrigiano, G. G. (2008). The self-tuning neuron: Synaptic scaling of excitatory synapses. *Cell, 135*(3), 422–435. https://doi.org/10.1016/j.cell.2008.10.008
