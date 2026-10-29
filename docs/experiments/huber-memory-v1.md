# Pérdida de Huber en la memoria de Titans

`huber_memory_v1` es la pieza R1 de la segunda ola del estudio holográfico ([#782](https://github.com/GonxKZ/mars-titan/issues/782), [#786](https://github.com/GonxKZ/mars-titan/issues/786)). Sustituye la pérdida asociativa L2 de la memoria de Titans por la de Huber con un umbral por canal que depende de la entrada. La pérdida y el umbral están en `src/mars_titan/models/titans/huber_memory.py`, la escritura en `NeuralMemory.update` y la declaración en [su configuración](../../configs/experiments/huber-memory-v1.json). Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y alcance

La memoria de Titans escribe con el gradiente de ‖M(k) − v‖², así que un token cuyo valor la memoria predice muy mal escribe con una fuerza proporcional a su error. En datos financieros esos tokens suelen ser sesiones con saltos o noticias aisladas. R1 contrasta la hipótesis contraria a la de E3, que refuerza las escrituras de los errores grandes: que la memoria funciona mejor si recorta la influencia de cada canal con un residuo excesivo. El control es el núcleo con L2.

La pérdida solo cambia en la memoria por flujo de MAC y solo con `mac_online`, la única variante que escribe. La memoria compartida de PT4 conserva L2 y rechaza la pieza si se le pasa. El control local C de CM-v1 tampoco se combina con ella sin su propio contraste, porque su presupuesto está contrastado con el núcleo. El consumidor congelado sí la admite, porque llama al mismo MAC y registra el código nuevo en su identidad.

## Pérdida

Sea r = M(k) − v el residuo de un token y un flujo, con d canales. La pérdida es

$$
\ell(r)=\sum_{c=1}^{d}\begin{cases}r_c^2 & |r_c|\le\delta_c\\ 2\delta_c|r_c|-\delta_c^2 & |r_c|>\delta_c\end{cases}
$$

que es dos veces la función de Huber de cada canal. El factor 2 conserva la escala del núcleo, cuya pérdida es ‖r‖² y no ½‖r‖². Así, dentro del umbral el gradiente respecto al residuo es 2r, el mismo que con L2, y fuera vale 2δ_c sign(r_c). Equivale a escribir con el residuo recortado:

$$
\nabla_W\ell = J_W^\top\,2\,\mathrm{clip}(r,-\delta,\delta),
$$

que es la regla de Yaad (ecuación 26 de MIRAS) con el gradiente de ℓ2 dentro del umbral y el de ℓ1 escalado por δ fuera, aplicada canal a canal. Las dos ramas valen δ² en |r_c| = δ_c, de modo que la pérdida es continua y su derivada también.

## Umbral

$$
\delta_c=\kappa_c\,\rho,\qquad \kappa_c=\operatorname{softplus}(w^\top x+b_c),\qquad \rho=\frac{\lVert v\rVert_2}{\sqrt d}\ \text{sin gradiente},
$$

con x el token observado, el mismo del que salen α, η y θ. MIRAS fija un umbral por canal que depende de la entrada, pero no su parametrización. Esta es propia: una dependencia de rango uno con la entrada, como la de η y θ, un desplazamiento por canal y la magnitud cuadrática media del valor como escala, para que κ se lea como un error relativo. Un umbral relativo a la escala del propio residuo recortaría siempre la misma fracción de canales, también en una sesión normal. Con la escala del valor, un canal queda fuera cuando su error supera κ veces la magnitud típica de lo que la memoria intenta recordar.

w empieza en cero y b en softplus⁻¹(κ₀), sin consumir ningún generador, así que al empezar el umbral es κ₀ρ en todos los canales. El valor declarado es κ₀ = 1. La escala no lleva gradiente porque así el gradiente interno es el de Huber con un umbral fijo y la memoria no puede escapar del recorte aumentando ‖v‖. Los parámetros del umbral solo reciben gradiente de los canales recortados, porque dentro del umbral la pérdida no depende de δ.

## Control

| Modelo | Cambio | Pregunta que contrasta |
| --- | --- | --- |
| `reference` | Sin el componente, pérdida L2 | Si recortar los residuos grandes mejora el núcleo |
| `huber_memory` | Huber por canal con κ₀ = 1 | |

R1 no empieza siendo la referencia, a diferencia de las piezas con puerta a cero. Una pérdida de Huber igual a L2 necesita un umbral infinito, y con él el umbral no recibe nunca gradiente. La paridad se comprueba de otra forma: sin declarar la pieza el cálculo no cambia, y con un umbral por encima de todos los residuos las salidas, los estados y los gradientes coinciden bit a bit con el núcleo.

## Correspondencia entre ecuaciones, código y pruebas

| Ecuación o regla | Código | Prueba |
| --- | --- | --- |
| Pérdida por canal y continuidad en el umbral | `huber_elements` | `test_each_channel_is_quadratic_inside_the_threshold_and_linear_outside` |
| Gradiente igual a L2 dentro y recortado fuera | `huber_elements` | `test_the_gradient_equals_l2_inside_the_threshold_and_clips_it_outside` |
| Umbral, softplus⁻¹(κ₀) y escala sin gradiente | `huber_threshold`, `HuberMemory.bias` | `test_the_threshold_is_the_softplus_times_the_value_rms_without_its_gradient` |
| Escritura con Jᵀ·2 clip(r) | `NeuralMemory.update` | `test_the_inner_gradient_is_the_jacobian_times_the_clipped_residual` |
| Paridad con un umbral por encima de los residuos | `NeuralMemory.update` | `test_a_threshold_above_every_residual_reproduces_the_l2_core_bit_for_bit`, `test_neutral_settings_reproduce_the_reference_predictor`, `test_neutral_settings_reproduce_the_reference_run` |
| El umbral aprende de los canales recortados | parámetros `huber_weight` y `huber_bias` | `test_a_threshold_below_some_residuals_changes_the_writes_and_learns` |
| Sorpresa por fila para #791 | lista `surprise` de `update` | `test_the_surprise_of_each_row_adds_up_to_the_huber_loss` |
| Parámetros sin consumir RNG e identidad | `NeuralMemory.__init__`, `MemoryConfig.identity` | `test_huber_adds_only_its_threshold_and_does_not_consume_the_generator`, `test_the_identity_records_the_loss_only_when_declared` |

## Paridad e identidad

Sin declarar la pieza, la identidad de la memoria, de MAC y del predictor no cambia, y la pérdida se calcula con las mismas operaciones que antes. Al declararla, la identidad de la memoria añade `associative_loss` con κ₀, la forma de la pérdida, la del umbral, la escala y la fuente, y la del predictor añade `memory_huber`. Los dos parámetros nuevos, `huber_weight` y `huber_bias`, son parámetros compartidos del predictor. El estado rápido no cambia, así que el checkpoint de la memoria tampoco.

Con el control de sorpresa de #791, la sorpresa de cada fila es la pérdida de Huber que realmente escribe.

El recorte global de gradientes del recorrido calcula la norma de todos los parámetros, también la de los dos nuevos. Aunque sus gradientes valgan cero, cambian el orden de la suma, así que los gradientes recortados pueden diferir en el último bit. La prueba del recorrido compara por eso los gradientes con una cota que no se alcanza.

## Desviaciones y límites

- La parametrización del umbral es propia. MIRAS lo proyecta desde la entrada con una proyección de rango bajo y aquí la dependencia es de rango uno, más un desplazamiento por canal.
- El umbral usa la escala del valor y no una escala absoluta. En una sesión en la que crecen a la vez el valor y el error relativo se mantiene, el canal no se recorta.
- κ₀ = 1 es una elección previa sin datos. Con las entradas sintéticas de la medida de coste y los pesos iniciales, el 95,5 % de los canales empieza fuera del umbral.
- La pérdida no reproduce los resultados de MIRAS. Su ablación en modelado de lenguaje no compara Yaad con una memoria L2 como la de Titans.

## Coste

Medido en CPU con `benchmarks/core_variants_cost.py` sobre el commit 60f044142, sin pasos de optimizador y con su [recibo](../../reports/engineering/core-variants-20261028/cost-cpu.json): 256 filas, 8 instantes, 10 repeticiones y 2 hilos en FP32 estricto, en un AMD Ryzen 9 8945HS. La máquina compartía la CPU con otras baterías, con una carga media de 12 a 20. Frente a la referencia, la mediana del forward con grafo es 1,035 veces la suya, la del backward 0,974 y la de la inferencia 0,929. U2, que no añade ninguna operación, quedó en 1,121 en el forward, así que diferencias de ese tamaño son ruido de la máquina compartida y R1 no tiene un coste medible a esta precisión. Añade 128 parámetros.

Con esas entradas y los pesos iniciales, el 95,5 % de los canales queda fuera del umbral al empezar. La LayerNorm residual da a la lectura de la memoria una escala cercana a uno, mayor que la del valor, y R1 empieza escribiendo con el residuo recortado en casi todos los canales. Es una escritura acotada, próxima a la de signo, que pasa a la de L2 en cada canal cuyo error baja de κρ. κ₀ = 1 se mantiene como valor declarado, porque elegirlo con entradas sintéticas no diría nada de los datos reales. El análisis deberá registrar el κ aprendido y la fracción recortada durante el ajuste.

La medida CUDA y otra con la máquina libre quedan pendientes.

## Comprobación

Pendiente de la verificación final.

## Referencias

- Behrouz, A., Razaviyayn, M., Zhong, P. y Mirrokni, V. (2026). It's all connected: A journey through test-time memorization, attentional bias, retention, and online optimization. En *The Fourteenth International Conference on Learning Representations* (pp. 131306–131333). https://proceedings.iclr.cc/paper_files/paper/2026/hash/d55f39791f04745b2e0c8abebf3dd5d7-Abstract-Conference.html
- Behrouz, A., Zhong, P. y Mirrokni, V. (2025). Titans: Learning to memorize at test time. En *Advances in Neural Information Processing Systems 38* (pp. 113506–113543). https://doi.org/10.52202/085713-3786
