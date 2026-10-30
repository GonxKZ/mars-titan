# Semividas geométricas en el olvido de Titans

`scale_invariant_time_v1` es la pieza U2 de la segunda ola del estudio holográfico ([#782](https://github.com/GonxKZ/mars-titan/issues/782), [#786](https://github.com/GonxKZ/mars-titan/issues/786)). Reparte en escala geométrica las semividas iniciales del olvido α de la memoria de Titans, que en el núcleo son iguales para todas las filas. La regla está en `src/mars_titan/models/titans/scale_invariant_time.py`, el bias en `NeuralMemory.__init__` y la declaración en [su configuración](../../configs/experiments/scale-invariant-time-v1.json). Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y alcance

SITH y DeepSITH representan la historia con integradores cuyas constantes de tiempo están espaciadas en escala logarítmica. Así, la representación cambia poco al reescalar el tiempo y conviven trazas de una semana y de varios años. En la memoria de Titans, α ya es un vector con una tasa por fila de salida de cada matriz, pero `GateBias` da a todas las filas la misma semivida inicial, 256 sesiones en la receta base. U2 pregunta si repartir esas semividas alrededor del mismo centro ayuda a la memoria. El control son las constantes iguales del núcleo.

La pieza solo existe con `mac_online` y necesita `gate_bias`, porque reparte su semivida. η, θ y la memoria compartida de PT4 no cambian.

## Regla

Con h la semivida de `gate_bias`, n filas y o octavas,

$$
h_i=h\cdot 2^{\,o\left(\frac{2i}{n-1}-1\right)},\qquad i=0,\dots,n-1,
$$

que va de h/2^o a h·2^o con media geométrica h. Con o = 6 y h = 256, las 64 filas van de 4 a 16.384 sesiones, con un 14 % entre filas consecutivas. El bias de la fila i es el logit de α_i = 1 − 2^{−1/h_i}, calculado con la misma función que el control (`GateBias.logits`), de modo que con entrada nula (1 − α_i)^{h_i} = 1/2. Con la caja de PT1 el bias invierte su función α = α_lo + (1 − α_lo)σ(z), y una semivida que exige menos olvido que el suelo se rechaza al declarar. Con el suelo de 1/500 la semivida máxima es de unas 346 sesiones, así que la receta declarada solo es compatible con la memoria sin caja.

Como en `GateBias`, el bias es el valor inicial de un parámetro aprendido y la dependencia de la entrada no cambia. Los bias no consumen ningún generador.

## Control

| Modelo | Cambio | Pregunta que contrasta |
| --- | --- | --- |
| `reference` | Semivida de 256 sesiones en todas las filas | Si repartir las semividas en escala geométrica mejora el núcleo |
| `scale_invariant_time` | Seis octavas a cada lado, de 4 a 16.384 sesiones | |

U2 tampoco empieza siendo la referencia, porque su hipótesis es otra inicialización. Con o = 0 todas las filas reciben la semivida del control y el cálculo coincide bit a bit con él, que es la prueba de paridad.

## Correspondencia entre ecuaciones, código y pruebas

| Ecuación o regla | Código | Prueba |
| --- | --- | --- |
| Semividas geométricas con media h | `ScaleInvariantTime.half_lives` | `test_half_lives_are_geometric_around_the_declared_half_life` |
| Bias por fila con (1 − α_i)^{h_i} = 1/2 | `ScaleInvariantTime.alpha_logits`, `NeuralMemory.__init__` | `test_each_row_starts_with_the_forgetting_of_its_half_life` |
| Caja de PT1 | `GateBias.logits` con la estabilidad | `test_the_box_of_pt1_rejects_half_lives_that_need_less_forgetting_than_its_floor` |
| Paridad con o = 0 | la misma función del control | `test_zero_octaves_reproduce_the_reference_bit_for_bit`, `test_neutral_settings_reproduce_the_reference_predictor`, `test_neutral_settings_reproduce_the_reference_run` |
| Solo cambia el bias de α y no se consume RNG | `NeuralMemory.__init__` | `test_only_the_alpha_bias_changes_and_the_generator_is_not_consumed` |
| El bias aprende | parámetro `alpha_projection.bias` | `test_spread_half_lives_change_the_memory_and_their_biases_learn` |

## Paridad e identidad

Sin declarar la pieza, la identidad y el cálculo no cambian. Al declararla, la identidad de la memoria añade `alpha_time_scales` con las octavas, el centro, el espaciado y la fuente, y la del predictor añade `memory_time_scales`. No hay parámetros nuevos, solo otro valor inicial del bias de α.

## Desviaciones y límites

- DeepSITH fija sus constantes de tiempo e invierte la transformada de Laplace para obtener una línea temporal comprimida. Aquí las semividas son el valor inicial de un parámetro aprendido y no hay inversión, porque la memoria de Titans no es una transformada de Laplace.
- El estado rápido se reinicia en cada recorrido. En un recorrido de validación de unos pocos cientos de sesiones, las filas con semividas mayores se comportan casi como si no olvidaran.
- Las filas con olvido muy lento acumulan más escrituras. La LayerNorm residual de la receta acota la lectura, pero no el tamaño de los pesos rápidos.
- El orden de las semividas sigue el índice de las filas. Como los pesos iniciales de cada fila se sortean de forma independiente, el orden no favorece a ninguna.
- No reproduce ningún resultado publicado.

## Coste

U2 no añade parámetros ni operaciones, solo cambia el valor inicial del bias de α. Medido en CPU con `benchmarks/core_variants_cost.py` sobre el commit 60f044142, sin pasos de optimizador y con su [recibo](../../reports/engineering/core-variants-20261028/cost-cpu.json): 256 filas, 8 instantes, 10 repeticiones y 2 hilos en FP32 estricto, en un AMD Ryzen 9 8945HS. La máquina compartía la CPU con otras baterías, con una carga media de 12 a 20. Frente a la referencia, su mediana del forward con grafo es 1,121 veces la suya, la del backward 1,015 y la de la inferencia 0,999. Como el cálculo es el mismo, ese 12 % mide el ruido de la máquina compartida y sirve de cota para leer las otras dos piezas.

## Comprobación

El [recibo](../../reports/engineering/core-variants-20261028/verification.json) reúne la verificación conjunta de R1, U1 y U2 sobre `912d543f8`, ya fusionado con develop.

- 883 casos dirigidos con cobertura de ramas, 881 en verde y 2 omitidos porque necesitan una protección que permita aprender. Incluyen la medida de lesiones, los cuatro módulos con dobles de la identidad y las pruebas de la señal madura de #791 que llegaron con develop.
- `scale_invariant_time.py` cubre 19/19 sentencias y 4/4 ramas, con CRAP máximo 4. De las 228 líneas cambiadas frente a develop se ejecutan 212. Las 16 restantes son de la medida de coste: el calentamiento, la comprobación de un gradiente no finito y la orden de línea de comandos. Después, #813 añadió las pruebas de esas tres partes, y su [recibo](../../reports/engineering/core-variants-cost-20261030/verification.json) deja la medida cubierta salvo la llamada final de `__main__`, con sus nueve mutaciones dirigidas detectadas.
- La regresión de los módulos que importan el código cambiado, sobre `60f044142`, ejecutó 1.925 casos de `tests/training` y 6.651 del resto. Nueve de sus diez fallos eran dobles de configuración sin el campo `mature_reliability` de #791 y el décimo era el plazo de la medida de lesiones de #803. Los dos arreglos llegaron con develop y esas pruebas pasan en la verificación fusionada. De las omisiones, 125 son de la protección del aprendizaje, una espera a que se levante y 39 necesitan binarios nativos de simulación y de PPO sin compilar.
- Las 87 mutaciones dirigidas, 11 de ellas en `scale_invariant_time.py`, se detectan con las pruebas en verde antes y después de cada una. Las cuatro de la medida de coste se repitieron con el directorio de Git declarado, porque la copia aislada no es un repositorio y la medida registra su commit.
- El CRAP de `FinancialConfig.__post_init__` (2.010) es una deuda previa recogida en #801.

No hay entrenamiento ni pasos de optimizador en ninguna de estas comprobaciones.

## Referencias

- Jacques, B. G., Tiganj, Z., Howard, M. W. y Sederberg, P. B. (2021). DeepSITH: Efficient learning via decomposition of what and when across time scales. En *Advances in Neural Information Processing Systems 34* (pp. 27530–27541). https://proceedings.neurips.cc/paper_files/paper/2021/hash/e7dfca01f394755c11f853602cb2608a-Abstract.html
- Behrouz, A., Zhong, P. y Mirrokni, V. (2025). Titans: Learning to memorize at test time. En *Advances in Neural Information Processing Systems 38* (pp. 113506–113543). https://doi.org/10.52202/085713-3786
