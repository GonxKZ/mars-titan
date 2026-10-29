# Atención diferencial en MAC

`differential_attention_v1` es la pieza U1 de la segunda ola del estudio holográfico ([#782](https://github.com/GonxKZ/mars-titan/issues/782), [#786](https://github.com/GonxKZ/mars-titan/issues/786)). Resta a la atención de MAC un segundo mapa de atención ponderado por un escalar por cabeza que empieza en cero. El mapa está en `src/mars_titan/models/titans/differential_attention.py`, la resta en `TitansMAC.forward` y la declaración en [su configuración](../../configs/experiments/differential-attention-v1.json). Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y alcance

El Transformer diferencial calcula la atención como la diferencia de dos mapas softmax. La idea es que el ruido que comparten los dos mapas, la atención que se reparte sobre posiciones irrelevantes, se cancela al restarlos. U1 pregunta si esa resta ayuda a la atención de MAC. El control es la atención estándar del núcleo.

En la receta base, MAC atiende una secuencia corta: los cuatro tokens de la memoria persistente, la lectura de la memoria neuronal y el token de la decisión, seis posiciones en total. Con tan pocas posiciones hay poco ruido que cancelar, así que el resultado puede ser nulo y sigue siendo informativo. La pieza existe en las tres variantes de MAC. No existe en `transformer_direct`, que no tiene esta atención.

## Ecuación

Con P la secuencia [memoria persistente, lectura, segmento] y S las posiciones del segmento,

$$
o=\operatorname{MHA}(P)_S-W_O\Bigl[\lambda_h\,\operatorname{softmax}\Bigl(\frac{Q_{2,h}K_{2,h}^\top}{\sqrt{d_h}}+M\Bigr)V_h\Bigr]_{h},
$$

con Q₂ = P_S W_{Q₂}ᵀ y K₂ = P W_{K₂}ᵀ dos proyecciones nuevas, V_h los valores de la cabeza h de la propia MHA, W_O su proyección de salida, M la máscara causal de MAC y [·]_h la concatenación de las cabezas. El primer término es exactamente la atención de MAC, así que con la notación del artículo el resultado es W_O[(A₁ − λ_h A₂)V_h]_h. λ_h empieza en cero, de modo que al construir el modelo o coincide bit a bit con MAC, y λ recibe gradiente desde el primer paso. Q₂ y K₂ solo reciben gradiente cuando λ deja de ser cero.

Q₂ y K₂ se sortean con la cota de xavier del `in_proj` de la MHA, √(6/(4d)), con una semilla derivada de la de MAC y sin consumir el generador global. Solo se calculan las consultas del segmento, que son las únicas posiciones que MAC escribe y emite. El presupuesto de elementos de atención cuenta el segundo mapa.

## Control

| Modelo | Cambio | Pregunta que contrasta |
| --- | --- | --- |
| `reference` | Atención estándar de MAC | Si restar un segundo mapa mejora el núcleo |
| `differential_attention` | Segundo mapa con λ por cabeza desde cero | |

La variante añade 2d² + H parámetros, 8.196 con d = 64 y cuatro cabezas. El control no iguala esa capacidad. λ no tiene signo impuesto, así que el modelo también puede aprender a sumar los mapas, y el análisis deberá registrar el signo aprendido de cada cabeza.

## Correspondencia entre ecuaciones, código y pruebas

| Ecuación o regla | Código | Prueba |
| --- | --- | --- |
| Segundo mapa por cabeza, máscara y W_O | `DifferentialMap.forward` | `test_the_second_map_matches_its_equation_head_by_head` |
| Causalidad dentro del segmento | máscara de MAC en `DifferentialMap.forward` | `test_each_segment_position_only_reads_earlier_positions` |
| Resta sobre las posiciones del segmento | `TitansMAC.forward` | `test_mac_subtracts_the_second_map_from_the_attention_of_the_segment` |
| Paridad al empezar en las tres variantes | λ inicial nulo | `test_a_new_differential_map_starts_exactly_at_the_reference`, `test_neutral_settings_reproduce_the_reference_predictor`, `test_neutral_settings_reproduce_the_reference_run` |
| Semilla derivada y cota del `in_proj` | `DifferentialMap.__init__` | `test_the_projections_use_a_derived_seed_and_the_bound_of_the_in_proj` |
| Presupuesto con el segundo mapa | `TitansMAC.forward` | `test_the_attention_budget_counts_the_second_map` |
| Q₂ y K₂ aprenden con λ no nulo | `DifferentialMap.forward` | `test_lambda_away_from_zero_trains_the_second_projections` |

## Paridad e identidad

Sin declarar la pieza, la identidad de MAC conserva su huella anterior, que la prueba compara con la de la versión 1. Al declararla, la identidad de MAC añade `differential_attention` con la forma de cada mapa, la ponderación, la salida, las consultas, la inicialización y la fuente, y la del predictor la misma entrada. Los parámetros nuevos son `differential.query`, `differential.key` y `differential.weight`, compartidos del predictor.

Con el recorte global de gradientes del recorrido, la norma total incluye el gradiente de λ, así que los gradientes recortados de la referencia pueden cambiar en el coeficiente del recorte. La prueba del recorrido compara por eso los gradientes con una cota que no se alcanza.

## Desviaciones y límites

- λ es un escalar directo por cabeza que empieza en cero. El artículo usa λ = exp(λ_{q1}·λ_{k1}) − exp(λ_{q2}·λ_{k2}) + λ_init con λ_init = 0,8 − 0,6·e^{−0,3(l−1)}.
- No hay normalización por cabeza ni el factor (1 − λ_init) del artículo, y V tiene la dimensión de la cabeza en vez del doble. Las tres cambiarían la salida al empezar.
- El primer mapa es la MHA de MAC sin cambios, con sus propias Q y K de la dimensión de la cabeza.
- La secuencia de MAC es corta, de modo que el efecto de cancelar ruido puede ser pequeño.
- El control local C de CM-v1 no se combina con la pieza, porque su presupuesto está contrastado con el núcleo.
- No reproduce ningún resultado publicado.

## Coste

Medido en CPU con `benchmarks/core_variants_cost.py` sobre el commit 60f044142, sin pasos de optimizador y con su [recibo](../../reports/engineering/core-variants-20261028/cost-cpu.json): 256 filas, 8 instantes, 10 repeticiones y 2 hilos en FP32 estricto, en un AMD Ryzen 9 8945HS. La máquina compartía la CPU con otras baterías, con una carga media de 12 a 20. Frente a la referencia, la mediana del forward con grafo es 0,996 veces la suya, la del backward 1,006 y la de la inferencia 0,937. U2, que no añade ninguna operación, quedó en 1,121 en el forward, así que estas diferencias son ruido de la máquina compartida. Con seis posiciones, el segundo mapa es pequeño frente a la escritura de la memoria. Añade 8.196 parámetros.

La medida CUDA y otra con la máquina libre quedan pendientes.

## Comprobación

El [recibo](../../reports/engineering/core-variants-20261028/verification.json) reúne la verificación conjunta de R1, U1 y U2 sobre `912d543f8`, ya fusionado con develop.

- 883 casos dirigidos con cobertura de ramas, 881 en verde y 2 omitidos porque necesitan una protección que permita aprender. Incluyen la medida de lesiones, los cuatro módulos con dobles de la identidad y las pruebas de la señal madura de #791 que llegaron con develop.
- `differential_attention.py` cubre 33/33 sentencias y 2/2 ramas, con CRAP máximo 2. De las 228 líneas cambiadas frente a develop se ejecutan 212. Las 16 restantes son de la medida de coste: el calentamiento, la comprobación de un gradiente no finito y la orden de línea de comandos.
- La regresión de los módulos que importan el código cambiado, sobre `60f044142`, ejecutó 1.925 casos de `tests/training` y 6.651 del resto. Nueve de sus diez fallos eran dobles de configuración sin el campo `mature_reliability` de #791 y el décimo era el plazo de la medida de lesiones de #803. Los dos arreglos llegaron con develop y esas pruebas pasan en la verificación fusionada. De las omisiones, 125 son de la protección del aprendizaje, una espera a que se levante y 39 necesitan binarios nativos de simulación y de PPO sin compilar.
- Las 87 mutaciones dirigidas, 13 de ellas en `differential_attention.py`, se detectan con las pruebas en verde antes y después de cada una. Las cuatro de la medida de coste se repitieron con el directorio de Git declarado, porque la copia aislada no es un repositorio y la medida registra su commit.
- El CRAP de `FinancialConfig.__post_init__` (2.010) es una deuda previa recogida en #801.

No hay entrenamiento ni pasos de optimizador en ninguna de estas comprobaciones.

## Referencias

- Behrouz, A., Zhong, P. y Mirrokni, V. (2025). Titans: Learning to memorize at test time. En *Advances in Neural Information Processing Systems 38* (pp. 113506–113543). https://doi.org/10.52202/085713-3786
- Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G. y Wei, F. (2025). Differential Transformer. En *The Thirteenth International Conference on Learning Representations*. https://arxiv.org/abs/2410.05258
