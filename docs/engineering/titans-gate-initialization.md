# Inicialización de las puertas de la memoria neuronal

La memoria neuronal de [Titans-MAC](titans-memory-core.md) calcula sus tasas `α`, `η` y `θ` con proyecciones lineales de la observación seguidas de una sigmoide. En la identidad v1 esas proyecciones no tienen bias. Con los parámetros iniciales, `α` vale cerca de 0,5 y la memoria rápida pierde la mitad de su contenido en cada observación. El [entrenador cronológico](titans-chronological-trainer.md) midió la consecuencia en su fixture: la norma de cada capa pasó de 5,6 a 1,4·10⁻¹⁰ en 36 observaciones y el gradiente de la cabeza bajó de 10⁻⁷ a 10⁻¹³ en cuatro tramos.

Este documento recoge qué fijan las fuentes, el análisis de retención, las alternativas comparadas y la opción implementada. `MemoryConfig(gate_bias=GateBias(...))` crea una identidad nueva del núcleo. El valor por defecto `gate_bias=None` conserva la identidad, los pesos y las salidas de v1 byte a byte. No se ha ejecutado ningún entrenamiento ni paso de optimizador.

## Qué fijan las fuentes

Las actas de [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf) definen la actualización en sus ecuaciones 1 a 3:

$$S_t=\eta_t S_{t-1}-\theta_t\nabla\ell(M_{t-1};x_t),\qquad M_t=(1-\alpha_t)M_{t-1}+S_t.$$

El texto exige que `η_t` y `θ_t` dependan de la entrada y declara `α_t ∈ [0, 1]^{d_in}`. Con `α_t → 0` la memoria se actualiza sin olvidar y con `α_t → 1` se vacía. El [preprint v1](https://arxiv.org/abs/2501.00663v1) añade en la sección 3.2 que sus experimentos usan tasas por token y que hacerlas constantes por chunk es una simplificación posible. Ninguna de las dos versiones fija la forma funcional de las tasas, la existencia de bias ni su inicialización. El apéndice E.1 solo describe el optimizador externo. Los PDF consultados son los del [recibo de fuentes](../../reports/research/titans-mac-source-audit-20261008.json), con hashes `9b0f0c4a…` y `a65e4a7d…`.

La sección 3.3 de las actas describe además una memoria con residual y LayerNorm, `M(x) = x + LN(W₁σ(W₂x))`. La adaptación compacta del proyecto no tiene residual, LayerNorm ni bias en la memoria, y esa desviación ya estaba documentada. Es relevante aquí porque, sin residual, `M = 0` produce lecturas nulas.

Como contraste se revisó la implementación no oficial [`lucidrains/titans-pytorch`](https://github.com/lucidrains/titans-pytorch/blob/1d40c445fa1794fb28582c721786482b5b683e47/titans_pytorch/neural_memory.py), licencia MIT, commit `1d40c445fa1794fb28582c721786482b5b683e47` del 13 de julio de 2026 (SHA-256 del archivo `dae13545…`). No se ha copiado ni ejecutado código. Sus decisiones son estas:

- Las tres tasas usan `nn.Linear` con bias y sigmoide. `θ` se escala con `max_lr`, cuyo valor por defecto es 1,0 y 0,1 en su ejemplo de entrenamiento. El olvido entra como `1 − α` en el scan.
- Con la inicialización por defecto de PyTorch el bias es pequeño, de modo que `α` y `η` también empiezan cerca de 0,5.
- Ofrece `init_decay_bias`, `init_momentum_bias` e `init_adaptive_step_bias`. Esas opciones ponen a cero los pesos de la proyección y fijan el bias, así que eliminan la dependencia de la entrada al inicio.
- Su memoria por defecto incluye LayerNorm y residual, por lo que la lectura no se anula aunque los pesos rápidos decaigan.

Hay precedentes publicados de fijar la escala temporal de una puerta con su bias. [Mamba](https://arxiv.org/abs/2312.00752v2) inicializa el parámetro Δ, que describe como un término de bias, con `τ_Δ⁻¹(Uniform([0.001, 0.1]))` (sección 3.6). [xLSTM](https://papers.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html) señala entre sus limitaciones que la inicialización de la puerta de olvido debe elegirse con cuidado y compara varias inicializaciones del bias en su tabla 6. Son arquitecturas distintas y no trasladan un valor concreto a esta memoria.

En resumen, la dependencia de la entrada y los rangos proceden del artículo. La sigmoide, la escala `theta_max` y la ausencia de bias de v1 fueron decisiones de esta adaptación. El bias y sus valores iniciales son una decisión nueva de este proyecto.

## Retención con olvido constante

Si `α` fuese constante, la contribución de una escritura pasada se multiplicaría por `(1 − α)` en cada observación. La semivida en observaciones es

$$h=\frac{\log 0{,}5}{\log(1-\alpha)},\qquad \alpha=1-2^{-1/h}.$$

| `α` | Semivida | Factor tras 8 observaciones |
| --- | --- | --- |
| 0,5 | 1 | 0,0039 |
| 0,1 | 6,6 | 0,43 |
| 0,01077 | 64 | 0,917 |
| 0,002704 | 256 | 0,979 |

En v1, los logits de `α` son `A x` con `A ~ U(±1/√D)`. En el adaptador financiero con parámetros iniciales, la salida de atención que escribe en memoria mide entre 0,10 y 0,16 y la desviación de los logits es de 0,01. `α` queda entre 0,49 y 0,51, es decir, una semivida de una observación. Tras 36 observaciones el factor es 0,5³⁶ = 1,5·10⁻¹¹, coherente con la caída de 5,6 a 1,4·10⁻¹⁰ medida en la PR #377. El BPTT truncado de 8 decisiones del entrenador atraviesa además un factor 0,0039 en cada tramo.

## Estabilidad del origen sin bias ni residual

La memoria de dos capas es `M(k) = W₂ GELU(W₁ k)`. Como `GELU(0) = 0` y no hay bias, `W₁ = W₂ = 0` es un punto fijo con lecturas nulas. Cerca de él, `GELU(z) ≈ z/2` y `GELU'(0) = 1/2`. Para la pérdida sumada `ℓ = ‖M(k) − v‖²`, con claves unitarias, la derivación propia da a primer orden

$$\nabla_{W_2}\ell\approx -v k^\top W_1^\top,\qquad \nabla_{W_1}\ell\approx -W_2^\top v k^\top.$$

Sea `C = E[v kᵀ]` la asociación media y `σ` su mayor valor singular. Con tasas constantes, el modo de `(W₁, W₂)` alineado con el par singular evoluciona junto con su momentum según la matriz `[[η, θσ], [η, 1 − α + θσ]]`. Su polinomio característico cumple `p(1) = α(1 − η) − θσ`. El origen deja de atraer la dinámica linealizada cuando

$$\rho=\frac{\theta\,\sigma}{(1-\eta)\,\alpha}>1.$$

La condición se obtiene para tasas constantes y una asociación promediada, así que es una aproximación. No describe cada flujo con tasas por fila variables ni garantiza un equilibrio concreto. Con claves unitarias, `σ ≤ E‖v‖`.

Con los valores iniciales de v1 (`α ≈ η ≈ 0,5`, `θ ≈ 0,05`) haría falta `σ > 5`. En el adaptador financiero inicial `‖v‖` vale entre 0,06 y 0,09, así que el colapso hacia lecturas nulas es la dinámica esperada. Una memoria lineal de una capa no tiene ese atractor, porque su gradiente en `W = 0` vale `−2 v kᵀ`. Para un par fijo con `‖k‖ = 1` alcanza `W*k = c/(α + c) · v`, con `c = 2θ/(1 − η)`. Esa fracción vale 0,29 con las tasas iniciales de v1 y 0,99 con la opción elegida.

## Alternativas comparadas

| Opción | Formulación | Ventajas | Inconvenientes | Decisión |
| --- | --- | --- | --- | --- |
| A. Bias inicializado | `α = σ(A x + b_α)`, `η = σ(e·x + b_η)`, `θ = θ_max σ(t·x + b_θ)` | Conserva el rango (0, 1) del artículo. Los pesos y la dependencia de la entrada son los de v1. Un bias aprendible desplaza la tasa media | Añade 2 + D parámetros y obliga a una identidad nueva | Elegida |
| B. Escala acotada | `α = α_max σ(A x)`, con `α_max = 2α₀` | Mismo `state_dict` que v1 | `α ≤ α_max` impide vaciar la memoria y contradice `α → 1` del artículo. La sensibilidad relativa a la entrada se reduce a la mitad | Descartada |
| C. Pesos a cero y bias | Opción `init_*_bias` de la implementación no oficial | Tasas iniciales exactas | Sin dependencia de la entrada al inicio. Cambiar la entrada no cambia `α_t` | Descartada |
| D. Residual y LayerNorm | Memoria de la sección 3.3 de las actas | Las lecturas no se anulan | Cambia la arquitectura de la memoria y no la tasa de olvido | Fuera de alcance, otra identidad |
| E. Semividas por fila | Rango de `b_α` por fila, al estilo de Mamba | Varias escalas temporales | Las filas rápidas cumplen peor `ρ > 1` y añade hiperparámetros sin evidencia | No implementada |

Con la misma tasa media inicial, A y B producen trayectorias casi iguales en los fixtures (tabla de medidas). La diferencia está en lo que el ajuste externo puede alcanzar después. En A, la entrada desplaza el logit igual que en v1 y la sensibilidad relativa es `∂ log α/∂z = 1 − α ≈ 1`. En B es `1 − σ(z) ≈ 0,5`. La derivada de `α` respecto a su logit lineal es `α_max σ(1 − σ) ≈ 0,00135` en B, frente a `α(1 − α) ≈ 0,0027` en A.

## Opción implementada

```python
from mars_titan.models.titans import GateBias, MemoryConfig

memory = MemoryConfig(dim=64, depth=2, gate_bias=GateBias(alpha_half_life=256, eta=0.5, theta=0.05))
```

`GateBias` declara las tasas iniciales para entrada nula. Sus bias son

$$b_\alpha=\operatorname{logit}(1-2^{-1/h}),\qquad b_\eta=\operatorname{logit}(\eta_0),\qquad b_\theta=\operatorname{logit}(\theta_0/\theta_{max}).$$

`b_α` se calcula como `log(−expm1(−ln2/h)) + ln2/h` para evitar cancelación. Se exige `1 ≤ h ≤ 10⁶`, `0 < η₀ < 1` y `0 < θ₀ < theta_max`. Los bias son constantes y no consumen RNG, de modo que los pesos de las proyecciones coinciden con los de v1 y la única diferencia inicial son los bias. Se registran como parámetros compartidos, dentro del rol `shared` del entrenador.

La identidad de la memoria añade `gate_bias` con los tres valores y `init = constant_logit_bias_v1_weight_draws`. Con `None` la clave no aparece y la huella coincide con la de v1. Los contratos v1 y con bias rechazan los parámetros y estados del otro.

Los valores propuestos son hiperparámetros declarados, no óptimos demostrados:

- `h = 256` observaciones, algo más de un año bursátil de 252 sesiones y cuatro veces la ventana de 64 sesiones que ya atiende el codificador de precios. `b_α = −5,9103`.
- `η₀ = 0,5` y `θ₀ = 0,05 = theta_max/2`. Dan `b_η = b_θ = 0`, así que al inicio `η` y `θ` coinciden con los de v1 para cada entrada y la intervención inicial se limita a `α`.
- `h = 64` no cumple `ρ > 1` con la escala del token fusionado y colapsa en los fixtures. `h = 1024` y `η₀ = 0,9` también evitan el colapso, con olvido más lento o un momentum mayor. Quedan como contrastes posibles, sin evidencia predictiva para preferirlos.

## Correspondencia ecuación, código y prueba

Las pruebas están en [`test_gate_bias.py`](../../tests/models/titans/test_gate_bias.py), salvo que se indique otro archivo.

| Ecuación o contrato | Código | Prueba |
| --- | --- | --- |
| `α_t = σ(A x_t + b_α) ∈ (0, 1)^D`, `b_α = logit(1 − 2^{−1/h})` | `GateBias.logits`, `NeuralMemory.__init__` | `test_logits_reproduce_the_declared_rates_and_half_life`, `test_zero_input_starts_at_declared_rates_and_longer_half_life_retains_more` |
| `η_t = σ(e·x_t + b_η)`, `θ_t = θ_max σ(t·x_t + b_θ)` | Mismas funciones | Las dos anteriores y `test_rates_stay_inside_their_ranges_and_still_depend_on_the_input` |
| Dependencia de la entrada | Pesos de v1 sin cambios | `test_rates_stay_inside_their_ranges_and_still_depend_on_the_input` |
| Ecuación 3 con las tasas nuevas | `NeuralMemory.update`, sin cambios | `test_linear_update_with_gate_bias_matches_the_manual_equation` |
| Derivadas respecto a bias, entrada, pesos y momentum | `update(..., differentiable=True)` | `test_two_layer_update_gradcheck_covers_gate_biases_inputs_and_state`, diferencias finitas en FP64 |
| Identidad v1 intacta | `MemoryConfig.identity` | `test_default_configuration_keeps_the_v1_identity_and_parameters_byte_for_byte` |
| Identidad nueva con los pesos de v1 | `MemoryConfig.identity`, `NeuralMemory.__init__` | `test_gate_bias_is_a_new_identity_that_only_adds_constant_biases_to_v1_draws`, `test_v1_and_gate_bias_contracts_reject_each_other_parameters_and_states` |
| Retención en 64, 256 y 1024 observaciones | Recorrido MAC completo | `test_gate_bias_keeps_fast_memory_norms_inside_the_declared_retention`, `test_gate_bias_keeps_outer_gradients_finite_and_far_from_v1_collapse` |
| Inicialización en CPU sin consumir RNG global | `NeuralMemory.__init__` | [`test_initialization_device.py`](../../tests/models/titans/test_initialization_device.py) |
| Validación | `GateBias.__post_init__`, `MemoryConfig.__post_init__` | `test_invalid_gate_bias_values_are_rejected`, `test_memory_configuration_rejects_theta_outside_theta_max_and_foreign_types` |
| Adaptador financiero y recetas | `FinancialConfig`, `FinancialPredictor.__init__`, recetas cronológicas | `test_financial_default_keeps_the_v1_contract_and_parameters`, `test_financial_controls_build_the_declared_gate_bias_and_pair_from_mac_online`, `test_financial_configuration_rejects_incomplete_or_invalid_gate_bias`, `test_chronological_recipes_declare_the_gate_bias_decided_in_issue_27` |

Las cotas de las pruebas largas son técnicas y propias de su fixture. Con `D = 16`, dos flujos y tokens de norma cercana a 1, la norma de cada capa debe quedar entre la mitad del decaimiento puro `2^{−T/h}` y su valor inicial. Los gradientes respecto a la cabeza, las tres proyecciones de las puertas, claves, valores y consulta deben ser finitos, mayores que 10⁻¹² y no caer más de cuatro órdenes entre 64 y 1024 observaciones. En v1 esos gradientes quedan más de veinte órdenes por debajo en todas las ventanas, y ya a las 64 observaciones son del orden de 10⁻³⁵.

## Medidas en fixtures

[`benchmarks/titans_gate_retention.py`](../../benchmarks/titans_gate_retention.py) recorre MAC con `D = 64`, dos capas, cuatro cabezas, cuatro tokens persistentes, cuatro flujos y un token por observación durante 4096 observaciones. Usa tres flujos de entrada: ruido con norma cercana a 1, como el token fusionado inicial, ruido de varianza unidad y cuatro prototipos alternos con ruido. Solo ejecuta pasos hacia delante, escrituras asociativas y `autograd.grad` de la suma de `head(salida)` en ventanas de 8 observaciones. El [recibo](../../reports/engineering/titans-gate-retention-20261009.json) conserva normas, tasas, asociación media, gradientes y versiones. La ejecución tardó 411 s en CPU con dos hilos.

Norma de `W₁` tras 4096 observaciones en FP64, con `ρ` medido con la asociación media entre paréntesis. Todas parten de 8,0.

| Variante | Token fusionado | Varianza unidad | Prototipos recurrentes |
| --- | --- | --- | --- |
| v1 | 0 (0,013) | 0 (0,007) | 0 (0,026) |
| A, `h = 64`, `η₀ = 0,5` | 1,3·10⁻⁸ (0,61) | 5,6·10⁻¹⁴ (0,32) | 0,59 (1,19) |
| A, `h = 256`, `η₀ = 0,5`, elegida | 0,31 (2,43) | 0,48 (1,28) | 1,35 (4,75) |
| A, `h = 256`, `η₀ = 0,9` | 0,37 (12,2) | 2,45 (6,4) | 1,43 (23,6) |
| A, `h = 1024`, `η₀ = 0,5` | 0,61 (9,7) | 2,37 (5,1) | 1,67 (19,0) |
| B, `α_max = 2α₀` | 0,31 (2,44) | 0,48 (1,28) | 1,35 (4,77) |

En los 18 casos, `ρ < 1` coincide con normas que tienden a cero y `ρ > 1` con normas positivas a las 4096 observaciones. El cociente por token, que usa `E‖v‖` en lugar de `σ`, no basta para predecirlo. Con `h = 64` y varianza unidad vale 3,2 y aun así la memoria colapsa, porque las direcciones aleatorias se cancelan en la asociación media. La opción elegida con varianza unidad está cerca del umbral y su norma todavía baja de 0,57 a 0,48 entre 2048 y 4096 observaciones.

Gradientes del funcional lineal con el token fusionado, en FP64, para ventanas que terminan en 64, 1024 y 4096 observaciones:

| Variante | Cabeza | Pesos de la proyección de `α` | Proyección de consulta |
| --- | --- | --- | --- |
| v1 | 7,9·10⁻³⁵ / 0 / 0 | 4,6·10⁻³⁷ / 0 / 0 | 2,1·10⁻⁶⁷ / 0 / 0 |
| A, `h = 256`, elegida | 0,030 / 4,2·10⁻⁴ / 4,7·10⁻⁴ | 6,0·10⁻⁶ / 6,3·10⁻⁸ / 5,5·10⁻⁸ | 1,4·10⁻³ / 3,6·10⁻⁷ / 2,3·10⁻⁷ |
| B, `α_max = 2α₀` | 0,030 / 4,3·10⁻⁴ / 4,7·10⁻⁴ | 3,0·10⁻⁶ / 3,2·10⁻⁸ / 2,8·10⁻⁸ | 1,4·10⁻³ / 3,6·10⁻⁷ / 2,3·10⁻⁷ |

Con la opción elegida los gradientes se estabilizan a partir de unas 1000 observaciones, cuando la memoria alcanza su nivel de equilibrio. En v1 son nulos en coma flotante desde la ventana de 1024. B reproduce las normas de A con la misma tasa media, pero la derivada respecto a la proyección de `α` es la mitad, como anticipa el análisis.

En FP32 la opción elegida termina con normas de 0,27, 0,50 y 1,31 en los tres flujos. v1 llega a cero exacto antes de 256 observaciones y sus gradientes son cero desde la primera ventana medida. Los pesos iniciales en FP32 proceden de otro sorteo, por lo que estas cifras no se comparan dígito a dígito con FP64.

## Integración en el adaptador y las recetas

La issue #27 registra la decisión de usar la identidad corregida en la campaña sobre la edición desde 2000 y conservar v1 solo como referencia. `FinancialConfig` añade `gate_bias`, con `None` por defecto. Acepta un `GateBias` o un diccionario JSON con exactamente `alpha_half_life`, `eta` y `theta`, y comprueba `θ₀ < theta_max` al construirse. Los controles con MAC pasan el valor a `MemoryConfig`. La identidad registra `memory_gate_bias` en las cuatro variantes, también en `transformer_direct`, porque `copy_paired_parameters` compara la configuración sin variante ni semilla. Solo `mac_online` usa las tasas, pero `mac_frozen` y `mac_disabled` reciben los mismos bias por la copia emparejada.

Las recetas [`chronological-training.json`](../../configs/titans/chronological-training.json) y [`chronological-training-quantile.json`](../../configs/titans/chronological-training-quantile.json) declaran `{"alpha_half_life": 256.0, "eta": 0.5, "theta": 0.05}`. Siguen en estado `propuesta_sin_ejecutar`. Las huellas de `FinancialConfig` sin `gate_bias` coinciden con las de develop, según las pruebas y el [recibo de verificación](../../reports/engineering/titans-gate-initialization-verification-20261009.json).

## Límites

La salida de MAC es `y ⊙ M(y)`. En los fixtures su escala absoluta es pequeña, entre 10⁻⁵ y 10⁻² según la entrada, porque multiplica dos magnitudes pequeñas. El bias evita el colapso de la memoria, pero no cambia esa escala, que pertenece a la puerta de salida y al ajuste externo.

Estas medidas proceden de fixtures aleatorios en CPU. No predicen el comportamiento con datos reales, no seleccionan hiperparámetros por validación y no acreditan una mejora predictiva. La comprobación CUDA queda pendiente:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 uv run --no-sync python benchmarks/titans_gate_retention.py \
  --device cuda:0 --output reports/engineering/titans-gate-retention-cuda-<fecha>.json
```
