# Escala de Laplace con β-NLL

Este documento describe F1 del estudio de variantes de [#742](https://github.com/GonxKZ/mars-titan/issues/742), que se implementa en [#749](https://github.com/GonxKZ/mars-titan/issues/749). `laplace_scale_head_v1` es un componente opcional de `FinancialConfig` (`laplace_scale_head`) que añade a la cabeza escalar una escala por fila y cambia la pérdida del ajuste por una NLL de Laplace ponderada. Desde la integración con F4 es un objetivo de salida del marco común de `output_objectives`, igual que el objetivo de IC por sesión. Los dos sustituyen la pérdida MAE de la cabeza escalar y no se pueden declarar a la vez.

Está integrada en el predictor, en el recorrido cronológico y en la escritura de predicciones de la ventana. Se ha comprobado con pruebas en CPU sobre fixtures. La comprobación CUDA está pendiente porque la GPU está ocupada. No se ha entrenado ni evaluado con datos financieros y nada de lo que sigue es una mejora observada.

## Motivación

La [fase 0](../../reports/research/scale-diagnostics-20261023/README.md) mostró que las predicciones de los pilotos de octubre eran mucho más amplias de lo que justificaba su ordenación. A1 (`scale_shrink_v1`) corrige esa amplitud con un factor común y A2 (`neutral_abstention_v1`) se abstiene cuando el intervalo calibrado contiene el cero. Las dos necesitan una medida de incertidumbre por fila que la cabeza escalar no tiene. F1 la aprende con las mismas etiquetas maduras y sin cambiar el gradiente de la mediana.

## Ecuaciones

Sea h el estado codificado que ya recibe la cabeza escalar y μ su salida. La escala es

$$b=\operatorname{softplus}(w^\top h+c)+b_{\min}.$$

El ajuste sustituye el error absoluto por la β-NLL de [Seitzer et al. (2022)](https://arxiv.org/abs/2203.09168) con β = 1, aplicada a la Laplace:

$$\mathcal L=\frac1N\sum_i \operatorname{sg}(b_i)\left(\frac{\lvert y_i-\mu_i\rvert}{b_i}+\log b_i\right).$$

sg detiene el gradiente. Por fila, antes de dividir por N,

$$\frac{\partial\mathcal L}{\partial\mu}=\operatorname{sign}(\mu-y),\qquad \frac{\partial\mathcal L}{\partial b}=\operatorname{sg}(b)\,\frac1b\left(1-\frac{\lvert y-\mu\rvert}{b}\right).$$

La mediana recibe exactamente el gradiente del error absoluto del control. La escala anula su gradiente medio en b = E|y − μ|, la desviación absoluta media, que es el estimador de máxima verosimilitud de la escala de una Laplace. Sin el factor sg(b), el término |y − μ|/b daría a las filas con escala pequeña un peso 1/b en el gradiente de μ. Ese es el efecto que la β-NLL corrige.

`laplace_beta_nll` escribe la pérdida como |y − μ|·(sg(b)/b) + sg(b)·log b. Tiene el mismo valor, porque sg(b)/b vale exactamente uno en coma flotante, y su derivada respecto a μ es exactamente sign(μ − y)/N, igual que `l1_loss`, sin el redondeo de multiplicar y dividir por b. El valor de la pérdida que registran los contadores del recorrido es esta β-NLL. La selección sigue usando el MAE por sesión de la predicción emitida, que es μ.

Los pesos w empiezan en cero y c en softplus⁻¹(`initial_scale` − b_min), de modo que b vale `initial_scale` en todas las filas. Con w = 0 el gradiente de b no llega al tronco: en la inicialización, el gradiente de todos los parámetros de la referencia coincide bit a bit con el del control L1. La diferencia entre la variante y el control aparece cuando w deja de ser cero y el gradiente de la escala llega al tronco compartido. Las pruebas comprueban los dos casos.

La declaración usa b_min = 10⁻⁶ y `initial_scale` = 0,01 en unidades del residuo diario. Son valores declarados, no ajustados con datos. El suelo mantiene finitos 1/b y log b en FP32. Con softplus, el gradiente respecto al logit z es (1 − |y − μ|/b)·σ(z). Cuando b se acerca al suelo, σ(z) ≈ b − b_min y el término negativo queda acotado por |y − μ|.

## Cuantiles emitidos para A1 y A2

La predicción puntual sigue siendo μ. Con `rows` (inferencia de una partición), el recorrido escribe además los cuantiles de la Laplace(μ, b) en los cinco niveles de `quantile_head_v1`:

$$q_\tau=\mu+b\log(2\tau)\ \ (\tau\le 1/2),\qquad q_\tau=\mu-b\log(2(1-\tau))\ \ (\tau>1/2).$$

q₀,₅ es exactamente μ y los cinco valores son no decrecientes en coma flotante. `run_titans_window` los guarda en las columnas `quantile_*` del Parquet de predicciones, igual que la cabeza de cuantiles. A2 puede calibrarlos con CQR por mercado sin código nuevo. A1 usa μ. La escala se recupera como b = (q₀,₉ − q₀,₅)/log 5. El comparador de la criba todavía identifica las columnas de cuantiles por `output = quantile_head_v1`. Declarar estas columnas como cuantiles de Laplace queda para el protocolo de [#746](https://github.com/GonxKZ/mars-titan/issues/746).

## Desviaciones y límites

- El artículo plantea la β-NLL para la gaussiana con el peso sg(σ^{2β}). Aquí el peso es sg(b), el que deja el gradiente de μ igual al del error absoluto. Es una adaptación propia.
- Se omite la constante log 2 de la NLL de Laplace. No cambia ningún gradiente.
- El suelo b_min es una decisión propia, no del artículo.
- La escala comparte el estado codificado con la mediana. Una escala con su propio tronco sería otro contraste.
- Con la memoria v1 sin residual ni LayerNorm, el estado codificado del fixture es casi nulo ([escala de la salida de MAC](../engineering/titans-mac-output-scale.md)) y la escala no varía entre filas. Las pruebas del recorrido usan la memoria de la receta histórica.
- F1 se rechaza con la cabeza de cuantiles, con una receta distinta de `mae`, con destilación, en el postentrenamiento y en el ajuste Dust inicial, que no tienen todavía una pérdida declarada para la escala. El lector episódico consume μ como cualquier otro padre y no ajusta la escala. El objetivo declara `scalar_point`, porque su punto emitido es la cabeza escalar sin transformar. El objetivo de IC por sesión no lo cumple y el lector lo sigue rechazando.

## Correspondencia entre ecuaciones, módulo y prueba

| Ecuación o regla | Módulo | Prueba |
| --- | --- | --- |
| ∂L/∂μ = sign(μ − y) y ∂L/∂b = sg(b)(1/b)(1 − \|y − μ\|/b), en FP32 y FP64 | `models/titans/laplace_scale.py`, `laplace_beta_nll` | `test_both_gradients_match_their_closed_form`, también contra `l1_loss` |
| El gradiente es el de la NLL ponderada con el peso fijo | `laplace_beta_nll` | `test_scale_gradient_is_that_of_the_weighted_nll_with_the_weight_held_fixed`, con `gradcheck` en FP64 |
| Valor sg(b)·(NLL − log 2) | `laplace_beta_nll` | `test_value_is_the_detached_scale_times_the_laplace_nll_without_log_two`, contra `scipy.stats.laplace` |
| Óptimo en b = E\|y − μ\| | `laplace_beta_nll` | `test_the_scale_gradient_vanishes_at_the_mean_absolute_deviation` |
| b = softplus(w·h + c) + b_min, w = 0 al empezar | `LaplaceScale` | `test_scale_starts_at_the_declared_value_without_global_rng` |
| Cuantiles q_τ de la Laplace | `laplace_quantiles` | `test_quantiles_are_those_of_the_laplace_and_keep_the_median`, contra `scipy.stats.laplace.ppf` |
| La cabeza de la mediana recibe el gradiente L1 y el tronco solo difiere con w ≠ 0 | `FinancialPredictor._prepare` | `test_gradients_reach_the_scale_and_the_median_keeps_the_l1_gradient` |

Las pruebas del predictor están en `tests/models/titans/test_laplace_scale_head.py` y las del recorrido en `tests/training/test_financial_run_laplace_scale.py`. Con un optimizador que solo registra gradientes, el recorrido comprueba que cada actualización usa la β-NLL con dos salidas por etiqueta, que todos los gradientes de la referencia coinciden bit a bit con los del control L1 mientras w = 0, que la repetición por bloques reproduce los gradientes y que alterar el sufijo futuro del corpus no cambia los gradientes anteriores al corte. También comprueba que las filas emitidas llevan los cuantiles de cada predicción y que la identidad del recorrido registra la escala.

## Paridad, estado y recuperación

Sin declarar `laplace_scale_head`, `FinancialConfig.identity()` no cambia, el predictor no crea el módulo y `PreparedDecisions.outputs` vale None. Con la escala, la identidad registra `output` y `output_objective` como cualquier objetivo de salida, los parámetros de la referencia conservan sus valores iniciales con la misma semilla, la mediana inicial coincide bit a bit y el generador global no avanza. `PreparedDecisions.outputs` contiene [μ, b] por fila. Solo se añaden `output_objective.weight` y `output_objective.bias`, 65 parámetros con H = 64, que pertenecen al papel `shared` del optimizador. No hay estado por flujo. La escala viaja en el `state_dict` del checkpoint y la prueba de ida y vuelta reproduce las escalas tras cargarlo.

## Controles

La [declaración](../../configs/experiments/laplace-scale-head-v1.json) parte de la receta histórica con máscaras, con la cabeza escalar y la pérdida `mae`. El control es la misma receta sin cambios del predictor, es decir, el objetivo L1 con los mismos pasos. Las dos configuraciones son `mac_online`, están desactivadas por defecto y no acceden al test final. La variante se descarta si su MAE por sesión de μ no mejora al del control con el contraste de la criba. La utilidad de la escala para A2 se contrasta allí, frente a sus propios controles de igual cobertura.

## Coste medido

La medida CPU del 28 de octubre usa 256 y 1.024 activos, ocho instantes con grafo, 64 sesiones de precios, 384 componentes de noticias, 512 de gráficos, 26 conceptos fundamentales y 140 indicadores macro. El predictor tiene H = 64 y la memoria de la receta histórica. Se usaron cuatro hilos, FP32 estricto, SDPA MATH, un calentamiento y siete repeticiones con 256 activos y cinco con 1.024, con orden rotatorio. El equipo tenía otras cargas, con una media de carga de un minuto entre 5 y 7 al empezar. Son entradas sintéticas y ningún parámetro cambia. Los recibos [de 256](../../reports/engineering/scale-head-horizons-20261027/cost-256.json) y [de 1.024 activos](../../reports/engineering/scale-head-horizons-20261027/cost-1024.json) conservan las muestras, las huellas de código y la carga.

| Activos | Modelo | Parámetros adicionales | Forward, pérdida y backward de ocho instantes | Relación |
| ---: | --- | ---: | ---: | ---: |
| 256 | Control L1 | 0 | 0,673 s | 1,00 |
| 256 | F1 | 65 | 0,671 s | 1,00 |
| 1.024 | Control L1 | 0 | 3,330 s | 1,00 |
| 1.024 | F1 | 65 | 3,273 s | 0,98 |

Los tiempos son medianas. La diferencia queda dentro de la variación entre repeticiones: la escala añade una proyección de H a 1 por fila. El pico de RAM del proceso fue de 1,45 GB y 3,41 GB, incluidos los otros modelos medidos. No se han medido VRAM ni energía. Esta medida describe el coste técnico, no el de un entrenamiento ni su utilidad predictiva.

```bash
CUDA_VISIBLE_DEVICES=-1 uv run python -m benchmarks.scale_head_horizons_cost \
  --rows 256 --instants 8 --repetitions 7 --warmup 1 --threads 4 \
  --output "${MARS_SCALE_HEAD_COST:?}"
```

## Comprobación y trabajo pendiente

La comprobación del 27 y 28 de octubre se hizo en CPU, sin pasos de optimizador y con la GPU ocupada por otra carga.

- Las 91 pruebas dirigidas de F1 y F5 pasan con medida de cobertura. Tras integrar los últimos cambios de develop, esas pruebas y las de los consumidores congelados, Dust y variantes regionales pasan con 281 casos.
- La regresión de los módulos que recorren estos cambios pasa con 385 casos y 7 omitidos por el bloqueo o por el enlace nativo. Falla `test_core_without_bank_trains_only_the_core` del adaptador del lector, que también falla en develop desde [#772](https://github.com/GonxKZ/mars-titan/pull/772) y no pasa por F1 ni F5.
- `make test-fast` pasa 11.220 pruebas y omite 381. La guarda del bloqueo de aprendizaje pasa con 54 casos y una omisión prevista.
- Seis mutaciones dirigidas de F1, dentro de las quince de la familia, se detectan con los controles en verde antes y después. Se ejecutaron antes de integrar desde develop C1, C6 y el registro de sus fuentes, sobre el mismo código de F1.
- Con coverage 7.16.1 y radon 6.0.1, `laplace_scale.py` cubre 49/49 sentencias y 10/10 ramas, con CRAP máximo 10.
- El 28 de octubre, tras pasar F1 al marco de objetivos de salida de #760, la regresión de los 22 módulos que recorren estos cambios pasó 588 casos con una omisión de la guarda. Los cuatro fallos eran pruebas del recorrido de F1 que esperaban la interfaz anterior, y sus 9 casos pasan tras actualizarlas.

El [recibo de verificación](../../reports/engineering/scale-head-horizons-20261027/verification.json) conserva los recuentos y las huellas de los informes JUnit y de las mutaciones.

Queda pendiente:

- La comprobación en `cuda:0`, que espera a que la GPU quede libre.
- La declaración de los cuantiles de Laplace en el comparador de la criba de [#746](https://github.com/GonxKZ/mars-titan/issues/746).
- Cualquier evaluación predictiva, que sigue bloqueada.
