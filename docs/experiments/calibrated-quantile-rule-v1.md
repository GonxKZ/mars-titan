# Regla fija sobre cuantiles calibrados y no operar

G2 es la parte de políticas financieras del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#748](https://github.com/GonxKZ/mars-titan/issues/748)). Compara políticas que leen cuantiles calibrados. PPO y KLPO ([#137](https://github.com/GonxKZ/mars-titan/issues/137)) necesitan entrenar y siguen bloqueados mientras rija el bloqueo de aprendizaje. Este trabajo implementa los dos controles que pueden descartarlas, sin aprender nada y conectados al entorno financiero auditado ([#365](https://github.com/GonxKZ/mars-titan/issues/365)). Está en `src/mars_titan/simulation/calibrated_quantile_rule.py` y se ha comprobado con cintas sintéticas. No hay ningún resultado financiero medido con datos.

## Regla

La cinta del entorno lleva una puntuación por activo y sesión. Para esta regla es la emisión de A2 (`neutral_abstention_v1`):

$$
s_i=\begin{cases}q_{0,5,i}&\text{si } l_i>0 \text{ o } u_i<0,\\0&\text{en otro caso,}\end{cases}
$$

con $[l_i,u_i]$ el intervalo central calibrado de cobertura nominal 0,8. Con la CQR estática de G se obtiene con `neutral_abstention.apply`. Con la calibración de A3 (`regime_weighted_conformal_v1`), la misma puntuación sale de `neutral_abstention.emit_calibrated`. En cada decisión, `calibrated_quantile_rule_v1` elige la exposición

$$
e_t=\begin{cases}1&\text{si existe un activo válido con } s_i>0,\\0&\text{en otro caso.}\end{cases}
$$

Un activo es válido si tiene cierre valorado, puntuación definida y no se ha dado de baja, la misma marca que ya da la observación del entorno. El entorno reparte la exposición por igual en el cuartil superior de las puntuaciones positivas (`positive_top_quartile_equal_weight`), así que la regla solo compra activos con una afirmación alcista calibrada. Como el entorno no admite ventas en corto, una afirmación bajista equivale a no operar con ese activo.

El control `no_trade` se queda en efectivo en todas las decisiones. Es `evaluation.fixed_policy("cash")`, la referencia que ya usa la campaña de políticas. `compare(env)` evalúa los dos sobre el mismo entorno con `evaluation.evaluate` y exige el reparto anterior, porque con el reparto 1/N la puntuación no decidiría qué se compra.

## Por qué estos controles

Una política aprendida que lea los mismos cuantiles debe mejorar a una regla fija que ya usa esa información y a no operar. Si no supera a la regla, el aprendizaje no añade nada a la afirmación calibrada. Si no supera a no operar, no hay valor financiero que justificar con sus costes. La regla tampoco se ajusta: no tiene umbrales elegidos con datos, porque el intervalo y su cobertura los fija A2 en G.

## Causalidad

La regla solo lee la observación del entorno en el instante de decidir. La puntuación de la cinta en esa sesión procede de una predicción emitida en ella, y la calibración de A2 se ajusta en G y rechaza cualquier emisión anterior a su última etiqueta. La cartera ejecuta las órdenes en la apertura siguiente con las reglas del entorno. Una cartera arruinada o sin valoración tiene una observación nula y la regla se queda en efectivo.

## Correspondencia con pruebas

Las pruebas están en `tests/simulation/test_calibrated_quantile_rule.py`.

| Elemento | Código | Prueba |
| --- | --- | --- |
| $e_t$ con $s_i>0$ estricto | `rule_policy` | `test_the_rule_invests_only_when_some_asset_has_a_positive_claim` |
| Solo activos válidos | `rule_policy` | `test_a_claim_on_an_asset_without_a_valued_close_does_not_invest` |
| Reparto en afirmaciones alcistas | entorno | `test_the_exposure_goes_only_to_assets_with_an_upward_claim` |
| $s_i$ desde A2 | `neutral_abstention.apply` | `test_the_rule_reads_the_a2_claims_of_the_calibrated_intervals` |
| No operar y reproducibilidad | `compare` | `test_no_trade_keeps_the_capital_in_cash` |
| Formato de la observación | `rule_policy` | `test_observations_without_the_environment_layout_are_rejected` y `test_a_ruined_or_unvalued_book_stays_in_cash` |

## Límites

- La observación divide la puntuación por `score_scale` y la guarda en float32. Una mediana positiva menor que unos $10^{-45}$ veces la escala se leería como cero. Con residuos diarios no ocurre, pero queda declarado.
- Las cintas reconstruidas con datos reales comprueban que las predicciones coinciden con la huella del recibo de cada ventana. Las puntuaciones de A2 son otra serie y necesitan su propio recibo antes de poder cargarse. Ese paso queda para la criba ([#746](https://github.com/GonxKZ/mars-titan/issues/746)) y para la campaña de políticas cuando se levante el bloqueo.
- El motor contable nativo comparte el formato de observación, pero no hay binarios compilados vigentes y no se ha probado aquí.

## Coste medido

La regla no tiene tensores ni parámetros, así que su coste es el del recorrido del entorno. `benchmarks/calibrated_quantile_rule_cost.py` mide `compare`, que evalúa la regla y no operar, sobre cintas sintéticas de semilla fija con 252 sesiones y una afirmación alcista en un 10 % de las filas. El recibo está en `reports/engineering/output-stages-20261023/calibrated-quantile-rule-cost.json`. Se tomó antes de rebasar la rama sobre `develop`, así que su campo `commit` no aparece en la historia publicada. Las huellas SHA-256 de las fuentes que guarda coinciden con las de esta versión. Son medianas de tres repeticiones con una carga media de la máquina en torno a 50, así que no son cotas superiores.

| Activos | Decisiones por política | `compare` (regla y no operar) | Una decisión de la regla |
| ---: | ---: | ---: | ---: |
| 64 | 251 | 2,24 s | 17,5 µs |
| 256 | 251 | 8,53 s | 12,6 µs |

Casi todo el tiempo es la contabilidad del entorno en Python. Con 256 activos cada paso tarda unos 17 ms y la decisión de la regla menos de una milésima de eso. No se midieron el motor contable nativo, las cintas reconstruidas con datos reales ni la energía.

## Mutaciones dirigidas

Se aplicaron a mano 9 mutaciones y se revirtieron después: desigualdad no estricta en la puntuación, quitar la marca de validez, invertir o cambiar `any` por `all` en la decisión, quitar la comprobación del resto de la observación y la del reparto, leer otro rasgo como puntuación, invertir con 0,75 en lugar de 1 y cambiar no operar por la regla fija del 50 %. Todas hacen fallar alguna prueba. La del resto sobrevivía al principio, porque la observación de siete valores ya fallaba por no tener ningún activo, y motivó el caso de quince valores.

## Pendiente

- Comparar PPO y KLPO con estos controles en las mismas cintas y con las mismas semillas cuando se levante el bloqueo de aprendizaje.
- Un recibo para cargar las puntuaciones de A2 en cintas reconstruidas con datos reales (#746).
