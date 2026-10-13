# Análisis de las trazas de prueba de cada familia

Trazas escritas el 13 de octubre de 2026 por las pruebas de #448 sobre los corpus técnicos de las pruebas, con trazas en cada paso y optimizadores que no cambian pesos. Se obtienen con `pytest -k trace` sobre los archivos de pruebas de trazas de cada familia, del postentrenamiento y de la etapa RL nativa, y `scripts/analyze_learning_traces.py --root` sobre su carpeta temporal. Comprueban que todas escriben el contrato común y que el análisis lo lee. Los valores no tienen significado científico.

| Ejecución | Familia | Unidad | Pasos con filas | Filas | Bytes | Presupuesto usado | Agotado |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `test_double_dqn_traces_keep_th0/traced/traces` | double_dqn | transiciones | 3 | 24 | 6480 | 0.04% | no |
| `test_klpo_traces_record_the_in0/out/traces` | klpo_terminal | oleadas | 1 | 6 | 2116 | 0.01% | no |
| `test_ppo_traces_are_not_duplic0/out/traces` | ppo_clip_full_kl_v1 | transiciones | 1 | 8 | 2157 | 0.01% | no |
| `test_ridge_writes_one_fit_even0/traces` | ridge | ajustes | 1 | 9 | 1964 | 0.19% | no |
| `test_titans_posttraining_trace0/traced/traces` | posttraining_titans_mac | pasos del optimizador | 22 | 944 | 13370 | 0.08% | no |
| `test_traces_follow_the_writes_0/resumed/traces` | b6 | escrituras de la memoria asociativa | 40 | 721 | 12121 | 0.07% | no |
| `test_traces_follow_the_writes_0/traced/traces` | b6 | escrituras de la memoria asociativa | 40 | 721 | 12121 | 0.07% | no |
| `test_traces_keep_the_candidate0/resumed/traces` | episodic_gru | pasos del optimizador | 14 | 1068 | 12954 | 0.08% | no |
| `test_traces_keep_the_candidate0/traced/traces` | episodic_gru | pasos del optimizador | 14 | 1068 | 10468 | 0.06% | no |
| `test_traces_keep_the_online_co0/traced/traces` | transformer_compact_online | pasos del optimizador | 40 | 1205 | 11633 | 0.07% | no |
| `test_traces_keep_the_reader_bi0/resumed/traces` | mars_titan_readout | pasos del optimizador | 43 | 2282 | 29192 | 0.17% | no |
| `test_traces_keep_the_reader_bi0/traced/traces` | mars_titan_readout | pasos del optimizador | 43 | 2282 | 26804 | 0.16% | no |
| `test_traces_keep_the_reference0/traced/traces` | neural_reference | pasos del optimizador | 4 | 68 | 4316 | 0.03% | no |
| `test_traces_keep_the_reference1/traced/traces` | neural_reference | pasos del optimizador | 4 | 68 | 4333 | 0.03% | no |
| `test_traces_keep_the_window_bi0/resumed/traces` | titans_mac | pasos del optimizador | 43 | 2457 | 29726 | 0.18% | no |
| `test_traces_keep_the_window_bi0/traced/traces` | titans_mac | pasos del optimizador | 43 | 2457 | 27301 | 0.16% | no |
| `test_traces_reach_every_fitted0/traced/jobs/US/fold-001/gru__full_continuation/fit-s42/run/traces` | posttraining | pasos del optimizador | 2 | 27 | 4048 | 0.02% | no |
| `test_traces_reach_every_fitted0/traced/jobs/US/fold-001/gru__fusion/fit-s42/run/traces` | posttraining | pasos del optimizador | 2 | 34 | 4134 | 0.02% | no |
| `test_traces_reach_every_fitted0/traced/jobs/US/fold-001/gru__fusion_full_rank/fit-s42/run/traces` | posttraining | pasos del optimizador | 2 | 34 | 4105 | 0.02% | no |
| `test_traces_reach_every_fitted0/traced/jobs/US/fold-001/gru__head/fit-s42/run/traces` | posttraining | pasos del optimizador | 2 | 34 | 5857 | 0.03% | no |
| `test_traces_reach_every_fitted0/traced/jobs/US/fold-001/gru__head_fusion/fit-s42/run/traces` | posttraining | pasos del optimizador | 2 | 61 | 4215 | 0.03% | no |
| `test_traces_reach_the_adapted_0/traced/traces` | posttraining_episodic_gru | pasos del optimizador | 43 | 280 | 7219 | 0.04% | no |
| `test_traces_reach_the_reader_c0/traced/traces` | posttraining_readout | pasos del optimizador | 10 | 659 | 10622 | 0.06% | no |
| `test_traces_resume_from_the_ch0/continuous/traces` | neural_reference | pasos del optimizador | 4 | 68 | 4321 | 0.03% | no |
| `test_traces_resume_from_the_ch0/paused/traces` | neural_reference | pasos del optimizador | 4 | 68 | 11823 | 0.07% | no |
| `test_xgboost_rounds_follow_the0/traces` | xgboost | rondas de boosting | 2 | 6 | 3736 | 0.36% | no |

## `test_double_dqn_traces_keep_th0/traced/traces`

Fases: validation. Cadencia: cada 1 transiciones.

Políticas: `validation/rl.episodes/all` 1, `validation/rl.improved/all` 0, `validation/rl.mean_liquidated_log_growth/all` 0.002211, `validation/rl.mean_log_growth/all` 0.002462, `validation/rl.mean_max_drawdown/all` 0.003922, `validation/rl.ruin_count/all` 0, `validation/rl.selection_score/all` 0.002211, `validation/rl.stale_evaluations/all` 2.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_klpo_traces_record_the_in0/out/traces`

Fases: validation. Cadencia: cada 1 oleadas.

Políticas: `validation/rl.episodes/all` 1, `validation/rl.improved/all` 1, `validation/rl.mean_liquidated_log_growth/all` -0.006414, `validation/rl.mean_log_growth/all` -0.006165, `validation/rl.ruin_count/all` 0, `validation/rl.selection_score/all` -0.006414.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_ppo_traces_are_not_duplic0/out/traces`

Fases: validation. Cadencia: cada 1 transiciones.

Políticas: `validation/rl.episodes/all` 1, `validation/rl.improved/all` 1, `validation/rl.mean_liquidated_log_growth/all` -0.006414, `validation/rl.mean_log_growth/all` -0.006165, `validation/rl.mean_max_drawdown/all` 0.006429, `validation/rl.ruin_count/all` 0, `validation/rl.selection_score/all` -0.006414, `validation/rl.stale_evaluations/all` 0.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_ridge_writes_one_fit_even0/traces`

Fases: validation. Cadencia: cada 50 ajustes.
Validación: mejor 0.125 en el paso 0, última 0.125.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_titans_posttraining_trace0/traced/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.05422 en el paso 0, última 0.05422.
Optimización: pérdida 0.04527 → 0.04516, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `predictor.mac` | 9.714e-05 | 0 | no |

Memoria de Titans:

- `predictor.mac.memory`: alpha_high_fraction_mean 0, alpha_last_mean 0.002708, alpha_low_fraction_mean 1, associative_loss_first 0.2217, associative_loss_last 0.003919, eta_high_fraction_mean 0, eta_last_mean 0.4981, eta_low_fraction_mean 0, theta_last_mean 0.05021, traced_tokens 106.

Adaptadores:

- `predictor.mac.persistent`: delta_l2 0, delta_relative 0.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_follow_the_writes_0/resumed/traces`

Fases: b6_calibration, b6_evaluation, b6_validation, validation. Cadencia: cada 1 escrituras de la memoria asociativa.
Validación: mejor 0.02638 en el paso 19, última 0.02638.

Corrección B6: `b6_calibration/associative.matrix_l2/all` 0.06719, `b6_calibration/associative.writes/all` 58, `b6_evaluation/associative.matrix_l2/all` 0.06564, `b6_evaluation/associative.writes/all` 110, `b6_validation/associative.matrix_l2/all` 0.06345, `b6_validation/associative.writes/all` 53.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_follow_the_writes_0/traced/traces`

Fases: b6_calibration, b6_evaluation, b6_validation, validation. Cadencia: cada 1 escrituras de la memoria asociativa.
Validación: mejor 0.02638 en el paso 19, última 0.02638.

Corrección B6: `b6_calibration/associative.matrix_l2/all` 0.06719, `b6_calibration/associative.writes/all` 58, `b6_evaluation/associative.matrix_l2/all` 0.06564, `b6_evaluation/associative.writes/all` 110, `b6_validation/associative.matrix_l2/all` 0.06345, `b6_validation/associative.writes/all` 53.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_keep_the_candidate0/resumed/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 5.855 en el paso 0, última 5.855.
Optimización: pérdida 0.9609 → 0.9002, fracción recortada 1.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.charts_bias` | 0.09533 | 0 | no |
| `model.charts_weight` | 0.5392 | 0 | no |
| `model.fundamentals_bias` | 0 | 0 | no |
| `model.fundamentals_weight` | 0 | 0 | no |
| `model.fusion_bias` | 0.1275 | 0 | no |
| `model.fusion_weight` | 1.497 | 0 | no |
| `model.head_bias` | 0.3479 | 0 | no |
| `model.head_weight` | 12.7 | 0 | no |
| `model.initial_bias` | 0.207 | 0 | no |
| `model.initial_weight` | 4.043 | 0 | no |
| `model.macro_bias` | 0.02534 | 0 | no |
| `model.macro_weight` | 0.02534 | 0 | no |
| `model.news_bias` | 0 | 0 | no |
| `model.news_weight` | 0 | 0 | no |
| `model.price_bias_hh` | 0.2187 | 0 | no |
| `model.price_bias_ih` | 0.4077 | 0 | no |
| `model.price_weight_hh` | 0.3032 | 0 | no |
| `model.price_weight_ih` | 0.2828 | 0 | no |
| `model.query_bias` | 4.504e-12 | 0 | no |
| `model.query_weight` | 1.593e-10 | 0 | no |
| `model.step_logit` | 0.03253 | 0 | no |
| `model.update_bias` | 2.53e-06 | 0 | no |
| `model.update_weight` | 0.0001022 | 0 | no |
| `model.value_bias` | 7.24e-07 | 0 | no |
| `model.value_weight` | 2.513e-06 | 0 | no |

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_keep_the_candidate0/traced/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 5.855 en el paso 0, última 5.855.
Optimización: pérdida 0.9609 → 0.9002, fracción recortada 1.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.charts_bias` | 0.09533 | 0 | no |
| `model.charts_weight` | 0.5392 | 0 | no |
| `model.fundamentals_bias` | 0 | 0 | no |
| `model.fundamentals_weight` | 0 | 0 | no |
| `model.fusion_bias` | 0.1275 | 0 | no |
| `model.fusion_weight` | 1.497 | 0 | no |
| `model.head_bias` | 0.3479 | 0 | no |
| `model.head_weight` | 12.7 | 0 | no |
| `model.initial_bias` | 0.207 | 0 | no |
| `model.initial_weight` | 4.043 | 0 | no |
| `model.macro_bias` | 0.02534 | 0 | no |
| `model.macro_weight` | 0.02534 | 0 | no |
| `model.news_bias` | 0 | 0 | no |
| `model.news_weight` | 0 | 0 | no |
| `model.price_bias_hh` | 0.2187 | 0 | no |
| `model.price_bias_ih` | 0.4077 | 0 | no |
| `model.price_weight_hh` | 0.3032 | 0 | no |
| `model.price_weight_ih` | 0.2828 | 0 | no |
| `model.query_bias` | 4.504e-12 | 0 | no |
| `model.query_weight` | 1.593e-10 | 0 | no |
| `model.step_logit` | 0.03253 | 0 | no |
| `model.update_bias` | 2.53e-06 | 0 | no |
| `model.update_weight` | 0.0001022 | 0 | no |
| `model.value_bias` | 7.24e-07 | 0 | no |
| `model.value_weight` | 2.513e-06 | 0 | no |

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_keep_the_online_co0/traced/traces`

Fases: online_calibration, online_evaluation, online_validation, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.1617 en el paso 19, última 0.1617.
Optimización: pérdida 0.06091 → 0.06029, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.encoders` | sin dato | sin dato | sin dato |
| `model.fusion` | sin dato | sin dato | sin dato |
| `model.head` | sin dato | sin dato | sin dato |
| `model.price_encoder` | sin dato | sin dato | sin dato |

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_keep_the_reader_bi0/resumed/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.05276 en el paso 0, última 0.05276.
Optimización: pérdida 0.04524 → 0.04487, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `readout.query_projection` | 6.095e-05 | 0 | no |
| `readout.refinement` | 0.00727 | 0 | no |
| `readout.step_logit` | 0.0002736 | 0 | no |
| `readout.value_projection` | 0.001943 | 0 | no |

Memoria de Titans:

- `predictor.mac.memory`: alpha_high_fraction_mean 0, alpha_last_mean 0.002708, alpha_low_fraction_mean 1, associative_loss_first 0.2217, associative_loss_last 0.003919, eta_high_fraction_mean 0, eta_last_mean 0.4981, eta_low_fraction_mean 0, theta_last_mean 0.05021, traced_tokens 212.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_keep_the_reader_bi0/traced/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.05276 en el paso 0, última 0.05276.
Optimización: pérdida 0.04524 → 0.04487, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `readout.query_projection` | 6.095e-05 | 0 | no |
| `readout.refinement` | 0.00727 | 0 | no |
| `readout.step_logit` | 0.0002736 | 0 | no |
| `readout.value_projection` | 0.001943 | 0 | no |

Memoria de Titans:

- `predictor.mac.memory`: alpha_high_fraction_mean 0, alpha_last_mean 0.002708, alpha_low_fraction_mean 1, associative_loss_first 0.2217, associative_loss_last 0.003919, eta_high_fraction_mean 0, eta_last_mean 0.4981, eta_low_fraction_mean 0, theta_last_mean 0.05021, traced_tokens 212.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_keep_the_reference0/traced/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.08277 en el paso 2, última 0.08277.
Optimización: pérdida 0.01283 → 0.01274, fracción recortada sin dato.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.encoders` | 0.0403 | 0 | no |
| `model.fusion` | 0.1157 | 0 | no |
| `model.head` | 0.2317 | 0 | no |
| `model.price_encoder` | 0.0185 | 0 | no |

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_keep_the_reference1/traced/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.1942 en el paso 2, última 0.1942.
Optimización: pérdida 0.0264 → 0.02817, fracción recortada sin dato.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.encoders` | 0.05234 | 0 | no |
| `model.fusion` | 0.4874 | 0 | no |
| `model.head` | 0.4145 | 0 | no |
| `model.price_encoder` | 0.1687 | 0 | no |

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_keep_the_window_bi0/resumed/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.05422 en el paso 0, última 0.05422.
Optimización: pérdida 0.04527 → 0.04516, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `predictor.encoders` | 7.255e-06 | 0 | no |
| `predictor.fusion` | 0.0001823 | 0 | no |
| `predictor.head` | 0.1015 | 0 | no |
| `predictor.mac` | 0.0002349 | 0 | no |
| `predictor.price_encoder` | 4.951e-05 | 0 | no |

Memoria de Titans:

- `predictor.mac.memory`: alpha_high_fraction_mean 0, alpha_last_mean 0.002708, alpha_low_fraction_mean 1, associative_loss_first 0.2217, associative_loss_last 0.003919, eta_high_fraction_mean 0, eta_last_mean 0.4981, eta_low_fraction_mean 0, theta_last_mean 0.05021, traced_tokens 212.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_keep_the_window_bi0/traced/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.05422 en el paso 0, última 0.05422.
Optimización: pérdida 0.04527 → 0.04516, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `predictor.encoders` | 7.255e-06 | 0 | no |
| `predictor.fusion` | 0.0001823 | 0 | no |
| `predictor.head` | 0.1015 | 0 | no |
| `predictor.mac` | 0.0002349 | 0 | no |
| `predictor.price_encoder` | 4.951e-05 | 0 | no |

Memoria de Titans:

- `predictor.mac.memory`: alpha_high_fraction_mean 0, alpha_last_mean 0.002708, alpha_low_fraction_mean 1, associative_loss_first 0.2217, associative_loss_last 0.003919, eta_high_fraction_mean 0, eta_last_mean 0.4981, eta_low_fraction_mean 0, theta_last_mean 0.05021, traced_tokens 212.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_reach_every_fitted0/traced/jobs/US/fold-001/gru__full_continuation/fit-s42/run/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 1.344 en el paso 0, última 1.344.
Optimización: pérdida 0.2995 → 0.2995, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.encoders` | 0.09019 | 0 | no |
| `model.fusion` | 0.2807 | 0 | no |
| `model.head` | 0.3299 | 0 | no |
| `model.price_encoder` | 0.0345 | 0 | no |

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_reach_every_fitted0/traced/jobs/US/fold-001/gru__fusion/fit-s42/run/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 1.344 en el paso 0, última 1.344.
Optimización: pérdida 0.2995 → 0.2995, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.fusion` | 0.02473 | 0 | no |

Adaptadores:

- `model.fusion.0.weight`: delta_l2 0, delta_relative 0.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_reach_every_fitted0/traced/jobs/US/fold-001/gru__fusion_full_rank/fit-s42/run/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 1.344 en el paso 0, última 1.344.
Optimización: pérdida 0.2995 → 0.2995, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.fusion` | 0.2595 | 0 | no |

Adaptadores:

- `model.fusion.0.weight`: delta_l2 0, delta_relative 0.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_reach_every_fitted0/traced/jobs/US/fold-001/gru__head/fit-s42/run/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 1.344 en el paso 0, última 1.344.
Optimización: pérdida 0.2995 → 0.2995, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.head` | 0.3299 | 0 | no |

Adaptadores:

- `model.head.bias`: delta_l2 0, delta_relative 0.
- `model.head.weight`: delta_l2 0, delta_relative 0.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_reach_every_fitted0/traced/jobs/US/fold-001/gru__head_fusion/fit-s42/run/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 1.344 en el paso 0, última 1.344.
Optimización: pérdida 0.2995 → 0.2995, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.fusion` | 0.01279 | 0 | no |
| `model.head` | 0.3299 | 0 | no |

Adaptadores:

- `model.fusion.0.weight`: delta_l2 0, delta_relative 0.
- `model.head.bias`: delta_l2 0, delta_relative 0.
- `model.head.weight`: delta_l2 0, delta_relative 0.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_reach_the_adapted_0/traced/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.3976 en el paso 0, última 0.3976.
Optimización: pérdida 0.04344 → 0.0587, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `head.parametrizations` | 0.1102 | 0 | no |

Adaptadores:

- `head..bias`: delta_l2 0, delta_relative 0.
- `head..weight`: delta_l2 0, delta_relative 0.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_reach_the_reader_c0/traced/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.1582 en el paso 0, última 0.1582.
Optimización: pérdida 0.04907 → 0.05394, fracción recortada 0.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `predictor.fusion` | 0.01395 | 0 | no |
| `predictor.head` | 0.1079 | 0 | no |
| `predictor.mac` | 0.01891 | 0 | no |
| `readout.query_projection` | 1.316e-06 | 0 | no |
| `readout.value_projection` | 0.00055 | 0 | no |

Memoria de Titans:

- `predictor.mac.memory`: alpha_high_fraction_mean 0, alpha_last_mean 0.002691, alpha_low_fraction_mean 1, associative_loss_first 27.09, associative_loss_last 22.29, eta_high_fraction_mean 0, eta_last_mean 0.5194, eta_low_fraction_mean 0, theta_last_mean 0.04869, traced_tokens 26.

Adaptadores:

- `predictor.fusion.0.weight`: delta_l2 0, delta_relative 0.
- `predictor.head.bias`: delta_l2 0, delta_relative 0.
- `predictor.head.weight`: delta_l2 0, delta_relative 0.
- `predictor.mac.attention.out_proj.weight`: delta_l2 0, delta_relative 0.
- `predictor.mac.query_projection.weight`: delta_l2 0, delta_relative 0.
- `readout.query_projection.weight`: delta_l2 0, delta_relative 0.
- `readout.value_projection.weight`: delta_l2 0, delta_relative 0.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_resume_from_the_ch0/continuous/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.07677 en el paso 2, última 0.07677.
Optimización: pérdida 0.01192 → 0.01045, fracción recortada sin dato.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.encoders` | 0.03332 | 0 | no |
| `model.fusion` | 0.1187 | 0 | no |
| `model.head` | 0.2168 | 0 | no |
| `model.price_encoder` | 0.01332 | 0 | no |

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_traces_resume_from_the_ch0/paused/traces`

Fases: train, validation. Cadencia: cada 1 pasos del optimizador.
Validación: mejor 0.07677 en el paso 2, última 0.07677.
Optimización: pérdida 0.01192 → 0.01045, fracción recortada sin dato.

| Grupo | Mediana de la norma del gradiente | Mediana de la razón de actualización | Se mueve |
| --- | ---: | ---: | --- |
| `model.encoders` | 0.03332 | 0 | no |
| `model.fusion` | 0.1187 | 0 | no |
| `model.head` | 0.2168 | 0 | no |
| `model.price_encoder` | 0.01332 | 0 | no |

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.

## `test_xgboost_rounds_follow_the0/traces`

Fases: validation. Cadencia: cada 2 rondas de boosting.
Validación: mejor 0.25 en el paso 4, última 0.25.

Integridad: 0 valores no finitos y 0 estadísticas con valores no finitos en su tensor.
