# Revisión de bibliotecas científicas reutilizables

Esta revisión contrasta las piezas científicas que el proyecto ha escrito por su cuenta con las bibliotecas maduras que resuelven lo mismo. Se hizo el 10 de octubre de 2026 sobre `develop` en `c41afcc1` y se sigue en [#490](https://github.com/GonxKZ/mars-titan/issues/490). La parte de HPC, datos e infraestructura está en la [revisión de bibliotecas de ingeniería](../engineering/library-review.md) ([#485](https://github.com/GonxKZ/mars-titan/issues/485)).

Para cada candidata se indica qué pieza propia sustituiría o contrastaría, la versión elegida con su licencia, si resuelve con el lock del proyecto, cómo se mantiene, qué riesgos tiene y un veredicto entre cuatro: adoptar, referencia de paridad solo en pruebas, evaluar con medida o descartar. Cuando una biblioteca solo sirve para leer su código o su documentación, el veredicto es lectura.

## Conclusiones

- **Adoptar `arch` 8.0.0 con `statsmodels` 0.15.0** ([#493](https://github.com/GonxKZ/mars-titan/issues/493)). Con la misma semilla, `CircularBlockBootstrap` genera exactamente los mismos índices que `circular_block_indices`, así que el bootstrap propio queda contrastado sin cambiarlo. Además aporta lo que el proyecto no tiene: Diebold-Mariano con la corrección de Harvey, SPA, MCS, Reality Check, StepM y la longitud de bloque de Politis y White.
- **Cuatro referencias de paridad solo en pruebas.** PEFT 0.21.0 para los adaptadores ([#494](https://github.com/GonxKZ/mars-titan/issues/494)), scoringrules 0.11.0 para pinball y puntuación de intervalo ([#495](https://github.com/GonxKZ/mars-titan/issues/495)), MAPIE 1.5.0 para la CQR estática ([#496](https://github.com/GonxKZ/mars-titan/issues/496)) y sb3-contrib 2.9.0 para la pérdida QR-DQN ([#497](https://github.com/GonxKZ/mars-titan/issues/497)). En la comprobación preliminar las cuatro coincidieron con el código propio hasta el redondeo de FP64, salvo los gradientes de DoRA, que difieren por una desviación ya declarada.
- **titans-pytorch sigue como oráculo de la memoria neuronal**, ahora identificado también por su versión de PyPI. La [revisión de implementaciones de Titans](titans-reference-implementations.md#actualización-del-10-de-octubre-de-2026) recoge la actualización.
- **Ninguna biblioteca sustituye una pieza del núcleo.** Unas no cubren lo que el proyecto necesita (adaptar los pesos de una GRU con DoRA, cohortes con etiquetas retrasadas, PPO con penalización KL fija, la recurrencia exacta de Titans). Otras no resuelven con el lock, arrastran muchas dependencias, han dejado de mantenerse o, como los núcleos de Triton de fla, usan TF32 por defecto.

## Método

**Versión elegida.** La más reciente publicada en PyPI hasta el 26 de septiembre de 2026, dos semanas antes de la revisión. Las fechas son las de subida de la rueda en PyPI. Cuando hay una versión posterior se indica en la tabla, sin usarla.

**Compatibilidad.** Cada candidata se resolvió con `uv pip compile` contra las restricciones exportadas de `uv.lock` (`uv export --frozen --all-extras`), Python 3.12, Linux x86-64, el índice cu130 de PyTorch y `--no-build`. La resolución no instala nada. Una candidata «resuelve» si encaja con todas las versiones fijadas del proyecto, incluidos torch 2.14.0+cu130, NumPy 2.5.3 y pandas 3.0.6. Las resoluciones no fijaron fecha límite, así que algunas dependencias transitivas salieron más recientes que el corte (por ejemplo `wrapt` 2.5.0, del 27 de septiembre, o `tensordict` 0.14.3, del 8 de octubre). La adopción tiene que resolver con `exclude-newer`.

**Lectura del código.** Las ruedas se descargaron en un directorio temporal fuera del repositorio y se leyeron sin ejecutarlas, salvo en la comprobación siguiente.

**Comprobación preliminar.** Un entorno temporal fuera del repositorio, sin torch propio, reutilizó los paquetes del entorno del proyecto y añadió PEFT, accelerate, scoringrules, MAPIE, arch, statsmodels, stable-baselines3 y sb3-contrib con `--no-deps`. Se ejecutó solo en CPU, con dos hilos, TF32 desactivado y algoritmos deterministas. No hubo optimizador, pasos de entorno ni ajuste de ningún modelo. Sus cifras son preliminares y no son un recibo del repositorio. Cada issue de paridad tiene que reproducirlas en pruebas con tolerancias declaradas antes de ejecutar.

## Tabla de veredictos

| Candidata | Versión y fecha | Licencia | Resolución con el lock | Pieza propia o hueco | Veredicto |
| --- | --- | --- | --- | --- | --- |
| [arch](https://github.com/bashtage/arch) | 8.0.0, 21-10-2025 | NCSA | Sí, con statsmodels, patsy, formulaic, interface-meta y wrapt | `evaluation/paired_comparisons.py`. Faltan DM, SPA y MCS | Adoptar, [#493](https://github.com/GonxKZ/mars-titan/issues/493) |
| [statsmodels](https://github.com/statsmodels/statsmodels) | 0.15.0, 27-08-2026 | BSD-3-Clause | Sí | Falta Diebold-Mariano | Adoptar con arch, [#493](https://github.com/GonxKZ/mars-titan/issues/493) |
| [PEFT](https://github.com/huggingface/peft) | 0.21.0, 15-09-2026 (hay 0.21.2) | Apache-2.0 | Sí, con accelerate | `models/predictive_adaptation.py` | Paridad en pruebas, [#494](https://github.com/GonxKZ/mars-titan/issues/494) |
| [scoringrules](https://github.com/frazane/scoringrules) | 0.11.0, 06-06-2026 | Apache-2.0 | Sí, sin otros paquetes | `models/quantile_head.py`, `evaluation/forecast_scores.py`, `integrity/independent_scores.py` | Paridad en pruebas, [#495](https://github.com/GonxKZ/mars-titan/issues/495) |
| [properscoring](https://github.com/TheClimateCorporation/properscoring) | 0.1, 12-11-2015 | Apache-2.0 | Sí | Igual que scoringrules | Descartar, sin versiones desde 2015 |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | 1.5.0, 05-08-2026 | BSD-3-Clause | Sí, sin otros paquetes | `calibration/conformal_quantiles.py` | Paridad en pruebas, [#496](https://github.com/GonxKZ/mars-titan/issues/496) |
| [crepes](https://github.com/henrikbostrom/crepes) | 0.9.1, 12-06-2026 | BSD-3-Clause | Sí | CQR estática por grupos | Descartar, duplica a MAPIE y no cubre el retraso |
| [puncc](https://github.com/deel-ai/puncc) | 0.9.3, 23-06-2026 | MIT | No, exige NumPy < 2 | CQR estática | Descartar |
| [empyrical-reloaded](https://github.com/stefan-jansen/empyrical-reloaded) | 0.5.12, 01-06-2025 | Apache-2.0 | No, exige peewee < 3.17.4, que no tiene ruedas | `evaluation/financial_metrics.py` | Lectura de convenciones |
| [quantstats](https://github.com/ranaroussi/quantstats) | 0.0.81, 13-01-2026 | Apache-2.0 | Sí, con yfinance, curl-cffi, websockets, seaborn, peewee y protobuf | `evaluation/financial_metrics.py` | Descartar |
| [filterpy](https://github.com/rlabbe/filterpy) | 1.4.5, 10-10-2018 | MIT | No, solo distribución de fuentes | Regla `kalman` de `memory/associative_memory.py` (PT3) | Descartar |
| statsmodels, espacio de estados | 0.15.0 | BSD-3-Clause | Sí | Regla `kalman` (PT3) | Descartar, la regla ya se contrasta con formas cerradas |
| [scikit-learn-extra](https://github.com/scikit-learn-contrib/scikit-learn-extra) | 0.3.0, 27-03-2023 | BSD-3-Clause | No, sin rueda para Python 3.12 | `cm/medoids.py` | Descartar |
| [kmedoids](https://github.com/kno10/python-kmedoids) | 0.5.5, 23-05-2026 | GPL-3.0-or-later | Sí, sin otros paquetes | `cm/medoids.py`, `cm/anchored_medoids.py` | Evaluar con medida en un entorno aislado |
| [titans-pytorch](https://github.com/lucidrains/titans-pytorch) | 0.5.5, 13-07-2026 | MIT | Sí, con 17 paquetes más | `models/titans/` | Oráculo ya en uso, sin issue nueva |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) y fla-core | 0.5.2, 27-07-2026 | MIT | Sí, con einops | Memorias lineales y regla delta | Lectura de las referencias `naive` |
| [GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) | Repositorio, sin versión en PyPI | NVIDIA Source Code License-NC | No evaluada | Ninguna | Lectura, su licencia no comercial no encaja con un proyecto MIT |
| [TorchRL](https://github.com/pytorch/rl) | 0.14.0, 10-09-2026 | MIT | Sí, con tensordict, hoptorch, orjson y pyvers | `native/src/ppo_objectives.cpp` | Lectura |
| [TRL](https://github.com/huggingface/trl) | 1.14.0, 25-09-2026 (hay 1.15.0) | Apache-2.0 | Sí, con accelerate, datasets, aiohttp y 10 más | `native/src/group_relative.cpp` | Lectura |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | 1.2.0, 22-05-2023 | MIT | No, exige Python < 3.11 y huggingface-hub < 0.12 | PPO y DQN nativos | Lectura de sus scripts |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | 2.9.0, 15-06-2026 | MIT | Sí, sin otros paquetes | PPO nativo | Solo como dependencia de sb3-contrib |
| [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | 2.9.0, 15-06-2026 | MIT | Sí | `native/src/quantile_dqn.cpp` | Paridad en pruebas, [#497](https://github.com/GonxKZ/mars-titan/issues/497) |
| [EnvPool](https://github.com/sail-sg/envpool) | 1.2.7, 15-09-2026 | Apache-2.0 | Sí, con 11 paquetes y recursos de MuJoCo | Entorno nativo por lotes | Descartar |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), vectorización | 1.3.0, ya fijada (hay 1.4.0) | MIT | Ya en el lock | `simulation/environment.py` y entorno nativo | Descartar la ampliación |
| [neuralforecast](https://github.com/Nixtla/neuralforecast) | 3.2.2, 08-09-2026 (hay 3.3.0) | Apache-2.0 | Sí, con 26 paquetes, entre ellos Lightning, Ray y Optuna | `models/baselines/dlinear.py` | Lectura |
| [darts](https://github.com/unit8co/darts) | 0.47.0, 04-09-2026 | Apache-2.0 | Sí, con 14 paquetes y sin modelos de torch salvo el extra | Referencias de previsión | Descartar |
| [TensorBoard](https://github.com/tensorflow/tensorboard) | 2.21.0, 29-06-2026 | Apache-2.0 | Sí, con grpcio, protobuf y werkzeug | `learning_traces/recorder.py` | Descartar |
| [MLflow](https://github.com/mlflow/mlflow) | 3.16.1, 16-09-2026 (hay 3.17.0) | Apache-2.0 | Sí, con 51 paquetes | `learning_traces/recorder.py` | Descartar |
| [Aim](https://github.com/aimhubio/aim) | 3.29.1, 08-05-2025 | Apache-2.0 | No, exige filelock < 4 | `learning_traces/recorder.py` | Descartar |

Las licencias de la tabla son compatibles con usarlas como dependencia de un proyecto MIT, salvo kmedoids (GPL) y GatedDeltaNet-2 (no comercial), que se tratan aparte. Ninguna cubre los derechos del proyecto ni al revés.

## Adaptadores frente a PEFT

`models/predictive_adaptation.py` implementa con `torch.nn.utils.parametrize` las formas de tensor `residual`, `low_rank` (LoRA), `dora`, `gain_rows` ((IA)³ por filas) y `gain_columns`, y las de módulo `parallel_adapter` y `serial_adapter` con `BottleneckAdapter`. Se aplican a cualquier tensor del modelo, incluidos los pesos de `nn.GRU`. La [variedad de adaptadores](../engineering/adapter-variety.md) ya cita PEFT como implementación de referencia, pero ninguna prueba lo compara.

Lo que cubre PEFT 0.21.0, leído en su código:

- LoRA sobre `Embedding`, `Conv1d`, `Conv2d`, `Conv3d`, `MultiheadAttention`, `Linear` y `Conv1D` de transformers. No hay soporte de módulo para `nn.GRU` ni `nn.LSTM`.
- `target_parameters` adapta cualquier `nn.Parameter` con LoRA, pero rechaza DoRA, dropout, `lora_bias` y `fan_in_fan_out`. Es la única vía para los pesos de una GRU.
- DoRA con la norma por fila de salida separada del grafo, como propone la sección 4.3 de [Liu et al. (2024)](https://arxiv.org/abs/2402.09353). El proyecto no la separa y lo declara como desviación.
- (IA)³ sobre `Linear`, `Conv2d`, `Conv3d` y `Conv1D`. `bias="all"` se aproxima a BitFit y existe `ln_tuning`.
- Prefix tuning necesita los `past_key_values` de transformers. No hay adaptadores de cuello de botella. `_register_custom_module` permite registrar capas propias.

Sustituir el código propio por PEFT obligaría a mantener dos caminos (módulos y `target_parameters`), perdería DoRA y los cuellos de botella en la GRU y cambiaría las identidades de los brazos de [#364](https://github.com/GonxKZ/mars-titan/issues/364). Como referencia de paridad, en cambio, aporta una implementación externa y mantenida (nueve versiones en el último año).

Resultado preliminar con los mismos pesos copiados:

| Comparación | Salida | Gradientes |
| --- | --- | --- |
| LoRA en `nn.Linear` frente a `LowRankDelta`, FP64 | 8,9·10⁻¹⁶ | 5,7·10⁻¹⁴ |
| LoRA en `nn.Linear` frente a `LowRankDelta`, FP32 | 9,5·10⁻⁷ | 1,5·10⁻⁵ |
| DoRA frente a `WeightDecomposedDelta`, FP64 | 1,8·10⁻¹⁵ | Difieren (174 en FP64 y 83 en FP32) por la norma separada |
| (IA)³ con `feedforward_modules=[]` frente a `RowGain` y `SharedRowGain` | 8,9·10⁻¹⁶ en FP64, 4,8·10⁻⁷ en FP32 | No comparados |
| LoRA con `target_parameters=["gru.weight_ih_l0"]` frente a la parametrización propia | 0 exacto | No comparados |

En DoRA la magnitud de PEFT corresponde a la norma del padre más el desplazamiento que entrena el proyecto. Las cifras son diferencias absolutas máximas. Veredicto: referencia de paridad solo en pruebas, en un grupo de dependencias separado del entorno de ejecución ([#494](https://github.com/GonxKZ/mars-titan/issues/494)).

## Puntuaciones probabilísticas

El proyecto calcula la pérdida pinball en `models/quantile_head.py` y en `evaluation/forecast_scores.py`, y la puntuación de intervalo en `forecast_scores.py` y en la reimplementación independiente de `integrity/independent_scores.py`. No calcula CRPS, y [las métricas](metrics.md) aclaran que la pinball media no es el CRPS.

[scoringrules](https://frazane.github.io/scoringrules) 0.11.0 exige Python 3.12 o superior, depende solo de NumPy y SciPy y ofrece `quantile_score`, `interval_score`, `weighted_interval_score` y `crps_quantile`, con motores de NumPy, Numba, torch y JAX. La comprobación preliminar dio 3,5·10⁻¹⁸ entre `quantile_score` y `pinball_loss` y 0 exacto entre `interval_score` y `session_scores`. `crps_quantile` resultó igual a dos veces la pinball media sobre la rejilla de cuantiles (6,9·10⁻¹⁸). Es una aproximación del CRPS a partir de cuantiles ([Gneiting y Raftery, 2007](https://doi.org/10.1198/016214506000001437), [Bracher et al., 2021](https://doi.org/10.1371/journal.pcbi.1008618)), así que no añade una métrica nueva a la comparación.

properscoring 0.1 cubre lo mismo pero no publica versiones desde 2015. Veredicto: scoringrules como referencia de paridad en pruebas ([#495](https://github.com/GonxKZ/mars-titan/issues/495)), properscoring descartado.

## Calibración conformal

`calibration/conformal_quantiles.py` aplica la CQR de [Romano, Patterson y Candès (2019)](https://arxiv.org/abs/1905.03222) por grupo con el estadístico de orden ⌈(n + 1)(1 − α)⌉. `calibration/online_conformal.py` (PT2, [#454](https://github.com/GonxKZ/mars-titan/issues/454)) sigue el cuantil de la puntuación con el método de [Angelopoulos, Candès y Tibshirani (2023)](https://arxiv.org/abs/2307.16895), con cohortes que maduran con retraso y una cota propia para ese retraso.

- **MAPIE 1.5.0.** `ConformalizedQuantileRegressor`, con un estimador ya ajustado que devuelve cuantiles fijos y `symmetric_correction=True`, dio exactamente las mismas correcciones que `fit_conformal_quantiles` en coberturas de 0,8 y 0,95. Su `TimeSeriesRegressor` con `method="aci"` implementa [Gibbs y Candès (2021)](https://arxiv.org/abs/2106.00170), que actualiza el nivel α y no el cuantil. No sirve como paridad de PT2. Veredicto: referencia de paridad de la CQR estática ([#496](https://github.com/GonxKZ/mars-titan/issues/496)).
- **crepes 0.9.1.** Predictores conformales estáticos con categorías de Mondrian. Duplica lo que ya cubre MAPIE. Descartado.
- **puncc 0.9.3.** Exige NumPy < 2. Descartado.
- **[aangelopoulos/conformal-time-series](https://github.com/aangelopoulos/conformal-time-series)** (`b729c3f5`, MIT, sin cambios desde noviembre de 2023). Es el código del artículo de PT2, pero son scripts y no un paquete, y usan `np.infty`, que NumPy 2 eliminó. Queda como lectura de la regla sin retraso.

## Inferencia estadística de la comparación

`evaluation/paired_comparisons.py` remuestrea bloques circulares de días UTC con todos los activos y modelos juntos y da intervalos simultáneos por máximo estudentizado. `evaluation/financial_metrics.py` usa el mismo bootstrap para Sharpe, Sortino y drawdown. El proyecto no tiene Diebold-Mariano, SPA, MCS ni Reality Check, aunque la [revisión adversarial](adversarial-review.md) ya señala el problema de reutilizar datos de [White (2000)](https://doi.org/10.1111/1468-0262.00152) y MT-036 ([#36](https://github.com/GonxKZ/mars-titan/issues/36)) proponía `arch.bootstrap` como implementación contrastada.

`arch` 8.0.0 ofrece `IIDBootstrap`, `CircularBlockBootstrap`, `StationaryBootstrap`, `MovingBlockBootstrap`, `optimal_block_length` ([Politis y White, 2004](https://doi.org/10.1081/ETC-120028836), con la [corrección de 2009](https://doi.org/10.1080/07474930802459016)), `SPA` ([Hansen, 2005](https://doi.org/10.1198/073500105000000063)), `RealityCheck`, `StepM` ([Romano y Wolf, 2005](https://doi.org/10.1111/j.1468-0262.2005.00615.x)) y `MCS` ([Hansen, Lunde y Nason, 2011](https://doi.org/10.3982/ECTA5771)). `statsmodels` 0.15.0 añade `diebold_mariano_test` ([Diebold y Mariano, 1995](https://doi.org/10.1080/07350015.1995.10524599)) con la corrección de [Harvey, Leybourne y Newbold (1997)](https://doi.org/10.1016/S0169-2070(96)00719-4).

Comprobaciones hechas:

- Con el mismo `numpy.random.Generator`, `CircularBlockBootstrap(...).update_indices()` repetido R veces da exactamente los mismos índices que `circular_block_indices(rng, R, P, L)` en cinco combinaciones de días, bloque y réplicas (P = 250, 251, 1.000, 37 y 10). Los dos códigos sortean ⌈P/L⌉ inicios uniformes, recorren el bloque de forma circular y truncan a P.
- `SPA`, `MCS`, `optimal_block_length` y `diebold_mariano_test` funcionaron con NumPy 2.5.3 y pandas 3.0.6 sin avisos, y repetir con la misma semilla dio los mismos resultados.

Riesgos. La última versión de `arch` es del 21 de octubre de 2025, anterior a pandas 3.0.0, aunque el repositorio sigue recibiendo cambios y la comprobación no mostró problemas. Las pruebas de paridad deben detectar cualquier rotura al actualizar. `optimal_block_length` calculado con datos de evaluación sería una forma de ajustar el bootstrap con el futuro, así que solo se usa como diagnóstico sobre desarrollo. DM, SPA y MCS suponen estacionariedad de las diferencias de pérdida y no corrigen por sí mismos la selección entre muchas variantes que no se hayan conservado. Por eso se declaran como análisis secundarios antes de ver resultados, con el test final cerrado.

Veredicto: adoptar como dependencia del extra `research` ([#493](https://github.com/GonxKZ/mars-titan/issues/493)), sin sustituir el bootstrap propio, que sigue siendo el contraste principal.

## Métricas de cartera

`evaluation/financial_metrics.py` anualiza con 252 sesiones, usa tasa libre de riesgo nula, Sharpe con `ddof = 1` y Sortino con la raíz de la media de min(r, 0)². Las funciones de empyrical-reloaded siguen esas mismas convenciones, pero la versión 0.5.12 no resuelve con el lock porque exige `peewee < 3.17.4`, que no tiene ruedas. quantstats resuelve, pero arrastra yfinance y otros paquetes de red, y publicó cinco versiones entre el 26 y el 27 de septiembre. Las métricas propias ya tienen pruebas con ejemplos calculables a mano. Veredicto: empyrical como lectura de convenciones y quantstats descartado.

## Regla de Kalman de la memoria asociativa

La regla `kalman` de `memory/associative_memory.py` (PT3, [#455](https://github.com/GonxKZ/mars-titan/issues/455)) incorpora ruido de cohorte correlacionado y un recorte de Huber opcional. `tests/memory/test_kalman_associative_memory.py` ya la compara con la forma de covarianza densa y con el filtro de Kalman escalar. filterpy 1.4.5 es de 2018 y solo se distribuye como fuentes. El espacio de estados de statsmodels admite covarianzas de observación generales, pero está pensado para series con una dimensión de observación fija y estimación por máxima verosimilitud. Expresar cohortes de tamaño variable con un componente común exigiría rehacer el problema para obtener una comprobación que ya dan las dos formas cerradas. Veredicto: descartar las dos.

## Medoids de CM-v1

`cm/medoids.py` usa una construcción voraz con intercambios y tiene un oráculo de enumeración exacta para tamaños pequeños. `cm/anchored_medoids.py` añade los anclajes. kmedoids 0.5.5 implementa FasterPAM ([Schubert y Rousseeuw, 2021](https://doi.org/10.1016/j.is.2021.101804)) en Rust y resuelve sin otros paquetes. Serviría para comprobar, en tamaños donde la enumeración exacta no es viable, si la búsqueda propia llega al mismo coste o a uno menor. FasterPAM también es una búsqueda local, así que no certifica el óptimo, y una heurística de medoids no hereda el factor de aproximación de metric k-median. Su licencia GPL-3.0-or-later impide incluirlo en el entorno del proyecto o copiar su código. Veredicto: evaluar con medida en un entorno aislado si [#293](https://github.com/GonxKZ/mars-titan/issues/293) necesita tamaños mayores que los del oráculo. scikit-learn-extra 0.3.0 no tiene rueda para Python 3.12 y se descarta.

## Titans y memorias

La [revisión de implementaciones de Titans](titans-reference-implementations.md) se ha actualizado con estos hechos:

- La rueda de titans-pytorch 0.5.5 en PyPI tiene los mismos archivos de la memoria y del MAC, byte a byte, que el commit `1d40c445` que usa el arnés de paridad. El oráculo puede fijarse también por su versión de PyPI.
- `fla-core` 0.5.2 contiene la misma derivada de LayerNorm de `fla/ops/titans` que la revisión del 9 de octubre encontró incorrecta. Después de esa versión el archivo solo ha cambiado por la retirada de `head_first`.
- Las referencias `naive` de fla para la regla delta con puerta, Gated DeltaNet-2 y TTT están escritas en PyTorch y convierten todas las entradas a FP32, también las FP64, así que no sirven como oráculo FP64. Sirven como lectura de las ecuaciones. En los núcleos de Triton, `tl.dot` usa TF32 por defecto con entradas FP32, y fla solo fuerza IEEE en GPU sin TF32 y en algunos núcleos. No cumplen FP32 estricto sin fijar `TRITON_F32_DEFAULT=ieee` y comprobarlo.
- El código oficial de [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) existe, pero su licencia es no comercial y no encaja con un proyecto MIT, así que queda como lectura.
- Sigue sin haber código oficial de Titans, ATLAS, MIRAS ni Nested Learning.

La regla delta con puerta de fla decae el estado antes de calcular el residuo, `S ← exp(g)·S` y luego `S ← S + β·k(v − kᵀS)`. La regla `delta` de `memory/associative_memory.py` calcula el residuo con el estado anterior, `M ← ρ·M + η·k(v − kᵀM)`. Coinciden solo con retención 1, de modo que fla no puede usarse como referencia de la memoria con olvido sin reescribir la regla.

Veredicto: titans-pytorch sigue como oráculo con el arnés existente y sin vendorizar. fla y Gated DeltaNet-2 quedan como lectura. La adaptación financiera no cambia.

## Aprendizaje por refuerzo y posentrenamiento

La etapa nativa en C++ implementa PPO con penalización KL fija (`−ratio·A + β·KL(antigua ‖ nueva)` sobre seis acciones y sin GAE) en `native/src/ppo_objectives.cpp`, QR-DQN con Huber de parámetro κ en `native/src/quantile_dqn.cpp` y GRPO, Dr. GRPO, DAPO estático y GSPO en `native/src/group_relative.cpp`. KLPO, de Zhang et al. (2026), está en `models/klpo.py`. El oráculo FP64 en NumPy de `native/tests/rl_variety_reference.py` genera el fixture que comprueba `tests/simulation/test_rl_variety_reference.py`. Las bibliotecas externas se valoran como segunda referencia independiente, no como sustitutas.

- **sb3-contrib 2.9.0.** `quantile_huber_loss(current, target, cum_prob=None, sum_over_quantiles=True)` implementa la pérdida de [Dabney et al. (2018)](https://arxiv.org/abs/1710.10044) con κ = 1 fijo. Pasando `cum_prob` en FP64 coincidió con la fórmula del oráculo con 1,7·10⁻¹⁸. Sin `cum_prob` lo calcula en FP32. Veredicto: referencia de paridad para κ = 1 ([#497](https://github.com/GonxKZ/mars-titan/issues/497)). Stable-Baselines3 solo entra como su dependencia. Su PPO usa recorte en lugar de penalización KL y `set_random_seed` toca los generadores globales.
- **TorchRL 0.14.0.** Tiene GAE funcional, `ClipPPOLoss`, `KLPENPPOLoss` (KL de la política antigua a la nueva con β adaptativo, que queda fijo con `increment = decrement = 1`) y `DQNLoss`. Coincide en la dirección de la KL con la etapa nativa, pero exige tensordict y se distribuye como ruedas compiladas por plataforma cuya compatibilidad binaria con torch 2.14 no se ha comprobado. El oráculo NumPy ya da una referencia independiente para PPO. Veredicto: lectura.
- **TRL 1.14.0.** `GRPOConfig` ofrece `loss_type` con `grpo`, `dr_grpo`, `dapo` (por defecto), `bnpo`, `cispo`, `sapo`, `luspo` y `vespo`, `importance_sampling_level="sequence"` para GSPO y `scale_rewards` por grupo, lote o ninguno. Todo está dentro de un entrenador de modelos de lenguaje y no expone la pérdida como función aislada. Veredicto: lectura de las normalizaciones de cada variante.
- **CleanRL 1.2.0.** No instala con Python 3.12. Sus scripts de un solo archivo sirven como lectura.
- **EnvPool 1.2.7.** Solo ejecuta sus entornos integrados, que habría que ampliar en C++ dentro de su marco, y el entorno nativo del proyecto ya avanza por lotes. Descartado.
- **Vectorización de Gymnasium.** `SyncVectorEnv` y `AsyncVectorEnv` repetirían en Python lo que ya hace el entorno nativo. Gymnasium sigue fijado en 1.3.0 para `environments/prediction.py`, `episodes/prediction.py`, `simulation/environment.py` y `check_env`. Descartado ampliarla.

## Referencias de previsión

`models/baselines/dlinear.py` adapta el código de los autores de DLinear (Apache-2.0). neuralforecast 3.2.2 incluye DLinear, PatchTST e iTransformer, y su media móvil usa el mismo relleno por replicación de (k − 1)/2 a cada lado que el proyecto. Pero instalarlo trae 26 paquetes, entre ellos PyTorch Lightning, Ray y Optuna, y la paridad de DLinear ya está cubierta por el código original. darts 0.47.0 tiene DLinear, TSMixer y un PatchTST preentrenado como modelo fundacional (`patchtst_fm`), no tiene iTransformer y sus modelos neuronales necesitan otro extra. Ninguno añade brazos a la campaña. Veredicto: neuralforecast como lectura de PatchTST e iTransformer y darts descartado.

## Seguimiento de experimentos

`learning_traces/recorder.py` ([#467](https://github.com/GonxKZ/mars-titan/pull/467)) guarda estadísticas acotadas en partes Parquet reanudables, con un presupuesto de bytes y el cursor dentro del checkpoint. El [observatorio](../../src/mars_titan/observatory/) las lee. TensorBoard 2.21.0, MLflow 3.16.1 y Aim 3.29.1 guardarían otro estado fuera del checkpoint que no se recupera con el mismo cursor. MLflow trae 51 paquetes (servidor web, cliente de Docker y SDK de Databricks) y Aim no resuelve con el lock. Veredicto: descartar los tres.

## Issues derivadas

| Issue | Tipo | Biblioteca |
| --- | --- | --- |
| [#493](https://github.com/GonxKZ/mars-titan/issues/493) | Adopción | arch 8.0.0 y statsmodels 0.15.0 |
| [#494](https://github.com/GonxKZ/mars-titan/issues/494) | Paridad | PEFT 0.21.0 |
| [#495](https://github.com/GonxKZ/mars-titan/issues/495) | Paridad | scoringrules 0.11.0 |
| [#496](https://github.com/GonxKZ/mars-titan/issues/496) | Paridad | MAPIE 1.5.0 |
| [#497](https://github.com/GonxKZ/mars-titan/issues/497) | Paridad | sb3-contrib 2.9.0 |

Las referencias de paridad comparten el grupo de dependencias `reference`, fuera del entorno de ejecución, y la marca de pytest `external_reference`. Su omisión falla con `MARS_TITAN_REQUIRE_REFERENCE=1`, como ya hace `native_binding` con su variable.

## Límites

Las resoluciones de uv comprueban compatibilidad de versiones, no de comportamiento. La comprobación preliminar usó fixtures aleatorios pequeños en CPU y no cubre CUDA. Los repositorios se leyeron en las partes relevantes para cada pieza y puede haber detalles que solo se vean ejecutándolos. Nada de esta revisión mide rendimiento predictivo, entrena ni ejecuta pasos de optimizador, y el bloqueo de aprendizaje sigue vigente.
