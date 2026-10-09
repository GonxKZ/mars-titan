# Pruebas

Las pruebas se agrupan por la responsabilidad que comprueban:

| Directorio | Alcance |
| --- | --- |
| `tests/data/` | Disponibilidad temporal, inventario, preparación y materialización de datos |
| `tests/models/` | Entradas comunes, referencias predictivas, persistencia y reanudación |
| `tests/native/` | Configuración de CMake, avisos, análisis estático y sanitizadores |
| `tests/tooling/` | Biblioteca documental, fuentes públicas, exportador y observatorio |

Las pruebas de herramientas no acceden a la red. Comprueban identificadores, metadatos, integridad, formatos, límites de descarga y conservación de archivos anteriores. Las pruebas científicas usan datos sintéticos o artefactos locales controlados y distinguen las medidas de coste de los resultados predictivos.

Desde la raíz del repositorio se puede ejecutar el conjunto completo:

```bash
uv run --locked pytest
```

Las pruebas que requieren CUDA comprueban su disponibilidad y no cambian a CPU de forma silenciosa. Para ejecutar solo las comprobaciones compatibles con CPU se deben seleccionar los módulos correspondientes, por ejemplo:

```bash
uv run --locked pytest tests/tooling tests/native
```

Cada prueba protege una propiedad concreta. La cobertura y las pruebas de mutación ayudan a localizar lógica poco comprobada, pero no sustituyen los casos de comportamiento ni acreditan por sí solas la reproducibilidad de un entrenamiento.

## Protección del aprendizaje

La protección local `~/.local/state/mars-titan/training-hold-2000.json`, o la ruta indicada en `MARS_TITAN_TRAINING_HOLD`, bloquea el aprendizaje mientras no declare `training_allowed: true`. Un archivo ausente no bloquea y un valor que no sea booleano detiene la ejecución. Python la lee con `learning_blocked()` y `require_learning_allowed()` en `src/mars_titan/training/learning_hold.py`. Los ejecutables nativos usan la misma variable y la misma ruta mediante `native/src/learning_hold.cpp`.

### Puntos de entrada protegidos

`require_learning_allowed()` lanza `LearningHoldError` al inicio de cada punto de entrada que ajusta parámetros, antes de abrir fuentes, crear salidas o escribir checkpoints. Esta excepción no hereda de `RuntimeError`, así que los lanzadores no la registran como un fallo ordinario del intento.

| Ámbito | Puntos de entrada |
| --- | --- |
| Referencias neuronales | `run_reference_case`, `run_search`, `run_temporal_search`, `run_reference_campaign` y `training/real_campaign.run_campaign` |
| Referencias tabulares | `run_tabular_reference`, `run_external_reference`, `run_tabular_search` y `baseline_queue.run_queue` |
| Campaña con máscaras | `masked_campaign.run_campaign`, al empezar y antes de cada trabajo, y `candidate_walk_forward.fit_window` y `carry_window` |
| Adaptador predictivo | `run_predictive_case`, `run_predictive_study` y `klpo_queue.run_queue` |
| Postentrenamiento | `posttraining/run.run_case`, `posttraining/queue.run_queue`, `run_completion` y las etapas tabular y de postentrenamiento de la compleción |
| Sondas y mediciones | `train_budget_grid`, `run_temporal_probe`, `profile_case`, `run_reference_probe` y `models/baselines/campaign.run_campaign` |
| Primitivas de ajuste | `fit_ridge_blocks`, `fit_boosting_batches`, `fit_external_boosting` y `fit_hmm` |
| Simulación y refuerzo | `FinancialTrainer.run`, `simulation/campaign.run_campaign`, `run_adaptive_campaign` y `prepare_adaptation_scenarios` con `fit_markov=True` |
| Titans-MAC | `titans_walk_forward.run_titans_window`, `carry_titans` y `ChronologicalTrainer.run` cuando recibe un optimizador de `torch.optim` |
| MARS-TITAN | `mars_titan_walk_forward.run_mars_titan_window`, `carry_mars_titan` y `ReadoutTrainer.run` con cualquier optimizador |
| CM-v1 | `cm_v1_factorial.run_cm_v1_core_window`, `run_cm_v1_window` y `carry_cm_v1`, antes de leer la declaración o las fuentes |
| Scripts | `run_native_ppo.py` en modo de entrenamiento, `benchmark_native_ppo.py`, `benchmark_adaptive_rl.py`, `run_financial_comparators.py` y `run_titans_walk_forward.py` |
| Ejecutables nativos | `mars-titan-ppo` en modo de entrenamiento, después de validar argumentos y antes de leer fuentes, y `mars-titan-adapter-control` |

`ChronologicalTrainer.run` en `src/mars_titan/training/financial_run.py` solo aplica la protección con un optimizador de `torch.optim`. Así sus pruebas recorren el bucle con un optimizador propio que registra llamadas sin modificar pesos. `CandidateChronologicalTrainer.run` en `src/mars_titan/training/candidate_run.py` llama a `require_learning_allowed()` con cualquier optimizador, y sus pruebas usan `learning_doubles` porque el suyo solo registra gradientes. `ReadoutTrainer.run` en `src/mars_titan/training/mars_titan_run.py` también la aplica con cualquier optimizador y sus pruebas usan `learning_doubles`. `carry_window`, `carry_titans`, `carry_mars_titan` y `carry_cm_v1` también comprueban la protección aunque no ajustan, porque producen las predicciones de una ventana de la campaña.

### Lo que no se bloquea

La preparación de objetivos (`prepare_corpus_targets`), la codificación (`encode_corpus`), la generación de escenarios sin HMM, `temporal_search --check` y las órdenes `check`, `prepare`, `sources`, `posttraining check` y `throughput` de `run_masked_campaign.py` siguen permitidas. `throughput` recorre forward y backward sin crear optimizadores de PyTorch: Titans-MAC, la GRU candidata, los lectores de MARS-TITAN y los núcleos y lectores de CM-v1 llegan hasta el paso con un optimizador de la medición que no modifica pesos, se exige que los pesos no cambien, también los del padre congelado de cada lector, y su gancho rechaza cualquier paso. `posttraining run` se detiene con la protección como `run_stage`. `ablation run` no ajusta nada, pero la etapa de ablación de modalidades se detiene con la protección antes de crear salidas y antes de cada trabajo pendiente, porque es una evaluación científica, y su gancho rechaza cualquier paso de optimizador. `ablation check` y `ablation sources` siguen permitidas. También la inferencia congelada, como la auditoría de `run_native_ppo.py --audit-run` y su ruta en `mars-titan-ppo` sobre fuentes sintéticas o la etapa de evaluación de la compleción. La evaluación de una política sobre cintas reconstruidas (esquema 4 de `mars-titan-ppo` y `mars-titan-klpo`) sí se detiene con la protección antes de leer cintas o crear salidas, porque usa el histórico real. Las estadísticas que describen datos o residuos sin ajustar un modelo (`fit_standardizer`, `fit_normalization`, `ActionGrid.fit`, `fit_volatility` y el estadístico de orden de `fit_conformal_quantiles`) tampoco se bloquean. Esta protección actúa sobre el ajuste de parámetros y no impide por sí sola la evaluación científica ni el análisis de campañas (`scripts/finish_real_campaign.py` y `src/mars_titan/evaluation/`).

### Comportamiento en las pruebas

- `tests/conftest.py` registra un gancho global previo a cada paso de `torch.optim`. La prueba que intente ese paso se omite antes de modificar pesos o estados del optimizador. Sin protección, o con `training_allowed: true`, el gancho no se instala.
- El mismo archivo convierte `LearningHoldError` en una omisión con su motivo, tanto en la preparación como en la llamada. Mientras rige la protección, las pruebas que alcanzan un punto de entrada protegido se omiten antes de ajustar.
- El fixture `learning_doubles` apunta `MARS_TITAN_TRAINING_HOLD` a una protección temporal permitida. Lo usan pruebas cuyo aprendizaje está sustituido por binarios o ejecutores simulados, como el lanzador PPO nativo, la campaña adaptativa, la campaña con máscaras y la retención tabular con un estimador fijo. `tests/posttraining/conftest.py` también lo aplica mientras rige la protección, porque el postentrenamiento solo ajusta mediante `torch.optim`. El gancho de PyTorch sigue activo en todas ellas.
- `tests/training/test_learning_hold_guards.py` recorre cada punto de entrada con protecciones temporales. Bloqueado, falla antes del doble que sustituye el ajuste y sin crear salidas. Permitido o sin protección, llega a ese doble. También comprueba que la preparación de objetivos, la codificación, `temporal_search --check` y la auditoría congelada no se bloquean. Las pruebas del binario real necesitan `MARS_TITAN_PPO_EXECUTABLE`.
- `tests/posttraining/test_learning_hold_entrypoints.py` comprueba lo mismo en los cinco puntos de entrada del postentrenamiento y que la etapa de evaluación congelada no se bloquea.
- `tests/simulation/test_native_ppo_runner.py` se omite mientras rige la protección porque entrena con el binario real.
- En CTest, `learning_hold` comprueba la lectura nativa y `adapter_control_cli_hold` el rechazo del control de adaptadores con una protección temporal. `adapter_control_cli` se omite mediante el mensaje del bloqueo.

Siguen requiriendo una selección explícita las pruebas que ajustan sin pasar por un punto de entrada protegido: `tests/models/test_boosting_selection.py` y `tests/training/test_external_convergence.py` (llaman a `xgb.train`), `tests/models/test_ridge_normal_equations.py` (resuelve el sistema normal de la ridge) y las pruebas de CTest que usan directamente las bibliotecas de PPO, KLPO y adaptadores, por ejemplo `ppo_training`, `ppo_policy` y `adapter_control`.
