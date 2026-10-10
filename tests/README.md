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

Las pruebas que requieren CUDA, un binario nativo o Node.js comprueban su disponibilidad y, si falta, se omiten con su motivo. Nunca cambian a CPU de forma silenciosa. La [suite completa en condiciones reales](#suite-completa-en-condiciones-reales) describe cómo ejecutarlas todas. Para ejecutar solo las comprobaciones compatibles con CPU se deben seleccionar los módulos correspondientes, por ejemplo:

```bash
uv run --locked pytest tests/tooling tests/native
```

Las pruebas que necesitan el enlace episódico nativo leen su ruta en `MARS_TITAN_EPISODIC_NATIVE` y, sin ella, se omiten con su motivo para que la suite CPU siga funcionando. Las paridades de MARS-TITAN llevan además la marca `native_binding`. La comprobación local debe exigirlas con el [modo estricto](#modo-estricto), que comprueba los binarios y carga el enlace antes de recoger ninguna prueba. Con ese modo, cualquier omisión de una prueba marcada cuenta como fallo:

```bash
MARS_TITAN_REQUIRE_NATIVE=1 \
  uv run --locked pytest tests/memory/test_mars_titan_session_parity.py tests/memory/test_mars_titan_variant.py
```

La orden necesita los cinco binarios declarados como en la [suite completa](#binarios-nativos). Si falta alguno, si una ruta no existe o si el enlace episódico no carga, el modo estricto termina con un error de uso. Con `MARS_TITAN_REQUIRE_NATIVE=0` o sin la variable, el comportamiento es el de la suite CPU.

Las paridades con bibliotecas externas (PEFT, scoringrules, MAPIE y sb3-contrib) llevan la marca `external_reference` y necesitan el grupo de dependencias `reference`, que no forma parte del entorno de ejecución. Sin el grupo se omiten con su motivo. `MARS_TITAN_REQUIRE_REFERENCE=1` exige el grupo con las versiones exactas de `pyproject.toml` antes de recoger pruebas y convierte en fallo cualquier omisión de una prueba marcada:

```bash
uv sync --locked --group reference
MARS_TITAN_REQUIRE_REFERENCE=1 uv run --locked pytest -q -rs -m external_reference
```

Una prueba con las dos marcas, como la que compara el núcleo QR-DQN con sb3-contrib, solo falla por omitirse si se exigen las dos cosas. Esa prueba también necesita `rl_variety_tests` junto a `mars-titan-ppo`, que la comprobación del inicio no exige, así que con `MARS_TITAN_REQUIRE_NATIVE=1` la ausencia de ese ejecutable la hace fallar aunque no se exija el grupo.

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

La preparación de objetivos (`prepare_corpus_targets`), la codificación (`encode_corpus`), la generación de escenarios sin HMM, `temporal_search --check` y las órdenes `check`, `prepare`, `views`, `sources`, `posttraining check`, `throughput`, `storage`, `budget` y `schedule` de `run_masked_campaign.py` siguen permitidas. `budget` solo lee recuentos declarados, manifiestos de vistas o la decisión, la maduración y la presencia de los objetivos para contar filas, y `schedule` no lee datos. `storage` solo lee manifiestos de vistas y escribe tablas sintéticas en un directorio temporal. `throughput` recorre forward y backward sin crear optimizadores de PyTorch: Titans-MAC, la GRU candidata, los lectores de MARS-TITAN y los núcleos y lectores de CM-v1 llegan hasta el paso con un optimizador de la medición que no modifica pesos, se exige que los pesos no cambien, también los del padre congelado de cada lector, y su gancho rechaza cualquier paso. `posttraining run` se detiene con la protección como `run_stage`. `regenerate` se detiene antes de abrir fuentes, porque predice ventanas reales de la campaña aunque no ajuste nada, y `rolling` se detiene en la fase base con la campaña. `ablation run` no ajusta nada, pero la etapa de ablación de modalidades se detiene con la protección antes de crear salidas y antes de cada trabajo pendiente, porque es una evaluación científica, y su gancho rechaza cualquier paso de optimizador. `ablation check` y `ablation sources` siguen permitidas. También la inferencia congelada, como la auditoría de `run_native_ppo.py --audit-run` y su ruta en `mars-titan-ppo` sobre fuentes sintéticas o la etapa de evaluación de la compleción. La evaluación de una política sobre cintas reconstruidas (esquema 4 de `mars-titan-ppo` y `mars-titan-klpo`) sí se detiene con la protección antes de leer cintas o crear salidas, porque usa el histórico real. Las estadísticas que describen datos o residuos sin ajustar un modelo (`fit_standardizer`, `fit_normalization`, `ActionGrid.fit`, `fit_volatility` y el estadístico de orden de `fit_conformal_quantiles`) tampoco se bloquean. Esta protección actúa sobre el ajuste de parámetros y no impide por sí sola la evaluación científica ni el análisis de campañas (`scripts/finish_real_campaign.py` y `src/mars_titan/evaluation/`).

### Comportamiento en las pruebas

- `tests/conftest.py` registra un gancho global previo a cada paso de `torch.optim`. La prueba que intente ese paso se omite antes de modificar pesos o estados del optimizador. Sin protección, o con `training_allowed: true`, el gancho no se instala.
- El mismo archivo convierte `LearningHoldError` en una omisión con su motivo, tanto en la preparación como en la llamada. Mientras rige la protección, las pruebas que alcanzan un punto de entrada protegido se omiten antes de ajustar.
- El fixture `learning_doubles` apunta `MARS_TITAN_TRAINING_HOLD` a una protección temporal permitida. Lo usan pruebas cuyo aprendizaje está sustituido por binarios o ejecutores simulados, como el lanzador PPO nativo, la campaña adaptativa, la campaña con máscaras y la retención tabular con un estimador fijo. `tests/posttraining/conftest.py` también lo aplica mientras rige la protección, porque el postentrenamiento solo ajusta mediante `torch.optim`. El gancho de PyTorch sigue activo en todas ellas.
- `tests/training/test_learning_hold_guards.py` recorre cada punto de entrada con protecciones temporales. Bloqueado, falla antes del doble que sustituye el ajuste y sin crear salidas. Permitido o sin protección, llega a ese doble. También comprueba que la preparación de objetivos, la codificación, `temporal_search --check` y la auditoría congelada no se bloquean. Las pruebas del binario real necesitan `MARS_TITAN_PPO_EXECUTABLE`.
- `tests/posttraining/test_learning_hold_entrypoints.py` comprueba lo mismo en los cinco puntos de entrada del postentrenamiento y que la etapa de evaluación congelada no se bloquea.
- `tests/simulation/test_native_ppo_runner.py` se omite mientras rige la protección porque entrena con el binario real.
- En CTest, `learning_hold` comprueba la lectura nativa y `adapter_control_cli_hold` el rechazo del control de adaptadores con una protección temporal. `adapter_control_cli` se omite mediante el mensaje del bloqueo.

Siguen requiriendo una selección explícita las pruebas que ajustan sin pasar por un punto de entrada protegido: `tests/models/test_boosting_selection.py` y `tests/training/test_external_convergence.py` (llaman a `xgb.train`), `tests/models/test_ridge_normal_equations.py` (resuelve el sistema normal de la ridge) y las pruebas de CTest que usan directamente las bibliotecas de PPO, KLPO y adaptadores, por ejemplo `ppo_training`, `ppo_policy` y `adapter_control`.

## Suite completa en condiciones reales

La suite general omite con su motivo las pruebas que necesitan un binario nativo, CUDA, Node.js o una activación explícita. Esa omisión es correcta en una ejecución parcial, pero una verificación completa debe compilar los binarios, declararlos y comprobar CUDA. Los pasos siguientes repiten la verificación de la issue #374 y no ejecutan ningún ajuste: el gancho de `tests/conftest.py` y las protecciones de los puntos de entrada siguen activos.

### Entorno

Se prepara un entorno propio con todos los extras, sin tocar otros entornos del equipo:

```bash
uv sync --locked --all-extras --group reference
uv run --locked python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

La versión de PyTorch debe terminar en `+cu130`. `uv run --locked` no retira los extras ya instalados. Sin el extra `data` falta `lxml` y las pruebas macro fallan al construir el árbol HTML. Node.js es opcional y solo lo usa el validador de la web del observatorio.

### Binarios nativos

Se compilan desde `native/` y desde el mismo commit que se prueba, con dos trabajos:

```bash
cmake --preset native-release -DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON
cmake --build --preset native-release -j 2
cmake --preset native-ppo-release
cmake --build --preset native-ppo-release -j 2
cmake --preset native-candidate-cuda -DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON -DMARS_TITAN_BUILD_SIMULATION=ON
cmake --build --preset native-candidate-cuda -j 2
```

| Variable | Artefacto | Sin ella |
| --- | --- | --- |
| `MARS_TITAN_EPISODIC_NATIVE` | `build/native/native-candidate-cuda/_episodic_native.cpython-312-x86_64-linux-gnu.so` | Se omiten el banco episódico, la GRU candidata, MARS-TITAN, CM-v1 y sus sesiones |
| `MARS_TITAN_NATIVE_LIBRARY` | `build/native/native-release/libmars_titan_simulation.so` | Se usa la de `native-release` si existe, si no se omiten las variantes nativas |
| `MARS_TITAN_SIM_EXECUTABLE` | `build/native/native-release/mars-titan-sim` | Igual que la biblioteca |
| `MARS_TITAN_PPO_EXECUTABLE` | `build/native/native-ppo-release/mars-titan-ppo` | Se omiten las pruebas del binario PPO real |
| `MARS_TITAN_KLPO_EXECUTABLE` | `build/native/native-ppo-release/mars-titan-klpo` | Se usa el de `native-ppo-release` si existe |
| `MARS_TITAN_UNADJUSTED_EDITION` | Edición de precios sin ajustar, solo lectura | Se omiten las pruebas de humo con la edición real |

El enlace episódico debe ser el de `native-candidate-cuda`. El de `native-release` no incluye el candidato GRU y deja sin ejecutar las pruebas de la GRU en sesiones financieras. Una ruta declarada que no existe nunca se trata como ausente y hace fallar la prueba. Desde la raíz del repositorio:

```bash
B=$PWD/build/native
export MARS_TITAN_EPISODIC_NATIVE=$B/native-candidate-cuda/_episodic_native.cpython-312-x86_64-linux-gnu.so
export MARS_TITAN_NATIVE_LIBRARY=$B/native-release/libmars_titan_simulation.so
export MARS_TITAN_SIM_EXECUTABLE=$B/native-release/mars-titan-sim
export MARS_TITAN_PPO_EXECUTABLE=$B/native-ppo-release/mars-titan-ppo
export MARS_TITAN_KLPO_EXECUTABLE=$B/native-ppo-release/mars-titan-klpo
```

### Modo estricto

`MARS_TITAN_REQUIRE_NATIVE=1` detiene la sesión antes de recoger pruebas si falta declarar alguno de los cinco binarios, si su ruta no existe o si el enlace episódico no carga, por ejemplo porque se compiló con otro PyTorch. Durante la sesión, una prueba marcada con `native_binding` que se omite cuenta como fallo. `MARS_TITAN_REQUIRE_REFERENCE=1` hace lo mismo con el grupo `reference` y la marca `external_reference`. `MARS_TITAN_REQUIRE_CUDA=1` detiene la sesión si CUDA no está visible. Así un binario sin compilar o una GPU no disponible no se confunden con omisiones esperadas. Fuera de este modo las pruebas que dependen de CUDA usan `requires_cuda` de `tests/suite_support.py` y se omiten con su motivo, sin pasar nunca a CPU.

### Parte CPU

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 MARS_TITAN_REQUIRE_NATIVE=1 MARS_TITAN_REQUIRE_REFERENCE=1
for part in data models training simulation memory posttraining evaluation environments episodes calibration cm native tooling; do
  uv run --locked pytest -q -rs tests/$part \
    --ignore=tests/models/test_boosting_selection.py \
    --ignore=tests/training/test_external_convergence.py \
    --ignore=tests/models/test_ridge_normal_equations.py
done
uv run --locked pytest -q -rs tests/test_suite_support.py
```

Las tres exclusiones son las pruebas que ajustan sin pasar por un punto de entrada protegido, descritas al final de la sección anterior. Cada carpeta en su propio proceso acota la memoria al pico de la más pesada. En la verificación del 9 de octubre de 2026, `tests/training` llegó a unos 5,7 GiB de memoria residente y el resto de carpetas se quedó por debajo de 1,1 GiB. Si otras sesiones de pytest comparten el directorio temporal, cada parte necesita su propio `--basetemp`, porque pytest borra los `pytest-N` antiguos de las demás sesiones y las pruebas en curso pierden sus archivos.

### Parte CUDA

Las pruebas CUDA son las que la parte CPU omite por falta de CUDA, que se reconocen por su motivo en la salida de `-rs`. Se repiten con la GPU visible, por tandas cortas de pocos archivos y con la GPU reservada en exclusiva durante cada tanda si otras cargas la comparten:

```bash
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_REQUIRE_CUDA=1 MARS_TITAN_REQUIRE_NATIVE=1
uv run --locked pytest -q -rs <archivos o identificadores de la tanda>
```

Las pruebas que aplican pasos de optimizador en CUDA se omiten igualmente por la protección. Las que exigen `MARS_TITAN_CUDA_INTEGRATION=1` aplican esos pasos y quedan fuera mientras rija el bloqueo de aprendizaje. `MARS_ENCODER_INTEGRATION=1` activa la extracción con los codificadores reales y es una comprobación aparte. Las ocho pruebas de escenarios con HMM necesitan `hmmlearn==0.3.3`, que no forma parte de `uv.lock` y se añade con `uv run --with hmmlearn==0.3.3`. Como ajustan el HMM, también esperan al levantamiento del bloqueo.
