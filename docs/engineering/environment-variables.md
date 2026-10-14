# Variables de entorno

Esta es la tabla única de las variables `MARS_TITAN_*` que lee el código del proyecto, más `MARS_ENCODER_INTEGRATION`. `scripts/check_repository.py` comprueba que cada variable leída en `src/`, `scripts/`, `tests/`, `benchmarks/`, `native/` o `configs/` aparece aquí y que ninguna de la tabla ha dejado de usarse. Las opciones de CMake con el mismo prefijo, como `MARS_TITAN_SANITIZER` o `MARS_TITAN_BUILD_EPISODIC_PYTHON`, son variables de la caché de CMake y se describen en [native/README.md](../../native/README.md).

Una variable vacía equivale a no declararla. Una ruta declarada que no existe nunca se trata como ausente: la prueba o la orden que la usa falla.

## Ejecución y aprendizaje

| Variable | La lee | Significado | Sin declarar |
| --- | --- | --- | --- |
| `MARS_TITAN_TRAINING_HOLD` | `training/learning_hold.py` y `native/src/learning_hold.cpp` | Ruta de la protección del aprendizaje. Bloquea los ajustes mientras no declare `training_allowed: true` ([protección del aprendizaje](../../tests/README.md#protección-del-aprendizaje)) | `~/.local/state/mars-titan/training-hold-2000.json`. Si el archivo no existe, no bloquea |
| `MARS_TITAN_INPUT_CACHE_MIB` | `training/reference_run.py` | Presupuesto en MiB de la caché de tablas del corpus, entre 1 y 4096 | Caché por defecto de `CorpusDataset` |
| `MARS_TITAN_DECODE_WORKERS` | `training/input_pipeline.py` | Hilos de decodificación del lector cronológico, de 0 a 16 | 0, la ruta secuencial |
| `MARS_TITAN_PREFETCH_BATCHES` | `training/input_pipeline.py` | Lotes adelantados por el lector, de 0 a 64 | 0 |
| `MARS_TITAN_GROUP_CACHE_MIB` | `training/input_pipeline.py` | Límite en MiB de la caché de grupos decodificados, hasta 8192 | 4096 |
| `MARS_TITAN_DIGEST_CACHE` | `training/corpus_inputs.py` | Ruta absoluta de un archivo de huellas compartido entre procesos de una campaña | Cada proceso calcula sus huellas |
| `MARS_TITAN_SHARED_INDICES` | `training/window_indices.py` | Ruta absoluta de la raíz donde los trabajos de una campaña comparten por vista sus índices de observaciones. La fijan la campaña y el recorrido por ventanas, que la borran por vista al cerrar cada ventana | Cada trabajo construye sus índices en `indices` de su carpeta |
| `MARS_TITAN_NVTX` | `nvtx_ranges.py` | `1` abre rangos NVTX por fase para Nsight Systems. No cambia las salidas | `0`, sin rangos |
| `MARS_TITAN_HARDWARE_PROFILE` | `hardware/hardware_profiles.py` | Perfil de [configs/hardware](../../configs/hardware) que nombra la máquina, como `rtx4070-laptop`. Lo lee `declared_profile()` | `declared_profile()` se detiene y pide declararlo |
| `MARS_TITAN_SLOT_VRAM_MIB` | `training/campaign_slots.py` la escribe para cada proceso hijo | VRAM en MiB asignada a la plaza de la campaña | No se lee fuera de la campaña |

## Binarios nativos

`make native-build` deja estos artefactos en `build/native/`. Los presets y la compilación se describen en [tests/README.md](../../tests/README.md#binarios-nativos) y [native/README.md](../../native/README.md).

| Variable | Artefacto | Preset | Sin declarar |
| --- | --- | --- | --- |
| `MARS_TITAN_EPISODIC_NATIVE` | Enlace episódico de Python, `_episodic_native*.so` | `native-candidate-cuda` | Se omiten el banco episódico, la GRU candidata, MARS-TITAN, CM-v1 y sus sesiones |
| `MARS_TITAN_NATIVE_LIBRARY` | `libmars_titan_simulation.so` | `native-release` | Se usa la de `build/native/native-release` si existe. Si no, se omiten las variantes nativas |
| `MARS_TITAN_SIM_EXECUTABLE` | `mars-titan-sim` | `native-release` | Igual que la biblioteca |
| `MARS_TITAN_PPO_EXECUTABLE` | `mars-titan-ppo` | `native-ppo-release` | La etapa de políticas usa el de `build/native/native-ppo-release`. Algunas pruebas exigen declararlo |
| `MARS_TITAN_KLPO_EXECUTABLE` | `mars-titan-klpo` | `native-ppo-release` | Se usa el de `build/native/native-ppo-release` si existe |
| `MARS_TITAN_CLANG_TIDY` | `clang-tidy` para `tests/native/test_diagnostics.py` | Ninguno | Se busca en `PATH` y, si falta, se omite la prueba |

## Pruebas

| Variable | Significado | Sin declarar |
| --- | --- | --- |
| `MARS_TITAN_REQUIRE_NATIVE` | `1` exige los cinco binarios y que el enlace episódico cargue antes de recoger pruebas. Una prueba `native_binding` omitida cuenta como fallo | `0` |
| `MARS_TITAN_REQUIRE_CUDA` | `1` detiene la sesión si CUDA no está visible | `0` |
| `MARS_TITAN_REQUIRE_REFERENCE` | `1` exige el grupo `reference` con sus versiones exactas. Una prueba `external_reference` omitida cuenta como fallo | `0` |
| `MARS_TITAN_CUDA_INTEGRATION` | `1` activa las integraciones CUDA que aplican pasos de optimizador. Solo procede fuera del bloqueo de aprendizaje | Se omiten |
| `MARS_ENCODER_INTEGRATION` | `1` activa la extracción con los codificadores reales en `cuda:0` | Se omite |
| `MARS_TITAN_UNADJUSTED_EDITION` | Ruta de una edición de precios sin ajustar para las pruebas de humo, que solo la leen | Se omiten |
| `MARS_TITAN_LISTING_STATUS` | Ruta del `listing-status.json` de esa edición, que las mismas pruebas leen junto a ella | Se omiten |

Las comprobaciones CUDA que se ejecutan de forma explícita (`tests/**/cuda_*_check.py` y `tests/training/candidate_memory_check.py`) eligen el dispositivo y el recibo con variables propias. Con `cpu` ensayan la misma lógica sin GPU y no acreditan CUDA.

| Variable | Comprobación | Sin declarar |
| --- | --- | --- |
| `MARS_TITAN_GRU_CHECK_DEVICE`, `MARS_TITAN_GRU_CHECK_REPORT` | `tests/memory/cuda_gru_session_check.py` | `cuda:0`, sin recibo |
| `MARS_TITAN_M3_CHECK_DEVICE` | `tests/memory/cuda_mature_error_check.py` | `cuda:0` |
| `MARS_TITAN_B6_CHECK_REPORT` | `tests/memory/cuda_mars_titan_correction_check.py` | Sin recibo |
| `MARS_TITAN_EPISODIC_READOUT_CHECK_REPORT` | `tests/models/titans/cuda_episodic_readout_check.py` | Sin recibo |
| `MARS_TITAN_REGENERATION_CHECK_DEVICE`, `MARS_TITAN_REGENERATION_CHECK_REPORT` | `tests/training/cuda_regeneration_check.py` | `cuda:0`, sin recibo |
| `MARS_TITAN_CANDIDATE_RUN_CHECK_DEVICE`, `MARS_TITAN_CANDIDATE_RUN_CHECK_REPORT` | `tests/training/cuda_candidate_run_check.py` | `cuda:0`, sin recibo |
| `MARS_TITAN_MARS_RUN_CHECK_DEVICE`, `MARS_TITAN_MARS_RUN_CHECK_REPORT` | `tests/training/cuda_mars_titan_run_check.py` | `cuda:0`, sin recibo |
| `MARS_TITAN_CM_CONTROL_CHECK_DEVICE`, `MARS_TITAN_CM_CONTROL_CHECK_REPORT` | `tests/training/cuda_cm_v1_control_check.py` | `cuda:0`, sin recibo |
| `MARS_TITAN_CANDIDATE_MEMORY_CHECK_DEVICE`, `MARS_TITAN_CANDIDATE_MEMORY_CHECK_ASSETS`, `MARS_TITAN_CANDIDATE_MEMORY_CHECK_REPORT` | `tests/training/candidate_memory_check.py` | `cpu`, activos `128,256`, sin recibo |
| `MARS_TITAN_CANDIDATE_ADAPTERS_CHECK_DEVICE`, `MARS_TITAN_CANDIDATE_ADAPTERS_CHECK_REPORT` | `tests/posttraining/cuda_candidate_adapters_check.py` | `cuda:0`, sin recibo |
| `MARS_TITAN_READOUT_ADAPTERS_CHECK_DEVICE`, `MARS_TITAN_READOUT_ADAPTERS_CHECK_REPORT` | `tests/posttraining/cuda_readout_adapters_check.py` | `cuda:0`, sin recibo |
| `MARS_TITAN_TITANS_ADAPTERS_CHECK_DEVICE` | `tests/posttraining/cuda_titans_adapters_check.py` | `cuda:0` |

## Servicios de usuario

Las unidades de [configs/systemd](../../configs/systemd) leen sus valores de un archivo `.env` local, fuera del repositorio.

| Variable | Unidad | Significado |
| --- | --- | --- |
| `MARS_TITAN_RUNTIME`, `MARS_TITAN_SCENARIOS`, `MARS_TITAN_BINARY`, `MARS_TITAN_CAMPAIGN_CONFIG`, `MARS_TITAN_OUTPUT` | `mars-titan-adaptive-rl.service` | Runtime congelado, catálogo de escenarios, binario, configuración y salida de la campaña RL ([guía](adaptive-rl.md)) |
| `MARS_TITAN_GPU_STATE`, `MARS_TITAN_CAMPAIGN_SUMMARY`, `MARS_TITAN_PREPARATION_STATE`, `MARS_TITAN_ACTIVITY_ARGS`, `MARS_TITAN_CAMPAIGN_SERVICE` | `mars-titan-campaign-health@.service` | Estado de la GPU, resumen, preparación, actividades opcionales y servicio observado, por defecto `mars-titan-scientific-campaign.service` ([guía](campaign-health.md)) |
| `MARS_TITAN_REVIEW_PYTHON` | `mars-titan-prediction-review@.service` | Entorno uv con el que se revisan las predicciones ([guía](prediction-review.md)) |
