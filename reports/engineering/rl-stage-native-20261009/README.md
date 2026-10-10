# Etapa RL nativa sin aprendizaje: CUDA, perfil y caudal (9 de octubre de 2026)

Este informe mide el recorrido de las políticas de la etapa RL sobre cintas reconstruidas de la edición sin ajustar, con el binario compilado con el backend CUDA de LibTorch, en `cuda:0` y en CPU (issue #365). El bloqueo de aprendizaje siguió vigente durante todo el trabajo. Ninguna medida ejecuta Adam, `update_ready` ni `terminal_step`, y las tres pruebas CTest que aplican pasos de Adam (`ppo_policy`, `ppo_training` y `ppo_gru_packing`) no se lanzaron. Las puntuaciones de las cintas son sintéticas, así que las cifras describen coste de cálculo y nunca calidad de una política.

El [resumen estructurado](summary.json) recoge entorno, órdenes, medianas, huellas de paridad, concurrencia, memoria y proyección.

## Entorno

| Elemento | Valor |
| --- | --- |
| CPU | AMD Ryzen 9 8945HS, 8 núcleos y 16 hilos, AVX-512 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (Max-Q), 8.188 MiB, capacidad 8.9, driver 595.91.07 |
| RAM | 30 GiB visibles |
| Perfil de energía | `power-saver`, sin cambios |
| Sistema | Linux 7.0.0-38 |
| LibTorch | PyTorch 2.14.0+cu130 (CUDA 13.0, cuDNN 9.24) |
| Compilador | Clang 21.1.8, CMake 4.2.3, Ninja, `-j 2`, Release |
| Perfiladores | Kineto de LibTorch, Nsight Systems 2026.3.2 |
| Hilos y precisión | un hilo de ATen e interop, algoritmos deterministas, FP32 IEEE sin TF32, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

Otros agentes compartían la máquina durante las medidas, con cargas medias entre 8 y 16 en 16 hilos. Por eso cada comparación se hizo intercalando sus dos lados en la misma sesión, y los valores absolutos son orientativos. `perf` no pudo usarse porque `perf_event_paranoid` vale 4 en el equipo y no se cambió la configuración de seguridad para obtener cifras.

Las cintas se prepararon con [`benchmarks/rl_stage_tapes.py`](../../../benchmarks/rl_stage_tapes.py) sobre `unadjusted-prices-20261009/edition-v1`, con el mismo camino que la etapa (`window_tapes`), 400 candidatos por orden alfabético y la regla de universo de la etapa hasta 128 activos. EE. UU. usa la ventana de política `fold-018`, con ajuste en `fold-014` a `fold-016` (252, 253 y 252 sesiones) y validación en `fold-017` (251). China usa `fold-012`, con ajuste en `fold-008` a `fold-010` y validación en `fold-011`. Las cuatro cintas de cada mercado tienen 128 activos y la observación tiene 770 valores.

## Compilación CUDA

`cmake --preset native-ppo-release -DMARS_TITAN_LIBTORCH_ENABLE_CUDA=ON`, con el entorno de PyTorch del proyecto, configura en 7,7 s y compila los 103 objetivos en 4 min 5 s con un pico de 0,65 GiB. `libnvJitLink` se resuelve desde la instalación CUDA 13.4 del sistema y no desde la rueda de PyTorch, algo que conviene fijar si se empaquetan los binarios. CTest pasa 26 de 26 pruebas sin las tres que ejecutan Adam, incluida la nueva `ppo_collection`.

## Perfil del recorrido

`mars-titan-policy-benchmark` (en [`native/benchmarks/policy_collection.cpp`](../../../native/benchmarks/policy_collection.cpp)) mide por separado el paso del entorno con 16 carriles, la copia de observaciones, la inferencia con copia de vuelta, la GAE de un recorrido de 1.024 transiciones, el forward y backward de un minilote de 64 del objetivo PPO, la recogida de `PpoTrainer::advance` antes de su primera actualización, el primer paso de un entrenador recién creado, una oleada completa de KLPO hasta `ready` con el forward y backward de su objetivo, y un episodio de evaluación con un carril. Al terminar exige que los parámetros conserven su huella y que no haya pasos de optimizador.

La traza de Kineto repartió el tiempo de forma distinta según el dispositivo. En CPU dominaba el coste fijo de muchos operadores ATen pequeños. En `cuda:0` la GPU pasaba la mayor parte del paso ociosa entre lanzamientos de kernels muy cortos y esperas al copiar acciones y valores al host. Una MLP de 770×64×64 con 16 filas hace unos 1,7 MFLOP por forward, demasiado poco para ocupar la GPU. En los dos dispositivos aparecieron dos costes evitables en el código del proyecto, no en LibTorch.

Nsight Systems lo confirmó restando una traza de 64 pasos medidos de `PpoTrainer` a otra de 192 sobre las cintas de EE. UU. Antes de esta rama cada paso lanzaba 60 kernels, hacía 8 copias (5 al host y 3 a la GPU) y esperaba 8 sincronizaciones de stream. Ahora lanza 44 kernels, hace 5 copias (4 y 1) y espera 5 sincronizaciones. Las huellas del recorrido coinciden también bajo el perfilador.

## Optimizaciones conservadas

1. **Validación incremental de la oleada KLPO.** `KlpoTerminalCollector::collect_tick` validaba la oleada entera en cada paso, con un coste cuadrático en la longitud del episodio. Ahora `validate_klpo_batch` recibe cuántos pasos de cada episodio ya superaron la validación, comprueba siempre cabecera, calendarios y presupuesto y, de cada episodio, repite el último paso validado y los añadidos. La validación completa sigue al guardar un punto de control, al restaurarlo y al formar el objetivo.
2. **Reutilización del forward de bootstrap en PPO.** El validador de cada paso ya calculaba la salida de la política para la observación siguiente. Con la MLP sin contexto ni Double DQN, `PpoTrainer` guarda esa salida con su observación y los pasos del optimizador, y el paso siguiente muestrea de ella con `PpoPolicy::sample` si la observación coincide byte a byte (`at::equal`) y no ha habido actualización. Una pausa, una restauración, una actualización o un fallo descartan la salida guardada. La GRU, Double DQN y el contexto conservan el forward repetido.

Mediana de tres ejecuciones intercaladas de la versión anterior (`6fee0ef2`, con el mismo medidor) y de esta rama:

| Medida | Dispositivo | EE. UU. antes | EE. UU. después | China antes | China después |
| --- | --- | ---: | ---: | ---: | ---: |
| Paso de recogida KLPO (µs) | CPU | 2.157 | 815 | 2.234 | 741 |
| Oleada KLPO completa (s) | CPU | 0,55 | 0,21 | 0,55 | 0,18 |
| Paso de recogida KLPO (µs) | `cuda:0` | 1.726 | 680 | 1.661 | 694 |
| Oleada KLPO completa (s) | `cuda:0` | 0,44 | 0,18 | 0,40 | 0,17 |
| Paso de `PpoTrainer` (µs) | `cuda:0` | 647 | 526 | 688 | 554 |
| Transiciones PPO por segundo | `cuda:0` | 21.150 | 25.027 | 20.622 | 24.628 |

La oleada KLPO tiene 243 pasos y 3.878 transiciones en EE. UU. La recogida KLPO es entre 2,4 y 3 veces más rápida y el paso de PPO en `cuda:0` un 19 % más corto. Las ejecuciones de CPU se hicieron con carga media de 15 a 16 y las de CUDA con 8, así que solo se comparan dentro de cada dispositivo.

## Paridad

Las huellas SHA-256 coinciden entre la versión anterior y esta rama en los dos mercados y dispositivos: recorrido PPO completo (observaciones, acciones, probabilidades, valores, recompensas, bootstrap, máscaras y pesos de acción), estado de las carteras de los 16 entornos y estado del muestreador tras 544 pasos en `cuda:0`, y registro serializado y muestreador de la oleada KLPO en CPU y `cuda:0`. En CPU, la recogida PPO solo admite el diagnóstico de 32 transiciones y sus huellas también coinciden. `mars-titan-sim --trace` con `hold_initial` y `rebalance_50` sobre las cintas de validación de los dos mercados produce trazas idénticas salvo `identity_sha256`, que cambia porque incluye la huella de las fuentes nativas. Las reglas de acciones A de China, los cortes walk-forward y la auditoría no se tocaron.

La nueva prueba `native/tests/ppo_collection_tests.cpp` recorre 15 pasos de dos carriles con cintas de 4 y 7 sesiones y reproduce cada paso con una política independiente de la misma semilla que hace el forward y el muestreo por su cuenta. Exige acciones, probabilidades, valores y pesos idénticos, que el bootstrap de cada paso sea el valor del siguiente salvo en los cierres, el mismo estado final del muestreador y la misma continuación tras restaurar a mitad del recorrido. `klpo_episodes_tests` añade el contraste entre la validación incremental y la completa para todos los prefijos.

## CPU frente a GPU

Las ejecuciones de CPU y `cuda:0` de la versión optimizada se intercalaron en la misma sesión, con carga media de 13 a 14. Mediana de tres ejecuciones de cada mediana p50:

| Componente | CPU EE. UU. | `cuda:0` EE. UU. | CPU China | `cuda:0` China |
| --- | ---: | ---: | ---: | ---: |
| Paso del entorno con 16 carriles (µs) | 146 | 144 | 165 | 161 |
| Acción PPO de 16 filas con copia de vuelta (µs) | 105 | 251 | 103 | 272 |
| Bootstrap con copia de vuelta (µs) | 73 | 119 | 71 | 127 |
| Acción de Double DQN (µs) | 137 | 425 | 112 | 500 |
| Forward y backward de un minilote de 64 (µs) | 816 | 1.889 | 764 | 1.921 |
| Forward sin gradiente de 64 filas (µs) | 206 | 110 | 211 | 102 |
| GAE de 1.024 transiciones en FP64, siempre en CPU (µs) | 934 | 878 | 924 | 894 |
| Primer paso de un entrenador nuevo (µs) | 479 | 938 | 462 | 1.031 |
| Paso estable de `PpoTrainer` (µs) | no admitido | 570 | no admitido | 604 |
| Oleada KLPO completa (s) | 0,12 | 0,22 | 0,11 | 0,21 |
| Forward y backward del objetivo KLPO de la oleada (s) | 0,06 | 0,04 | 0,06 | 0,04 |
| Episodio de evaluación de un carril (ms) | 8,6 | 36,1 | 9,4 | 38,5 |
| `VmHWM` del proceso (MiB) | 483 | 1.292 | 484 | 1.292 |

Para estas redes, un proceso en CPU es más rápido que uno en `cuda:0` en casi todo el recorrido. La acción y el minilote tardan entre 2,3 y 2,6 veces más en la GPU, la oleada KLPO casi el doble y el episodio de evaluación unas cuatro veces más. Solo ganan en `cuda:0` el forward sin gradiente de 64 filas, que Double DQN usa para sus objetivos, y el objetivo KLPO, que procesa la oleada completa de una vez. La recogida PPO en CPU solo puede medirse con el primer paso de un entrenador recién creado, porque el diagnóstico de CPU admite 32 transiciones. Ese paso no reutiliza el bootstrap y en `cuda:0` tarda entre 1,6 y 1,7 veces el paso estable, y aun así en CPU queda por debajo del paso estable de la GPU.

La etapa exige hoy `cuda:0` y solo admite en CPU el diagnóstico de 32 transiciones. Las huellas de CPU y `cuda:0` difieren entre sí, porque el muestreador y la aritmética de cada dispositivo dan secuencias distintas. Pasar las ejecuciones reales a CPU cambiaría por tanto su identidad y es una decisión del protocolo, no una optimización neutra. Esta rama no la toma.

## Procesos simultáneos

Se lanzaron K procesos independientes del medidor optimizado sobre las cintas de EE. UU. con `--repeat`, 12 repeticiones de las recogidas PPO y KLPO en `cuda:0` y 20 oleadas KLPO en CPU. Se descarta la primera repetición de cada proceso y el caudal agregado suma el de todos. Los intervalos de los procesos se solaparon al menos en un 90 %. MPS usó un servidor propio con directorios privados, que se cerró al terminar. Todos los procesos de cada caso produjeron las mismas huellas que el proceso aislado.

| Configuración | K | Paso PPO por proceso (µs) | Aceleración PPO | Oleada KLPO por proceso (s) | KLPO agregado (transiciones/s) | Aceleración KLPO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 1 | no admitido | | 0,106 | 38.018 | 1,00 |
| CPU | 2 | no admitido | | 0,113 | 71.003 | 1,87 |
| CPU | 4 | no admitido | | 0,114 | 141.967 | 3,73 |
| CPU | 8 | no admitido | | 0,168 | 192.699 | 5,07 |
| `cuda:0` sin MPS | 1 | 576 | 1,00 | 0,192 | 20.908 | 1,00 |
| `cuda:0` sin MPS | 2 | 840 | 1,37 | 0,233 | 34.498 | 1,65 |
| `cuda:0` sin MPS | 4 | 1.282 | 1,79 | 0,353 | 45.448 | 2,17 |
| `cuda:0` con MPS | 1 | 565 | 1,00 | 0,178 | 22.589 | 1,00 |
| `cuda:0` con MPS | 2 | 579 | 1,95 | 0,192 | 41.803 | 1,85 |
| `cuda:0` con MPS | 4 | 613 | 3,67 | 0,213 | 75.262 | 3,33 |

Sin MPS, los contextos se reparten la GPU por tiempo y cuatro procesos apenas rinden 1,8 veces uno. Con MPS, cuatro procesos rinden entre 3,3 y 3,7 veces uno con salidas idénticas, así que si la etapa sigue en `cuda:0`, MPS es la forma de llenar la GPU. En CPU, cuatro procesos escalan casi linealmente y ocho llegan a 5,1 veces, con 8 núcleos físicos y otros agentes ocupando la máquina. Cuatro procesos CPU recogen KLPO casi al doble de velocidad que cuatro procesos CUDA con MPS, y cada proceso CUDA también ocupa un núcleo para lanzar kernels.

Cada proceso CPU adicional sumó unos 267 MiB al cgroup de la medida. Cada proceso CUDA sumó unos 830 MiB de RAM (780 MiB con MPS) y unos 330 MiB de VRAM (unos 270 MiB con cuatro clientes MPS). Con cuatro procesos CUDA la GPU usaba 2.061 MiB sin MPS y 1.884 MiB con MPS, incluidos entre 736 y 797 MiB de otros procesos.

## Memoria

El pico de unos 0,6 GiB anotado para `mars-titan-sim` en [políticas nativas sobre cintas reconstruidas](../../../docs/engineering/native-policy-real-tapes.md) no corresponde al proceso C++. En Linux, el `ru_maxrss` que devuelve `wait4` para un hijo conserva el máximo de la imagen anterior a `exec`, que es la del proceso que lo lanza. `VmHWM`, que es lo que el binario escribe en `executable_peak_rss_bytes`, solo cuenta la imagen nueva:

| Proceso padre | `VmHWM` del padre | `ru_maxrss` del hijo | `VmHWM` de `mars-titan-sim` |
| --- | ---: | ---: | ---: |
| Python pequeño | 15 MB | 40 MB | 40 MB |
| Python con 640 MiB reservados | 686 MB | 686 MB | 40 MB |

`mars-titan-sim --compare` con la cinta de validación de EE. UU. y 128 activos usa unos 40 MB, y `/usr/bin/time` lanzado desde la shell da lo mismo. No hay memoria que reducir en el simulador.

Los binarios con LibTorch pesan más. El medidor llega a 483 MiB de `VmHWM` en CPU, de los que unos 265 MiB son anónimos y unos 225 MiB páginas de bibliotecas compartidas entre procesos. En `cuda:0` llega a 1.292 MiB, con unos 840 MiB anónimos del contexto CUDA y de las bibliotecas cargadas. Las cifras por proceso adicional de la sección anterior son las que cuentan para planificar procesos simultáneos.

## Proyección de la etapa A

`native_policy_hours` (en `src/mars_titan/simulation/policy_throughput.py`) aplica los tiempos medidos al plan de la etapa A: 2.160 trabajos, con 1.368 ajustes (792 de KLPO, 432 de los tres brazos PPO y 144 de Double DQN) y 792 referencias. Cada ajuste recorre 262.144 transiciones con 16 entornos, recorridos de 1.024, cuatro épocas, minilotes de 64 y validación cada 16.384 transiciones. La función cuenta la recogida, la GAE y los minilotes de PPO, un minilote y dos forwards sin gradiente por transición elegible de Double DQN tras su calentamiento de 256, las oleadas completas de KLPO con su objetivo, todas las validaciones y la evaluación con tres costes. La estimación anterior de `policy_hours` contaba en Double DQN los minilotes de PPO, 16 veces menos de los que hace.

El paso de Adam no se ha medido, así que todas las horas son una cota inferior. En CPU, el paso de PPO es el primer paso de un entrenador nuevo, sin reutilizar el bootstrap. El paso de Double DQN se estima en los dos dispositivos como ese primer paso con la acción de Double DQN en lugar de la de PPO, lo que en `cuda:0` lo sobrestima en menos de 0,3 h en total. Las referencias usan el paso de evaluación con red, que también es una cota superior de su coste. Las horas con varios procesos dividen las de un proceso por la aceleración medida en la recogida KLPO en CPU y por la menor de PPO y KLPO con MPS, y suponen que las actualizaciones escalan igual, algo que no se ha medido.

| Escenario | Horas | Sin gradientes de actualización | Double DQN | KLPO | Cada brazo PPO |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cuda:0`, un trabajo cada vez (protocolo actual) | 31,9 | 5,5 | 23,0 | 3,9 | 1,7 |
| `cuda:0`, 4 procesos sin MPS | 17,8 | | | | |
| `cuda:0`, 4 procesos con MPS | 9,6 | | | | |
| CPU, un trabajo cada vez | 18,3 | 3,1 | 13,1 | 2,7 | 0,85 |
| CPU, 4 procesos | 4,9 | | | | |
| CPU, 8 procesos | 3,6 | | | | |

Double DQN concentra más del 70 % de las horas por sus 261.888 minilotes por ajuste. La etapa A suma 44.839.458 pasos de Adam: 37.711.872 de Double DQN, 2.359.296 por cada brazo PPO y 49.698 de KLPO. Cada 100 µs por paso de Adam añaden 1,25 h en serie.

Nota del 10 de octubre de 2026 (PR #430). La etapa declara ahora cinco referencias (se añaden 1/N mensual e índice de mercado) y cuatro costes de evaluación (0, 5, 10 y 20 pb), así que la etapa A tiene 1.221 referencias en lugar de 792. Con los mismos tiempos medidos de este informe, `native_policy_hours` da 31,97 h en `cuda:0` y 18,28 h en CPU con un trabajo cada vez, frente a las 31,9 y 18,3 h de la tabla, y los pasos de Adam no cambian. Las cifras de la tabla se conservan tal como se midieron.

## Mutación dirigida

Se aplicaron de uno en uno 16 defectos. En C++, los seis de la reutilización y la validación incremental fallan: comparar sin la observación, muestrear con `argmax`, guardar el forward de otra entrada, no repetir el último paso validado, quitar la cota de pasos validados y quitar la comprobación del tamaño del recuento. Sobrevive uno en el colector, que calcula el recuento después de añadir el paso y por tanto da por validado el paso nuevo. El colector solo añade pasos que construye él mismo y ninguna interfaz pública permite inyectar uno inválido, así que ninguna prueba puede distinguirlo sin un punto de inyección que no existe en producción. La oleada completa se vuelve a validar al formar el objetivo y en cada punto de control, de modo que ese defecto retrasaría la detección, pero no dejaría pasar un registro inválido al gradiente. En Python, los nueve de `native_fit_work` y `native_policy_hours` fallan: calentamiento de Double DQN, validación final y cadencia de KLPO, número de forwards de Double DQN, GAE de PPO, oleadas redondeadas hacia arriba, oleada con una sola ventana, tiempos negativos y coste de un traslado. Este último sobrevivió en la primera pasada y lo mata ahora `test_native_carry_evaluates_each_cost_without_fitting`.

## Comprobaciones

| Comprobación | Resultado |
| --- | --- |
| CTest en `native-ppo-release` con CUDA, sin las tres pruebas de Adam | 26/26 |
| `test_native_policy_tapes.py`, `test_native_real_tapes.py`, `test_native_ppo_launcher.py`, `test_policy_throughput.py` y `test_campaign_stage.py` con los binarios de esta compilación | 168 pasan, incluidas las 6 de la edición real |
| clang-tidy con la configuración del proyecto sobre las fuentes cambiadas | sin avisos |
| `ruff check`, `ruff format --check` y `scripts/check_repository.py` | correctos |

## Pendiente

Queda fuera de esta rama todo lo que aplica pasos de optimizador, porque el bloqueo de aprendizaje sigue vigente:

1. Las tres pruebas CTest con Adam en la compilación CUDA, con la GPU reservada para una sola carga:

   ```bash
   ctest --test-dir build/native/native-ppo-release -R '^(ppo_policy|ppo_training|ppo_gru_packing)$' --output-on-failure
   ```

2. El coste de la actualización con Adam en cada motor. Con la salida de `benchmarks/rl_stage_tapes.py` en `$T`, se escriben las configuraciones de un ajuste real de cada brazo y se lanza un ajuste corto con pausa. PPO llega a 1.024 pasos de Adam con 16.384 transiciones, Double DQN a 16.128 y KLPO a unas 16 oleadas con 65.536:

   ```bash
   D=$HOME/.cache/mars-titan-tmp/rl-update-cost
   uv run python - "$D" <<'PY'
   import json, sys
   from pathlib import Path
   from mars_titan.simulation.native_policy_runs import klpo_config, ppo_config
   from mars_titan.simulation.policy_plan import load_stage, plan_stage
   stage = load_stage("configs/simulation/historical-masked-rl-stage-a.json")
   out = Path(sys.argv[1])
   out.mkdir(parents=True, exist_ok=True)
   for job in plan_stage(stage):
       path = out / f"{job['arm']}.json"
       if job["kind"] == "fit" and job["market"] == "US" and not path.exists():
           build = klpo_config if job["engine"] == "native_klpo" else ppo_config
           path.write_text(json.dumps(build(stage, job), indent=1))
   PY
   for arm in ppo_clip_full_kl double_dqn klpo_terminal; do
     bin=build/native/native-ppo-release/mars-titan-ppo; stop=16384
     [ $arm = klpo_terminal ] && bin=build/native/native-ppo-release/mars-titan-klpo && stop=65536
     /usr/bin/time -v uv run python scripts/run_native_ppo.py --binary $bin --config $D/$arm.json \
       --output $D/$arm-run $(for t in $T/US/*-train-*; do echo --train-tape $t; done) \
       --validation-tape $(echo $T/US/*-validation-*) --stop-after $stop > $D/$arm.log 2>&1
   done
   ```

   El coste de cada paso de Adam se obtiene restando al tiempo de pared la recogida y las validaciones medidas aquí y dividiendo por los pasos alcanzados. Con esa cifra, `native_policy_hours` deja de ser una cota inferior.

3. La decisión de dispositivo. Mantener `cuda:0` con un trabajo cada vez cuesta al menos 31,9 h, y con cuatro procesos bajo MPS al menos 9,6 h. Admitir ejecuciones reales en CPU con ocho procesos bajaría a unas 3,6 h, pero exige cambiar la validación de `PpoTrainingConfig`, asumir nuevas identidades de ejecución y repetir la paridad y la recuperación en CPU. Esa elección corresponde al protocolo de la campaña.
4. La estimación de la etapa en `campaign_throughput.py` sigue usando `policy_hours`. Sustituirla por `native_policy_hours` con estas medidas toca un archivo que comparten otras ramas y se deja para una integración aparte.
