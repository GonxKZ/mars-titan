# Comprobaciones CUDA sin pasos de optimizador (9 de octubre de 2026)

Con la codificación de la edición histórica v3 terminada, la GPU quedó libre y se ejecutaron en `cuda:0` las comprobaciones pendientes que no aplican pasos de optimizador (issue #374). El bloqueo de aprendizaje siguió vigente y el gancho de `tests/conftest.py` estuvo activo en todas las ejecuciones. Ninguna comprobación mide utilidad predictiva. Son contrastes técnicos de paridad CPU/CUDA, recuperación y memoria sobre fixtures.

El [resumen estructurado](summary.json) recoge órdenes, tiempos, picos de memoria, huellas de los binarios y las medidas de memoria. Los recibos de cada comprobación están en esta carpeta.

## Entorno

| Elemento | Valor |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (Max-Q), 8.188 MiB, capacidad 8.9, driver 595.91.07 |
| Perfil de energía | `power-saver`, sin cambios |
| Python y PyTorch | 3.12.14, PyTorch 2.14.0+cu130 (CUDA 13.0, cuDNN 9.24) |
| Otras bibliotecas | NumPy 2.5.3, PyArrow 25.0.1, XGBoost 3.3.0, CuPy 14.2.0 |
| Compilador | Clang 21.1.8, CMake 4.2.3, Ninja, `-j 2` |
| Hilos | `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Commit base | `2f0e5f72` (develop) |

Las comprobaciones se ejecutaron sobre `2f0e5f72` con las correcciones de pruebas de esta rama, que no cambia `src/`. Los recibos de los medidores de Titans se regeneraron tras corregir su procedencia. Cada comprobación se lanzó sola en la GPU, tras consultar `nvidia-smi` y `torch.cuda.is_available()`. El pico de VRAM procede de `nvidia-smi` cada 200 ms e incluye unos 612 MiB del escritorio. Los picos del asignador de PyTorch están en los recibos. La máquina compartía la CPU con la generación de objetivos y con otros procesos, así que los tiempos son orientativos.

## Compilaciones

| Preset | Opciones | Tiempo | Resultado |
| --- | --- | ---: | --- |
| `native-candidate-cuda` | enlace episódico Python y simulación, LibTorch con CUDA | 85,6 s | enlace `144b7534…` |
| `native-release` | con el enlace episódico añadido para comparar su huella | 41,6 s | `mars-titan-sim`, biblioteca de simulación y enlace `230cdc5c…` |
| `native-ppo-release` | PPO con LibTorch | 137,9 s | `mars-titan-ppo` |

El caso M2 de `tests/memory/cuda_mature_error_check.py` exige el enlace `e7559ed4…`. Esa huella la fija la constante `NATIVE_SHA` del propio archivo, introducida en `0bf7fd0b`, y corresponde a una compilación `native-release` del 8 de octubre en el worktree `financial-session-v2-20261008`, con enlace episódico, sin candidato y sin backend CUDA de LibTorch. El binario incorpora el hash de sus fuentes y una identidad de compilación con rutas absolutas de include, de modo que ninguna compilación de develop hecha en otro worktree puede reproducirla.

## Resultados

| Comprobación | Resultado | Tiempo | VRAM pico |
| --- | --- | ---: | ---: |
| `ctest candidate_cuda` | 1/1 pasa | 2,7 s | 934 MiB |
| GRU en sesiones financieras | 1 pasa, error máximo 2,2·10⁻⁸ en FP32 y 1,2·10⁻¹⁶ en FP64 | 9,0 s | 810 MiB |
| Paridad de la GRU candidata (script) | 6 casos, error máximo 1,5·10⁻⁶ en FP32 y 1,6·10⁻¹⁵ en FP64 | 4,8 s | 842 MiB |
| Caché externa de XGBoost | 2 pasan tras la corrección 1 | 4,7 s | 1.438 MiB |
| Referencias con máscaras | 115 pasan tras la corrección 2 | 6,8 s | 928 MiB |
| Cuantiles y `test_reference_run` | 24 pasan, 26 omitidas por el bloqueo | 14,1 s | 932 MiB |
| Adaptadores predictivos | 5 pasan tras la corrección 3 | 2,9 s | 900 MiB |
| Puertas Titans (retención) | completo, repetición idéntica bit a bit, diferencia con CPU de 3,6·10⁻⁶ en FP32 y 5,7·10⁻¹⁴ en FP64 | 454,2 s | 955 MiB |
| Entrenador de la GRU candidata | 6 pasan | 23,4 s | 892 MiB |
| Memoria de la candidata, 128, 256 y 512 activos | 4 pasan | 728,5 s | 2.004 MiB |
| Titans, `test_financial_run` CUDA | 1 pasa | 8,5 s | 904 MiB |
| Titans, memoria cronológica (dos recetas) | completo | 3,6 s | 1.890 MiB |
| Titans, escala de la salida de MAC | completo, tres ejecuciones idénticas bit a bit | 174,1 s | 918 MiB |
| Postentrenamiento con cuantiles | 3 pasan | 4,4 s | 902 MiB |
| MARS-TITAN ampliado (M1 y M3, K = 1 y 4) | 10 pasan tras la corrección 4 | 60,2 s | 902 MiB |
| CM-v1, B y B+C con `accumulation_rows` nulo y 2 | 8 pasan tras la corrección 4 | 57,0 s | 965 MiB |
| M3 (`-k m3`) | 1 pasa con el enlace de develop | 14,4 s | 902 MiB |
| M2 (`-k m2`) | falla con el enlace de develop por la huella, pasa con el binario `e7559ed4` | 15,6 s | 902 MiB |
| `ctest ppo_inputs` | 1/1 pasa, sin GPU | 0,1 s | 612 MiB |
| Delta CUDA de KLPO sin Adam | MLP y GRU idénticos al recibo de `be88f4d9` | 1,1 s | 922 MiB |
| Selección CUDA de la PR #380 | 81 pasan, 61 omitidas por el bloqueo | 16,0 s | 1.147 MiB |

Las 61 omisiones de la selección de la PR #380 son 57 entradas de ajuste detenidas por la protección y 4 pasos de AdamW interceptados por el gancho. Ninguna falla.

La caché de XGBoost alcanza 1.438 MiB porque CuPy y XGBoost reservan su propio pool. Ninguna comprobación superó 2.004 MiB de VRAM total.

## Fallos y diagnóstico

1. **Caché externa de XGBoost (fallo de la prueba).** Desde #388, `fit_external_boosting` comprueba la protección en su entrada, así que las dos pruebas se omitían sin construir la caché, aunque `xgb.train` ya se sustituía por una excepción. Ahora usan `learning_doubles`.
2. **Referencias con máscaras (fallo de la prueba).** Por la misma causa, la prueba CUDA y 17 pruebas CPU de `test_masked_reference_run.py` se omitían. El sustituto de AdamW no modifica pesos y exige pesos sin cambios en cada paso. Con `learning_doubles` pasan las 2 CUDA y las 18 CPU.
3. **Adaptadores nulos en el Transformer (fallo de la prueba).** En entrenamiento, la copia con adaptadores difería del padre en 1,5·10⁻⁸ aunque sus pesos efectivos son idénticos bit a bit. `torch.matmul` resuelve la proyección de entrada no contigua de `MultiheadAttention` con `mm` (tras plegar la entrada) cuando un peso requiere gradiente y con `bmm` cuando no. En CUDA son núcleos distintos y redondean distinto. Se comprobó con una reproducción mínima de `linear` sobre una entrada transpuesta. La prueba exige ahora pesos efectivos idénticos y salidas con tolerancia relativa 1e-6 y absoluta 1e-7. En inferencia sigue exigiendo igualdad exacta y CPU la conserva también en entrenamiento.
4. **CM-v1 y MARS-TITAN ampliado (fallo de la prueba).** En un proceso nuevo, `torch.cuda.reset_peak_memory_stats(0)` lanza `Invalid device argument` hasta que CUDA se inicializa. Las dos comprobaciones reiniciaban el pico antes de la primera reserva y fallaban sus 8 y 10 casos sin comparar nada. Se añade `torch.cuda.init()` antes del reinicio.
5. **Recogida de la suite completa (fallo real en develop).** `tests/posttraining/test_campaign_stage.py` (#397) y `tests/simulation/test_campaign_stage.py` (#401) comparten nombre de módulo en carpetas sin paquete, y pytest detiene la suite completa al recogerla. Se renombra el de postentrenamiento a `test_adapter_campaign_stage.py` y su referencia en la documentación. El de simulación conserva el nombre porque hay trabajo abierto que lo modifica.
6. **Declaración de MARS-TITAN (fallo real en develop).** #410 añadió `m3` a los bancos permitidos a propósito, pero una prueba seguía exigiendo su ausencia. La prueba nueva exige que los bancos declarados coincidan con `MARS_BANKS` y que M3 tenga su receta.
7. **Recibos de los medidores de Titans (fallo real del código).** `titans_gate_retention.py` y `titans_mac_output_scale.py` escribían `CUDA_VISIBLE_DEVICES=-1`, un alcance «en CPU» y una orden sin `--device` aunque se ejecutaran en `cuda:0`. Ahora registran las opciones y el entorno reales, con una prueba unitaria. Los recibos de esta carpeta se regeneraron después de la corrección.
8. **Huella del caso M2.** No es un fallo del código: la prueba fija un binario concreto. Con el enlace de develop falla la aserción. Con el binario archivado `e7559ed4` pasa sin cambios ([recibo](m2-mature-error-cuda.json)), y con la huella de develop sustituida localmente también pasa ([diagnóstico](m2-mature-error-cuda-diagnostic.json)), con los mismos errores máximos que el recibo anterior (9,3·10⁻¹⁰ en FP32 y 3,5·10⁻¹⁸ en FP64). Conviene decidir si la prueba debe fijar el binario o su hash de fuentes.
9. **Orden de la paridad de la candidata.** `tests/models/candidate/cuda_candidate_check.py` es un script con `argparse`. La orden de la lista lo pasaba a pytest, que no recoge ninguna prueba. Se ejecutó como script con `--output`.

## Observaciones sin cambio de código

- La GRU nativa de la candidata emite en CUDA el aviso de cuDNN de pesos no contiguos, que obliga a compactarlos en cada llamada. `native/src/candidate.cpp` llama a `at::gru` sin aplanar los pesos, a diferencia de `native/src/ppo_policy.cpp`. No se ha medido su coste.
- En la escala de la salida de MAC, CPU y CUDA coinciden con `gate_bias` sin residual (4·10⁻¹⁶ en FP64), pero con residual y LayerNorm las trayectorias se separan con el número de pasos (1,7 % en 2.048 pasos y 4,2 % en 4.096 con `gate_bias`). Tres ejecuciones CUDA dan resultados idénticos bit a bit. La receta de campaña usa esa memoria, así que una ejecución en CPU y otra en CUDA no deben compararse como si fueran la misma.
- Con los fixtures pequeños, MARS-TITAN y CM-v1 tardan el doble en CUDA que en CPU, porque el trabajo se reduce a lanzamientos. No es representativo de la campaña.
- M2 y M3 alcanzan 69 MB del asignador, frente a 19 MB del recibo M2 anterior, también con el binario archivado. El aumento procede del lado Python de develop y queda dentro del límite de 128 MiB de la prueba.

## Medidas de memoria

### GRU candidata

Receta propuesta en FP32 (`update_instants=8`, `block_rows=128`, banco de 1.024) con dimensiones reales de entrada. Pico neto del asignador de PyTorch durante un ajuste sin pasos ([recibo](candidate-memory-cuda.json)).

| Activos | Sin acumulación | `accumulation_rows=128` | `accumulation_rows=1024` | Recomputación | Ambas |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 357,9 MiB | 156,0 MiB | 156,0 MiB | 144,9 MiB | 148,2 MiB |
| 256 | 622,1 MiB | 189,5 MiB | 191,6 MiB | 151,5 MiB | 149,3 MiB |
| 512 | 1.157,8 MiB | 256,6 MiB | 258,7 MiB | 164,7 MiB | 150,9 MiB |

Sin acumulación el pico crece unos 2,1 MiB por activo entre 256 y 512. Con `accumulation_rows=128` crece unos 0,26 MiB por activo. La recomputación de un bloque de 128 filas pasa de 5,6 ms a 8,3 ms en forward y backward (frente a 2,0 ms del forward sin grafo) y da gradientes idénticos bit a bit. La acumulación cambia el orden de suma y difiere hasta 9·10⁻⁸. El tiempo del ajuste completo apenas cambia entre configuraciones (unos 8, 16 y 32 s con 128, 256 y 512 activos).

Una extrapolación lineal, no medida, sitúa el ajuste sin acumulación de 4.000 activos por encima de los 8 GiB, y con `accumulation_rows=128` en torno a 1,1 GiB.

### Titans-MAC

Grafo guardado por fila e instante antes del backward, con dimensiones reales y 128 filas ([receta por defecto](titans-chronological-memory-cuda.json) y [receta de campaña](titans-chronological-memory-campaign-cuda.json)). Las estimaciones por tramo son del propio medidor, con `accumulation_rows=128`.

| Variante (receta de campaña) | Bytes por fila e instante | Pico medido con backward | Tramo de 1.024 flujos sin y con acumulación | Tramo de 5.023 flujos sin y con acumulación |
| --- | ---: | ---: | ---: | ---: |
| `transformer_direct` | 241.521 | 307 MiB | 1,84 GiB y 288 MiB | 9,04 GiB y 494 MiB |
| `mac_disabled` | 243.314 | 321 MiB | 1,86 GiB y 418 MiB | 9,11 GiB y 1,10 GiB |
| `mac_frozen` | 292.235 | 367 MiB | 2,23 GiB y 466 MiB | 10,94 GiB y 1,14 GiB |
| `mac_online` | 393.895 | 481 MiB | 3,01 GiB y 565 MiB | 14,74 GiB y 1,24 GiB |

En CUDA cada fila guarda unos 16 KB más que en CPU (393.895 frente a 375.558 bytes en `mac_online`). Con la población completa, ninguna variante de Titans cabe en 8 GiB sin acumulación según estas estimaciones, y todas caben con `accumulation_rows=128`.

## No ejecutado

- Las comprobaciones con pasos de optimizador siguen bloqueadas: `tests/training/test_joint_temporal_cuda.py`, `tests/training/test_cn_temporal_cuda.py`, la paridad Ridge en `cuda:0`, las pruebas CUDA de políticas RL y las 160 exclusiones de la PR #380 que ajustan modelos fuera de PyTorch.
- `scripts/run_masked_campaign.py throughput` y el caudal de RL, que necesitan las vistas preparadas tras los objetivos.
- Del delta de KLPO se repitió solo el tramo de actores de la sonda privada de #362. El tramo de referencia y oleada forzada no forma parte de su versión final.

## Suite CPU con binarios nativos

Con `CUDA_VISIBLE_DEVICES=-1`, la biblioteca y `mars-titan-sim` de `native-release`, `mars-titan-ppo` y el enlace episódico, y las exclusiones de la PR #380 (los archivos mixtos se excluyeron completos), la suite da 6.625 pruebas superadas, 22 fallos y 360 omisiones en 30 min 38 s. 21 fallos son pruebas que exigen CUDA y no se sustituyen por CPU, y pasan u omiten por el bloqueo en la selección CUDA. El restante era la aserción de M3 (diagnóstico 6), ya corregida. Las omisiones son 260 entradas de ajuste y 49 pasos de optimizador detenidos por la protección, 48 por CUDA ausente o ventana exclusiva y 3 por la edición real sin declarar. Los grupos que fallaban por falta de binarios en la PR #380 pasan ahora: 75 de `test_native_runner`, 13 de `test_ppo_objective_config`, 36 del ciclo financiero nativo y 19 de la GRU en sesiones.
