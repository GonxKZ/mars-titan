# Perfilado de CPU, memoria y GPU

Este documento recoge cómo se perfilan los recorridos de MARS-TITAN en el equipo de trabajo sin cambiar su configuración de seguridad. Las medidas y sus conclusiones están en el [informe de perfilado](../../reports/engineering/profiling-20261010/README.md).

## Herramientas

| Herramienta | Versión | Instalación | Uso |
| --- | --- | --- | --- |
| py-spy | 0.4.2 | `uv tool install py-spy==0.4.2`, fuera del entorno del proyecto | Muestreo del tiempo de Python por función |
| memray | 1.20.0 | Grupo `dev` de `pyproject.toml` | Pico de memoria atribuido por asignación |
| Nsight Systems | 2026.3.2 | Con el toolkit CUDA del sistema | Trazas CUDA y rangos NVTX |
| `torch.cuda.nvtx` | La de PyTorch | Ya incluida en torch | Rangos por fase de los entrenadores |

`perf_event_paranoid` vale 4 y el controlador de NVIDIA tiene `RmProfilingAdminOnly=1`, así que `perf`, los contadores de Nsight Compute y py-spy enganchado a un proceso ya en marcha necesitan permisos de administrador. Esa configuración no se cambia. py-spy funciona lanzando el proceso como hijo suyo, porque `ptrace_scope=1` permite seguir a los descendientes, y Nsight Systems traza CUDA y NVTX sin muestreo de CPU.

Todas las medidas van dentro de `memslot`, con `TMPDIR` en disco y no en `/tmp`, y con una sola carga GPU a la vez (`memslot gpu`). Ninguna da pasos de optimizador.

## py-spy

```bash
memslot gpu -- py-spy record --rate 100 --threads --nonblocking --format raw \
  --output <salida>.collapsed -- python benchmarks/chronological_replay.py <vista> <trabajo> \
  --family titans --arm titans_mac_online ... --events <eventos>
```

Por defecto py-spy detiene el proceso en cada muestra. En el informe esa pausa alargó mac_online entre un 22 % y un 33 % y el lector M1 entre un 36 % y un 90 %. Con `--nonblocking` lee la pila sin detenerlo y el coste no se distingue del ruido, a cambio de perder entre el 1 % y el 14 % de las muestras. py-spy imprime al terminar las muestras y los errores, y conviene anotarlos con el reparto. Los dos modos ordenan igual las funciones principales.

Con `--gil` solo se guardan las muestras en las que el hilo tiene el GIL. PyTorch lo suelta mientras ejecuta un operador llamado desde Python, así que comparar las dos capturas separa el tiempo del intérprete del tiempo dentro de los operadores y de las esperas a la GPU. `reports/engineering/profiling-20261010/pyspy_split.py` hace ese reparto por función dentro de `_train_pass`.

`--native` añade los marcos de C++ pero no sirve en estos recorridos. py-spy detiene el proceso en cada muestra y desenrollar las pilas de libtorch de todos los hilos tarda más que el intervalo de muestreo. En mac_online el recorrido pasó de 37 s a más de cinco minutos, con la GPU trabajando mientras el proceso estaba detenido, así que el reparto resultante no representa el recorrido real.

## memray

```bash
ARROW_DEFAULT_MEMORY_POOL=system memslot suite --max 9G -- \
  uv run --no-sync python -m memray run --native --aggregate -o <salida>.bin <guion> ...
uv run --no-sync python reports/engineering/profiling-20261010/memray_peak.py <salida>.bin <reparto>.json
```

El modo agregado guarda solo el máximo de memoria viva y las asignaciones que no se liberaron, sin escribir cada asignación. `ARROW_DEFAULT_MEMORY_POOL=system` hace que Arrow reserve con `malloc` y memray vea esas asignaciones con su pila. Con el asignador por defecto de Arrow solo se verían los bloques grandes que pide al sistema. La medida sin memray y con el asignador por defecto da el pico real del proceso para compararlo con la declaración del planificador, porque memray alarga el recorrido varias veces y sube el RSS.

En un proceso que inicia CUDA el máximo de memray incluye las proyecciones del controlador en el espacio de direcciones, que no son memoria del anfitrión. En la validación de XGBoost marcó 11,7 GB con un RSS de 5,4 GiB. En esos procesos el pico se compara con el RSS y memray solo sirve para atribuir las asignaciones del anfitrión.

## Rangos NVTX

`src/mars_titan/nvtx_ranges.py` abre un rango por fase en los entrenadores de Titans-MAC (`financial_run.py`), del lector de MARS-TITAN (`mars_titan_run.py`) y en el lector cronológico (`financial_observations.py`). Están apagados por defecto. `MARS_TITAN_NVTX=1` los enciende al importar el módulo y falla si PyTorch no trae CUDA. Cualquier otro valor distinto de `0` es un error.

```bash
MARS_TITAN_NVTX=1 memslot gpu -- nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output <salida> python benchmarks/chronological_replay.py ...
nsys stats --report nvtx_pushpop_sum,nvtx_gpu_proj_sum <salida>.nsys-rep
```

| Rango | Qué abarca |
| --- | --- |
| `titans.read`, `readout.read` | Cada `next` del lector de eventos en el bucle de ajuste |
| `titans.labels`, `readout.labels` | Etiquetas que maduran en el instante |
| `titans.update`, `readout.update` | El paso completo, del backward al sellado de los parámetros |
| `titans.backward` y `readout.replay` | Backward del tramo, o repetición de los bloques maduros en el lector |
| `readout.checks` | Comprobaciones diferidas de la repetición y suma de pérdidas |
| `titans.clip`, `readout.clip` | Recorte de la norma de los gradientes |
| `titans.optimizer`, `readout.optimizer` | Paso del optimizador, puesta a cero y sellado |
| `titans.truncate` | Truncamiento de los flujos tras el paso |
| `titans.checkpoint`, `readout.checkpoint` | Escritura de un checkpoint de recuperación |
| `titans.observe`, `readout.observe` | Emisión de las predicciones del instante |
| `titans.validate`, `readout.validate` | Comprobación de las entradas en CPU |
| `titans.control_plan`, `readout.event_plan` | Plan del control C o del instante en el lector |
| `titans.prepare`, `readout.prepare` | Forward de Titans-MAC del bloque |
| `titans.emit`, `readout.emit` | Copia de las predicciones al anfitrión y registro |
| `titans.penalty` | Penalización del control C |
| `readout.snapshot`, `readout.apply`, `readout.encode`, `readout.admit` | Instantánea del banco, lectura episódica, codificación y admisión |
| `reader.event`, `reader.decode` | Montaje de un instante y decodificación de un grupo Parquet en el lector |

Los rangos son marcas del anfitrión. No sincronizan la GPU, no crean tensores ni tocan el generador aleatorio. Un rango se abre y se cierra en el mismo hilo, así que los del lector con prefetch aparecen en el hilo productor. `tests/tooling/test_nvtx_ranges.py` comprueba el cierre ante excepciones, el reparto por hilos y que apagados no llaman a NVTX. `tests/training/test_nvtx_parity.py` repite los recorridos de Titans-MAC y del lector con los rangos apagados y encendidos y exige las mismas predicciones, los mismos gradientes y el mismo historial bit a bit.
