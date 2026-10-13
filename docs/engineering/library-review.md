# Revisión de bibliotecas y herramientas: HPC, datos e infraestructura

Fecha: 10 de octubre de 2026. Tarea [#485](https://github.com/GonxKZ/mars-titan/issues/485).

Esta revisión contrasta bibliotecas y herramientas de cálculo, datos, compilación, calidad, perfilado y operación local con los caminos que el proyecto ya ha medido. La parte científica, RL y evaluación se revisa por separado. No se ha instalado nada en los entornos compartidos, no se ha creado ningún entorno con torch y no se ha lanzado ninguna medida GPU. Los veredictos se apoyan en las medidas publicadas en los informes enlazados, en las fuentes de cada proyecto y en resoluciones de uv sin instalación.

## Criterios

Una candidata solo entra si mejora un camino medido sin romper las reglas que ya rigen el cálculo del proyecto:

- FP32 estricto, sin TF32, BF16, autocast ni `torch.compile`. Un cambio de núcleo que altere el orden de las sumas necesita paridad medida, como el codificador de último token de [#482](https://github.com/GonxKZ/mars-titan/pull/482).
- Resultados idénticos bit a bit cuando el cambio es solo de rendimiento.
- uv como única vía de paquetes, ejecución local, 8 GB de VRAM, 32 GB de RAM y perfil de energía sin cambios.
- Sin cambiar la seguridad del equipo para obtener cifras (`perf_event_paranoid`, permisos de contadores de la GPU).
- Versiones publicadas al menos dos semanas antes de esta revisión, es decir, el 26 de septiembre de 2026 o antes. Cuando existe una versión más nueva se indica.

Las versiones, fechas y licencias salen de la API JSON de PyPI, de las versiones publicadas en GitHub y del archivo de licencia de cada repositorio. La compatibilidad se comprobó con `uv lock` sobre una copia de `pyproject.toml` y `uv.lock`, restringida a Linux x86-64, con cada candidata en un extra propio. Así uv resuelve contra torch 2.14.0+cu130, numpy 2.5.3, pyarrow 25.0.1, CuPy 14.2.0 y XGBoost 3.3.0 sin crear entornos ni instalar paquetes.

## Caminos medidos

| Camino | Medida | Límite observado | Fuente |
| --- | --- | --- | --- |
| Titans-MAC y núcleos de CM-v1 | mac_online con bloque 1.024: 8.360 filas/s. La GPU trabaja en torno al 19 % del tiempo | CPU: bucle de Python del entrenador (48 % fuera de operadores), lanzamientos (8,7 %) y `fill_` del modo determinista (10,4 %). En la GPU domina la atención FP32 de CUTLASS (23 %) | [Núcleos de la campaña](../../reports/engineering/campaign-kernels-20261009/README.md) |
| Lectores de MARS-TITAN y CM-v1 | M1 con bloque 512: 1.156 filas/s | `cm/medoids.py::_Costs._distance`, 70 de 150 s en M1 y 98 de 155 s en BCM | El mismo informe |
| Referencias neuronales | Paso con CUDA Graphs de 1,4 a 7,9 ms por lote de 256, es decir, de 32.000 a 187.000 muestras/s de cálculo | La lectura. Tras #469 un trabajo rinde 40.400 filas/s (GRU), 33.300 (Transformer) y 48.500 (DLinear) | [Referencias](../../reports/engineering/campaign-kernels-20261009/references.md) y [tubería de la campaña](../../reports/engineering/campaign-pipeline-20261009/README.md) |
| Lector cronológico | 36 filas/s en US+CN fold-012 con la caché LRU de 1 GiB. Tras #469, 4.988 en serie y hasta 9.711 con hilos, con 6,1 a 6,6 GiB de RSS | Caché que no aguantaba el acceso cíclico y SHA-256 de unos 54 GB en cada proceso nuevo (1.415 s), ya corregidos | [Tubería de la campaña](../../reports/engineering/campaign-pipeline-20261009/README.md) |
| Ridge y XGBoost | Gram FP64 a 0,285 TFLOPS, más rápida que el lector. Lector de 16.000 filas/s en CN fold-000 y 163 en el arranque de US+CN fold-012 | Lectura y disco. Las rondas de XGBoost no se han podido medir con el bloqueo | [Ajuste tabular](../../reports/engineering/campaign-tabular-20261009/README.md) |
| Etapa RL nativa | Paso de `PpoTrainer` de 526 a 570 µs en `cuda:0`, con 146 µs del entorno de 16 carriles. Recogida KLPO en CPU de 741 a 815 µs por paso | Operaciones ATen pequeñas y copias al host. Para estas redes la CPU es más rápida que la GPU | [Etapa RL nativa](../../reports/engineering/rl-stage-native-20261009/README.md) |
| Codificación de gráficos de la v3.1 | 625 a 805 gráficos/s en la GPU con lotes de 8 | El dibujo de cada gráfico en CPU, de 3,6 a 4,0 ms por muestra | [Edición v3.1](../data/edition-v3-1.md) |
| Compilación nativa | 103 objetivos de `native-ppo-release` en 4 min 5 s con `-j 2` | Sin desglose entre compilación y enlazado | Etapa RL nativa |
| RAM | OOM global el 9 de octubre. Las familias cronológicas declaran 9.216 MiB en US+CN y XGBoost 8.192 MiB | Picos de RSS sin atribuir por asignación | [Tubería de la campaña](../../reports/engineering/campaign-pipeline-20261009/README.md) |

## Veredictos

Las columnas recogen versión y licencia, el camino que mejoraría con su evidencia, la compatibilidad con CUDA 13, torch 2.14 cu130, Python 3.12, FP32 estricto y determinismo, el coste y los riesgos, y el veredicto.

### Inferencia y GPU

| Candidata | Versión y licencia | Camino y evidencia | Compatibilidad | Coste y riesgos | Veredicto |
| --- | --- | --- | --- | --- | --- |
| TensorRT y Torch-TensorRT | TensorRT 11.3.0.99 (9 sep 2026, licencia propietaria de NVIDIA). Torch-TensorRT 2.14.0 (BSD-3-Clause, 10 sep 2026), que exige TensorRT 11.1 | Ninguna inferencia sin autograd domina un recorrido medido. Titans-MAC actualiza su memoria por gradiente también al predecir, en las referencias la lectura sigue por debajo del paso con grafo, la MLP de RL rinde más en CPU y los codificadores congelados ya fijaron su identidad en la v3.1 | Hay rueda `torch_tensorrt-2.14.0+cu130` para cp312 en el índice cu130 de PyTorch. TF32 viene activo (`disable_tf32=False`). La elección de núcleos por tiempo y la fusión de capas cambian el orden de las sumas, como `torch.compile`, que ya se descartó por una diferencia de 2,4·10⁻⁷ | `tensorrt` y `tensorrt-cu13` 11.1 se publican como sdist que descargan una rueda de bibliotecas de 3,74 GB. No se resolvieron con uv para no construirlas | Descartar. Reabrir si una inferencia sin autograd domina un recorrido, con `disable_tf32=True` y paridad medida frente a PyTorch FP32 |
| cuDNN frontend | 1.30.0 (Apache-2.0 y MIT, 23 sep 2026) | Las convoluciones de ResNet18 ya pasan por cuDNN 9.24 desde PyTorch. La atención FP32 usa `fmha_cutlassF_f32` | Su SDPA en Ampere y Ada solo admite fp16 y bf16. Resuelve con uv, pero arrastra CuTe DSL 4.8.0, bajo EULA de NVIDIA, para cu12 y cu13 | Dependencias pesadas (91 MB solo la biblioteca cu13 de CuTe DSL) sin camino FP32 | Descartar |
| cuBLASLt | Dentro de PyTorch 2.14 y CUDA 13.0 | Ningún GEMM domina. La GPU de mac_online está ocupada un 19 % y el núcleo mayor es la atención | `torch.backends.cuda.preferred_blas_library` permite preferir cuBLASLt, pero PyTorch usa cuBLAS por defecto. Cambiarlo cambia los núcleos y puede cambiar bits | El uso directo exige una extensión C++ propia | Descartar y mantener el valor por defecto |
| CUTLASS y CuTe | CUTLASS 4.8.0 (BSD-3-Clause en C++, 22 sep 2026). CuTe DSL 4.8.0 bajo EULA de NVIDIA | PyTorch ya usa núcleos de CUTLASS para la atención FP32. El perfil no muestra un núcleo que justifique uno propio | Compatible con sm_89 y CUDA 13 | Escribir y mantener núcleos con su referencia numérica | Descartar |
| Triton sin `torch.compile` | 3.8.0 (MIT, 28 ago 2026), ya instalado como dependencia de torch 2.14.0+cu130 | Podría fusionar las operaciones elementales de la actualización de memoria de Titans-MAC y reducir lanzamientos (8,7 % de la CPU). Sin medir | `tl.dot` usa TF32 con fp32 por defecto y `enable_fp_fusion=True` contrae productos y sumas en FMA. Habría que fijar `input_precision="ieee"` y desactivar esa fusión | El bucle de Python (48 %) no desaparece y Triton también lanza desde Python. CUDA Graphs atacan el mismo coste sin cambiar núcleos y se están midiendo en Titans-MAC | Descartar por ahora. Reabrir si los grafos no se pueden aplicar a Titans-MAC |
| CUDA Graphs en Titans-MAC | Dentro de PyTorch 2.14 | En las referencias dieron de 1,4 a 3,0 veces sin cambiar bits (#482) | No aplica | No aplica | Fuera de esta revisión, porque se mide en otra línea de trabajo |
| NVIDIA DALI y nvImageCodec | DALI 2.3.0 (Apache-2.0, 28 ago 2026, rueda de 186 MB). nvImageCodec 0.9.0.20 (Apache-2.0, 14 jul 2026) | Los gráficos son PNG de 224×224 dibujados con PIL. La CPU se va en dibujarlos, no en decodificarlos | nvImageCodec decodifica en la GPU JPEG, JPEG 2000 y TIFF. PNG solo pasa por su extensión de OpenCV en CPU. Resuelven con uv (DALI baja `packaging` de 26.3 a 26.2) | La v3.1 codifica con lotes de 8, el mayor medido con vectores iguales a los de un gráfico por llamada, y la identidad del codificador registra ese lote | Descartar |
| RAPIDS cuDF y cuML | cuDF 26.8.1 y cuML 26.8.0 (Apache-2.0, agosto de 2026) | Ridge ya acumula la Gram FP64 por bloques fuera de memoria, con bits iguales a la ruta anterior. cuML 26.08 no tiene k-medoids (su módulo `cluster` ofrece aglomerativo, DBSCAN, HDBSCAN, KMeans y espectral), y las distancias de `cm/medoids.py` acumulan con `hypot`, que una distancia por GEMM no reproduce | La resolución con uv baja numpy de 2.5.3 a 2.4.6, pyarrow de 25.0.1 a 23.0.1 y pandas de 3.0.6 a 3.0.3, y añade numba 0.64.0 y libcudf (314 MB) | Cambiaría la pila de lectura de la edición y competiría por los 8 GB de VRAM | Descartar |

### Datos

| Candidata | Versión y licencia | Camino y evidencia | Compatibilidad | Coste y riesgos | Veredicto |
| --- | --- | --- | --- | --- | --- |
| Arrow C++ y Parquet | pyarrow 25.0.1 (Apache-2.0, 10 ago 2026), ya en `uv.lock`. Arrow 26.0.0 salió el 9 de octubre | Lectores en Python y nativos (`financial_parquet.cpp` enlaza Arrow y Parquet de la rueda de pyarrow). El límite del lector cronológico era la caché y la huella por proceso, ya corregidas en #469 | Compatible | Ninguno nuevo | Mantener. Pasar a la 26 cuando cumpla dos semanas y con las huellas de lectura comprobadas |
| polars y duckdb | polars 1.44.2 y duckdb 1.5.5 (MIT), ya en `uv.lock` | Auditoría y preparación de la edición | Compatibles | Ninguno nuevo | Mantener |
| Lance | pylance 12.0.0 (Apache-2.0, 17 sep 2026, rueda de 80 MB) | Su acceso aleatorio por fila atacaría la lectura cíclica de unos 5.000 activos por instante, pero ese límite ya lo resolvió la caché de #469 | Resuelve con uv sin cambiar versiones fijadas | Convertir la edición duplicaría unos 47 GB con 33 GB libres y obligaría a rehacer las huellas de integridad | Descartar por ahora. Reabrir si la lectura vuelve a limitar tras #469 |
| WebDataset | 1.0.2 (BSD-3-Clause, 19 jun 2025) | Pensada para flujos secuenciales de muestras en tar, no para cortes transversales por instante con máscaras | Resuelve con uv | Formato nuevo y otra conversión | Descartar |
| torchdata | 0.11.0 (BSD-3-Clause, 20 feb 2025) | `StatefulDataLoader` duplicaría el cursor confirmado que ya guardan los checkpoints | Resuelve con uv, pero su tabla de compatibilidad llega a torch 2.6.0 y no declara versión para la 2.14 | Mantenimiento incierto | Descartar |
| Zarr | 3.4.0 (MIT, 15 sep 2026) | Matrices N-dimensionales por trozos. Los datos son tablas por activo con columnas heterogéneas y máscaras | Resuelve con uv | Conversión de formato | Descartar |

### C++ y compilación

| Candidata | Versión y licencia | Camino y evidencia | Compatibilidad | Coste y riesgos | Veredicto |
| --- | --- | --- | --- | --- | --- |
| pybind11 | 3.1.0 en PyPI (BSD-3-Clause, 6 ago 2026). Los enlaces usan la copia que trae LibTorch | `candidate_python.cpp` pasa tensores con `torch/csrc/utils/pybind.h` y `EpisodicPython.cmake` exige las cabeceras pybind11 de LibTorch | Ligado al ABI de LibTorch 2.14 | Ninguno nuevo | Mantener |
| nanobind | 3.1.0 (BSD-3-Clause, 18 sep 2026) | Las llamadas cruzan el lenguaje con lotes grandes y no hay sobrecarga de enlace medida | No trae los conversores de tensores de PyTorch. Habría que pasar por DLPack | Reescribir dos módulos de enlace | Descartar |
| Eigen | 5.0.1 (MPL-2.0, 11 nov 2025). El sistema tiene 3.4.0 | El álgebra lineal nativa pasa por LibTorch | Compatible | Otra dependencia sin camino medido | Descartar |
| oneTBB | 2023.1.0 (Apache-2.0, 13 jul 2026). El sistema tiene 2022.3.0 | La etapa RL escala por procesos (KLPO en CPU, 3,73 veces con cuatro y 5,07 con ocho) y ATen corre con un hilo. `simulation_files.cpp` ya usa `std::jthread` | Compatible | Sobresuscripción con los hilos de ATen y de los lectores | Descartar |
| Highway y xsimd | Highway 1.4.0 (Apache-2.0 o BSD-3-Clause, 23 abr 2026). xsimd 14.3.0 (BSD-3-Clause, 15 jul 2026) | Ningún bucle nativo vectorizable domina. El paso del entorno reparte sus 146 µs entre muchas operaciones pequeñas | La CPU admite AVX-512, pero una reducción SIMD cambia el orden de las sumas | Dispatch por ISA y pruebas de paridad | Descartar. Medir antes LTO y PGO |
| Google Benchmark | 1.9.5 (Apache-2.0, 21 ene 2026) | `native/benchmarks/` mide recorridos completos con huellas y JSON. Un microbenchmark aislado no basta para decidir | Compatible | Otro arnés | Descartar |
| Catch2 y GoogleTest | Catch2 3.16.0 (BSL-1.0, 25 ago 2026). GoogleTest 1.18.0 (BSD-3-Clause, 10 ago 2026) | CTest ya ejecuta pruebas que se comprueban solas, con etiquetas, sanitizadores y cobertura LLVM con CRAP | Compatibles | Migrar las pruebas nativas sin un fallo que lo motive | Descartar |
| ccache y sccache | ccache 4.12.3 instalado (GPL-3.0 o posterior, herramienta externa sin enlazar). sccache 0.18.0 (Apache-2.0, 14 sep 2026), no instalado | La compilación de `native-ppo-release` tarda 4 min 5 s y cada worktree vuelve a compilar las mismas fuentes | `CMAKE_CXX_COMPILER_LAUNCHER`, sin cambiar los objetos | Caché en disco que hay que acotar | ccache: adoptado solo en los presets de desarrollo tras medirlo ([#487](../../reports/engineering/native-build-variants-20261010/README.md)). sccache: descartar porque cumple el mismo papel |
| mold | 2.40.4 instalado (MIT). La última elegible es 2.42.1 | Enlazado de los ejecutables con LibTorch. Sin medir | `CMAKE_LINKER_TYPE=MOLD` con CMake 4.2. `BuildIdentity.cmake` registra el enlazador, así que cambia la identidad compilada | Solo para desarrollo. No debe usarse en los binarios de la campaña, aunque todavía nada lo impide | Adoptado solo en los presets de desarrollo tras medirlo ([#487](../../reports/engineering/native-build-variants-20261010/README.md)) |
| LTO y PGO | `MARS_TITAN_ENABLE_IPO` y `MARS_TITAN_PGO` ya existen, desactivadas | Paso del entorno (146 µs del paso PPO) y recogida KLPO en CPU (741 a 815 µs por paso). Sin medir | Release compila con `-fno-fast-math` y `-ffp-contract=off`, así que las huellas deberían coincidir | Tiempo de compilación y un perfil PGO representativo | Medidos sin ganancia reproducible ([#486](../../reports/engineering/native-build-variants-20261010/README.md)). Release los mantiene desactivados |

### Calidad y pruebas

| Candidata | Versión y licencia | Camino y evidencia | Compatibilidad | Coste y riesgos | Veredicto |
| --- | --- | --- | --- | --- | --- |
| hypothesis | 6.168.1 (MPL-2.0, 23 sep 2026) | Pruebas por propiedades de códecs, formatos y contratos de lectura. Sin medir | Resuelve con uv | Hay que fijar semillas y guardar los ejemplos que fallan | Usar solo en desarrollo cuando una propiedad describa el contrato mejor que un fixture |
| mutmut y cosmic-ray | mutmut 3.8.0 (BSD-3-Clause, 12 sep 2026). cosmic-ray 8.7.0 (MIT, 9 ago 2026) | La mutación dirigida ya está en los informes, por ejemplo 22 mutantes en el ajuste tabular y 4 en los grafos de las referencias | Resuelven con uv | Una campaña por módulo repite una suite de unas 4.777 pruebas, con CUDA tras un candado y memoria acotada | Descartar y mantener la mutación dirigida |
| pytest-xdist | 3.8.0 (MIT, 1 jul 2025) | La suite corre por trozos en dos plazas de 6 GiB. Sin medir | Resuelve con uv | Cada trabajador importa torch y puede agotar la plaza | Evaluar con benchmark |
| pytest-benchmark | 5.3.0 (BSD-2-Clause, 23 ago 2026) | Los benchmarks del proyecto son scripts con recibos JSON del recorrido completo | Resuelve con uv | Duplicaría el formato de los recibos | Descartar |
| pyright y mypy | pyright 1.1.414 (MIT, 10 sep 2026). mypy 2.3.1 (MIT, 15 ago 2026) | Solo el 7,2 % de las 3.976 funciones de `src/` anota el retorno y el 8,6 % de los parámetros, así que un comprobador tendría pocos tipos que contrastar | Resuelven con uv | Mucho ruido inicial | Descartar por ahora |
| coverage.py con CRAP | coverage.py 7.16.1 (Apache-2.0, 13 sep 2026). La 7.16.2 salió el 27 de septiembre | Los informes ya la usan con Radon 6.0.1, pero cada uno calcula CRAP a su manera y algunos con la 7.16.2 | Resuelve con uv | Ninguno relevante | Adoptar con un cálculo común, como el nativo |

### Perfilado y memoria

| Candidata | Versión y licencia | Camino y evidencia | Compatibilidad | Coste y riesgos | Veredicto |
| --- | --- | --- | --- | --- | --- |
| Nsight Systems | 2026.3.2, instalado con CUDA 13.4 (licencia de NVIDIA) | Ya confirmó los lanzamientos, copias y sincronizaciones de la etapa RL | La traza CUDA funciona sin permisos especiales | Ninguno nuevo | Adoptado |
| Nsight Compute | 2026.3.1, instalado | Contadores por núcleo | El controlador tiene `RmProfilingAdminOnly=1`, que limita los contadores a administradores | Exige cambiar la seguridad del equipo | Descartar mientras no se autorice |
| Perfilador de PyTorch | Kineto en torch 2.14 | Ya repartió CPU y GPU de mac_online, de los lectores y de la etapa RL | Compatible | Ninguno nuevo | Adoptado |
| perf | Del sistema | Contadores de CPU | `perf_event_paranoid` vale 4 | Exige cambiar la seguridad del equipo | Descartar |
| py-spy | 0.4.2 (MIT, 24 abr 2026). Admite Python 3.12 desde la 0.4.0 | El 48 % de la CPU de mac_online queda fuera de los operadores de PyTorch | Lanzado como padre del proceso funciona con `ptrace_scope=1`. `--native` muestra los marcos de C | Muestreo con poca sobrecarga | Usar solo en desarrollo |
| memray | 1.20.0 (Apache-2.0, 7 ago 2026), Linux y macOS | OOM del 9 de octubre y declaraciones de RAM por trabajo sin atribuir por asignación | Sigue asignaciones de extensiones nativas y de hilos | El modo nativo ralentiza | Usar solo en desarrollo |
| scalene | 2.3.0 (Apache-2.0, 12 may 2026) | Solapa con py-spy, memray y el perfilador de PyTorch | Resuelve con uv | Otra herramienta para lo mismo | Descartar |
| NVTX | `torch.cuda.nvtx` ya viene en torch. El paquete `nvtx` 0.2.16 no hace falta | Ninguna fase de los entrenadores está anotada, así que las trazas de Nsight Systems se leen por núcleo | Compatible | Ninguno si los rangos van apagados por defecto | Adoptar en el perfilado |

### Operación local

| Candidata | Versión y licencia | Camino y evidencia | Compatibilidad | Coste y riesgos | Veredicto |
| --- | --- | --- | --- | --- | --- |
| safetensors | 0.8.0 (Apache-2.0, 9 jun 2026), ya en `uv.lock` | Ya carga los pesos de los codificadores congelados (`use_safetensors=True`). Los checkpoints usan `torch.save` y `torch.load(weights_only=True)` con esquema e identidad, y guardan RNG, cursor y otros estados que no son tensores | Compatible | Para checkpoints haría falta un manifiesto aparte | Mantener en los codificadores y descartar para checkpoints |
| torch.distributed.checkpoint | Dentro de torch 2.14 | Pensado para estados repartidos entre procesos. Aquí hay una sola GPU | Compatible | Complejidad sin necesidad | Descartar |
| systemd y memslot | systemd del sistema y el envoltorio local `memslot` | Desde el OOM del 9 de octubre acota cada trabajo con `MemoryMax` y plazas por clase | Funciona | Es local y queda fuera del repositorio | Mantener |
| nvidia-ml-py | 13.615.71 (BSD-3-Clause, 25 sep 2026) | El observatorio lee NVML con `ctypes` en `observatory/telemetry.py`: temperatura, memoria, uso, potencia, reloj y motivos de reducción | Compatible | Otra dependencia para algo que ya funciona y tiene pruebas | Descartar. Reabrir si #451 necesita campos que el envoltorio no cubre |
| DCGM | 4.6.1 (Apache-2.0, 19 ago 2026) | Telemetría pensada para clústeres | Su guía indica funcionalidad limitada en GPU que no son de centro de datos. Necesita el servicio `nv-hostengine` instalado como paquete del sistema | Servicio con privilegios | Descartar |
| MPS | Del controlador | Medido: tres trabajos de transformer_direct rinden 2,51 veces uno y cuatro procesos RL de 3,3 a 3,7 veces | Compatible | Ninguno nuevo | Mantener |

## Adopciones y medidas recomendadas

Ninguna biblioteca de GPU de la lista mejora un camino medido sin romper el FP32 estricto o la igualdad de bits. Lo que sí compensa revisar está en las herramientas locales, ordenado por impacto esperado:

| Issue | Qué se adopta o mide | Por qué |
| --- | --- | --- |
| [#488](https://github.com/GonxKZ/mars-titan/issues/488) | py-spy y memray como herramientas de desarrollo y rangos `torch.cuda.nvtx` apagados por defecto | Atacan los dos costes que hoy no se pueden atribuir: el 48 % de CPU en el bucle de Python de mac_online y los picos de RAM que llevaron al OOM |
| [#486](https://github.com/GonxKZ/mars-titan/issues/486) | LTO y PGO en el recorrido nativo de la etapa RL | Las opciones ya existen y nunca se han medido. La ganancia queda acotada por la parte del paso que es código propio |
| [#489](https://github.com/GonxKZ/mars-titan/issues/489) | coverage.py 7.16.1 y un cálculo común de CRAP para Python | Hace comparables entre informes unas cifras que hoy cada uno calcula a su manera |
| [#487](https://github.com/GonxKZ/mars-titan/issues/487) | ccache y mold en la compilación nativa, solo en desarrollo | Reduce el tiempo de compilación entre worktrees sin tocar los binarios de la campaña |
| [#491](https://github.com/GonxKZ/mars-titan/issues/491) | pytest-xdist dentro de una plaza de `memslot` | Puede acortar la suite, pero solo si el pico de varios trabajadores cabe en 6 GiB |

Los costes medidos más grandes no los resuelve una biblioteca. Las distancias de medoids ocupan la mayor parte del recorrido de los lectores y acumulan con `hypot` dimensión a dimensión, así que una distancia por GEMM (cuML, cuBLAS o SciPy) cambiaría el redondeo. Las dos opciones que conservan los bits (repartir los bloques entre los hilos del trabajo o reutilizar las distancias de los pares que siguen en el banco) ya están descritas en el informe de núcleos. El relleno de memoria del modo determinista (10,4 % de la CPU) depende de una opción de PyTorch, y la lectura de la campaña la resuelve #469.

## Compatibilidad comprobada con uv

Cada fila es una resolución de `uv lock` sobre la copia del proyecto con la candidata en un extra. «Sin cambios» significa que ningún paquete ya fijado cambia de versión.

| Candidatas | Resultado |
| --- | --- |
| cuDF 26.8.1 y cuML 26.8.0 | Resuelve, pero baja numpy a 2.4.6, pyarrow a 23.0.1 y pandas a 3.0.3, y añade 27 paquetes |
| DALI 2.3.0 | Resuelve y baja `packaging` de 26.3 a 26.2 |
| nvImageCodec 0.9.0.20, pylance 12.0.0, Zarr 3.4.0, WebDataset 1.0.2, torchdata 0.11.0 | Sin cambios |
| cuDNN frontend 1.30.0 | Sin cambios, con 8 paquetes nuevos, entre ellos CuTe DSL y protobuf |
| hypothesis 6.168.1, mutmut 3.8.0, pytest-xdist 3.8.0, pytest-benchmark 5.3.0, pyright 1.1.414, mypy 2.3.1 y coverage 7.16.1 | Sin cambios |
| cosmic-ray 8.7.0 | Sin cambios, con 18 paquetes nuevos |
| py-spy 0.4.2, scalene 2.3.0, memray 1.20.0, nvidia-ml-py 13.615.71 y nvtx 0.2.16 | Sin cambios |
| nanobind 3.1.0 y pybind11 3.1.0 | Sin cambios |
| Torch-TensorRT 2.14.0+cu130 | No resuelto. Sus dependencias `tensorrt` y `tensorrt-cu13` 11.1 son sdist cuya construcción descarga 3,74 GB, y se detuvo para no ocupar disco |

## Fuentes

- Índice JSON de PyPI para cada paquete: `https://pypi.org/pypi/<paquete>/<versión>/json`, con fecha de subida, licencia y ruedas.
- Versiones publicadas y licencias en GitHub de [CUTLASS](https://github.com/NVIDIA/cutlass/releases), [cuDNN frontend](https://github.com/NVIDIA/cudnn-frontend/releases), [TensorRT](https://github.com/NVIDIA/TensorRT/releases), [Torch-TensorRT 2.14.0](https://github.com/pytorch/TensorRT/releases/tag/v2.14.0), [DALI](https://github.com/NVIDIA/DALI/releases), [nvImageCodec](https://github.com/NVIDIA/nvImageCodec), [cuDF](https://github.com/rapidsai/cudf/releases), [cuML](https://github.com/rapidsai/cuml/releases), [Lance](https://github.com/lance-format/lance/releases), [torchdata](https://github.com/meta-pytorch/data), [Zarr](https://github.com/zarr-developers/zarr-python/releases), [nanobind](https://github.com/wjakob/nanobind), [pybind11](https://github.com/pybind/pybind11/releases), [oneTBB](https://github.com/uxlfoundation/oneTBB/releases), [Highway](https://github.com/google/highway/releases), [xsimd](https://github.com/xtensor-stack/xsimd), [Google Benchmark](https://github.com/google/benchmark/releases), [Catch2](https://github.com/catchorg/Catch2/releases), [GoogleTest](https://github.com/google/googletest/releases), [ccache](https://github.com/ccache/ccache/releases), [sccache](https://github.com/mozilla/sccache/releases), [mold](https://github.com/rui314/mold/releases), [DCGM](https://github.com/NVIDIA/DCGM), [py-spy](https://github.com/benfred/py-spy/releases), [memray](https://github.com/bloomberg/memray), [scalene](https://github.com/plasma-umass/scalene/releases), [safetensors](https://github.com/huggingface/safetensors/releases) y [Arrow](https://github.com/apache/arrow/releases). Eigen en su [página de versiones de GitLab](https://gitlab.com/libeigen/eigen/-/releases).
- [Tabla de atención de cuDNN frontend](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md), con los tipos admitidos por arquitectura.
- [API de Torch-TensorRT](https://docs.pytorch.org/TensorRT/py_api/dynamo.html), donde `disable_tf32` vale `False` por defecto, y [matriz de soporte de TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html).
- [`triton.language.dot`](https://github.com/triton-lang/triton/blob/v3.8.0/python/triton/language/core.py) y [opciones del compilador NVIDIA de Triton 3.8.0](https://github.com/triton-lang/triton/blob/v3.8.0/third_party/nvidia/backend/compiler.py).
- [`torch.backends` de PyTorch 2.14](https://docs.pytorch.org/docs/2.14/backends.html) para `preferred_blas_library`.
- [Guía de inicio de DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html), con las plataformas admitidas.
- Módulos `cluster` y `linear_model` de [cuML 26.08.00](https://github.com/rapidsai/cuml/tree/v26.08.00/python/cuml/cuml).
