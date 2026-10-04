# Consultas episódicas C++20 y comprobación CUDA

La [tarea #200](https://github.com/GonxKZ/mars-titan/issues/200) elimina la materialización de la matriz de claves en cada consulta de los comparadores episódicos. Se conserva el cambio porque mantiene la trayectoria numérica y reduce la mediana del tiempo de proceso en dos series completas de RL. Las [mediciones estructuradas](episodic-query-20261004.json) incluyen repeticiones, dispersión, memoria, identidades y límites.

## Representación y recuperación

El banco conserva una matriz FP64 preparada para GEMV de LibTorch. Cada escritura convierte solo su fila. Una consulta provisional guarda esa fila en `std::array`, la sustituye mediante una vista `std::span` y la restaura al salir, incluso ante una excepción. Se reutiliza el [guard genérico de LibTorch 2.14](https://github.com/pytorch/pytorch/blob/v2.14.0/c10/util/ScopeExit.h), sin añadir otra biblioteca ni exigir C++23. La destrucción de objetos automáticos durante la propagación de excepciones está descrita en el apartado 14.3 del [borrador público N4861](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/n4861.pdf).

La consulta usa la misma GEMV, normalización, orden de desempate y filtros temporales. Los snapshots conservan claves FP32 con almacenamiento propio. La recuperación reconstruye la matriz FP64 desde esas claves. El ámbito de un banco sigue admitiendo un único controlador, sin acceso concurrente. La API pública y el formato persistente no cambian.

Para un banco completo, las asignaciones de almacenamiento de tensores por consulta confirmada pasan de 533.248 a 8.960 bytes. En la consulta provisional pasan de 533.504 a 8.960. El contador de c10 no mide todas las asignaciones de la STL ni las internas de BLAS.

La representación residente aumenta 256 KiB por banco, hasta 4 MiB con 16 bancos. No es una reducción de toda la memoria del proceso. El presupuesto comprobado incluye ahora la fila FP64 y conserva la admisión existente de 2 MiB por banco. Los picos RSS de la primera serie pasan de 1355,21 a 1370,42 MiB en PPO episódico y de 1353,98 a 1369,16 MiB con HMM. Incluyen el resto del ejecutor y sus temporales.

## Medidas

Equipo: Ryzen 9 8945HS, RTX 4070 Laptop de 8188 MiB y controlador 595.91.07. Compilación Clang 21.1.8 en C++20 y Release, con avisos tratados como errores, `-fno-fast-math` y `-ffp-contract=off`. LibTorch 2.14.0 utiliza su runtime CUDA 13.0. Se observan instrucciones `cvtps2pd` en las conversiones generadas por el compilador, sin introducir intrinsics ni una ISA obligatoria.

El ejecutable [mars-titan-memory-benchmark](../../native/benchmarks/episodic_query.cpp) mide consultas, consultas provisionales y preparación con confirmación. Cada caso realiza 4096 operaciones, un calentamiento y siete repeticiones por implementación, alternando el orden. La huella de identificadores y puntuaciones coincide en todos los casos. La tabla resume el recorrido que incluye preparación, consulta y escritura.

| Capacidad del banco | Referencia, mediana (ms) | Caché FP64, mediana (ms) | Factor de aceleración |
| --- | ---: | ---: | ---: |
| 32 | 30,077 | 23,888 | 1,26 |
| 256 | 47,767 | 34,953 | 1,37 |
| 1024 | 127,895 | 84,928 | 1,51 |

Estos microbenchmarks CPU compartieron el equipo con la campaña. La comparación completa CUDA pausó esa carga y mantuvo las aplicaciones del escritorio. Utilizó 16 entornos, 256 fuentes sintéticas de entrenamiento y 128 de validación, 16 activos y 256 sesiones por mundo, 75.873.867 bytes de entrada, semilla 42 y un trabajador. Cada ejecución completa 8192 transiciones de aprendizaje. Las 11.264 transiciones observadas incluyen calentamiento. Cada serie tiene un calentamiento y cinco repeticiones por implementación y variante, con orden alternado.

| Serie | Variante | Referencia, mediana (s) | Caché FP64, mediana (s) | Reducción del tiempo |
| --- | --- | ---: | ---: | ---: |
| Inicial | PPO episódico | 8,3331 | 8,1359 | 2,37 % |
| Inicial | PPO episódico con HMM | 8,6345 | 7,9841 | 7,53 % |
| Réplica | PPO episódico | 7,7826 | 7,3805 | 5,17 % |
| Réplica | PPO episódico con HMM | 8,5331 | 7,7297 | 9,42 % |

El tiempo incluye preparación, aprendizaje, validación, checkpoints y lanzador. La dispersión es apreciable y está conservada en el JSON. Las medianas no acreditan un porcentaje universal ni significación estadística. Los 24 pares, incluidos los calentamientos, conservan exactamente pesos, RNG, pasos y momentos de Adam, selección y 270.336 filas de trazas. Solo se excluye de la comparación de metadatos la identidad de compilación de cada traza, previamente contrastada con su propio recibo.

## Comprobaciones y límites

Pasan 14 CTests Release, 14 con ASan/UBSan y 63 pruebas de integración sin omisiones. Seis casos CUDA comprueban pausa y recuperación en las dos variantes episódicas y las semillas 42, 43 y 44. Esos seis casos se repitieron después de ampliar la comparación a los momentos de Adam. Clang Static Analyzer y clang-tidy no señalan diagnósticos propios. Las tres mutaciones dirigidas se detectan: omitir la restauración, no actualizar la caché y devolver snapshots FP64.

LLVM 21 mide 457 de 461 líneas y 181 de 242 ramas en `episodic_memory.cpp`. La consulta y la recuperación ejecutan todas sus líneas instrumentadas. Lizard 1.17.31 calcula complejidad 20 para la consulta y CRAP 20 con la convención `CCN² × (1 − cobertura)³ + CCN`. Estas cifras no demuestran ausencia de defectos. La prueba de asignación fallida verifica que el banco confirmado se puede consultar y que el token sigue siendo válido después de la excepción.

La instrumentación sigue las opciones compatibles descritas por [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) y [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html). LibTorch y Arrow son dependencias precompiladas sin esa instrumentación. No se ha añadido concurrencia ni se han repetido TSan o MSan.

[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) registra, en una comprobación adicional de 96 transiciones, 4576 kernels, 449 sincronizaciones de stream, 101 transferencias H2D con 302.940 bytes y 348 D2H con 1.065.393 bytes. El perfil incluye el arranque y sirve para localizar llamadas pequeñas y transferencias. No sustituye una medida estable de toda la campaña. No se han medido energía, coste monetario ni pico VRAM del proceso.

[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html), con comprobación completa de fugas, termina con código 99 y señala 136.316.933 bytes en nueve asignaciones al cerrar. La referencia anterior reproduce los mismos tamaños y número de avisos. No aparecen accesos inválidos en esos ensayos. Se conserva el diagnóstico, sin supresiones, y su seguimiento en [#202](https://github.com/GonxKZ/mars-titan/issues/202). El [pool de cuBLAS de LibTorch](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/cuda/CublasHandlePool.cpp) documenta retención de espacios de trabajo y restricciones de destrucción. Esa evidencia no convierte el diagnóstico en una prueba sin avisos.

## Reproducción

El instrumento y los límites se conservan en el repositorio. Desde `native/`:

```bash
cmake --preset native-ppo-release -DMARS_TITAN_WARNINGS_AS_ERRORS=ON
cmake --build --preset native-ppo-release
ctest --preset native-ppo-release
../build/native/native-ppo-release/mars-titan-memory-benchmark \
  --capacity 1024 --iterations 4096 --mode step
```

La versión de referencia corresponde a `7caa43a`. El benchmark completo reutiliza [benchmark_adaptive_rl.py](../../scripts/benchmark_adaptive_rl.py), con los dos ejecutables, las variantes `ppo_episodic` y `ppo_episodic_hmm`, cinco repeticiones, 8192 transiciones y un trabajador. El JSON identifica los binarios, entradas y resultados. La prueba [test_episodic_cuda.py](../../tests/simulation/test_episodic_cuda.py) requiere activar `MARS_TITAN_CUDA_INTEGRATION=1` en una ventana CUDA exclusiva y ejecutar pytest mediante `uv run --with hmmlearn==0.3.3 python -m pytest`.

La campaña activa conserva su revisión y ejecutable congelados. La nueva identidad de compilación requiere otra ejecución, aunque la paridad observada sea exacta. Los ensayos son técnicos y sintéticos. No evalúan generalización financiera, no abren el test final y no implementan ni entrenan el candidato MARS-TITAN.
