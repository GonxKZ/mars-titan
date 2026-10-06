# Almacenamiento de pesos GRU en LibTorch

El comparador PPO nativo registraba los cuatro pesos recurrentes en almacenamientos separados. Cada llamada CUDA a `at::gru` necesitaba preparar de nuevo el bloque que utiliza cuDNN. La red ahora llama a la primitiva de compactación de LibTorch después de trasladar los parámetros al dispositivo y después de cargar un checkpoint. Conserva nombres, valores, referencias de Adam y esquema de serialización. La ruta CPU mantiene su almacenamiento anterior.

La implementación sigue el mecanismo del [frontend C++ de LibTorch](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/torch/csrc/api/src/nn/modules/rnn.cpp#L163). Las vistas de los parámetros conservan la propiedad del bloque compartido. La compactación se ejecuta sin registrar operaciones de autograd y bajo el dispositivo de esos parámetros. No introduce una implementación propia de GRU ni cambia su aritmética.

## Recorrido completo y movimiento de datos

La [evidencia numérica](gru-weight-packing-20261006.json) corresponde a una RTX 4070 Laptop de 8 GiB, Ryzen 9 8945HS, Clang 21.1.8, LibTorch 2.14.0 y runtime CUDA 13.0. Ambos binarios utilizan C++20 y Release con `-O3`, sin instrumentación, LTO ni PGO. Cada carga usa 16 entornos sintéticos, un trabajador, rollouts de 1.024 transiciones y minibatches de 64. Se conserva la semilla y el protocolo del comparador. Se descarta una pareja de calentamiento y se alterna el orden de ejecución.

| Transiciones de entrenamiento | Parejas medidas | Referencia, mediana de segundos | Candidata, mediana de segundos | Reducción de la mediana |
| ---: | ---: | ---: | ---: | ---: |
| 1.024 | 3 | 5,276 | 4,675 | 11,39 % |
| 8.192 | 3 | 11,039 | 9,890 | 10,41 % |
| 32.768 | 5 | 23,320 | 22,571 | 3,21 % |

El tiempo incluye preparación, entrenamiento, validación, checkpoints y lanzador. Los procesos CPU del escritorio continuaron activos. La dispersión es considerable en las dos cargas mayores y tres de las once parejas medidas son más lentas. Las medianas bajan en las tres cargas, pero no acreditan una aceleración uniforme ni significación estadística. El informe conserva todas las repeticiones, incluida la réplica mayor que se añadió al observar variación en la carga intermedia.

Una pasada separada con Nsight Systems 2026.3.2, sobre la carga de 1.024 transiciones, registra 936.700.160 bytes en 20.228 copias dentro de la GPU en la referencia, frente a 760.064 bytes en 2.372 copias en la candidata. Las cargas útiles CPU a GPU siguen en 69.625.756 bytes y las de GPU a CPU en 3.658.576. Son bytes declarados por las operaciones de copia, no una medida del tráfico físico completo de DRAM.

El pico de RAM de la carga mayor está alrededor de 1,49 GB. Las pasadas sin instrumentación no miden el pico de VRAM. El perfil registra 188.759.560 bytes de asignaciones dinámicas de dispositivo en la referencia y 190.856.712 en la candidata, 2 MiB más. Ese registro no incluye toda la memoria del contexto y del controlador. No se han medido energía ni coste monetario.

La preparación del bloque añade coste al crear y recuperar la red. En cinco repeticiones calentadas, la mediana de construcción pasa de 0,283 a 0,422 ms y la carga de 2,154 a 2,390 ms. Ambas versiones cargan los mismos bytes del checkpoint anterior desde memoria. Estas medidas sincronizan CUDA antes y después de cada operación, excluyen la lectura de disco y la destrucción, y su primera medida tampoco incluye crear el contexto inicial.

## Paridad y recuperación

Los catorce pares, incluidos los calentamientos, conservan exactamente el estado científico. La comprobación abarca 42 pares de bundles retenidos, pesos, RNG, Adam, rollout, selección, evaluaciones, contadores y 317.440 filas de trazas por lado. Se validan 5.442 artefactos y sus vínculos con la identidad de cada ejecución. Las diferencias de compilación y formato físico del archivo se comprueban por separado, sin excluir cambios de valores. El estado también coincide entre repeticiones de cada tamaño.

La prueba nativa comprueba los seis pares de intervalos de los pesos y rechaza doce solapamientos completos o parciales. Exige cuatro pasos Adam tras la primera actualización y ocho tras continuar. También comprueba que cambian parámetros y momentos. El binario anterior falla al exigir almacenamiento compacto. La candidata pasa esa condición y carga un checkpoint CUDA completo del anterior, conservando 71 tensores y la siguiente actualización.

Pasan las pruebas CPU, ASan/UBSan, detección de fugas CPU, clang-tidy y el análisis estático de Clang. El probe final añade cierre explícito a sus dos archivos de salida. Un límite de tamaño de archivo cero reproduce el fallo previo, que devolvía éxito con un CSV vacío. Ahora devuelve error. Esta corrección de escritura se comprobó en CPU y no cambia la producción ni los binarios medidos.

## Recursos retenidos al cerrar

Compute Sanitizer 13.4, con comprobación completa de fugas, termina con código 99 y señala 159.399.445 bytes en 17 asignaciones. El total incluye 13 bytes de memoria host fijada. La versión anterior reproduce la misma distribución por tamaño, API y clase propietaria. Añadir seis construcciones y seis cargas no cambia ese total en ninguno de los dos probes. No aparecen otros errores de memcheck en esos diagnósticos.

Las trazas pasan por espacios de trabajo y handles de cuBLAS, cuDNN y los allocators de dispositivo y host de LibTorch. El [pool de cuBLAS](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CublasHandlePool.cpp#L392) conserva recursos de proceso y la [destrucción de handles cuDNN](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cudnn/Handle.cpp#L12) está deshabilitada en esa revisión. Un bloque cacheado de 2 MiB se atribuye a backward cuDNN en la referencia y a Adam en la candidata. Sus pilas no son idénticas.

La igualdad de segmentos no distingue subbloques vivos de memoria reservada. Solo permite afirmar que este recorrido no observa un aumento introducido por la compactación. No convierte el diagnóstico en una comprobación limpia ni acredita ausencia general de fugas. Se conservan los logs sin supresiones, vaciado de cachés ni reinicios de contexto. El cierre de recursos sigue en [#202](https://github.com/GonxKZ/mars-titan/issues/202).

El cambio se conserva por eliminar preparación repetida mediante la biblioteca existente, mantener paridad y reducir las medianas de las cargas medidas. La referencia es `d2cca84` y la reproducción utiliza [benchmark_adaptive_rl.py](../../scripts/benchmark_adaptive_rl.py) con `--variants ppo_gru`, ambos binarios y los presupuestos de la tabla. Las campañas científicas mantienen sus versiones congeladas. Estas medidas técnicas usan escenarios sintéticos, no evalúan generalización financiera y no implementan ni entrenan el candidato MARS-TITAN.
