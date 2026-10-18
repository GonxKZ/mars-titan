# Reservas CUDA al cerrar LibTorch

Comprobación del 18 de octubre de 2026 para [#202](https://github.com/GonxKZ/mars-titan/issues/202). El candidato termina 300 ciclos de construcción, forward, backward, recuperación y destrucción sin crecimiento observado de memoria activa o reservada después del primer ciclo. Se conservan las tolerancias originales y no hay pasos de optimizador. Esto delimita el comportamiento de estas formas, no demuestra ausencia de fugas en cualquier ejecución.

El [recibo](verification.json) identifica código, binario, fuentes, argumentos, versiones, medidas y hashes de los logs. Las comprobaciones usan la RTX 4070 Laptop de 8 GiB, PyTorch 2.14.0+cu130, runtime CUDA 13.0, cabeceras del Toolkit 13.4, Clang 21.1.8 y C++20. Compute Sanitizer informa versión 2026.3.0.0, build 38855100. Se mantiene `power-saver`, FP32 y FP64 sin TF32, un hilo por runtime y una carga CUDA propia simultánea. El escritorio, el navegador y el observatorio siguen activos.

## Atribución y cierre

El diagnóstico [`cuda_lifetimes.cpp`](../../../native/benchmarks/cuda_lifetimes.cpp) cruza los segmentos y bloques del asignador con las direcciones de los mapas de espacios de trabajo cuBLAS/cuBLASLt. Consulta también el asignador de host fijado. Se sincroniza después de destruir los tensores. Los mapas se leen con su mutex compartido. Las direcciones se contrastan con las del mismo proceso instrumentado, no con otro proceso que reserve la misma cantidad.

| Carga bajo memcheck | Repeticiones | Bytes señalados al salir | Asignaciones | Errores de acceso detectados |
| --- | ---: | ---: | ---: | ---: |
| Inicialización sin operaciones | 1 | 0 | 0 | 0 |
| GEMM y backward | 1 | 136.316.929 | 8 | 0 |
| GRU aislada y backward | 30 | 25.179.665 | 10 | 0 |
| Candidato y recuperación | 30 | 167.788.049 | 20 | 0 |
| Candidato, liberación de workspaces y recreación | 1 + 1 | 167.788.049 | 20 | 0 |

Las veinte asignaciones del candidato se reparten en 67.108.864 bytes de workspaces de PyTorch, 33.554.432 bytes de caché inactiva de dispositivo, 67.110.912 bytes cuya pila pasa por creación de handles cuBLAS, 13.840 bytes por creación de handles cuDNN y un byte de caché host de PyTorch. Ningún bloque activo del asignador queda sin referencia de workspace. La suma incluye nueve bytes de host fijado, por lo que no es una medida de VRAM.

Los mapas de cuBLAS se conservan como objetos globales. En la revisión de PyTorch usada, el [pool de handles](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CublasHandlePool.cpp) tiene un destructor deliberadamente vacío. La implementación de [cuDNN](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cudnn/Handle.cpp) tampoco invoca `cudnnDestroy`, por problemas documentados del orden de cierre. El [asignador host](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/cuda/CachingHostAllocator.cpp) mantiene sus bloques para reutilización. Las pilas de creación atribuyen esas reservas al proveedor, pero no permiten inspeccionar todos sus estados internos.

`clearCublasWorkspaces()` reduce los bytes activos del asignador de 67.108.864 a cero. Los 100.663.296 bytes reservados permanecen en la caché, y las asignaciones internas de los proveedores siguen presentes al salir. Repetir la carga recrea los workspaces y conserva la paridad. Por tanto, esta llamada aislada no elimina el informe de cierre ni acredita una mejora de rendimiento. Solo existe como modo explícito del diagnóstico. No se añade al predictor ni a los entrenadores.

Se conserva la salida 99 de Compute Sanitizer. No se han usado supresiones, `emptyCache`, destrucción manual de handles ni reinicios del contexto. El informe histórico de nueve asignaciones y 136.316.933 bytes permanece intacto. Sus dos reservas host, de uno y cuatro bytes, pasan por el asignador de host y la lectura de escalares. Ese proceso antiguo no emitía snapshots, así que la atribución por dirección solo corresponde a los ensayos nuevos.

## Paridad y límites de las comprobaciones

El fixture del candidato se comparte con la prueba CUDA original. Comprueba cuatro casos por precisión, K = 1 sin banco y K = 1, 2 y 4 con banco. Conserva salidas, gradientes, identificadores de recuperación y recuperación exacta del checkpoint. El máximo error de salida frente a CPU fue 6,66 × 10⁻¹⁶ en FP64 y 2,38 × 10⁻⁷ en FP32, tanto en 300 ciclos ordinarios como en 30 instrumentados. No se han ampliado tolerancias.

Los 300 ciclos necesitaron 179,50 segundos y un pico RSS de 1.375.571.968 bytes. Los 30 ciclos con memcheck necesitaron 305,54 segundos y 2.764.775.424 bytes de RSS. Son tiempos de proceso con instrumentación distinta, no una comparación de velocidad entre modelos. No se midieron pico de VRAM durante cada operación, percentiles de latencia ni energía. En todos los snapshots posteriores al primer ciclo fueron constantes los bytes activos, asignados y reservados de dispositivo y los activos y reservados de host.

Pasan las dos pruebas CUDA de CTest, las dos de opciones y rechazo de CLI y la prueba de opciones con ASan/UBSan y detección de fugas. Clang-Tidy y Clang Static Analyzer no emiten diagnósticos propios en las tres unidades revisadas. El compilador aplica los avisos estrictos y el análisis experimental de lifetime compatible con Clang 21. El intento de ejecutar el candidato CUDA con ASan/UBSan falla antes de las comprobaciones: `cudaGetDeviceCount()` devuelve error 2, `out of memory`. Esa ejecución no se cuenta como superada. Compute Sanitizer sí completa las cargas indicadas.

El [analizador](../../../scripts/analyze_cuda_lifetimes.py) tiene 37 pruebas. Rechaza snapshots incoherentes, logs truncados, totales incompatibles y registros de finalización incorrectos. Distingue ausencia de instrumentación de cero errores y no oculta crecimiento de memoria host. Cinco mutaciones dirigidas sobre direcciones, totales, propiedad, crecimiento y finalización fueron detectadas. Coverage.py 7.16.1 cubre 159 de 160 sentencias y 76 de 78 ramas. Radon 6.0.1 da complejidad máxima 28 y CRAP máximo 28, con la convención `C² × (1 − cobertura de sentencias)³ + C`. La sentencia no cubierta es la entrada de módulo. Estas medidas no acreditan que todo el programa esté libre de errores.

El resultado de esta investigación es una limitación de cierre atribuida y reproducible en la biblioteca, junto con comprobaciones del candidato. No se ha demostrado un defecto de pérdida de referencias en el código propio para estas cargas. No hay cambios de precisión, política de memoria o arquitectura de producción, ni resultados de aprendizaje.
