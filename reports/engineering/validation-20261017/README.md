# Verificación técnica local del 17 de octubre de 2026

Se conciliaron 12.010 casos de pytest: 11.398 superados y 612 omitidos con motivo registrado. No quedaron casos sin clasificar ni fallos de pytest sin resolver. La comparación usa identificadores de caso, por lo que repetir una prueba no aumenta ese total. El [recibo](verification.json) conserva los recuentos por selección, los motivos de omisión, las huellas de los informes y los fallos iniciales. Los casos que requieren aprendizaje permanecen bloqueados y no cuentan como superados.

La suite general se ejecutó sobre `df1324982290564083e72f39913e04e706b7fa79`. Las correcciones se comprobaron por separado sobre la rama de la [PR #647](https://github.com/GonxKZ/mars-titan/pull/647). Este informe no presenta un entrenamiento, una comparación científica ni una mejora predictiva. El test final de 2024 siguió cerrado.

## Selección y resultados

La recogida normal contiene 11.961 casos y la recogida explícita añade 49. También se contrastaron las definiciones `test_*` de los archivos Python con ambos inventarios. No quedaron definiciones fuera de esa recogida. El programa CUDA independiente del candidato y CTest se registran aparte.

El Makefile omitía `tests/hardware`, dos archivos de la raíz y las comprobaciones `cuda_*_check.py`. Ahora incluye esas rutas, fija `cuda:0` en los objetivos CUDA y admite rutas con espacios para los binarios. Sus cinco pruebas comprueban las órdenes que realmente ejecuta. La guarda de aprendizaje declaraba siete usos del optimizador registrador en un archivo que ya tenía ocho. Se corrigió el inventario y se repitieron los dos casos que habían fallado.

La tabla conserva la primera ejecución de cada bloque CPU. Los dos fallos de la raíz se muestran aunque sus repeticiones ya estén resueltas. Hardware, las repeticiones dirigidas y CUDA aparecen en el JSON. Las pruebas adicionales con la edición real verifican contabilidad y paridad con acciones fijas, sin evaluar una política aprendida.

| Bloque CPU | Superadas | Omitidas | Fallos iniciales | Segundos de pytest |
| --- | ---: | ---: | ---: | ---: |
| root | 52 | 1 | 2 | 164,06 |
| data | 1974 | 7 | 0 | 328,31 |
| models | 1485 | 175 | 0 | 161,28 |
| training | 2554 | 223 | 0 | 5106,01 |
| simulation | 1146 | 102 | 0 | 588,01 |
| memory | 710 | 0 | 0 | 323,45 |
| posttraining | 587 | 67 | 0 | 702,58 |
| evaluation | 1284 | 0 | 0 | 108,99 |
| environments | 137 | 0 | 0 | 27,55 |
| episodes | 33 | 0 | 0 | 2,46 |
| calibration | 57 | 0 | 0 | 1,75 |
| cm | 199 | 0 | 0 | 8,99 |
| integrity | 305 | 2 | 0 | 110,33 |
| learning_traces | 53 | 3 | 0 | 16,11 |
| native | 24 | 0 | 0 | 79,91 |
| tooling | 473 | 0 | 0 | 67,92 |

Las comprobaciones CUDA abarcaron la selección marcada, los 16 archivos explícitos, los codificadores congelados, el programa candidato y las medidas de memoria. Los casos que necesitan pasos de optimizador no se activaron. Ocho omisiones también requieren `hmmlearn 0.3.3`, que no está en `uv.lock`, y ajustan un HMM. El recibo distingue cada intento y no convierte una omisión en un aprobado.

Los avisos también se revisaron. Gymnasium emitió dos por límites infinitos del espacio de observaciones. PyTorch informó de la inicialización del contexto primario de cuBLAS en un hilo de autograd y del carácter parcial de su modo de diagnóstico de sincronizaciones. No se cambiaron límites de observación ni se silenciaron esos avisos para obtener otra salida.

Un aviso distinto, sobre `AccumulateGrad` y streams, permitió localizar una retención innecesaria en `ReferenceStepGraph`. Tras calcular el backward, la salida capturada seguía enlazada a sus nodos de autograd cuando el siguiente lote parcial usaba la ruta eager. Ahora conserva el búfer mediante [`detach`](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html), que comparte el almacenamiento sin conservar ese grafo. La prueba nueva falla en la versión anterior. Los 16 casos CUDA y los cinco casos CPU del módulo pasan con el aviso tratado como error. La comparación entre la captura y la ruta eager conserva salidas, gradientes y RNG bit a bit.

En un diagnóstico con un grafo reutilizado por proceso, las salidas de RNN y Transformer retenían respectivamente 66 y 163 nodos de autograd y ahora no retienen ninguno. El pico CUDA medido fue igual entre versiones. No se acredita una aceleración: las medianas de seis lotes fueron 14,53 y 14,54 ms para RNN, y 30,05 y 34,76 ms para Transformer, con dispersión y carga CPU concurrente. Coverage.py 7.16.1 registra 70 de 72 sentencias y 20 de 22 ramas del módulo. Radon 6.0.1 da CCN máximo 9 y CRAP máximo 9,070 con la cobertura de sentencias por función. Las dos líneas sin cubrir son rechazos de entrada anteriores al lanzamiento. Estas cifras son diagnósticos, no una garantía de corrección.

## Diagnósticos nativos

CTest completó 44 casos CPU del conjunto de controles, 19 del candidato y el caso CUDA del candidato. La comprobación CUDA del modo de solo evaluación de políticas se ejecutó directamente. La selección CUDA del build de controles devolvió código 8 porque no tenía casos registrados, y queda identificada como tal.

ASan y UBSan completaron 58 casos, con detección de fugas y pruebas breves de fuzzing. Los cuatro casos afectados por las primeras correcciones se repitieron con el código nuevo. La preparación de reglas se comprobó después con 95 casos de integración bajo ASan/UBSan y seis paridades sobre datos de 2023. Otras cinco comprobaciones explícitas de la edición real revisaron las cintas y la contabilidad.

TSan completó seis de sus ocho casos con los límites originales. `simulation` y `episodic_projection` superaron esos límites y, al repetirlos sin retirar aserciones, terminaron en 64,69 y 132,44 segundos. Sus límites instrumentados pasan a 240 y 480 segundos. Sin instrumentación conservan 60 y 120 segundos.

MSan también agotó los límites de esos dos casos con `-O0`, incluso al ampliarlos a 900 segundos. La configuración `-O1 -g`, con libc++ y libc++abi instrumentadas, completó los ocho casos. Un control negativo siguió detectando una lectura de memoria sin inicializar. La [guía de LLVM](https://clang.llvm.org/docs/MemorySanitizer.html#usage) recomienda `-O1` o superior para un coste razonable. Este cambio no modifica Release ni desactiva la instrumentación.

Se revisaron los 57 archivos C++ de `native/src`, incluidos los dos enlaces con Python, y el control CUDA nuevo. Se corrigieron los avisos de nombres de trazas, argumentos del lanzador y acceso a acumuladores. La sección `.text` del núcleo de medoids conserva sus 4.493 bytes y la misma huella en Release. Tres excepciones locales del linter documentan dos firmas públicas cuyo orden se conserva y el indicador atómico que debe cambiar tras `fork`. No se desactivaron familias de diagnósticos. El análisis de rutas no emitió hallazgos en los artefactos registrados. Su objetivo general incluye ahora el lanzador de aprendizaje, con dos pruebas de su ejecución y de su conexión.

Los runtimes precompilados de terceros no están completamente instrumentados. Estas comprobaciones corresponden al equipo Linux utilizado y no acreditan pruebas en Windows u otras arquitecturas.

## Memoria y tiempo medidos

Equipo: Ryzen 9 8945HS, RTX 4070 Laptop de 8 GiB, controlador 595.91.07 y perfil `power-saver`. Python 3.12.14, PyTorch 2.14.0+cu130, NumPy 2.5.3, PyArrow 25.0.1, Clang 21.1.8 y clang-tidy 21.1.6. Se mantuvieron dos hilos por proceso Python y una carga GPU propia a la vez. Los límites de memoria acotan las ejecuciones, pero no son mediciones del pico.

En la GRU candidata con banco episódico, la recomputación conservó exactamente los gradientes del caso sin acumulación y redujo el pico incremental del asignador CUDA. Las entradas eran fixtures con las dimensiones históricas. Las llamadas al optimizador eran registros sin modificación de parámetros.

| Activos por instante | Sin recomputación, MiB | Con recomputación, MiB | Gradientes iguales bit a bit |
| --- | ---: | ---: | --- |
| 128 | 355,44 | 144,14 | Sí |
| 256 | 621,15 | 150,75 | Sí |
| 512 | 1150,78 | 163,97 | Sí |

Las variantes que cambian el orden de acumulación presentan una diferencia máxima de gradiente de 8,94 × 10⁻⁸. La tabla no representa toda la VRAM del proceso. El mayor pico de cgroup observado fue 2.689.122.304 bytes y el mayor muestreo NVML, realizado cada segundo, fue 1.474 MiB. Este muestreo puede perder picos breves.

Titans-MAC y sus controles también se midieron con 128 flujos y ocho instantes, sin optimizador. El JSON separa un diagnóstico inicial sin calentamiento y otro con dos calentamientos y siete repeticiones por variante. Registra grafo conservado, parámetros, asignador, forward, backward y tiempo total del diagnóstico. No incluye lectura de corpus, colas de madurez ni recuperación del recorrido completo. No es una comparación de calidad entre arquitecturas.

Una caché experimental de la máscara de atención conservó salidas y gradientes, pero no mostró ventaja en las dos formas medidas. No se incorporó al núcleo. La [preparación de reglas](../native-rule-preparation-20261017.json) sí redujo trabajo repetido: evita copiar intervalos, analiza cada fecha una sola vez y reserva buffers con el tamaño conocido. Con 30 fechas pasó de 27 a 11 asignaciones y de 4.566 a 1.974 bytes solicitados por llamada. El recibo separa el contador de asignaciones del cronometraje Release y conserva ambos programas de medida. Son bytes acumulados y coste de esa función, no un ahorro demostrado en todo el predictor.

El recorrido CLI con una cinta sintética de cuatro activos chinos y acciones fijas se midió con dos calentamientos y 21 repeticiones por versión, alternando su orden. Incluye inicio del proceso, lectura, reconstrucción, simulación, auditoría y escritura de traza. Las medianas fueron 64,81 y 65,69 ms, con intervalos observados de 59,23 a 110,24 y de 59,61 a 75,84 ms. Los picos máximos de RSS fueron 36.992 y 37.004 KiB. No demuestra una aceleración del recorrido. Las trazas coinciden bit a bit salvo por la identidad del binario, comprobada por separado. La contabilidad coincide con la referencia Python. El intento inicial que comparaba también esas identidades distintas se conserva en el recibo.

Los tiempos tuvieron carga CPU concurrente y páginas de archivos calientes. No se midieron energía ni una mejora general del caudal del sistema. Las configuraciones científicas y sus presupuestos no cambiaron por estas medidas.

## Asignaciones CUDA pendientes

Compute Sanitizer mantiene código 99. El candidato produjo 167.788.049 bytes en veinte asignaciones al salir. Un [control que enlaza solo LibTorch](../../../native/benchmarks/torch_cuda_allocation.cpp) reprodujo 136.316.928 bytes en siete asignaciones. Sus 30 repeticiones mantuvieron constantes los bytes asignados y reservados. El candidato también mantuvo constantes sus valores en 30 repeticiones con paridad y recuperación comprobadas.

No se vaciaron cachés, suprimieron avisos ni reiniciaron contextos para obtener otra salida. La [incidencia #202](https://github.com/GonxKZ/mars-titan/issues/202) continúa abierta. El control orienta la atribución de una parte de las asignaciones, pero no explica por sí solo todas las del candidato ni acredita ausencia general de fugas.

## Alcance matemático y reproducción

El [contraejemplo formal](../../research/lean/README.md) y el [cálculo racional del calentamiento](../../../docs/research/titans-memory-certificate.md) precisan las condiciones de PT1. La contracción del escritor con las mismas entradas efectivas no se extiende automáticamente a la atención realimentada de MAC. El export Lean pasó los verificadores registrados. Se conservan también el fallo de resolución aislada de dependencias y el límite agotado de `--fresh`. No se verificaron por ello las familias 325 y 125 ni se demostró novedad o superioridad predictiva.

Las órdenes de los lanzadores están en [tests/README.md](../../../tests/README.md) y los perfiles de diagnóstico, en [native/README.md](../../../native/README.md). El recibo conserva las versiones y las huellas de los binarios y de cada resultado. Su campo `diagnostic_reproduction` guarda el texto exacto de los programas auxiliares, que reutilizan los módulos y fixtures del repositorio. Se ejecutaron desde su raíz mediante `uv run --no-sync --offline python`, con los programas en un directorio de salida privado. Los logs, XML, perfiles y datos voluminosos permanecen fuera de Git.

## Observatorio y documentos

Los 55 casos de los módulos del observatorio pasaron con Node 24.21.0. Playwright 1.63.0 y Chrome 153.0.8010.47 comprobaron la página sobre registros ficticios. La primera ejecución pública pasó 19 de 20 casos: al cerrar el detalle, una publicación de estado podía sustituir una navegación por URL todavía pendiente. La prueba nueva fuerza ese orden de eventos y fallaba antes de la corrección. El observatorio lee ahora la URL vigente antes de aplicar el cambio del detalle. Una guarda adicional no fue necesaria en los recorridos comprobados y se retiró tras sobrevivir a una mutación dirigida. La versión final pasa 22 de 22 casos públicos y ocho de ocho en modo directo. Incluyen teclado, móvil, CSV, texto hostil, recuperación de red y protección de la fase reservada.

La generación local del documento terminó con referencias resueltas en su último log. Los avisos de referencias de las primeras pasadas no se confundieron con el estado final. El README también concilia las vistas v3.1, los quince modelos de MARS-TITAN, B6 y las opciones de memoria con la configuración vigente. Ninguna de estas comprobaciones ejecutó cargas científicas en GitHub Actions.
