# Comprobación de la fusión regional

`2abecb549` incorpora `regional_fusion_v1` como componente opcional anterior a MAC. Sus cinco nodos reciben las representaciones de precios, noticias, gráficos, fundamentales y macro. Alterna pasos locales y actualizaciones de un resumen, con límites conjuntos de trabajo. La [especificación](../../../docs/experiments/regional-fusion-v1.md) describe las ecuaciones y los ocho controles. No hay aprendizaje, corpus real ni resultados predictivos en este informe.

El componente conserva los estados intermedios solo durante una predicción. No almacena recuerdos históricos en varias regiones y no aprende a escoger herramientas. El selector y sus contrastes siguen pendientes en [#701](https://github.com/GonxKZ/mars-titan/issues/701). La campaña completa y el test final permanecen cerrados.

## Comportamiento comprobado

Las pruebas contrastan la media enmascarada con un cálculo manual, las aristas locales, la influencia real del resumen en el ciclo siguiente y la diferencia entre frecuencias de actualización. Incluyen separación de activos, partición y permutación del lote, ausencia total, entradas ausentes extremas en FP64, gradientes por diferencias finitas, aislamiento del RNG, precisiones, dispositivos y rechazos de NaN/Inf. Los checkpoints regionales rechazan campos ausentes o extra, aristas distintas, formas, precisión y configuración incompatibles antes de copiar sus tensores.

La integración reutiliza el constructor cronológico y la cabeza de cuantiles. Se comprueba con las cuatro referencias financieras, entradas estrictas y enmascaradas, sin añadir transiciones MAC por las vueltas del grafo. El enlace M0/M1 con PT4 y PT2 publica sus contadores en el historial financiero. Un corte después de confirmar conserva el estado y reproduce la siguiente emisión. Las versiones anteriores y los runtimes históricos conservan sus identidades propias.

La comprobación de paridad utilizó una referencia capturada antes de editar el predictor. Sus 16 condiciones combinan cuatro referencias, CPU/CUDA y FP32/FP64, con tres observaciones consecutivas. Con la ampliación desactivada se conservaron exactamente parámetros, salidas, gradientes, estado rápido y configuración, incluida la lectura de esos pesos anteriores. Los [recibos de captura](baseline-capture.json) y [comparación](baseline-verify.json) identifican las fuentes por sus huellas. Ambos conservan el HEAD de captura porque la edición se comprobó antes del commit. Las huellas distintas del archivo distinguen las dos implementaciones. Los pesos de referencia permanecen fuera de Git.

La regresión inicial ejecutó 904 casos. Pasaron 902 y fallaron dos controles del inventario de excepciones de fixtures. Tres analizadores anteriores contenían cinco usos ya existentes de `learning_doubles` que el inventario no declaraba. `4fef82697` añadió sus recuentos 1, 3 y 1, tras comprobar que solo procesan fixtures. La repetición focalizada pasó 55 casos, incluidos los controles negativos de ese inventario. No se cambió la protección global ni se eliminaron sus rechazos. El [registro de comprobaciones](verification.json) conserva el fallo y la repetición por separado. No se presenta como una única ejecución completa sin fallos.

La unión por identidad de esa regresión, su repetición y las comprobaciones adicionales contiene 955 casos aprobados en su última ejecución. Incluye 89 casos regionales, 37 de ellos CUDA, y ocho casos del contrato de los benchmarks. Los 120 casos marcados CUDA de la selección ampliada se conciliaron con el XML, sin omisiones. No son 955 observaciones científicas ni una ejecución única. Hay una advertencia conocida del modo de diagnóstico de sincronización de PyTorch.

Las [13 mutaciones dirigidas](mutations.json) fueron detectadas. La referencia inicial y la copia restaurada pasaron 47 casos CPU cada una, sin omisiones. Se alteraron máscaras, número y orden de pasos, conexiones entre activos, gradientes, límite conjunto, detección de valores no finitos, configuración de checkpoints, inserción del módulo y contadores archivados. Las pruebas CUDA son independientes de esa selección de mutación. El resultado no es una cobertura de todas las mutaciones posibles.

La [cobertura y CRAP](quality.json) se obtienen con coverage.py y Radon mediante la herramienta común. La convención de CRAP utiliza cobertura de sentencias y la complejidad ciclomática de cada función. La selección cubre seis módulos, incluidos componentes anteriores. Las fronteras sin ejecutar permanecen visibles en el JSON. Estas medidas no acreditan ausencia de fallos ni precisión financiera.

Se ejecutaron 1.189/1.322 sentencias y 456/584 ramas en 110 funciones. En el módulo regional fueron 130/131 sentencias y 39/40 ramas. Su `forward` ejecutó todas sus sentencias y ramas, con complejidad 6 y CRAP 6. El mayor CRAP regional fue 15 en `_checked_presence`. En la configuración financiera anterior quedan rechazos sin recorrer y su CRAP es 403,60. El alcance de esta selección no permite afirmar cobertura completa de ese módulo previo.

## Medidas y reproducción

El [protocolo de coste](cost-protocol.json) se fijó antes de las medidas principales. Se utilizó `cuda:0` de la RTX 4070 Laptop, sin TF32 ni autocast y con algoritmos deterministas. Las condiciones se ejecutaron en orden fijo, sin pruebas, mutaciones ni perfiles propios simultáneos. No se cambió la carga gráfica del equipo ni se midió energía. Los percentiles son empíricos y su resolución está limitada por el número de repeticiones.

La [medida residente](operator-cost.json) compara ocho configuraciones con 64 y 256 filas, H = 64, FP32 y FP64, dos calentamientos y doce repeticiones. Los pesos se inicializan en FP32 y después se convierten a la precisión declarada. Se verifica la igualdad de parámetros compartidos entre configuraciones. La fusión incluye las proyecciones de las otras fuentes y excluye el codificador de precios. El consumidor incluye codificador, fusión, MAC y cabeza, con sus verificaciones. Las transferencias iniciales y la construcción del estado quedan fuera de esas latencias y dentro del tiempo de proceso.

La [medida de sesión](session-cost.json) incluye apertura, calentamiento, tres emisiones, dos recepciones de resultados, cierre censurado, publicación, poda, lectura histórica y recuperación. Usa H = 32, FP64, bloques de 64 y capacidad episódica 64, con almacenamiento ext4 en NVMe. Compara el mismo contexto PT4, PT2 online, parámetros del tronco y archivo, añadiendo solo el grafo regional en anillo. No incluye codificadores externos ni filas reales. Sus tiempos no se suman ni se restan de la medida residente, que usa otras formas y otro recorrido.

Para 256 filas residentes en FP32, estas son las medianas por llamada. Los JSON conservan también las otras precisiones, tamaños, percentiles y muestras individuales.

| Configuración | Fusión, ms | Consumidor, ms |
| --- | ---: | ---: |
| Original | 0,287 | 8,513 |
| Nodos aislados sin resumen | 2,649 | 11,907 |
| Mensajes uniformes y resumen | 3,175 | 12,504 |
| Atención completa y resumen | 2,845 | 11,141 |
| Anillo y resumen | 3,031 | 11,428 |
| Anillo sin resumen | 2,506 | 11,504 |
| Anillo, cuatro resúmenes | 3,585 | 11,854 |
| Anillo, un resumen | 2,778 | 11,788 |

La referencia almacena 98.247 parámetros y cada configuración regional 139.463. Los 41.216 parámetros adicionales no cambian los bytes de estado persistente por flujo. Para el anillo con resumen y 256 filas, el pico asignado del consumidor pasa de 125.569.536 a 125.745.664 bytes en FP32. Son 176.128 bytes adicionales en esta condición de inferencia, con parámetros congelados. No estima la memoria de entrenamiento con sus activaciones y gradientes.

La sesión completa produce estos tiempos por trayectoria, con seis repeticiones tras dos calentamientos:

| Filas por emisión | Configuración | p50, ms | p95, ms | Filas emitidas/s |
| --- | --- | ---: | ---: | ---: |
| 64 | Contexto y PT2 originales | 3.742,74 | 4.042,48 | 51,27 |
| 64 | Más anillo regional | 3.898,41 | 4.137,91 | 49,88 |
| 256 | Contexto y PT2 originales | 10.989,15 | 11.500,54 | 70,29 |
| 256 | Más anillo regional | 10.921,29 | 12.374,83 | 68,92 |

Con 256 filas, una mediana ligeramente menor coincide con peor media, caudal y cola. No se declara una aceleración. El archivo retenido pasa de 11.824.214 a 11.831.830 bytes y el pico de VRAM asignada de 80.438.784 a 80.523.264 bytes. Los contadores, la identidad y la configuración añadidos también ocupan espacio. El proceso residente tarda 8,87 s internos y 12,90 s externos. La medida completa tarda 246,01 s internos y 249,63 s externos. Los [costes de proceso](process-cost.json) registran los picos de RAM medidos por `/usr/bin/time`, separados de los del propio script.

Un [perfil instrumentado](profiles.json) de tres llamadas con 256 filas FP32 registra 756 llamadas a `cudaLaunchKernel` en la referencia y 1.419 con anillo. Las llamadas a atención pasan de 6 a 18. Hay más lanzamientos y primitivas pequeñas, además del control y las verificaciones del consumidor en CPU. El recuento de `_local_scalar_dense` permanece en seis. El perfil no aporta contadores suficientes para un Roofline ni demuestra saturación de cómputo o ancho de banda. No se ha sustituido una primitiva madura por un kernel propio ni se afirma que el recorrido esté optimizado hasta el límite del equipo.

El coste adicional está medido. Su utilidad predictiva sigue desconocida. Para contrastarla faltan entrenamientos emparejados y evaluación temporal autorizada sobre las mismas filas, con los controles de capacidad y de comunicación declarados. Este informe no selecciona un configuración mediante MAE ni habilita la campaña.

Los [artefactos externos](external-artifacts.json) conservan las huellas de logs, XML, cobertura y referencia de pesos. Incluyen las fases rojas y los errores de preparación. Una comparación inicial de FP32 construía un modelo primero en FP64 y otro directamente en FP32. Se corrigió el fixture para emparejar la inicialización, sin cambiar tolerancias ni el predictor. La paridad histórica de 16 condiciones se verificó con su referencia capturada. Esos errores no se cuentan como detecciones de mutantes.

Para repetir las pruebas regionales en el entorno CUDA del proyecto:

```bash
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_REQUIRE_CUDA=1 uv run --no-sync pytest -q tests/models/titans/test_regional_fusion.py tests/models/titans/test_financial_regional_fusion.py tests/memory/test_regional_fusion_session.py tests/tooling/test_regional_benchmarks.py
```

Los archivos de salida del coste deben ser nuevos. La medida de sesión necesita el backend episódico nativo configurado y un directorio de almacenamiento existente y verificado en el dispositivo que se quiera medir. `MARS_TITAN_COST_STORAGE` designa ese directorio. No debe apuntar a `tmpfs` si se pretende medir SSD.

```bash
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python reports/engineering/regional-fusion-20261021/benchmark.py --rows 64 256 --warmup 2 --repetitions 12 --output /tmp/mars-titan-regional-operator.json
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python reports/engineering/regional-fusion-20261021/session_benchmark.py --rows 64 256 --warmup 2 --repetitions 6 --storage-root "${MARS_TITAN_COST_STORAGE:?Se necesita un directorio de almacenamiento verificado}" --output /tmp/mars-titan-regional-session.json
```

Los scripts rechazan tamaños no admitidos, repeticiones excesivas, filas duplicadas y salidas incompatibles antes de acceder a CUDA. La construcción y las transferencias forman parte del tiempo total de proceso, aunque no todas pertenezcan a la latencia interna de cada recorrido. La VRAM de los recibos es la del asignador de PyTorch, no toda la memoria del dispositivo. Los picos de RAM proceden del proceso, no de una suma por configuración.

La revisión de vocabulario posterior detectó un término no admitido en la descripción del benchmark de sesión. Se corrigió solo ese texto. El [registro de revisión](source-revision.json) conserva las dos huellas y un parche que reconstruye la fuente medida. Los recibos mantienen su fuente original. La modificación no altera las operaciones ni justifica repetir las medidas de coste.
