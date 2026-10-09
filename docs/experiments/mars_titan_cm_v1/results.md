# Comprobaciones técnicas de CM-v1

El 8 de octubre de 2026 se ejecutaron 77 pruebas CPU de los mecanismos aislados, sin construir modelos, optimizadores ni recorridos de aprendizaje. Pasaron los casos de matriz cero, normal, nilpotente y no normal, el contraejemplo alternante, las derivadas numéricas fuera de empates y los casos degenerados comprobados como finitud. Las pruebas también cubren entradas NaN/Inf, conversiones de precisión, presupuestos y conservación del RNG ajeno.

M se contrastó con un oráculo independiente de enumeración pequeña. Se probaron candidatos diferentes de los clientes, IDs compartidos, duplicados con multiplicidad, empates, permutaciones, ambas distancias, capacidades, límites y desbordamientos. Un caso L1 entero pasa de coste greedy 86 a 77 mediante dos intercambios. Es un fixture técnico y no una mejora predictiva.

La suite detectó 15 mutaciones dirigidas en corrección angular, autovalor extremo, gradientes, memoria retenida, norma espectral, distancia euclídea, identidad compartida, desbordamiento entero, límite de pares, multiplicidad, orden canónico, intercambios, enumeración y tamaño de los IDs Unicode. Se ejecutaron en copias aisladas, comprobando el origen de importación.

La estimación de autograd incluye las fases y las reducciones, también para matrices escalares. Una sonda de 27 combinaciones de orden, rejilla y bloque no superó la estimación, tanto al sumar tensores guardados como al contar almacenamientos únicos. La corrección de este desglose conserva exactamente valores y gradientes en 12 casos de cuatro tipos y tres formas. Las medidas de tiempo siguientes corresponden al cálculo anterior a ese ajuste del contador de bytes.

Coverage.py 7.16.2 registró sentencias y ramas. Radon 6.0.1 calculó complejidad ciclomática. CRAP se calculó como `CCN² (1 - cobertura_sentencias)³ + CCN`, dentro del rango de cada función. C alcanzó 100 % de sentencias y ramas, con máximo CCN y CRAP 17. M alcanzó 98,81 % de sentencias y 97,37 % de ramas, con máximo CCN 16 y CRAP 16,021. Estos diagnósticos no demuestran ausencia de defectos.

## Coste CPU medido

El [benchmark reproducible](../../../scripts/benchmark_cm_mechanisms.py) utiliza NumPy 2.5.3, PyTorch 2.14.0+cu130 y Python 3.12.14 sobre un AMD Ryzen 9 8945HS. Todas las operaciones se ejecutaron en CPU. La unidad local tenía una cuota del 200 % de CPU y un límite de proceso de 2 GiB. Se utilizaron dos hilos Torch y de las bibliotecas numéricas, con un hilo interop. No se controlaron otras aplicaciones del equipo.

C usa 64 ángulos y bloques de 8. Se realizó un calentamiento y cinco repeticiones por forma. La variante con derivada calcula una penalización y su gradiente, sin actualización de parámetros.

| Orden | Matrices | Derivada | Mediana | Intervalo observado |
| --- | ---: | --- | ---: | ---: |
| 16 | 1 | No | 1,08 ms | 1,07–1,14 ms |
| 32 | 2 | Sí | 15,90 ms | 15,79–17,15 ms |
| 64 | 1 | No | 14,92 ms | 14,36–16,32 ms |

M usa distancia euclídea, capacidad indicada y candidatos sintéticos distintos de los clientes. Se realizó un calentamiento y tres repeticiones por forma. El coste final se contrastó por separado con `math.dist` y `math.fsum`, con tolerancia relativa `1e-14`.

| Clientes | Candidatos | Dimensión | Capacidad | Mediana | Pico rastreado | Buffers estimados |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 16 | 4 | 3 | 2,24 ms | 184.720 B | 243.968 B |
| 1.024 | 32 | 8 | 4 | 46,41 ms | 608.240 B | 832.000 B |
| 4.096 | 64 | 8 | 4 | 242,23 ms | 1.843.008 B | 3.094.528 B |

En la última forma se calcularon 2.048.000 pares de distancia, unos 8,45 millones de pares por segundo según la mediana. El volumen de coordenadas de entrada fue de 266.240 bytes. El tiempo total del comando, incluidas importaciones, fue 3,64 segundos. El recorrido medido internamente consumió 2,37 segundos, incluidos calentamientos, medidas adicionales y referencias. El máximo RSS observado por `time` fue 653.952 KiB para el comando completo. `tracemalloc` registra asignaciones rastreadas de Python y NumPy, no todos los espacios de trabajo nativos ni el RSS. Los tamaños anteriores son cargas técnicas declaradas, no una medición del banco episódico integrado.

Para repetir pruebas y medidas en un entorno preparado:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync --offline pytest -q tests/cm
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync --offline python scripts/benchmark_cm_mechanisms.py
```

El benchmark imprime las repeticiones, dispersión, versiones, formas, buffers y límites en JSON. No se midieron energía, coste económico, VRAM o transferencias. No se sustituye una implementación anterior ni se afirma una aceleración.

## Factorial sobre Titans-MAC

El 9 de octubre de 2026 se comprobó en CPU la implementación del [factorial](factorial.md), con `CUDA_VISIBLE_DEVICES=-1`, dos hilos y un archivo de pruebas cada vez. Ninguna prueba ejecuta pasos de optimizador. Los bucles llegan al paso con un registrador de gradientes que no modifica pesos y las entradas de ventana se recorren con la protección del aprendizaje simulada.

Las pruebas nuevas cubren la penalización en el entrenador (11), las lecturas del operador (22), los Jacobianos completos (10), el factorial (19), su campaña (17), la paridad con la sesión también con centros fijos y las protecciones de las tres entradas. La regresión de las 49 suites afectadas, de C/M, Titans, sesiones, campaña, comparación, entrenadores y postentrenamiento, dio 1.297 pruebas superadas y 7 omitidas por falta de CUDA o por el bloqueo. El ensayo de la comprobación CUDA en CPU superó sus 4 casos, lo que no acredita CUDA. Después de rebasar sobre `develop` con la etapa de políticas se repitieron las suites que comparten plan, campaña, medición o protecciones, junto con las de la etapa de políticas, y pasaron 345 pruebas con 2 omitidas por falta de CUDA o por el bloqueo. Entre ellas está la que exige que la etapa de políticas admita un brazo del factorial como predictor y rechace un núcleo auxiliar.

La mutación dirigida aplicó 26 cambios en una copia aislada, comprobando que las pruebas importaban el código de esa copia. En la primera pasada se detectaron 21 de 25. Tres supervivientes mostraban huecos de prueba: el contador de términos descartados sin etiquetas, la salida de los núcleos auxiliares y una declaración cambiada al trasladar. Las pruebas añadidas los detectan. La cuarta era equivalente, porque en `first_read` la búsqueda restringida ya devuelve las posiciones del primer paso. Se retiró esa elección redundante y la mutación que elimina la restricción se detecta. Las mutaciones detectadas cambiaban la media de los términos por su suma u omitían el término, dejaban términos de un tramo en el siguiente, fijaban la selección por bloque, ignoraban el contador de cada flujo, conservaban términos en validación, aceptaban el diagnóstico o la acumulación, invertían el objetivo de la identidad, omitían el control del padre o el gemelo disabled, dejaban peso en B, quitaban la retención con centros fijos, ajustaban el núcleo de C sin C, cambiaban el núcleo de B+M, daban traslado o recibo a los auxiliares, quitaban el factor 2 del operador fijo, invertían el orden de los productos, marcaban como contraejemplo un operador expansivo o movían la base del refinamiento.

Estas comprobaciones son técnicas. No miden memoria ni caudal en `cuda:0`, no ajustan ningún brazo y no aportan un efecto de C o de M sobre el error.

## Comprobaciones pendientes

La [comprobación CUDA de C](../../../reports/research/cm-mechanisms-verification-20261008.json) contrasta float32, float64, complex64 y complex128 sobre dos matrices 2×2, con 17 ángulos y bloques de tres. Se comprobaron valores y gradientes de las tres penalizaciones. La diferencia máxima observada de valores fue `6,66e-16` y la de gradientes inferior a `1e-15`. El producto alternante mantiene el autovalor 2,25. El pico fue de 17,33 MB asignados por Torch, con contexto CUDA adicional y una aplicación externa activa. Esto no certifica cotas ni mide velocidad de un modelo.

El [control local de MAC](mac_local_control.md) ya identifica la transición conjunta de pesos rápidos y momentum y calcula `RᵀJR` mediante productos Jacobiano-vector. Se han ejecutado 295 pruebas CPU de Titans y C/M, 23 sondas independientes y seis casos CUDA de modos y precisiones. Los casos CUDA comprueban gradientes seleccionados, RNG y recuperación exacta de la siguiente predicción y actualización rápida. El [recibo](../../../reports/research/mac-local-control-verification-20261008.json) distingue las versiones y los costes técnicos. El diagnóstico comprimido puede omitir direcciones expansivas y no certifica estabilidad.

El [consumidor financiero congelado](../../engineering/financial-session-v2.md) integra núcleo, banco, lectura, observaciones sin etiqueta y recuperación combinada bajo M0/M1. Su [recibo](../../../reports/engineering/financial-session-completion-20261008.json) registra pruebas CPU y CUDA, el contrato numérico explícito y los fallos corregidos. La selección M conserva su backend CPU. La composición factorial con sus controles está implementada en el [factorial](factorial.md). Quedan pendientes su ejecución y el estudio científico. Estas comprobaciones no prueban una mejora de error ni estabilidad global.

La [composición de M2 con el diagnóstico C](../../../reports/engineering/m2-local-control-20261009.json) pasa una prueba CPU con ocho recorridos de cuatro flujos, FP64, K=1 y B_mem=4. Cada modo de C recorre una referencia en lotes de cuatro filas, el orden inverso de activos y etiquetas, filas individuales en orden inverso y una continuación tras un corte antes de publicar. En cada evento confirmado, el diagnóstico conserva exactamente predicciones, estado rápido, índices U/S/R, errores selectivos, archivos nativos del banco, episodios y etiquetas pendientes frente al modo desactivado. La comparación excluye solo los identificadores de modelo y de predicción, que incorporan la configuración de C. El orden inverso y la recuperación reproducen exactamente la referencia. Las filas individuales conservan los IDs y usan tolerancia `1e-12` para las cantidades flotantes. En los cinco eventos de decisión, C mide los dos mismos flujos con la selección fijada para el grupo lógico. Los RNG globales de Torch, NumPy y Python no cambian, y una guarda hace fallar la prueba si se construye un optimizador o se llama a la retropropagación.

Siete mutaciones temporales de la sesión financiera hicieron fallar la prueba. Omitían las mediciones, alteraban una emisión, seleccionaban flujos por bloque físico, consumían el RNG de Torch o de NumPy, construían un optimizador o modificaban las claves pendientes. La versión anterior de la prueba no detectaba el consumo de NumPy ni el optimizador. La validación de producción rechaza por sí misma las claves pendientes modificadas, por lo que esa mutación no acredita la nueva comparación del banco. No se ejecutaron pasos de optimizador ni se estimaron etiquetas. La penalización durante aprendizaje, CUDA de esta combinación y su efecto predictivo siguen sin comprobarse.

No se ha reproducido la prueba formal de los manuscritos auditados ni su algoritmo de aproximación k-median. No hay aprendizaje, evaluación financiera, resultado de validación ni evidencia de estabilidad de MARS-TITAN atribuible a estas funciones.
