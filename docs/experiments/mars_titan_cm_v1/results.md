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

## Comprobaciones pendientes

Faltan la selección del operador completo de Titans-MAC, la integración temporal del banco, la recuperación de su estado y la paridad con C y M desactivados. También faltan comprobaciones CUDA y la comparación CPU/CUDA de los caminos que se integren. Las pruebas CPU no las sustituyen.

No se ha reproducido la prueba formal de los manuscritos auditados ni su algoritmo de aproximación k-median. No hay aprendizaje, evaluación financiera, resultado de validación ni evidencia de estabilidad de MARS-TITAN atribuible a estas funciones.
