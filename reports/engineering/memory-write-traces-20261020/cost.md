# Coste técnico de las trazas de memoria

Ocho casos de 64 y 256 filas, dimensión 64 y profundidad 2, en FP32 y cuda:0. Se comparan actualización por flujo y escritura compartida, sin ampliación y con PT5 más plasticidad. Estos últimos dos mecanismos cambian juntos entre esas filas y no se separa su coste mediante esta sonda. Dentro de cada caso se comparan las mismas entradas y parámetros, sin trazas, cada paso y cada 16 pasos. Se conserva exactamente el estado final y ningún parámetro cambia.

Son fixtures sin datos financieros, con 16 escrituras por trayectoria, tres escrituras de calentamiento y tres repeticiones con orden rotado. Cada llamada completa se sincroniza para medirla. El calentamiento cubre el cálculo, no inicializa previamente cada volcado Parquet. Los tiempos de volcado y de bloque se publican aparte. No hay pasos de optimizador ni acceso al test final.

| Filas | Escritura | PT5 y plasticidad | p50 sin trazas (ms) | p50 cada paso (ms) | p50 cada 16 (ms) |
| ---: | --- | --- | ---: | ---: | ---: |
| 64 | Por flujo | No | 1,787 | 4,781 | 1,961 |
| 64 | Por flujo | Sí | 3,910 | 8,215 | 3,934 |
| 64 | Compartida | No | 1,618 | 4,507 | 1,614 |
| 64 | Compartida | Sí | 3,657 | 8,109 | 3,565 |
| 256 | Por flujo | No | 2,226 | 4,773 | 2,173 |
| 256 | Por flujo | Sí | 5,092 | 9,373 | 4,724 |
| 256 | Compartida | No | 3,234 | 7,980 | 3,298 |
| 256 | Compartida | Sí | 6,707 | 14,071 | 6,714 |

Los percentiles describen 48 llamadas por condición y caso. El p50 de una cadencia de 16 pasos oculta el paso que registra y no estima su sobrecoste amortizado. Los pequeños descensos de algunas medianas no se interpretan como aceleración. Se conservan p95, p99, tiempos de los tres bloques y del volcado, bytes y pico del asignador en [cost.json](cost.json). No se extrapolan estos valores a una decisión financiera completa, al entrenamiento ni a latencias extremas.

La sonda final consumió 8.895 segundos internos y alcanzó 1.304.633.344 bytes de RSS de proceso. El escritorio siguió activo. No se midieron energía ni volumen total de transferencias. La cadencia de la campaña no se activa ni se selecciona con este fixture.

La primera sonda conservaba referencias de un caso al siguiente al reiniciar el contador de pico. Se preservó fuera de Git y se repitió tras corregir esas vidas. Sus comparaciones de coste no se utilizan. [cost-audit.json](cost-audit.json) identifica ambas fuentes y sus tiempos internos, sin esconder la sonda sustituida ni sumarla al presupuesto científico.

El [programa ejecutado](benchmark.py) mantiene la misma huella que la sonda final. Acepta `--output` con una carpeta nueva, exige la RTX y rechaza otro proceso de cómputo CUDA. Registra fuentes, software, configuración y presencia de procesos. Se ejecuta localmente con uv y el entorno CUDA del proyecto. Los artefactos generados deben permanecer fuera de Git.

Esta medida cubre el interior de la memoria con parámetros iniciales. La paridad de todas las familias y su sobrecoste sobre las formas completas de v3.1 siguen pendientes en #448.
