# Coste inicial del estudio de 64 y 256 activos

Cuatro núcleos Titans-MAC con proyección `linear_v1`, semilla 42 y dos épocas
completaron sus recorridos con el runtime `6b037a7262d4c2d97105b194a137f8b2f5767b01`.
El ajuste del padre usa las filas admitidas desde 2000 anteriores a abril de 2021,
con selección y calibración en 2021 y evaluación del padre en 2022.
El lector episódico y la política de consultas todavía no están evaluados en este recibo.

| Activos y representación | Tiempo externo (s) | Actualizaciones | Pico RSS (GiB) | Pico asignado por PyTorch en CUDA (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 64, original | 193,11 | 1.308 | 1,927 | 194,77 |
| 64, geometría | 189,99 | 1.308 | 1,915 | 194,77 |
| 256, original | 366,11 | 1.308 | 2,615 | 595,77 |
| 256, geometría | 360,06 | 1.308 | 2,629 | 595,77 |

El tiempo externo incluye lanzamiento, ejecución y comprobación del recibo.
Los picos CUDA pertenecen al asignador de PyTorch. No incluyen toda la memoria
del contexto del controlador. El RSS corresponde al proceso completo. Se utilizó
la RTX 4070 Laptop con PyTorch 2.14.0+cu130, FP32 sin TF32 y dos hilos CPU.
Durante la ronda hubo comprobaciones CPU de ingeniería en paralelo.

Pasar de 64 a 256 conserva 1.308 actualizaciones y amplía las filas con objetivo
de ajuste de 187.158 a 750.751 por época. El tiempo del núcleo original aumenta de
193,11 a 366,11 segundos. Es una observación de esta carga, con una ejecución por
combinación. No hay un intervalo de incertidumbre de rendimiento ni evidencia
de que la geometría acelere en general el ajuste.

Los cuatro intentos terminaron con recibos cerrados, sin parada forzada ni exceso
de sus límites. Quedan 56.490,73 segundos del presupuesto conjunto de 57.600.
La extrapolación de los núcleos usa un coste constante entre semillas y
proyecciones. Las etapas de lectores, colección, selección y controles aún no
tienen estimación propia. No se modifica el catálogo según las métricas de 2023.

[Recibo, hashes y presupuestos](verification.json). Las cuatro tareas pertenecen
al [protocolo de escalado](../../../docs/experiments/refinement-scaling-v1.md).
Su finalización no acredita una mejora predictiva del controlador ni completa
los 288 trabajos del catálogo. La campaña completa y el test final siguen cerrados.
