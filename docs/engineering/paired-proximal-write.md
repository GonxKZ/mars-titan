# Escritura proximal compartida de B2

La corrección W y el mapa de consultas T de un área de [B2](../experiments/area-query-bridge-v1.md) reciben las mismas claves, pesos, tasa y retención. La primera implementación resolvía ambas escrituras por separado. El perfil de tres ciclos con cuatro áreas registró 24 factorizaciones de Cholesky, aunque solo había doce sistemas distintos.

`memory/paired_proximal.py:write_proximal_pair` comparte los productos de las claves y la factorización. Mantiene separados los lados derechos y las llamadas a `cholesky_solve`. Así evita concatenar columnas o cambiar las dimensiones de cada solución respecto a la referencia. No añade otro factor experimental ni modifica los parámetros del predictor.

## Sistema común y validación

Para las claves K, pesos Ω, objetivos financieros Y y rasgos de destino Z:

$$
G=K^\mathsf{T}\Omega K,\qquad LL^\mathsf{T}=I+\eta G,
$$
$$
W^+=\operatorname{cholsolve}(\rho W+\eta K^\mathsf{T}\Omega Y,L),\qquad
T^+=\operatorname{cholsolve}(\rho T+\eta K^\mathsf{T}\Omega Z,L).
$$

La pareja recibe una única `MatureFeedback`, que define claves, metadatos y pesos comunes, y un segundo conjunto de valores. Exige dos estados proximales con la misma dimensión de clave, tasa, olvido, contador y cursor. La validación canónica comprueba la cohorte compartida. Los valores del segundo estado se comprueban por separado en forma, dispositivo, precisión y finitud.

Una cohorte vacía conserva ambos estados. Un error de factorización o un resultado no finito rechaza la pareja sin publicar ninguno. La función no cambia sus entradas ni guarda factores entre eventos. Los estados resultantes mantienen sus matrices, contadores, cursor y formato de serialización anteriores.

## Inserción en el recorrido

`AreaQueryBridgeState.write_corrections` aplica la pareja por área solo con la regla proximal. La escritura individual sigue disponible como referencia. B1 sin puente y la regla delta conservan su recorrido. El control con ganancia cero escribe los mismos mapas que B2 y recibe la misma optimización.

La selección de destinos de los mapas se calcula por arista. En el control barajado solo se conserva un buffer de destino a la vez. No se prepara una lista de copias de toda la cohorte por área. La generación mantiene el dominio RNG y el orden canónico registrados.

La sesión fija el código del helper en su identidad cuando B2 proximal lo utiliza. La declaración del modelo, las filas, las recetas y las políticas temporales no cambian. El presupuesto de estado persistente sigue siendo el de B2. La factorización compartida solo reduce trabajo y temporales de una llamada.

## Comprobaciones

La batería de integración aprobó 296 casos CPU y dejó pendientes los tres casos CUDA. Una ampliación posterior de 52 pruebas del helper incluye tipos de estado incorrectos y fallo comunicado por la biblioteca numérica. No cambió el código de producción entre esas dos ejecuciones.

Se compararon lecturas, escrituras y recuperación con el código real de `3e89cc5df`, en procesos separados. Los quince casos coinciden bit a bit e incluyen delta, proximal, ganancia cero, ambos barajados, dos y cuatro pasos, 512 rasgos, tasa cero y ausencia de olvido. Son comprobaciones sobre fixtures, sin una medida de calidad financiera.

El perfil posterior registra doce factorizaciones en esos tres ciclos. Las 24 soluciones se conservan y los productos matriciales pasan de 72 a 60 en el tramo combinado. Los tiempos del perfil instrumentado no se usan como benchmark.

El benchmark [paired_area_write.py](../../benchmarks/paired_area_write.py) carga el código de referencia desde el commit local en un espacio de nombres separado. Usa el mismo entorno numérico para ambas versiones, comprueba paridad y alterna el orden de las operaciones. El ciclo medido reúne lectura y escritura de la memoria. No incluye el predictor CUDA, la codificación ni la persistencia de una sesión.

## Medida antes y después

Se ejecutaron dos pasadas locales en Ryzen 9 8945HS, con PyTorch 2.14.0+cu130, FP64 y dos hilos. Cada caso usa cinco calentamientos y cuarenta repeticiones. Se midieron B2, ganancia cero y destinos barajados con 64, 256 y 1.024 filas. Ambas versiones conservan el mismo estado inicial y las mismas observaciones. Las [muestras y la comparación](../../reports/engineering/paired-proximal-write-20261026/comparison.json) incluyen los nueve casos de cada pasada.

| Pasada | Filas | Ciclo anterior p50 | Ciclo compartido p50 | Reducción |
| --- | ---: | ---: | ---: | ---: |
| 1 | 64 | 13,467 ms | 11,500 ms | 14,6 % |
| 1 | 256 | 46,969 ms | 40,861 ms | 13,0 % |
| 1 | 1.024 | 130,685 ms | 120,777 ms | 7,6 % |
| 2 | 64 | 19,776 ms | 16,656 ms | 15,8 % |
| 2 | 256 | 30,164 ms | 25,436 ms | 15,7 % |
| 2 | 1.024 | 82,311 ms | 77,038 ms | 6,4 % |

La mediana del ciclo bajó en los nueve casos de ambas pasadas. Los tiempos absolutos variaron con las cargas concurrentes, que no se detuvieron. Estas medidas no se comparan directamente con la tabla de B2 aislada, porque el benchmark conserva ambas implementaciones y mide su ejecución alternada. La ruta de lectura no se modificó y sus pequeñas diferencias de tiempo no se atribuyen a una optimización.

Se mantiene el estado numérico de 530.432 bytes. El [recibo de verificación](../../reports/engineering/paired-proximal-write-20261026/verification.json) separa paridad, perfil instrumentado y benchmark. El RSS de cada proceso incluye el entorno y ambas implementaciones, por lo que no mide su diferencia de pico. Tampoco se midió energía.

Las [once mutaciones dirigidas](../../reports/engineering/paired-proximal-write-20261026/mutations.json) fueron detectadas entre referencias verdes. Comprueban pesos, alineación, retención, tasa, maduración, overflow, estado de factorización, despacho, RNG, buffers y pin del código. Coverage.py 7.16.1 midió las 31 sentencias y diez ramas del helper. Radon 6.0.1 da complejidad 10 y CRAP 10. La [calidad del conjunto](../../reports/engineering/paired-proximal-write-20261026/quality.json) conserva las partes no cubiertas de los módulos relacionados. Estos diagnósticos no demuestran ausencia de errores.

La optimización se conserva por la reducción repetida del coste y la paridad observada. El predictor CUDA completo y las tres pruebas de interoperabilidad siguen pendientes. No hay mejora predictiva medida ni un cambio de política financiera.
