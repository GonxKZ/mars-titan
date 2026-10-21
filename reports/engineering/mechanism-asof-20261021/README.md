# Vista histórica de evidencias de mecanismos

Se añadió una lectura por generación y corte exclusivo a `FinancialSession`. Reconstruye resultados a partir de registros nativos confirmados, con alternativas previas a etiquetas, pendientes, exclusiones y censuras separadas. Permite consultar un prefijo de una sesión que después recibió más información, sin recurrir a su recibo actual. El código de producción comprobado es `ee5603ecc5e675f08482709490616e64a1fdb541`.

Una etiqueta con disponibilidad anterior pero recepción posterior queda pendiente en el prefijo. La vista completa conserva los resultados del recibo confirmado. Se comprueban las huellas, la correspondencia por activo y fecha, el ancla y los límites antes de cada lectura. No se modifica el predictor, la memoria ni el estado confirmado. La [descripción del contrato](../../../docs/engineering/mechanism-collection-asof.md) detalla sus límites.

## Comprobaciones ejecutadas

La regresión seleccionada pasó 348 casos en 367,98 segundos, incluidos 32 marcados CUDA. Una revisión posterior añadió dos casos CPU que también pasaron: esquema nativo alterado y cambio de cabeza confirmado mientras se lee. No hubo fallos, errores ni omisiones en esas ejecuciones seleccionadas. No es la suite completa del proyecto ni una evaluación financiera. El archivo nuevo contiene 27 casos, seis de ellos CUDA. Los datos y etiquetas son fixtures identificados y no hay pasos de optimizador.

Se detectaron doce mutaciones dirigidas. Once se ejecutaron en una copia aislada con 19 casos CPU pasando antes y después. La duodécima retiró la detección del cambio de cabeza, con su caso pasando antes y después y fallando bajo la mutación. No se presenta este resultado como cobertura de todas las mutaciones posibles.

La cobertura de ramas se midió con coverage.py 7.16.1. La complejidad y CRAP se calcularon con Radon 6.0.1 y `scripts/summarize_python_quality.py`, usando regiones y cobertura de sentencias de coverage.py. En los tres módulos seleccionados se ejecutaron 778/922 sentencias y 278/388 ramas. `collection_at` tiene 58/58 sentencias, 14/14 ramas, complejidad 21 y CRAP 21. Los operandos de cada condición no son casos independientes en esta convención. Las funciones antiguas de los módulos también aparecen en el [informe completo de calidad](quality.json), incluidas rutas asociativas y de admisión que esta selección no ejercita.

El primer ciclo detectó la ausencia de la API. Después se corrigió un intento de serializar directamente los bytes de la colección al calcular una huella JSON. La huella ahora incorpora la identidad del paquete, que cubre esos bytes. Se añadió correspondencia explícita entre filas y generaciones de recepción. Un caso de exclusión tenía inicialmente un prefijo admisible y se corrigió con entradas de prueba nulas en una fecha cuyo prefijo real es insuficiente. No se estimaron etiquetas residuales para construir estos fixtures. Los recibos de esas tentativas se conservan aparte y no se suman como pruebas adicionales.

## Coste observado

RTX 4070 Laptop de 8188 MiB, controlador 595.91.07, PyTorch 2.14.0+cu130, CUDA 13.0, Python 3.12.14 y dos hilos CPU. Se seleccionó `cuda:0` expresamente y se mantuvo FP64. El [protocolo](cost-protocol.json) fijó tres calentamientos y veinte repeticiones por modo, con orden rotado y sincronización CUDA por llamada. No hubo otras cargas propias durante las medidas y no se detuvieron aplicaciones ajenas. Esto no acredita exclusividad del equipo.

| Activos | Lectura | p50 (ms) | p95 (ms) | p99 (ms) | Llamadas/s | Paquete y registros (bytes) |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Recibo actual | 8.514 | 9.543 | 10.580 | 115.45 | No medido |
| 4 | Vista completa | 13.186 | 20.596 | 108.823 | 51.58 | 22658 |
| 4 | Vista parcial | 12.180 | 13.595 | 14.262 | 80.24 | 21193 |
| 64 | Recibo actual | 22.967 | 24.824 | 25.077 | 43.25 | No medido |
| 64 | Vista completa | 32.046 | 32.970 | 37.929 | 31.10 | 107299 |
| 64 | Vista parcial | 27.887 | 29.757 | 30.345 | 35.66 | 94641 |
| 256 | Recibo actual | 51.329 | 61.078 | 62.264 | 18.46 | No medido |
| 256 | Vista completa | 68.491 | 85.445 | 164.749 | 13.28 | 378284 |
| 256 | Vista parcial | 54.682 | 66.334 | 67.728 | 17.38 | 329815 |

El recibo actual y la vista completa tienen las mismas alternativas, resultados y censuras. La vista parcial conserva la mitad de las etiquetas como pendientes y se reproduce igual antes y después de recibir las restantes. Sus tiempos corresponden a trabajos distintos. La reconstrucción verifica más registros que la lectura del recibo actual y en esta medida tarda más. No se reivindica una aceleración. Las colas con veinte repeticiones son descriptivas, con valores extremos conservados en las muestras.

El proceso completo tardó 21,838 s desde el comienzo del procedimiento y 0:25.61 según la medida externa, que incluye el arranque. El pico registrado por Python fue 1547751424 bytes y el externo 1614336 KiB. Incluyen la creación de datos de prueba y el predictor residente. El pico del allocator CUDA durante las lecturas coincidió con los 67.395.072 bytes residentes en los tres tamaños. No es la VRAM total del sistema ni un presupuesto exclusivo del lector. La reserva del allocator se registra en [cost.json](cost.json). No se midió energía.

Los bytes de la tabla suman el paquete y los registros nativos. La integridad de los bloques de trazas se comprueba con los límites propios del archivo. Esos bloques pueden contener trazas posteriores y no se contabilizan en ese campo. El lector no toma resultados posteriores para construir su supervisión. Los bytes indicados no equivalen a todo el tráfico físico ni al pico de memoria.

La primera medida se conserva con su hash. Se repitió al añadir identidades de datos, parámetros, colección y registros al manifiesto, sin cambiar el código de producción medido. La tabla usa esta segunda ejecución por su trazabilidad, sin escoger los tiempos más favorables. Las dos ejecuciones quedan identificadas.

## Alcance pendiente

El [ajuste protegido](../mechanism-fit-20261021/README.md) mantiene su requisito de sesiones completas cerradas antes del final del tramo. La nueva vista no se ha integrado en ese lector ni demuestra que la normalización, el solver o el ajuste funcionen en su recorrido positivo. No se entrenó ningún selector y no hay mejora predictiva medida.

La API exige un máximo explícito de 256 registros. Solicitar muchas vistas repite recorridos. Falta un lector conjunto que amortice ventanas largas y un contrato de fuentes para usarlo en el ajuste. La huella de la vista es una comprobación de integridad, no una firma externa ni una prueba de aprendizaje. Continúan pendientes la selección temporal y la evaluación de una trayectoria autónoma. La campaña completa y el test final permanecen cerrados.

Los [resultados de pruebas](verification.json), [mutaciones](mutations.json), [calidad](quality.json), [coste](cost.json) y [hashes de artefactos externos](external-artifacts.json) separan las comprobaciones ejecutadas del trabajo pendiente. Los registros voluminosos y las muestras individuales siguen fuera de Git.

En un entorno local con la extensión nativa y CUDA compatibles ya configuradas, los puntos de entrada utilizados son:

```bash
uv run --no-sync pytest -q tests/memory/test_mechanism_collection_asof.py
uv run --no-sync python reports/engineering/mechanism-asof-20261021/benchmark.py --output /tmp/mars-asof-cost.json
```

El benchmark exige una ruta de salida nueva. No permite reemplazar un resultado anterior ni pasar implícitamente a CPU.
