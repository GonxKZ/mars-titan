# Archivo de mecanismos y madurez por calendario

El código de este incremento está en `526e41fa6`. Añade un archivo sellado para conservar las emisiones y actualizaciones de mecanismos, una consulta de resultados aún pendientes en PT2 y la madurez prevista por calendario. Continúa [#701](https://github.com/GonxKZ/mars-titan/issues/701). El consumidor sigue siendo provisional. Faltan su publicación conjunta con `FinancialSession`, los cortes transaccionales y el selector aprendido.

Las pruebas y medidas usan fixtures identificados. No hay corpus real, ajustes, pasos de optimizador, estudio predictivo nuevo ni acceso al test final. No se deduce una mejora financiera de estas comprobaciones.

## Qué conserva el archivo

[`MechanismHistory`](../../../src/mars_titan/memory/mechanism_history.py) reutiliza `SessionArtifacts` y su sello nativo. Los eventos se agrupan en bloques inmutables. El manifiesto identifica modelo, configuración, generaciones contiguas, corte y bytes. Cada evento lleva su generación, tipo, corte, huella del registro nativo y payload JSON acotado. El módulo valida ese transporte. La correspondencia entre observaciones, acciones, puntos, cuantiles y resultados deberá comprobarla el enlace financiero.

El archivo no tiene otra cabeza de publicación. `append` prepara una propuesta y `live_references` enumera lo que una generación deberá conservar al podar. Las pruebas recuperan la primera emisión después de siete eventos, varios bloques, poda y restauración. Esto no acredita todavía la recuperación de una sesión financiera completa ni su confirmación atómica.

Por defecto admite 32 eventos por bloque, un objetivo de 4 MiB codificados, un máximo de 8 MiB de JSON por evento y 512 bloques. Los límites globales de archivos y bytes de `SessionArtifacts` también se aplican, incluidas las propuestas transitorias. Superarlos produce `HistoryCapacityError` y no permite borrar evidencia para continuar. Estos presupuestos no son una cota del pico de RAM ni garantizan que quepa cualquier universo histórico.

Los eventos se codifican una sola vez como JSON canónico dentro de bytes. Al ampliar el bloque no se reconstruyen sus payloads anteriores. La caché conserva únicamente firmas de archivo y resúmenes, con un máximo de `max_chunks + 2` entradas. Si cambia la firma, se verifica de nuevo. La recuperación fuerte relee todos los bloques. Las huellas detectan corrupción, no autentican un origen hostil.

La opción `zlib1` usa compresión sin pérdida y fija la versión de zlib en la identidad. La descompresión limita la salida mediante `max_length` y rechaza flujos incompletos, sobrantes o demasiado grandes. La [referencia de Python 3.12](https://docs.python.org/3.12/library/zlib.html) documenta esa API. `bufsize` no sustituye ese límite. Se verificaron los bits de valores FP64, incluidos cero con signo, subnormales y el máximo finito, junto con enteros de 64 bits y texto Unicode. `none` sigue siendo el valor por defecto.

## Frontera temporal

`CohortFeedback.unresolved_keys` distingue las etiquetas que PT2 aún espera de las que ya recibió y de toda la cola financiera. La consulta no exporta los cuantiles. Una cohorte cancelada no autoriza perder las pérdidas financieras válidas de otras filas.

`PrefixTargetVerifier.expected_maturity` usa la siguiente sesión del calendario admitido. Respeta festivos, horario de verano y medias sesiones sin consultar retornos futuros. En el contrato auditado, la disponibilidad de precios coincide con la decisión cinco minutos después del cierre. El calendario no garantiza que exista el resultado. Sin siguiente sesión anterior al periodo reservado devuelve `None`.

El consumidor conserva las filas sin esa madurez y registra `unavailable_label_calendar` cuando retorna de una acción online. Los otros mercados mantienen su capacidad. Las acciones estáticas conservan su ruta. La recepción real de resultados continúa sujeta a la validación nativa que falta conectar.

## Comprobaciones

La [selección de regresión](verification.json) pasó 629 casos en 152,76 s, sin fallos ni omisiones. Incluye los 63 casos marcados CUDA, identificados por colección y conciliados con el XML aprobado. Los subprocesos internos no se cuentan como nuevos casos. El archivo aporta 47 casos. Hay pruebas de tipos exactos, continuidad, capacidad antes de escribir, corrupción con tamaño y `mtime` restaurados, caché acotada, aislamiento y recuperación fuerte. La selección conserva una advertencia de PyTorch sobre su diagnóstico experimental de sincronización.

Las [16 mutaciones dirigidas](mutations.json) quedaron detectadas, con 63 casos aprobados en la referencia inicial y la restaurada de la copia CPU. Un caso CUDA quedó fuera de esa selección CPU y se ejecutó en la regresión completa. Los fallos introducidos cubren claves recibidas, madurez, reparto de capacidad, tipos, integridad, huecos históricos, tamaño JSON, descompresión, conservación numérica y trabajo repetido. Son mutaciones seleccionadas, no todas las posibles. Las funciones de prueba no cambiaron después de medirlas. El formato y un import sin uso sí se corrigieron antes de la regresión final.

La [cobertura y CRAP](quality.json) usan coverage.py 7.16.1 y Radon 6.0.1. Sobre seis módulos y 74 funciones se ejecutaron 894/963 sentencias y 283/350 ramas. CRAP emplea cobertura de sentencias. El máximo es 47,62 en `FrozenMechanismConsumer.begin`. En el archivo nuevo es 30,27 en `verify`. No se afirma cobertura completa del proyecto ni ausencia de errores.

El desarrollo conservó los intentos fallidos. La primera colección del test de calendario falló por su import antes de comprobar comportamiento. Se trasladó al conjunto de fixtures que utiliza. Dos casos iniciales del archivo tenían un conflicto en el nombre del argumento de su factoría. El cargador de la referencia necesitó el contexto real del paquete para resolver su import relativo. El primer fixture de coste tenía un punto distinto de su mediana. Se corrigió y esa ronda se conserva como diagnóstico. Esos errores de preparación no se cuentan como detecciones de mutantes. Los [artefactos externos](external-artifacts.json) identifican logs, XML, recibos originales y medidas de cobertura por sus huellas.

## Coste del archivo

El [benchmark](benchmark.py) mide empaquetado, verificación incremental, serialización, escritura sellada y poda conservando dos generaciones de referencias. Añade contenido distinto en cada paso. Usa 8 o 64 eventos previos y 64 o 256 filas por evento, dos calentamientos y doce repeticiones. La recuperación fuerte y el digest semántico se miden fuera de cada latencia y dentro del proceso completo. No hay predictor, GPU, aprendizaje ni publicación de `FinancialSession`.

Se preserva la [referencia inicial de diccionarios](dictionary_reference.py) para reproducir el coste anterior. Solo es una referencia de este experimento. La [comparación](comparison.json) concilia el mismo digest semántico en las cuatro condiciones para diccionarios, bytes sin compresión y bytes con `zlib1`.

La primera ronda utilizó `/tmp`, que en este equipo es `tmpfs`. Sus recibos se conservan como coste en memoria: [diccionarios](cost-tmpfs-dictionary.json), [bytes](cost-tmpfs-none.json) y [compresión](cost-tmpfs-zlib1.json). No se presentan como latencia de SSD. Después se guardó el [protocolo de almacenamiento](cost-protocol.json) y se repitieron las tres versiones sobre ext4 en el SSD local, sin otras pruebas o perfiles propios simultáneos. Los recibos de [diccionarios](cost-ext4-dictionary.json), [bytes](cost-ext4-none.json) y [compresión](cost-ext4-zlib1.json) conservan las muestras. La tabla recoge p50 en milisegundos de esa segunda ronda.

| Eventos previos | Filas por evento | Diccionarios | Bytes sin comprimir | Bytes con zlib1 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 64 | 324,08 | 9,04 | 9,32 |
| 8 | 256 | 994,26 | 40,14 | 30,49 |
| 64 | 64 | 148,98 | 13,63 | 13,94 |
| 64 | 256 | 879,88 | 48,06 | 30,56 |

Evitar la reconstrucción del historial reduce el trabajo observado en las cuatro cargas. La compresión ahorra bytes pero no gana siempre en latencia frente a bytes sin comprimir. En la condición de 64 eventos previos y 64 filas, su p95 pasa de 15,59 a 17,53 ms. No se elige una configuración a partir de su mejor repetición.

En estos fixtures variados, `zlib1` reduce los bytes del historial entre un 29,76 % y un 30,19 % respecto a diccionarios y entre un 57,65 % y un 58,42 % respecto a bytes sin comprimir. En la condición mayor son 6.320.903, 10.613.127 y 4.412.487 bytes, respectivamente. Se cuentan por separado los bytes de las propuestas retenidas. La codificación sin compresión requiere más espacio que la referencia de diccionarios. Las primeras pruebas con campos constantes eran mucho más compresibles y no representan el ahorro en datos reales.

El tiempo externo de las cuatro cargas sobre ext4 fue 134,27 s, 10,78 s y 9,74 s. Incluye arranque, recuperación fuerte y lectura completa para el digest. `/usr/bin/time -v` registró picos de RSS de 674.078.720, 670.330.880 y 674.295.808 bytes. No se observa un ahorro de RAM relevante atribuible a la compresión. Los recibos distinguen este contador externo del máximo observado dentro de Python.

El equipo es un Ryzen 9 8945HS, con 16 hilos lógicos y unos 32 GB de RAM. El archivo es una carga CPU de serialización y almacenamiento. Las comprobaciones neuronales utilizaron explícitamente `cuda:0`, RTX 4070 Laptop de 8.188 MiB, controlador 595.91.07 y PyTorch 2.14.0+cu130. Se conservó la aplicación gráfica existente. No se controlaron frecuencias, temperatura o cachés, no se midió energía y el orden de versiones fue fijo. Los percentiles proceden de pocos eventos cuyo historial crece, no de una distribución estacionaria de peticiones. Falta medir el recorrido financiero integrado.

## Reproducción

En el entorno del proyecto, con el enlace nativo ya configurado, la selección completa se enumera en `verification.json`. Las pruebas nuevas pueden ejecutarse con:

```bash
MARS_TITAN_REQUIRE_CUDA=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:. uv run --no-sync pytest -q tests/calibration/test_cohort_unresolved.py tests/memory/test_prefix_maturity_calendar.py tests/models/titans/test_mechanism_maturity.py tests/memory/test_mechanism_history.py tests/memory/test_mechanism_history_encoding.py
```

Para medir almacenamiento persistente hay que fijar `TMPDIR` a un directorio existente del dispositivo que se quiere comprobar. Cada salida debe ser nueva. La referencia conserva su comando propio:

```bash
PYTHONPATH=src:. uv run --no-sync python reports/engineering/mechanism-history-20261021/benchmark.py --implementation dictionary --output /tmp/mars-history-dictionary.json
PYTHONPATH=src:. uv run --no-sync python reports/engineering/mechanism-history-20261021/benchmark.py --compression none --output /tmp/mars-history-bytes.json
PYTHONPATH=src:. uv run --no-sync python reports/engineering/mechanism-history-20261021/benchmark.py --compression zlib1 --output /tmp/mars-history-zlib1.json
```

El formato del benchmark publicado tiene el mismo AST que el ejecutado en la ronda `tmpfs`. Solo cambió la distribución de sus cadenas para cumplir Ruff. La ronda ext4 usó el archivo publicado sin cambios. Las huellas medidas y publicadas se conservan.

La integración durable debe añadir estado auxiliar e historial a la misma generación nativa, validar su semántica financiera, conservar las referencias al podar y comprobar cortes antes y después de publicar. El archivo aislado no completa ese trabajo. El selector aprendido y su evaluación emparejada también siguen pendientes. Las variantes anteriores continúan identificadas y desactivadas. La campaña completa y el test final permanecen cerrados.
