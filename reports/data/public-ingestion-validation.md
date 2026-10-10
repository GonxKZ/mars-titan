# Validación de la ingesta pública con instantáneas inmutables

Autor: Gonzalo García Lama. Comprobación: 10 de octubre de 2026.

Este informe recoge la evidencia de MT-067 (#69). El actualizador manual de fuentes públicas pasa a reutilizar contenidos ya confirmados, a enviar peticiones condicionales cuando el proveedor publicó validadores y a mantener un índice de capturas. Las capturas siguen fuera del benchmark congelado y ninguna decide por sí sola su incorporación a un experimento.

## Qué cambia en el actualizador

La lógica reutilizable está en [`public_snapshots.py`](../../src/mars_titan/data/public_snapshots.py) y el transporte sigue en [`refresh_public_sources.py`](../../scripts/refresh_public_sources.py), sin una segunda implementación de la descarga.

- **Índice derivado.** `data/manifests/public-snapshots/index.json` se reconstruye desde el manifiesto inicial y los `manifest.json` confirmados de `data/external/`. Guarda la huella de cada manifiesto, los registros que hacen falta para reutilizar y los contenidos únicos por SHA-256. Es una función pura de esos archivos, de modo que perderlo o escribirlo a medias no cambia ninguna captura. La siguiente ejecución lo vuelve a generar.
- **Reutilización por huella.** Si una respuesta válida tiene los mismos bytes que un contenido confirmado, el registro remite al archivo existente con `content_reused=true` y `reused_from_run_id`. El archivo se vuelve a leer y su huella se comprueba antes de aceptarlo. Un archivo alterado o ausente obliga a guardar otra copia.
- **Peticiones condicionales.** El ETag y el Last-Modified de la respuesta final, tras redirecciones, se guardan con la URL solicitada. La siguiente petición a la misma fuente y URL los envía como `If-None-Match` e `If-Modified-Since`, solo si la captura válida más reciente los trajo y su contenido sigue íntegro. Un 304 se registra como `not_modified` y vuelve a pasar por el validador de formato con el contenido anterior. Un 304 sin petición condicional es un fallo. Los valores con saltos de línea o caracteres de control no se reenvían.
- **Interrupciones.** Una carpeta de ejecución sin manifiesto no es una captura. Sus archivos no se reutilizan ni aportan validadores, y el manifiesto siguiente la lista en `previous_interrupted_runs`. Si falla la escritura del índice después del manifiesto, la captura queda confirmada y el índice se reconstruye en la próxima ejecución.
- **Presupuesto y parada.** Se mantienen los límites de 100 MiB por ejecución, 10 MiB y 45 segundos por fuente, la ausencia de reintentos y la suspensión del host tras 403 o 429. Se añade un techo de disco de 512 MiB para el almacén, que cuenta los archivos confirmados, las ejecuciones interrumpidas y lo nuevo. Al alcanzarlo, la fuente falla sin guardar bytes, aunque se puede seguir remitiendo a contenidos ya guardados.
- **Una sola actualización.** Un bloqueo de archivo sobre el almacén rechaza una segunda ejecución simultánea, que podría duplicar contenidos o escribir un índice sin la otra captura.
- **Disponibilidad.** Cada registro lleva `available_at_utc=null`. La hora de adquisición (`acquired_at_utc`) no acredita la publicación de cada dato, y `maturity` explica por qué la disponibilidad sigue sin establecerse. La confirmación de la ingesta es el `finished_at_utc` del manifiesto.

## Comprobación real con el almacén local

Se copió el almacén local (la captura inicial y las tres ejecuciones de septiembre) a la copia de trabajo de la rama y se lanzaron dos ejecuciones seguidas con la selección por defecto de ocho fuentes. La primera no tenía validadores previos porque los manifiestos de septiembre no los guardaban.

| Fuente | 1.ª ejecución `20261010T123641.516510Z` | 2.ª ejecución `20261010T123722.075681Z` | Validadores observados |
| --- | --- | --- | --- |
| FRED-MD | 200, contenido nuevo (edición `2026-09-md.csv`, 812 filas) | 200, mismos bytes, remite a la 1.ª | Ninguno |
| FRED-QD | 200, contenido nuevo (edición `2026-09-qd.csv`) | 200, mismos bytes, remite a la 1.ª | Ninguno |
| French diario | 200, contenido nuevo hasta el 31 de agosto de 2026 | 304, contenido anterior validado | ETag y Last-Modified |
| BCE USD/EUR | 200, 65 observaciones hasta el 9 de octubre | 304, contenido anterior validado | Last-Modified |
| Cboe VIX | 200, 9.291 filas hasta el 9 de octubre | 304, contenido anterior validado | ETag débil y Last-Modified |
| Treasury | 200, siete sesiones de octubre | 200, mismos bytes, remite a la 1.ª | Ninguno |
| GSCPI por versiones | 200, 349 filas hasta septiembre de 2026 | 200, mismos bytes, remite a la 1.ª | Ninguno |
| RSS monetario de la Fed | 200, 15 entradas | 304, contenido anterior validado | ETag y Last-Modified |

La primera ejecución transfirió 2.232.380 bytes y guardó 2.004.320. Ningún archivo coincidía con septiembre porque todas las fuentes habían publicado datos nuevos. La segunda transfirió 1.557.122 bytes, un 30,2 % menos, y no guardó ninguno. Esa cifra incluye la página de FRED que resuelve los enlaces `current.csv`, que se descarga siempre. Las ocho fuentes terminaron en estado válido y la orden devolvió código 0, también con las cuatro respuestas 304.

La reducción depende de cada proveedor. FRED, Treasury y New York Fed no enviaron validadores en estas respuestas, así que siguen descargándose enteros y solo se ahorra el disco. Las consultas del BCE y de Treasury cambian de URL con la fecha, por lo que sus validadores solo sirven dentro del mismo día o mes.

Estas dos capturas se conservan fuera de Git en el directorio privado de estado de MARS-TITAN, con la misma estructura `data/external/<run_id>/`. Las huellas de sus manifiestos son `d84493012cef70c2…` y `118d24aefcf2aa98…`. El índice versionado de esta rama describe el almacén del proyecto, que todavía no las incluye. Si se incorporan a ese almacén, `uv run python -m mars_titan.data.public_snapshots` regenera el índice sin acceder a la red.

## Estado del almacén del proyecto

El [índice](../../data/manifests/public-snapshots/index.json) referencia el manifiesto inicial sin modificarlo (`9455a61254926edf…`) y las tres ejecuciones de septiembre. No hay ejecuciones interrumpidas.

| Medida | Bytes |
| --- | --- |
| Citados por registros válidos | 7.178.186 |
| Contenidos únicos por huella | 5.657.300 |
| Ocupados en disco por archivos confirmados | 7.178.186 |

Los 1.520.886 bytes de diferencia son copias repetidas de septiembre, anteriores a la reutilización: FRED-MD, FRED-QD, French, la matriz GSCPI y tres copias del RSS. Se conservan porque las capturas son inmutables. El índice remite a la primera copia íntegra de cada contenido.

## Pruebas

| Comprobación | Resultado |
| --- | --- |
| [`test_public_snapshots.py`](../../tests/data/test_public_snapshots.py) | 23 pruebas: descarga idéntica, 304 con revalidación, 304 sin condición, contenido cambiado, solo la última captura válida, URL distinta, copia corrompida antes y durante la ejecución, interrupción antes del manifiesto, fallo del índice, captura inicial, límite de disco con interrupciones, bloqueo, manifiesto corrupto, rutas fuera del proyecto, árbol intacto fuera del almacén, inyección de cabeceras y transporte con `--dump-header` |
| [`test_public_source_reconciliation.py`](../../tests/data/test_public_source_reconciliation.py) | 11 pruebas de la reconciliación de MT-066 |
| [`test_public_sources.py`](../../tests/tooling/test_public_sources.py) | Las 41 pruebas anteriores de formatos, límites y transporte, sin cambios |
| Mutación dirigida | 32 de 32 mutantes detectados sobre reutilización, validadores, 304, interrupciones, disco, bloqueo, cabeceras, índice y reconciliación |

Ninguna prueba accede a la red. La prueba del árbol comprueba que dos ejecuciones solo añaden su carpeta y el índice, con el resto de archivos del proyecto falso (un benchmark y el catálogo) intactos.

## Límites

- La captura nueva no se incorpora a ningún experimento. Hace falta una decisión explícita sobre versión, disponibilidad y derechos antes de evaluarla, y los datos actuales sin versiones no sirven como disponibilidad histórica.
- No se interpreta `Retry-After`, no hay reintentos y no se instala cron, demonio ni tarea en GitHub Actions. La ejecución sigue siendo manual.
- Un 304 confirma que el proveedor no ha cambiado ese recurso según sus validadores. No prueba que el contenido fuera el publicado en una fecha anterior.
- La extensión puede retirarse sin afectar al estudio principal. Ninguna canalización científica lee las carpetas de ejecución de esta herramienta ni el índice. Otras carpetas de `data/external/`, como las de las adquisiciones macro o de China, pertenecen a otros procesos y el índice las ignora.

```bash
uv run pytest tests/tooling/test_public_sources.py tests/data/test_public_snapshots.py tests/data/test_public_source_reconciliation.py
uv run python scripts/refresh_public_sources.py --source fed_monetary_rss
uv run python -m mars_titan.data.public_snapshots
```
