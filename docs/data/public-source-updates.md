# Actualización manual de fuentes públicas

Autor: Gonzalo García Lama.

La orden [refresh_public_sources.py](../../scripts/refresh_public_sources.py) crea capturas fechadas de los complementos del [catálogo público](../../data/catalogs/public-sources.json). Es una herramienta de adquisición y validación de formatos. No implementa modelos, no transforma el objetivo financiero y no incorpora datos al benchmark. Tampoco modifica `dataset/`, las capturas iniciales ni `data/manifests/public-snapshots.json`.

## Uso desde la raíz del proyecto

```bash
uv run python scripts/refresh_public_sources.py --list
uv run python scripts/refresh_public_sources.py --source fed_monetary_rss
uv run python scripts/refresh_public_sources.py --source fred_md --source ecb_usd_eur
uv run python scripts/refresh_public_sources.py
```

`--list` lee solo el catálogo local y no hace peticiones ni crea directorios de captura. La salida indica qué fuentes se incluyen por defecto y cuáles admiten selección explícita. Se puede repetir `--source` para seleccionar varias fuentes. Un identificador desconocido, bloqueado o sin validación previa se rechaza antes de descargar.

Por defecto se seleccionan fuentes con `status=downloaded_validated`, `auth=none` y un validador admitido. Los documentos financieros PDF se excluyen de esa selección. Las fuentes que devolvieron errores de SEC, GDELT o un XLSX inválido no se reintentan automáticamente. Una fuente solo se habilitará después de una nueva comprobación explícita del catálogo.

Para capturar de nuevo el documento fijo de Apple se requiere seleccionarlo:

```bash
uv run python scripts/refresh_public_sources.py --source apple_fy2026_q3_financials
```

Su registro conserva `refresh_mode=fixed_document`. Descargar ese enlace otra vez no descubre el trimestre siguiente. El script no busca automáticamente nuevos informes corporativos ni afirma que un archivo sin cambios contenga resultados más recientes.

## Qué se actualiza en cada ejecución

| Fuente | Resolución de la descarga |
| --- | --- |
| FRED-MD y FRED-QD | Se consulta una vez la página oficial del catálogo y se resuelven sus enlaces etiquetados `current.csv`, distinguiendo las rutas mensual y trimestral. No se reutiliza a ciegas la dirección de agosto de 2026. |
| Tipo USD/EUR del BCE | Se reconstruyen `startPeriod` y `endPeriod` para 90 fechas naturales, incluida la fecha UTC de ejecución. Se conservan los demás parámetros de la consulta. |
| Curva del Treasury | Se sustituye el parámetro de mes por el mes UTC actual. Las demás opciones permanecen en la consulta. |
| Factores French, VIX, matriz GSCPI y RSS oficial | Se consulta el enlace corriente de cada fuente. Su cadencia editorial determina cuándo cambian realmente los datos. |
| PDF financiero | Se conserva el documento expresamente identificado. Solo se adquiere mediante selección explícita. |

El periodo más reciente del archivo puede ser anterior al día de adquisición. La etiqueta `current.csv` indica qué edición ofrece el proveedor, no que la observación se haya publicado hoy. El script no construye URLs de vintages por conjetura si no encuentra el enlace oficial.

## Capturas y procedencia

Cada ejecución crea un directorio nuevo con identificador UTC de microsegundos, por ejemplo `data/external/20260918T160000.123456Z/`. Una colisión de nombre se rechaza. Los archivos completos se publican sin reemplazar destinos existentes y después se escribe el manifiesto de esa ejecución. Un directorio sin manifiesto completo no constituye una captura confirmada.

El manifiesto registra el hash del catálogo, sus metadatos por fuente, URL del catálogo, URL solicitada y URL efectiva, estado HTTP, tipo de contenido, tiempo de adquisición, bytes y SHA-256. Añade el resultado del validador, la madurez temporal y cualquier fallo. En FRED conserva también la URL y el hash de la página usada para resolver el enlace. Las rutas locales del manifiesto son relativas a la raíz del proyecto.

Todos los registros conservan `benchmark_eligible=false`. Una descarga válida no acredita un dato point-in-time. Hay que auditar revisiones, publicaciones, ajustes y correspondencia con activos antes de usarla en un experimento. La matriz de vintages GSCPI, por ejemplo, contiene etiquetas mensuales que no equivalen a horas exactas de publicación.

Estas capturas no son una sincronización incremental. No se presupone ahorro mediante ETag, ni se reemplazan archivos antiguos cuando el contenido sea idéntico. Los hashes permiten comprobar posteriormente esa igualdad. El crecimiento de disco debe revisarse antes de programar descargas recurrentes.

## Límites y comprobaciones

La ejecución es serial. El catálogo puede reducir los límites, pero no elevarlos por encima de 100 MiB transferidos por ejecución, 10 MiB por fuente y 45 segundos de presupuesto de peticiones por fuente. El descubrimiento de FRED consume presupuesto y no se repite para la segunda frecuencia. Se cuenta también el cuerpo de las respuestas fallidas cuando el transporte informa de sus bytes. Se deja aproximadamente un segundo entre peticiones y no se realizan reintentos automáticos.

El transporte utiliza `curl` del sistema, desactiva su configuración personal y permite únicamente HTTPS, también en redirecciones. No solicita claves ni resuelve verificaciones de navegador. La inspección de PDF requiere además `pdfinfo`. Ambos programas procesan las respuestas como datos. El contenido descargado no se ejecuta como código.

Una respuesta 403 o 429 suspende nuevas peticiones al mismo host durante esa ejecución, incluidas otras URLs del catálogo. El manifiesto distingue la respuesta original de las fuentes omitidas sin petición. Otros proveedores pueden continuar. No se cambia de red ni de identidad para eludir el rechazo. La herramienta no interpreta `Retry-After` ni reanuda automáticamente al proveedor en ese proceso.

Los [validadores](../../scripts/public_source_formats.py) comprueban esquemas CSV conocidos, fechas, anchura de filas y valores numéricos finitos. Los marcadores explícitos de ausencia se cuentan y no se imputan. El RSS exige titulares, enlaces y fechas con zona horaria. Treasury exige observaciones fechadas y campos de rendimientos. El validador Stooq está preparado para una fuente OHLCV previamente validada, pero no habilita un endpoint bloqueado del catálogo.

XML rechaza `DOCTYPE`, declaraciones de entidades y codificaciones no admitidas. ZIP admite miembros almacenados o DEFLATE y verifica CRC, nombres, número de miembros y tamaño descomprimido, con techo de 50 MiB y rechazo de relaciones de compresión excesivas. Los errores de descompresión se registran como fallos de formato. Un ZIP sin estructura de libro no pasa por XLSX. En XLSX se comprueban componentes XML y celdas, sin evaluar fórmulas ni reconstruir el calendario de Excel. Los PDF requieren cabecera, final de archivo y aceptación por `pdfinfo`, sin extraer automáticamente tablas.

Si una fuente falla, no se publica un archivo de datos válido para ella. El manifiesto conserva el fallo y las demás fuentes pueden continuar dentro del presupuesto. La orden termina con código distinto de cero si cualquier fuente seleccionada falla. Esto permite detectar errores en una ejecución manual o futura programación, sin presentarlos como una actualización completa.

## Frecuencia de trabajo

La ejecución es manual. No se ha instalado cron, un servicio persistente ni una tarea en la nube. Para una futura programación se puede preparar una orden con ruta absoluta al proyecto y a `uv`, empezando por una fuente pequeña. Ejemplo de plantilla no instalada:

```cron
17 8 * * 1-5 cd "/RUTA/AL/PROYECTO" && /RUTA/A/uv run python scripts/refresh_public_sources.py --source fed_monetary_rss
```

Antes de utilizarla deben fijarse la zona horaria del programador, el tratamiento de códigos de salida, la retención de capturas y las condiciones del proveedor. La hora elegida no implica que un comunicado esté disponible ni que el mercado esté abierto. Una API o una interfaz MCP no garantiza actualidad, cobertura histórica o permiso de redistribución.

## Verificación de la herramienta

Las [pruebas sin red](../../tests/tooling/test_public_sources.py) cubren formatos, límites, enlaces de FRED, cambio de fechas, selección de fuentes, rutas locales, aislamiento de fallos y conservación de capturas existentes. Pueden ejecutarse con:

```bash
uv run pytest tests/tooling/test_public_sources.py
```

La descarga real de comprobación se limita a un RSS oficial pequeño. Su evidencia es el manifiesto de una ejecución nueva, no una prueba de disponibilidad futura de todas las fuentes. Las condiciones y derechos siguen siendo los del catálogo y de cada productor.

La captura inicial del actualizador, `20260918T193326.801397Z`, conserva una limitación anterior a esta corrección: registró el inicio con microsegundos y el fin truncado a segundos. Por ello su manifiesto puede aparentar que la ejecución terminó antes de comenzar. Se conserva intacto y no debe utilizarse para calcular duraciones subsegundo. La versión corregida registra las marcas UTC con precisión uniforme de microsegundos, comprobada mediante una prueba con reloj fijo.
