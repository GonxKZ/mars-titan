# Fuentes públicas complementarias y primera instantánea

Autor: Gonzalo García Lama. Adquisición: 18 de septiembre de 2026.

Se han descargado y validado nueve archivos de ocho proveedores, con un tamaño conjunto de 5.149.883 bytes (4,91 MiB). Aportan paneles macroeconómicos, factores de mercado, divisas, volatilidad, curvas de tipos, una matriz de ediciones de GSCPI, titulares monetarios oficiales y un documento financiero de Apple con tablas. La adquisición se ha ejecutado sin claves, cuentas nuevas, suscripciones ni pagos. No se ha incorporado ninguno de estos archivos al benchmark congelado ni se ha entrenado un modelo con ellos.

El [catálogo de fuentes](../../data/catalogs/public-sources.json) conserva catorce registros, incluidos endpoints fallidos, alternativas públicas y el descubrimiento bloqueado de Stooq. El [manifiesto](../../data/manifests/public-snapshots.json) registra las peticiones, los códigos HTTP, el formato comprobado, tamaño, SHA256 y ruta de cada archivo válido. La ampliación documental conserva sin cambios los ocho archivos iniciales, que sumaban 2.001.180 bytes, y añade un PDF de 3.148.703 bytes, dentro del límite adicional de 10 MiB. Las copias están en `data/external/2026-09-18/`, para consulta local. No se presupone autorización para redistribuirlas.

## Qué se ha obtenido

Las fechas de las siete primeras filas son periodos de observación, no fechas de publicación. En el RSS son fechas de publicación declaradas por el feed. En el PDF se separan periodo económico y publicación del comunicado. Las filas y columnas se han contado leyendo los archivos, sin convertir los valores ausentes en ceros.

| Fuente y archivo | Contenido validado | Cobertura de la copia | Aportación y límite |
| --- | --- | --- | --- |
| FRED-MD, `fred-md-2026-08.csv` | 811 filas mensuales, 126 series y una columna de fecha. | Enero de 1959 a julio de 2026. | Contexto macro estadounidense amplio. Es la edición corriente revisada, con 967 celdas numéricas ausentes. |
| FRED-QD, `fred-qd-2026-08.csv` | 270 filas trimestrales, 245 series y fecha. | Primer trimestre de 1959 a segundo de 2026, según las etiquetas del proveedor. | Cuentas y magnitudes de baja frecuencia. Las 2.135 ausencias impiden suponer un panel completo. |
| French, `french-factors-daily.zip` | ZIP íntegro y CSV con 26.296 fechas y cuatro columnas numéricas. | 1 de julio de 1926 a 31 de julio de 2026. | Mkt-RF, SMB, HML y RF. Retornos y factores agregados, no precios individuales ni una base de valores excluidos. |
| BCE, `ecb-usd-eur-2026-08-09.csv` | 35 observaciones de `EXR.D.USD.EUR.SP00.A` con metadatos SDMX. | 3 de agosto a 18 de septiembre de 2026. | Tipo de referencia USD por EUR, confirmado por los campos de moneda y unidad del CSV. No representa una operación ejecutable. |
| Cboe, `cboe-vix-history.csv` | 9.275 filas y apertura, máximo, mínimo y cierre de VIX. | 2 de enero de 1990 a 17 de septiembre de 2026. | Contexto de volatilidad implícita. No es un activo directamente negociable ni volatilidad futura realizada. |
| Treasury, `treasury-yields-2026-09.xml` | XML con doce observaciones diarias de la curva nominal. | 1 a 17 de septiembre de 2026. | Rendimientos par a diferentes vencimientos. No son precios ni retornos de bonos. |
| New York Fed, `nyfed-gscpi-vintages.csv` | 348 filas mensuales y 57 columnas de edición, de Jan-22 a Sep-26. | Septiembre de 1997 a agosto de 2026. | Permite estudiar revisiones entre columnas. Sus etiquetas mensuales todavía necesitan fecha y hora de publicación contrastadas. Hay 1.596 celdas ausentes. |
| Reserva Federal, `fed-monetary-press-rss.xml` | Quince entradas con título, enlace, fecha y resumen del propio feed. | 8 de abril a 16 de septiembre de 2026. | Metadatos y texto breve de comunicación monetaria oficial. No es un archivo completo de noticias financieras ni el cuerpo de las páginas enlazadas. |
| Apple, `apple-fy2026-q3-financial-statements.pdf` | PDF de tres páginas, validado con `pdfinfo`. Primera página renderizada y revisada. | Trimestre terminado el 27 de junio de 2026, comunicado del 30 de julio de 2026. | Estados financieros condensados no auditados con tablas y texto. Un emisor y un documento, sin extracción tabular estructurada ni historial completo de versiones. |

En las series CSV y en el ZIP de French no se detectaron fechas duplicadas en la granularidad inspeccionada ni fechas de referencia posteriores al día de descarga. Esto comprueba estructura y coherencia básica. No demuestra ausencia de revisiones, cobertura completa o corrección económica de todos los valores.

Las rutas de FRED etiquetadas como `current.csv` en la [página oficial](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) apuntaban a los archivos versionados `2026-rev-08-md.csv` y `2026-rev-08-qd.csv`. El manifiesto conserva esos destinos exactos. Sus primeras filas contienen códigos de transformación y otros metadatos, que no se han contado como observaciones.

## Fuentes inaccesibles y comprobaciones que evitaron falsas descargas

| Petición | Resultado observado | Decisión |
| --- | --- | --- |
| SEC submissions de Apple, CIK 0000320193 | HTTP 403. | No hay archivo de presentaciones descargado. Se conserva el fallo y no se cambia identidad ni se utiliza otra red para eludirlo. |
| SEC companyfacts de Apple | HTTP 403. | No hay nuevos fundamentales XBRL descargados. El endpoint permanece documentado para un acceso permitido posterior. |
| GDELT DOC, lista de hasta 25 noticias del 17 de septiembre | HTTP 200, pero la respuesta no superó el esquema esperado. Una consulta diagnóstica posterior recibió HTTP 429. | Se detuvieron las solicitudes. No se atribuye una lista de noticias válida a esta consulta ni se descargan artículos de los medios. |
| XLSX de GSCPI enlazado por la interfaz oficial | HTTP 200 y un ZIP que solo contenía un tema Office, sin `xl/workbook.xml` ni hojas. | Se rechazó como libro de datos. El código HTTP y la extensión no se consideran prueba de validez. |
| Primera validación del CSV alternativo de GSCPI | El validador no reconocía `#N/A` como ausencia. | Se conservó el intento fallido, se añadió ese marcador y se validó una nueva respuesta. No se sustituyó por cero. |
| Stooq, descubrimiento de un ejemplo EOD de AAPL.US | La portada devolvió HTTP 200 con una verificación de navegador mediante JavaScript. | No se pudo descubrir el enlace ofrecido ni revisar sus condiciones. Se detuvo el acceso, sin resolver el reto, cambiar identidad o deducir endpoints de terceros. No se descargó un CSV de precios. |

El CSV alternativo de GSCPI procede del recurso de datos utilizado por el JavaScript de su propia [interfaz oficial](https://www.newyorkfed.org/research/policy/gscpi). No se ha sorteado una restricción de acceso. La primera copia recibida como XLSX no tenía datos utilizables y el CSV público sí superó la comprobación estructural.

SEC [documenta](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) que sus API públicas no necesitan autenticación y que sus datos XBRL y de presentaciones se actualizan durante el día. Eso no garantiza acceso desde cualquier entorno. Se utilizó el identificador honesto `MARS-TITAN academic research (+https://github.com/GonxKZ/mars-titan)`, sin inventar un correo de contacto. Las peticiones fueron seriales y espaciadas, por debajo del máximo publicado de [diez solicitudes por segundo](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).

## Qué añade cada modalidad y qué sigue faltando

Los paneles FRED permiten ampliar la auditoría del [catálogo macroeconómico](macro-catalog.md) con observaciones reales. No equivalen a descargar las 140 variables de ese catálogo ni a validar su uso retrospectivo. Su información llega a ritmos distintos y hay series que comienzan más tarde o terminan antes. La fecha máxima de una tabla tampoco significa que todas sus columnas estén actualizadas hasta ese periodo.

Los factores de [French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) sirven para comparaciones de exposición y para revisar convenciones de construcción. El archivo corriente incorpora revisiones y cambios de metodología, incluido el paso FIZ/CIZ que documenta el proveedor. Un factor diario no sustituye automáticamente al retorno de mercado del intervalo apertura-cierre utilizado por el protocolo. Ese alineamiento debe resolverse antes de residualizar.

VIX, tipos y divisas aportan contexto financiero distinto de los precios de las acciones. La [página de Cboe](https://www.cboe.com/tradable-products/vix/vix-historical-data) distingue la metodología actual de VIX de su versión anterior VXO. La historia anterior a la introducción de un método requiere tratamiento explícito. La [documentación Treasury](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed) define consultas por mes y año, pero no acredita un archivo point-in-time de cada corrección. El [BCE](https://data.ecb.europa.eu/data/datasets/EXR/EXR.D.USD.EUR.SP00.A) publica un tipo de referencia, no un precio de bróker.

El RSS monetario de la Fed incorpora una modalidad textual pequeña y rastreable. Se conserva lo que el organismo distribuye en su [feed público](https://www.federalreserve.gov/feeds/feeds.htm), sin visitar ni descargar cuerpos de noticias. Si más adelante se usa GDELT, sus resultados seguirán siendo metadatos de descubrimiento. La fecha `seendate` no sustituye la publicación original y el límite de resultados impide tratar una respuesta como cobertura exhaustiva. Los derechos del medio sobre titulares, texto e imágenes no desaparecen por aparecer en un agregador.

El PDF financiero se descubrió en el enlace `View PDF` del [comunicado oficial de Apple](https://www.apple.com/newsroom/2026/07/apple-reports-third-quarter-results/). Es una fuente corporativa directa y distinta de las API SEC rechazadas. La primera página contiene cuentas de resultados, unidades, columnas comparativas de tres y nueve meses y desgloses por segmento y categoría. La revisión visual comprueba que esas tablas son legibles. No convierte el PDF en una tabla XBRL reconciliada ni acredita su disponibilidad en una hora histórica precisa. La fecha interna de creación del PDF tampoco sustituye la publicación.

Este ejemplo amplía el material documental, tabular y de presentación visual disponible para investigar extracción multimodal. No aporta una colección de gráficos de precios ni una nueva muestra representativa de empresas. Los derechos reservados y las [condiciones de Apple](https://www.apple.com/legal/internet-services/terms/site.html) se mantienen. El archivo ofrecido para descarga se conserva para consulta local, sin inferir autorización para republicarlo.

Esta adquisición no añade precios completos por activo, cotizaciones intradiarias en tiempo real, profundidad de mercado, disponibilidad de préstamo ni un universo histórico completo que incluya empresas excluidas. Tampoco reconstruye las normas de negociación o la pertenencia a índices de cada periodo. Las dos API SEC probadas no han aportado archivos utilizables en esta sesión.

China continúa con una brecha específica. El componente global de GSCPI puede reflejar condiciones chinas, pero no sustituye series NBS/PBOC ni publicaciones de empresas e instrumentos chinos. No se ha descargado una nueva base de acciones chinas ni acreditado simetría US/CN. El [catálogo macro](macro-catalog.md) mantiene sus candidatos pendientes y el alcance científico principal sigue delimitado por el protocolo.

## Separación temporal y derechos

Cada snapshot pertenece a una adquisición exploratoria de 2026. No reemplaza archivos existentes de FinMultiTime ni modifica su manifiesto congelado. Si una fuente llega a utilizarse en un experimento, se deberá seleccionar una copia concreta, registrar su hash, construir su disponibilidad histórica y cerrar la selección antes del test final. La actualización de fuentes y la evaluación de modelos son procesos separados.

Los archivos válidos se distinguen por su madurez:

- `latest_revised_not_point_in_time`: archivos corrientes de FRED, French, BCE, Cboe y Treasury. Su pasado puede incorporar información posterior a cada observación.
- `vintage_matrix_release_timestamps_pending`: matriz GSCPI con ediciones identificadas por mes. Falta verificar el instante real de cada edición y la preservación de sus columnas históricas. Los valores anteriores a 2022 siguen siendo un backcast anterior a su presentación pública.
- `current_feed_requires_publication_audit`: RSS actual. `pubDate`, captura y eventual modificación del comunicado deben mantenerse separados. Guardar hoy una entrada antigua no demuestra qué contenido existía entonces.
- `publisher_dated_document_version_history_unverified`: PDF vinculado a un comunicado con fecha conocida, pero sin hora original ni historial de sustituciones contrastados. El cierre contable no determina la disponibilidad.

El catálogo también reserva estados para metadatos SEC y descubrimiento GDELT, aunque estas peticiones hayan fallado. Todos los registros tienen `benchmark_eligible=false`. Ninguno se activa por compartir fecha de calendario con un precio. Horas y zonas originales se contrastarán y convertirán a UTC, aplicando las reglas del [contrato de datos](data-contract.md).

El acceso gratuito describe lo observado en estos endpoints, no una licencia universal. Cboe indica derechos reservados. French y los paneles distribuidos por FRED pueden incluir condiciones de productores originales. La licencia MIT de MARS-TITAN no cubre esos datos. En el catálogo, `license_unknown=true` indica que no se ha verificado un permiso suficiente de redistribución para esa copia.

El [BCE](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html) permite reutilizar su información bajo condiciones de exactitud, atribución y declaración de modificaciones, con excepciones. Por ello se documenta su permiso de forma específica. Incluso con una licencia de reutilización, la publicación de datos constituye una decisión separada de esta descarga local.

## Reutilización de la configuración

El catálogo separa URL de documentación, URL de descarga, formato esperado, validador, cadencia, condiciones, modalidad y limitaciones PIT. Fija un presupuesto de 100 MiB en total, 10 MiB por respuesta, 45 segundos por petición y ausencia de reintentos automáticos. Las alternativas llevan `fallback_for`, de modo que el XLSX inválido y el CSV válido de GSCPI no se confunden como dos magnitudes distintas. Stooq conserva `download_url=null` porque no se verificó un enlace de descarga. No debe incluirse entre los endpoints ejecutables.

El PDF de Apple tiene una URL fija de una publicación concreta. Obtener otro trimestre requerirá descubrir y contrastar su enlace oficial. La política `publisher_full_text=false` se refiere a artículos de medios de comunicación. El documento financiero oficial solicitado se identifica mediante `official_financial_documents=true`, con sus condiciones específicas.

La orden [refresh_public_sources.py](../../scripts/refresh_public_sources.py) reutiliza estos campos y crea una nueva carpeta fechada. Resuelve los enlaces corrientes de FRED en la página oficial, actualiza las fechas de consulta del BCE y Treasury y conserva peticiones fallidas sin sobrescribir capturas anteriores. Ante HTTP 403 o 429 suspende las peticiones al mismo host durante esa ejecución. No interpreta `Retry-After`, no reintenta automáticamente ni cambia de identidad. La validación de formato, tamaño, fechas, ausencias y hash precede a marcar una descarga como válida.

La [guía de actualización manual](public-source-updates.md) recoge las ocho fuentes renovables habilitadas y el PDF fijo que requiere selección explícita. La captura nueva del RSS oficial comprobó el recorrido completo de descarga, validación y manifiesto. No acredita la disponibilidad futura del resto de proveedores ni constituye una sincronización incremental.

El sondeo inicial se conserva en `tmp/fetch_public_probe.py`, como material temporal. La utilidad mantenida no es una implementación científica ni un servicio de datos en producción. La adquisición se apoya en HTTP y API públicas. Un conector MCP futuro no cambiaría las condiciones del proveedor, los límites de cobertura o la necesidad de conservar procedencia y versiones.
