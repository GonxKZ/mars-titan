# Validación de las fuentes públicas complementarias

Autor: Gonzalo García Lama. Reconciliación: 10 de octubre de 2026.

Este informe cierra la revisión de MT-066 (#68). Contrasta el [catálogo público](../../data/catalogs/public-sources.json) con los manifiestos y los archivos conservados, y fija una decisión por uso para cada fuente. El detalle de contenido, cobertura y condiciones de cada muestra está en [fuentes públicas complementarias](../../docs/data/free-data-sources.md). Aquí se recoge lo que se ha comprobado y lo que se decide con esa evidencia.

## Método

[`public_source_reconciliation.py`](../../src/mars_titan/data/public_source_reconciliation.py) reconstruye el índice de capturas sin acceder a la red y clasifica cada fuente por la evidencia de sus registros:

- **Adquirida y validada**: al menos un registro válido cuyo archivo existe y conserva la huella del manifiesto.
- **Documento fijo**: lo mismo para un PDF financiero de una publicación concreta.
- **Intento fallido**: registros sin ninguna respuesta válida.
- **Bloqueada en el descubrimiento**: el acceso se detuvo antes de conocer un enlace de descarga.

El estado del catálogo tiene que coincidir con esa clase. También se comprueba que cada fuente declara `benchmark_eligible=false` y al menos una decisión por uso con la carencia que cubriría y su evidencia. La orden termina con error si encuentra una incoherencia. El resultado completo está en [`public-source-validation.json`](public-source-validation.json), con la huella del catálogo (`298053b78f09d854…`) y de los cuatro manifiestos reconciliados.

```bash
uv run python -m mars_titan.data.public_source_reconciliation --output reports/data/public-source-validation.json
```

La ejecución sobre el almacén local del proyecto no encontró incoherencias en las catorce fuentes.

## Reconciliación del catálogo

La columna de intentos indica intentos registrados, registros válidos y contenidos distintos. Los diagnósticos son peticiones que el manifiesto inicial anota sin conservar cuerpo.

| Fuente | Estado del catálogo | Evidencia | Intentos | HTTP de los fallos |
| --- | --- | --- | --- | --- |
| `fred_md` | `downloaded_validated` | Adquirida y validada | 2 / 2 / 1 | |
| `fred_qd` | `downloaded_validated` | Adquirida y validada | 2 / 2 / 1 | |
| `french_factors_daily` | `downloaded_validated` | Adquirida y validada | 2 / 2 / 1 | |
| `ecb_usd_eur` | `downloaded_validated` | Adquirida y validada | 2 / 2 / 2 | |
| `nyfed_gscpi` | `failed` | Intento fallido | 1 / 0 / 0 | 200 con un ZIP sin libro (diagnóstico 200) |
| `cboe_vix` | `downloaded_validated` | Adquirida y validada | 2 / 2 / 2 | |
| `treasury_yields` | `downloaded_validated` | Adquirida y validada | 2 / 2 / 2 | |
| `sec_aapl_submissions` | `failed` | Intento fallido | 1 / 0 / 0 | 403 |
| `sec_aapl_companyfacts` | `failed` | Intento fallido | 1 / 0 / 0 | 403 |
| `gdelt_article_metadata` | `failed` | Intento fallido | 1 / 0 / 0 | 200 con esquema no válido (diagnóstico 429) |
| `nyfed_gscpi_vintages` | `downloaded_validated` | Adquirida y validada | 3 / 2 / 1 | 200 rechazado por `#N/A` en la primera validación |
| `fed_monetary_rss` | `downloaded_validated` | Adquirida y validada | 4 / 4 / 1 | |
| `apple_fy2026_q3_financials` | `downloaded_validated` | Documento fijo | 1 / 1 / 1 | |
| `stooq_aapl_eod` | `blocked_at_source_discovery` | Bloqueada en el descubrimiento | 1 / 0 / 0 | 200 con verificación de navegador |

Los fallos de SEC, GDELT y Stooq siguen registrados como tales. No se han repetido: una comprobación posterior solo se hará con un acceso permitido, sin cambiar de identidad, de red ni resolver retos de navegador. Un HTTP 200 tampoco se ha tomado como prueba de formato correcto. Tres de los fallos devolvieron 200 con un contenido que no superó su validador.

La comprobación real del 10 de octubre, descrita en la [validación de la ingesta](public-ingestion-validation.md), volvió a obtener las ocho fuentes renovables con contenido válido. FRED ya enlazaba las ediciones de septiembre (`2026-09-md.csv` y `2026-09-qd.csv`), VIX y BCE llegaban al 9 de octubre y la matriz GSCPI sumaba la versión de septiembre. Esas capturas se conservan fuera del almacén del proyecto y no cambian la tabla anterior.

## Decisiones por uso

`admitted_exploratory` significa que la copia puede usarse en análisis locales que no alimentan ningún modelo. `pending` exige una comprobación concreta antes de decidir. `excluded` descarta ese uso con la evidencia actual. Ninguna decisión vuelve elegible una fuente para el benchmark.

| Fuente | Uso propuesto | Decisión | Motivo principal |
| --- | --- | --- | --- |
| FRED-MD | Entrada macro mensual de un experimento | Excluida | `current.csv` revisa periodos anteriores. El catálogo macro asigna a cada serie ALFRED o un archivo de publicación |
| FRED-MD | Contraste local de cobertura, unidades y ausencias | Admitida para exploración local | Capturas íntegras con huella |
| FRED-QD | Entrada macro trimestral | Excluida | El fin del trimestre no es su publicación |
| FRED-QD | Contraste local de magnitudes trimestrales | Admitida para exploración local | Capturas íntegras con huella |
| French diario | Revisar la convención del factor de mercado del objetivo residual | Pendiente | Cierre a cierre frente al intervalo de apertura a cierre del protocolo, con revisiones y cambio FIZ/CIZ |
| French diario | Sustituir el factor de mercado del objetivo | Excluida | Crearía otra identidad de supervisión y el archivo no es point-in-time |
| BCE USD/EUR | Contexto de divisas | Pendiente | El catálogo macro prevé DEXUSEU con versiones. La consulta solo cubre 90 días del presente |
| GSCPI XLSX | Libro del índice | Excluida | ZIP sin libro. Lo sustituye el CSV de versiones |
| Cboe VIX | Volatilidad implícita como contexto | Pendiente | Fuera de los 140 indicadores, sin versiones y con derechos reservados |
| Treasury | Curva de tipos como entrada | Excluida | El catálogo macro usa ALFRED o archivo de publicación para DGS3MO hasta DGS30 |
| Treasury | Contraste local con H.15 | Admitida para exploración local | Capturas íntegras con huella |
| SEC submissions | Presentaciones con hora de aceptación | Pendiente | 403 registrado |
| SEC companyfacts | Hechos XBRL con fecha de presentación | Pendiente | 403 registrado |
| GDELT | Metadatos de noticias | Pendiente | Esquema no válido y 429. `seendate` no acredita la publicación original |
| GSCPI por versiones | Fuente del GSCPI en la edición histórica | Excluida | La edición descarga el mismo CSV con `macro_gscpi` y su propio recibo |
| GSCPI por versiones | Estudio local de revisiones entre versiones | Admitida para exploración local | 348 filas y 57 columnas de versión. Las etiquetas no son horas de publicación |
| RSS de la Fed | Comunicados monetarios como modalidad textual | Pendiente | Quince entradas recientes con `pubDate` sin auditar |
| Apple Q3 FY2026 | Ejemplo de extracción de tablas y texto | Admitida para exploración local | PDF aceptado por `pdfinfo`, comunicado del 30 de julio de 2026 sin hora verificada |
| Apple Q3 FY2026 | Fundamentales de entrada | Excluida | Un emisor y un documento sin historial de versiones |
| Stooq | Precios EOD como contraste externo | Excluida | Verificación de navegador sin resolver y condiciones sin revisar. Una comprobación permitida podría reabrirla |

El texto completo de cada decisión, con su carencia, está en el campo `use_decisions` del catálogo.

## Carencias que cubren y que no cubren

Las fuentes no macro se han revisado sin esperar al catálogo macro completo. El PDF de Apple y las API de SEC apuntan a la misma carencia: la fecha de cierre de un fundamental no equivale a su publicación. El PDF sirve como ejemplo documental, pero las API que darían horas de aceptación siguen sin respuesta permitida. El RSS de la Fed y GDELT tampoco amplían la historia de noticias: las de la copia de FinMultiTime empiezan en abril de 2009 en US y en marzo de 2020 en CN ([cobertura histórica](../../docs/data/historical-coverage.md)), y estas fuentes solo cubren el periodo reciente.

Las fuentes macro remiten a la edición histórica (MT-056) cuando el catálogo macro ya les asigna una vía con versiones, como ocurre con el GSCPI, DEXUSEU y los rendimientos Treasury. Las copias corrientes solo quedan para contrastes locales. La actualización periódica de estas capturas corresponde a MT-067 (#69).

## Condiciones de uso

Las condiciones se volvieron a leer en las páginas de cada productor el 10 de octubre de 2026. El campo `terms_review` del catálogo guarda la fecha, las páginas consultadas, el estado de la revisión, lo que permiten para el uso local y para la redistribución, y un resumen de sus condiciones. La reconciliación exige esa revisión en las catorce fuentes, rechaza un uso admitido para exploración local si las condiciones no lo permiten y comprueba que `license_unknown` solo sea falso cuando la redistribución con atribución está verificada.

| Fuente | Revisión | Uso local | Redistribución | Condición principal |
| --- | --- | --- | --- | --- |
| FRED-MD y FRED-QD | Verificada en los términos de FRED y del St. Louis Fed | Personal, educativo y no comercial | No concedida | Las series con copyright de terceros solo admiten ese uso sin permiso del propietario. El panel no tiene una licencia única |
| French diario | Sin declaración | No declarado | No concedida | Solo figura el copyright de Fama y French. Por eso su uso sigue pendiente |
| BCE USD/EUR | Verificada | Permitido | Con atribución | Reproducción exacta, cita del BCE y declaración de modificaciones |
| GSCPI (XLSX y CSV) | Verificada | Permitido | Con atribución | Licencia no exclusiva del New York Fed con aviso, atribución, cambios marcados y mismas condiciones |
| Cboe VIX | Verificada | No declarado | No concedida | Materiales protegidos sin licencia sobre los datos |
| Treasury | Sin declaración del Departamento | Permitido por 17 U.S.C. § 105 | Sin verificar | La ley excluye del copyright las obras federales, pero el feed no lo confirma y parte de cotizaciones de mercado |
| SEC (dos API) | Página no accesible (HTTP 403) | Sin verificar | Sin verificar | No se conserva ningún archivo |
| GDELT | Verificada | Permitido | Con atribución | Uso libre citando el proyecto. Los derechos de los medios sobre titulares siguen aparte |
| RSS de la Fed | Verificada | Permitido | Con atribución | Dominio público salvo indicación contraria, citando a la Junta |
| Apple Q3 FY2026 | Verificada | Personal y no comercial | No concedida | Sin modificar, sin retirar avisos y sin publicar copias |
| Stooq | Bloqueada antes de las condiciones | Sin verificar | Sin verificar | No se conservan precios |

Nueve fuentes mantienen `license_unknown=true` porque no tienen un permiso de redistribución verificado. Las que quedan admitidas para exploración local (FRED-MD, FRED-QD, Treasury, la matriz GSCPI y el PDF de Apple) tienen condiciones que permiten ese uso. Ninguna copia se redistribuye: los archivos siguen fuera de Git, el acceso gratuito no equivale a una licencia y la licencia MIT del proyecto no cubre estos datos.

## Lo que no cambia

Ninguna decisión altera el benchmark congelado, sus entradas ni sus predicciones. Una fuente admitida para exploración local no puede alimentar un experimento sin superar antes el contrato temporal y una decisión experimental registrada antes de evaluar. Descargar hoy un valor revisado no lo convierte en una versión histórica.
