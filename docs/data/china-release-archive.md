# Indicadores chinos a partir de comunicados fechados

El lector recupera seis conceptos del catálogo: IPC interanual nacional, producción industrial interanual mensual, PMI manufacturero oficial, saldo de M2, flujo mensual de financiación a la economía real y su saldo. Cada entrada de [china-release-sources.json](../../data/catalogs/china-release-sources.json) identifica el documento, su SHA-256, el concepto, el periodo y el límite documental de disponibilidad.

La edición inicial contiene 88 documentos de NBS y del Banco Popular de China. Su intervalo de comprobación mensual va de octubre de 2022 a noviembre de 2023. Incluye algunas publicaciones anteriores para inicializar el contexto. Los archivos HTML permanecen en almacenamiento local. El manifiesto público contiene sus referencias y huellas, sin redistribuir esos documentos bajo la licencia del proyecto.

El archivo se valida antes de calcular el panel. Deben existir publicaciones para cada concepto y mes del intervalo declarado. En producción industrial, el [comunicado de enero y febrero de 2023](https://www.stats.gov.cn/english/PressRelease/202303/t20230317_1937565.html) contiene un dato conjunto. Se registra como exclusión de la serie mensual y no se convierte en dos observaciones. Las decisiones posteriores conservan el último dato mensual conocido y su periodo de referencia. Esto no equivale a disponer de una nueva observación cada día.

## Fecha y unidades

NBS publica versiones inglesas que pueden ser posteriores al comunicado chino. El lector conserva el día declarado y el día de la ruta archivada. Admite la cifra después del más tardío, al terminar ese día en `Asia/Shanghai`, y aplica la siguiente sesión estricta del mercado de destino. La fecha de la ruta es un límite conservador del archivo consultado, no una prueba de la hora original de publicación. `publication_timestamp_verified` permanece en `false`.

Los documentos de PBoC deben tener fechas coincidentes en sus metadatos y cabecera. También se admite la reproducción oficial de la oficina financiera de Guangdong cuando identifica a PBoC como fuente. En ese caso se usa el día de la reproducción, sin adelantarlo al del original. El manifiesto de esta edición usa directamente los comunicados de PBoC.

El [informe financiero de noviembre de 2023](https://www.pbc.gov.cn/diaochatongjisi/116219/116225/b21be1ab65c94424a612a7666d605a1b/index.html) publica M2 como saldo. Los informes de financiación distinguen flujo mensual, acumulado y saldo. El lector exige la cifra del periodo indicado, sin calcular un saldo sumando flujos ni deducir un flujo mensual restando acumulados de distintas versiones.

Los importes se expresan en miles de millones de yuanes. Un `万亿元` equivale a 1.000 unidades de salida y un `亿元` a 0,1. La conversión utiliza aritmética decimal antes de almacenar `float64`. Se guardan la cifra y unidad originales. El lector rechaza saldos no positivos, valores no finitos, desbordamientos, fechas incompatibles y resultados ambiguos.

PBoC recoge cambios de perímetro en los comunicados, incluida la incorporación de determinadas entidades financieras desde enero de 2023. Se conserva cada documento completo y su procedencia. El panel representa las cifras publicadas en cada momento, no una historia homogeneizada retrospectivamente. Los informes no acreditan un ajuste estacional para esos saldos y flujos, por lo que su metadato indica `published_without_adjustment_statement`.

## Preparación y comprobación

`uv run python -m mars_titan.data.macro_releases --help` describe la ejecución. Requiere manifiesto, catálogo, caché de fuentes, directorio de salida nuevo, mercado y fechas. Cada HTML de la caché se llama con su SHA-256 seguido de `.html`. `--download` recupera únicamente los archivos ausentes desde los dominios admitidos, con límites de tamaño y tiempo. No sigue redirecciones. Una respuesta distinta de la huella registrada exige revisar la fuente y crear otro recibo, no sobrescribir la evidencia anterior.

La salida contiene `macro.parquet`, `observations.parquet`, `source-manifest.json`, `report.json` y los originales en `sources/`. La publicación del directorio es atómica y no reemplaza una edición existente. Los errores de descarga pueden reintentarse conservando los archivos ya comprobados en la caché. La publicación atómica utiliza `renameat2` de Linux, como la puerta de cobertura macro existente.

El periodo final solicitado no puede superar el mes siguiente al último mes de referencia del archivo. Este límite impide prolongar indefinidamente cifras antiguas cuando no se han recuperado nuevas publicaciones. La cobertura del archivo y la presencia de los 140 indicadores por decisión son comprobaciones distintas. El panel debe pasar después por la [puerta de admisión](macro-admission.md).

La primera preparación local ha producido 22.644 filas para 3.774 sesiones de 2009 a 2023, con 255 observaciones de los comunicados y 1.808 valores disponibles. Las fechas anteriores al archivo quedan ausentes. No se ha entrenado ningún modelo con este panel por completar su adquisición. Las pruebas y medidas ejecutadas se registran en [china-releases-quality.json](../../reports/resources/china-releases-quality.json).
