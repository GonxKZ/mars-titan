# Cobertura histórica de 2000 a 2023

La copia inspeccionada no permite formar una historia desde 2000 con cuatro modalidades y los 140 indicadores obligatorios. US dispone de precios desde el 3 de enero de 2000, pero sus noticias comienzan en abril de 2009. CN dispone de precios desde el 4 de enero de 2006 y noticias desde el 31 de marzo de 2020. El GSCPI y el contrato de versiones de estrés financiero limitan además la cesta actual al periodo reciente. Esta conclusión se refiere a los archivos conservados, no a todas las fuentes que puedan existir.

El índice textual inspeccionado ocupa 2.201.612.288 bytes. Sus 5.586 recibos completos concilian 4.469.917 registros con el [recibo del índice original](../../reports/data/corpus-index-20260922.json). Se consultaron fechas, estados y localizadores, sin leer los cuerpos. `PRAGMA quick_check` devolvió `ok`.

El censo conserva 4.784 identidades US y 892 CN. Tienen archivos de las cuatro fuentes 2.639 US y 810 CN. El [resumen verificable](../../reports/data/historical-coverage-20261007.json) conserva los motivos de falta de modalidades y las huellas de la auditoría. Las firmas de tamaño, mtime y ctime de los 216.453 originales coinciden con el inventario. Esto no constituye una nueva verificación de los 116 GB por contenido.

## Fechas comprobadas y unidades de recuento

| Componente | Estados Unidos | China |
| --- | --- | --- |
| Precios válidos anteriores a 2024, todos los archivos | 16.090.522 filas, 2000-01-03 a 2023-12-29 | 2.891.924 filas, 2006-01-04 a 2023-12-29 |
| Noticias originales con fecha declarada anterior a 2024 | 2.436.231 registros, desde 2009-04-08 hasta 2023-12-16 | 1.053.678 registros, desde 2020-03-31 hasta 2023-12-31 |
| Noticias preparadas de activos con cuatro fuentes | 1.687.392 filas, disponibilidad 2009-04-13 20:05 UTC a 2023-12-18 21:05 UTC | 553.392 filas, disponibilidad 2020-04-01 07:05 UTC a 2023-12-29 07:05 UTC |
| Hechos contables preparados originales | 2.982.064 filas, disponibilidad 2009-04-16 20:05 UTC a 2023-12-29 21:05 UTC | Cero hechos admitidos en la preparación original |
| Hechos CN revisados en la unión actual | No corresponde | 287 hechos en 49 activos, disponibilidad 2022-03-11 07:05 UTC a 2023-04-03 07:05 UTC |
| Sesiones completas con la cesta macro actual | 397, desde 2022-06-01 20:05 UTC hasta 2023-12-28 21:05 UTC | 387, desde 2022-06-02 07:05 UTC hasta 2023-12-29 07:05 UTC |

Los hechos revisados CN comprenden cierres de 2020 a 2022. Su disponibilidad sigue siendo la de las publicaciones de 2022 y 2023. Los 193.946 registros contables originales CN cuyo cierre está entre 2000 y 2023 carecen todos de publicación declarada y de campos de unidad no vacíos. El único registro CN con cierre de 2000 pertenece a resultados. Los balances comienzan en 2001. Ninguna de esas fechas acredita por sí sola CNY, CAS o publicación contemporánea.

Los registros US preparados se cuentan por `available_at`. El [informe contable anterior](../../reports/data/company-corpus-audit.json) cuenta 2.982.123 hechos por `filed`. Son filtros distintos y no deben intercambiarse. No se han vuelto a recorrer las 158.623.107 ocurrencias US ni los aproximadamente 89 GB contables que respaldan aquella auditoría.

El último texto US preparado queda disponible el 18 de diciembre de 2023. Con la ventana vigente de cinco sesiones, el límite de presencia textual llega como máximo al 22 de diciembre. Es una cota del componente textual, no una nueva admisión conjunta por activo ni una etiqueta madura.

## Recuento anual

Precios y ventanas pertenecen a instrumentos con cuatro fuentes presentes. Noticias y hechos son filas preparadas con disponibilidad en ese año. Una ventana de 64 precios permite estudiar un gráfico causal regenerado, pero no cuenta como una muestra. El resumen conserva los recuentos de los 24 años por mercado, incluidos los ceros, y las fechas extremas por componente. Los localizadores y mapas por archivo permanecen en los recibos privados enlazados por hash.

### US

| Año | Filas de precios | Ventanas de 64 | Noticias preparadas | Hechos disponibles | Indicadores originales con algún valor | Sesiones actuales con 140 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 280768 | 167989 | 0 | 0 | 63 | 0 |
| 2001 | 287791 | 232944 | 0 | 0 | 68 | 0 |
| 2002 | 301374 | 246909 | 0 | 0 | 68 | 0 |
| 2003 | 311206 | 259262 | 0 | 0 | 68 | 0 |
| 2004 | 324573 | 272714 | 0 | 0 | 69 | 0 |
| 2005 | 339274 | 291060 | 0 | 0 | 87 | 0 |
| 2006 | 352927 | 309353 | 0 | 0 | 87 | 0 |
| 2007 | 373843 | 329527 | 0 | 0 | 87 | 0 |
| 2008 | 391903 | 344607 | 0 | 0 | 88 | 0 |
| 2009 | 398488 | 359111 | 314 | 12872 | 93 | 0 |
| 2010 | 411819 | 371658 | 7178 | 49620 | 97 | 0 |
| 2011 | 427156 | 389535 | 12711 | 122223 | 103 | 0 |
| 2012 | 439423 | 401465 | 27846 | 181036 | 103 | 0 |
| 2013 | 461718 | 425456 | 49039 | 189029 | 103 | 0 |
| 2014 | 488703 | 457758 | 66788 | 192186 | 112 | 0 |
| 2015 | 514441 | 484463 | 79011 | 199882 | 113 | 0 |
| 2016 | 533664 | 508523 | 97972 | 205451 | 116 | 0 |
| 2017 | 553688 | 527455 | 127201 | 217487 | 116 | 0 |
| 2018 | 588156 | 560554 | 131978 | 240111 | 116 | 0 |
| 2019 | 625519 | 598616 | 95438 | 267804 | 124 | 0 |
| 2020 | 654981 | 633102 | 118581 | 273317 | 124 | 0 |
| 2021 | 657984 | 647236 | 141742 | 274782 | 125 | 0 |
| 2022 | 657412 | 650721 | 219382 | 275981 | 125 | 148 |
| 2023 | 656309 | 651360 | 512211 | 280283 | 125 | 249 |

### CN

| Año | Filas de precios | Ventanas de 64 | Noticias preparadas | Hechos disponibles | Indicadores originales con algún valor | Sesiones actuales con 140 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 0 | 0 | 0 | 0 | 63 | 0 |
| 2001 | 0 | 0 | 0 | 0 | 68 | 0 |
| 2002 | 0 | 0 | 0 | 0 | 68 | 0 |
| 2003 | 0 | 0 | 0 | 0 | 68 | 0 |
| 2004 | 0 | 0 | 0 | 0 | 69 | 0 |
| 2005 | 0 | 0 | 0 | 0 | 87 | 0 |
| 2006 | 107469 | 72879 | 0 | 0 | 87 | 0 |
| 2007 | 117215 | 105958 | 0 | 0 | 87 | 0 |
| 2008 | 126710 | 114724 | 0 | 0 | 88 | 0 |
| 2009 | 128038 | 119099 | 0 | 0 | 93 | 0 |
| 2010 | 138259 | 125875 | 0 | 0 | 97 | 0 |
| 2011 | 150434 | 139659 | 0 | 0 | 103 | 0 |
| 2012 | 157065 | 147064 | 0 | 0 | 103 | 0 |
| 2013 | 154806 | 148210 | 0 | 0 | 103 | 0 |
| 2014 | 161493 | 153534 | 0 | 0 | 112 | 0 |
| 2015 | 165605 | 153555 | 0 | 0 | 113 | 0 |
| 2016 | 168872 | 162149 | 0 | 0 | 116 | 0 |
| 2017 | 175608 | 170849 | 0 | 0 | 116 | 0 |
| 2018 | 179441 | 174174 | 0 | 0 | 116 | 0 |
| 2019 | 184216 | 117768 | 0 | 0 | 124 | 0 |
| 2020 | 190779 | 182077 | 87232 | 0 | 124 | 0 |
| 2021 | 194309 | 191131 | 150587 | 0 | 125 | 0 |
| 2022 | 195596 | 193881 | 152370 | 275 | 125 | 145 |
| 2023 | 196009 | 195530 | 163203 | 12 | 125 | 242 |

En CN la columna de hechos disponibles usa exclusivamente los 49 activos revisados, mientras precios y noticias cubren los 810 preparados. Las columnas no describen una población ya intersectada. Los 140 indicadores con algún valor en un año tampoco garantizan 140 en todas sus sesiones.

## Qué limita la historia macro

El panel original contiene 63 indicadores observados en alguna sesión de 2000 y llega a 125 en 2021. Los 15 restantes se recuperaron después con fuentes separadas. La edición US recuperada de 2009 a 2023 alcanza 93 indicadores en 2009, 109 en 2011 y 131 en 2021. Solo completa los 140 desde noviembre de 2022 bajo STLFSI4. La edición posterior declara de forma separada la composición STLFSI3→STLFSI4, los retardos diarios por observaciones válidas y el historial de anuncios chinos.

El resumen separa cinco ediciones macro. Conserva sus sesiones completas por año, los identificadores totalmente ausentes o parcialmente ausentes y las fechas observadas de los conceptos que limitan la ampliación. Los años con las mismas listas de ausencias se agrupan sin mezclar ediciones. Las columnas del panel original y del actual no son intercambiables. El actual contiene únicamente 2022 y 2023, sin un recálculo nuevo de los años anteriores.

| Serie o familia | Evidencia local | Consecuencia |
| --- | --- | --- |
| GSCPI y cambio mensual | Presentación pública en 2022. Regla admitida desde el cierre del mes de versión de mayo de 2022. Primeras decisiones US 2022-06-01 y CN 2022-06-02 | Una estimación retrospectiva de 1997 no sirve como publicación de 1997 o 2000 |
| STLFSI3 y STLFSI4 | Versiones admitidas desde 2022-01-13 y 2022-11-10 | La composición actual no acredita una serie publicada anterior a enero de 2022 |
| NFCI | Primera versión conservada admitida 2011-05-25 | La observación de 2007 en ese archivo solo se conoce desde esa versión |
| Seis anuncios NBS/PBOC | Archivos actuales de referencia abril de 2022 a noviembre de 2023. Primera cota de publicación entre 2022-05-06 y 2022-05-19 | Requieren otros anuncios originales para ampliar hacia atrás. El histórico de PMI dentro de un anuncio no se retrofecha |
| Tipos y divisas con historia antigua pero vintages locales tardíos | DGS10 conserva periodos desde 1988 pero su primer vintage local es 2005-06-28. DEXCHUS empieza en versiones de 2014-03-18. El tipo de depósito euro empieza en versiones de 2021-03-11 | Son huecos de procedencia temporal de esta adquisición. No prueban que el dato no existiera en aquellos años |
| Series con inicio de observación más reciente | Los periodos locales de DFEDTARL/U empiezan en 2008-12-16, SOFR en 2018-04-03 y DFII5/10 en 2003-01-02 | No se deben prolongar a 2000 ni sustituir por otra definición sin un contrato distinto |

La consulta de todos los paneles inspeccionados devuelve cero filas anteriores a 2024 con `available_at > prediction_at`. Es una comprobación de sus metadatos, no una nueva validación de las fórmulas o de todas las publicaciones originales.

En 2023 la edición US tiene una sesión macro incompleta por `brent_wti_spread`. CN tiene completas sus 242 sesiones de ese año. Resolver únicamente el GSCPI preliminar podría añadir siete sesiones US del 20 al 31 de mayo de 2022, según la [auditoría previa de GSCPI](../../reports/data/gscpi-release-audit-20261006.json). La revisión FRASER no acreditó captura contemporánea y esas siete sesiones continúan sin admitir.

## Edición histórica adicional con ausencias explícitas

Se define una edición adicional con todos los datos utilizables desde 2000 y
máscaras de ausencia. Su preparación y verificación están pendientes. La
comparación estricta que exige las cuatro modalidades y los 140 indicadores
observados se conserva como control separado, con sus propias filas y resultados.
Los recuentos de esta auditoría no cambian ni acreditan que la nueva edición
esté implementada.

La presencia de un archivo, un valor observado igual a cero y una ausencia son
estados distintos. Un cero observado conserva su máscara de presencia. Una
entrada ausente o temporalmente inadmisible conserva su máscara de ausencia y
su causa. Si se desconoce la publicación, la entrada permanece ausente y no se
usa el cierre contable como fecha de publicación.
No hace falta disponer de noticias de 2000 para registrar que faltan en esta
copia, ni se inventan textos o cifras para completar esa modalidad.

Los precios y objetivos necesarios para una fila deben ser reales y válidos.
Una máscara no permite crear precios, rendimientos ni etiquetas maduras. Un
indicador macro solo se calcula cuando su fórmula, sus entradas y su
disponibilidad lo permiten. En caso contrario queda ausente, sin sustituir su
definición ni adelantar observaciones futuras.

Todos los modelos y etapas de la misma comparación reciben los mismos IDs,
particiones temporales, máscaras y objetivos. Los resultados se desglosan por
patrón de cobertura, con denominadores explícitos. La edición histórica y el
control estricto mantienen identidades y análisis separados. La pausa de
aprendizaje continúa hasta preparar y verificar la edición histórica adicional.
Este contrato no inicia ni reanuda modelos.

## Tramos de ampliación que respetan las fuentes

1. Conservar el panel desde 2000 con las ausencias reales y recalcular sobre cada calendario solo cuando exista una nueva fuente fechada. La recuperación de archivos o retardos no acredita una publicación contemporánea del GSCPI anterior a 2022. Mantener los mismos 140 como requisito completo impide una historia desde 2000 con los contratos actuales.

2. Para US anterior a abril de 2009 hacen falta noticias históricas y publicaciones contables con fecha acreditada. La copia actual de FinMultiTime no las aporta. Los archivos históricos de noticias y de presentaciones de emisores son candidatos a una nueva adquisición con procedencia explícita, no fuentes ya verificadas por esta auditoría.

3. Para CN pueden revisarse informes anteriores a los ya transcritos, con sus PDF y anuncios originales. Hay metadatos de cierres desde 2000/2001, pero no publicación ni unidad en esos JSON. La ampliación debe recuperar esos datos por documento. Antes de marzo de 2020 faltan también noticias en esta copia y antes de enero de 2006 faltan precios. No se puede extender un activo a fechas anteriores a su propia historia.

4. Para macros publicados antes de sus primeros vintages locales, contrastar archivos oficiales de la misma serie y unidad, por ejemplo H.15 para los tipos, publicaciones del banco central para sus tipos oficiales y anuncios NBS/PBOC. Son vías de adquisición pendientes, no cobertura concedida. La fórmula conserva el máximo de disponibilidad de sus componentes y su calentamiento.

5. Los PNG originales son semestrales y su nombre no acredita cuándo podía utilizarse cada píxel. Para decisiones diarias anteriores solo procede regenerar gráficos desde las 64 sesiones históricas válidas. Hay 10.121.378 ventanas US y 2.668.116 CN entre instrumentos con cuatro fuentes, antes de cruzarlas con noticias, fundamentales o macro. Estos totales concilian exactamente con el informe previo de ventanas.

## Fuentes oficiales candidatas, sin admisión

Una sonda posterior conservó cinco documentos oficiales, 476.088 bytes, sin
redirecciones ni reintentos. Las ediciones H.15 del
[10 de enero](https://www.federalreserve.gov/releases/h15/20000110/h15.htm) y del
[18 de enero de 2000](https://www.federalreserve.gov/releases/h15/20000118/h15.htm)
aportan 60 cifras diarias de seis tipos. Coinciden con los valores locales
numéricos, pero sus HTML declaran actualización web el 11 de octubre de 2001.
La revisión posterior de los PDF de esas ediciones se describe al final de este apartado.
La coincidencia numérica no prueba que los bytes capturados ahora fueran los
publicados en 2000, ni acredita una hora de publicación contemporánea.

Tres documentos del BCE identifican la facilidad de depósito y distinguen
anuncio de entrada en vigor: [21 de enero de 1999](https://www.ecb.europa.eu/press/pr/date/1999/html/pr990121.en.html),
[4 de noviembre de 1999](https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/1999/html/is991104.en.html)
y [5 de octubre de 2000](https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2000/html/is001005.pt.html).
Los tres tipos coinciden con las versiones locales no vacías. Se conserva el
intervalo local a NULL del 16 de junio de 2021 y su restauración al día siguiente.
Esa retirada no se elimina ni se cuenta como discrepancia numérica.

Son 63 cifras candidatas, todavía sin incorporar al catálogo ni a los paneles.
La secuencia BCE no es completa y no permite extender tipos entre fechas
alejadas. H.15 no aporta en este lote los días de fin de semana necesarios para
DFF. Ninguna sonda sustituye SOFR o GSCPI ni resuelve la historia multimodal.
La revisión PDF se conserva en un recibo separado del resumen inicial.

El contraste posterior de los PDF del [10 de enero](https://www.federalreserve.gov/releases/h15/20000110/h15.pdf)
y del [18 de enero](https://www.federalreserve.gov/releases/h15/20000118/h15.pdf)
revisó sus cuatro páginas. Las 60 cifras diarias y 36 celdas agregadas coinciden
con los HTML. Los documentos imprimen las fechas de publicación, pero no una
hora. Sus metadatos internos carecen de zona horaria y declaran modificaciones
anteriores a la creación, por lo que no se utilizan para fijar la disponibilidad.
El recibo de esta revisión tiene SHA-256
`dce22ab2e5e55590d846645c6c69748bf63dc4c8e5e72b329b1a87714c226e5d`.
Esta evidencia permite preparar una edición documental con disponibilidad
conservadora por fecha. Todavía requiere su contrato de admisión y no acredita
que los bytes descargados sean una captura contemporánea de 2000.

## Reproducción y límites

El resumen enlaza por SHA-256 los tres recibos de cálculo, sus scripts y los fallos conservados. Las tres pasadas se ejecutaron con `uv run --no-sync --offline`, CUDA desactivada, dos hilos, una cuota de dos CPU y un límite de 2 GiB de memoria. No se modificó el entorno ni se realizaron descargas.

| Pasada confirmada | Tiempo interno | RSS máximo |
| --- | ---: | ---: |
| inventory_prices_prepared | 60,798 s | 265,33 MiB |
| macro | 1,838 s | 208,17 MiB |
| cn_original_periods | 17,905 s | 125,29 MiB |

La suma de tiempos internos confirmados es 80,541 s. No es el tiempo total de la auditoría interactiva. El núcleo verificó 15.477 archivos derivados o de índice y 5.840.656.338 bytes. La pasada CN contrastó además 2.430 fuentes y 595.799.355 bytes. Los mapas detallados de hashes están en los tres recibos, enlazados desde el JSON final.

Se conservan dos fallos del arnés privado. El primero encontró una ruta SQLite almacenada como bytes y se corrigió su lectura. El segundo encontró que DuckDB intentaba importar pytz al convertir timestamps y se resolvió usando Arrow, sin instalar paquetes. También se conserva el informe macro anterior a fijar explícitamente UTC. Ningún fallo modificó las fuentes.

Los modelos continúan parados. La auditoría y las sondas no resuelven la condición de cobertura histórica ni autorizan reanudar aprendizaje. No se generaron muestras, objetivos, factores nuevos ni resultados de modelos. No se eligieron activos por puntuaciones. La auditoría no cambia la cesta macro, las representaciones, los calendarios ni la campaña detenida.
