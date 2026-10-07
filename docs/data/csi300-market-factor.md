# Factor retrospectivo CSI 300

El factor chino utiliza la apertura y el cierre diarios publicados en la tabla
CSI 300 del [boletín mensual de SSE](https://www.sse.com.cn/aboutus/publication/monthly/index/).
La adquisición de 2022 y 2023 conserva 24 respuestas, sus fechas de descarga y
huellas. Las 484 sesiones coinciden con XSHG, 242 por año, sin duplicados ni
ausencias. La apertura no se reconstruye con el cierre anterior.

`csi300_factor` lee una adquisición terminada, contrasta la consulta
`COMMON_SSE_ZQZS_M_CSI300_INDEX_C` y decodifica su envoltorio JSONP sin ejecutar
código. Rechaza claves duplicadas, meses posteriores a 2023, precios no positivos
o incoherentes y archivos que superan los presupuestos. Los meses declarados se
contrastan con un calendario completo. Las sesiones ausentes se registran y no
se rellenan, incluso cuando falta un mes entre dos meses adquiridos.

La salida incluye `prices.parquet`, `market-factors.json` y `report.json`. El
descriptor puede pasarse a la codificación del corpus mediante
`--market-factors`. Identifica CN, el índice `000300`, el archivo y su SHA256.
Los puntos del índice se conservan en float64 con los dos decimales publicados,
sin añadir ajustes. Los originales fijados conservan su representación textual.
El Parquet real ocupa 17.477 bytes.

La disponibilidad al cierre más cinco minutos es una convención de
reconstrucción retrospectiva. La adquisición actual no acredita las versiones
diarias originales ni la fecha de publicación de cada mensual. Tampoco se ha
verificado una política de correcciones históricas. El informe declara
`point_in_time_verified=false`. El índice sirve para etiquetas predictivas, no
para simular una operación ejecutable. `financial_simulation_ready=false`
conserva esa distinción. La procedencia depende de las respuestas y recibos
previamente contrastados, que el lector no vuelve a autenticar contra la web.

El retorno factorial de una sesión es `close / open - 1`. El residualizador
existente estima alfa y beta con un máximo de 252 sesiones y un mínimo de 126
parejas cuyos cierres ya hayan madurado. El retorno del índice de la siguiente
sesión forma parte de la etiqueta y no entra en las características ni en ese
ajuste. Iniciar el factor en enero de 2022 no aporta calentamiento suficiente
para junio de 2022.

La [comprobación real](../../reports/data/csi300-factor-20261006.json) contrasta
las implementaciones de referencia y NumPy sobre precios preparados de Ping An.
Las 484 filas coinciden exactamente en coeficientes, etiquetas y exclusiones.
De sus 169 fechas candidatas con noticias, precios, fundamentales y macro,
168 tienen etiqueta admisible. La última se excluye porque cruza el corte anual.
La [codificación posterior](chinese-multimodal-samples.md) ya permite recorrer esas
muestras, pero todavía no constituye un corpus listo para la comparación china.

La creación, reutilización y contraste completo tardaron 0,55 segundos en un
proceso, con 146.596 KiB de RAM máxima. Es una ejecución funcional por método,
sin estimación de aceleración. Pasan 77 pruebas relacionadas y cinco mutaciones
dirigidas. La cobertura del módulo es del 90,86 % de sentencias y del 75,71 % de
ramas. El recibo declara herramientas, complejidad y convención de CRAP.

La CLI se ejecuta con `PYTHONPATH=src uv run --no-sync python -m
mars_titan.data.csi300_factor`. Requiere `--acquisition` y `--output`. Las opciones
`--calendar-start` y `--calendar-end` permiten fijar el calendario. El destino
debe estar separado de la adquisición y del dataset. Una interrupción anterior
a la publicación no deja una edición parcial visible. La reutilización vuelve
a comprobar fuentes, contenido y descriptor y rechaza identidades distintas.

## Ampliación con los boletines de 2021

`csi300_history.extend_csi300_history` une una edición base con tablas revisadas
del [archivo oficial de SSE](https://www.sse.com.cn/aboutus/publication/monthly/documents/).
Recibe el directorio base, un manifiesto de revisión, un destino nuevo y el
calendario CN. No descarga ni interpreta documentos durante la importación.
La revisión previa identifica el índice, el mes, la página y las columnas
apertura, máximo, mínimo y cierre. Sus PDF y CSV quedan fijados por SHA256.

El manifiesto `reviewed_csi300_history` declara mercado `CN`, símbolo `000300`
y una lista `sources`. Cada entrada conserva `month` en formato `YYYYMM`,
`source_url`, `pdf_path`, `pdf_sha256`, `csv_path`, `csv_sha256`,
`pdf_page_1_based`, `printed_page` y `title`. Las rutas son relativas al
manifiesto. El CSV mantiene las columnas `session`, `open`, `high`, `low`,
`close`, `source_pdf_sha256`, `pdf_page_1_based` y `printed_page`.

El importador limita cada PDF a 32 MiB, cada CSV a 1 MiB y cada tabla a 31
filas. Comprueba decimales, OHLC, calendario, localizadores, procedencia y
ausencias. Un solapamiento idéntico conserva la fila base y queda contado.
Un precio distinto en la misma sesión impide la unión. La salida se publica
de una vez tras volver a comprobar las fuentes. Su reutilización contrasta
también el contenido Parquet y el descriptor, sin reescribirlos. La comparación
conserva los tipos JSON, incluidos los campos anidados. Un cero no sustituye
al booleano `false` y un decimal no sustituye al recuento entero.

La [verificación de 2021](../../reports/data/csi300-history-20261007.json)
recupera doce tablas y 243 sesiones. La extracción por columnas de Poppler y
una extracción independiente por coordenadas de MuPDF coinciden en las 972
celdas OHLC. Las páginas se revisaron visualmente. Todas son la página física
19, con número impreso 16 en febrero y marzo y 13 en los demás meses.

La edición ampliada contiene 727 sesiones de 2021 a 2023, sin huecos ni
duplicados. Las 484 filas anteriores y los archivos de la edición base siguen
idénticos. Continúa siendo una reconstrucción retrospectiva. Estos documentos
no acreditan versiones diarias originales ni convierten el índice en un
precio ejecutable.

En Ping An y Vanke, los dos cálculos residuales coinciden exactamente sobre
727 filas por empresa. La historia adicional recupera 18 y 28 etiquetas,
respectivamente. Las divisiones anuales preliminares pasan a 266 filas de
entrenamiento y 451 de validación entre ambas empresas. Persisten dos cortes
anuales y dos finales. También cambian 114 y 126 etiquetas que ya eran
admisibles, porque el ajuste OLS utiliza más observaciones anteriores a la
decisión. Se necesita otra identidad de supervisión, sin sustituir resultados
de la edición previa.

La creación tarda 0,120 segundos y la reutilización 0,109. El recorrido con
ambos contrastes residuales tarda 0,574 segundos dentro del proceso y alcanza
138.600 KiB de RAM. Son comprobaciones funcionales de una repetición, sin
calentamiento ni afirmación de aceleración. El proceso completo medido desde
fuera tarda 1,00 segundos y alcanza 139.752 KiB.

Pasan 129 pruebas relacionadas y siete mutaciones dirigidas. La cobertura del
importador es del 91,82 % de sentencias y del 81,63 % de ramas. El recibo
registra también la complejidad por función y la convención de CRAP. Estas
comprobaciones no acreditan versiones históricas que las fuentes no aportan.
