# Precios sin ajustar, acciones corporativas y bajas

Autor: Gonzalo García Lama. Trabajo de la issue #379, del 9 de octubre de 2026.

El entorno de negociación real ([`simulation/market.py`](../../src/mars_titan/simulation/market.py)) solo admite una cinta histórica con `price_basis="unadjusted"`, acciones corporativas completas y retornos de salida para los activos que dejan de cotizar. La población preparada desde 2000 conserva los precios del archivo original, que están ajustados retrospectivamente, y está formada casi solo por empresas que seguían cotizando en 2025. Este documento recoge las fuentes revisadas, el método que reconstruye los precios negociados, la concordancia medida, la edición resultante y la decisión sobre las bajas. El [recibo público](../../reports/data/unadjusted-prices-20261009.json) conserva las capturas, la identidad del método, los recuentos y las tasas, sin incluir precios de terceros.

Nada de este trabajo entrena, ajusta ni evalúa modelos. La edición es un artefacto de datos separado. No modifica `dataset/`, la población preparada ni la codificación histórica v3.

## Qué distribuye el archivo original

Cada CSV de `dataset/time_series` tiene las columnas `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits`. La fecha incluye la zona horaria del mercado y las series terminan el 28 de marzo de 2025 en Estados Unidos y el 31 de marzo de 2025 en China. La [auditoría de precios](../../reports/data/price-audit.md) ya había conservado los eventos sin reaplicarlos, y la [revisión de integridad del refuerzo](../engineering/rl-environment-integrity.md) midió que el 69 % de los cierres US y el 84 % de los chinos no caen en un múltiplo de céntimo.

El README de FinMultiTime declara licencia MIT y no documenta el proveedor de precios ni su ajuste. El formato coincide con la salida de `history()` de la biblioteca yfinance, que sirve datos de Yahoo. Lo trato como hipótesis de trabajo, no como procedencia acreditada. El método se acepta o se rechaza con los datos y con fuentes independientes, no por esa atribución.

Los datos muestran cuatro convenciones del proveedor:

- OHLC ajustados por splits y dividendos, con el mismo factor para las cuatro columnas de una sesión.
- Volumen ajustado solo por splits.
- Dividendo publicado en la fecha ex y expresado en acciones posteriores a los splits estrictamente posteriores. Por ejemplo, el dividendo de AAPL de agosto de 2012 figura como 0,094643, que es 2,65 entre 28.
- Razón de split en la fecha ex, con valores no enteros en los repartos de acciones chinos (1,3 para «10 送 3») y en algunas escisiones (1,046 para IBM en noviembre de 2021).

## Inventario de fuentes

Todas las peticiones fueron seriales, separadas al menos dos segundos, sin reintentos, sin claves privadas ni cuentas nuevas y con un User-Agent que identifica el proyecto. Un 403 o 429 suspendía el host durante la ejecución. Cada captura guarda el cuerpo sin modificar, la URL, la fecha UTC, las cabeceras, el estado HTTP, los bytes, el SHA-256 y las condiciones conocidas. El manifiesto completo está en el estado privado (`captures-manifest.jsonl`) y su resumen, sin cuerpos, en el recibo público. Las fuentes que no conceden redistribución quedan solo en el estado privado.

| Fuente | Qué aporta | Acceso comprobado | Condiciones | Uso |
| --- | --- | --- | --- | --- |
| Bolsa de Shanghái, `yunhq.sse.com.cn`, serie diaria | OHLCV negociado desde la admisión, también de códigos dados de baja | 10 historias completas de 80 a 332 KB, HTTP 200. El código 600087, excluido en 2014, conserva su serie | No se capturaron condiciones de reutilización. Uso privado sin redistribución | Contraste de cierres CN |
| Bolsa de Shanghái, lista de valores con cotización terminada | 143 bajas de acciones A del tablero principal con fechas de admisión y baja | 6 páginas de 25 registros | Igual que la anterior | Análisis de bajas |
| Bolsa de Shenzhen, informe `1793_ssgs` (终止上市) | 208 bajas, 187 de acciones A, con fechas | 11 páginas | No capturadas. Uso privado | Análisis de bajas |
| Bolsa de Shenzhen, series históricas | `getHistoryData` devuelve unas 200 sesiones recientes. `1815_stock_snapshot` no devolvió registros para fechas de 2010, 2023 y 2025 | 5 sondeos | No capturadas | No hay serie oficial antigua de Shenzhen |
| EODHD con su clave demo pública | Cierres negociados, dividendos sin ajustar con sus fechas y splits | 12 capturas de AAPL, AMZN, TSLA y VTI entre 2000 y 2023 | La documentación capturada limita la clave demo a seis tickers. Uso privado de validación | Contraste de cierres y eventos US |
| Alpha Vantage con su clave demo pública | `TIME_SERIES_DAILY` sin ajustar, dividendos y splits de IBM. `LISTING_STATUS` de bajas solo para el 10 de julio de 2014 | 5 capturas | Términos capturados (PDF): uso personal y no comercial, que incluye investigación, sin facilitar los datos a terceros | Contraste de IBM y prueba de la lista de bajas |
| Banco Mundial, WDI `CM.MKT.LDOM.NO` | Empresas domésticas cotizadas en EE. UU. y China, 1995-2024, con origen WFE | 1 captura JSON | El catálogo del Banco Mundial publica WDI con CC BY 4.0. La página capturada redirige a las condiciones generales y no contiene ese texto, así que la licencia queda por capturar | Cobertura anual |
| SEC EDGAR (`data.sec.gov`) | Dividendos declarados en XBRL y avisos Form 25 de exclusión, de dominio público | HTTP 403 «Undeclared Automated Tool» | Exige un User-Agent con correo de contacto. No se envió ninguno | Pendiente de autorizar un contacto |
| Nasdaq.com, API de cotizaciones | Histórico y dividendos | 2 tiempos de espera agotados | Términos de uso personal | No utilizado |
| Nasdaq Data Link, `WIKI/PRICES` | Cierres sin ajustar y eventos hasta 2018 | HTTP 403, exige clave | Condiciones de Nasdaq Data Link | No utilizado |

Stooq ya figuraba como bloqueado por una verificación de navegador en las [fuentes complementarias](free-data-sources.md), y sus series están ajustadas. CRSP, Compustat, Norgate o Sharadar ofrecen precios y retornos de salida completos, pero son de pago y no se han consultado.

La disponibilidad de una respuesta no acredita un permiso de redistribución. El recibo público solo contiene metadatos, recuentos y tasas derivadas.

## Método

### Inversión del ajuste

Para un activo, sea `P_t` el cierre negociado, `R_u` la razón de split en la sesión `u` y `d_e` el dividendo por acción en la fecha ex `e`. El proveedor publica

```text
K_t = prod_{u > t} R_u                  splits posteriores a t
C_t = P_t / K_t                         cierre ajustado por splits
D_e = d_e / K_e                         dividendo publicado
f_e = 1 - D_e / C_{e-1}                 factor multiplicativo de dividendo
A_t = C_t * phi * prod_{e > t} f_e      cierre ajustado publicado
```

`phi` recoge las distribuciones que el proveedor aplicó después de la última fila del archivo. Si se escribe `G_t = 1 / (phi * prod_{e > t} f_e)`, la definición de `f_e` da `A_{e-1} * G_e = C_{e-1} - D_e`, y de ahí

```text
G_{e-1} = G_e + D_e / A_{e-1}
P_t = K_t * A_t * (gamma + S_t),    S_t = sum_{t < e <= c} D_e / A_{e-1}
```

La inversa del factor de dividendo es aditiva. `gamma = G_c` es una constante por activo que acumula todas las distribuciones posteriores al corte `c = 2023-12-31`, incluidas las que el proveedor aplicó sin publicarlas en el archivo. Por eso la reconstrucción no necesita leer precios ni dividendos posteriores al corte. Las razones de split posteriores al corte sí se leen, solo como escala de `K_t`. El volumen negociado es `V_t / K_t` y el dividendo por acción negociada es `D_e * K_e`.

En la exploración inicial se aplicó la inversión exacta con todas las filas del archivo y `gamma = 1`. Ese recorrido sí leyó filas y eventos de 2024 y 2025. Situó en rejilla el 83,5 % de los cierres US y el 94,2 % de los chinos, frente a un 1,6 % y un 0,8 % esperados por azar. Pero 579 activos US quedaban por debajo del 10 %. El análisis por tramos mostró que cada tramo entre fechas ex se corregía con su propia constante y que esas constantes derivaban lentamente. Es lo que produce una recursión que arranca con un factor terminal distinto de uno, y coincide con un archivo descargado después de alguna fecha ex de abril de 2025. La formulación con `gamma` corrige ese defecto sin mirar el periodo sellado.

Cuando coinciden split y dividendo, el proveedor no divide el dividendo publicado por el split del mismo día, aunque sí divide por él el cierre anterior. El rendimiento aplicado es `d_e * R_e / P_{e-1}`, mayor que el económico. La edición invierte esa convención tal cual, porque busca el precio negociado y no corregir el ajuste. En la exploración, sobre los 12 activos chinos de Shenzhen con al menos dos coincidencias, la convención del proveedor sitúa en rejilla entre el 99,4 % y el 100 % de los cierres en 11 de ellos. La alternativa económicamente coherente se queda entre el 8 % y el 86 %. El duodécimo, ZTE, falla con ambas.

### Unidad mínima de cotización

| Mercado y periodo | Unidades admitidas | Fuente |
| --- | --- | --- |
| EE. UU., antes del 28-08-2000 | 1/256 USD, la fracción más fina en uso | Calendario de decimalización de la SEC (Release 34-42360) y GAO-05-535 |
| EE. UU., del 28-08-2000 al 08-04-2001 | 1/256 o 0,01 USD | La NYSE completó la conversión el 29-01-2001 y Nasdaq el 09-04-2001. El CSV no identifica la bolsa de cotización |
| EE. UU., desde el 09-04-2001 | 0,01 USD desde 1 USD y 0,0001 USD por debajo | Regla 612 de la Regulation NMS (17 CFR 242.612) |
| China, acciones A | 0,01 CNY | Reglas de negociación de SSE y SZSE, recogidas en las [reglas del mercado chino](../engineering/china-market-rules.md) |

Las referencias de la SEC y de la GAO no se capturaron, porque el dominio de la SEC rechaza el acceso automatizado sin contacto. Antes de 2005 ninguna regla impedía cotizar por debajo del céntimo bajo 1 USD. La unidad de 0,0001 se usa como la más fina admisible.

### Tolerancias y probabilidad de azar

Un cierre cae en rejilla si su distancia al múltiplo más próximo no supera `2e-6` veces el precio. La probabilidad de acertar por azar con un residuo uniforme es `min(1, sum 2 * tol * P / unidad)`, y el exceso sobre el azar es `(observado - azar) / (1 - azar)`. Con precios altos el azar crece. A 1.700 CNY ronda el 68 %, y por encima de 2.500 USD toda fila acierta. En esos activos la rejilla apenas discrimina y la evidencia depende de las fuentes externas.

La tolerancia se fijó después de la exploración. En 150 activos por mercado con validación superior al 99 %, los residuos relativos de los aciertos tienen una mediana de 8,1e-8 en EE. UU. y 5,8e-8 en China, y un cuantil 0,999 de 9,8e-7 y 9,4e-7. Es el orden del redondeo a `float32` del archivo más el redondeo de los dividendos publicados.

### Estimación de la constante terminal

Los candidatos de `gamma` son exactamente los valores en `[1 - 2e-5, 1,25]` que llevan el último cierre válido a un múltiplo de su unidad. Se cuentan los aciertos en las últimas 60 sesiones válidas con una tolerancia de identificación de `3e-7`, cercana al cuantil 0,9 medido. Con la tolerancia de validación, dos candidatos contiguos de un precio alto empatan en sesiones de precio parecido. Se acepta el mejor si supera al segundo por un 10 % de la ventana y si sitúa al menos el 90 % de la ventana en rejilla con `2e-6`. Si hay empate, la ventana crece a 125, 250 y 500 sesiones con los candidatos que seguían por encima del 60 %. Los fallos se registran como `no_grid_solution`, `ambiguous`, `insufficient_rows` o `no_candidates`.

La ventana suele ser el último trimestre de 2023 y, si se amplía, llega a 2022. Ese periodo es validación en la campaña. `gamma` es una sola constante por activo y su función es recuperar un precio que ya era observable en cada fecha. No selecciona modelos ni umbrales. Aun así, el precio reconstruido es una reconstrucción retrospectiva y no una captura de la época.

### Verificación por tramos y controles

Las acciones corporativas dividen la serie en tramos. Se recorren desde el más reciente y el primero con al menos cinco filas y menos de un 90 % en rejilla detiene la verificación. Un tramo más corto que falla no basta para rechazar los anteriores, pero sus filas quedan sin verificar. La columna `verified` marca las filas aceptadas y `verified_since` es la primera de ellas. La granularidad es el tramo. Un ajuste no publicado dentro de un tramo invalida el tramo entero.

Los controles negativos repiten la reconstrucción, con su propia `gamma`, sin dividendos o con las fechas ex desplazadas una sesión hacia delante o hacia atrás. Se miden en las filas afectadas, anteriores al último dividendo previo a la ventana de ajuste. Un control que no encuentra `gamma` también se cuenta como fallo del control.

### Contraste externo

Un cierre coincide si dista de la fuente menos de media unidad de su redondeo publicado (0,005, o 0,00005 bajo 1 USD) más `2e-6` relativo. «Exacto» exige solo la holgura relativa. Los eventos se emparejan por fecha ex. Los dividendos coinciden con una tolerancia relativa de `5e-3` y los splits con `1e-6`.

## Resultados

### Muestra auditada

La muestra se eligió antes de construir la edición completa, por estratos de mercado, bolsa y antigüedad, ordenando por `SHA-256(semilla:mercado:símbolo)` con la semilla `mars-titan-unadjusted-20261009`. Tiene 40 activos US, 10 por estrato de inicio (2000, 2001-2009, 2010-2017 y desde 2018), y 18 chinos, 3 por bolsa y estrato (2006, 2007-2014 y desde 2015). Se excluyeron los activos inspeccionados durante la exploración. AAPL, AMZN, TSLA, IBM, VTI y 600519.SS forman aparte el grupo con fuente externa. Antes de construir la muestra fijé como aceptable una mediana por activo de al menos un 95 % en validación en ambos mercados, controles cerca del azar y al menos un 95 % de cierres externos coincidentes.

| Muestra aleatoria | Activos | Verificados | Parciales | Sin verificar | Sin `gamma` | Filas verificadas | Validación | Azar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EE. UU. | 40 | 27 | 5 | 1 | 7 | 79,5 % | 94,2 % | 1,7 % |
| China | 18 | 17 | 1 | 0 | 0 | 97,5 % | 97,7 % | 0,8 % |

La validación usa las filas anteriores a la ventana de ajuste de los activos con `gamma`. La mediana por activo es 1,0 en ambos mercados. Cinco de los siete fallos US son fondos cotizados con cierres finales fuera de rejilla (CFA, GSEW, RESD, TPIF y ULTR). Son fondos de poco volumen y la explicación más simple es que el proveedor registre el último cruce, que puede no caer en céntimo. No lo he comprobado con otra fuente. AMRN tiene sus últimos cierres de 2023 en múltiplos de 0,20 USD sin split posterior en el archivo. Es coherente con un contrasplit aplicado por el proveedor después de la última fila, que `gamma` no puede representar. USAS cotiza por debajo de 1 USD con cierres en céntimos y varios candidatos empatan. Entre los parciales, GSK falla en las sesiones anteriores a la escisión de Haleon de julio de 2022. El archivo registra una agrupación de 0,8 el 19 de julio y un split de 1,226 el 22 de julio. Los tres cierres entre ambas fechas y el tramo anterior quedan fuera de rejilla.

En las filas afectadas por dividendos, el método sitúa en rejilla el 92,4 % (EE. UU.) y el 97,5 % (China). Los controles bajan al 0,6-2,1 % y al 0,5-0,8 %, o no encuentran `gamma`, como en 18 a 20 de los 24 activos US y en 2 de los 18 chinos.

### Contraste con fuentes externas

| Activos | Fuente | Sesiones comparadas | Coinciden | Exactas | En filas verificadas |
| --- | --- | ---: | ---: | ---: | ---: |
| 10 de Shanghái | SSE, serie oficial | 31.015 | 95,7 % | 95,7 % | 99,6 % de 29.676 |
| AAPL, AMZN, TSLA, VTI | EODHD | 21.143 | 99,9 % | 81,7 % | igual, todas verificadas |
| IBM | Alpha Vantage | 6.008 | 99,8 % | 97,4 % | igual, todas verificadas |

Las coincidencias no exactas con EODHD son diferencias de fracciones de céntimo en los periodos anteriores a splits recientes, de AAPL hasta 2020 y de TSLA hasta 2019. Los cierres fraccionarios reconstruidos de 2000, como 106 9/16, aparecen allí con cuatro decimales. Es coherente con un redondeo de la propia fuente, que reconstruye sus cierres antiguos desde series ajustadas. Las demás discrepancias son sesiones aisladas en las que el cierre del proveedor difiere de la fuente mientras las sesiones vecinas coinciden. No se corrigen. En 600428.SS, de las 1.339 filas que la verificación por tramos deja fuera, 1.191 no coinciden con SSE, y en las verificadas coinciden 2.896 de 2.903.

En las diez series de Shanghái hay 1.163 sesiones del proveedor que SSE no publica. En 600519.SS son 67, todas con volumen cero, y 65 repiten el cierre anterior. En 600239.SS son 488, con 487 de volumen cero y el cierre anterior repetido. Son sesiones de suspensión que el proveedor rellena. Una cinta de negociación no debe ejecutar órdenes en esas filas.

Los eventos coinciden en fecha e importe: 46 de 46 dividendos y 4 de 4 splits de AAPL, 96 de 96 dividendos y el split de IBM, y los splits de AMZN y TSLA, que no pagaron dividendos en el periodo. En VTI coinciden 90 de 91 dividendos y el restante difiere en un día de la fecha ex. El split de IBM de 2021 es la escisión de Kyndryl, registrada como split de 1,046 por ambas fuentes.

### Edición completa

Con la muestra dentro de los criterios, se ejecutó la edición sobre los 5.012 activos con precios. Tardó 204 segundos de reloj y 178 de CPU en un solo proceso con prioridad reducida, alcanzó 383 MB de memoria residente y ocupa 716 MB. No interfirió con la codificación, que usa unos seis núcleos y la GPU.

| Edición | Activos | Verificados | Parciales | Sin verificar | Sin `gamma` | Filas verificadas | Validación | Azar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EE. UU. | 4.202 | 3.377 | 278 | 41 | 506 | 91,4 % | 99,0 % | 1,7 % |
| China | 810 | 705 | 97 | 1 | 7 | 93,5 % | 95,0 % | 0,8 % |

En EE. UU. se verifican por completo 2.678 de las 2.932 series con sector declarado y 699 de las 1.270 marcadas `N/A`, que incluyen los fondos cotizados. 385 de esas 1.270 no encuentran `gamma`. En las filas afectadas por dividendos, el método alcanza el 99,0 % y el 94,5 %, y los controles el 1,4-2,2 % y el 0,8-1,2 %. `gamma` tiene una mediana de 1,0175 en EE. UU. y 1,0151 en China, con un máximo de 1,249 en sociedades hipotecarias de rendimiento alto. Hay 405.403 filas US y 110.565 chinas con volumen cero.

## La edición sin ajustar

La edición `1ac37278…` está en el estado privado `unadjusted-prices-20261009/edition-v1`, fuera del repositorio. Cada activo tiene `prices.parquet`, `events.parquet` y `receipt.json`. Los precios ocupan las mismas sesiones que la población preparada, con apertura, máximo, mínimo y cierre negociados, volumen, `split_multiplier`, `dividend_offset`, `on_grid` y `verified`. Los eventos llegan hasta el corte, con el importe publicado, el dividendo por acción negociada y la razón de split. El recibo conserva las huellas del CSV y de la preparación, el ajuste de `gamma`, la validación, los tramos fallidos y los controles.

La identidad combina el esquema, la política y las huellas del código. El manifiesto declara `price_basis="unadjusted_reconstructed"` y `corporate_actions_complete=false`. Una nueva ejecución verifica las huellas y reutiliza los activos sin cambios. Un artefacto alterado o una fuente distinta se reconstruyen o se rechazan.

```bash
PYTHONPATH=src uv run --no-sync python -m mars_titan.data.unadjusted_edition sample \
  --prepared <población preparada> --output <muestra.json> --exclude US:AAPL ...
PYTHONPATH=src uv run --no-sync python -m mars_titan.data.unadjusted_edition build \
  --prepared <población preparada> --dataset dataset --output <destino> [--assets <muestra.json>]
```

El código está en [`unadjusted_prices.py`](../../src/mars_titan/data/unadjusted_prices.py), [`unadjusted_edition.py`](../../src/mars_titan/data/unadjusted_edition.py) y [`unadjusted_evidence.py`](../../src/mars_titan/data/unadjusted_evidence.py). La preparación leyó el CSV con pandas y su analizador difiere del de pyarrow hasta en 3,8e-13 relativo en precios muy pequeños. La edición lo admite hasta 1e-12 y aplica los factores a los valores preparados.

La edición no satisface por sí sola el contrato de `MarketTape`. Una cinta real debe limitarse a filas verificadas, tratar el volumen cero como sesión sin negociación y declarar las acciones corporativas que use. Las escisiones registradas como splits fraccionarios y las ampliaciones no publicadas impiden afirmar que las acciones estén completas.

## Limitaciones

- La evidencia independiente cubre 15 activos. La rejilla es una comprobación interna y su tolerancia se eligió tras explorar toda la población.
- Las escisiones aparecen como splits no enteros o no aparecen. Las ampliaciones de capital chinas (配股) no figuran como eventos y explican parte de los tramos fallidos.
- Los contrasplits posteriores a la última fila del archivo no se pueden recuperar con `gamma`.
- Los fondos con poca negociación tienen cierres fuera de rejilla. Su base de precios queda sin verificar.
- Con precios altos la rejilla apenas discrimina. Esos activos dependen del contraste externo, que solo cubre 600519.SS.
- La serie del proveedor incluye sesiones de suspensión con volumen cero y cierre repetido.
- Las discrepancias aisladas del proveedor frente a la bolsa se conservan.
- No hay serie oficial antigua de Shenzhen. Los 341 activos de esa bolsa solo tienen la comprobación interna.

## Bajas: viabilidad y decisión

La población es de supervivientes. Según la [revisión de integridad](../engineering/rl-environment-integrity.md), solo dos de los 4.202 activos US terminan antes de diciembre de 2023, y los 810 chinos llegan al 31 de marzo de 2025 en el archivo. Siete códigos chinos del archivo figuran hoy en las listas oficiales de bajas, todos excluidos en 2025, después de la última fila.

**Estados Unidos.** No he encontrado una fuente libre y verificable de precios y retornos de salida de 2000 a 2023. Los retornos de salida de referencia proceden de CRSP, de pago. La clave demo de Alpha Vantage solo devuelve 430 valores dados de baja a 10 de julio de 2014, la mayoría entre 2009 y 2014 y sin precios. EDGAR permitiría localizar los avisos Form 25, pero exige declarar un correo de contacto. Nasdaq Data Link y las API de Nasdaq requieren clave o rechazan el acceso automatizado. No es viable ahora.

**China.** Es parcialmente viable. Las bolsas publican las listas de bajas: 101 de Shanghái y 129 de acciones A de Shenzhen entre 2000 y 2023, de ellas 43 en 2022 y otras 43 en 2023. Shanghái conserva las series diarias de códigos excluidos, comprobado con 600087 hasta su última sesión de junio de 2014. Falta el retorno de salida. Después de la última sesión, el valor depende del traslado al sistema de valores excluidos, de la contraprestación de una fusión o de una oferta. Exige leer anuncios caso a caso en CNINFO. Para Shenzhen no he encontrado series oficiales antiguas.

**Decisión.** No se incorporan bajas en esta edición. Los resultados predictivos y financieros sobre la población se declaran condicionados a seguir cotizando en marzo de 2025. Se propone este análisis de sensibilidad, que no se ha ejecutado por el bloqueo de aprendizaje:

1. Publicar con cada resultado la cobertura anual. La tabla compara las series con sector declarado del archivo y las empresas domésticas cotizadas según WDI. El archivo incluye emisores extranjeros, así que la proporción es orientativa. El recuento chino de WDI salta de 3.584 en 2018 a 12.730 en 2019, así que solo se usa hasta 2018.

   | Año | Series US con sector | Cotizadas US (WDI) | Proporción | Series CN | Cotizadas CN (WDI) | Bajas A en SSE y SZSE |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | 2000 | 1.310 | 6.917 | 19 % | 0 | 1.086 | 0 |
   | 2006 | 1.702 | 5.133 | 33 % | 466 | 1.421 | 12 |
   | 2010 | 1.932 | 4.279 | 45 % | 601 | 2.063 | 4 |
   | 2015 | 2.392 | 4.381 | 55 % | 687 | 2.827 | 7 |
   | 2018 | 2.721 | 4.013 | 68 % | 748 | 3.584 | 5 |
   | 2023 | 2.930 | 4.317 | 68 % | 810 | (ruptura) | 43 |

2. Repetir las métricas financieras con bajas sintéticas declaradas antes de ejecutar. Cada año se retira al azar una fracción de los activos evaluados (por ejemplo 2 %, 5 % y 8 %, y la tasa oficial china cuando exista), en una sesión aleatoria y con un retorno de salida de 0 %, −30 % o −100 %. Shumway (1997) estimó retornos de salida del orden de −30 % para las bajas por mal rendimiento en NYSE y AMEX, y Shumway y Warther (1999) del orden de −55 % en Nasdaq. Las semillas, las tasas y los retornos se fijan antes de abrir resultados.
3. Informar de las conclusiones que cambian de signo o de orden en algún escenario. Si ocurre, el resultado no debe presentarse como general.

Para cerrar la limitación en EE. UU. haría falta una fuente con licencia, como CRSP a través de una institución, o autorizar un contacto para EDGAR y construir las bajas desde los Form 25. En China, el siguiente paso sería revisar en CNINFO los anuncios de las bajas de los antiguos componentes del CSI 300.

## Comprobaciones

Las pruebas nuevas cubren la inversión sobre series generadas con la convención del proveedor, el almacenamiento en `float32`, los controles, la rejilla de cada periodo, la constante terminal con precios altos, la recuperación de la edición, las huellas y los lectores de evidencia. Trece mutaciones dirigidas se detectaron, entre ellas usar el cierre de la fecha ex, incluir el split del mismo día, identificar con la tolerancia de validación, quitar la unidad de céntimo en la transición de 2000-2001, no detener la verificación en un tramo fallido o verificar un tramo corto fuera de rejilla. No se midió la cobertura porque el entorno compartido no incluye `coverage` y no debía modificarse.

## Referencias

- Shumway, T. (1997). The delisting bias in CRSP data. *The Journal of Finance, 52*(1), 327-340.
- Shumway, T., & Warther, V. A. (1999). The delisting bias in CRSP's Nasdaq data and its implications for the size effect. *The Journal of Finance, 54*(6), 2361-2379.
- U.S. Government Accountability Office. (2005). *Securities markets: Decimal pricing has contributed to lower trading costs and a more challenging trading environment* (GAO-05-535).
- U.S. Securities and Exchange Commission. (2000). *Order directing the exchanges and the NASD to submit a phase-in plan to implement decimal pricing* (Release No. 34-42360).
- Regulation NMS, Rule 612, Minimum pricing increment, 17 C.F.R. § 242.612.
- Xu, W., et al. (2025). FinMultiTime: A four-modal bilingual dataset for financial time-series analysis. arXiv:2506.05019.
