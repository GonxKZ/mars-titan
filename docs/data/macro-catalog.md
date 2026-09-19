# Catálogo macroeconómico de MARS-TITAN

Autor: Gonzalo García Lama. Verificación documental: 18 de septiembre de 2026.

El [catálogo CSV](../../data/catalogs/macro-indicators.csv) contiene 140 indicadores candidatos. Sus estados documentan la revisión inicial, no el estado de cada ejecución. Son 64 series de proveedores oficiales con metadatos contrastados, 70 transformaciones y seis candidatos del NBS/PBOC sin identificador estable verificado. Las 64 series incluyen 63 fichas FRED y el GSCPI del New York Fed.

La preparación posterior adquirió 59 series completas y calculó valores para 125 indicadores en al menos una fecha. Los paneles conservan también las ausencias y sus causas. La unidad del catálogo es una referencia de diseño, no sustituye las unidades históricas de cada versión. No se ha ejecutado una comparación confirmatoria predictiva. Véanse [preparación](preparation.md) y los informes de cálculo [US](../../reports/data/macro-US-calculation.json) y [CN](../../reports/data/macro-CN-calculation.json).

El catálogo sirve para seleccionar familias macro con un contrato temporal común. No propone introducir 140 columnas de forma automática ni afirma que una mayor cantidad de indicadores mejore la predicción. La selección pertenece al periodo de desarrollo. La cobertura estadounidense es más amplia que la china y no debe presentarse como simétrica.

## Contenido y estado

La descarga histórica no requiere una clave en la vía pública de ALFRED utilizada
en esta preparación. Se seleccionan todas las fechas de vintage del intervalo y
se valida el archivo devuelto. El formulario web puede cambiar. Un cambio de
estructura produce un error, no una descarga aparentemente correcta.

```bash
uv run python - <<'PY'
import csv
from pathlib import Path
from mars_titan.data.macro_acquisition import acquire_catalog
from mars_titan.data.storage import atomic_json

with Path("data/catalogs/macro-indicators.csv").open() as stream:
    catalog = list(csv.DictReader(stream))
report = acquire_catalog(
    catalog, Path("data/external/phase1-macro"),
    observation_start="1988-01-01", observation_end="2025-03-31",
    realtime_start="1990-01-01", realtime_end="2025-03-31", workers=2,
)
report["destination"] = "data/external/phase1-macro"
atomic_json(Path("reports/data/macro-acquisition.json"), report)
print(report["completed_series"], report["failed_series"])
PY
uv run mars-data macro --market US
uv run mars-data macro --market CN
```

La captura ejecutada conserva 59 series completas y dos errores pendientes, WTI
y Brent. No se presentan las 61 adquisiciones como correctas. Los errores y las
exclusiones se trasladan al cálculo mediante `execution_catalog`. Las series que
fallan después de algún lote no exponen sus filas parciales al motor.

| Familia | Series del proveedor y candidatos | Transformaciones |
| --- | ---: | ---: |
| Inflación | 7 | 16 |
| Empleo | 10 | 12 |
| Actividad | 7 | 7 |
| Vivienda | 2 | 3 |
| Cuentas nacionales | 7 | 8 |
| Tipos de interés | 13 | 12 |
| Liquidez y dinero | 6 | 2 |
| Crédito | 6 | 4 |
| Divisas | 6 | 2 |
| Materias primas | 3 | 2 |
| Condiciones financieras | 2 | 1 |
| Cadenas de suministro | 1 | 1 |
| Total | 70 | 70 |

`raw` significa serie publicada por un proveedor. Puede ser un índice calculado por ese proveedor, como NFCI, GSCPI o ventas minoristas reales. No significa que sea una medición directa ni que esté libre de revisiones. `derived` identifica una fórmula propuesta por MARS-TITAN. Las transformaciones no son fuentes independientes y no duplican evidencia.

Los estados tienen un significado limitado:

- `verified_metadata_not_ingested`: ficha oficial comprobada, observaciones y disponibilidad histórica todavía sin auditar.
- `provider_verified_identifier_pending`: existe el proveedor y la familia de información, pero no se ha validado un código estable. `series_id` contiene literalmente `no identifier verified` y la serie queda excluida de cualquier ejecución.
- `formula_defined_not_computed`: fórmula y dependencias definidas, sin valores calculados ni validación numérica con un panel real.

`core`, `secondary` y `optional` ordenan la auditoría propuesta. No son resultados de selección de variables. Una serie `core` que no supere la auditoría temporal se excluye igualmente. Los índices con estimación retrospectiva y los candidatos chinos sin identificador son optativos.

## Contrato de cada campo

`id` es el identificador interno único. `name`, `category` y `geography` describen la magnitud. `provider` conserva el productor y, cuando corresponde, la distribución a través de FRED. Los identificadores de FRED no se presentan como códigos nativos de BLS, BEA, BIS u OECD. `source_url` enlaza la ficha contrastada. En un derivado enlaza la primera dependencia y `input_ids` enumera todas mediante `|`.

`frequency` describe el periodo observado, no la frecuencia de publicación. `M` y `Q` son mes y trimestre. `Q_END` es un saldo de fin de trimestre. `D` representa los días con observación del proveedor y `D7` una serie diaria que también puede incluir fines de semana. `W_SAT`, `W_FRI` y `W_WED` indican la referencia semanal. `W_WED_LEVEL` es un saldo del miércoles y `W_WED_AVG` una media semanal terminada en miércoles. Estos últimos no son intercambiables.

`unit` utiliza `million = 10^6` y `billion = 10^9`, evitando la ambigüedad del billón español. `SA` significa ajuste estacional, `NSA` ausencia de ajuste y `SAAR` nivel desestacionalizado expresado a tasa anual. `percent_pa` es un tipo anual cotizado en porcentaje. Las diferencias entre dos porcentajes están en puntos porcentuales, no en porcentaje de variación. Los importes PBOC pendientes se normalizarían a miles de millones de CNY solo después de verificar la unidad del comunicado.

`formula` define exclusivamente derivados. `availability_rule` remite a las reglas siguientes. `vintage_policy` establece el tratamiento obligatorio de las revisiones. `verification_status` y `verified_on` documentan la comprobación realizada, no la actualidad de cada observación.

## Reloj de información

Cada observación futura necesitará al menos `period_start`, `period_end`, `published_at`, `vintage_at`, `source_first_available_at`, `ingested_at`, `source_timezone`, `source_version`, URL del comunicado y hash del archivo. En un flujo real, `available_at` no puede preceder ni a la publicación ni a la recepción efectiva. En una reconstrucción histórica, la descarga actual no es una recepción histórica: se necesita evidencia archivada de disponibilidad y una latencia de procesamiento declarada. Ambos modos deben quedar identificados y no mezclarse.

Todas las entradas cumplirán `available_at <= prediction_at`. Las horas se convierten a UTC a partir de la zona IANA original, principalmente `America/New_York`, `Asia/Shanghai` y `Europe/Frankfurt`. El cambio de hora europeo o estadounidense no se sustituye por un desplazamiento UTC fijo. La fecha del dato, la actualización de FRED y la fecha del comunicado son campos distintos.

Si existe una fecha de publicación fiable pero se desconoce la hora, se considera disponible después del cierre del siguiente día de negociación del mercado objetivo posterior a esa fecha local. Se contrasta además una demora de dos sesiones. La regla se aplica por mercado y deja constancia de la aproximación. Si falta también la fecha o la versión histórica, no basta con sumar treinta días al periodo: la observación queda fuera del análisis point-in-time estricto.

Las revisiones publicadas junto a un dato nuevo son información nueva. Pueden actualizar el estado desde su publicación, pero no reescriben las entradas o predicciones ya emitidas. El [contrato de datos](data-contract.md) y el [protocolo](../research/protocol.md) gobiernan el resto de la cronología.

## Reglas de publicación del CSV

Las horas habituales de la tabla orientan la búsqueda del comunicado. Siempre prevalecen la fecha y hora de la publicación efectiva, incluidas demoras extraordinarias. Un calendario previsto no acredita que la publicación se produjera. Las reglas se reconstruyen con la versión aplicable al año evaluado.

| Código | Evidencia y aplicación |
| --- | --- |
| `BLS_CPI`, `BLS_PPI`, `BLS_EMP` | Comunicado de precios o empleo correspondiente. Hora habitual 08:30 de Nueva York, contrastada con el [calendario BLS](https://www.bls.gov/schedule/2026/home.htm). Empleo y paro proceden de encuestas distintas aunque compartan publicación. |
| `BLS_JOLTS` | Comunicado JOLTS, normalmente a las 10:00. Su mes de referencia suele estar más retrasado que el de empleo. No copiar el calendario de PAYEMS. |
| `BEA_PI`, `BEA_GDP` | Publicaciones de renta y consumo o PIB del [calendario BEA](https://www.bea.gov/news/schedule). Hora habitual 08:30. Registrar estimación adelantada, segunda, tercera y revisiones de cuentas. |
| `DOL_CLAIMS` | Comunicado semanal de [solicitudes de desempleo](https://oui.doleta.gov/unemploy/claims.asp). Las solicitudes iniciales y continuadas tienen referencias distintas. Conservarlas aunque aparezcan en el mismo documento. |
| `FED_G17` | Comunicado [G.17](https://www.federalreserve.gov/releases/g17/), con calendario y revisiones de producción y capacidad. No anticipar al mes observado. |
| `CENSUS_HOUSING`, `CENSUS_RETAIL`, `CENSUS_ORDERS` | Comunicado concreto del [calendario Census](https://www.census.gov/economic-indicators/calendar-listview.html). No aplicar la fecha de ventas adelantadas a pedidos manufactureros ni a revisiones posteriores. |
| `FRED_REAL_RETAIL` | La serie RRSFS ya es un cálculo de St. Louis. Exigir disponibilidad de la versión publicada y de sus insumos, con la fecha más tardía. No reconstruir retrospectivamente con un IPC revisado hoy. |
| `FED_EFFR` | Tipo efectivo de la fecha de negociación, publicado después. Contrastar el [calendario y metodología del New York Fed](https://www.newyorkfed.org/markets/reference-rates/effr). DFF incluye días sin nueva negociación. No contarlos como nuevos anuncios. |
| `FOMC_TARGET` | Separar hora del anuncio y entrada en vigor. Los límites diarios representan el tipo vigente. Una decisión anunciada para el día siguiente puede ser un evento conocido, pero no el tipo efectivo de hoy. |
| `FED_H15` | Publicación [H.15](https://www.federalreserve.gov/releases/h15/) y disponibilidad del distribuidor. Un rendimiento fechado hoy no se considera automáticamente conocido al cierre bursátil de las 16:00. |
| `FED_H41` | Publicación [H.4.1](https://www.federalreserve.gov/releases/h41/). WALCL es un saldo del miércoles, WRESBAL y WTREGEN son medias semanales. El miércoles de referencia no es el momento de publicación. |
| `NYFED_RRP` | Resultado publicado de la operación [reverse repo](https://www.newyorkfed.org/markets/desk-operations/reverse-repo). Registrar finalización y publicación, sin deducir la hora del resultado a partir de la fecha diaria. |
| `FED_H6` | Publicación [H.6](https://www.federalreserve.gov/releases/h6/). Registrar cambios de definición, revisiones estacionales y el calendario mensual vigente. |
| `FED_H8`, `FED_H8_MONTHLY` | Publicación [H.8](https://www.federalreserve.gov/releases/h8/). TOTBKCR es semanal, BUSLOANS y CONSUMER del catálogo son mensuales. Una media mensual requiere la publicación de todos sus componentes y su versión. |
| `FED_H10` | H.10 difunde semanalmente valores diarios de divisas de la semana anterior. Su [documentación](https://www.federalreserve.gov/releases/h10/about.htm) indica actualización los lunes a las 16:15. Usar el comunicado realmente publicado y sus cambios por festivos. |
| `EIA_SPOT` | Publicación y archivo de precios al contado de EIA. Son cotizaciones de petróleo, no retornos de un futuro negociable. Sin hora acreditada se aplica la regla de fecha conocida y demora conservadora. |
| `STLFSI_RELEASE`, `NFCI_RELEASE` | Versión publicada del índice y de su metodología. No emplear hacia atrás la normalización o los coeficientes estimados con el histórico actual. Exigir archivo de vintages o limitar al seguimiento prospectivo. |
| `NYFED_SOFR` | SOFR corresponde a operaciones del día previo y se publica aproximadamente a las 08:00 del siguiente día hábil. Comprobar revisiones y su hora en la [documentación oficial](https://www.newyorkfed.org/markets/reference-rates/sofr). |
| `BIS_RELEASE` | Comunicado BIS efectivo, cobertura del trimestre o mes y edición del archivo. El ajuste por rupturas y el denominador PIB pueden revisarse. No equiparar el primer periodo de la serie a su primera disponibilidad pública. |
| `ECB_POLICY` | Comunicado del BCE y fecha de entrada en vigor de su facilidad de depósito. Validar el calendario histórico con [los tipos oficiales](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html). |
| `OECD_RELEASE`, `IMF_COMMODITY` | Edición de OECD o IMF y fecha comprobada de incorporación. Una media mensual alemana o de cobre no es un precio diario ejecutable. Sin archivo histórico de ediciones se excluye del PIT estricto. |
| `NYFED_GSCPI` | La [página oficial](https://www.newyorkfed.org/research/policy/gscpi) anuncia actualización a las 10:00 del cuarto día hábil del mes. Exigir la versión del indicador y su método que existían en la fecha evaluada. |
| `NBS_RELEASE`, `NBS_PMI` | Comunicado chino original, calendario y unidad exacta. La traducción inglesa puede publicarse después. Enero y febrero pueden difundirse conjuntamente para algunas magnitudes. No crear un enero mensual inexistente. |
| `PBOC_RELEASE` | Tabla o comunicado original de M2 o financiación agregada, con fecha y perímetro. Distinguir flujo del mes, acumulado y saldo. Las seis filas chinas pendientes no pasan a admisibles solo por aplicar un desfase. |
| `MAX_INPUT_AVAILABLE_AT` | El derivado aparece cuando estén disponibles todas las observaciones y versiones de su fórmula, incluidas las de los retardos. Recalcular solo para decisiones posteriores al evento. |
| `COMMON_PERIOD_MAX_INPUT_AVAILABLE_AT` | Lo anterior, usando además el último periodo común a las dependencias. No restar un TIPS de ayer a un Treasury de hoy y llamarlo diferencial de hoy. |

La tabla no fija retardos constantes de publicación que aparenten reconstruir un archivo histórico inexistente. Por ejemplo, H.10 permite verificar una demora real del canal utilizado. No se sustituye ese canal por una supuesta cotización intradiaria a la que el proyecto no tenía acceso.

## Vintages y fórmulas

`ALFRED_OR_RELEASE_ARCHIVE` exige seleccionar para cada decisión la versión con evidencia de publicación no posterior al corte. [ALFRED](https://alfred.stlouisfed.org/help) conserva versiones, pero puede añadirlas después de la publicación y no ofrece por sí solo una hora intradiaria universal. Hay que comprobar su cobertura por serie. `MODEL_VINTAGES_ONLY` añade que el cálculo y la versión del modelo también deben existir en esa fecha. `NO_VINTAGES_EXCLUDE` mantiene fuera del conjunto estricto a los candidatos pendientes. `DERIVE_FROM_ASOF_VINTAGES` hereda todas las restricciones de las dependencias.

En las fórmulas, `p` es el último periodo de referencia admisible a la hora de decisión. `x[p-k]` es el periodo anterior correspondiente, consultado en la misma instantánea permitida. Para datos mensuales y trimestrales se exige el periodo de calendario exacto. Para las series semanales se exige la semana indicada. En `D`, veintiuna observaciones son veintiún registros diarios válidos del proveedor, no necesariamente veintiuna sesiones del mercado de la acción. En `D7`, el retardo de veintiún días es natural. Un hueco no se convierte en cero ni se elimina para fabricar una ventana de calendario completa.

`mean(x[p-3:p])` incluye los cuatro periodos desde `p-3` hasta `p`. `^` significa potencia matemática. Las tasas usan `100*(x[p]/x[p-k]-1)` y requieren niveles positivos en ambos extremos. Ante valores no positivos, incluido un episodio excepcional del petróleo, la tasa se marca ausente y se conserva la causa. No se cambia a logaritmos ni se recorta usando información futura. La anualización de tres meses o un trimestre eleva el cociente a cuatro. No anualiza una rentabilidad negociada ni representa una previsión.

Los diferenciales Treasury menos TIPS son aproximaciones a compensación por inflación. Incluyen primas y diferencias de liquidez. `2*breakeven_10y-breakeven_5y` es una aproximación lineal basada en rendimientos a vencimiento constante, no una curva cero cupón ni una expectativa pura de inflación. Los cambios de tipos se miden en puntos porcentuales. Los saldos encadenados reales no deben sumarse como si fueran magnitudes nominales aditivas.

Ninguna fórmula contiene ventanas centradas, datos futuros o una normalización estimada con toda la muestra. Un escalado, PCA, residualizador macro o clasificador de regímenes añadido después tendrá que ajustarse dentro de cada ventana de entrenamiento. No se deduce su autorización ni su utilidad del tamaño del catálogo.

## Acceso, cobertura histórica y condiciones

La comprobación empleó páginas oficiales y descargas públicas, sin claves aportadas por el usuario, suscripciones ni servicios de pago. La [página FRED-MD/FRED-QD](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) ofrece CSV y archivos mensuales históricos. Se conservaron capturas corrientes de ambos paneles en la [adquisición complementaria](free-data-sources.md). No se han integrado como entradas del benchmark ni reconstruido como observaciones disponibles históricamente. Tampoco equivalen al cálculo de las 140 variables del catálogo. La API ordinaria de FRED requiere una clave gratuita y no se ha usado. La [actualización manual](public-source-updates.md) reutiliza los enlaces públicos verificados y respeta sus límites.

| Proveedor | Acceso contrastado o ruta primaria | Restricción pendiente |
| --- | --- | --- |
| FRED/ALFRED, BLS, BEA, Reserva Federal, Treasury, Census, DOL y EIA | Fichas oficiales y portales públicos enlazados en el catálogo y en las reglas. [Condiciones FRED](https://fred.stlouisfed.org/legal/) y notas de cada serie. | Auditar primeras publicaciones, cobertura de versiones, cambios de base y derechos ajenos. Acceso público no es licencia MIT. |
| BIS | [Descargas completas](https://data.bis.org/bulkdownload), [metodología de crédito](https://data.bis.org/topics/TOTAL_CREDIT) y fichas de tipos efectivos. | Cumplir [condiciones de uso](https://data.bis.org/help/legal). Las revisiones de rupturas, ponderaciones y PIB requieren ediciones históricas. |
| ECB, OECD e IMF | Fichas FRED contrastadas para `ECBDFR`, `IRLTLT01DEM156N` y `PCOPPUSDM`, con productor identificado. | No se ha validado una API directa ni una historia completa de ediciones. Comprobar términos y cobertura en el productor antes de una ingesta. |
| NBS | [Calendario 2026](https://www.stats.gov.cn/english/PressRelease/ReleaseCalendar/202512/t20251226_1962154.html), comunicados originales y traducciones. | Reconstruir archivo por fecha, identificador y unidad. No atribuir un dato a una fecha anterior por usar su traducción posterior. |
| PBOC | [Portal estadístico](https://www.pbc.gov.cn/en/3688247/3688975/index.html), con tablas de dinero y financiación agregada. | Identificadores estables, condiciones de reutilización y cambios de perímetro pendientes. No hay un panel PIT chino preparado. |
| New York Fed y Chicago Fed | Páginas de índices y tipos de referencia. | Acreditar vintages, versiones del método y derechos de insumos de terceros. Un backcast no fue necesariamente un dato público contemporáneo. |

La comprobación actual no acredita máximos y mínimos históricos, porcentaje de ausencias ni cobertura coincidente con cada activo de FinMultiTime. Esas cifras se medirán tras una ingesta autorizada. La existencia de una serie desde 1959 o de un backcast desde 1997 no prueba que la versión actual estuviera disponible entonces. Por ejemplo, el New York Fed presentó públicamente el GSCPI el [4 de enero de 2022](https://libertystreeteconomics.newyorkfed.org/2022/01/a-new-barometer-of-global-supply-chain-pressures/). Su historia retrospectiva anterior no se admite como indicador público contemporáneo. Desde su presentación siguen siendo necesarias las ediciones que se publicaron en cada fecha.

## Comprobaciones y siguiente uso

Las claves del CSV son únicas. Las dependencias de sus 70 derivados existen, no hay ciclos y los seis candidatos sin identificador no alimentan ningún derivado. Las fórmulas solo usan periodos presentes admisibles y pasados. La revisión de unidades evita sumar millones con miles de millones, confundir porcentajes con puntos porcentuales o mezclar saldos con medias semanales.

Antes de calcular variables se fijará el corte temporal, se verificará la historia de publicaciones de una familia reducida y se generará un informe de cobertura. Después se compararán bloques de indicadores, conservando el test final cerrado. Cada familia puede excluirse por ausencia de señal, redundancia, coste de auditoría o falta de vintages. Este documento aporta preparación a O1, O3, O4 y O5. No cierra esos objetivos.
