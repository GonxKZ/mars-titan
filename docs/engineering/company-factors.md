# Factores financieros de empresa

Los factores empresariales complementan el contexto macro, no lo sustituyen.
Se calculan por activo y presentación a partir de importes contables sin
transformar. La unidad de cálculo es el grupo formado por `period_end`, `filed`
y `accession`. Dos cifras con el mismo cierre no son compatibles si pertenecen
a presentaciones distintas.

## Catálogo inicial

Las abreviaturas corresponden a conceptos `us-gaap` en USD. A es `Assets`, AC
es `AssetsCurrent`, P es `Liabilities`, PC es `LiabilitiesCurrent` y PN es
`StockholdersEquity`. CC y CP son `AccountsReceivableNetCurrent` y
`AccountsPayableCurrent`.

| Identificador | Cálculo | Condición del denominador |
| --- | --- | --- |
| `current_ratio` | AC / PC | PC > 0 |
| `working_capital_to_assets` | (AC − PC) / A | A > 0 |
| `liabilities_to_assets` | P / A | A > 0 |
| `equity_to_assets` | PN / A | A > 0 |
| `receivables_to_assets` | CC / A | A > 0 |
| `payables_to_assets` | CP / A | A > 0 |
| `liabilities_to_positive_equity` | P / PN | PN > 0 |

El patrimonio negativo se conserva en PN / A. No se divide entre patrimonio
nulo o negativo. `Liabilities` es pasivo total, no deuda financiera. La guía de
la [SEC sobre estados financieros](https://www.sec.gov/about/reports-publications/investorpubsbegfinstmtguide)
explica los componentes del balance, el fondo de maniobra y el cociente entre
pasivo y patrimonio. La normalización por activo de la tabla es una decisión
de representación del proyecto, no una regla de inversión.

No se obtiene caja acumulando su variación. Tampoco se calculan margen, ROA,
ROE, PER o deuda neta cuando faltan ingresos, beneficio, acciones o deuda
financiera compatibles. Un campo sin `period_start` no demuestra por sí solo
que sea un saldo. Los siete ratios usan únicamente los conceptos de balance
definidos en el catálogo y rechazan sus registros con periodo inicial.

## Disponibilidad y ausencias

Cada resultado conserva componentes, fuentes, numerador, denominador y estado.
La disponibilidad de un resultado válido es la publicación más tardía de sus
componentes. Los eventos de una presentación se procesan en orden de llegada.
Un conflicto posterior invalida el factor desde que se conoce, sin reescribir
la versión anterior.

Faltar un componente, tener una unidad incompatible, encontrar valores
contradictorios, dividir entre un denominador no positivo y desbordar float64
son motivos distintos de ausencia. Una nueva presentación incompleta también
produce ausencias explícitas. Los grupos se registran antes de seleccionar los
componentes para que una presentación con solo caja no conserve silenciosamente
un ratio que otra presentación con solo activo total dejaría ausente.

El registro de grupos es conservador. Incluye observaciones `us-gaap` en USD
sin periodo inicial, aunque el concepto no intervenga en un ratio, además de
los componentes del catálogo cuya incompatibilidad deba explicarse. Una
taxonomía que valide individualmente otros conceptos sigue siendo necesaria
antes de convertirlos en nuevas variables.

## Representación y almacenamiento

La representación original permanece disponible con 24 canales. La opción
`mars-data encode --company-factors` produce la versión 2 del esquema, con
45 canales: 15 valores, 15 máscaras y 15 antigüedades. Los ocho conceptos
originales preceden a los siete ratios en el orden de la tabla.

Se mantiene la transformación fija `sign(x) * log1p(abs(x))`, seguida de las
máscaras y `log1p(antigüedad_en_días)`. No se ajusta ninguna estadística usando
validación. Un valor ausente ocupa cero y máscara cero. No es una observación
contable igual a cero.

`company-factors.parquet` conserva los cálculos en float64 y la trazabilidad.
`samples.parquet` almacena los vectores en listas de tamaño fijo float32.
El manifiesto identifica las fórmulas, el código, el orden de los conceptos
y la huella del Parquet de factores. La reutilización comprueba ambos Parquet.
Para conservar un experimento se debe elegir un directorio de salida nuevo.

Se procesa un activo cada vez. La entrada y la expansión tienen límites de
filas independientes del tamaño total del corpus. La escritura usa bloques de
hasta 1.024 filas y publicación atómica. Solo se conserva en memoria la parte
de los factores necesaria para las consultas temporales. Los originales no se
modifican. La implementación usa Python y PyArrow. No se ha identificado una
necesidad de kernels C++ o CUDA para estos siete cocientes.

La [comprobación ejecutada](../../reports/data/company-factors.md) separa la
materialización, la cobertura por decisión y los entrenamientos de diagnóstico.
