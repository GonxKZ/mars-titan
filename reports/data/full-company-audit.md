# Cobertura contable del universo con cuatro fuentes

La auditoría del 22 de septiembre de 2026 recorrió los 7.917 archivos contables
de los 2.639 instrumentos US que tienen las cuatro fuentes en el inventario
corregido. No aplicó el filtro de sector del panel técnico. Verificó y leyó
88.987.082.255 bytes sin modificar originales.

Hay 2.238 instrumentos con algún hecho admisible publicado antes de 2024.
En 390 no se emitió ningún hecho y en 11 solo aparecen publicaciones desde 2024.
Tener archivos de balance, flujos y patrimonio no demuestra que contengan datos.

Las 158.623.107 ocurrencias anteriores a 2024 se descomponen en 155.632.281
duplicadas, 2.982.123 hechos únicos admitidos y 8.703 registros con periodo
inválido o cierre posterior a la publicación. No se detectaron claves con
valores contradictorios. La deduplicación se hizo por espacio de nombres,
concepto, unidad, inicio y cierre del periodo, `filed` y `accn`.

## Conceptos y unidades

El catálogo conserva 27 nombres contables y 93 pares concepto-unidad. De ellos,
92 aparecen en desarrollo. `AccountsPayableCurrent:CNY` solo aparece en 2023
y se mantiene separado de la selección de variables de desarrollo. Las
unidades monetarias, por acción, acciones y magnitudes adimensionales no se
mezclan ni se convierten por coincidencia de nombre.

No se encontró `CashAndCashEquivalentsAtCarryingValue` antes de 2024. Sí hay
variaciones de caja y otros flujos. Ninguno sustituye automáticamente ese saldo.
Ocho conceptos tienen registros con y sin `start`, incluidos flujos conocidos.
Las etiquetas estructurales `stock_without_start` y `flow_with_start` describen
la forma del registro, no una clasificación contable validada por sí misma.

El inventario de conceptos no abre automáticamente nuevos canales de entrenamiento.
La representación implementada admite ocho conceptos de balance y siete ratios
en USD. Los ratios compatibles en CAD, CNY e ILS quedan documentados como
extensiones pendientes. No se presentan como variables ya calculadas para los
modelos ni se sustituyen importes en USD por otras monedas.

## Grupos compatibles en USD

| Factor | Activos hasta 2022 | Grupos hasta 2022 | Activos en 2023 | Grupos en 2023 |
| --- | ---: | ---: | ---: | ---: |
| Liquidez corriente | 1.673 | 137.160 | 1.651 | 13.569 |
| Fondo de maniobra / activo | 1.668 | 136.596 | 1.651 | 13.569 |
| Pasivo / activo | 1.890 | 141.687 | 1.859 | 15.188 |
| Patrimonio / activo | 2.149 | 172.887 | 2.092 | 17.338 |
| Cuentas por cobrar / activo | 1.369 | 98.315 | 1.269 | 9.812 |
| Cuentas por pagar / activo | 1.487 | 112.392 | 1.424 | 11.289 |
| Pasivo / patrimonio positivo | 1.805 | 124.431 | 1.697 | 13.381 |

Los siete ratios coinciden en una misma presentación en 914 activos de
desarrollo, con 48.408 grupos, y en 767 activos de 2023, con 5.579 grupos.
Un grupo exige activo, unidad, cierre, `filed` y `accn` comunes, componentes sin
periodo inicial y denominador positivo. El patrimonio negativo se conserva
cuando aparece en el numerador.

Estas cifras son grupos contables y se particionan por `filed`. No son muestras
diarias de entrenamiento. Todavía hay que aplicar disponibilidad conservadora,
ventanas de precios, verificación de noticias, gráficos, macro y maduración de
etiquetas. Una publicación al final del año puede quedar disponible en el
siguiente. Los hechos de 2024 en adelante se contaron como descartados sin usar
sus conceptos o valores para seleccionar variables.

## Coste y comprobación

La pasada completa tardó 1.983,55 segundos desde la entrada al programa,
33,06 minutos, sin incluir arranque e importaciones. El tiempo acumulado de
procesamiento por activo fue 1.933,69 segundos. El RSS máximo fue 129,04 MiB.
Se utilizó un proceso con Python 3.12.14 e ijson 3.5.1, backend `yajl2_c`, sin GPU.

Los totales se conciliaron con los 2.639 checkpoints por activo y los CSV.
Se comprobaron las huellas de 2.650 artefactos. La conciliación se repitió sobre
los resúmenes confirmados. El recorrido es recuperable por activo y no
acumula todos los hechos de todas las empresas en memoria.

La primera ejecución del script encontró un contador no inicializado ante un
archivo sin hechos. El caso se corrigió y se probó antes de la pasada completa.
Los intentos anteriores quedaron conservados por separado. Sus tiempos no se
incluyen en los 33,06 minutos de la pasada final.

La presencia y compatibilidad contable ya están cuantificadas. La campaña de
entrenamiento completa sigue pendiente de la intersección multimodal verificada
y de materializar los factores para todos los activos que resulten admitidos.

El [resumen numérico](company-corpus-audit.json), el
[catálogo de conceptos](company-concept-catalog.csv) y el
[catálogo de ratios](company-ratio-catalog.csv) permiten inspeccionar los
recuentos por periodo, unidad y concepto. Los CSV públicos normalizan los saltos
de línea a LF, sin cambiar valores. Los recibos por activo y sus fuentes se
conservan localmente. La huella del inventario de origen es
`faabf457d3ce7b308ec6c6887fac4e6ee68847a1b28aeffebba5a8ddf40517e3`.
