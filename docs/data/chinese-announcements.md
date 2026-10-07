# Catálogo recuperable de anuncios contables

`china_announcements` consulta los metadatos públicos de
[CNINFO](https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search)
para localizar documentos del censo. Recibe la cola Parquet del
[inventario de balances](chinese-balance-inventory.md), un destino y una ventana
explícita de publicaciones anterior a 2024. La categoría inicial es
`category_ndbg_szsh`, correspondiente a anuncios anuales. Se conservan también
los registros ajenos al censo, con su clasificación.

La consulta utiliza páginas de treinta registros y un máximo de cien páginas
por intervalo. Cuando el total declarado supera 3.000, divide el intervalo de
fechas. Los hijos deben conciliar con el total y los anuncios ya observados del
padre. Un día que exceda el límite detiene el catálogo. Los totales, las fechas,
los identificadores y `hasMore` se comprueban en cada página.

No se utiliza `totalpages` como número fiable de páginas. La comprobación real
de un intervalo con 31 anuncios devolvió `totalpages=1` en ambas respuestas,
con treinta registros en la primera y uno en la segunda. Se usa el techo del
cociente entre el total declarado y el tamaño de página y se concilia la
población recibida.

Cada respuesta conserva cuerpo, petición y recibo con sus hashes. Un cursor
atómico enlaza las respuestas confirmadas. Al reanudar se reproduce ese historial
y se verifica el siguiente intervalo y página. Un corte después de guardar la
respuesta y antes de confirmar el cursor permite recuperarla sin repetir la
petición. Un bloqueo de escritor impide recoger dos veces el mismo catálogo
en paralelo.

Las peticiones se separan al menos dos segundos. El transporte tiene treinta
segundos de espera y conserva hasta 1 MiB por respuesta. Se registra cuántos
bytes se leyeron y se retuvieron, y se concilian con `Content-Length`, incluidos
sus espacios exteriores. La recuperación vuelve a comprobar esa relación.
Cambiar un indicador de truncamiento no convierte una respuesta incompleta en
completa. Una respuesta 403 o 429 detiene la recogida sin reintentos automáticos.

La API es `collect_chinese_announcements(queue, output, *, publication_start,
publication_end, max_requests)`. La ejecución local utiliza `PYTHONPATH=src uv
run --no-sync python -m mars_titan.data.china_announcements`, con las mismas
opciones expresadas con guiones.
El presupuesto de peticiones es obligatorio y puede ampliarse en otra invocación
con la misma configuración. El cursor distingue una colección parcial de una
colección completada o bloqueada.

## Consulta por emisor

La opción por emisor añade cuatro argumentos, que deben suministrarse juntos:

- `--issuer-symbol`, con un símbolo de la cola, como `000066.SZ`.
- `--issuer-receipt`, con el `receipt.json` de una respuesta pública conservada,
  junto a su `body.json`.
- `--issuer-receipt-sha256`, con la huella esperada de ese recibo.
- `--issuer-announcement-id`, con el identificador del anuncio elegido dentro de
  la respuesta.

La API usa los mismos nombres con guiones bajos. El código y el `orgId` se
obtienen del anuncio y se contrastan con el símbolo. No se recibe un `orgId`
libre ni se construye uno a partir del código. La evidencia debe proceder de una
captura pública completa con HTTP 200, con la petición anual original, las
fechas y el vínculo al documento conservados. Una captura con transporte
inyectado no sirve como evidencia pública.

El filtro `stock=código,orgId` forma parte de cada petición y de su recibo. Cada
página, incluida su recuperación, debe contener únicamente ese código, mercado
y `orgId`. Se mantienen los controles de población, duplicados, intervalos y
`hasMore`. Si la ventana incluye la publicación del anuncio elegido, ese anuncio
debe aparecer antes de completar la recogida. Una respuesta vacía o la pérdida
del anuncio conocido bloquean ese ámbito. Fuera de esa ventana, una respuesta
vacía solo acredita que el proveedor no devolvió resultados.

La configuración fija también los metadatos normalizados de ese anuncio. Cuando
su ID aparece, la fecha, el instante de publicación, el enlace PDF, el título y
los demás campos deben coincidir con la evidencia. Esta comprobación se aplica
al consumir y al recuperar respuestas, incluso si la fecha original queda fuera
de la ventana solicitada. No basta con conservar el ID de un anuncio distinto.

Estas colecciones usan configuración e informe de versión 2 y el ámbito
`issuer_annual_category_only`. Conservan las rutas y hashes de la evidencia, que
se vuelven a comprobar antes de pedir datos, al recuperar y antes de confirmar.
La espera mínima es de cinco segundos. `--max-requests` sigue siendo el máximo
de peticiones nuevas por invocación de esa colección, incluidas las páginas y
subdivisiones de fechas. No es un presupuesto para recorrer todo el censo.

Cada emisor necesita una salida separada. Las capturas generales permanecen en
versión 1, con su comportamiento y recuperación originales. Como la identidad
incluye hashes del código, las capturas antiguas se reanudan desde su runtime
congelado. No se migran ni se ignoran sus controles para abrirlas con otro código.
Tampoco se transforma una colección general existente en una colección por emisor.

`completed` indica que se han conciliado las respuestas del ámbito solicitado.
`issuer_history_complete=false`, `period_coverage_verified=false` y
`financial_values_admitted=false` evitan interpretar ese estado como historia
completa, cobertura de todos los cierres o admisión de cifras.

El [piloto por emisor](../../reports/data/chinese-issuer-announcements-20261007.json)
utilizó el código y `orgId` de un anuncio oficial de `000066.SZ`. Tres peticiones
devolvieron cuatro anuncios únicos. La ventana 2022–2023 dio cuatro y cada año
por separado dio dos, con unión exacta. Todas las filas pertenecían al emisor y
los dos anuncios ya conservados mantuvieron sus metadatos. `totalpages` devolvió
cero pese a contener filas y no se utiliza para acreditar la población.
Esta prueba justifica la opción, sin demostrar estabilidad general ni una
aceleración del catálogo completo. La integración de la opción reproduce esos
cuerpos guardados sin red y recupera sus salidas sin repetir peticiones.

## Comprobaciones ejecutadas

El [recibo real](../../reports/data/chinese-announcements-20261007.json)
contiene un piloto completado con 51 anuncios y dos peticiones, de los que tres
pertenecen al censo. La reutilización no realiza peticiones ni reescribe los
archivos. La ventana completa de publicaciones de 2022 y 2023 pasó de 24 a
48 respuestas confirmadas y de 630 a 1.350 anuncios tras reanudarse. Los bytes
y fechas de modificación de las respuestas anteriores permanecieron iguales.
Ese corte todavía no acredita que se haya recorrido toda la ventana.

La recogida posterior se detuvo con HTTP 504 tras conservar 1.710 anuncios.
Un segundo intento reutilizó sus sesenta respuestas correctas, sin modificar
el primero, y aumentó la espera mínima a cinco segundos. Alcanzó 1.860 anuncios
y volvió a detenerse ante HTTP 504. Las respuestas y ambos fallos permanecen
conservados. La colección completa sigue pendiente y no se ha identificado
la causa interna del error del proveedor.

Pasan 45 pruebas del módulo y cuatro comprobaciones privadas adicionales.
Se detectaron doce mutaciones dirigidas. La cobertura es del 89,90 % de
sentencias y del 75,00 % de ramas. El recibo registra la complejidad y los
límites de la revisión.

## Avance del 7 de octubre de 2026

Un catálogo nuevo consulta publicaciones del 3 de abril de 2022 al 31 de
diciembre de 2023. Las tres invocaciones usan el mismo código, configuración y
cursor recuperable. El [recibo de progreso](../../reports/data/chinese-catalogue-progress-20261007.json)
conserva sus métricas y hashes. Los intentos anteriores con HTTP 504 mantienen
su evidencia separada y no se suman a estos recuentos.

| Invocación | HTTP 200 nuevos | Anuncios nuevos | Anuncios acumulados | Bytes recibidos | Tiempo (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 | 64 | 1.770 | 1.770 | 1.222.139 | 408,29 |
| 02 | 64 | 1.835 | 3.605 | 1.204.316 | 412,32 |
| 03 | 64 | 1.903 | 5.508 | 1.205.747 | 411,01 |

Cada invocación agotó su presupuesto de 64 peticiones, sin errores HTTP ni
reintentos automáticos. Se aplicó una espera mínima de cinco segundos. El menor
intervalo observado entre inicios fue de 5,493 segundos y el mayor pico RSS del
proceso fue de 179,39 MiB. Los tiempos incluyen las pausas y la verificación
final de fuentes, pero no la preparación ni el inventario posterior.

Al reanudar se verificaron las 64 y las 128 respuestas anteriores,
respectivamente, sin repetirlas ni modificar su contenido. Las 1.770 y las
3.605 filas previas también permanecieron idénticas. Los anuncios acumulados
proceden de páginas confirmadas. Las siete respuestas que provocaron una
división de intervalos se conservan como sondas, sin volver a sumar sus anuncios.

Están conciliados los intervalos del 3 al 22 de abril de 2022, con 2.615
anuncios, y del 23 al 27, con 2.083. El siguiente cursor es la página 28 del
tramo del 28 de abril al 2 de mayo, con 810 de sus 2.630 registros declarados
recogidos. Quedan seis tareas de intervalo y página, detalladas en el recibo.
Las filas incorporadas contienen publicaciones del 6 al 30 de abril de 2022.
Ese rango observado no acredita haber completado la ventana solicitada ni los
intervalos pendientes.

De los 5.508 anuncios, 740 pertenecen a 366 instrumentos del censo. La
clasificación por título identifica 359 posibles informes completos de 2021 o
2022, correspondientes a 351 empresas fuera de las
[25 con edición contable admitida](chinese-balance-batch02.md). Son 103 empresas
candidatas más en la segunda invocación y otras 139 en la tercera. Los resúmenes,
las versiones inglesas y los títulos de otro tipo se conservan por separado.
No se han descargado esos PDF ni contrastado sus cifras. Este avance no añade
balances admitidos ni amplía por sí solo el corpus de entrenamiento.

El catálogo contiene anuncios, resúmenes y correcciones. Cada enlace PDF se
contrasta con su identificador y fecha. Esta fase no descarga los documentos,
no escoge una revisión por defecto y no admite importes contables. Las
publicaciones de los cierres trimestrales necesitan sus categorías específicas.
`period_coverage_verified=false` y `financial_values_admitted=false` mantienen
explícito ese alcance.
