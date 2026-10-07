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
publication_end, max_requests)`. La CLI `python -m
mars_titan.data.china_announcements` recibe las mismas opciones con guiones.
El presupuesto de peticiones es obligatorio y puede ampliarse en otra invocación
con la misma configuración. El cursor distingue una colección parcial de una
colección completada o bloqueada.

## Comprobaciones ejecutadas

El [recibo real](../../reports/data/chinese-announcements-20261007.json)
contiene un piloto completado con 51 anuncios y dos peticiones, de los que tres
pertenecen al censo. La reutilización no realiza peticiones ni reescribe los
archivos. La ventana completa de publicaciones de 2022 y 2023 pasó de 24 a
48 respuestas confirmadas y de 630 a 1.350 anuncios tras reanudarse. Los bytes
y fechas de modificación de las respuestas anteriores permanecieron iguales.
Ese corte todavía no acredita que se haya recorrido toda la ventana.

Pasan 45 pruebas del módulo y cuatro comprobaciones privadas adicionales.
Se detectaron doce mutaciones dirigidas. La cobertura es del 89,90 % de
sentencias y del 75,00 % de ramas. El recibo registra la complejidad y los
límites de la revisión.

El catálogo contiene anuncios, resúmenes y correcciones. Cada enlace PDF se
contrasta con su identificador y fecha. Esta fase no descarga los documentos,
no escoge una revisión por defecto y no admite importes contables. Las
publicaciones de los cierres trimestrales necesitan sus categorías específicas.
`period_coverage_verified=false` y `financial_values_admitted=false` mantienen
explícito ese alcance.
