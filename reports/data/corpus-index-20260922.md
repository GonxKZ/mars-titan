# Índice textual completo de ambos mercados

El recorrido del 22 de septiembre de 2026 confirma los 5.586 archivos de noticias
del inventario y conserva 4.469.917 registros. Comprueba 14.382.105.211 bytes contra
sus hashes, sin modificar los originales. No selecciona un panel ni filtra por
sector. El [recibo del índice](corpus-index-20260922.json) conserva los conteos y
la identidad de la implementación.

| Mercado | Instrumentos con alguna fuente | Con las cuatro fuentes | Registros textuales | Registros de instrumentos con cuatro fuentes |
| --- | ---: | ---: | ---: | ---: |
| Estados Unidos | 4.784 | 2.639 | 3.049.555 | 2.021.684 |
| China | 892 | 810 | 1.420.362 | 1.377.245 |

El primer recuento es la unión de identidades del inventario. No significa que
todos tengan precios ni que sean empresas cotizadas distintas y verificadas.
Las 4.213 series de precios estadounidenses son otra población. Las fuentes
incompletas se conservan para explicar cobertura y permitir su reconciliación.

## Estados de los registros

Hay 2.428.080 registros pendientes de contraste editorial, 1.061.815 que necesitan
procedencia adicional, 980.008 posteriores al corte declarado y 14 sin contenido
utilizable. Los cuatro recuentos suman el total. Ninguno recibe una admisión
editorial por tener campos presentes.

El corte de esta clasificación es el 31 de diciembre de 2023 y utiliza la fecha
declarada. No acredita su zona ni disponibilidad. Los datos reservados forman
parte del inventario, no del entrenamiento o de la selección de modelos.

El censo chino y el estadounidense conservan las mismas reglas de integridad,
pero no se confunden sus formatos. Los resúmenes chinos sin URL permanecen
pendientes de procedencia. Las fechas de publicación contable y el factor de
mercado chino siguen siendo requisitos para construir sus muestras supervisadas.

## Recuperación e integridad

La segunda ejecución reutiliza los 5.586 archivos confirmados después de volver
a contrastar sus hashes. Conserva exactamente los recuentos por mercado y estado,
la identidad del catálogo y el tamaño de la base. No inserta de nuevo los
registros. El [recibo de continuación](corpus-index-resume-20260922.json) identifica
esta ejecución por separado.

La [comprobación estructural independiente](corpus-index-verification-20260922.json)
recorre todas las posiciones y ordinales, reconcilia las longitudes y comprueba
la integridad SQLite y sus claves foráneas. También relee 11.151 registros en los
extremos de los archivos. La [comprobación completa de registros](corpus-record-integrity-20260922.json)
contrasta después los 4.469.917 hashes de bytes y las 4.469.917 identidades textuales
con el original. Tardó 33,47 segundos. No valida la veracidad editorial.

## Recursos medidos

| Ejecución | Proceso completo | Tiempo interno del indexador | RSS máximo |
| --- | ---: | ---: | ---: |
| Creación del índice | 334,78 s | 312,20 s | 183,35 MiB |
| Continuación y verificación de fuentes | 29,84 s | 7,79 s | 182,81 MiB |

El tiempo interno empieza después de construir y comprobar los metadatos del
catálogo. No equivale al tiempo completo de la orden. Los registros externos de
[creación](../resources/corpus-index-process-time.txt) y
[continuación](../resources/corpus-index-resume-process-time.txt) conservan ambas
ejecuciones. La primera coincidió con pruebas locales y la segunda con una
comprobación de integridad. La caché del sistema no se vació. No son mediciones
aisladas de rendimiento máximo ni una comparación de aceleración entre motores.

La base ocupa 2.201.612.288 bytes y guarda metadatos, índices y localizadores.
No contiene los cuerpos originales. Esa diferencia de tamaño no es una medida
de compresión íntegra del dataset y la base sigue necesitando sus fuentes.
El código usa SQLite y lecturas acotadas en CPU. No se han añadido kernels propios.

La [verificación del código](../resources/corpus-catalog-quality.json) registra
832 pruebas correctas sin omisiones y ocho mutaciones dirigidas detectadas.
La revisión independiente localizó la pérdida de recibos de fuentes con error.
Se corrigió con dos regresiones antes de integrar. Cobertura y CRAP se conservan
con sus convenciones, sin presentarlos como garantía de ausencia de defectos.

El índice completo es una entrada para el contraste de noticias y la preparación
por bloques. No es un Parquet supervisado ni acredita nuevos entrenamientos.
Los brazos US, CN y US+CN siguen pendientes de completar sus datos admisibles,
la lectura supervisada y la reanudación operativa de las referencias.
