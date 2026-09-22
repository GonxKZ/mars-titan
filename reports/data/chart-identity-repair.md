# Corrección de identidades en el inventario de imágenes

Se ha corregido la clasificación de gráficos en rutas con distinta profundidad.
La colección identifica el mercado, y la carpeta inmediata y el prefijo del PNG
deben coincidir en el activo. Las discrepancias y los formatos desconocidos quedan
rechazados. La documentación Markdown de la carpeta sigue siendo metadato.

La clasificación tiene una versión independiente de la inspección de contenido.
Si tamaño, mtime y ctime no cambian, se conservan hash, cabecera y fecha de
inspección anteriores. Un rechazo semántico tampoco borra esa evidencia. La
instantánea de contenido no debe cambiar por corregir una etiqueta de instrumento.

## Comprobación sobre el corpus

El [resultado verificado](source-reclassification-verified.json) procede de una
copia separada del inventario. El inventario original y los archivos fuente se
conservan. La revisión nueva está en
`data/interim/source-inventory-v3-verified-20260922.sqlite`, fuera de Git.

- 216.453 registros reclasificados, con cero errores.
- 168.886 identidades corregidas, de 195.382 gráficos comprobados.
- 2.639 identidades US y 810 CN con las cuatro fuentes presentes.
- Cero cambios en huellas, cabeceras, fechas de inspección, tamaños o marcas físicas frente al inventario anterior.
- Instantánea de contenido `f206a659be9e19f2cd6913488843c480c06808ce089c19cef95cd62d672c1619`, igual a la anterior.
- 8,864 segundos y 64,590 MiB de RSS máximo en la pasada registrada.

No se recalcularon las huellas de los 116 GB. Se reutilizó la evidencia de contenido
cuando las marcas de archivo coincidían. El registro incorpora hashes del código
y de la base resultante. `PRAGMA quick_check` devolvió `ok`.

## Fallos encontrados y pruebas

El [primer intento](source-reclassification.json) detectó incorrectamente
`image/image.md` como un gráfico. Se añadió una regresión y se conservó como
documentación. La [siguiente pasada](source-reclassification-final.json) terminó
sin errores en los datos reales.

La revisión independiente detectó después que rechazar una clasificación antigua
podía borrar su hash aunque los bytes no hubieran cambiado. La nueva prueba
reprodujo ese fallo y ahora conserva huella, cabecera y fecha de inspección, incluso
al repetir la clasificación rechazada. También se corrigió el recuento de contenidos
duplicados para no confundir errores semánticos con archivos sin hash conocido.

Las 22 pruebas del inventario pasan. La suite completa terminó con 466 pruebas
correctas, sin omisiones, con CUDA y los codificadores reales. Una mutación que
elimina la igualdad entre carpeta y prefijo fue detectada por dos pruebas.
La [evidencia de calidad](chart-identity-quality.json) recoge cobertura, complejidad
y CRAP con sus convenciones.

La corrección no acredita que una imagen original estuviera disponible en una
fecha de predicción ni que existan noticias o fundamentales admisibles para esa
fecha. Los gráficos de entrenamiento siguen regenerándose con precios pasados.
