# Cobertura al terminar la pasada editorial

La instantánea que alimenta la edición de 470 muestras conserva 4.469.917
registros textuales. La pasada recorrió los candidatos pendientes del índice,
pero no completó su verificación. Una limitación temporal de consulta deja el
registro para otro intento, no lo convierte en una noticia verificada.

| Estado en la instantánea | Registros |
| --- | ---: |
| Artículo completo admitido | 125 |
| Rechazado | 11 |
| No verificable con la captura disponible | 1.793 |
| Pendiente de reintento | 269.803 |
| Necesita procedencia adicional | 3.218.163 |
| Reservado por el corte temporal | 980.008 |
| Sin contenido utilizable | 14 |
| Total | 4.469.917 |

Los 125 registros admitidos pertenecen a 93 instrumentos. Al exigir las otras
fuentes y su intersección temporal se preparan 71 activos estadounidenses y
66 conservan muestras materializadas. No se aplica un máximo de empresas ni
un filtro por sector actual.

| Mercado | Candidatos del inventario | Con las cuatro fuentes | Preparados con noticia completa | Con muestras materializadas |
| --- | ---: | ---: | ---: | ---: |
| Estados Unidos | 4.784 | 2.639 | 71 | 66 |
| China | 892 | 810 | 0 | 0 |

En Estados Unidos, 2.568 instrumentos con las cuatro fuentes todavía no tienen
una noticia completa admitida en esta instantánea. Los 810 candidatos chinos
con cuatro fuentes tampoco cumplen ese requisito. No se han sustituido sus
resúmenes por cuerpos inventados ni se les han asignado fechas contables
de publicación a partir del cierre de periodo.

## Qué impide ampliar el entrenamiento

De los reintentos, 268.895 corresponden al plazo local entre consultas, 901 a
respuestas 404 y siete a respuestas 429. El primer grupo no acredita un fallo
del editor ni falsedad del artículo. Requiere continuar respetando el plazo.
Tampoco una respuesta 404 basta para rechazar el contenido del dataset.

La categoría de procedencia reúne casos distintos. En China contiene 1.053.678
registros del tramo no reservado. El formato original observado conserva fecha,
título y resumen, sin una URL editorial estructurada. En Estados Unidos hay
2.156.348 registros sin un editor resuelto por el adaptador actual y 8.137 que
ya requerían procedencia al indexarlos. No es correcto describir todo ese grupo
como noticias corruptas.

Una inspección acotada de 100 registros, comprobados contra sus hashes, distinguió
40 resúmenes chinos, 20 registros estadounidenses sin título y 40 registros de
Nasdaq cuyo editor todavía no resuelve el adaptador. Esa inspección ayuda a
separar problemas de formato y cobertura. No estima sus proporciones sobre todo
el corpus ni acredita una verificación editorial de esos 100 registros.

La siguiente ampliación depende de recuperar fuentes y extender los adaptadores
con capturas contrastadas. Aumentar la RAM, utilizar toda la VRAM o añadir C++ no
resuelve esa falta de evidencia. La cola conserva los pendientes y puede
continuarse sin modificar las instantáneas utilizadas para entrenar.

## Trazabilidad

La instantánea SQLite y su SHA-256 permanecen en
`data/processed/training-edition-post-scan-20260923/`. Los manifiestos de cobertura,
muestras y etiquetas identifican sus fuentes. Los datos y capturas no se publican
en Git. Los [resultados de la edición](../baselines/post-scan-reference-study.md)
utilizan exclusivamente esa población congelada.
