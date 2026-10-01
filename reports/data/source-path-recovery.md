# Recuperación de procedencia en nombres de archivo

La preparación inicial del universo recorrió 5.676 candidatos. En Estados
Unidos preparó 2.639 activos y registró 2.145 sin todas las fuentes. En China,
808 activos fallaron al escribir su procedencia en SQLite. El nombre original
de sus archivos contenía bytes no válidos como texto UTF-8. El fallo no
demostraba corrupción del cuerpo de las noticias.

La representación de `source_file` ahora usa escape porcentual reversible de
los bytes originales. La versión del recibo y su marcador de codificación
impiden confundirla con la convención anterior. Los archivos originales y las
huellas de su contenido no cambian.

La prueba real de recuperación de `000001.SZ` preparó 4.366 precios y 1.487
noticias, sin errores de proceso. Recorrió 3.616 registros de texto y excluyó
1.260 duplicados y 869 registros reservados por fecha. No obtuvo hechos
contables con publicación acreditada, por lo que no demuestra que el activo
ya sea entrenable con cuatro modalidades.

El recorrido de prueba se detuvo expresamente después de ese activo. No se
presenta como comprobación de los otros 891 candidatos chinos. Se conserva la
edición anterior con errores y la recuperación requiere una nueva salida.

Las regresiones incluyen nombres con bytes inválidos y porcentajes literales,
admisiones y exclusiones, recuperación de la ruta exacta, conservación de los
bytes de origen y continuación después de un cuerpo Unicode inválido. La
prueba del cuerpo inválido sigue rechazándolo, sin reparar ni inventar texto.
