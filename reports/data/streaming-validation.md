# Preparación acotada y comprobación por bloques

La codificación ya no carga el panel macro completo ni acumula todas las muestras
del activo antes de escribirlas. Consulta grupos Parquet por fecha y confirma las
muestras por bloques en un archivo temporal. El archivo final solo se sustituye
cuando termina correctamente la escritura.

## Límites implementados

| Componente | Límite inicial |
| --- | --- |
| Fuentes para normalizar un activo | 256 MiB en total, antes de decodificar |
| Archivo de noticias | 100.000 registros y 1.048.576 caracteres por registro |
| Deduplicación de fundamentales | 100.000 hechos únicos |
| Partición preparada | 100.000 filas y 64 MiB de buffers Arrow acumulados |
| Grupo macro | 100.000 filas, 64 MiB y decodificación por bloques de 64 filas |
| Caché macro | Dos grupos retenidos, más el grupo que se esté construyendo |
| Identificador y unidad macro | 128 y 256 caracteres respectivamente |
| Bloque de muestras | 256 filas por defecto, configurable entre 1 y 1.024 |

Las unidades e identificadores se inspeccionan como diccionarios antes de expandir
cadenas repetidas. El tamaño Parquet sin compresión no basta para acotar esa
expansión. Las pruebas incluyen un grupo pequeño en disco cuya unidad repetida
expandía varios MiB en memoria.

Estos límites no equivalen al RSS total. Python, Arrow, PyTorch, sus copias y sus
asignadores tienen costes adicionales. Noticias y hechos contables siguen
materializándose por activo dentro de los límites descritos. Si se exceden, la
preparación falla explícitamente. No se presenta esta entrega como conversión
incremental ilimitada de todo el universo multimodal.

## Paridad y recuperación

La [comprobación real](bounded-materialization.json) reconstruyó las 65 muestras
estrictas de MNST y DECK usando grupos de ocho. Las tablas resultantes son
exactamente iguales a las anteriores, incluidas modalidades, fechas, procedencia
y contexto macro. ABM y CSGS conservan cero muestras. La segunda pasada reutilizó
los cuatro activos mediante sus huellas.

La recuperación se confirma por activo. Un fallo dentro de un archivo conserva
su versión anterior y no publica un manifiesto nuevo. Se reutilizan las
representaciones ya confirmadas en la caché, pero se repite la escritura del
activo interrumpido. No se promete recuperación a mitad de un grupo Parquet.

## Medición del acceso macro

Se compararon la lectura anterior y la nueva sobre las mismas 65 decisiones y
el mismo archivo de 888.860 filas. Hubo tres repeticiones por ruta, en procesos
separados y alternando el orden referencia/nueva. Las seis produjeron el mismo
SHA-256 de los vectores calculados.

| Ruta | Mediana del tiempo | Mediana del pico RSS |
| --- | ---: | ---: |
| Panel completo en memoria | 4,667 s | 1.156,26 MiB |
| Consulta por grupos | 1,683 s | 140,45 MiB |

En esta carga, la razón entre medianas es aproximadamente 2,77 y el pico de RAM
se reduce alrededor del 88 %. No es una mejora medida de entrenamiento. La caché
del sistema operativo no estaba controlada y parte de la suite local coincidió
con estos ensayos CPU. Los [registros individuales](../resources/macro-access-comparison.json)
conservan esa limitación, hashes y cifras exactas. No se extrapola al universo
completo ni se atribuye un óptimo general.

## Pruebas

La suite completa pasó con 414 pruebas, incluida CUDA. Se verificaron fronteras
entre grupos, tamaños distintos con filas idénticas, fallos intermedios, límites
previos a la decodificación, cadenas de diccionario y conservación de originales.
La revisión independiente detectó el riesgo de expansión y se corrigió mediante
dos pruebas que fallaron antes del cambio.

En los módulos de bloques y muestras, coverage.py registró 262 de 285 sentencias
y 93 de 118 ramas, un 88,09 % combinado. El escritor atómico tiene complejidad 6
y CRAP 6,002 según Radon 6.0.1, usando cobertura de sentencias por función y
`C² × (1 − cobertura)³ + C`. Tres mutaciones dirigidas fueron detectadas: omitir
el límite de cadenas, omitir la publicación atómica y omitir el límite de filas.

La futura muestra principal sigue pendiente de cobertura. Por ese motivo, #50
mantiene abierto ese criterio aunque el recorrido acotado del piloto esté probado.
