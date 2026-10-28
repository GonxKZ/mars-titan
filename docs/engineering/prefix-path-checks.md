# Ensayo de optimización de las firmas de archivo

El perfil de B4 localizó trabajo repetido en `PrefixTargetVerifier`: 1.792 comprobaciones de firma y 19.712 llamadas `lstat` en un paso de 64 activos. Se ensayó recorrer las rutas con `os.path.dirname` y `os.lstat`, evitando objetos `Path` temporales y pasando de tres a dos consultas de metadatos al archivo objetivo. El candidato sigue comprobando todos los componentes, relee el objetivo al final y conserva dispositivo, inode, tamaño, mtime y ctime en nanosegundos. No añade una caché de rutas ni hace atómica su resolución.

El candidato no se adopta. Se conserva la implementación de producción y se archivan su [código y pruebas completos](../../reports/engineering/prefix-path-cost-20261027/candidate.patch) como experimento reproducible sobre `093bb69431077a9d6963ab053eed714110196f3c`.

El microbenchmark intercaló ocho pasadas medidas tras un calentamiento, con 2.000 firmas por pasada. En rutas de ocho y quince niveles, el cociente entre tiempos de referencia y candidato fue 1,51 y 1,48. Las firmas coincidieron exactamente. Ese resultado no sustituye una medida de la sesión completa.

La primera comparación de sesiones siguió el orden referencia, candidato, candidato, referencia, con tres repeticiones por modo. La última referencia tardó más del doble que la primera. Se conservan [sus datos](../../reports/engineering/prefix-path-cost-20261027/unrestricted-comparison.json), pero la diferencia agrupada no se atribuye al código.

Antes de repetir se fijó otro [protocolo](../../reports/engineering/prefix-path-cost-20261027/affinity-protocol.json): afinidad del propio proceso a los núcleos físicos 0 y 1, dos hilos, un calentamiento y cinco repeticiones por modo. No se cambió el modo de energía ni se detuvieron aplicaciones. El criterio exigía reducir al menos un 5 % la mediana agrupada en ambos modos, mejorar la referencia vecina en los dos procesos candidatos y limitar al 10 % la deriva entre referencias.

La confirmación usa fixtures CPU FP64 de B2 con 256 activos, con B4 desactivado o con veinte diagnósticos. Conserva las dimensiones compactas y el procedimiento del [benchmark de sesión](../experiments/lesion-protocol-v1.md#medición-de-la-sesión-durable). Se observaron estas medianas:

| Modo | Referencia (ms) | Candidato (ms) | Reducción observada |
| --- | ---: | ---: | ---: |
| Sin B4 | 1.577,8 | 1.462,2 | 7,33 % |
| Veinte diagnósticos | 1.889,1 | 1.818,6 | 3,73 % |

La deriva entre referencias fue del 9,58 % sin B4 y del 4,40 % con diagnósticos. El segundo modo queda por debajo del margen y el segundo proceso candidato tampoco mejora su referencia vecina. Por eso se conserva la referencia. La [comparación completa](../../reports/engineering/prefix-path-cost-20261027/affinity-comparison.json) contiene todas las muestras.

Las predicciones y contadores coinciden por activo y fecha en los checkpoints de los pasos 128 y 130. Se realizaron 40.960 comparaciones de valores sobre 160 checkpoints de ambos ensayos. Son comprobaciones repetidas de los mismos fixtures, no observaciones financieras independientes. Cada ejecución conserva además la comprobación completa de memoria y recuperación de su propio contrato. No se comparan como iguales los identificadores de orígenes distintos.

El candidato aprobó 39 pruebas focalizadas sin omisiones y detectó cuatro mutaciones dirigidas con referencia verde antes y después. La función propuesta cubre 14/14 sentencias y 6/6 ramas, con CRAP 6. El archivo candidato completo cubre 162/184 sentencias y 46/64 ramas, con coverage.py 7.16.1 y Radon 6.0.1. Esa cobertura pertenece al código archivado, no al archivo de producción restaurado. Las pruebas incluyen enlaces al mismo inode, cambios durante la revisión de padres, archivos no regulares y equivalencia exacta de la evidencia temporal con la referencia. No acreditan ausencia de todas las carreras de resolución de rutas.

El patch incluye la implementación y las pruebas del candidato. Se comprobó que reproduce los cambios y que se puede aplicar sobre la referencia mediante `git apply --check`. Los [recibos](../../reports/engineering/prefix-path-cost-20261027/verification.json) identifican las fuentes y mantienen el resultado negativo. No se ejecutaron entrenamientos, corpus real, evaluaciones científicas ni el test final.
