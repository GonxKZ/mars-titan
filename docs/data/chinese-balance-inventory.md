# Inventario de balances chinos para reconciliación

`china_inventory` recorre el censo preparado y localiza los registros de balance
de los cierres solicitados. Contrasta los hashes del censo, de cada preparación
y de los originales. No convierte el cierre de un periodo en fecha de anuncio
ni atribuye moneda, escala, norma contable o consolidación cuando faltan.

La API es `inventory_chinese_balances(manifest, output, *, periods=PERIODS)`.
La ejecución `PYTHONPATH=src uv run --no-sync python -m
mars_titan.data.china_inventory` recibe `--manifest`, `--output`
y `--period` repetible. Por defecto selecciona el 31 de diciembre de 2021 y de
2022. Permite hasta 128 fechas ISO únicas y ordenadas entre 1990 y 2023. Son
fechas de cierre, no una clasificación de informes anuales auditados.

La primera pasada comprueba el JSON completo y conserva los ordinales y las
fechas de selección. La segunda materializa únicamente los registros
seleccionados. Una publicación declarada desde 2024, o una publicación no vacía
que no puede interpretarse, queda sin inspección de sus campos contables. Las
fechas de cierre mal formadas producen un error del activo. Los errores y las
exclusiones conservan sus motivos y no se convierten en hechos admitidos.

`inventory.json` contiene el censo, fuentes, estados y rangos `queue_rows` que
enlazan con `reconciliation-queue.parquet`. Los índices empiezan en cero y el
extremo `stop` queda fuera del rango. Los detalles por cierre viven solo en el
Parquet: ordinales, huellas, campos, metadatos declarados y motivos pendientes.
`report.json` resume la edición y sus medidas. El directorio se publica de una
vez tras revalidar fuentes y archivos. Un destino existente se rechaza.

Los límites son 1.024 activos, 64 MiB por fuente, 100.000 registros por fuente,
4.096 registros seleccionados por fuente y 100.000 seleccionados en total.
Cada registro admite hasta 512 campos. Las cifras se leen como `Decimal`.
Las huellas completas distinguen la igualdad de todos los campos y la igualdad
al omitir `update_flag`. Esa marca no acredita por sí sola qué revisión estaba
disponible en una fecha histórica. Las grafías `.SS` del censo y `.SH` del
original se conservan de forma explícita.

## Recorrido real

La [verificación](../../reports/data/chinese-balance-inventory-20261007.json)
concilia 892 candidatos, con 810 preparados y 82 sin las fuentes requeridas.
Los dos cierres predeterminados generan 1.620 trabajos y 1.807 registros.
En 187 cierres hay dos registros, de los que 177 pares solo difieren en
`update_flag`. Los otros diez necesitan contrastar también las demás diferencias.

Una segunda selección incluye ocho cierres, desde diciembre de 2021 hasta
septiembre de 2023, con todos los trimestres de 2022 y los tres primeros de 2023.
Genera 6.480 trabajos y 7.397 registros. Las 810 empresas tienen registros en
cada cierre. Ninguno declara publicación, moneda, escala o contexto contable
en los campos inventariados. La cola organiza el trabajo pendiente y mantiene
`temporal_admission_granted=false` y `training_ready=false`.

## Perfilado y tamaño

El perfil inicial registró 410.799 llamadas de cálculo de huellas, muchas de
ellas utilizadas solo para contar valores numéricos distintos. Esa comparación
utiliza ahora conjuntos de `Decimal`, sin convertir las cifras a float ni
modificar las huellas completas de los registros. El segundo perfil registra
14.794 llamadas de huellas. El recorrido de metadatos sigue siendo el mayor
coste observado.

Tres parejas de ejecuciones completas, alternando el orden, reducen la mediana
de 15,156 a 13,436 segundos, un 11,35 %. El JSON pasa de 19.790.766 a 793.536
bytes al retirar los detalles duplicados. La cola de 634.416 bytes es idéntica
en las seis ejecuciones. La mediana de RSS baja de 233.644 a 219.624 KiB.
Se usó una CPU y un límite de 1 GiB, con la campaña comparativa activa. Las
fuentes ya se habían leído y no se vació la caché del sistema. Estas medidas
corresponden al inventario y no indican aceleración del entrenamiento.

Pasan 73 pruebas y once mutaciones dirigidas. La cobertura es del 95,37 % de
sentencias y del 90,12 % de ramas. El recibo incluye los perfiles, repeticiones,
versiones, complejidad y convención de CRAP. La revisión focal comprueba la
selección temporal, los rangos de filas y diferencias decimales que desaparecerían
al convertir a binary64.
