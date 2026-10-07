# Tercer lote de balances chinos

La tercera ampliación añade doce empresas, 69 hechos contables y 2.546 muestras
reales con cuatro modalidades y los 140 indicadores macro. La edición china
resultante reúne 37 empresas y 7.920 muestras. Sigue siendo parcial respecto a
los 810 instrumentos con cuatro fuentes potenciales.

La [evidencia de la ampliación](../../reports/data/chinese-balance-batch03-20261007.json)
conserva por empresa las huellas de documentos, conciliaciones, hechos y muestras.
Los PDF suman 90.441.733 bytes y 2.927 páginas. Se inspeccionaron 76 páginas
pertinentes, sin presentar ese recuento como lectura íntegra de cada informe.

## Selección y contraste documental

Se eligieron los primeros símbolos pendientes con título exacto de informe
completo de 2021 entre dos catálogos parciales ya conservados. Para cada símbolo
se tomó la publicación más temprana observada que cumplía esa regla. Esto no
acredita la primera versión histórica ni convierte la selección en una muestra
representativa. La regla, los doce enlaces y los hashes quedan fijados antes de
codificar las nuevas entradas.

| Empresa | Publicación | Hechos admitidos | Muestras nuevas | Documento |
| --- | --- | ---: | ---: | --- |
| 000009.SZ | 2022-03-31 | 6 | 145 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-03-31/1212752790.PDF) |
| 000021.SZ | 2022-04-21 | 6 | 170 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-21/1213001411.PDF) |
| 000027.SZ | 2022-04-22 | 6 | 211 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-22/1213028283.PDF) |
| 000031.SZ | 2022-04-12 | 6 | 188 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-12/1212886659.PDF) |
| 000059.SZ | 2022-04-13 | 6 | 62 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-13/1212897106.PDF) |
| 000061.SZ | 2022-04-26 | 6 | 387 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-26/1213118045.PDF) |
| 000066.SZ | 2022-04-30 | 6 | 286 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-30/1213249559.PDF) |
| 000088.SZ | 2022-04-09 | 6 | 170 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-09/1212854858.PDF) |
| 000089.SZ | 2022-04-09 | 6 | 316 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-09/1212857408.PDF) |
| 000096.SZ | 2022-04-12 | 6 | 26 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-12/1212882308.PDF) |
| 000301.SZ | 2022-04-19 | 3 | 211 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-19/1212962265.PDF) |
| 000333.SZ | 2022-04-30 | 6 | 374 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-30/1213264815.PDF) |

La revisión contrasta balance consolidado, moneda CNY, unidad, norma CAS,
periodo y patrimonio total incluidos los minoritarios. Las 72 cifras
transcritas originan 69 hechos admitidos y 108 coincidencias con los registros
originales. Once coincidencias son decimales exactas y 97 conservan el mismo
valor binario de 64 bits que el importe publicado convertido a CNY. Las
comparaciones, diferencias y política numérica quedan registradas. No se utiliza
una tolerancia amplia para aceptar importes distintos.

Tres casos requieren distinguir columnas o versiones:

- En 000009, el cierre de 2020 está en la tercera columna. La segunda contiene
  la apertura de 2021 ajustada y no se utiliza como cierre anterior.
- En 000096, el saldo inicial de la tabla se contrasta con el resumen del cierre
  de 2020 y las notas sobre comparativos. La fecha no se deduce solo de la
  posición de una columna.
- En 000333, las cifras se presentan en miles de CNY. Se aplica el multiplicador
  1.000 y se usan las columnas consolidadas, separadas de las de la matriz. La
  reclasificación comparativa del transporte afecta a resultados, sin cambiar
  los tres totales de balance transcritos.

Los tres comparativos reexpresados de 2020 de 000301 no coinciden con el conjunto
original y quedan excluidos. Se admiten sus tres totales de 2021. La publicación
de abril de 2022 se conserva para los hechos que se utilizan. El materializador
aplica la regla conservadora de disponibilidad del calendario CN, sin atribuir
al documento una hora de publicación que no consta.

El primer intento de descarga de 000301 alcanzó 11.534.336 bytes y agotó su
límite local de 45 segundos. Se conserva el fallo. Un nuevo intento completo,
con máximo de 180 segundos, descargó los 32.356.879 bytes del PDF. No se unieron
fragmentos sin una identidad de versión suficiente. 000009, pendiente en un
lote anterior por otro timeout, se descargó y revisó en esta edición.

Una revisión independiente volvió a inspeccionar las tablas y notas,
extrajo de nuevo 28 páginas desde los PDF y comprobó 196 archivos. Contrastó
38 registros originales y 114 comparaciones de campo, incluidos los importes
rechazados. Las fuentes, recibos y hechos permanecieron intactos.

## Codificación y particiones

MiniLM y ResNet18 conservan exactamente la identidad de los codificadores usados
en US. La ejecución utilizó CUDA y comprobó las 2.546 nuevas fechas de decisión,
sus modalidades y las 140 máscaras macro. Los vectores contables coinciden con
una referencia Decimal de 50 dígitos transformada a `float32`. Repetir la
preparación reutiliza las salidas confirmadas sin nueva inferencia.

La ventana exclusiva de GPU duró 134,38 segundos. El proceso de codificación y
verificación tardó 118,84 segundos y reservó como máximo 612.368.384 bytes de
VRAM. La campaña estadounidense se reanudó al terminar. Son medidas de una
comprobación funcional, sin afirmar aceleración ni uso máximo del hardware.

La división anual contiene 2.737 filas de entrenamiento y 5.146 de validación.
Las restantes muestras conservan sus exclusiones por disponibilidad de la
etiqueta o por cruce del corte anual. Las diez ventanas posteriores mantienen
el protocolo US con calendario CN y recuperan expresamente las etiquetas
permitidas por sus nuevos cortes.

Las cuarenta particiones se recorrieron y verificaron. Utilizan 7.855 muestras
distintas y 4.182 filas de evaluación sin repeticiones entre meses. La mayor
ventana contiene 6.180 filas de entrenamiento, 813 de validación, 494 de
calibración y 343 de evaluación. La unión conserva los bytes de las ediciones
anteriores y su recuperación no reescribe los manifiestos confirmados.

Pasan 206 pruebas CPU relacionadas con fuentes, hechos, preparación, muestras,
historia y unión. La comprobación CUDA de este lote prepara datos reales. No es
un entrenamiento de comparadores chinos ni una evaluación de su capacidad
predictiva. El test final de 2024 sigue cerrado.

La edición [US+CN anterior](../engineering/joint-market-campaigns.md) conserva
su población de 25 empresas chinas. Esta entrega amplía la edición CN a 37,
compatible con el contrato común ya implementado. La campaña conjunta requiere
congelar su población actualizada antes de ejecutarla. Quedan por revisar las
publicaciones y versiones de los demás instrumentos, incluidos cierres
posteriores a los anuales utilizados aquí.
