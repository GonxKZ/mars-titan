# Ampliación de balances chinos del 7 de octubre

La revisión añade once empresas y 2.503 muestras reales a la
[edición de Ping An y Vanke](chinese-corpus.md). La unión contiene 3.224 muestras
de trece empresas, con precios, noticias, gráficos, fundamentales y las
140 posiciones macro completas. Es una ampliación de datos comprobada en CUDA.
No se han entrenado nuevos modelos con ella.

Se seleccionaron doce informes completos de 2021 entre los anuncios ya
conservados del catálogo parcial. Once descargas terminaron, con 61.863.494 bytes
y 3.051 páginas. La de 000009.SZ agotó 45 segundos y conserva el parcial y el
fallo, sin admitir cifras ni reintentar automáticamente. La selección no
acredita la primera versión histórica de cada informe ni una muestra
representativa del mercado.

## Hechos admitidos y exclusiones

Las páginas revisadas identifican balance consolidado, fechas de las columnas,
moneda de presentación CNY y normas CAS. Se contrastan activos, pasivos y
patrimonio total con minoritarios. Son tres conceptos contables, no todos los
campos del balance. El materializador conserva 60 hechos con 93 coincidencias
en los originales. Dieciséis son igualdades decimales y 77 cumplen la política
documentada de [compatibilidad binary64](chinese-fact-history.md#precisión-del-original-y-del-importe-publicado).

| Símbolo | Informe primario | Páginas físicas del balance | Hechos admitidos | Muestras completas |
| --- | --- | --- | ---: | ---: |
| 000016.SZ | [1212725368](https://static.cninfo.com.cn/finalpage/2022-03-30/1212725368.PDF) | 99–102 | 6 | 178 |
| 000039.SZ | [1212710745](https://static.cninfo.com.cn/finalpage/2022-03-29/1212710745.PDF) | 191–192 | 6 | 270 |
| 000060.SZ | [1212711064](https://static.cninfo.com.cn/finalpage/2022-03-29/1212711064.PDF) | 106–107 | 6 | 127 |
| 000069.SZ | [1212757289](https://static.cninfo.com.cn/finalpage/2022-03-31/1212757289.PDF) | 136–138 | 3 | 288 |
| 000099.SZ | [1212687423](https://static.cninfo.com.cn/finalpage/2022-03-28/1212687423.PDF) | 78–81 | 6 | 74 |
| 000157.SZ | [1212746809](https://static.cninfo.com.cn/finalpage/2022-03-31/1212746809.PDF) | 116–119 | 6 | 290 |
| 000166.SZ | [1212752277](https://static.cninfo.com.cn/finalpage/2022-03-31/1212752277.PDF) | 248–249 | 6 | 371 |
| 000338.SZ | [1212750799](https://static.cninfo.com.cn/finalpage/2022-03-31/1212750799.PDF) | 92–95 | 3 | 262 |
| 000402.SZ | [1212750976](https://static.cninfo.com.cn/finalpage/2022-03-31/1212750976.PDF) | 87–88 | 6 | 248 |
| 000423.SZ | [1212685160](https://static.cninfo.com.cn/finalpage/2022-03-26/1212685160.PDF) | 69–72 | 6 | 186 |
| 000488.SZ | [1212748493](https://static.cninfo.com.cn/finalpage/2022-03-31/1212748493.PDF) | 100–101 | 6 | 209 |

000039.SZ expresa los importes en miles de yuanes. Las tablas de 000060.SZ y
000069.SZ incluyen una columna de apertura ajustada de 2021, distinta del cierre
de 2020. En 000069.SZ y 000338.SZ, los seis totales comparativos reexpresados
de 2020 no coinciden con los originales. Quedan excluidos. Sus cifras de 2021
sí tienen coincidencias admitidas. Los comparativos aceptados de otras empresas
conservan la publicación de 2022, sin anticiparse a 2020.

El informe de 000423.SZ imprime una fecha de aprobación de 24 de marzo de 2021
en la página física 94. La disponibilidad se obtiene del anuncio CNINFO del
26 de marzo de 2022. Esa frase interna no se utiliza para fechar las cifras.
En todos los casos se aplica el siguiente cierre admisible tras la fecha
publicada. Los documentos originales y los intentos fallidos se conservan
localmente y no se redistribuyen.

## Muestras y comprobaciones

La codificación utiliza exactamente la identidad de MiniLM y ResNet18 empleada
en la campaña estadounidense, incluido el código congelado y sus pesos.
Cada vector contable se contrasta con una referencia Decimal de 50 dígitos
transformada a float32. Coinciden los 2.503 vectores. Se comprueban las fechas
de disponibilidad, valores finitos y las 140 máscaras macro activas. La
recuperación de las once ediciones conserva bytes y fechas de modificación
sin repetir inferencia.

El proceso CUDA de preparación y comprobación tarda 88,87 segundos y alcanza
1.999.916 KiB de RAM. PyTorch registra 570.390.528 bytes asignados y
612.368.384 reservados en la RTX 4070 Laptop. La ventana completa, incluida
la pausa recuperable y la reanudación, dura 97,94 segundos. Son medidas de
una ejecución funcional. No se midieron energía ni bytes transferidos entre
CPU y GPU, y no se afirma una aceleración del entrenamiento.

La unión con los dos activos anteriores conserva sus muestras y utiliza el
CSI 300 de 2021 a 2023. La división anual preliminar tiene 1.103 filas de
entrenamiento y 2.105 de validación. Las diez ventanas temporales reúnen
3.195 muestras distintas y 1.691 filas de evaluación, sin repeticiones entre
meses. Se han recorrido las cuarenta particiones y comprobado fechas,
dimensiones y valores. Las filas de entrenamiento que reaparecen en ventanas
posteriores no se suman como muestras independientes.

La unión, recuperación y lectura temporal tarda 12,33 segundos, con un pico
de 742.340 KiB de RAM. Pasan 172 pruebas de los módulos contables, codificación,
unión y particiones. No se han modificado esos módulos ni vuelto a medir su
cobertura o mutación. El [recibo](../../reports/data/chinese-balance-expansion-20261007.json)
enlaza las once reconciliaciones y conserva hashes, recuentos por activo y por
ventana, tiempos y límites.

Esta edición dejaba 797 de los 810 instrumentos con cuatro fuentes potenciales
sin una edición contable admitida. El [segundo lote](chinese-balance-batch02.md)
amplía la unión a 25 empresas y 5.374 muestras, con otra identidad. Los anuales de este lote no
aportan por sí solos actualizaciones contables de 2023. Hay que ampliar
publicaciones y comprobar sus versiones antes de considerar completa la
cohorte. Los textos mantienen la categoría `original_audited`, sin atribuirles
una verificación editorial externa.

La comparación CN puede utilizar los nueve componentes contables comunes
entre sus modelos. El brazo US+CN requiere una proyección versionada que
conserve por separado los 23 conceptos estadounidenses y los tres chinos,
sus máscaras y sus edades. También falta conectar los dos calendarios con el
coordinador y evaluar cada brazo sobre las mismas claves. El modo de
ponderación común disponible es `natural`. No se presenta `balanced_markets`
como implementado para Ridge y XGBoost.

La campaña estadounidense conserva sus entradas y runtime. El test final de
2024 permanece cerrado. La procedencia de los ajustes OHLC sigue siendo
necesaria para validar simulaciones financieras reales con posiciones.
