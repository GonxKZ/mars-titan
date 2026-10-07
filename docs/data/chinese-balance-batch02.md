# Segundo lote de balances chinos

El segundo lote añade doce empresas y 2.150 muestras reales con las cuatro
modalidades y los 140 indicadores macro completos. La unión con la
[edición anterior](chinese-balance-expansion.md) contiene 5.374 muestras de
25 empresas. Sus diez ventanas utilizan 5.327 muestras distintas y 2.844 filas
de evaluación, sin repeticiones entre ventanas. La cobertura sigue siendo
parcial y no se ha lanzado una campaña científica con esta edición.

Los doce informes se recuperaron desde anuncios ya conservados, sin consultas
nuevas al catálogo ni reintentos de descarga. Los PDF completos ocupan
51.160.483 bytes y tienen 2.984 páginas. La revisión visual abarca 71 páginas
y 72 cifras. El contraste con los originales admite 62 hechos y 94
coincidencias, con la política de precisión ya documentada.

| Símbolo | Informe primario | Hechos admitidos | Muestras completas |
| --- | --- | ---: | ---: |
| 000528.SZ | [1212752303](https://static.cninfo.com.cn/finalpage/2022-03-31/1212752303.PDF) | 3 | 168 |
| 000538.SZ | [1212688576](https://static.cninfo.com.cn/finalpage/2022-03-26/1212688576.PDF) | 6 | 240 |
| 000541.SZ | [1212769001](https://static.cninfo.com.cn/finalpage/2022-04-01/1212769001.PDF) | 6 | 143 |
| 000550.SZ | [1212727390](https://static.cninfo.com.cn/finalpage/2022-03-30/1212727390.PDF) | 6 | 296 |
| 000553.SZ | [1212747054](https://static.cninfo.com.cn/finalpage/2022-03-31/1212747054.PDF) | 6 | 53 |
| 000555.SZ | [1212750276](https://static.cninfo.com.cn/finalpage/2022-03-31/1212750276.PDF) | 6 | 160 |
| 000617.SZ | [1212810890](https://static.cninfo.com.cn/finalpage/2022-04-02/1212810890.PDF) | 6 | 137 |
| 000629.SZ | [1212710834](https://static.cninfo.com.cn/finalpage/2022-03-29/1212710834.PDF) | 3 | 146 |
| 000680.SZ | [1212712525](https://static.cninfo.com.cn/finalpage/2022-03-29/1212712525.PDF) | 6 | 95 |
| 000698.SZ | [1212730657](https://static.cninfo.com.cn/finalpage/2022-03-30/1212730657.PDF) | 4 | 70 |
| 000725.SZ | [1212750904](https://static.cninfo.com.cn/finalpage/2022-03-31/1212750904.PDF) | 4 | 382 |
| 000728.SZ | [1212710930](https://static.cninfo.com.cn/finalpage/2022-03-29/1212710930.PDF) | 6 | 260 |

La tabla se vincula a los recibos de cada revisión. Se conservan CNY, normas
CAS, balance consolidado, multiplicador y patrimonio total con minoritarios.
Los guiones impresos en algunos componentes no se convierten en ceros
documentados. Las columnas de apertura ajustada de 2021 se distinguen del
cierre de 2020.

Los seis comparativos reexpresados de 2020 de 000528 y 000629 no coinciden con
los originales y quedan excluidos. En 000698 y 000725 tampoco coinciden los
activos y el patrimonio de 2021. Se conservan los pasivos de 2021 y los tres
hechos admitidos de 2020. La selección por concepto utiliza el hecho más
reciente disponible para cada variable. Por tanto, los vectores de esas dos
empresas combinan cierres distintos, publicados en 2022. No se presentan como
balances de un mismo cierre y no se calculan ratios empresariales con ellos.
El periodo de cada hecho queda conservado en su procedencia.

La comprobación CUDA usa los codificadores congelados de la edición
estadounidense. Coinciden los 2.150 vectores contables con la referencia Decimal
de 50 dígitos transformada a float32. Se comprueban fechas, valores finitos y
las 140 máscaras macro. La recuperación de las doce ediciones conserva bytes
sin repetir inferencia.

El proceso completo de codificación y comprobación tarda 84,16 segundos.
`/usr/bin/time` registra 2.033.084 KiB de RAM. PyTorch alcanza 570.390.528 bytes
asignados y 612.368.384 reservados en CUDA. La unión, recuperación y lectura de
las cuarenta particiones tarda 16,92 segundos, con 757.852 KiB de RAM. Son
medidas funcionales de una repetición, sin medición de energía o transferencias
CPU/GPU ni afirmación de aceleración.

El [recibo](../../reports/data/chinese-balance-expansion-batch02-20261007.json)
conserva las doce reconciliaciones, recuentos, hashes, periodos por concepto y
límites. Quedan 785 de los 810 instrumentos con cuatro fuentes potenciales sin
edición contable admitida en este recorrido. La captura de anuncios sigue
siendo parcial y los anuales revisados no aportan por sí solos actualizaciones
contables de 2023. El test final permanece cerrado.
