# Historia contable china con publicaciones separadas

La preparación admite varias ediciones contables revisadas mediante
`additional_facts`. Cada una conserva su documento, anuncio, fecha de
disponibilidad y localizadores del original. La unión se ordena por disponibilidad
y periodo. Añadir un informe posterior no adelanta sus cifras ni sustituye las
observaciones anteriores.

El [anual de Ping An de 2021](https://static.cninfo.com.cn/finalpage/2022-03-10/1212533413.PDF)
acredita activos, pasivos y patrimonio consolidado de 2021 y 2020 en las páginas
físicas 137–138, expresados en millones de CNY. Se publicó el 10 de marzo de 2022.
Los seis hechos quedan disponibles desde el siguiente cierre, el 11 de marzo.
Los comparativos de 2020 conservan esa fecha.

Vanke tiene dos revisiones: el [anual de 2021](https://static.cninfo.com.cn/finalpage/2022-03-31/1212750450.PDF),
páginas físicas 142–143, y el [anual de 2022](https://static.cninfo.com.cn/finalpage/2023-03-31/1216273938.PDF),
páginas 131–132. Ambos presentan balances consolidados CAS en CNY, con céntimos.
El patrimonio total incluye los minoritarios, diferenciados del patrimonio
atribuible a la matriz. Sus disponibilidades son el 1 de abril de 2022 y el
3 de abril de 2023. Los tres totales comparativos de 2021 coinciden entre los
dos documentos.

Las revisiones de [Ping An](../../reports/data/china-pingan-2021-reconciliation-20261007.json),
[Vanke 2021](../../reports/data/china-vanke-2021-reconciliation-20261007.json) y
[Vanke 2022](../../reports/data/china-vanke-2022-reconciliation-20261007.json)
conservan los importes, páginas, unidades y coincidencias. No se publican los PDF.

## Precisión del original y del importe publicado

La igualdad decimal exacta sigue siendo la regla predeterminada. El JSON de
Vanke contiene, por ejemplo, `1757124444202.9499511719`, mientras el documento
publica `1757124444202.95`. Convertir ambos directamente a `float` oculta esa
diferencia. La política opcional `binary64_roundtrip` exige simultáneamente:

- La misma conversión a `float64` para el original y el importe publicado en CNY.
- Una unidad decimal publicada al menos dos veces mayor que la ULP de ese valor.
- Que cuantizar el original a esa unidad decimal produzca el importe publicado.

La unidad publicada se obtiene de los decimales conservados en la revisión y
del multiplicador declarado. La [ULP](https://docs.python.org/3.12/library/math.html#math.ulp)
se calcula con la biblioteca estándar, y [Decimal.from_float](https://docs.python.org/3.12/library/decimal.html#decimal.Decimal.from_float)
conserva su valor binario exacto para la comparación. No se utiliza una tolerancia
relativa general. Una variación de un céntimo o una precisión binaria insuficiente
para distinguirlo impiden la coincidencia.

Esta compatibilidad no demuestra qué programa serializó el original. Cada
revisión debe activar la política explícitamente y conserva el valor original,
su diferencia decimal, la ULP y la precisión publicada. El Parquet mantiene el
decimal documental en `value_exact` y su conversión a float64 en `value`.
La conversión no se presenta como almacenamiento exacto de los céntimos.

## Unión y comprobaciones

La unión rechaza ediciones repetidas, hechos duplicados o contradictorios de
la misma publicación, otros emisores y cortes incompatibles. Un ordinal del
original puede respaldar cifras coincidentes en dos publicaciones distintas.
Los recuentos distinguen hechos, coincidencias y registros originales únicos.
Se limitan a 16 ediciones, 100.000 hechos y 64 MiB agregados antes de descomprimir.
La preparación de una sola edición conserva la copia de sus bytes.

La [comprobación real](../../reports/data/chinese-fact-history-20261007.json)
obtiene doce hechos por empresa y los siguientes recorridos independientes:

| Activo | Muestras con cuatro modalidades y 140 macros | Entrenamiento anual preliminar | Validación anual preliminar |
| --- | ---: | ---: | ---: |
| Ping An | 335 | 104 | 211 |
| Vanke | 386 | 116 | 240 |

Las 721 muestras producen 671 etiquetas utilizables bajo esa división.
Se excluyen 46 por historial insuficiente del factor, dos por el cambio de año
y dos por el corte final. El factor empleado en esta edición comienza en 2022.
Recuperar más historia factorial es un trabajo separado, con otra identidad.

La ejecución CUDA contrasta todos los vectores contables con una referencia
Decimal de 50 dígitos después de convertir el resultado a float32. Los vectores
coinciden exactamente. También conserva íntegramente las 169 muestras de la
primera edición de Ping An y recupera ambas ediciones sin repetir inferencia.
El proceso completo tarda 33,62 segundos y alcanza 2.035.732 KiB de RAM.
PyTorch registra 570.390.528 bytes asignados y 612.368.384 reservados en CUDA.
No se midieron energía ni transferencias CPU/GPU.

Pasan 175 pruebas y seis mutaciones dirigidas. La cobertura y la convención
de CRAP constan en el recibo. La campaña activa se pausó de forma recuperable y
se reanudó al terminar la comprobación. La [unión posterior](chinese-corpus.md)
prepara el corpus común y diez ventanas de validación, calibración y evaluación
para esos dos activos. Falta ampliar la cobertura para la comparación china
completa. El test final de 2024 permanece cerrado.
