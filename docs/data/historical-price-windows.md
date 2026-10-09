# Ventanas históricas de todos los activos con precios

El censo de la copia disponible reúne 17.076.024 ventanas de 64 sesiones consecutivas entre 2000 y 2023. Incluye todos los activos con precios auditados, aunque falten noticias, fundamentales o archivos gráficos originales. Son ventanas candidatas por precios. Todavía no son muestras con objetivo residual y representaciones multimodales preparados.

| Mercado | Activos con archivo de precios | Filas de precios anteriores a 2024 | Ventanas de 64 sesiones |
| --- | ---: | ---: | ---: |
| US | 4.213 | 16.090.522 | 14.407.908 |
| CN | 810 | 2.891.924 | 2.668.116 |

Los 1.574 activos US que no tienen archivos de todas las fuentes aportan 5.057.402 filas de precios y 4.286.530 ventanas. Las 10.121.378 ventanas US restantes y las 2.668.116 CN coinciden exactamente con el recuento anterior de activos con cuatro fuentes. La presencia de esos archivos no significa que todas sus modalidades estén disponibles en cada decisión.

El [recibo del censo](../../reports/data/historical-price-windows-20261007.json) concilia 48 combinaciones de mercado y año, los activos sin sesiones anteriores al corte y los que no llegan a formar una ventana completa. Los huecos de calendario interrumpen la continuidad. No se completan precios ni se acorta el contexto. Los primeros precios conservados son del 3 de enero de 2000 en US y del 4 de enero de 2006 en CN. La [revisión de huecos](historical-price-gaps.md) atribuye la ausencia de ventanas CN entre abril y agosto de 2019 y las filas que se rechazaban por redondeo.

La pasada verifica los hashes de 5.023 archivos Parquet, que suman 852.835.869 bytes. Solo interpreta su columna de sesiones anterior a 2024. No lee importes OHLC, etiquetas ni predicciones. La validez de precios y las exclusiones se heredan de la auditoría fijada, sin afirmar que se hayan reconstruido los ajustes retrospectivos de la fuente.

El proceso tarda 23,786 segundos internamente y alcanza 196.739.072 bytes de RSS. La unidad local tarda 24,613 segundos y declara un pico de memoria de 774,2 M, que incluye la contabilidad del grupo y no equivale al RSS. Se limita a una CPU, dos GiB y un hilo de Arrow y BLAS. Es una pasada funcional, sin calentamiento ni comparación de aceleración. Se conservan dos fallos previos del arnés de recuento, uno por el tipo `large_string` y otro por el vector de fechas vacío.

La edición histórica con máscaras debe conciliar estas ventanas con los motivos de exclusión de su objetivo y con los errores de preparación. La máscara admite una entrada opcional ausente, pero no crea precios ni etiquetas. La pausa de aprendizaje continúa mientras se prepara y verifica la nueva edición.
