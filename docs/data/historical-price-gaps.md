# Huecos de precios en la edición desde 2000

Revisión del 9 de octubre de 2026 ([#426](https://github.com/GonxKZ/mars-titan/issues/426)). La edición codificada `encoded-history-v3-cpu-words` no contiene ninguna muestra CN entre el 29 de abril y el 1 de agosto de 2019. De enero a abril de ese año reúne unos 500 activos por sesión, cuando en diciembre de 2018 había unos 715. Este documento explica la causa, clasifica lo encontrado en todo el periodo de 2000 a 2023 y separa lo que se ha corregido en código de lo que queda pendiente de regenerar. No se han modificado `dataset/`, la edición v3, `targets-v3` ni las vistas, y no se ha ejecutado ningún aprendizaje.

## Cómo se pierde una ventana

La codificación solo admite una decisión cuando las 64 sesiones del calendario que terminan en ella tienen una fila de precio aceptada (`incomplete_price_window`). Es el único motivo de exclusión que aparece en los manifiestos de la edición v3 para ambos mercados. Una sesión sin precio elimina, por tanto, las 64 ventanas que la contienen. Además, la etiqueta de la sesión anterior queda como `missing_next_session`.

Para atribuir cada pérdida se reprodujo la regla a partir de los CSV originales y del calendario. La reproducción coincide activo a activo con la edición v3, tanto en las sesiones de precios preparadas como en las sesiones de las muestras, sin ninguna discrepancia en los 5.023 activos.

## Recorrido hacia atrás

| Eslabón | Comprobación | Resultado |
| --- | --- | --- |
| Calendario XSHG (`MarketClock`, exchange_calendars 4.13.2) | Sesiones de abril y mayo de 2019 | Contiene el 29 y el 30 de abril |
| Precios preparados (`prepared-accounting-v1`) | Activos con precio por sesión de 2006 a 2023 | El 29 y el 30 de abril de 2019 no tienen ningún activo. El 7 de enero de 2019 tiene 509 en lugar de unos 749 |
| Fuente `dataset/time_series/HS300_time_series` | Filas por fecha en los 810 CSV | Ningún archivo contiene el 29 ni el 30 de abril de 2019. El 7 de enero aparece en 749 archivos |
| Factor CSI 300 v16 | Sesiones de 2006 a 2023 | Completo, incluye ambas sesiones |
| Macro, noticias y gráficos | Motivos de exclusión del codificador | Con la política de máscaras no excluyen muestras. Todas las exclusiones son de ventana de precios |
| Capturas SSE de `unadjusted-prices-20261009` | Series diarias oficiales de diez valores de Shanghái | Los nueve que cotizaban en 2019 tienen el 29 y el 30 de abril y el 7 de enero |

El mes a mes de CN en 2019 queda así:

| Mes | Sesiones del calendario | Sesiones con precio | Sesiones con muestras | Muestras v3 | Etiquetas aceptadas |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2018-12 | 20 | 20 | 20 | 14.335 | 14.279 |
| 2019-01 | 22 | 22 | 22 | 11.600 | 11.335 |
| 2019-02 | 15 | 15 | 15 | 7.493 | 7.435 |
| 2019-03 | 21 | 21 | 21 | 10.464 | 10.379 |
| 2019-04 | 21 | 19 | 19 | 11.820 | 10.997 |
| 2019-05 | 20 | 20 | 0 | 0 | 0 |
| 2019-06 | 19 | 19 | 0 | 0 | 0 |
| 2019-07 | 23 | 23 | 0 | 0 | 0 |
| 2019-08 | 22 | 22 | 21 | 15.669 | 15.534 |

El [recibo](../../reports/data/historical-price-gaps-20261009.json) conserva estas cifras, las huellas de los manifiestos consultados y el desglose anual.

## Tres causas distintas

**Ausencia real en la copia.** El 29 y el 30 de abril de 2019 faltan en todos los CSV chinos. El calendario y el factor CSI 300 incluyen esas sesiones y las capturas oficiales de SSE confirman que hubo negociación. La ausencia procede de la copia de FinMultiTime y no de la preparación. Con la regla de 64 sesiones se pierden 65 sesiones completas, desde el 29 de abril hasta el 1 de agosto. La primera ventana sin esos dos días termina el 2 de agosto. Se registra como ausencia explícita. No se rellenan precios.

**Filas defectuosas en la fuente.** El 7 de enero de 2019, 240 valores de Shanghái tienen una apertura inferior al mínimo. En las capturas de SSE esa apertura coincide con la de la sesión anterior, de modo que la fila es incoherente y su exclusión es correcta. Esos activos pierden las ventanas del 7 de enero al 12 de abril, y eso explica el descenso hasta unos 500 activos por sesión. El mismo tipo de defecto aparece en 308 filas CN y 688 US anteriores a 2024. Las fechas con más casos son el 29 de mayo de 2020 en CN (37 filas) y el 5 de mayo de 2021 y el 5 de junio de 2023 en US (30 y 25). Siguen excluidas.

**Rechazo por redondeo.** Las series ajustadas de la fuente redondean cada precio por separado. Cuando una sesión cierra en su máximo o en su mínimo, el cierre puede quedar fuera del rango por una unidad de la última cifra. La lectura estricta rechazaba esas filas. Son 3.454 filas CN de 392 activos y 59.123 US de 2.243 activos, todas con un exceso relativo menor que 1,4e-15. Cada fila arrastra hasta 64 ventanas, así que se perdían 109.001 ventanas CN (un 4,1 %) y 1.466.381 US (un 10,2 %). En US la pérdida es mayor en los primeros años, con un 27,7 % de las ventanas de 2000 y un 13,3 % de las de 2010. No es una pérdida aleatoria. Elimina las etiquetas de las sesiones que cierran en un extremo y, en CN, más de la mitad de las filas planas sin volumen que la lectura estricta sí acepta cuando no hay redondeo. Este caso es un fallo de preparación y se ha corregido en código.

Fuera de 2019 no hay otra sesión del calendario sin ningún precio en CN ni en US. Las caídas bruscas de activos por sesión de CN (enero de 2019 y mayo de 2020) y de US (mayo de 2021 y junio de 2023) corresponden a filas defectuosas.

## Corrección implementada

`read_prices` y `audit_prices` aceptan ahora una tolerancia explícita `ordering_rtol`, que vale cero por defecto y no puede superar 1e-6. La orden `audit-prices --ordering-rtol 1e-9` crea otra política de auditoría, con otro hash, otro estado y otro destino. Una fila se admite solo si su único defecto es ese desorden. Conserva apertura, cierre y volumen y toma como máximo y mínimo la envolvente de los cuatro precios. La auditoría añade `ordering_rounded_rows`, el exceso relativo máximo y un `ordering_roundings.parquet` con los valores originales de cada fila. El lector auditado posterior sigue exigiendo el orden estricto y acepta esa envolvente sin cambios.

Sobre la copia real, con 1e-9 y antes de 2024, CN pasa de 2.891.924 a 2.895.378 filas y US de 16.090.522 a 16.149.645. Ninguna fila estricta cambia. Con tolerancia cero las salidas y los campos de auditoría son los de antes. La corrección no altera la edición v3. Entrará en una edición nueva solo cuando se apruebe su regeneración.

## Efecto en la comparación walk-forward

Los tramos de cada ventana son los del [protocolo v2](../research/walk-forward-2000.md). Dos ventanas CN, y las mismas del ámbito conjunto, contienen el hueco:

| Ventana | Tramo CN afectado | Sesiones con etiquetas | Etiquetas aceptadas |
| --- | --- | ---: | ---: |
| Evaluación de 2019 | Evaluación de 2019 | 178 de 244 | 115.801, frente a 173.005 en 2018 y 180.595 en 2020 |
| Evaluación de 2020 | Validación de abril a septiembre de 2019 | 59 de 125 | 41.349, frente a 88.551 en la validación anterior |

La calibración de ambas ventanas está completa. US no tiene sesiones vacías, aunque todas sus ventanas cambian con la corrección del redondeo.

Todos los brazos se evalúan con las mismas filas, por lo que el hueco no favorece a ningún modelo por sí mismo. Sí cambia lo que se mide. La evaluación CN de 2019 no incluye mayo, junio ni julio, que contienen la caída del 6 de mayo tras la escalada arancelaria, y de enero a abril excluye casi todos los valores de Shanghái. Su MAE por sesión describe un año incompleto y con otra composición, y su incertidumbre por bloques será mayor. La selección de la ventana de 2020 usa en CN menos de la mitad de las sesiones de validación habituales. Si se agregan años con el mismo peso, 2019 cuenta como un año completo con dos tercios de sus sesiones. Conviene informar los años CN 2019 y 2020 con su cobertura y repetir el resumen sin ellos como análisis de sensibilidad.

## Pendiente

La regeneración con la tolerancia nueva no se ha ejecutado. Sustituiría datos verificados y necesita aprobación expresa. El plan del 9 de octubre propone una edición nueva que reutilice sin cambios los activos no afectados y vuelva a preparar, codificar y etiquetar los 2.635 afectados.

Hay dos decisiones abiertas que no forman parte de esta corrección. La primera es admitir ventanas con sesiones ausentes en todo el mercado mediante una máscara por sesión. Recuperaría 47.570 ventanas CN de mayo a julio de 2019, pero cambia la entrada de precios de todos los modelos y exigiría otra identidad y otra comparación. La segunda es si las filas planas sin volumen, que suelen representar suspensiones, deben contar como observaciones. Ya se aceptan 111.211 en CN y la corrección no cambia ese criterio.
