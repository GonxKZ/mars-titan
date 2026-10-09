# Edición v3.1 de la historia desde 2000

Tarea [#428](https://github.com/GonxKZ/mars-titan/issues/428), 9 de octubre de 2026. La edición v3.1 cambia solo la lectura de precios de la edición `encoded-history-v3-cpu-words`. Corrige tres cosas: la conversión del texto numérico de la fuente, las filas que se rechazaban por redondeo y las ventanas que contienen sesiones sin ninguna fila en todo el mercado. Noticias, fundamentales, macro, codificadores y política de máscaras no cambian.

Este documento describe el código. La auditoría, la preparación, la codificación y los objetivos v3.1 no se han ejecutado sobre los datos y esperan aprobación. La edición v3, `targets-v3` y sus vistas siguen intactas. No se ha entrenado ni evaluado ningún modelo.

## Conversión exacta del texto de la fuente

Hasta la v3, los CSV de precios se leían con `pandas.read_csv` y su motor C por defecto. Ese conversor no redondea correctamente. Para una parte de los textos devuelve un `float64` vecino del que corresponde al decimal escrito. La referencia es `float(text)` de Python, que da el `float64` más próximo al decimal.

La orden `verify-number-parsing` relee los 5.023 CSV de precios y compara cada valor con esa referencia. El [recibo](../../reports/data/edition-v3-1-number-parsing-20261009.json) registra lo siguiente para las columnas `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends` y `Stock Splits`:

| Mercado | Archivos | Valores | Ausentes | Diferencias del lector exacto | Diferencias del lector v3 | Máximo en ULP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| US | 4.213 | 122.141.159 | 0 | 0 | 10.894.452 (8,9 %) | 2.768 |
| CN | 810 | 21.975.884 | 8 | 0 | 1.786.171 (8,1 %) | 7 |

Una diferencia de *k* ULP equivale a un error relativo inferior a *k*·2⁻⁵², así que el peor caso es menor que 6,2e-13. Es un error pequeño, pero sistemático, y afecta a ventanas, gráficos y objetivos. **La edición v3 conserva este error** y no se modifica. La v3.1 lo elimina.

`source_numbers.exact_floats` es ahora el único conversor de los CSV de precios. Acepta solo decimales ASCII con signo y exponente opcionales y aplica `float(text)`. Una celda vacía o un texto no numérico queda ausente, como antes, y cualquier otro tipo detiene la lectura. El volumen pasa a `float64` porque así lo produce el mismo conversor. La identidad `python_float_correctly_rounded_v1` entra en la configuración de la revisión de precios.

En el código de datos quedan otras dos lecturas de texto numérico con bibliotecas. Las cifras NBS de las ediciones macro se extraen con `pandas.read_html`. Se volvieron a extraer los 58 documentos conservados con todas las celdas como texto y las 298 observaciones coincidieron bit a bit en ambos mercados. Los precios sin ajustar usan el lector CSV de pyarrow, que coincidió con `float(text)` en 9.000 textos aleatorios en los que el motor C de pandas falló 1.375 veces. Ninguna de las dos lecturas alimenta la edición v3.1.

## Filas con desorden por redondeo

La fuente redondea cada precio por separado, de modo que un cierre en el máximo puede quedar una unidad de la última cifra por encima. La auditoría con `--ordering-rtol 1e-9` admite una fila cuando su único defecto es ese desorden y su exceso relativo, `max(top - high, low - bottom) / top`, no supera la tolerancia. La fila conserva exactamente sus cinco valores. No se recalcula ninguna envolvente.

La relajación afecta únicamente a la comprobación de orden. El estado de la auditoría guarda la tolerancia y la lectura auditada, la preparación y el gráfico la reciben de ahí. Cada fila admitida queda registrada con su fila de origen, su fecha y su exceso relativo. Las filas con otro defecto, como una apertura por debajo del mínimo, siguen excluidas.

## Ventanas con sesiones ausentes en todo el mercado

### Qué se admite

Una ventana cubre las 64 sesiones del calendario oficial que terminan en la decisión. Solo puede faltar una sesión en la que ningún archivo de la fuente del mercado tiene una fila, aceptada o excluida. Esas sesiones se buscan entre la primera sesión observada y la última, sin pasar del 31 de diciembre de 2023. Con los recuentos preparados, el resultado es el 29 y el 30 de abril de 2019 en CN y ninguna en US. La lista no se escribe a mano. `price_revision.market_absences` la deriva de los derivados de la auditoría y la guarda en `market-absent-sessions.json`.

Si a un activo le falta una sesión que otros activos sí tienen, la ventana sigue excluida como en la v3. Una ausencia declarada que tenga una fila en cualquier activo detiene la codificación.

### Contrato de la ventana

`data/price_windows.py` define el contrato versión 1. Lo guardan la configuración de la edición, el recibo de cada activo y la identidad de representación, así que una ventana v3.1 no puede leerse como una v3 ni al revés.

| Campo | Valor |
| --- | --- |
| Posiciones | Sesiones del calendario del mercado que terminan en la decisión, 64 |
| Canales | Cinco de precio y `price_present` |
| Relleno de un hueco | +0.0 exacto en los cinco canales y presencia 0 |
| Ancla | Cierre de la primera sesión presente |
| Escala del volumen | Media de las sesiones presentes |
| Calendarios | Inicio, fin y huella de las decisiones de cada mercado |
| Ausencias | Lista por mercado derivada de la fuente |

Una ventana sin huecos usa exactamente las mismas operaciones que la v3 y añade la presencia a uno. Las pruebas comparan los cinco canales bit a bit con el lector anterior sobre los mismos precios preparados. Frente a la v3 publicada, las ventanas cambiarán en la última cifra allí donde cambie la conversión del texto, y la comparación entre ediciones mide ese cambio.

En el gráfico, cada sesión presente se dibuja en su posición del calendario y un hueco deja su columna vacía. Una ventana completa produce el mismo PNG que antes.

### Canal frente a indicador separado

El bit de presencia viaja como sexto canal del mismo tensor. Un indicador aparte obligaría a cambiar la estructura de los lotes, las vistas, el corpus ordenado, las formas del entorno de refuerzo y la interfaz nativa para transportar un segundo tensor que siempre acompaña al primero. Con un canal, cada familia recibe el bit en el mismo sitio que los precios, la vista de información declara que los cinco canales dependen de él y una ablación no puede quitar el bit sin quitar los precios. El coste es que la primera capa de cada codificador de precios recibe seis entradas en lugar de cinco y las familias tabulares 64 columnas más. Las dos cosas ya exigen una identidad nueva.

### Relleno anulado en todas las familias

El relleno no es un dato. `gate_price_window` deja en +0.0 los cinco canales de cada paso con presencia 0 mediante una selección (`np.where` o `torch.where`), así que el relleno no recibe gradiente aunque se cambie su valor. Todas las familias leen el tensor de `CorpusDataset.price_windows` y lo filtran así:

| Consumidor | Punto de aplicación |
| --- | --- |
| Referencias RNN, LSTM, GRU, DLinear y Transformer | `MultimodalReference.forward`, antes del codificador de precios |
| Tabulares (Ridge y XGBoost) | `tabular_corpus._matrix`, antes de aplanar |
| Titans-MAC y MARS-TITAN | Validación en CPU antes de crear tensores, huella de la validación y filtro antes del codificador de precios |
| CM-v1 | Lee las mismas entradas a través del predictor de MARS-TITAN y de M3 |
| Puntuación M3 | La anomalía usa rendimientos entre cierres observados consecutivos |
| GRU candidata nativa | Valida el canal de presencia con la política histórica |

Las pruebas cambian solo el relleno (0, 3,75 y −1000) y exigen las mismas predicciones y los mismos gradientes en las cinco referencias, la misma matriz tabular y la misma anomalía M3. El validador rechaza un bit distinto de 0 o 1, un último paso ausente, menos de dos pasos presentes y cualquier relleno distinto de +0.0, incluido −0.0. Titans rechaza además un relleno escrito sobre el tensor después de la validación.

### Alternativa descartada

Se evaluó tratar el 29 y el 30 de abril de 2019 como días sin mercado. Un paso abarcaría entonces tres sesiones, el horizonte de la etiqueta de la sesión anterior cambiaría sin que nada lo declare, el calendario cambiaría para todos los activos y la ausencia quedaría oculta. La máscara mantiene el calendario oficial y deja la ausencia a la vista de cada modelo.

### Objetivos y refuerzo

El objetivo de una decisión cuya sesión siguiente falta en todo el mercado sigue sin existir. La decisión del 26 de abril de 2019 conserva `missing_next_session` porque el 29 no tiene precio y no se inventa ninguno. Las decisiones de mayo a julio cuya ventana contiene el hueco recuperan su etiqueta si la sesión siguiente tiene precio.

La etapa de refuerzo no consume las ventanas de los modelos. Sus cintas de precios no tienen filas esas dos sesiones, igual que en la v3, y el entorno las trata como una suspensión de todos los activos. No se ha cambiado su comportamiento. Los mundos sintéticos y la preparación de algunos postentrenamientos generan ventanas de cinco canales y fallan al recibir una fuente de seis, en lugar de mezclarlas.

## Herramientas de la regeneración

La regeneración prevista reutiliza todo lo que no depende de los precios.

- `price_revision.revise_prepared_prices` crea la preparación v3.1 a partir de `prepared-accounting-v1` y de la auditoría nueva. Enlaza con enlaces duros noticias y fundamentales tras comprobar su huella. Relee los precios con el lector auditado y los enlaza también si el Parquet es idéntico. Se detiene si un activo sin precios en el padre los gana.
- `encode_corpus --price-window --vector-carry` reutiliza los vectores de la v3 cuando el PNG o el texto son idénticos y el codificador es el mismo. La v3 ya comprobó que recodificar con la caché vacía da muestras idénticas. La codificación tiene tres pasadas para que la GPU solo calcule. `--collect` recorre en CPU cada activo en `collect/`, lo confirma si tiene todos sus vectores y, si le falta alguno, lo descarta y anota su texto o su PNG en `pending-vectors*.sqlite`. Ningún vector provisional llega a `samples/`. `--encode-pending` codifica en GPU solo esas entradas, una a una con la misma llamada que la codificación en línea, y admite tramos con `--max-pending` para liberar el candado. Una última pasada en CPU con `--reuse-only` completa los activos pendientes con `computed-vectors.sqlite` y falla si todavía falta algún vector. `--shard K N` reparte las pasadas de CPU entre procesos acotados.
- `edition_comparison.compare_editions` recorre las dos ediciones activo por activo con un registro reanudable. Cuenta sesiones nuevas y perdidas, registra cada gráfico que cambia y cualquier otra columna distinta, y mide el cambio de las ventanas comunes.

## Pruebas

Las pruebas cubren el contrato, la admisión en el codificador, la lectura desde el corpus, las vistas temporales y el corpus ordenado, las familias, M3, la vista de información, la GRU nativa y las tres herramientas. La mutación dirigida se resume en la PR.

## Pendiente

Falta ejecutar la auditoría con la tolerancia, la revisión de precios, la codificación, `targets-v3.1` y la verificación completa (verificador de la edición, `verify-targets` y comparación con la v3). Los recuentos esperados por la simulación son 2.824.687 ventanas CN y 15.874.289 US. Antes se medirá en una porción pequeña el tiempo de CPU, el de GPU y el disco.
