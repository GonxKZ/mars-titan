# Edición v3.1 de la historia desde 2000

Tarea [#428](https://github.com/GonxKZ/mars-titan/issues/428), 9 y 10 de octubre de 2026. La edición v3.1 parte de la edición `encoded-history-v3-cpu-words`. Corrige tres cosas en la lectura de precios: la conversión del texto numérico de la fuente, las filas que se rechazaban por redondeo y las ventanas que contienen sesiones sin ninguna fila en todo el mercado. Además, todos sus vectores se calculan en FP32 estricto, sin TF32. Noticias, fundamentales, macro, modelos de los codificadores y política de máscaras no cambian.

Este documento describe el código y una medida sobre una porción de 13 activos copiada fuera de las rutas oficiales. La auditoría, la preparación, la codificación y los objetivos v3.1 completos no se han ejecutado y esperan la orden del coordinador. La edición v3, `targets-v3` y la preparación `prepared-accounting-v1` siguen intactas. No se ha entrenado ni evaluado ningún modelo.

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

## Precisión de los codificadores

La especificación de la v3 registra `runtime_precision` con `matmul_tf32: false` y `cudnn_tf32: true`. El lanzador de la v3 activó TF32 en cuDNN, así que las convoluciones de ResNet18 se calcularon con mantisa de 10 bits. La v3.1 calcula todos sus vectores en FP32 estricto, como exige el proyecto para no perder precisión.

- `embeddings.strict_fp32()` desactiva TF32 en matmul y en cuDNN y fija `float32_matmul_precision("highest")`. La orden de codificación la llama al arrancar, en todas sus pasadas.
- `FrozenEncoders` falla al arrancar, antes de abrir CUDA o cargar pesos, si alguno de esos tres ajustes permite TF32. Antes de cada llamada comprueba además que la precisión efectiva sigue siendo la registrada.
- `runtime_precision` registra ahora cuatro campos: `dtype`, `matmul_tf32`, `cudnn_tf32` y `float32_matmul_precision`. La identidad del codificador cambia, de modo que ningún vector de la v3 puede confundirse con uno de la v3.1.
- `embeddings.encoder_spec` calcula esa identidad en CPU sin cargar los modelos. Las pasadas de CPU la usan para nombrar los vectores pendientes, y `FrozenEncoders` usa la misma función, así que ambas coinciden. La pasada de GPU lo comprueba antes de codificar.

Un vector solo se hereda de otra edición si las dos registran la misma identidad de codificador y esa identidad declara FP32 estricto. La v3 no cumple la segunda condición. Sus vectores de gráficos se calcularon con TF32 en cuDNN y se recalculan todos. Los textos de la v3 se heredan por otra vía, con un contraste por activo que se describe en la sección siguiente.

## Textos heredados de la v3

La v3 solo registra la precisión de toda la edición, con `cudnn_tf32: true`, y ningún metadato por vector separa los textos de los gráficos. La decisión de heredar sus textos se apoya en tres elementos.

El primero es cómo se calcularon. PyTorch mantiene TF32 desactivado en matmul por defecto y la v3 lo registra así (`matmul_tf32: false`). La marca de cuDNN solo afecta a las operaciones que pasan por cuDNN, como las convoluciones de ResNet18. El modelo de texto está hecho de capas lineales, atención y normalización, que no usan cuDNN, así que la marca no debería haber cambiado ningún texto. El segundo es la medida. En la porción, los 5.051 vectores de texto que comparten la v3 y la v3.1 coinciden bit a bit uno a uno, y las medias de noticias de las 40.443 sesiones comunes también. El tercero es que ni el argumento ni una porción de 13 activos garantizan el resto del universo, y por eso cada activo vuelve a comprobarlo.

1. `vector_carry.check_text_source` compara las identidades de los codificadores. Deben coincidir el modelo, el tokenizador, las versiones, la ubicación de la tabla de palabras y el lote de fragmentos de texto, y la edición de origen no puede haber usado TF32 en matmul. Solo pueden cambiar la precisión registrada, la huella del código del módulo y el lote de gráficos.
2. En la recogida, un texto que falta en la v3.1 y que la v3 ya tiene no pasa directamente a la GPU. Se reúne con los demás textos heredables del activo, en el orden en que el activo los pide, que es determinista. Al terminar el activo se elige la muestra: el primero, el último, uno de cada 64 y, en total, al menos 8 repartidos entre todos (todos si el activo tiene 8 o menos). Solo la muestra queda pendiente de GPU, pero se guarda el contenido de todos los textos heredables.
3. Tras codificar los pendientes, `encode_pending` compara cada texto de la muestra con su vector de la v3 y exige igualdad bit a bit. Si todos coinciden, el activo puede usar los vectores de la v3. Si alguno difiere, la misma pasada codifica en FP32 estricto todos los textos heredables del activo, que ya no usa ninguno de la v3.
4. La constancia de cada activo queda en `text-carry/<mercado>/<símbolo>.json`, con la edición de origen, las huellas de su configuración y de su codificador, la regla, los parámetros de la muestra, el número de textos heredables, contrastados y distintos (con los primeros contenidos que difieren), la decisión y los textos recodificados. La configuración de la edición registra la herencia en `text_carry`, y todas las pasadas deben declararla igual.
5. La pasada final busca primero los vectores calculados en FP32 estricto y solo después los de la v3, y únicamente para un activo con el contraste superado.

El límite está en el muestreo. Un texto fuera de la muestra que difiriera pasaría inadvertido, y la comparación previa a la sustitución tampoco lo vería, porque el vector heredado es precisamente el de la v3. Una diferencia que afectara a muchos textos de un activo aparecería en su muestra, que nunca deja más de 63 textos seguidos sin contrastar, pero una que afectara a textos sueltos podría escapar. Los gráficos se recalculan siempre y no pasan por esta herencia.

## Herramientas de la regeneración

La regeneración prevista reutiliza todo lo que no depende de los precios ni de la precisión.

- `price_revision.revise_prepared_prices` crea la preparación v3.1 a partir de `prepared-accounting-v1` y de la auditoría nueva. Enlaza con enlaces duros noticias y fundamentales tras comprobar su huella. Relee los precios con el lector auditado y los enlaza también si el Parquet es idéntico. Se detiene si un activo sin precios en el padre los gana.
- La codificación tiene tres pasadas para que la GPU solo calcule. `--collect` recorre en CPU cada activo en `collect/`, lo confirma si tiene todos sus vectores y, si le falta alguno, lo descarta y anota su texto o su PNG, con el activo que lo pide, en `pending-vectors*.sqlite`. Ningún vector provisional llega a `samples/`. `--encode-pending` codifica en GPU solo esas entradas, cada texto por separado y los gráficos en lotes de `--image-batch-size`, y admite tramos con `--max-pending` para liberar el candado. Un lote de cada 64, y siempre el último si es más corto, se contrasta con la codificación de su último gráfico solo, y cualquier diferencia detiene la pasada. Una última pasada en CPU con `--reuse-only` completa los activos con `computed-vectors.sqlite` y falla si todavía falta algún vector. `--shard K N` reparte las pasadas de CPU entre procesos acotados.
- Las anotaciones y los vectores calculados se confirman en SQLite por activo y por tandas de 512. Confirmar cada fila obligaba a esperar a que el disco sincronizara el registro, unos 12 ms por fila con la carga actual del equipo, y la recogida de la porción pasó de 1.043 s a 172 s al agruparlas.
- `--release-vectors` borra los PNG pendientes y los vectores de gráficos de `computed-vectors.sqlite` cuando todos los activos que los anotaron están confirmados. Los gráficos ya están en las muestras y los textos se conservan porque otros activos comparten noticias. Así el espacio adicional se limita al tramo de activos en curso.
- `edition_comparison.compare_editions` recorre las dos ediciones activo por activo con un registro reanudable. Cuenta sesiones nuevas y perdidas, registra cada gráfico que cambia y cualquier otra columna distinta, y mide el cambio de las ventanas comunes. Si los codificadores de las dos ediciones difieren, el vector de un mismo PNG puede cambiar y se registran su diferencia absoluta máxima, la relativa por componente y la relativa por norma.
- `--text-carry` nombra la edición cuyos textos se heredan con el contraste por activo descrito arriba. Se declara en la recogida y en la pasada final, y la pasada de GPU lo lee de la configuración.
- `factor_descriptor.describe_market_factors` fija el descriptor de factores de una preparación revisada, que se describe más abajo.

## Sustitución activo a activo

La v3 ocupa 57,8 GB en `samples/` y la v3.1 ocupará algo más, porque recupera ventanas. Con poco más de 40 GB libres no caben las dos. `--substitute-previous` y `--substitution-records` activan la sustitución de la v3 activo a activo:

1. La pasada que confirma un activo de la v3.1 llama a `edition_substitution.substitute_asset` justo después.
2. Se comprueba que las muestras nuevas tienen la huella de su recibo y que las muestras de la v3 tienen la del suyo.
3. `compare_asset` compara el activo con la v3. Solo se admiten los cambios declarados: sesiones nuevas, ventanas distintas, gráficos nuevos y vectores de gráficos recalculados por el cambio de codificador. La media de noticias debe coincidir exactamente, salvo en un activo cuyo contraste de textos falló y que los recodificó todos, donde su cambio se mide como el de los gráficos. Una sesión perdida o cualquier otra columna distinta detiene el recorrido.
4. El registro de la comparación, con las huellas de ambos activos, se escribe y se sincroniza en disco.
5. Solo entonces se borra el `samples.parquet` de la v3 de ese activo. Su manifiesto, su configuración y sus factores se conservan como constancia, igual que la configuración y el manifiesto de toda la edición.

Cualquier fallo de verificación lanza `SubstitutionError` fuera del registro de fallos por activo, así que el recorrido se detiene sin borrar nada. Un activo pendiente de GPU no está confirmado y no sustituye nada. Repetir el recorrido no vuelve a verificar un activo ya sustituido. Con `--min-free-disk-bytes`, ninguna pasada empieza un activo nuevo mientras el disco libre esté por debajo de la reserva, que para la v3.1 son 15 GB. La sustitución no toca las vistas, `targets-v3`, `prepared-accounting-v1` ni `embeddings.sqlite` de la v3. A partir del primer activo sustituido, la v3 deja de poder leerse como edición completa.

## Factor de mercado

El factor residual de Estados Unidos es SPY tal como lo deja la preparación. La revisión reescribe sus precios, así que el descriptor v2 deja de describir el archivo de la edición nueva. `factor_descriptor.describe_market_factors` crea un descriptor con identidad propia: apunta a los precios revisados de SPY, registra `number_parsing` y la tolerancia de orden, y guarda la huella del descriptor v2 sin abrir sus precios. El CSI300 chino no cambia y conserva la huella de su informe. La auditoría repite las comprobaciones del v2 (huella, precios finitos y positivos, OHLC coherente, sesiones crecientes del calendario anterior a 2024 y disponibilidad igual a la decisión) y aplica a SPY la tolerancia de redondeo que declaró la auditoría de precios. El CSI300 se audita sin tolerancia.

Aplicada a los descriptores v2, la auditoría nueva reproduce exactamente la del v2 en los dos mercados. Un ensayo con SPY revisado aparte, fuera de las rutas oficiales, recupera las dos sesiones que faltaban en el v2 (20 de julio y 29 de diciembre de 2000) y pasa de 6.035 a 6.037 filas. Cinco sesiones de 2000 quedan admitidas por redondeo, todas con un exceso relativo de entre 1,48e-16 y 1,65e-16, es decir, de una ULP. Tres de ellas ya estaban en el v2 porque el conversor de pandas movía algún precio una ULP y la fila quedaba ordenada por casualidad. Con la lectura exacta muestran el mismo desorden que las otras dos. El descriptor oficial se generará sobre la preparación v3.1 completa, cuando se ordene la regeneración.

## Medida sobre una porción

La medida usa 13 activos elegidos al azar con semilla 428 (8 de US y 5 de CN, incluido 600000.SS por su hueco de 2019), copiados con sus CSV fuera de las rutas oficiales. La v3 de esos activos se copió también, sin enlaces, para probar la sustitución sobre la copia. El equipo estaba compartido con otras cargas, con una carga media de 20 a 48 procesos en 16 hilos, así que los tiempos de reloj son pesimistas y los tiempos de CPU son más fiables. El recibo completo está en `reports/data/edition-v3-1-slice-20261010.json`.

| Pasada | Reloj | CPU | Resultado |
| --- | ---: | ---: | --- |
| Recogida (`--collect`) | 172 s | 156 s | 12 activos pendientes, 43.942 gráficos y 5.165 textos anotados, 1 activo sin muestras confirmado |
| GPU (`--encode-pending`), un gráfico por llamada | 684 s | | 43.942 gráficos en 221,5 s de llamadas (198 por segundo) y 5.165 textos en 134,8 s (38 por segundo) |
| Final con sustitución (`--reuse-only`) | 336 s | 176 s | 13 activos confirmados y sustituidos, 43.973 muestras |
| Repetición sobre activos confirmados | 65 s | 42 s | Nada que verificar ni borrar |
| Liberación (`--release-vectors`) | 7 s | 3 s | 43.942 gráficos y el registro de pendientes liberados |

La recogida cuesta 3,6 ms de CPU por muestra y la pasada final 4,0 ms, porque las dos dibujan el gráfico. La GPU estuvo ocupada un 18 % del tiempo de media: con un gráfico por llamada, el coste lo marcan la preparación en CPU y el lanzamiento de núcleos, no el cálculo. El asignador de Torch llegó a 188 MB (210 MB reservados) y el proceso a 360 MiB según `nvidia-smi`, con el contexto CUDA incluido. El reloj de la pasada de GPU incluye la carga de los modelos y un muestreo de `nvidia-smi` cada medio segundo que competía por la CPU, así que el caudal se toma del tiempo de las llamadas.

### Lotes de gráficos

Con un gráfico por llamada, la GPU pasa la mayor parte del tiempo esperando. Se midió en la misma GPU, con la llamada de producción y en FP32 estricto, cada tamaño de lote de 1 a 8 sobre 2.048 PNG reales de la porción. Todos los tamaños dieron vectores idénticos bit a bit a los de un gráfico por llamada, y el caudal subió de 326 a 805 gráficos por segundo. Con el modelo solo, lotes de 32, 64 y 128 ya no coinciden (hasta 9,2e-6 de diferencia absoluta) y tampoco son más rápidos que el de 8. Con TF32 activo en el mismo proceso, los 2.048 vectores difieren hasta 0,0052 y 8,5e-4 por norma, lo mismo que frente a la v3.

La porción se volvió a codificar entera con lotes de 8. Las 43.973 muestras de los 13 activos coinciden en todas sus columnas con las de la edición codificada de uno en uno. La pasada de GPU tardó 181 s de reloj, con 65,4 s de llamadas para 40.870 gráficos (625 por segundo, incluida la comprobación, porque los otros 3.072 ya se habían guardado) y 88,1 s para 5.165 textos, con menos carga en el equipo que la primera vez. El primer intento agotó el límite del asignador de 224 MiB que usaba la v3, y la pasada se reanudó con 1 GiB. El asignador llegó a 203 MB (264 MB reservados). La identidad del codificador registra el tamaño del lote, así que la edición se codifica entera con el mismo.

### Cambios frente a la v3

La porción pasa de 40.616 a 43.973 muestras (un 8,3 % más), sin perder ninguna sesión. Las 3.357 sesiones nuevas vienen de las filas recuperadas por la tolerancia y de las ventanas con el hueco de 2019. Hay 173 gráficos distintos en sesiones comunes. Dibujar la misma ventana con los precios de la v3 y con los de la v3.1 reproduce las dos huellas en todos los casos comprobados (BBW, CHRS, CKX y 600000.SS), así que el cambio se debe a la conversión exacta, que mueve algún precio lo justo para cambiar un píxel. Una fila recuperada no altera ventanas comunes, porque la v3 excluía cualquier ventana que la contuviera, y solo crea sesiones nuevas. Además, 497 ventanas comunes cambian como mucho 2,8e-14 por la misma conversión.

En las 40.443 sesiones comunes con el mismo PNG, ningún vector de gráfico coincide bit a bit con el de la v3. La diferencia absoluta máxima es 0,0070 sobre componentes de hasta 10,8 y la relativa por norma no pasa de 1,0e-3, con una mediana de 5,4e-4. Por componente, la mediana relativa es 7,6e-4 y el percentil 99 es 1,8e-2. Entre 214 y 342 componentes por activo valen cero en una codificación y no en la otra, porque sus activaciones quedan a un lado u otro del cero de la ReLU, y para ellos la diferencia relativa es 1. Ese es el efecto de calcular las convoluciones con TF32, y es la razón para recalcular todos los gráficos.

Los vectores de noticias recalculados en FP32 estricto dan agregados idénticos bit a bit a los de la v3 en las 40.443 sesiones comunes. Uno a uno, los 5.051 textos que tienen las dos ediciones coinciden bit a bit, y otros 114 textos de la v3.1 no estaban en la v3. Es coherente con que el modelo de texto no use cuDNN, y es la medida en la que se apoya la herencia de textos.

### Textos heredados

La porción se volvió a codificar con `--text-carry` sobre la v3 real, con lotes de 8 y la sustitución sobre otra copia de la v3.

| Pasada | Reloj | CPU | Resultado |
| --- | ---: | ---: | --- |
| Recogida | 163 s | 136 s | 5.051 textos heredables, 165 en la muestra (3,3 %), y 279 textos pendientes de GPU contando los 114 nuevos |
| GPU | 81 s | | 43.942 gráficos en 60,7 s de llamadas y 279 textos en 3,3 s. Los 12 activos con textos superan el contraste y ningún texto difiere |
| Final con sustitución | 150 s | 141 s | 13 activos confirmados y sustituidos, 43.973 muestras |

Las 43.973 muestras coinciden en todas sus columnas con las de la edición que recodificó todos sus textos, y la sustitución no encontró ninguna diferencia no declarada. La recogida apenas cambia de coste, porque buscar un texto en la v3 es una consulta por clave. La GPU codifica un 5,4 % de los textos que codificaba sin herencia.

### Disco

| Concepto | Porción | Por muestra |
| --- | ---: | ---: |
| Muestras v3.1 | 149,2 MB | 3.393 B |
| Muestras v3 sustituidas | 138,3 MB | 3.405 B |
| Registro de pendientes | 113,6 MB | 2.584 B |
| Vectores calculados | 203,8 MB | 4.635 B |
| Textos heredables en el registro de pendientes | 13,5 MB | 2.669 B por texto |

El registro de pendientes y los gráficos calculados son temporales. Tras la liberación, el archivo de vectores calculados conserva su tamaño, pero SQLite reutiliza las páginas libres en el siguiente tramo. De la copia de la v3 quedaron 0,76 MB de manifiestos, configuraciones y factores como constancia.

### Proyección a la edición completa

La simulación de la revisión cuenta 18.698.976 ventanas (15.874.289 en US y 2.824.687 en CN), y la v3 tiene 1.591.529 textos distintos. Con los costes de la porción, donde los intervalos recogen la diferencia entre las dos cargas del equipo:

| Concepto | Proyección |
| --- | ---: |
| CPU de la recogida | de 16,8 a 18,5 h |
| CPU de la pasada final | de 18,5 a 20,7 h |
| GPU para gráficos en lotes de 8 | 8,3 h |
| GPU para gráficos, uno por llamada | 26,2 h |
| GPU para textos sin herencia | de 7,5 a 11,5 h |
| GPU para textos heredados con contraste | menos de 1 h |
| Muestras v3.1 | 63,5 GB, 5,7 GB más que la v3 |
| Vectores de texto que se conservan, con herencia | unos 0,2 GB |
| Temporales por activo en curso | unos 27 MB |

La estimación de los textos heredados supone la misma proporción que en la porción, donde un 5,4 % de los textos pasó por la GPU entre la muestra y los textos nuevos. Un activo con pocos textos contrasta una proporción mayor, porque la muestra nunca baja de 8. Sin sustitución, la v3.1 no cabe junto a la v3 en el disco libre. Con sustitución y tramos de unos 300 activos, el espacio ocupado crece como mucho unos 15 GB sobre el actual (8,4 GB de temporales del tramo, unos 0,3 GB de textos heredables pendientes, 0,2 GB de textos calculados y 5,7 GB de crecimiento neto). Las pasadas de CPU se reparten entre procesos y pueden solaparse con la GPU. Con lotes de 8 y los textos heredados, la GPU necesita unas 9 h, y las dos pasadas de CPU suman de 35 a 39 h de CPU. Repasar un activo ya confirmado cuesta hasta unos 3 s de CPU, por lo que cada tramo recorre solo sus activos con `--shard` en lugar de repasar los anteriores.

## Pruebas

Las pruebas cubren el contrato, la admisión en el codificador, la lectura desde el corpus, las vistas temporales y el corpus ordenado, las familias, M3, la vista de información, la GRU nativa y las herramientas. Para la precisión comprueban que cada uno de los tres ajustes detiene el codificador antes de abrir CUDA, que la identidad calculada en CPU es la del codificador cargado y que una edición con TF32 no cede vectores. La sustitución se prueba con cada fallo de verificación (huella anterior, huella nueva, columna cambiada y sesión perdida), con un registro que no se puede escribir, con un activo pendiente de GPU y con la reserva de disco, y en todos los casos la v3 queda intacta. El descriptor de factores repite cada comprobación del v2 y prueba que la tolerancia de SPY no se aplica al CSI300. La herencia de textos se prueba con una edición de origen que registra TF32 en cuDNN: si la muestra coincide, la GPU solo codifica la muestra y las muestras finales son las de la v3. Si los textos difieren en el último bit, el activo recodifica todos sus textos, ninguno sale de la v3, las muestras coinciden con las de una codificación en FP32 estricto y la sustitución mide el cambio de la media de noticias en lugar de detenerse. También se prueban la posición de la muestra, la unión de muestras al recoger de nuevo un activo o al repartirlo entre fragmentos, la espera del contraste mientras queden pendientes, la preferencia por un vector calculado en FP32 estricto, una edición de origen modificada y las identidades incompatibles. La mutación dirigida se resume en la PR.

## Pendiente

Falta ejecutar la auditoría con la tolerancia, la revisión de precios, el descriptor de factores sobre la preparación completa, la codificación, `targets-v3.1` y la verificación completa (verificador de la edición, `verify-targets` y comparación con la v3). El plan acordado recorre la edición en tramos con `--shard`, gráficos en lotes de 8 con la comprobación de uno de cada 64, un límite del asignador de 1 GiB, textos heredados con contraste, sustitución con una reserva de 15 GB y liberación de los temporales tras cada tramo, dejando libre la GPU entre tramos. La ejecución espera la orden del coordinador.
