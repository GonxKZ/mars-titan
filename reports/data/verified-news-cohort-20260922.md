# Ampliación de noticias y muestra multimodal

La revisión del 22 de septiembre de 2026 incorpora ocho artículos completos y
mantiene un registro nuevo como no verificable. El manifiesto pasa de 36 a 45
revisiones: 21 admitidas, 13 no verificables y 11 rechazadas. No se sustituyen
los cuerpos originales ni se inventan noticias para completar fechas.

El contraste de cada artículo incluye texto editorial, tablas o transcripciones
publicadas cuando existen, declaraciones de posiciones, fecha y relación con
la empresa. La coincidencia con una página actual no demuestra una versión
histórica inmutable. Las ocho admisiones conservan esa limitación explícita.

## Fuentes contrastadas

| Activo | Fecha local | Fuente del editor | Decisión |
| --- | --- | --- | --- |
| MSFT | 2022-08-05 | [Resultados y valoración de Microsoft](https://www.fool.com/investing/2022/08/05/is-microsoft-stock-a-buy/) | Admitida, incluida la tabla completa. |
| NVDA | 2023-08-18 | [Participación de Appaloosa en Nvidia](https://www.fool.com/investing/2023/08/18/billionaire-david-tepper-boosts-stake-in-nvidia/) | Admitida. |
| AMZN | 2023-05-03 | [Resultados de Amazon](https://www.fool.com/investing/2023/05/03/3-things-smart-investors-learned-amazons-q1-report/) | Admitida. |
| MSFT | 2022-11-11 | [Compra de Activision](https://www.fool.com/investing/2022/11/11/activision-blizzard-stock-is-waiting-for-the-micro/) | No verificable como artículo completo, vídeo sin transcripción. |
| MSFT | 2022-09-13 | [Microsoft, Intel y HP](https://www.fool.com/investing/2022/09/13/why-intel-microsoft-and-hp-stocks-flopped-today/) | Admitida, con información propia sobre Microsoft. |
| MSFT | 2022-07-24 | [Negocio y valoración de Microsoft](https://www.fool.com/investing/2022/07/24/is-now-a-good-time-to-buy-microsoft-stock/) | Admitida, con transcripción textual completa. |
| NVDA | 2022-06-17 | [Negocio y valoración de Nvidia](https://www.fool.com/investing/2022/06/17/4-compelling-reasons-nvidia-is-my-newest-stock-buy/) | Admitida, con transcripción textual completa. |
| NVDA | 2022-01-31 | [Evolución y negocio de Nvidia](https://www.fool.com/investing/2022/01/31/if-invested-10000-nvidia-1999-how-much-today/) | Admitida. |
| NVDA | 2022-01-20 | [GPU y portátiles de Nvidia](https://www.fool.com/investing/2022/01/20/why-nvidia-could-crush-amd-once-again-in-2022/) | Admitida. |

Los tres primeros artículos proceden de una búsqueda dirigida previa. Los seis
siguientes se eligieron antes del contraste, tomando los tres primeros registros
por activo de colas ordenadas mediante `sha256("42:" + source_record_hash)`.
El periodo de esas colas termina en 2022. No se usaron rendimientos ni resultados
de los modelos para seleccionar artículos o sustituir el caso no admisible.

La cola adicional tenía 110 candidatos de MSFT y 465 de NVDA después de las
exclusiones previas. Sus filtros de longitud, entidad y editor sirven para
priorizar la revisión, no para aprobar una noticia. Quedan 569 registros de esas
colas sin contrastar. No se extrapola la tasa de admisión al conjunto original.

Las publicaciones y modificaciones declaradas de las páginas admitidas pertenecen
al día de la fecha local. Se conserva la fecha sin hora del proveedor y el retardo
por sesiones. Una fecha de grabación no reemplaza la publicación de su transcripción.
No se ha comprobado que estas transcripciones recojan cada palabra del vídeo.

## Intersección materializada

El [panel de ejecución](../../data/manifests/verified-cohort-20260922.json) incluye
los cuatro activos de la preparación anterior y las tres empresas incorporadas.
El sector actual no interviene como filtro. El conjunto pasa de 65 a 105 muestras,
con 80 etiquetas de entrenamiento y 25 de validación. Las 65 muestras anteriores
mantienen exactamente todos sus campos y representaciones.

| Activo | Entrenamiento | Validación | Total |
| --- | ---: | ---: | ---: |
| ABM | 0 | 0 | 0 |
| CSGS | 0 | 0 | 0 |
| DECK | 25 | 10 | 35 |
| MNST | 25 | 5 | 30 |
| MSFT | 15 | 0 | 15 |
| NVDA | 15 | 5 | 20 |
| AMZN | 0 | 5 | 5 |
| Total | 80 | 25 | 105 |

AMZN solo aparece en validación y MSFT solo en entrenamiento. El contraste
incluye así activos vistos y no vistos durante el ajuste. Sus errores se
desglosan por separado y no se interpreta el promedio como una muestra
representativa de todo el mercado.

Cada muestra contiene una ventana de 64 sesiones y cinco canales de precios,
384 posiciones textuales, 512 de gráficos, 45 de fundamentales y 420 de contexto
macro. Los bloques numéricos conservan máscaras y antigüedad. Se exige al menos
un hecho fundamental y un indicador macro observados, no que todas las variables
del catálogo estén disponibles. Los codificadores permanecen congelados y el
texto se procesa por fragmentos completos, sin recortarlo a un titular.

El [recibo de comprobación](verified-news-cohort-20260922.json) conserva huellas,
recuentos, auditoría de factores, tamaños y límites. Se comprobaron las fechas de
disponibilidad, los vectores finitos, las dimensiones y las 105 ventanas de precios.
Las etiquetas se obtuvieron exclusivamente en los tramos permitidos. No hay
muestras admitidas fuera de entrenamiento o validación y no se ha abierto el
test final.

## Recursos y límites

La preparación y codificación tardaron 17,60 segundos dentro de la medición y
19,54 segundos de [proceso completo](../resources/news-cohort-process-time.txt). La lectura interna de RSS al cerrar el
recibo fue 1.979,41 MiB. Los derivados nuevos se almacenan en Parquet y se leen
por bloques. Los originales y los conjuntos anteriores permanecen intactos.

Los archivos de origen del panel suman 443.114.244 bytes. Sus tablas de muestras
ocupan 494.765 bytes, pero no representan una compresión íntegra de esas fuentes.
Contienen solo la intersección admitida y sus representaciones. Excluyen los
históricos de precios, hechos contables, contexto macro compartido, cachés y pesos.
Las ventanas de precios se reconstruyen durante la lectura.

El contenido lógico float32 de una entrada suma 6.724 bytes. Un lote de 16, con
una etiqueta float32 por muestra, suma 107.648 bytes. Estas cifras no incluyen
objetos Python, activaciones, gradientes, optimizador, bibliotecas o reservas del
dispositivo. No se utilizan como sustitución del pico de RAM o VRAM medido.

Esta ampliación no completa la auditoría del corpus ni amplía la cobertura
anterior al corte de 2018 del selector de piloto. Las fechas, las exclusiones y
los límites de preentrenamiento de los codificadores siguen siendo relevantes.
Los resultados de esta cohorte deberán compararse entre modelos entrenados sobre
ella, sin mezclar sus métricas con las obtenidas sobre las 65 muestras anteriores.
