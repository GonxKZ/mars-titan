# Verificación recuperable de noticias

El registro editorial consume el [índice completo del corpus](corpus-catalog.md)
sin cambiar los archivos originales. Una fila del índice conserva su estado
aunque falte la URL, contenga un resumen o pertenezca al periodo reservado.
La cola no limita empresas ni selecciona noticias por sus retornos posteriores.

El adaptador actual lee páginas públicas de The Motley Fool. El título, la
fecha, la mención explícita del instrumento y todo el cuerpo editorial deben
coincidir con el registro. Un enlace candidato, una coincidencia de titular o
una página accesible no bastan para admitirlo.

## Contraste y procedencia

La comparación normaliza Unicode NFC y espacios. Conserva mayúsculas, palabras,
su orden, cifras, signos y unidades. Una negación añadida, una cifra distinta o
un párrafo truncado impiden verificar el cuerpo. Los intervalos admitidos señalan
caracteres del original y llevan una huella SHA-256. No se reescribe el artículo.

Solo se pueden retirar las plantillas contrastadas de Stock Advisor y el aviso
final conocido del distribuidor. Los marcadores publicitarios no bastan. El
bloque completo debe coincidir con una de cinco huellas de plantilla obtenidas
de las revisiones existentes. Se parametrizan únicamente el nombre repetido de
la empresa y la fecha de la promoción. Las declaraciones de posiciones del autor y del medio
se conservan. Las estructuras desconocidas no autorizan a eliminar párrafos.

El lector del editor separa las cotizaciones actuales insertadas en componentes
identificados del texto publicado. Conserva las tablas editoriales. Comprueba la
URL canónica y la identidad del artículo en sus metadatos. Exige una declaración
editorial reconocible. No da por completo un vídeo sin transcripción contrastada.
Los cambios de estructura que no encajan quedan pendientes de otro adaptador o
de revisión, sin recurrir a una extracción aproximada.

La correspondencia con una página actual no demuestra que su contenido estuviera
disponible exactamente así en el pasado. `historical_version_verified` permanece
en `false`. Una modificación declarada en un día posterior al de publicación
impide la admisión automática. Las modificaciones del mismo día conservan su
fecha en la evidencia, pero no se presentan como una certificación histórica.
La disponibilidad temporal sigue las [reglas de noticias](news-policy.md).

## Uso local

Crear la cola con todos los localizadores del índice y las revisiones existentes:

```bash
uv sync --locked --inexact --extra data
uv run --no-sync mars-data news-queue \
  --catalog data/interim/corpus-news-20260922.sqlite \
  --database data/interim/editorial-reviews.sqlite \
  --manual-reviews data/manifests/news-reviews.json
```

Consultar el estado o contrastar capturas ya descargadas, sin conexión:

```bash
uv run --no-sync mars-data news-status --database data/interim/editorial-reviews.sqlite
uv run --no-sync mars-data news-verify \
  --database data/interim/editorial-reviews.sqlite \
  --evidence data/external/editorial-evidence
```

`--network` permite consultar el editor autorizado. `--stop-after` interrumpe una
pasada tras ese número de decisiones confirmadas. No cambia el universo de la
cola ni constituye una muestra científica. Al repetir la orden se conservan los
resultados anteriores y se respetan los plazos de reintento. Las hipótesis de URL
derivadas de una publicación sindicada siguen necesitando el contraste completo.

| Estado | Significado |
| --- | --- |
| `pending` | El registro todavía no ha pasado por el verificador. |
| `retry` | Falta una captura o la consulta no pudo completarse. Tiene un plazo de reintento. |
| `needs_provenance` | Falta identificar o recuperar una fuente editorial compatible. |
| `verified_full_article` | Coinciden el cuerpo completo y los campos de identidad exigidos. |
| `unverifiable` | La evidencia consultada no permite verificar el registro. No implica falsedad. |
| `rejected` | Se conserva un rechazo explícito del registro de revisiones. |
| `reserved` | No se procesa para seleccionar modelos porque pertenece al periodo reservado. |

Los demás motivos del índice, como ausencia de cuerpo o registro demasiado
grande, también se conservan. `training_ready` no se activa por completar una
pasada editorial. Faltan todavía la admisión temporal y el cruce con las otras
modalidades.

## Recuperación y límites

SQLite confirma cada decisión por separado. Un corte no convierte una decisión
parcial en una revisión completa. La continuación comprueba la huella del índice
y del código de la política. Un índice con cambios WAL sin consolidar se rechaza.
Cada registro leído se contrasta con sus huellas de bytes y de identidad textual.
Si cambian los originales se interrumpe el proceso, sin etiquetar el cambio como
una noticia falsa.

Las revisiones manuales se importan sin alterar sus notas, estados o evidencias.
Una revisión del periodo reservado no se devuelve como entrada de entrenamiento.
`reviews_for_asset` consulta un activo y un mercado, con un máximo de 128 MiB de
JSON. No se construye un diccionario con millones de revisiones. El recorrido de
la cola mantiene bloques de 128 localizadores y lee un cuerpo cada vez.

Las capturas se guardan comprimidas y con fecha y SHA-256 en una base separada.
El límite predeterminado de esa base es 2 GiB, configurable con `--quota-bytes`.
No incluye la base de revisiones, los archivos temporales de SQLite ni los
originales. Una respuesta no puede superar 4 MiB y un registro original tiene
un presupuesto de lectura de 1 MiB. La captura se vuelve a comprobar al leerla.
No se borran evidencias para hacer sitio automáticamente.

El cliente acepta únicamente HTTPS del editor implementado. Resuelve direcciones
públicas y fija la dirección de conexión, manteniendo la validación TLS. No usa
proxies ni sigue redirecciones automáticamente. Conserva los plazos de consulta
y los fallos entre ejecuciones. Aplica el intervalo más restrictivo entre la
política del editor y los cinco segundos predeterminados. Los códigos 403, 429 y
503 no se sortean. `Retry-After` puede aplazar nuevas consultas. No se ejecutan
solicitudes simultáneas al mismo archivo de caché.

Las reglas de rutas de `robots.txt`, incluidos sus comodines, se interpretan con
[Protego](https://github.com/scrapy/protego). El contenido de la política y sus
fechas se confirman en una misma transacción. El plazo de socket no constituye
un límite estricto de duración para la resolución DNS y la recepción completa de
cabeceras. No se ha medido ese extremo con una simulación de transporte lento.

La cobertura obtenida depende de que las páginas sigan accesibles y de disponer
de adaptadores contrastados. Ese sesgo debe acompañar a cualquier evaluación.
El registro mantiene visibles las fuentes sin resolver, incluidas las noticias
chinas resumidas. No las transforma en artículos completos ni las elimina del
objetivo del corpus.

La estructura HTML observada se interpreta con [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/).
Los permisos de consulta proceden de la [política del editor](https://www.fool.com/robots.txt),
que el cliente vuelve a consultar al caducar su copia local.
