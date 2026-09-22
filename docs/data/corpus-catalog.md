# Catálogo completo e índice de noticias

El catálogo conserva todos los instrumentos del inventario de cada mercado,
incluidos los que carecen de alguna fuente. No selecciona por sector actual ni
impone un número máximo de empresas. La presencia de precios, noticias,
fundamentales y gráficos se registra por separado de su admisión para entrenar.

El índice textual recorre los archivos de ambos mercados y conserva su relación
con el activo. También permite contar únicamente los registros de instrumentos
con las cuatro fuentes, sin ocultar las noticias que todavía no tienen otra
modalidad con la que combinarse.

## Almacenamiento y recuperación

SQLite conserva el archivo de origen, ordinal, desplazamiento, longitud, huellas,
fecha declarada, título y URL cuando existen. No copia los cuerpos del dataset.
El texto se puede recuperar posteriormente desde su posición, contrastando su
huella. Las rutas se guardan como bytes para conservar nombres que no son UTF-8.

Un hash identifica los bytes exactos del registro, incluidos sus terminadores.
Otro conserva la identidad textual utilizada por el lector anterior, sin el
BOM inicial ni los terminadores. Los cuerpos tienen su propia huella. Compartir
URL no equivale a compartir cuerpo, fecha, empresa o revisión.

Cada archivo se procesa en una transacción. Solo se confirma tras comprobar
su hash completo y que no haya cambiado durante la lectura. Una interrupción
revierte los registros del archivo pendiente y conserva los anteriores.
Al continuar se comprueban otra vez las huellas de las fuentes ya confirmadas.

La continuación rechaza cambios de origen, catálogo, corte, presupuesto de
registro o implementación del indexador. Otra política necesita un índice nuevo.
No se sobrescribe una base de datos ajena ni se escribe dentro de los originales.

Las lecturas limitan cada registro a un MiB por defecto. Si lo supera, se recorre
sin acumularlo y se conservan posición, longitud, hash y motivo de exclusión.
La caché de páginas SQLite se configura en 16 MiB. Esto no es una cota de todo
el RSS, que incluye metadatos del catálogo, intérprete y buffers. El estado
agregado utiliza recibos por archivo, sin volver a leer todos los artículos.

## Estados y límites

`pending_verification` significa que existen campos para contrastar un artículo,
no que su contenido esté verificado. `needs_provenance` conserva los registros
que necesitan información adicional, incluidos los resúmenes chinos sin URL.
`reserved` separa las fechas declaradas posteriores al corte. Los registros
inválidos, demasiado grandes o con símbolo contradictorio tienen otro motivo
explícito. Un JSON con claves repetidas no se interpreta escogiendo el último
valor.

Los errores del inventario se conservan en `source_errors`, con archivo,
modalidad y motivo. Una fuente textual fallida sigue contando entre los archivos
esperados y evita declarar completo el índice. Los errores de otras modalidades
no borran las noticias ya indexadas. Las órdenes devuelven código 2 cuando existen
errores de fuentes, aunque el recorrido textual haya terminado.

El corte del índice se aplica a la fecha declarada. No acredita hora, zona ni
disponibilidad. Esas comprobaciones siguen siendo necesarias para admitir una
muestra. El índice no genera etiquetas ni abre la reserva de evaluación.

`index_complete` solo acredita que se han confirmado todos los archivos
textuales del catálogo. `training_ready` permanece falso. Todavía se necesita
contraste editorial, disponibilidad contable, ventanas válidas, contexto macro
y etiquetas maduras. Los 140 indicadores del catálogo y las cuatro modalidades
no se sustituyen por esta tabla de localizadores.

## Órdenes locales

El inventario debe pertenecer a la misma copia de datos y tener identidades
coherentes con sus rutas. Las siguientes órdenes no seleccionan un panel:

```bash
uv run --no-sync mars-data corpus-index \
  --source dataset \
  --inventory data/interim/source-inventory-v3-final-20260922.sqlite \
  --database data/interim/corpus-news-20260922.sqlite

uv run --no-sync mars-data corpus-status \
  --database data/interim/corpus-news-20260922.sqlite
```

Repetir la primera orden continúa desde los archivos confirmados. Un informe
adicional se guarda mediante `--report` en una ruta nueva, fuera de los originales.
Las órdenes no descargan noticias, no verifican su contenido editorial y no
entrenan modelos.
