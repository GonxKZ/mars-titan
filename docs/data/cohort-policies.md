# Cohortes de datos y preparación

La comparación distingue dos niveles de evidencia. La presencia de un archivo
no acredita su contenido editorial ni su disponibilidad histórica.

| Cohorte | Texto admitido | Afirmación permitida |
| --- | --- | --- |
| `original_audited` | Cuerpo o resumen distribuido, con controles estructurales, identidad y fecha declarada. | Reproducción del corpus original auditado, no verificación externa. |
| `externally_verified` | Cuerpo completo contrastado mediante la revisión ligada al registro. | Coincidencia con la evidencia editorial registrada, con sus límites históricos. |

Los resúmenes no se renombran como artículos completos. La cohorte original
conserva el tipo de contenido y señala títulos ausentes y zonas horarias no
acreditadas. Cuando falta el símbolo dentro del registro, su asociación procede
del archivo asignado por el inventario y queda identificada como tal. Una
contradicción explícita entre símbolo del registro e instrumento se rechaza.

Los rechazos editoriales demostrados se mantienen en ambas cohortes. Un caso
no verificable no equivale a contenido falso. Puede pertenecer a la cohorte
original, pero no adquiere por ello evidencia externa.

## Disponibilidad temporal

Una fecha declarada sin hora se retrasa al cierre de la siguiente sesión. En la
cohorte original, una hora sin zona se conserva como dato de origen y solo se
utiliza su fecha, con ese mismo retraso. No se inventa una zona ni una hora
precisa. La incertidumbre temporal queda en `quality_flags` y la regla se
identifica como `declared_date_without_verified_timezone`.

Este tratamiento pertenece al benchmark original y no demuestra cuándo se
publicó el texto. La cohorte estricta sigue rechazando horas sin zona acreditada.
Los offsets inválidos y la precisión temporal no soportada no se reparan
silenciosamente. Los registros posteriores al corte y los que solo estarían
disponibles después se conservan como exclusiones.

La preparación contable no cambia esta distinción. Cada cifra estadounidense
usa su presentación interna. Los periodos contables chinos sin divulgación
acreditada siguen pendientes. Ni `period_end` ni la fecha del contenedor son una
publicación válida por sí solos.

## Almacenamiento y recuperación

Las noticias se leen por registro. Un registro mayor de 1 MiB queda excluido con
su localizador, sin perder el siguiente. La deduplicación y la ordenación usan
SQLite temporal en disco, con caché de 8 MiB. Los Parquet se escriben por bloques
acotados. La cuota predeterminada del archivo SQLite es de 2 GiB por activo.
No incluye un límite exacto sobre su journal o los archivos temporales del sistema.

Los nombres originales pueden contener bytes que no representan UTF-8 válido.
Los recibos editoriales de versión 2 guardan `source_file` con escape porcentual
de los bytes del sistema de archivos. `source_file_encoding` identifica esa
regla. `unquote_to_bytes` recupera los bytes exactos, incluidos porcentajes
literales. Las huellas siguen calculándose sobre los originales. No se renombra
el archivo ni se intenta adivinar la codificación de su nombre. Esta regla se
aplica tanto a noticias admitidas como a exclusiones.

Cada activo conserva precios, fundamentales, noticias, exclusiones, fuentes y
recibo. La preparación comprueba sus hashes antes de reutilizarlo. Un cambio de
cohorte, fuentes, calendario o implementación necesita otra edición. Las
interrupciones permiten retomar activos confirmados y rehacer el activo incompleto.

El recorrido congela primero una instantánea de las revisiones editoriales.
Una verificación posterior no modifica esa edición. El catálogo completo incluye
activos incompletos y fallidos, sin filtro de sector ni límite de empresas.

Desde la raíz del repositorio:

```bash
uv run --no-sync python -m mars_titan.data.corpus_preparation \
  --source dataset \
  --inventory data/interim/source-inventory-v3-final-20260922.sqlite \
  --reviews data/interim/editorial-reviews-v2-20260922.sqlite \
  --output data/processed/original-audited-20260923 \
  --cohort original_audited --market all
```

`--stop-after-assets` permite una pausa operativa. No define la muestra final.
Repetir la orden sin ese corte vuelve a comprobar los activos y conserva los
artefactos válidos. Un fallo de un archivo no oculta los restantes activos.

`completed` significa que terminó el recorrido de preparación. No acredita que
todas las empresas tengan muestras ni que haya terminado un entrenamiento.
La intersección temporal, las representaciones, las etiquetas y los modelos
tienen comprobaciones posteriores. Las ediciones anteriores se conservan con
sus políticas originales.
