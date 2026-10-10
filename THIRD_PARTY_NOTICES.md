# Material de terceros

La licencia MIT cubre únicamente los derechos que corresponden al autor del repositorio. No modifica la licencia, las condiciones de acceso ni los derechos sobre datos, artículos, libros, marcas o documentos académicos ajenos.

| Material | Procedencia | Tratamiento |
| --- | --- | --- |
| FinMultiTime | Autores y repositorio del dataset. Véase su ficha | Copia local en `dataset/`. Conservar procedencia y comprobar derechos de noticias y datos subyacentes antes de redistribuir. Una etiqueta MIT en la ficha no acredita derechos de cada proveedor. |
| Artículos y libros | Catálogos de `docs/references/` | Enlaces y reseñas originales versionados. PDF de consulta en `library/`, ignorados por Git. Acceso gratuito no equivale a permiso de redistribución. |
| Documento inicial MARS-TITAN | Documento propio aportado por Gonzalo García Lama | Antecedente conservado en `docs/research/source/`. Se mantiene sin modificar para distinguir la propuesta inicial de sus revisiones. |
| Dependencias | Proyectos de terceros en `uv.lock` | Sus licencias se mantienen. Revisarlas antes de distribuir un entorno, imagen o aplicación derivada. |
| uPlot 1.6.32 | Paquete npm [`uplot@1.6.32`](https://www.npmjs.com/package/uplot/v/1.6.32), de Leon Sorokin | Copia sin modificar de `dist/uPlot.iife.min.js` y `dist/uPlot.min.css` en `site/vendor/uplot/`, con su [licencia MIT](site/vendor/uplot/LICENSE). El observatorio la sirve desde su propio origen, sin CDN. |
| Atkinson Hyperlegible Next y Mono | Paquetes `@fontsource-variable/atkinson-hyperlegible-next@5.3.0` y `@fontsource-variable/atkinson-hyperlegible-mono@5.3.0`. Copyright de los autores de los proyectos Atkinson Hyperlegible Next y Mono | Subconjunto latino variable en `site/fonts/`, sin modificar. Licencia SIL Open Font License 1.1 en `site/fonts/ofl-atkinson-hyperlegible-next.txt` y `site/fonts/ofl-atkinson-hyperlegible-mono.txt`. |
| Newsreader | Paquete `@fontsource-variable/newsreader@5.3.0`. Copyright de The Newsreader Project Authors | Subconjunto latino con eje óptico en `site/fonts/`, sin modificar. Licencia SIL Open Font License 1.1 en `site/fonts/ofl-newsreader.txt`. |
| Escala cividis | Nuñez, Anderton y Renslow (2018), [doi:10.1371/journal.pone.0199239](https://doi.org/10.1371/journal.pone.0199239), muestreada de matplotlib 3.11 | `site/charts.mjs` guarda 17 colores de la escala e interpola entre ellos. Se cita la fuente en el código y en la documentación del observatorio. |
| DLinear | [LTSF-Linear, versión 0c113668](https://github.com/cure-lab/LTSF-Linear/tree/0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6) | El módulo `src/mars_titan/models/baselines/dlinear.py` adapta su descomposición y sus mapas temporales. Conserva el aviso original y la [licencia Apache 2.0](docs/licenses/dlinear-apache-2.0.txt). La fusión multimodal y la tarea residual se documentan como adaptación, no reproducción del benchmark original. |

No se eluden muros de pago, autenticación ni restricciones de acceso. Los libros comerciales se consultan mediante biblioteca, préstamo o adquisición legítima. Si una URL deja de permitir la descarga, se registra el fallo y se conserva la referencia bibliográfica.
