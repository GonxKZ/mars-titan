# Observatorio de MARS-TITAN

La página estática consulta el registro público de campañas y permite importar un JSON local. Muestra origen, actividad, avance observado, recuperación y métricas de validación. La importación permanece en memoria y no se envía a ningún servidor.

El contrato de versión 2 divide el historial en páginas de 64 registros, cargadas bajo demanda. La interfaz también admite las instantáneas anteriores de versión 1. Cada archivo tiene un límite de 8 MiB, cada página admite hasta 128 registros y cada curva hasta 500 puntos. Los valores desconocidos son `null`.

Las comparaciones predictivas exigen la misma población y definición de medida. La generación sintética y la evaluación financiera tienen vistas separadas. La pérdida de un optimizador financiero no se interpreta como MAE. Las fases reservadas y el test permanecen ocultos. El productor excluye estos datos antes de publicarlos, porque ocultarlos en pantalla no protegería el archivo descargable.

La fecha del último progreso, la observación del proceso y la preparación del paquete publicado son distintas. Un proceso sin una observación reciente se muestra como «Sin actualización». La interfaz no deduce que se haya pausado. El navegador consulta cada 60 segundos mientras la pestaña está visible y conserva la última página válida si la red o el contrato fallan.

El recolector local observa las campañas cada 15 segundos y publica los cambios ordinarios cada cinco minutos. Los estados terminales tienen prioridad. Pages sirve un paquete estático identificado por los SHA del frontend y de los datos, sin ejecutar entrenamientos ni recolección en Actions. La [guía del recolector](https://github.com/GonxKZ/mars-titan/blob/develop/docs/engineering/campaign-observatory.md) describe fuentes, límites y recuperación.

## Código y comprobaciones

`state.mjs` valida contratos y resuelve visibilidad, comparación y CSV. `app.js` gestiona controles, peticiones y DOM. `index.html` y `styles.css` definen la presentación adaptable. No hay servidor de aplicación, bibliotecas remotas, fuentes descargadas ni telemetría.

```bash
node --test site/tests/*.test.mjs
uv run --no-sync python -m http.server 8000 --bind 127.0.0.1 --directory site
```

El servidor permite abrir `http://127.0.0.1:8000`. Se termina con `Ctrl+C`. Los módulos y las peticiones requieren un origen HTTP.

Las pruebas de navegador de `site/tests/` usan una instalación local de Playwright y Chrome. Comprueban escritorio, móvil, teclado, importación local, separación de actividades, carga paginada y conservación de la página anterior tras un HTTP 503. Los datos ficticios de estas pruebas no se publican como resultados científicos.

`display-browser.mjs` comprueba anchos entre 320 y 1440 px, solapes del catálogo, etiquetas estables al paginar, magnitudes sin duplicar y selección de la medida dentro de los 240 puntos visibles. Recibe como primer argumento la ruta de `playwright/index.mjs` y admite un segundo argumento para guardar cobertura V8. El [informe de revisión](../reports/observatory-display-review.json) registra las pruebas y sus límites.

Trabajo vinculado a [MT-068](https://github.com/GonxKZ/mars-titan/issues/70).
