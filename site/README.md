# Observatorio de entrenamiento de MARS-TITAN

Autor: Gonzalo García Lama. Desarrollo asociado a [MT-068](https://github.com/GonxKZ/mars-titan/issues/70).

Esta interfaz estática permite consultar resúmenes de ejecuciones e importar un JSON local sin enviarlo a ningún servidor. El estado inicial no contiene ejecuciones ni resultados. La interfaz no entrena modelos, no controla el ordenador y no sustituye el registro experimental.

## Plan de diseño

La cronología de una ejecución organiza la pantalla. El estado observado queda separado del protocolo conceptual y de los modelos planificados. La información se alinea a la izquierda, con una anchura de lectura limitada y espacio suficiente entre grupos.

| Token | Valor | Uso |
| --- | --- | --- |
| Papel | `#ffffff` | Fondo y lectura principal |
| Plano | `#f2f6fb` | Cronología y zonas de contexto |
| Tinta | `#172b46` | Texto y datos |
| Secundario | `#526780` | Explicaciones y procedencia |
| Azul | `#335fe6` | Acciones y selección |
| Petróleo | `#177b72` | Estado confirmado y método |

Los títulos usan Noto Serif o una serif local equivalente. El texto y los números usan Noto Sans o una sans local equivalente, con cifras tabulares. No se descargan fuentes. La escala propuesta es 14, 16, 20, 26 y 34 píxeles, con adaptación a pantallas pequeñas.

Se han contrastado dos composiciones. Una cuadrícula de tarjetas repetiría contenedores sin aportar información. Se elige una cronología horizontal y un registro de lectura continua, con detalle lateral solo cuando existe una ejecución.

```text
MARS-TITAN                 Actualizar · Importar JSON
Seguimiento   Comparativa   Método   Historial

Estado del registro                   Origen y fecha
Preparación → Entrenamiento → Validación → Evaluación
Avance y curva observada               Recuperación

Comparación del mismo grupo y fase
Método, fórmulas y modelos planificados
Historial consultable y exportación
```

La revisión previa al desarrollo ha eliminado indicadores gigantes sin datos, porcentajes de acierto y curvas de ejemplo. La cronología es el único rasgo visual dominante. Los controles nombran acciones concretas. Los estados vacíos explican qué información falta.

## Contrato y límites

`data/observatory.json` usa `schema_version: 1`. Los números desconocidos son `null`. El navegador valida el contenido antes de mostrarlo y conserva el último resumen válido si una actualización falla. El límite de lectura es 8 MiB por resumen, con un máximo de 128 ejecuciones y 500 puntos por curva. La petición tiene un tiempo máximo y no hay solicitudes simultáneas. El gráfico muestra los últimos 240 puntos del registro seleccionado y lo indica en su descripción.

`comparison_group` identifica un contrato experimental compartido de datos, universo, target, cortes, agregación de métricas y evaluación. Solo se comparan ejecuciones completadas de ese mismo grupo y fase. Un grupo no debe mezclar resultados de folds o definiciones de target incompatibles. La interfaz no puede verificar esos supuestos científicos a partir de un nombre.

Una ejecución `running` sin heartbeat reciente se muestra como «Sin actualización», no como pausada. `status` y `phase` pueden ser `null`, que se muestra como información desconocida. El estado vacío significa «Sin ejecuciones registradas», sin inferir si el ordenador está encendido. Las métricas y curvas de `test` y `evaluation` permanecen ocultas mientras `test_released` no sea `true` y la ejecución no haya terminado, también en CSV. El productor del JSON público debe excluir resultados protegidos antes de publicarlos, ya que ocultarlos en una pantalla no protege el archivo fuente.

La consulta pública se actualiza cada 60 segundos cuando la pestaña está visible. Un reloj local revisa cada 15 segundos los indicadores cuyo heartbeat todavía es reciente, sin peticiones ni reconstrucción de filas si no cambia el filtro. Ambos temporizadores se detienen al ocultar o abandonar la página. La salud se vuelve a comprobar al regresar.

La importación local detiene la consulta pública y conserva el archivo solo en memoria. Su heartbeat también puede caducar. No se utiliza `localStorage`, no hay telemetría y no se envían archivos importados. Las fechas se contrastan con `generated_at` y con su orden de inicio, actualización, historia y checkpoint, conservando precisión de microsegundos.

## Organización

El historial visible corresponde a los resúmenes de la instantánea seleccionada. No representa todo el archivo científico. El exportador inicial mantiene el intento actual por ejecución. Los intentos anteriores y eventos completos se conservan en el registro privado que gestionará MT-031, sin cargarlos todos en el navegador.

- `state.mjs` contiene validación y funciones puras de estado, comparación y CSV.
- `app.js` conecta controles, lectura del resumen y actualización del DOM.
- `index.html` y `styles.css` definen contenido accesible, diseño adaptable e impresión.
- `tests/state.test.mjs` verifica contratos y casos límite con `node:test`.

La separación por responsabilidad mantiene las funciones pequeñas y favorece composición y comprobación aislada. Se reutilizan las mismas reglas de visibilidad en pantalla y CSV. No hay frameworks, servidor de aplicación ni dependencias de ejecución. Se rechazan entradas inválidas al recibirlas y se mide el tamaño antes de añadir herramientas o capas nuevas.

## Comprobación local

```bash
node --test site/tests/*.test.mjs
```

Para inspeccionar la página desde la raíz del proyecto:

```bash
uv run --no-sync python -m http.server 8000 --bind 127.0.0.1 --directory site
```

Abrir `http://127.0.0.1:8000` y terminar el servidor con `Ctrl+C` al acabar. Los módulos y `fetch` requieren un origen HTTP, por lo que no se recomienda abrir el archivo directamente. El contenido de `site/` se puede publicar directamente en GitHub Pages. Esta carpeta no configura GitHub Actions ni contiene credenciales.

La comprobación opcional con navegador usa una instalación local de Playwright y Chrome. El propio script abre un servidor HTTP efímero en `127.0.0.1` y lo cierra al terminar:

```bash
node site/tests/browser-smoke.mjs /ruta/a/playwright/index.mjs /ruta/a/fixture-publica.json
```

El segundo argumento es opcional. Si se aporta, debe ser una salida temporal del exportador real con 32 ejecuciones y más de 1 MiB, usada solo para comprobar integración. El script usa datos sintéticos identificados dentro de la prueba, nunca modifica `site/data/observatory.json` y guarda capturas en `/tmp/`.

## Revisión y medidas

La revisión visual ha comprobado escritorio de 1440 píxeles y móvil de 390. Se han corregido el espacio excesivo del pie de curva y un desbordamiento causado por la etiqueta accesible de una tabla. El diagrama del método pasa a una secuencia textual en móvil para conservar la legibilidad. La cronología mantiene su carácter conceptual cuando el registro está vacío. El indicador general usa gris si no hay actividad confirmada.

Las pruebas puras cubren validación, valores desconocidos, cronología, caducidad, separación de intentos y fases, comparabilidad, resultados protegidos y CSV seguro. La prueba de navegador añade teclado, movimiento reducido, filtros, importación sin red, conservación del último resumen válido, impresión y límites de tamaño. Se ha importado una salida del CLI de 1.435.027 bytes, con 32 ejecuciones y 500 puntos por curva. También se han rechazado archivos y respuestas HTTP de más de 8 MiB, incluso sin `Content-Length`.

Los cinco recursos iniciales ocupan aproximadamente 77,5 kB sin compresión y 22,3 kB al comprimirlos individualmente con gzip en esta revisión. Son tamaños de archivos, no tiempos de carga medidos en GitHub Pages. No hay fuentes, bibliotecas o estilos remotos. El registro publicado continúa vacío y las medidas usadas en las pruebas no representan experimentos científicos.

El cierre del desarrollo exige revisar el contrato del exportador y las pruebas junto con la página. Una captura visual no demuestra la validez científica de los experimentos que se registren después.
