# Auditoría de uso del observatorio

Esta auditoría revisa el observatorio publicado en `site/` antes de su rediseño (#451). Describe qué preguntas responde, qué cuesta encontrar, qué datos publicados no aparecen, cómo se comporta con muchos registros y qué problemas de accesibilidad tiene. Las conclusiones del final orientan el rediseño. No evalúan modelos ni resultados científicos.

## Método

Se sirvió el sitio de `develop` (`f2eef70c`) contra una copia de lectura del índice público y sus 51 páginas, tomada el 9 de octubre de 2026 a las 23:10 UTC desde la salida del recolector en marcha. La copia contiene 3.319 registros de 88 campañas y ocupa 211 KB de índice más 11,9 MB de páginas. El recolector, su estado y la copia de publicación no se modificaron.

La revisión usó Chrome 153 sin interfaz controlado por Playwright 1.63, a 1440 × 1000 y a 390 × 844 píxeles. Un servidor local registró cada petición con sus cabeceras condicionales. El coste de CPU se midió con `Performance.getMetrics` de Chrome durante 125 segundos con la pestaña visible y sin interacción. El contraste se calculó con la fórmula de luminancia relativa de WCAG 2.2.

## Preguntas que responde hoy

| Pregunta | Respuesta actual | Problema |
| --- | --- | --- |
| ¿Hay algo en marcha? | La primera ejecución del índice, ordenado con las activas primero. | Si no hay ninguna activa, abre una comprobación técnica de septiembre sin relación con lo actual. |
| ¿Cómo va una campaña? | Una línea de texto por campaña dentro de un desplegable cerrado. | No muestra ventanas, brazos ni semillas. Solo cuenta las completadas, no las pausadas, bloqueadas o pendientes. |
| ¿Cómo aprende una ejecución? | Una curva SVG de la ejecución elegida, limitada a sus últimos 240 puntos. | No se pueden comparar curvas de varias ejecuciones. El recorte solo se declara en la descripción accesible. |
| ¿Qué modelo va mejor? | Tabla de ejecuciones completadas del mismo grupo y fase. | Solo usa la página visible de 64 registros. Un grupo repartido entre páginas aparece incompleto. |
| ¿Dónde está una ejecución concreta? | Selector de 64 identificadores y búsqueda dentro de la página. | No hay búsqueda global ni enlace directo a una ejecución. |
| ¿Cuánto falta? | Ninguna respuesta. | Los recuentos previstos existen, pero no se muestra avance ni estimación. |
| ¿Cómo está el equipo? | Ninguna respuesta. | No hay temperatura, VRAM, RAM ni disco, ni siquiera en local. |

## Lo que cuesta encontrar

- El resumen de campañas, que es la información más útil del índice, queda plegado bajo «Campañas y recuentos previstos».
- El historial se recorre página a página entre 52 páginas. Filtros, CSV y comparación se aplican solo a la página consultada. La nota lo advierte, pero obliga a navegar a ciegas.
- Los identificadores públicos (`neural-real-expanded-20261006-fold-003-…`) mezclan campaña, etapa y ventana en una cadena larga. El selector los recorta en móvil.
- Ninguna vista tiene dirección propia. No se puede enviar un enlace a una ejecución, una campaña o un tramo de una curva.
- Las dependencias entre campañas existen en el índice, pero no se muestran. Una campaña «Bloqueada» no dice de qué depende.

## Datos publicados que no se muestran

El índice y las páginas ya contienen información que la interfaz no usa o esconde en el detalle de una sola ejecución:

- Por campaña: `dependencies`, `planned_runs`, `registered_runs` y los recuentos por estado (`paused`, `not_started`, `blocked`). En la copia hay 2 ejecuciones en pausa y 7 campañas bloqueadas.
- Por ejecución: `started_at` (1.310 registros), `elapsed_seconds` (2.424), `vram_peak_mib` (1.957), `ram_peak_mib` (1.189) y `samples_per_second` (992). Permiten ver coste y caudal de toda la campaña, no solo de una ejecución.
- `metadata.method` separa KLPO completo, Monte Carlo y exacto, REINFORCE y controles neuronales. La interfaz lo enseña como texto suelto.
- `configuration_sha256` y `source_sha256` acreditan la procedencia de cada cifra, pero no se muestran.
- 108 ejecuciones tienen `financial_validation`, visible solo una a una.
- 2.166 ejecuciones tienen curva por época (36.524 puntos en total). Solo se puede ver una curva cada vez.

Hay además medidas en los recibos que el recolector no publica. Cada época de un `run.json` neuronal contiene el MAE y el caudal de entrenamiento, el MAE por sesión de validación y la duración de la época. El recibo guarda también la validación inicial, la mejor época y si hubo parada temprana. Sin esos campos no se pueden dibujar curvas de entrenamiento frente a validación.

Conviene aclarar una medida publicada. En los recibos neuronales, `samples_per_second` procede de la pasada de validación, no del entrenamiento. La interfaz lo rotula como «Muestras por segundo» sin más contexto, lo que invita a compararlo con un caudal de entrenamiento.

## Rendimiento con muchos datos

| Medida | Valor observado |
| --- | --- |
| Primer contenido con datos | 423 ms en local |
| CPU de la pestaña en reposo | 0,14 s en 125 s (0,11 %) |
| Nodos del DOM | 1.942 |
| Peticiones por consulta | 2 (`observatory.json` y `deployment.json`) |
| Cabeceras condicionales enviadas | Ninguna |

La consulta usa `fetch` con `cache: "no-store"` y sin `If-None-Match`. Cada minuto descarga de nuevo 211 KB aunque el índice no haya cambiado, unos 12,4 MB por hora y pestaña. Cada respuesta se valida y se vuelve a dibujar entera.

El coste en reposo es bajo y debe conservarse. Lo que no escala es el dibujo. La curva crea un círculo SVG por punto y descarta todo lo anterior a los últimos 240. Con las trazas de #448 o con varias curvas superpuestas, ese enfoque crearía decenas de miles de nodos. La tabla del historial se reconstruye completa con cada cambio de filtro.

## Accesibilidad

- No hay modo oscuro. `color-scheme` está fijado en `light`.
- El texto secundario (`#526780`) alcanza 5,8:1 sobre blanco y 5,4:1 sobre el fondo de los paneles. Cumple AA. El gris `#8c9bae` (2,8:1) solo se usa en un borde decorativo.
- Varios textos bajan a 9, 10 u 11 píxeles. En móvil, las etiquetas de la curva SVG se escalan con el `viewBox` y quedan en torno a 6 píxeles.
- La curva tiene título y descripción, pero no una tabla equivalente con sus valores.
- El indicador de conexión se oculta por debajo de 760 píxeles, de modo que en móvil no se ve si la última consulta falló.
- El enlace para saltar al contenido, el foco visible, la semántica de las tablas y el escape del texto (incluido un intento de inyección en la prueba existente) funcionan.

## Decisiones que se derivan

1. **Empezar por la campaña.** La vista inicial debe responder qué se está ejecutando y cuánto falta, con una matriz de ventanas, brazos y semillas en lugar de una ejecución arbitraria.
2. **Cargar todo el historial de forma progresiva** con páginas inmutables descargadas una sola vez, para que comparación, búsqueda y CSV usen todos los registros.
3. **Consultar el índice de forma condicional** y no volver a validar ni dibujar si la respuesta es 304.
4. **Dibujar en canvas** con una biblioteca pensada para series largas, con tabla equivalente y reducción declarada cuando se aplique.
5. **Separar las medidas por su significado**: caudal de validación frente a entrenamiento y VRAM asignada frente a reservada, con unidades en cada etiqueta.
6. **Mostrar la procedencia** de cada cifra con su página publicada y las huellas del recibo y la configuración.
7. **Ofrecer un modo local en directo** que no dependa de los cinco minutos de publicación de Pages y que muestre también el estado del equipo.
8. **Modo oscuro, textos de al menos 12 píxeles y estado de conexión visible en móvil.**
