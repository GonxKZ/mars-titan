# Revisión de los tres enlaces de X

La [revisión ampliada](social-followup.md) incorpora la imagen de doce ecuaciones de Trackmind y precisa qué conceptos pueden aprovecharse. La imagen no acredita un artículo identificable ni el código anunciado.

Fecha de comprobación: 18 de septiembre de 2026. Alcance: documentación y contraste de fuentes. No se ha instalado software, ejecutado código de los posts ni iniciado experimentos científicos.

Los originales devolvieron HTTP 403 al lector web. Se recuperaron posteriormente mediante la API pública de FxTwitter, que devolvió `code: 200`, el identificador solicitado, la URL original y el autor correspondiente. Esta es una recuperación mediante un intermediario: se conserva esa limitación de procedencia y no se afirma haber leído los originales directamente en X. Las afirmaciones técnicas aprovechables se contrastaron con sus fuentes oficiales.

## Registro de acceso

| Enlace proporcionado | Fecha indicada por la copia recuperada, UTC | Resultado |
|---|---|---|
| [0xTrackmind, 2098778516380610740](https://x.com/0xTrackmind/status/2098778516380610740) | 12-09-2026, 14:19:26 | Original: 403. [API secundaria](https://api.fxtwitter.com/0xTrackmind/status/2098778516380610740): contenido y post citado recuperados. |
| [shubh6200, 2097175586951270505](https://x.com/shubh6200/status/2097175586951270505) | 08-09-2026, 04:09:58 | Original: 403. [API secundaria](https://api.fxtwitter.com/shubh6200/status/2097175586951270505): texto y enlace externo recuperados. |
| [RuujSs, 2096650137053667533](https://x.com/RuujSs/status/2096650137053667533) | 06-09-2026, 17:22:01 | Original: 403. [API secundaria](https://api.fxtwitter.com/RuujSs/status/2096650137053667533): contenido y artículo del post citado recuperados. |

También se probaron búsquedas por los tres identificadores exactos, sin resultados. Los puntos de acceso oficiales de oEmbed y sindicación no fueron accesibles desde el lector web. FxTwitter tampoco produjo contenido en ese lector, pero sí mediante una petición HTTP de solo lectura desde terminal. No se usaron cuentas ni credenciales.

## 0xTrackmind: el artículo recuperado no coincide con la promesa

La publicación anuncia un supuesto artículo cuantitativo con doce pasos, fórmulas y código Python completo, y promete una implementación inmediata con un asistente. Sin embargo, la copia recuperada cita [otra publicación del mismo autor](https://x.com/0xTrackmind/status/2080996366122172732), que contiene el artículo [You Were Taught to Fear Randomness and Called It Bad Luck. The Real Thing Can Be Calculated.](https://x.com/i/article/2080989573128024064).

Ese contenido es un ensayo divulgativo de seis apartados sobre azar, incertidumbre y riesgo. En el texto y las entidades recuperadas no aparecen un artículo identificable, DOI, repositorio ni bloques de código Python. Por tanto, el vínculo con el supuesto documento de doce pasos no queda acreditado. No se incorpora como evidencia de una metodología completa ni de un sistema operativo. Esta conclusión se refiere a la copia obtenida, sin afirmar que no exista otro recurso fuera de ella.

Utilidad para el proyecto: recordatorio informal de separar señal y ruido. Los fundamentos estadísticos deben citarse desde la [bibliografía financiera](finance-sources.json), no desde esta promoción. No se añade ninguna referencia científica nueva a partir de este enlace.

## shubh6200: latencia de memoria y rendimiento

El texto recuperado enlaza directamente a [Memory Latency, de Algorithmica](https://en.algorithmica.org/hpc/cpu-cache/latency/), material de Sergey Slotin. La página oficial fue accesible. Explica la diferencia entre latencia y ancho de banda, utiliza recorridos dependientes de memoria para medir latencia y analiza los efectos de la jerarquía de caché y la frecuencia de CPU. Es documentación técnica de autor, no una publicación científica revisada por pares.

Su relación con MARS-TITAN es secundaria: puede orientar el análisis de rendimiento de la carga y preparación de datos cuando existan mediciones. La caché de CPU no es la memoria neuronal de Titans. Tampoco demuestra una mejora de un modelo en CUDA. En esta fase se conserva como lectura complementaria para el futuro perfilado. No justifica cambiar la canalización ni aplicar optimizaciones antes de identificar un cuello de botella.

## RuujSs: LEAN y un artículo distinto sobre arquitectura

El post recomienda LEAN como motor de negociación algorítmica con Python y C#. El [repositorio oficial de QuantConnect](https://github.com/QuantConnect/Lean) confirma su arquitectura dirigida por eventos, soporte de backtesting y licencia Apache-2.0. Esto acredita la existencia y naturaleza del software. La licencia del motor no constituye una licencia de los datos de mercado que pueda utilizar.

El post citado es [2096241405190602967](https://x.com/RuujSs/status/2096241405190602967), cuyo artículo se titula [How to Use Loop and Graph Engineering to Build an Alpha Engine (The Complete Build)](https://x.com/i/article/2094863867713347584). Se trata de una propuesta divulgativa de organización de investigación, riesgo, asignación, ejecución y monitorización. No debe confundirse con documentación oficial de LEAN ni con un artículo revisado por pares.

La lectura de sus fragmentos de código recuperados muestra dos límites concretos: la rutina de DSR llama a `_expected_max_sr()` sin incluir su implementación y el ejemplo de deterioro suma ventanas por debajo del umbral sin exigir que sean consecutivas, aunque el texto sí lo afirma. Por ello no se adopta como código listo para usar. Tampoco se da por demostrado el control del error estadístico por aumentar un umbral tras cada ensayo. La referencia canónica del DSR ya figura como `bailey2014dsr` en [finance-sources.json](finance-sources.json). El artículo social no la sustituye.

Para el proyecto, LEAN queda como herramienta que podría valorarse si una fase posterior requiere simulación de órdenes y ejecución por eventos. La comparativa predictiva no obliga a integrarlo. La separación entre investigación y evaluación puede documentarse sin crear ahora agentes autónomos, infraestructura de negociación ni conexiones con un bróker.

## Incorporación a la documentación del proyecto

- Algorithmica: lectura complementaria para medir el coste computacional de la carga y preparación de datos.
- LEAN: referencia oficial de software para una posible evaluación financiera posterior. No es una dependencia decidida ni instalada.
- Artículos de X: registro de ideas y límites de verificación. No sustentan afirmaciones sobre rentabilidad, causalidad, superioridad de MARS-TITAN o funcionamiento de fondos institucionales.

Estos enlaces no cambian la base científica reunida en [neural-review.md](neural-review.md) y en la revisión financiera. Su valor es aportar pistas cuya procedencia y aplicabilidad se han comprobado por separado.
