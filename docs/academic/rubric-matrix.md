# Matriz de evaluación y evidencias de MARS-TITAN

Autor del proyecto: Gonzalo García Lama. Revisión: 18 de septiembre de 2026.

La matriz traduce los ocho criterios de la rúbrica suministrada, p. 1, a evidencias del proyecto de Tipo 3. El [manifiesto](source-manifest.json) identifica la copia local. Los pesos se han comprobado en la representación visual del PDF. La segunda página es una continuación de la exportación y no introduce criterios adicionales. Los descriptores de sobresaliente se sintetizan sin alterar su sentido.

«Preparado» se refiere únicamente a material documental disponible: la [propuesta presentada](proposal.md), los [requisitos académicos](requirements.md), la [revisión del documento original](../research/original-review.md) y esta matriz. «Pendiente» identifica trabajo necesario para demostrar el criterio. Ninguna fila equivale a una nota concedida o a experimentos ya ejecutados.

## Pesos oficiales

| ID | Bloque | Criterio | Peso sobre el total | Máximo sobre 10 |
| --- | --- | --- | ---: | ---: |
| C1 | Estructura | Estilo y formato académico. Estilo formal/académico, formato de bibliografía | 10 % | 1,0 |
| C2 | Estructura | Estructura/apartados | 10 % | 1,0 |
| C3 | Contenido | Alcance | 10 % | 1,0 |
| C4 | Contenido | Marco Teórico y referencias | 10 % | 1,0 |
| C5 | Contenido | Desarrollo específico de la contribución | 20 % | 2,0 |
| C6 | Contenido | Relación entre objetivos, planteamiento, desarrollo y conclusiones | 10 % | 1,0 |
| C7 | Exposición | Presentación del TFE/Exposición | 10 % | 1,0 |
| C8 | Exposición | Dominio del contenido | 20 % | 2,0 |
| **Total** | **Estructura 20 % + contenido 50 % + exposición 30 %** | **Ocho criterios** | **100 %** | **10,0** |

Si `c1, ..., c8` son las valoraciones de los criterios en escala 0-10, la suma ponderada es:

`nota = 0,10·c1 + 0,10·c2 + 0,10·c3 + 0,10·c4 + 0,20·c5 + 0,10·c6 + 0,10·c7 + 0,20·c8`.

La exposición aporta `0,10·c7 + 0,20·c8`, con un máximo de 3 puntos. Para aprobar es imprescindible obtener **al menos 1,5 puntos en este bloque**, además de alcanzar la nota global de aprobado. No se establece en la rúbrica un mínimo independiente para cada uno de sus dos criterios. Esta condición consta en la nota al pie de la rúbrica, p. 1, en el Reglamento, art. 12.2, p. 7, y en la Guía, pp. 31-32. Sus versiones están identificadas en el [manifiesto de fuentes](source-manifest.json).

## Descriptores y estado de las evidencias

| Criterio | Qué exige el nivel sobresaliente | Evidencia concreta de MARS-TITAN | Preparado | Pendiente |
| --- | --- | --- | --- | --- |
| C1. Estilo y formato, 10 % | Redacción académica fluida, elegante y sin errores. Todas las citas del texto en la lista de referencias. Uso de APA. | Memoria conforme a la plantilla, terminología consistente, ecuaciones definidas, tablas y figuras legibles, citas comprobadas y bibliografía normalizada. | Registro de requisitos y detección del sistema de citas numérico del original, que debe adaptarse. | Obtener instrucciones de formato, completar las referencias y revisar la memoria terminada. No se presupone una edición de APA no especificada en las fuentes. |
| C2. Estructura/apartados, 10 % | Todos los apartados, desarrollo lógico y continuo, coherencia entre secciones y extensión adecuada. | Secuencia problema y objetivos → estado del arte → datos y método → resultados → discusión → conclusiones. Anexos que permitan comprobar decisiones sin interrumpir el argumento. | Alcance organizado en seis objetivos y separación de normativa, propuesta y revisión crítica. | Ajustar el índice a la plantilla oficial y completar todos los apartados. La existencia de documentos de planificación no sustituye una memoria integrada. |
| C3. Alcance, 10 % | Contribución interesante y significativa. Tema bien formulado y justificado. Objetivos correctos, coherentes, alcanzables y realistas. | Problema financiero concreto, universo y horizonte definidos, comparación empírica relevante y alcance viable con 8 GB de VRAM. La aportación se mide por el conocimiento obtenido. | Tipo 3 y seis objetivos delimitados. Identificación de extensiones que pueden desbordar el trabajo. | Auditar los datos, comprobar viabilidad computacional y ejecutar una comparación suficiente para sostener la contribución. No se exige prometer una mejora. |
| C4. Marco teórico y referencias, 10 % | Marco y estudios previos del nivel exigible. Referencias relevantes y con cobertura adecuada. Correcta contextualización. | Síntesis crítica de FinMultiTime, predicción residual, memoria neural, adaptación temporal, incertidumbre y evaluación financiera. Diferencias precisas entre métodos publicados y adaptaciones propias. | Inventario inicial de referencias del original y de los aspectos que requieren ampliación y contraste. | Leer y verificar fuentes primarias, ampliar la cobertura metodológica y justificar cada alternativa. Una mención a TITANS, ATLAS o SEAL no demuestra una reproducción fiel. |
| C5. Desarrollo de la contribución, 20 % | Desarrollo claro y bien planteado. Resultados coherentes con los objetivos, bien descritos y correctamente discutidos. | Conjunto point-in-time auditable, modelos y ablaciones comparables, protocolo temporal reproducible, predicciones fuera de muestra, métricas con incertidumbre y mediciones de recursos. | Especificación del tipo de evidencia y revisión de riesgos de fuga, actualización de memoria, comparación y medición. | Implementar, verificar y ejecutar los experimentos. Generar tablas y figuras desde artefactos reales. Las gráficas de expectativas del original no son resultados. |
| C6. Coherencia de objetivos y conclusiones, 10 % | Tema y objetivos bien planteados. Conclusiones originales derivadas del trabajo y expresadas con terminología propia. Limitaciones y prospectiva coherentes y fundamentadas. | Una conclusión por objetivo vinculada a resultados identificables. Distinción entre mejora, ausencia de mejora, resultado inconcluso y experimento no realizado. | Correspondencia de O1-O6 con evidencias y problemas del original. | Completar resultados y redactar conclusiones proporcionadas. No inferir causalidad económica, rentabilidad futura o preparación para producción a partir de los resultados retrospectivos. |
| C7. Presentación/exposición, 10 % | Ideas claras y ordenadas, síntesis, comunicación fluida, ajuste al tiempo, tranquilidad y confianza, claridad y atención sostenida de la audiencia. | Presentación centrada en problema, comparación, hallazgos y límites. Gráficos legibles. Exposición ensayada de unos 10 minutos, dentro del máximo de 15. | Guion temporal y criterios de selección de contenido incluidos a continuación. | Preparar diapositivas con resultados reales, ensayar con cronómetro y revisar audio, cámara, conexión y legibilidad. No existe todavía evidencia de desempeño oral. |
| C8. Dominio del contenido, 20 % | Diapositivas usadas como apoyo. Respuestas que demuestren dominio del tema. Interés y entusiasmo al explicarlo. | Capacidad de justificar decisiones de datos, etiquetas, memoria, particiones, métricas, incertidumbre y costes, así como de reconocer límites y explicar un resultado negativo. | Banco de cuestiones técnicas para orientar la preparación. | Defensa oral practicada por el estudiante, con respuestas apoyadas en sus experimentos. Un documento escrito no acredita por sí mismo dominio ni autoría. |

La matriz no anticipa un sobresaliente. La comisión evaluadora decide la calificación y puede apartarse de la propuesta del director: Reglamento, art. 9.4, p. 5 y art. 10.3, p. 6.

## Trazabilidad de los seis objetivos

| Objetivo | Evidencia mínima que permite discutir su cumplimiento | Criterios principalmente relacionados | Estado |
| --- | --- | --- | --- |
| O1. Datos point-in-time | Inventario reproducible, procedencia y condiciones de uso, fechas de disponibilidad por modalidad, datos excluidos y comprobaciones temporales. | C3, C5, C6, C8 | Preparado el alcance. Pendiente la evidencia empírica completa. |
| O2. Retornos residuales | Horizonte y fórmula, datos de mercado y sector disponibles, coeficientes estimados con pasado y etiquetas accesibles solo una vez observadas. | C4, C5, C6, C8 | Preparado el problema. Pendiente implementación y validación. |
| O3. Memoria, regímenes e incertidumbre | Regla de sorpresa y actualización definida, lectura anterior a la escritura correspondiente, estado de memoria trazable y calibración evaluada. | C3, C4, C5, C8 | Preparada la revisión del diseño. Pendiente validación de cada mecanismo. |
| O4. Referencias y ablaciones | Comparaciones clásicas y neuronales con mismos datos y particiones. Variantes sin memoria, memoria global y sin sorpresa. Presupuesto de ajuste documentado. | C4, C5, C6, C8 | Preparada la estructura comparativa. Pendientes ejecuciones comparables. |
| O5. Evaluación temporal | Resultados de error, dirección, calibración, Rank IC y simulación long-short con costes, drawdown y turnover. Resultados por ventana e intervalos apropiados. | C5, C6, C7, C8 | Preparadas las familias de métricas. Pendientes resultados fuera de muestra. |
| O6. Análisis crítico y recursos | Fallos documentados, aportación de modalidades, medidas de tiempo y VRAM, limitaciones, resultados negativos y líneas futuras justificadas. | C3, C5, C6, C7, C8 | Preparados los riesgos a examinar. Pendientes conclusiones sustentadas en mediciones. |

Los objetivos son compromisos de investigación de la propuesta. No se convierten en afirmaciones de que todos los módulos serán útiles o de que el sistema completo será la mejor alternativa.

## Preparación de la defensa oral

La Guía, §5.2, p. 28, fija 15 minutos como máximo individual y recomienda 10. El siguiente reparto de 10 minutos es una decisión de preparación, no una distribución impuesta por UNIR.

| Intervalo | Contenido | Qué debe quedar claro |
| --- | --- | --- |
| 0:00-1:00 | Problema, modalidad Tipo 3 y pregunta principal | Qué se compara y qué conocimiento se pretende obtener. |
| 1:00-2:30 | Datos y objetivo residual, O1-O2 | Qué información estaba disponible en cada decisión y qué se intenta predecir. |
| 2:30-4:00 | Modelos, memoria e incertidumbre, O3 | Diferencia verificable entre la propuesta y sus referencias, sin detalles que no afecten a los resultados. |
| 4:00-5:00 | Referencias, ablaciones y protocolo, O4-O5 | Por qué las comparaciones son justas y cómo se controla la fuga temporal. |
| 5:00-8:00 | Resultados principales, O5 | Magnitud, estabilidad e incertidumbre de las diferencias, incluidos resultados desfavorables. |
| 8:00-9:30 | Discusión y recursos, O6 | Qué mecanismos aportan, cuáles fallan y qué limita la generalización. |
| 9:30-10:00 | Conclusiones y continuidad | Respuesta a la pregunta inicial, límites de la evidencia y siguiente experimento justificado. |

Las diapositivas deben apoyar el discurso. La Guía, pp. 29-31, recomienda elementos visuales, síntesis, lenguaje académico y contacto con la comisión. También recomienda disponer de PDF si se utiliza PowerPoint. La defensa incluye preguntas posteriores y exige documentación de identidad. No hay en los documentos suministrados un número obligatorio de diapositivas.

### Preguntas que requieren dominio de la evidencia

| Cuestión probable | Evidencia con la que debe poder responderse |
| --- | --- |
| ¿Qué ocurre con una noticia posterior al cierre o un fundamental sin fecha de publicación fiable? | Regla de disponibilidad, ejemplo trazable y efecto de la exclusión o del supuesto conservador. |
| ¿Cómo puede la memoria usar un retorno futuro sin introducir fuga? | Cronología de predicción, maduración de la etiqueta y actualización. La respuesta debe mostrar que ese retorno no se usa antes de conocerse. |
| ¿Por qué un retorno residual no prueba causalidad? | Definición del ajuste mercado/sector y explicación de qué asociaciones elimina y qué hipótesis no permite identificar. |
| ¿La mejora proviene de la memoria o de más parámetros, datos o ajuste? | Ablaciones, capacidad y presupuesto documentados, mismas particiones y resultados por semilla o ventana. |
| ¿En qué se diferencia una adaptación propia de una reproducción de TITANS? | Correspondencia entre el método publicado y la implementación, con cambios y limitaciones explícitos. |
| ¿Qué significa un resultado negativo o no concluyente? | Diferencias e intervalos, estabilidad temporal y análisis de potencia o limitaciones de muestra, sin convertir ausencia de evidencia en equivalencia. |
| ¿Cómo se transforma la predicción en el resultado económico presentado? | Regla de cartera, momento de ejecución, rotación, costes y restricciones de la simulación. |
| ¿La incertidumbre está calibrada y cómo se ha elegido el umbral de abstención? | Conjunto de calibración/validación separado del test, curvas de calibración o cobertura y relación entre riesgo y cobertura. |
| ¿Se ha respetado el presupuesto de 8 GB? | Registro de GPU, precisión, tamaños de lote y secuencia, pico de VRAM y tiempo, incluyendo el coste de obtener representaciones cuando corresponda. |
| ¿Qué se ha hecho personalmente y cómo se justifica cada decisión? | Historial de trabajo, resultados reproducibles y explicaciones del estudiante. Cumplimiento de los requisitos de autoría y herramientas recogidos en la Guía, p. 22. |

## Comprobación previa a una entrega final

- Cada cifra presentada puede reconstruirse desde un experimento identificado. Las ilustraciones conceptuales se distinguen de los resultados.
- Cada objetivo tiene una respuesta basada en evidencia o una limitación explícita. No quedan hipótesis redactadas como conclusiones.
- Las citas y los materiales reutilizados tienen origen comprobado y cumplen las condiciones aplicables.
- La memoria cumple la plantilla y las instrucciones disponibles en el campus. La versión de predepósito está cerrada.
- La exposición se ha ensayado dentro del tiempo y el estudiante puede responder sobre resultados y límites sin leer una respuesta preparada.

Estas comprobaciones preparan la revisión del trabajo. No sustituyen las autorizaciones del director ni la evaluación de la comisión.
