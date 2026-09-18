# Requisitos académicos de MARS-TITAN

Autor del proyecto: Gonzalo García Lama. Revisión documental: 18 de septiembre de 2026.

Este documento recoge los requisitos verificables en los documentos académicos suministrados y su aplicación a un proyecto de Tipo 3, «Comparativa de soluciones». La [propuesta presentada](proposal.md) define el alcance del proyecto. Su inclusión en el repositorio no acredita aprobación por la dirección académica ni autorización de depósito o defensa.

## Fuentes y alcance de la revisión

| Clave | Documento suministrado | Identificación y paginación |
| --- | --- | --- |
| G | Guía del Trabajo Fin de Estudios. Máster Universitario en Inteligencia Artificial ([registro de la copia local](source-manifest.json)) | 33 páginas PDF. Las páginas citadas coinciden con la numeración impresa a partir de la página 3. |
| R | Reglamento de Trabajos de Fin de Grado y de Fin de Máster de UNIR ([registro de la copia local](source-manifest.json)) | 9 páginas PDF. El historial de la p. 9 identifica la versión 10.1, aprobada el 29/01/2026 y en vigor desde el 30/01/2026. |
| U | Rúbrica de evaluación ([registro de la copia local](source-manifest.json)) | 2 páginas PDF. Los ocho criterios y sus descriptores están en la p. 1. La p. 2 contiene la continuación de la exportación, con los coeficientes. |

Las referencias siguientes usan páginas PDF, contadas desde la primera página del archivo. La revisión se limita a estas copias: no presupone que incluyan posteriores actualizaciones del campus. Los PDF institucionales se conservan como fuentes locales de consulta, excluidos del control de versiones.

Se distinguen las exigencias documentales de las fuentes, sus recomendaciones y las decisiones adoptadas para concretar MARS-TITAN. Una decisión del proyecto no se presenta como obligación de UNIR. «Preparado» indica que existe documentación de trabajo. No significa «realizado», «validado» ni «autorizado».

## Modalidad, alcance y propuesta

| ID | Requisito o indicación | Fuente exacta | Aplicación al proyecto |
| --- | --- | --- | --- |
| A01 | El TFE es obligatorio, original e inédito. Tiene una carga de 12 ECTS y exige superar la defensa ante una comisión. | G, §1.1, p. 3. R, art. 2.1-2.2, p. 1. | Ajustar la comparación y el volumen de ingeniería a esta carga. No se fija aquí una equivalencia en horas que las fuentes no proporcionan. |
| A02 | El trabajo debe aplicar o desarrollar competencias de la titulación, con supervisión del director. En este máster se orienta a investigar técnicas de IA y/o desarrollar soluciones de IA para un problema concreto. | R, art. 2.1, p. 1. G, §1.1, p. 4. | Comparar modelos de predicción financiera multimodal y analizar sus límites metodológicos. |
| A03 | El Tipo 3 exige estudiar alternativas para una misma tarea mediante un análisis empírico y pormenorizado. Identificar el problema, experimentar y discutir ventajas y desventajas a partir de los datos obtenidos. Una comparación previa de características no basta. | G, §3.2, p. 13. | La aportación central será una comparación reproducible de MARS-TITAN, referencias clásicas y neuronales y ablaciones. La superioridad de MARS-TITAN es una hipótesis. |
| A04 | La tipología es única. Pueden seleccionarse varias líneas de trabajo. La guía incluye aprendizaje automático, aprendizaje profundo y procesamiento del lenguaje natural. | G, §3.3, p. 15. | Tipo 3 como modalidad. Las líneas elegidas deben coincidir con la solicitud presentada, sin dar por acreditada su aceptación. |
| A05 | La solicitud debe incluir modalidad individual/grupal, datos de contacto, título provisional, tipo, líneas, aportación y justificación de novedad, bibliografía, origen de los datos y cuestionario específico. El título debe sintetizar la aportación en «no más de 12 o 15» palabras. | G, §3.1, pp. 8-10. R, art. 8.1, p. 4. | Conservar el texto de la propuesta presentada y verificar su adecuación al formulario del campus. No modificar unilateralmente el título presentado para satisfacer un recuento interpretado. |
| A06 | Una vez aprobada la propuesta y asignado el director no se puede cambiar de tema. La guía contempla concretar aspectos con el director. | G, pp. 7 y 9. | Documentar los ajustes de alcance dentro del tema de la propuesta y las decisiones que requieran la conformidad del director. No consta aquí una aprobación previa. |
| A07 | El tema y el desarrollo deben ser realizables con la dedicación correspondiente a los ECTS de la materia. La guía recomienda una cuestión concreta y abordable, propia de un TFE. | R, art. 8.2, p. 4. G, §3.1, p. 8. | Priorizar el experimento mínimo defendible. Dejar las extensiones costosas como trabajo futuro si no contribuyen a la comparación principal. |

El carácter individual de este proyecto es una decisión de su propuesta. Las disposiciones de trabajo grupal no se trasladan como obligaciones al trabajo de Gonzalo García Lama. La experiencia en producción corresponde al Tipo 1 y la demostración práctica de una herramienta al Tipo 2 (G, pp. 11-12). No se importan como requisitos propios del Tipo 3.

## Escritura, estructura y referencias

| ID | Requisito o indicación | Fuente exacta | Evidencia que deberá conservarse |
| --- | --- | --- | --- |
| A08 | Se debe utilizar la plantilla de elaboración del TFE y consultar las instrucciones específicas del aula. Cada tipo puede tener una estructura recomendada en esas instrucciones. | G, p. 4 y §3.2, pp. 10-11. | Memoria final trasladada a la plantilla vigente, con comprobación de sus apartados y formato. Los Markdown del repositorio son documentos de trabajo. |
| A09 | La rúbrica valora la redacción académica, el formato cuidado y la corrección lingüística. Para sobresaliente exige que las citas del texto aparezcan en referencias y que se utilice normativa APA. | U, p. 1, «Estilo y formato académico», 10 %. | Revisión de estilo y correspondencia entre citas y bibliografía. Normalización de las referencias según las instrucciones de la titulación. |
| A10 | Deben estar presentes y desarrollados los apartados, con coherencia y continuidad. Para sobresaliente se valora también una extensión adecuada. | U, p. 1, «Estructura/apartados», 10 %. | Hilo argumental entre problema, objetivos, método, resultados, discusión y conclusiones. Revisión de la estructura oficial cuando esté disponible. |
| A11 | El marco teórico y las referencias deben ser relevantes, cubrir adecuadamente el campo y contextualizar la contribución. | U, p. 1, «Marco Teórico y referencias», 10 %. | Fuentes primarias leídas y contrastadas. Identificación de qué se toma de cada trabajo y qué pertenece al diseño propio. |
| A12 | Los objetivos deben ser coherentes, alcanzables y realistas. Las conclusiones deben derivar del trabajo, incluir limitaciones y ofrecer una prospectiva fundamentada. | U, p. 1, criterios «Alcance» y «Relación entre objetivos, planteamiento, desarrollo y conclusiones». | Trazabilidad de los seis objetivos a experimentos, resultados y conclusiones, incluidos resultados negativos. |

Los tres documentos suministrados **no fijan un intervalo numérico de páginas, una edición concreta de APA, un número mínimo de referencias ni medidas de márgenes, interlineado o tipografía**. No se atribuyen a UNIR reglas como «APA 7» o un mínimo/máximo de páginas sin consultar la plantilla y las instrucciones que faltan. La recomendación de brevedad del título se refiere a la propuesta, no a la extensión de la memoria.

## Autoría, originalidad, datos y uso de herramientas

| ID | Exigencia documental | Fuente exacta | Aplicación y estado |
| --- | --- | --- | --- |
| A13 | El trabajo debe ser original e inédito y no haber sido plagiado, presentado o publicado anteriormente, en todo o en parte. | R, art. 2.2, p. 1. | Mantener la propuesta original como antecedente del mismo proyecto, separada de la futura memoria. La incorporación o publicación de materiales previos debe revisarse conforme a esta regla. Este registro no certifica su admisibilidad. |
| A14 | El plagio impide la autorización o produce suspenso. La falsa autoría implica la apertura de expediente disciplinario por falta muy grave. | R, art. 11.1-11.2, pp. 6-7. | Conservar procedencia de textos, código, datos y figuras, y referencias verificables. No existe en estas fuentes un porcentaje numérico que convierta por sí solo el trabajo en aceptable. |
| A15 | Un exceso de información ajena puede hacer que el trabajo no sea original aunque esté citada. La similitud con trabajos anteriores del estudiante puede hacer que no sea inédito aunque esté referenciada. | R, art. 11.4, p. 7. | La contribución propia debe consistir en decisiones justificadas, implementación, experimentación y análisis. Citar no sustituye esa contribución. |
| A16 | El estudiante debe aportar evidencias de autoría ante problemas de originalidad. Si sube material a una aplicación antiplagio y ello produce coincidencias posteriores, le corresponde demostrar que no es plagio. | G, §4.4, p. 22. R, art. 11.3, p. 7. | Conservar versiones y registros de elaboración. No subir la memoria a servicios antiplagio como paso automático de este proyecto. |
| A17 | Se considera conducta antiacadémica elaborar el trabajo con ayuda de terceros que no haya sido expresamente autorizada por el director. El director orienta, pero no es coautor ni responsable de la autoría del estudiante. | R, art. 8.4, pp. 4-5. G, §4.1, p. 18. | Cualquier autorización necesaria debe quedar acreditada. No se presume concedida ni se redacta una declaración ficticia de trabajo sin ayuda. |
| A18 | La guía desaconseja completamente las herramientas de IA para elaborar el trabajo. Si se emplean, el director debe validar un uso adecuado y ético. Debe reflejarse expresamente en el documento y deben incluirse referencias específicas a las herramientas conforme a las normas de citación. El uso o la citación inadecuados pueden impedir la defensa y hacer perder la convocatoria. | G, §4.4, p. 22. | Pendiente de verificar la validación del director y de consultar el manual de IA para estudiantes de UNIR al que remite la guía. Documentar el uso real y sus referencias antes de cualquier entrega que lo requiera. Este requisito no equivale a una prohibición absoluta ni autoriza a omitir información. |
| A19 | No pueden utilizarse datos de terceros, personales o de otra índole, sin autorización de su titular. | R, art. 8.3, p. 4. | Registrar procedencia, licencia y condiciones de uso de FinMultiTime y de cada fuente complementaria. El acceso a una descarga no acredita por sí mismo permiso para redistribuirla. |
| A20 | Si se requiere recabar datos personales de terceros, especialmente sanitarios, hay que obtener autorización del Comité de Ética antes de iniciar la recogida. Sin ella no se permite depósito ni defensa. | R, art. 2.7, p. 2. | Exigencia condicional. No se afirma que se aplique automáticamente a series de precios. Debe revisarse si la investigación incorpora datos personales. |

La obligación de informar sobre herramientas usadas para **elaborar el proyecto** se distingue de estudiar **modelos de IA como objeto de investigación**. No se incluye una declaración personal prefabricada ni se da por validado el uso de ninguna herramienta.

## Seguimiento, entregas y depósito

| ID | Requisito o indicación | Fuente exacta | Control del proyecto |
| --- | --- | --- | --- |
| A21 | Las entregas deben seguir la programación del aula y sus actividades. Hay tres borradores incrementales y un predepósito. No se admiten entregas adicionales, intermedias o extraordinarias fuera de ese esquema ordinario. | G, §4.4, pp. 21-22. | Trasladar al calendario las fechas reales del campus, todavía no documentadas aquí. Las seis fases de investigación no sustituyen estas entregas. |
| A22 | La primera entrega debería contener contexto, estado del arte, bibliografía y objetivos claros. La segunda debe mostrar avance sustancial. La tercera debe incluir todos los capítulos completamente desarrollados. | G, §4.4, pp. 24-25. | Preparar entregas acumulativas, aplicando las observaciones recibidas. La tercera incluye resultados, introducción y conclusiones, no solo el diseño. |
| A23 | Las valoraciones intermedias son APTO/COMPLETO o NO APTO/INCOMPLETO y no determinan la nota final. Entregas ausentes o insuficientes pueden impedir autorizar el depósito. | G, §4.4, pp. 21-23. | Conservar correcciones y resolución de observaciones sin equiparar una entrega apta a una calificación final. |
| A24 | El predepósito contiene la versión terminada y no admite cambios de forma ni de contenido. Tras el visto bueno se deposita el mismo documento. El plazo es inamovible. | G, §4.4, p. 25, §5.1, p. 27. | Verificar bibliografía, anexos, cifras, formato y archivos antes del predepósito. Identificar la versión exacta mediante su huella y conservarla. |
| A25 | El director decide si procede el depósito y autoriza la defensa sobre el documento definitivo depositado en PDF. Solo pueden ser convocados estudiantes autorizados. La autorización no supone aprobar. | R, art. 5.2, p. 3. Art. 9.1 y 9.8, p. 5. | Pendientes la memoria final y las autorizaciones del procedimiento académico. Preparar documentación no equivale a efectuar el depósito. |
| A26 | Como regla general deben estar superadas las demás materias antes de la defensa. Existen excepciones académicas expresas. La matrícula da derecho a ordinaria y extraordinaria en el mismo curso, con los requisitos establecidos. | R, art. 2.3-2.5, p. 2. Art. 7.2, p. 4. | Comprobar expediente y convocatoria con el campus, sin presumir que se cumple una excepción. |
| A27 | Tras no autorizar la defensa o no superarla en ordinaria existe un periodo adicional de tutela hasta el depósito extraordinario. La guía prevé una última entrega extraordinaria y mantiene la decisión de autorización del director. | R, art. 9.5-9.6, p. 5. G, §5.5, pp. 32-33. | La convocatoria extraordinaria sirve para completar o mejorar trabajo ya realizado. No se planifica como sustitución de todo el desarrollo ordinario. |

La guía establece cuatro sesiones con el director, una inicial y una tras cada borrador, y sitúa la comunicación académica en el correo interno del aula (G, §4.2, pp. 19-20). Recomienda copias de seguridad del borrador (G, pp. 25-26). La planificación concreta se completará con el calendario del campus.

## Defensa, evaluación y publicación institucional

| ID | Requisito o indicación | Fuente exacta | Implicación |
| --- | --- | --- | --- |
| A28 | La evaluación del proyecto corresponde a una comisión de dos especialistas. El presidente debe ser doctor y el secretario profesor en activo de UNIR. La propuesta de nota del director no es vinculante. | R, art. 4, p. 3. Art. 9.2 y 9.4, p. 5. Art. 10.3, p. 6. | Preparar evidencias comprensibles para evaluadores ajenos al desarrollo cotidiano del proyecto. |
| A29 | La rúbrica pondera estructura 20 %, contenido 50 % y exposición 30 %. Para aprobar se necesita al menos el 50 % de la puntuación de defensa: 1,5 puntos de sus 3 puntos posibles. | U, p. 1. R, art. 12.2, p. 7. G, pp. 31-32. | Consultar la [matriz completa](rubric-matrix.md). Una buena memoria no compensa suspender la defensa. |
| A30 | La presentación individual tiene un máximo de 15 minutos. Se recomiendan 10. Después hay preguntas. La duración total del acto puede ampliarse para evaluar las competencias, sin superar dos horas. | G, §5.2, p. 28, §5.3, p. 30. | Ensayar una exposición de 10 minutos, con margen respecto al máximo y preparación específica de las respuestas. |
| A31 | Entre depósito y defensa transcurren entre 30 y 90 días naturales. El calendario se comunica al menos 10 días naturales antes. | R, art. 10.1, p. 5. G, §5.3, p. 29. | No inventar fecha de defensa. Incorporar la convocatoria real cuando se reciba. |
| A32 | La defensa es pública, incluido el turno de preguntas. Salvo contradicción con la memoria del título, es en línea, se graba y exige identificación. La guía indica llevar presentación y DNI o pasaporte. | R, art. 2.6, p. 2. Art. 10.7, p. 6. G, pp. 29-31. | Preparar la documentación, comprobar conexión y legibilidad de figuras, y disponer de la presentación en PDF si se utiliza PowerPoint, como recomienda la guía. |
| A33 | Se debe demostrar dominio del contenido y autoría. Problemas graves en la defensa pueden implicar suspenso. | R, art. 10.5, p. 6. G, pp. 30-31. U, p. 1, «Dominio del contenido», 20 %. | Poder explicar datos, decisiones temporales, comparaciones, resultados negativos y límites sin depender de un texto leído. |
| A34 | La nota se expresa de 0 a 10 con un decimal: suspenso 0,0-4,9. Aprobado 5,0-6,9. Notable 7,0-8,9. Sobresaliente 9,0-10. La impugnación de la calificación tiene un plazo máximo de 72 horas desde la defensa. | R, art. 12.1-12.3, p. 7. G, §5.4, p. 32. | La matriz sirve para revisar evidencias, no para prometer una calificación. |
| A35 | La mención de matrícula de honor puede proponerse desde 9,7, con los límites de concesión del reglamento. El repositorio público institucional puede incorporar trabajos desde 9,0 si el estudiante lo autoriza expresamente, bajo licencia Creative Commons. | R, art. 12.7 y art. 13, p. 8. | Ninguna de estas posibilidades es automática. La publicación institucional es distinta de mantener un repositorio técnico del proyecto. |

## Decisiones del proyecto y trazabilidad de objetivos

Las siguientes decisiones concretan la propuesta presentada por Gonzalo García Lama. No son requisitos impuestos por la guía. Los números identifican objetivos y fases de investigación, **no carpetas** ni las tres fases administrativas del TFE.

| Objetivo/fase | Alcance acordado para el proyecto | Evidencia de cumplimiento prevista |
| --- | --- | --- |
| O1. Datos y disponibilidad temporal | Auditar precios, noticias, fundamentales y representaciones visuales de FinMultiTime y construir un conjunto utilizable con información disponible en cada instante. | Inventario, procedencia, reglas de disponibilidad, exclusiones, particiones y verificaciones de fuga temporal. |
| O2. Objetivo residual | Definir retornos residuales respecto al mercado y, si los datos lo permiten, al sector. | Fórmula, horizonte, estimación temporal de coeficientes y comprobación de disponibilidad de las etiquetas. |
| O3. Memoria e incertidumbre | Diseñar memoria de eventos con sorpresa basada en error, anomalía y relevancia económica, información de régimen y estimación de incertidumbre. | Especificación implementable, reglas de lectura/actualización y calibración, con cada mecanismo evaluado. |
| O4. Referencias y ablaciones | Comparar modelos clásicos y neuronales. Estudiar variantes sin memoria, con memoria global y sin criterio de sorpresa. | Configuraciones comparables y resultados de las variantes bajo el mismo protocolo. |
| O5. Evaluación | Evaluar error, dirección, calibración, Rank IC y simulación long-short, considerando costes, drawdown y rotación mediante validación temporal walk-forward. | Resultados fuera de muestra, incertidumbre de las estimaciones y trazabilidad de predicciones y costes. |
| O6. Análisis crítico | Analizar fallos, contribución de modalidades y coste computacional bajo un presupuesto experimental de 8 GB de VRAM. | Discusión sustentada en resultados y mediciones de recursos. Límites y trabajo futuro. |

El cumplimiento académico se apoya en la calidad de la comparación y del análisis, también cuando los resultados no confirman las hipótesis. La causalidad económica, la rentabilidad futura y la aptitud para producción no se dan por demostradas mediante residualización, ablaciones o un backtest.

## Pendientes de confirmación documental

- Obtener la plantilla y las instrucciones de elaboración del TFE vigentes en el campus. Comprobar estructura, extensión, formato y edición de citación, si la especifican.
- Incorporar el calendario real, el estado de la propuesta y las indicaciones del director, sin inventar fechas ni aprobaciones.
- Consultar el manual de IA para estudiantes y documentar la validación y las referencias exigibles al uso real de herramientas.
- Acreditar las condiciones de utilización y redistribución de datos y materiales, y revisar la incorporación de antecedentes al trabajo inédito.
- Completar los experimentos y redactar la memoria y la defensa a partir de sus resultados. La documentación preparatoria por sí sola no satisface el Tipo 3.
