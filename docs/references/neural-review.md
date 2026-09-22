# Revisión de memoria adaptativa, multimodalidad e incertidumbre

MARS-TITAN. Autor: Gonzalo García Lama. Revisión comprobada el 18 de septiembre de 2026. Este documento sirve de apoyo a la comparación experimental. Las propuestas de arquitectura son hipótesis que se deben contrastar, no resultados del proyecto.

Las 17 referencias tienen ficha y entrada bibliográfica: 15 se mantienen en `neural-sources.json` y `neural.bib`. EnbPI (`xu2021enbpi`) y ACI (`gibbs2021aci`) usan las entradas canónicas de `finance-sources.json` y `finance.bib` para evitar duplicados. Se ha comprobado un PDF abierto de cada trabajo. En TTT y MIRAS se distingue la publicación de la copia arXiv accesible. Acceso abierto y permiso de redistribución son cuestiones diferentes. El campo `redistribution` permanece en `unknown` cuando no se ha verificado expresamente ese permiso.

## Correspondencia con los objetivos

| Objetivo | Uso de esta revisión |
|---|---|
| 1. Datos multimodales y disponibilidad temporal | FinMultiTime, FinBERT y antecedentes de fusión financiera. Revisión de qué información estaba disponible en cada instante. |
| 2. Retorno residual | Distinguir el objetivo del proyecto de las tareas de sentimiento, dirección y sorpresa de beneficios usadas por otros autores. La justificación financiera del residual corresponde a su revisión específica. |
| 3. Arquitectura e incertidumbre | Titans, TTT y MIRAS como base. ATLAS, SEAL y Nested Learning para delimitar avances y alcance. |
| 4. Referencias y ablaciones | DLinear, TCN, PatchTST y comparaciones con la misma entrada y presupuesto. |
| 5. Evaluación walk-forward | Comparación temporal y evaluación de intervalos con EnbPI y ACI, sin utilizar etiquetas antes de que estén disponibles. |
| 6. Análisis crítico | Separar evidencia publicada, extrapolaciones al problema financiero y límites de recursos y datos. |

## Reseñas de fuentes primarias

### 1. FinMultiTime: `xu2025finmultitime`

**Estado:** preprint v2 de 2025, declarado en revisión. Reúne precios, noticias, gráficos y tablas. Es la fuente del conjunto de datos, pero su alineación no acredita disponibilidad histórica. Para este proyecto interesa reconstruir una muestra verificable y documentar qué modalidades pueden usarse sin anticipación. El PDF conserva campos de plantilla editorial: no debe citarse como una publicación ACM de 2018. [Ficha y versión](https://arxiv.org/abs/2506.05019v2).

### 2. Titans: `behrouz2025titans`

**Estado:** NeurIPS 2025, revisado por pares. Combina atención y memoria neuronal que cambia durante la inferencia, con mecanismos de sorpresa y olvido. La versión publicada es la referencia principal, por delante del preprint 2501.00663. Permite fundamentar un módulo pequeño de memoria y estudiar cuándo compensa actualizarlo. Sus experimentos de lenguaje y series temporales no demuestran rentabilidad bursátil. Una adaptación compacta debe identificarse como tal y compararse con la misma arquitectura sin actualización. [Publicación](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html).

### 3. TTT: `sun2025ttt`

**Estado:** ICML 2025, revisado por pares. Preprint inicial de 2024. El estado recurrente es un modelo que se actualiza mediante una tarea autosupervisada. Las variantes TTT-Linear y TTT-MLP ayudan a separar la capacidad de memoria de su regla de aprendizaje. Los experimentos principales, de 125 millones a 1.300 millones de parámetros, no constituyen un presupuesto razonable de reproducción para este proyecto. Una memoria lineal pequeña sería una adaptación experimental. [Actas](https://proceedings.mlr.press/v267/sun25h.html).

### 4. MIRAS: `behrouz2026miras`

**Estado:** ICLR 2026, corroborado por el PDF indexado de OpenReview y la página del autor. Copia arXiv accesible de 2025. Organiza el diseño alrededor de la memoria asociativa, su objetivo interno, la retención y el algoritmo de actualización. Esta separación resulta útil para diseñar ablaciones comprensibles. No exige implementar sus tres modelos completos. En MARS-TITAN puede servir para justificar por qué se cambia un componente cada vez. [Ficha](https://openreview.net/forum?id=gZyEJ2kMow), [confirmación del autor](https://alibehrouz.com/publications/).

### 5. ATLAS: `behrouz2025atlas`

**Estado:** versión arXiv de 2025 comprobada. El primer autor lo lista en ICML 2026, sin ficha independiente de actas accesible verificada aquí. Extiende la actualización de memoria para considerar contexto presente y pasado. Los autores distinguen esta memorización del aprendizaje persistente entre contextos independientes. La idea aprovechable sería un historial acotado de observaciones pasadas. No implica que el sistema reescriba su código ni justifica reutilizar datos posteriores al instante de predicción. [Artículo](https://arxiv.org/abs/2505.23735), [estado declarado por el autor](https://alibehrouz.com/publications/).

### 6. SEAL: `zweiger2025seal`

**Estado:** NeurIPS 2025, revisado por pares. El modelo genera datos y directrices para su propio ajuste. Un bucle de aprendizaje por refuerzo premia adaptaciones útiles. Esta actualización persistente es distinta de mantener un estado de memoria contextual. El trabajo aporta una inspiración sobre adaptación, pero reproducir su generación, ajuste y refuerzo ampliaría demasiado el proyecto. Su implementación completa queda fuera del núcleo experimental. La autoría de las actas difiere de la del preprint. La bibliografía sigue el PDF publicado. [Publicación](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6b41e04c41726e2a60e456d0a2b961ab-Abstract-Conference.html).

### 7. Nested Learning: `behrouz2025nested`

**Estado:** NeurIPS 2025, revisado por pares. Presenta problemas de optimización anidados, memoria a distintas escalas y el módulo HOPE. Amplía el contexto de Titans y muestra que el estado del arte ha seguido avanzando. Es adecuado para la discusión y las posibles extensiones. No es necesario añadir HOPE a la comparativa mínima. Sus resultados tampoco sustituyen una evaluación propia sobre datos financieros. [Publicación](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html).

### 8. DLinear: `zeng2023dlinear`

**Estado:** AAAI 2023, revisado por pares. Este trabajo muestra el valor de modelos lineales sencillos frente a arquitecturas de series temporales más complejas en sus pruebas de rendimiento. DLinear es una referencia útil por su bajo coste y facilidad de interpretación experimental. No implica que los Transformers sean siempre inadecuados. Hay que ajustarlo al mismo objetivo residual, entradas y horizonte, y evitar comparar predicción de precios con predicción de retornos. [Artículo](https://ojs.aaai.org/index.php/AAAI/article/view/26317).

### 9. PatchTST: `nie2023patchtst`

**Estado:** ICLR 2023, revisado por pares. Utiliza fragmentos de la serie como tokens y comparte pesos entre canales. Ofrece una referencia de atención temporal con una representación más económica que un token por observación. Para el proyecto conviene reducir profundidad y anchura y medir el coste real. Su diseño independiente por canal no constituye por sí solo un mecanismo de fusión multimodal. Las entradas auxiliares deben incorporarse con un protocolo explícito. [Versión de autor](https://arxiv.org/abs/2211.14730v2).

### 10. TCN: `bai2018tcn`

**Estado:** preprint técnico de 2018. Publicación revisada por pares no confirmada. Estudia convoluciones temporales dilatadas y conexiones residuales frente a redes recurrentes. Es una referencia compacta útil para comprobar si una memoria adaptativa mejora sobre un campo receptivo fijo. Aquí «causal» significa que la convolución no utiliza observaciones futuras. No significa que el modelo identifique efectos económicos causales. El campo receptivo debe cubrir un historial comparable al de los demás modelos. [Artículo](https://arxiv.org/abs/1803.01271).

### 11. FinBERT: `araci2019finbert`

**Estado:** trabajo de máster de 2019 depositado en arXiv, no artículo revisado por pares verificado. Adapta BERT al lenguaje financiero y evalúa sentimiento. Puede utilizarse como extractor congelado de noticias en inglés, guardando representaciones para todos los modelos. Su éxito en sentimiento no prueba que prediga retornos. Hay que registrar checkpoint y fecha, porque la disponibilidad histórica del encoder también afecta a la interpretación de un backtest. No debe aplicarse sin validación a noticias chinas. [Trabajo](https://arxiv.org/abs/1908.10063), [modelo asociado](https://huggingface.co/ProsusAI/finbert).

### 12. StockNet: `xu2018stocknet`

**Estado:** ACL 2018, revisado por pares. Combina tuits e historial de precios en una tarea de dirección bursátil con variables latentes temporales. Es un antecedente directo de predicción financiera multimodal y justifica comparar texto, precios y su combinación. Su muestra, periodo y objetivo difieren del proyecto. No permite deducir una mejora universal al añadir noticias ni atribuir a cada texto un efecto causal sobre el precio. [Actas](https://aclanthology.org/P18-1183/).

### 13. Texto y tablas financieras: `koval2024financial`

**Estado:** Findings of EMNLP 2024, revisado por pares. Predice beneficios trimestrales frente a expectativas de analistas mediante documentos, variables financieras y contexto macroeconómico. Aporta un diseño con referencias unimodales y análisis de la fusión. Resulta relevante para organizar las comparaciones, aunque la sorpresa de beneficios no equivale a una rentabilidad residual diaria. Sus resultados deben describirse dentro de esa tarea, sin trasladar directamente magnitudes al proyecto. [Actas](https://aclanthology.org/2024.findings-emnlp.486/).

### 14. Deep Ensembles: `lakshminarayanan2017ensembles`

**Estado:** NeurIPS 2017, revisado por pares. Combina modelos entrenados de forma independiente para estimar incertidumbre predictiva. Es una referencia práctica frente a aproximaciones bayesianas más costosas. Para una GPU pequeña se pueden entrenar miembros de forma secuencial, a costa de más tiempo total. El desacuerdo entre redes no asegura buena calibración en cambios de régimen. Deben medirse cobertura, anchura y puntuaciones predictivas, además del error de la media. [Artículo](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html).

### 15. MC dropout: `gal2016dropout`

**Estado:** ICML 2016, revisado por pares. Relaciona dropout con inferencia bayesiana aproximada y permite obtener varias predicciones estocásticas con un único modelo. Es una alternativa de bajo almacenamiento, aunque multiplica las pasadas de inferencia. En una red con memoria adaptable, cada pasada debe partir de una copia del mismo estado: de otro modo se mezclarían incertidumbre y evolución de la memoria. Su interpretación aproximada no garantiza intervalos calibrados en mercados cambiantes. [Actas](https://proceedings.mlr.press/v48/gal16.html).

### 16. EnbPI: `xu2021enbpi`

Entrada canónica compartida con la revisión financiera: [finance-sources.json](finance-sources.json) y [finance.bib](finance.bib).

**Estado:** ICML 2021, revisado por pares. Construye intervalos secuenciales a partir de predictores bootstrap y residuos. Su teoría ofrece cobertura marginal aproximadamente válida bajo condiciones sobre los errores y su dependencia. No una garantía sin hipótesis para cualquier serie. Ayuda a plantear la evaluación temporal de incertidumbre. Una red residual financiera puede aportar el predictor base, pero la calibración solo debe incorporar errores cuyo objetivo ya se haya observado. [Actas y título publicado](https://proceedings.mlr.press/v139/xu21h.html).

### 17. ACI: `gibbs2021aci`

Entrada canónica compartida con la revisión financiera: [finance-sources.json](finance-sources.json) y [finance.bib](finance.bib).

**Estado:** NeurIPS 2021, revisado por pares. Adapta la calibración conforme cambia la distribución y estudia la frecuencia de cobertura a largo plazo. Esta propiedad no equivale a cobertura condicional para cada activo o fecha, ni garantiza intervalos estrechos. Para el proyecto interesa evaluar cobertura por periodos y anchura, junto al promedio global. El retraso con que se observa un retorno futuro debe respetarse también al actualizar el calibrador. [Artículo](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).

## Auditoría bibliográfica de FinMultiTime

Hay cuatro detalles del PDF que condicionan el experimento. Se eliminan fechas de anuncio de algunas tablas y se propagan cifras contables dentro de su ventana de reporte. Además, se generan gráficos por semestres y se describe un universo mayormente referido a abril de 2025. El PDF v2 informa de 5.586 valores, mientras la ficha arXiv indica 5.105. Estos detalles exigen una auditoría propia de disponibilidad, cobertura y selección. La resolución intradía de las noticias no acredita precios intradía: la tabla 2 describe OHLCV diario. [PDF, secciones 2.1–2.2 y tabla 2](https://arxiv.org/pdf/2506.05019v2), [ficha arXiv](https://arxiv.org/abs/2506.05019v2).

El repositorio de datos enlazado por el artículo es [Wenyan0110 en Hugging Face](https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting). Es público y su ficha declara MIT. Esa declaración no resuelve por sí sola los derechos sobre noticias y datos de terceros. El visor muestra nueve imágenes, no el tamaño completo del archivo descargable. La ficha enumera 4.213 valores estadounidenses y 858 chinos: tampoco coincide exactamente con los recuentos del artículo. No se ha localizado un repositorio GitHub oficial de código en los enlaces primarios revisados. El botón GitHub automático de [HF Papers](https://huggingface.co/papers/2506.05019) conduce a otro proyecto, PyDGC, y no debe citarse como implementación de FinMultiTime.

Las siguientes son decisiones metodológicas propuestas para el proyecto, no propiedades ya verificadas de la copia local:

- Usar la fecha y hora de publicación de una noticia, con zona horaria y corte de decisión explícitos. Si solo existe la fecha, aplicar una convención conservadora de disponibilidad y medir su efecto.
- Incorporar cifras contables desde su fecha efectiva de presentación. Si no puede recuperarse, excluirlas del experimento principal y explicar la limitación. Un retraso fijo es solo una aproximación que requiere sensibilidad.
- Reconstruir gráficos de ventanas que terminen en el instante de decisión. Una imagen semestral no puede informar predicciones realizadas antes de que ese semestre termine. Además, un gráfico de OHLCV es otra representación del precio, no una fuente independiente de información.
- Calcular recuentos, cobertura por modalidad y población final a partir del manifiesto local. Si no se dispone de composición histórica y empresas desaparecidas, describir la evaluación como una muestra retrospectiva seleccionada.
- Ajustar normalización, imputación y selección de variables dentro de cada entrenamiento. Una extracción de texto congelada es reutilizable, pero se debe registrar su posible contaminación temporal por preentrenamiento posterior al periodo estudiado.

## Alcance razonable para MARS-TITAN

La contribución defendible consiste en comprobar si una memoria adaptativa compacta aporta valor incremental al pronóstico residual, frente a alternativas comparables. La literatura consultada motiva esa pregunta. No proporciona su respuesta. El uso de una variable residual, una máscara temporal o una ablación no identifica por sí mismo un efecto causal. Las expresiones adecuadas son «capacidad predictiva», «aportación incremental» y «sensibilidad a la modalidad».

Una implementación mínima puede extraer el texto una sola vez con pesos congelados, proyectarlo junto a las variables temporales y añadir una memoria pequeña. Se propone comenzar por precios y noticias y ampliar a tablas o imágenes únicamente cuando su disponibilidad esté justificada. Si se cambia el método de fusión a la vez que la memoria, será difícil atribuir una mejora al componente estudiado. Esta es una decisión de diseño inferida de los mecanismos de [Titans](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html) y la organización de [MIRAS](https://arxiv.org/abs/2504.13173), no una arquitectura validada por esas publicaciones.

| Comparación propuesta | Pregunta que permite contestar |
|---|---|
| Predicción cero y regresión lineal regularizada | ¿Existe señal suficiente para justificar modelos más complejos? |
| DLinear, TCN y PatchTST pequeños | ¿Mejora el método frente a familias temporales distintas? |
| Mismo modelo multimodal sin módulo de memoria | ¿Compensa añadir el módulo bajo un presupuesto comparable? |
| Misma memoria con y sin actualización durante evaluación | ¿La adaptación añade valor sobre la capacidad aprendida durante entrenamiento? |
| Precios frente a precios y texto con fusión fija | ¿Qué aporta la modalidad textual bajo el mismo protocolo? |
| Memoria con retención y actualización simplificadas | ¿Qué parte de su regla explica el comportamiento observado? |

Cada fila necesita el mismo objetivo, universo, fechas y entradas disponibles. Igualar aproximadamente parámetros ayuda, pero no sustituye registrar tiempo de entrenamiento, latencia y memoria máxima. Los hiperparámetros se eligen en validación temporal. El conjunto final de prueba no decide qué arquitectura, semillas o periodos se publican.

La memoria debe seguir una secuencia operativa inequívoca: incorpora observaciones disponibles, produce una predicción y solo más adelante puede incorporar el objetivo cuando se haya realizado. Una actualización autosupervisada sobre entradas disponibles y un ajuste supervisado tras recibir el retorno son mecanismos diferentes y se documentarán por separado. Las particiones temporales deben reconstruir su estado inicial sin arrastrar información de otra evaluación. También hay que decidir si cada activo tiene estado propio o si existe una memoria compartida. Mezclar activos mediante el orden arbitrario de los lotes no es una justificación económica.

ATLAS, SEAL y HOPE ayudan a situar el proyecto, pero implementar los tres además de Titans no es necesario para resolver esta pregunta. No se incluye generación autónoma de nuevas reglas, modificación del código, aprendizaje por refuerzo de una política de ajuste ni un agente de negociación. Esa delimitación mantiene el esfuerzo en una comparación reproducible y permite explicar una ausencia de mejora como resultado válido.

## Incertidumbre y presupuesto de 8 GB

No se ha ejecutado ningún entrenamiento en esta revisión. La viabilidad con una RTX 4070 de 8 GB queda pendiente de un piloto que mida memoria máxima, tiempo por época y tiempo de inferencia sobre la muestra real. No debe darse por demostrada a partir del número de parámetros. Las activaciones, la diferenciación de las actualizaciones internas y el historial retenido también consumen memoria.

Como primera opción se propone una salida de cuantiles o una cabeza probabilística pequeña, seguida de calibración temporal. Un ensemble de tres modelos compactos, entrenados de forma secuencial, puede servir de contraste si el piloto permite asumir ese coste. Tres es una propuesta de presupuesto, no una cifra óptima obtenida de la literatura. MC dropout ofrece otra referencia, con copias independientes del estado de memoria. Se debe elegir una alternativa principal y evitar multiplicar variantes sin una pregunta concreta. [Deep Ensembles](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html), [MC dropout](https://proceedings.mlr.press/v48/gal16.html).

Los intervalos se evaluarán por cobertura, anchura y periodo, además de una puntuación adecuada como pérdida de cuantiles o interval score. Una cobertura cercana al nivel objetivo puede conseguirse con intervalos poco útiles, por lo que hay que valorar ambas dimensiones. EnbPI y ACI ofrecen referencias para calibración secuencial. Sus garantías deben describirse con sus condiciones y sin convertir un promedio temporal en una garantía para cada operación. [EnbPI](https://proceedings.mlr.press/v139/xu21h.html), [ACI](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).

Para una futura ejecución se usará `uv`, se comprobarán `nvidia-smi` y `torch.cuda.is_available()`, y se seleccionará `cuda:0` explícitamente. Las representaciones de texto se procesarán por lotes y podrán conservarse en disco, con versión del modelo y de los datos. El resultado experimental distinguirá mejora predictiva, calidad de incertidumbre y utilidad financiera tras costes. Ninguna se da por supuesta en esta revisión.
