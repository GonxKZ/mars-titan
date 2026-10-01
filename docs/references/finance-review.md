# Revisión financiera para MARS-TITAN

Autor: Gonzalo. Fecha de verificación: 18 de septiembre de 2026.

Esta selección reúne 18 fuentes para justificar el diseño financiero y la evaluación del proyecto. Es una revisión dirigida a las decisiones del proyecto, no una revisión sistemática exhaustiva. Distingo los resultados publicados de las propuestas para MARS-TITAN. Ninguna de las fuentes demuestra por adelantado que su arquitectura vaya a generar rentabilidad.

El catálogo [finance-sources.json](finance-sources.json) conserva enlaces, versiones, estado de acceso y derechos conocidos. [finance.bib](finance.bib) contiene las referencias bibliográficas. Un PDF disponible para lectura no implica permiso para subirlo al repositorio. `unknown` significa que no se ha verificado una licencia de redistribución. Las fuentes vivas sin año único tienen `year: null`. Su fecha de consulta se registra por separado.

Cada referencia se vincula con los objetivos de datos, retorno residual, arquitectura, comparativa, evaluación o análisis crítico mediante los números 1 a 6. Esta numeración permite seguir la aportación de la bibliografía y no determina la estructura de carpetas.

## Fuentes seleccionadas

### 1. SEC: marcas temporales de EDGAR

U.S. Securities and Exchange Commission, sin fecha única. [Webmaster Frequently Asked Questions](https://www.sec.gov/about/webmaster-frequently-asked-questions). Fases 1 y 5.

La documentación distingue el periodo del informe, su aceptación y sus cambios posteriores. También advierte que aceptación y disponibilidad pública no son simultáneas. Para alinear tablas y textos contables propongo conservar ambas fechas y aplicar una regla explícita de disponibilidad antes de predecir. La fecha de cierre del trimestre no sirve como fecha de publicación. Acceso: HTML oficial completo. Reutilización: permitida por la SEC para contenido gubernamental y filings públicos, con excepciones para ciertos materiales de terceros.

### 2. Sesgo por exclusiones bursátiles

Tyler Shumway (1997). [The Delisting Bias in CRSP Data](https://doi.org/10.1111/j.1540-6261.1997.tb03818.x). [PDF del autor](https://www.tylergshumway.org/Shumway-DelistingBiasCRSP-1997.pdf). Fases 1, 5 y 6.

El artículo documenta que omitir retornos de exclusión puede sesgar los resultados. Motiva revisar empresas desaparecidas, identificadores históricos y bajas del universo. Un conjunto formado con componentes actuales de un índice necesita una limitación explícita si no permite reconstruir su composición histórica. El estudio no garantiza que otra base esté libre de este problema. Acceso: PDF del autor, respuesta HTTP comprobada. Redistribución: desconocida. El documento remite a condiciones de JSTOR.

### 3. Modelo factorial de referencia

Eugene F. Fama y Kenneth R. French (1993). [Common risk factors in the returns on stocks and bonds](https://doi.org/10.1016/0304-405X(93)90023-5). Fases 2, 4 y 5.

Proporciona la base para separar exposición al mercado, tamaño y valor en acciones. Conviene evitar una confusión bibliográfica: sus cinco factores conjuntos de acciones y bonos no son los cinco factores de acciones de 2015. Para el proyecto usaría FF3 como referencia de residualización y declararía ventana, intercepto y frecuencia. Acceso: ficha y resumen editoriales. PDF libre primario no confirmado. Redistribución del texto: no autorizada de forma abierta en lo verificado.

### 4. Contraste con cinco factores

Eugene F. Fama y Kenneth R. French (2015). [A five-factor asset pricing model](https://doi.org/10.1016/j.jfineco.2014.10.010). Fases 2, 4, 5 y 6.

Añade rentabilidad operativa e inversión al modelo de acciones. Sirve para comprobar si una señal atribuida a MARS-TITAN depende del modelo de riesgo usado para definir la variable objetivo. El propio trabajo identifica patrones que el modelo no explica bien. No constituye una definición universal del riesgo. Acceso: ficha editorial y [manuscrito de 2014](https://ssrn.com/abstract=2287202), cuya descarga no se pudo verificar. SSRN indica derechos reservados y permiso necesario para reutilizar.

### 5. Datos y versiones de los factores

Kenneth R. French, sin fecha única. [Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Fases 1, 2 y 5.

Es el origen primario para las series y sus convenciones. La página documenta el paso de CRSP FIZ a CIZ en enero de 2025 y ofrece algunos archivos históricos. Propongo registrar fecha de descarga, definición, unidades y hash. Una versión actual puede incorporar revisiones. La existencia de factores diarios no demuestra su disponibilidad operativa diaria inmediata. Acceso: datos y documentación públicos. Redistribución: licencia específica no verificada.

### 6. Referencias financieras con aprendizaje automático

Shihao Gu, Bryan Kelly y Dacheng Xiu (2020). [Empirical Asset Pricing via Machine Learning](https://doi.org/10.1093/rfs/hhaa009). [Manuscrito del autor, 2019](https://dachxiu.chicagobooth.edu/download/ML_BKP.pdf). Fases 2, 4, 5 y 6.

Compara familias de modelos para predecir retornos, incluyendo regresiones, árboles y redes. Es una referencia útil para que la comparación no se limite a arquitecturas profundas. Sus conclusiones pertenecen a su universo, frecuencia y variables. No pueden trasladarse directamente a datos multimodales diarios. Propongo reutilizar el principio comparativo, con los mismos cortes y presupuesto de selección para todos los candidatos. Redistribución del manuscrito: desconocida.

### 7. Interpretación del Sharpe

Andrew W. Lo (2002). [The Statistics of Sharpe Ratios](https://doi.org/10.2469/faj.v58.n4.2453). [Resumen primario de CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios). Fases 5 y 6.

Estudia el error de estimación del Sharpe y el efecto de la dependencia temporal. Justifica acompañar la estimación puntual de incertidumbre y revisar los supuestos de anualización. Multiplicar mecánicamente por la raíz del número de periodos puede distorsionar la comparación cuando hay autocorrelación. Acceso: resumen primario. No se verificó un PDF completo libre del autor o editor. Redistribución: desconocida.

### 8. Costes de negociación

Andrea Frazzini, Ronen Israel y Tobias J. Moskowitz (2018). [Trading Costs](https://www.aqr.com/insights/research/working-paper/trading-costs). Fases 5 y 6.

El trabajo de AQR relaciona costes e impacto con características de las operaciones y del mercado. Motiva separar rentabilidad bruta y neta, y estudiar sensibilidad al coste y al tamaño negociado. Las condiciones de ejecución de una institución no equivalen a las de esta simulación. Acceso: ficha primaria y enlace a SSRN. Texto completo no verificado. El PDF automático de la página contiene solo un resumen de dos páginas. Redistribución: desconocida.

### 9. Probabilidad de sobreajuste del backtest

David H. Bailey, Jonathan M. Borwein, Marcos López de Prado y Qiji Jim Zhu (2017). [The Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322). [Manuscrito del autor, 2015](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf). Fases 4, 5 y 6.

Plantea PBO mediante validación combinatoria para estudiar cómo se degrada la posición relativa de la estrategia seleccionada. Su aplicación requiere resultados comparables de las alternativas ensayadas. Lo considero un diagnóstico complementario. Las particiones combinatorias no reproducen automáticamente una operación causal walk-forward ni solucionan cambios de régimen. Deben explicarse su construcción y sus límites. Redistribución del manuscrito: desconocida.

### 10. Sharpe deflactado

David H. Bailey y Marcos López de Prado (2014). [The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality](https://doi.org/10.3905/jpm.2014.40.5.094). [PDF del autor](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf). Fases 4, 5 y 6.

El DSR ajusta la evaluación por selección entre ensayos y por no normalidad. Requiere declarar cómo se estima el número efectivo de alternativas, además de momentos y longitud de la muestra. No debe describirse como probabilidad universal de que exista una ventaja negociable. Tampoco corrige datos filtrados desde el futuro. Acceso: manuscrito completo. El PDF conserva todos los derechos reservados.

### 11. Descubrimientos y pruebas múltiples

Campbell R. Harvey, Yan Liu y Heqing Zhu (2016). [… and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhv059). [PDF del autor](https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF). Fases 4, 5 y 6.

Explica por qué la búsqueda repetida de factores altera la interpretación de la significación. Para el proyecto implica registrar también pruebas fallidas, modalidades descartadas y criterios de selección. Su umbral empírico no debería convertirse en una constante universal aplicada a cualquier métrica o experimento. Acceso: artículo completo en la página universitaria del autor. El PDF indica derechos reservados y permisos a través de Oxford University Press.

### 12. Calibración adaptativa

Isaac Gibbs y Emmanuel Candès (2021). [Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html). [PDF](https://proceedings.neurips.cc/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf). Fases 3, 4, 5 y 6.

ACI adapta el nivel de calibración según se observan errores. La garantía de frecuencia temporal a largo plazo no equivale a cobertura condicional en cada régimen, activo o fecha. Su utilidad debe medirse junto con amplitud de intervalos y comportamiento tras cambios. En horizontes múltiples, la actualización solo puede usar etiquetas ya observadas. Acceso: actas completas de NeurIPS. Redistribución: licencia específica no verificada.

### 13. Cobertura sin intercambiabilidad

Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas y Ryan J. Tibshirani (2023). [Conformal prediction beyond exchangeability](https://doi.org/10.1214/23-AOS2276). [Preprint](https://arxiv.org/pdf/2202.13415). Fases 3, 5 y 6.

El trabajo analiza variantes ponderadas y cuantifica pérdidas de cobertura cuando falla la intercambiabilidad. Ayuda a formular límites concretos para datos financieros dependientes y distribuciones cambiantes. Ponderar observaciones recientes no elimina cualquier deriva ni garantiza automáticamente cobertura nominal en un panel bursátil. Acceso: preprint completo. Licencia verificada: distribución no exclusiva de arXiv. No concede por sí sola redistribución general por terceros.

### 14. Intervalos secuenciales con ensembles

Chen Xu y Yao Xie (2021). [Conformal prediction interval for dynamic time-series](https://proceedings.mlr.press/v139/xu21h.html). [PDF](https://proceedings.mlr.press/v139/xu21h/xu21h.pdf). Fases 3, 4, 5 y 6.

EnbPI es una referencia útil para separar la calidad del predictor de la calibración del intervalo. Su cobertura marginal aproximada depende de supuestos sobre los errores y la calidad del estimador, incluyendo condiciones de mezcla. Los resultados no autorizan afirmar cobertura exacta para cualquier cambio estructural ni para todas las acciones simultáneamente. Acceso: artículo completo de ICML en PMLR. Redistribución: licencia específica no verificada.

### 15. Manual abierto de predicción

Rob J. Hyndman y George Athanasopoulos (2021). [Forecasting: Principles and Practice, tercera edición](https://otexts.com/fpp3/). Fases 4 y 5.

El libro completo puede leerse legalmente en línea. La [sección 5.10](https://otexts.com/fpp3/tscv.html) explica la evaluación con origen móvil y varios horizontes. Es útil para describir el protocolo y los modelos sencillos antes de introducir memoria adaptativa. Su desarrollo no sustituye los controles específicos del panel financiero. La edición web se actualiza: anoto consulta y versión. No se verificó un PDF oficial completo ni una licencia concreta de redistribución.

### 16. Marco económico del rendimiento residual

John H. Cochrane (2005). [Asset Pricing, edición revisada](https://www.johnhcochrane.com/asset-pricing). Princeton University Press, ISBN 9780691121376. Fases 2, 5 y 6.

Referencia de apoyo para estudiar primas de riesgo y evaluación respecto a un modelo factorial. La página del autor enlaza la editorial, material docente y un capítulo de muestra. Se registra como libro comercial, sin atribuirle una lectura completa en esta revisión ni ofrecer copias no autorizadas. Los materiales docentes tienen condiciones propias: el [programa del curso](https://www.johnhcochrane.com/syllabus-and-reading-list) prohíbe redistribuir sus PDFs. El acceso al curso no convierte el libro en una obra abierta.

### 17. Manual profesional de aprendizaje financiero

Marcos López de Prado (2018). [Advances in Financial Machine Learning](https://www.wiley-vch.de/en?isbn=9781119482086&option=com_eshop&view=product). Wiley, ISBN 9781119482086. Fases 1, 4, 5 y 6.

El índice editorial permite localizar capítulos de validación financiera, selección de hiperparámetros, backtesting y cambios estructurales. Es una referencia bibliográfica para profundizar en purga y embargo cuando se solapan etiquetas. La duración debe justificarse con el horizonte y la construcción de muestras. Acceso: ficha e índice del editor, no texto completo. Libro comercial, sin permiso abierto verificado. No se ha descargado una copia externa.

### 18. Cambios de régimen

James D. Hamilton (2008). [Regime-Switching Models](https://econweb.ucsd.edu/~jhamilto/pub_metrics), en The New Palgrave Dictionary of Economics. Fases 3, 4, 5 y 6.

Ofrece contexto econométrico para contrastar la adaptación de memoria con una explicación explícita por regímenes. Una clasificación estimada retrospectivamente debe distinguirse de un estado inferido con información disponible al predecir. Se verificó la referencia en la lista del autor. El [manuscrito de 2005](https://econweb.ucsd.edu/~jhamilton/palgrav1.pdf) está localizado, pero no se confirmó descarga por un problema del certificado TLS. Redistribución: desconocida. Su revisión detallada queda pendiente.

## Implicaciones para el diseño del proyecto

Las siguientes decisiones son una propuesta metodológica derivada de las fuentes, no resultados experimentales ni recetas atribuidas literalmente a sus autores.

### Definición de la variable objetivo y de la información disponible

Para un horizonte diario, fijaría primero el modelo `rᵉ(i,t) = α(i) + β(i)' f(t) + ε(i,t)`. En la fecha de decisión `t`, estimaría los parámetros con una ventana pasada y datos cuya disponibilidad esté acreditada. La etiqueta futura sería `y_res(i,t+1) = rᵉ(i,t+1) − α̂(i,t) − β̂(i,t)' f(t+1)`. El retorno en exceso es `rᵉ = r − r_f`. Si se decide conservar el intercepto, el objetivo pasa a ser `rᵉ − β̂'f`, una magnitud distinta que debe nombrarse expresamente. El contraste FF3/FF5 examinaría la sensibilidad a esta elección. [Fama y French, 1993](https://doi.org/10.1016/0304-405X(93)90023-5), [Fama y French, 2015](https://doi.org/10.1016/j.jfineco.2014.10.010).

El factor realizado en `t+1` puede formar parte de la etiqueta evaluada después. No puede entrar como entrada conocida en `t`. Las estimaciones tampoco deben usar una regresión sobre toda la muestra. Las versiones revisadas de factores merecen un registro específico. Predecir esta variable objetivo no garantiza por sí solo neutralidad factorial de los pesos negociados: las exposiciones de cartera se comprueban por separado. [Biblioteca de French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

Para cada noticia, tabla o informe conservaría `event_time`, `published_at`, `available_at` y `ingested_at` cuando existan. Si solo se conoce la fecha, declararía una regla conservadora y estudiaría su sensibilidad. Imágenes y gráficos deberían construirse con datos truncados al corte. Cambios de ticker, componentes históricos, exclusiones y revisiones contables requieren auditoría. No deben inferirse del estado actual. [SEC](https://www.sec.gov/about/webmaster-frequently-asked-questions), [Shumway, 1997](https://doi.org/10.1111/j.1540-6261.1997.tb03818.x).

### Comparación y validación

| Decisión | Comparación propuesta | Qué permite interpretar |
| --- | --- | --- |
| Valor del aprendizaje | Predicción cero, modelo lineal regularizado, árboles y arquitectura propuesta | Ganancia frente a referencias financieras sencillas |
| Valor de las modalidades | Precio solo. Adición de texto, tablas e imágenes. Combinación completa | Aportación incremental con idéntico universo y cortes |
| Valor de la memoria | Sin memoria. Memoria congelada. Memoria adaptativa | Efecto de adaptación manteniendo comparable el resto |
| Incertidumbre | Predictor puntual con calibración común. ACI. EnbPI si es viable | Si mejora el predictor, el calibrador o ambos |
| Robustez financiera | Retorno bruto y neto. FF3 y FF5. Subperiodos definidos previamente | Dependencia de costes, modelo de riesgo y entorno |

La tabla es una propuesta de ablaciones. Las referencias financieras se apoyan en [Gu, Kelly y Xiu](https://doi.org/10.1093/rfs/hhaa009). Los contrastes de incertidumbre, en [ACI](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) y [EnbPI](https://proceedings.mlr.press/v139/xu21h.html).

La prueba principal seguiría cortes cronológicos comunes a todo el panel, con selección dentro del pasado de cada ventana. Imputación, escalado, selección de variables, estimación factorial y calibración formarían parte de ese corte. Se eliminarían del entrenamiento las etiquetas que todavía no hayan terminado o no estén disponibles al predecir. La separación adicional se justificaría por el solapamiento real. Un embargo arbitrario no garantiza causalidad. Se registrarían reinicios de memoria y actualizaciones en línea para que cada predicción pueda reconstruirse. [Hyndman y Athanasopoulos, sección 5.10](https://otexts.com/fpp3/tscv.html), [López de Prado, índice editorial](https://www.wiley-vch.de/en?isbn=9781119482086&option=com_eshop&view=product).

### Evaluación financiera e incertidumbre

Además de error predictivo y correlación de ordenación por fecha, reportaría rentabilidad neta, Sharpe con incertidumbre, caída máxima, rotación, exposición bruta y neta, y costes de ejecución. La simulación declararía el instante negociable, el precio utilizado, préstamo de títulos si hay cortos y límites de liquidez. Para evitar ambigüedad, distinguiría volumen negociado `Σ|w − w_previo_ajustado|` de rotación de un solo sentido `½Σ|w − w_previo_ajustado|`. El coste por unidad se aplicaría a la magnitud correspondiente. [Frazzini, Israel y Moskowitz](https://www.aqr.com/insights/research/working-paper/trading-costs), [Lo](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios).

Registraría todas las configuraciones evaluadas y reservaría un bloque final sin selección reiterada. PBO y DSR se usarían como diagnósticos con sus supuestos declarados. No reemplazan ese historial ni convierten el backtest en una prueba de rentabilidad futura. [PBO](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf), [DSR](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), [Harvey, Liu y Zhu](https://doi.org/10.1093/rfs/hhv059).

En incertidumbre separaría cobertura, amplitud e infracobertura por tiempo, activo y entorno. Un intervalo marginal del 90 % no permite afirmar una probabilidad del 90 % para un activo concreto ni cobertura simultánea del panel. Las actualizaciones del calibrador esperarían a que madure el horizonte de retorno. La pérdida de intercambiabilidad y los cambios de distribución se tratarían como limitaciones explícitas y verificables. [Barber y colaboradores](https://doi.org/10.1214/23-AOS2276), [Gibbs y Candès](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).

## Límites de esta revisión

Se han verificado procedencia y metadatos mediante páginas de autores, organismos, universidades, congresos y editoriales. Ocho enlaces de PDF completo se han comprobado. Las descargas locales, hashes y conservación de versiones deben registrarse aparte. No se ha eludido ningún acceso restringido. Las fichas de libros comerciales y los resúmenes no equivalen a una lectura completa. Hamilton queda pendiente de lectura detallada. Los resultados de un modelo de regímenes no deben atribuirse a este capítulo sin consultar su desarrollo.

Estas fuentes no verifican la calidad del conjunto de datos local, la composición histórica de sus universos ni las licencias de cada noticia o imagen. Tampoco comparan MARS-TITAN con sistemas propietarios de inversión. El contraste con prácticas cuantitativas profesionales procede de métodos públicos de investigación y ejecución. Cualquier afirmación de superioridad necesitará resultados propios reproducibles y netos de costes.
