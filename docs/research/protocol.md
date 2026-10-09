# Protocolo de investigación

Autor: Gonzalo García Lama. Versión de trabajo ampliada: 18 de septiembre de 2026. La [ampliación de investigación](research-expansion.md) concreta memoria, recurrencia, datos macro y presupuesto.

## Pregunta y alcance

El estudio compara soluciones para predecir retornos residuales de los mercados estadounidense y chino disponibles en FinMultiTime. La campaña de referencias amplía el recorrido a todo el universo admisible, con comparaciones por mercado y conjunta. Se quiere determinar si una memoria adaptativa de eventos aporta información útil frente a modelos sin esa memoria, bajo las mismas condiciones de datos y evaluación. El trabajo no depende de demostrar superioridad: una comparación que identifique un efecto nulo o un coste desproporcionado también responde a la pregunta.

La contribución prevista comprende el protocolo temporal, una adaptación identificable de Titans-MAC y una comparación de sus ampliaciones. La [corrección arquitectónica del 8 de octubre](titans-mac-architecture.md) conserva la referencia GRU con banco episódico y añade un Transformer compacto, Titans-MAC sin las ampliaciones y MARS-TITAN con ellas. Cada mecanismo se contrasta por separado. La adaptación financiera no reproduce por sí misma los resultados del artículo y su novedad debe justificarse frente al [estado del arte](../references/neural-review.md).

La edición histórica desde 2000 conserva todos los datos utilizables con ausencias explícitas y las mismas filas entre modelos. La comparación estricta permanece separada. El corpus histórico y sus objetivos residuales se completaron y verificaron el 9 de octubre de 2026 ([recibo](../../reports/data/historical-edition-v3-targets-20261009.json)), pero entrenamientos, postentrenamientos, pilotos y evaluaciones científicas siguen sin ejecutarse y la protección local continúa activa. La [protección del aprendizaje](../../tests/README.md#protección-del-aprendizaje) detiene cada punto de entrada que ajusta parámetros antes de abrir fuentes o crear salidas, en Python y en los ejecutables nativos. Las comprobaciones técnicas de implementación no levantan esa condición.

Quedan fuera del núcleo las operaciones reales, la conexión a un bróker, las recomendaciones de inversión, la autoedición de modelos al estilo SEAL y el entrenamiento de grandes codificadores multimodales. La ampliación incorpora experimentos separados de refuerzo predictivo, decisiones financieras simuladas y ajuste conjunto. Su criterio principal sigue siendo el error predictivo, no el beneficio simulado. Varias escalas de memoria y kernels propios siguen condicionados a evidencia. La campaña china y la conjunta requieren resolver sus publicaciones, identidad y factor de mercado, sin dar por utilizables las cuatro modalidades solo porque existan sus archivos.

## Hipótesis registrables

| Hipótesis | Contraste previsto | Qué permitiría sostenerla |
| --- | --- | --- |
| H1. La memoria aporta señal adicional | Modelo completo frente al mismo codificador sin memoria. | Diferencia de MAE fuera de muestra favorable, intervalo por bloques temporales y consistencia entre ventanas. Informar también tamaño y coste del efecto. |
| H2. La escritura selectiva importa | Sorpresa propuesta frente a escritura uniforme y sorpresa puramente predictiva. | Mejora que no se explique solo por más parámetros, más actualizaciones o más cómputo. |
| H3. La representación multimodal afecta al resultado | Comparar codificadores y formas de fusión conservando las cuatro modalidades y las mismas muestras. | Diferencias que no se expliquen por cambios de cobertura. Una ablación que retire una modalidad requiere revisar explícitamente el alcance. |
| H4. La incertidumbre ayuda a abstenerse | Riesgo-cobertura, calibración y simulación con/sin abstención. | Menor error o pérdida a coberturas comparables. Informar operaciones descartadas, costes y periodos sin señal. |

Métrica primaria propuesta: **MAE del retorno residual**. Rank IC medio por sesión será una métrica secundaria prioritaria. Los resultados económicos son secundarios y no sustituyen la pregunta predictiva. La hipótesis principal, la métrica y la familia de comparaciones se congelarán antes de ejecutar el test final. No se fija una mejora porcentual esperada sin piloto que justifique su relevancia práctica.

## Universo y selección del subconjunto

La ampliación del 23 de septiembre de 2026 introduce [dos cohortes diferenciadas](../data/cohort-policies.md), original auditada y verificación externa estricta. La política acompaña cada edición y sus resultados. Esta ampliación no cambia retrospectivamente la admisión ni las conclusiones de las campañas anteriores.

Actualización de la campaña de referencias: el [contrato del corpus completo](../engineering/comparison-campaign.md)
amplía el recorrido a todos los instrumentos y muestras admisibles, sin el límite
de 128 activos. Las cifras de piloto del párrafo siguiente describen la
planificación inicial. La ampliación conserva las reglas temporales, las cuatro
modalidades y el test cerrado. El candidato posterior deberá compararse con
referencias ajustadas sobre los mismos datos de entrenamiento.

El piloto propone hasta 64 activos estadounidenses y la comparación principal hasta 128, con frecuencia diaria y un único horizonte principal de una sesión. El número final y las fechas se fijarán con la auditoría de cobertura y el tiempo medido, sin seleccionar por rentabilidad futura ni por disponibilidad durante todo el test. La selección se basará en información del periodo inicial y una regla determinista registrada. Conservará altas, bajas, cambios de símbolo y fechas de exclusión cuando existan. Se inventariará toda la copia y se prepara el recorrido por bloques. Ampliar a 256 activos, al universo completo o a China requiere una decisión de presupuesto registrada.

La [revisión de precios sin ajustar y bajas](../data/unadjusted-prices.md) mide el sesgo de supervivencia de la población desde 2000. Casi todos los activos siguen cotizando en marzo de 2025 y no hay una fuente libre y verificable de retornos de salida en EE. UU. Mientras no se incorporen bajas, los resultados predictivos y financieros se declaran condicionados a esa supervivencia y se acompañan del análisis de sensibilidad descrito en ese documento.

El universo de FinMultiTime no equivale a una lista de constituyentes históricos del S&P 500. Si no se reconstruye la pertenencia temporal, las conclusiones se limitarán explícitamente al universo retrospectivo disponible. No se impondrá como requisito que un activo sobreviva hasta el último día. Un manifiesto recogerá la versión, los archivos utilizados, los hashes, la regla de selección y cada motivo de exclusión. Véase la [ficha inicial](../data/finmultitime-card.md).

## Reloj de decisión y objetivo

Convención inicial para evitar ejecutar al cierre ya observado: construir la predicción después del cierre de la sesión t, cuando los precios de esa sesión estén disponibles, y simular entrada en la apertura de la siguiente sesión y salida en su cierre. Las horas reales y los márgenes de disponibilidad se obtendrán del calendario de mercado, con zona `America/New_York`, y se almacenarán en UTC. Una sesión no equivale a 24 horas naturales.

Para el horizonte principal, el retorno bruto es el retorno simple apertura-cierre de la siguiente sesión. El índice de mercado utilizará el mismo intervalo de negociación. La serie de SPY existe en la copia local, pero hay que comprobar sus ajustes antes de usarla como proxy. Una variante cierre-cierre sería un experimento predictivo distinto. No se mezclarán sus resultados ni se trasladarán automáticamente a la simulación ejecutable.

El objetivo residual por activo i es:

$$y_{i,t}=r^{OC}_{i,t+1}-\hat\alpha_{i,t}-\hat\beta_{i,t}r^{OC}_{m,t+1}.$$

Los coeficientes se estimarán exclusivamente con pares de retornos cuyo cierre ya sea conocido en t. Propuesta inicial: ventana móvil de 252 sesiones y mínimo de 126 observaciones. Descartar los casos sin historial suficiente. El residualizador se calcula según la misma regla temporal para todas las muestras y modelos, sin ajustar coeficientes con el periodo completo. El retorno futuro del mercado forma parte de la **etiqueta**, nunca de las entradas de la predicción.

El [ejemplo reproducible del objetivo](target-definition.md) aplica esta fórmula con 252 pares técnicos y el calendario real. Distingue decisión, apertura, cierre anticipado y maduración cuando el factor está disponible después que el activo, sin cargar datos del corpus ni entrenar un predictor.

La extensión sectorial añadiría un factor y su coeficiente con la misma regla. Requiere pertenencia sectorial histórica y una serie de comparación verificable. La etiqueta `Sector` actual del CSV no basta para acreditarlo. Residualizar no demuestra causalidad ni garantiza que la cartera resultante sea neutral al mercado. Se informarán también los retornos brutos y las exposiciones de la simulación.

## Modalidades y disponibilidad

Todas las entradas deben cumplir `available_at <= prediction_at`. El calendario contable, la fecha del texto y el nombre de un archivo son indicios, no pruebas suficientes de disponibilidad. Las reglas completas se especifican en el [contrato de datos](../data/data-contract.md).

La edición estricta exige precios, texto, fundamentales y gráficos, además de los 140 indicadores macro observados. La histórica conserva las cuatro posiciones de modalidad y el contexto macro con máscaras y causas de ausencia, sin eliminar una fila por una modalidad opcional ausente. Los precios y objetivos válidos siguen siendo necesarios. No se cambia la edición ni la población entre modelos. Cuando una noticia solo tiene fecha, se aplica un desplazamiento conservador hasta el cierre de la siguiente sesión posterior a esa fecha. Si no se puede acreditar su disponibilidad, no se convierte en una entrada conocida.

Las tablas usarán cada hecho y su versión de publicación. Ni el fin del trimestre ni un filing exterior justifican retrospectivamente todas sus cifras. Los gráficos se regenerarán con una ventana que termine en t, evitando usar imágenes semestrales para predecir días interiores. Embeddings, normalizadores, selección de variables y diccionarios se registrarán por versión y corte de entrenamiento. Un codificador preentrenado publicado después del periodo evaluado puede introducir conocimiento retrospectivo: debe declararse y no presentarse como una simulación histórica estricta de disponibilidad del modelo.

El [catálogo macroeconómico](../data/macro-catalog.md) amplía los candidatos por familias. Ninguna fila se admite por aparecer en el catálogo. Se requieren observaciones, versiones y disponibilidad histórica verificables. Una revisión conocida hoy es un evento nuevo y no reemplaza retroactivamente las entradas pasadas. La comparación actual mantiene el contexto macro en todos los modelos. Una variante que lo retire queda fuera del alcance vigente.

## Memoria y orden de actualización

La memoria neuronal de Titans, su momentum, los parámetros persistentes del artículo y el banco episódico son estados distintos. Los parámetros compartidos y persistentes quedan congelados al evaluar. La adaptación de pesos rápidos usa un objetivo asociativo y entradas disponibles bajo una política explícita. El banco usa eventos elegibles y error financiero maduro. Los retornos futuros no pueden entrar en el estado antes de estar disponibles. La sorpresa económica es una puntuación operativa, no una estimación de un efecto causal identificado.

La predicción y la escritura episódica siguen este orden. Cualquier actualización asociativa local dentro de Titans-MAC conserva un estado de entrada común para la sesión y declara por separado sus efectos y su granularidad:

1. Construir una instantánea común con datos ya disponibles. Consultar el estado previo.
2. Emitir y conservar las predicciones de **todos** los activos de esa sesión con ese estado.
3. Incorporar a la cola de actualización los errores anteriores que hayan madurado, identificados por `label_available_at`.
4. Calcular la puntuación de escritura con información permitida y actualizar el estado para decisiones posteriores.
5. Guardar los eventos aceptados/rechazados y un identificador del estado resultante.

Esta convención es conservadora: una etiqueta conocida en t puede influir a partir de la siguiente decisión. Para las escrituras simultáneas se fija un orden canónico por `(label_available_at, event_id, asset_id)`, independiente del orden de carga. La permutación de activos debe conservar tanto las predicciones actuales como el estado final y las predicciones posteriores. Una agregación conjunta sería una variante distinta que exigiría especificar su reducción. Las actualizaciones autosupervisadas de entradas observadas, si se incluyen, tendrán una variante y un registro separados de la adaptación supervisada por etiquetas maduras.

Cada ventana de evaluación reinicia la memoria y optimizador interno. El calentamiento solo puede usar historia previa al comienzo de evaluación y etiquetas maduras. No se hereda el estado del final de otra ventana. En la campaña desde 2000 los cuatro controles de Titans-MAC siguen la [política común de memoria](../engineering/titans-chronological-trainer.md#política-de-memoria-en-inferencia): cada tramo parte del estado inicial y observa antes 12 meses de entradas sin etiquetas. La GRU episódica empieza cada tramo con el banco vacío y solo admite etiquetas que maduran dentro de él. Se comparará memoria congelada con adaptación online predeterminada. Los hiperparámetros y reglas de actualización permanecerán fijados durante la evaluación.

Los refinamientos K = 1, 2 y 4 de la [referencia episódica](candidate-architecture.md) son otro eje y no cuentan las actualizaciones de memoria neuronal. Las lecturas episódicas usan la misma instantánea y no consultan etiquetas futuras. La política de K se selecciona en validación. Los errores de escritura proceden de predicciones conservadas, no recalculadas posteriormente. Cambiar el codificador requiere reconstruir o migrar de forma comprobada los episodios. CM-v1 conserva un baseline identificado y un factorial propio, sin cambiar B al cambiar de arquitectura.

## Particiones, ajuste y calibración

Se usará walk-forward expansivo, sujeto a la cobertura real. La configuración inicial plantea al menos tres años de entrenamiento, seis meses de validación, tres de calibración y tres de evaluación por ventana, avanzando tres meses. Son duraciones de diseño, no fechas ya validadas del dataset. El último año elegible se reservará como test final cronológico, sin usarlo para elegir arquitectura, modalidades o costes.

La edición desde 2000 aplica estas duraciones con evaluación anual en el [protocolo walk-forward v2](walk-forward-2000.md), que fija también el primer año evaluado de cada mercado, la purga por intervalo y la regla común de parada.

La validación se usa para hiperparámetros, selección de modelo y umbral de escritura. La calibración se reserva para intervalos y abstención. Cada frontera debe purgar ejemplos cuyo intervalo de etiqueta alcance el siguiente tramo. Ningún ejemplo de entrenamiento puede tener `label_available_at` posterior al corte de ajuste. Un margen conservador de una sesión se evaluará para el horizonte principal. Para horizontes mayores debe derivarse del intervalo real de las etiquetas, no de un número fijo heredado.

El modelo final se ajustará con la historia autorizada por el protocolo, se calibrará en el tramo reservado anterior al test y se evaluará una vez sobre el último año. Si se permite actualización de memoria durante ese año, será parte de la política online predefinida, sin selección ni cambios humanos basados en los resultados. Las variantes de adaptación se compararán con esa misma restricción.

```mermaid
flowchart LR
    T[Pasado<br/>ajuste] --> V[Validación<br/>selección]
    V --> C[Calibración<br/>intervalos y abstención]
    C --> E[Ventana futura<br/>evaluación]
    E --> N[Siguiente corte<br/>nuevo ajuste con pasado]
```

La [búsqueda implementada para la ampliación](../engineering/reference-search.md) fija doce configuraciones por familia principal, semilla `42` para seleccionar y semillas `42`, `43` y `44` para los finalistas. Se registra cada intento, incluidos errores y descartes. Este diseño sustituye el presupuesto inicial de diez configuraciones para las nuevas campañas, sin reinterpretar los experimentos anteriores. La reserva final permanece cerrada durante esa selección.

El [plan de cómputo](../engineering/compute-plan.md) distingue el cribado, la confirmación y el análisis. La disponibilidad 24/7 no sustituye una estimación de tiempo. Las ejecuciones prolongadas cumplirán el [contrato de checkpoints](../engineering/checkpoint-recovery.md), con datos, estados, versiones y posición confirmada recuperables.

## Métricas e interpretación

Para predicción: MAE, MSE, dirección del retorno residual, Rank IC por sesión y su distribución. Los empates y ceros tienen una regla explícita. Sesiones con muy pocos activos válidos o predicciones constantes no generan correlaciones válidas y deben contabilizarse, no convertirse en cero silenciosamente. La agregación principal será por sesión para evitar que días con más activos dominen el resultado. En la edición desde 2000 se informan además, como análisis secundario declarado el 9 de octubre de 2026 antes de cualquier resultado, las mismas métricas por [estratos de presencia de noticias y fundamentales](metrics.md#estratos-por-presencia-de-modalidades). Son descriptivas. Una diferencia entre estratos no mide el efecto de una modalidad, que exigiría enmascararla en las mismas filas.

Para incertidumbre: cobertura empírica y anchura de intervalos al 80 % y 95 %, pérdida de cuantiles, [puntuación de intervalo](metrics.md#puntuación-de-intervalo), [Brier y ECE de la probabilidad implícita de subida](metrics.md#probabilidad-implícita-de-subida-brier-y-ece) y curvas riesgo-cobertura. NLL se calculará solo cuando el modelo defina una densidad predictiva evaluable, ya que una lista de cuantiles no la determina. La calibración se evalúa también por régimen, activo y periodo cuando el tamaño lo permite. El núcleo utiliza un calibrador congelado antes de evaluar. En la edición desde 2000 es la [calibración común CQR](metrics.md#calibración-común-de-intervalos), ajustada por mercado en el tramo de calibración de cada ventana. Está implementada y comprobada con valores calculados a mano, sin aplicarse todavía a predicciones reales. ACI, si se incorpora, será una política online separada, con hiperparámetros fijados y actualizaciones por etiquetas maduras. No permite seleccionar umbrales o variantes después de ver el test. La [bibliografía conformal](../references/finance-review.md) no autoriza asumir intercambiabilidad en mercados cambiantes, y la cobertura bajo dependencia es una cuestión empírica del estudio.

La simulación económica ordenará activos por señal al cierre de t y abrirá posiciones en t+1. Como regla inicial, usar cuartiles extremos con exposición larga total +0,5 y corta -0,5: exposición neta cero y bruta uno, aunque beta no necesariamente cero. Los pesos se fijan al abrir y se liquida al cerrar. Deben incluirse costes de ambos lados y todas las operaciones de apertura y cierre. No basta con restar el cambio entre pesos objetivo de días sucesivos si cada día se parte de efectivo. Para una variante de cartera mantenida entre días, se usarían pesos desplazados por rendimientos y otro registro de operaciones.

Los retornos netos se calculan con retornos reales de activos, no sumando residuos como si fueran beneficios. Se estudiarán costes ilustrativos de 0, 5, 10 y 20 puntos básicos **por lado**, separados de préstamo de valores, impacto y deslizamiento. Los valores son escenarios de sensibilidad, no estimaciones observadas de ejecución. Si faltan disponibilidad de préstamo o precios ejecutables, se declara la limitación de la posición corta. La abstención deja capital en efectivo. Se informa su porcentaje y la exposición real, sin renormalizar silenciosamente el riesgo.

Se publicarán retorno acumulado neto, drawdown máximo, rotación, exposición, número de operaciones y Sharpe diario con convención de anualización y análisis de dependencia. La [cartera por cuartiles](long-short-portfolio.md) concreta esta regla en la versión 4 de la comparación, con precios negociados reconstruidos y las reglas de las acciones A, y declara sus simplificaciones. La incertidumbre se estimará remuestreando bloques de sesiones que conserven juntos los activos, sobre diferencias pareadas entre modelos. Longitud de bloque y sensibilidad se predefinen con el periodo de desarrollo. Las semillas no se tratarán como nuevos mercados independientes. La búsqueda múltiple se reflejará en los contrastes y en el registro de ensayos. PBO y DSR serán análisis complementarios cuando sus supuestos y datos de entrada estén justificados.

La política financiera por refuerzo es otro experimento, distinto de la ordenación larga y corta descrita anteriormente. Su alcance previsto es efectivo o posiciones largas, sin apalancamiento, con ejecución en aperturas posteriores y cuentas separadas por moneda. No reutiliza retornos residuales como beneficios. Las restricciones de cada instrumento y fecha, incluidas las de China, deben estar acreditadas antes de simular sus operaciones. En la campaña desde 2000, la [etapa de políticas](training-campaign-2000.md#etapa-de-políticas-por-ventana) tiene dos niveles. KLPO terminal y tres referencias sin aprendizaje se aplican a todos los predictores con productor en la campaña, y las variantes PPO y Double DQN solo al Transformer compacto y a Titans-MAC `mac_online`. La etapa ajusta cada política con los tramos de evaluación de ventanas anteriores, la selecciona con un criterio de cartera sobre la validación y nunca con el MAE del predictor, y usa cintas de [precios negociados reconstruidos](../data/unadjusted-prices.md). Las [reglas de las acciones A](../engineering/china-market-rules.md) se aplican en los motores Python y nativo con paridad paso a paso, pero `mars-titan-ppo` y `mars-titan-sim` solo admiten todavía cintas sintéticas y KLPO no tiene aún una orden financiera ejecutable.

El [entorno predictivo causal](../engineering/causal-prediction-environment.md) separa observaciones, acciones y etiquetas maduras. Su implementación no acredita que estén terminados la política financiera, el ajuste conjunto o sus campañas. Los tres experimentos deben conservar padres, población y presupuestos comparables, y mantener el error predictivo como criterio principal.

### Qué pregunta responde cada comparación

La [matriz de comparaciones](metrics.md#matriz-de-comparaciones-y-atribución-por-componentes) se declaró el 10 de octubre de 2026, antes de cualquier resultado, sobre las mismas filas, ventanas y remuestreo que la comparación walk-forward. Cada fila de la tabla es una o varias familias de contrastes con su propia corrección múltiple. Todas se miden con el MAE residual por sesión como métrica principal y además con pinball, intervalos, Brier y ECE del signo, dirección, Rank IC y la cartera. Las políticas y los diagnósticos de arquitectura se añadirán como vistas cuando se declaren sus métricas.

| Comparación | Pregunta en lenguaje llano | Estado de los brazos |
| --- | --- | --- |
| Entre familias | ¿Qué familia predice mejor que cada una de las demás con las mismas filas? | En la campaña |
| MARS-TITAN frente a cada familia | ¿Cada variante de MARS-TITAN (M0 a M3, K = 2 y 4, primera lectura y B6) mejora a cada referencia, y cuánto? | En la campaña, salvo la primera lectura y B6, que entran con los componentes de integración |
| CM-v1 frente a cada familia | ¿B, B+C, B+M y B+C+M mejoran a cada referencia? | En la campaña |
| Núcleo frente a la referencia pública | ¿Nuestro Titans-MAC predice como la implementación pública de referencia, y cuánto añaden nuestras ampliaciones sobre ella? | Pendiente de #434 y de su aprobación |
| Cadena por etapas | ¿Posentrenar el padre del año anterior mejora a reentrenar desde cero, a trasladarlo sin cambios o a continuarlo entero? | Pendiente del plan por etapas |
| Control en línea | ¿La ventaja de MARS-TITAN viene solo de seguir aprendiendo con etiquetas maduras? Un Transformer recibe las mismas etiquetas en el mismo instante | Pendiente del plan por etapas |
| Escalera de Titans | ¿Cuánto añade cada pieza, en un orden fijado, del Transformer compacto a MARS-TITAN con M3? | En la campaña |
| Dejar uno fuera | ¿Cuánto se pierde al quitar cada pieza del MARS-TITAN completo, junto con lo que depende de ella? | Cinco de ocho en la campaña. Faltan tres brazos candidatos |
| Efectos condicionados | ¿Cuánto aporta una pieza según lo que el modelo ya tiene? | Los pares que existen. El resto necesita candidatos |
| Interacciones | ¿Dos piezas se suman, se refuerzan o se pisan? Actualización de Titans con banco, K con escritura por error, B6 con banco y C con M | C con M en la campaña. Las otras tres necesitan candidatos |
| Shapley | ¿Cómo se reparte la mejora entre piezas sin depender del orden? | C y M en la campaña. Error frente a anomalía y actualización frente a banco necesitan candidatos. Las ocho piezas de Titans no admiten un reparto porque 236 de sus 256 combinaciones activan una pieza sin aquella de la que depende |

Los brazos candidatos, su coste estimado y su prioridad están en el [informe de brazos que faltan](../../reports/engineering/component-attribution-20261010/README.md). Ninguno se ha añadido al plan de la campaña. Las familias de la comparación walk-forward (controles de Titans, políticas de escritura, refinamientos, factorial de CM-v1 y referencias frente al control cero) siguen respondiendo a sus propias preguntas.

## Reglas para cerrar el estudio

Una afirmación de mejora exige datos y particiones comparables, una magnitud relevante y variabilidad reportada. Un valor p aislado no basta. Si el modelo no supera una referencia, se analizará si el límite procede de la señal, los datos, la capacidad, la actualización o el presupuesto. La discusión diferenciará asociación, contribución de un componente y causalidad económica.

Antes de abrir el test final se cerrarán el [registro de experimentos](experiment-matrix.md), las exclusiones, las fórmulas, el número de ensayos y la política de memoria. Cada resultado de la memoria tendrá un vínculo a su configuración, commit, manifiesto de datos, predicciones y entorno. El [procedimiento de apertura única](../engineering/final-test-opening.md) fija el orden, las condiciones y el registro, e impide repetir la apertura aunque falle. Está implementado sin el paso que lee 2024, así que no acredita ninguna apertura.
