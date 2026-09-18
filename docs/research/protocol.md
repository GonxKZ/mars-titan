# Protocolo de investigación

Autor: Gonzalo García Lama. Versión de trabajo ampliada: 18 de septiembre de 2026. La [ampliación de investigación](research-expansion.md) concreta memoria, recurrencia, datos macro y presupuesto.

## Pregunta y alcance

El estudio compara soluciones para predecir retornos residuales de un subconjunto del mercado estadounidense disponible en FinMultiTime. Se quiere determinar si una memoria adaptativa de eventos aporta información útil frente a modelos sin esa memoria, bajo las mismas condiciones de datos y evaluación. El trabajo no depende de demostrar superioridad: una comparación que identifique un efecto nulo o un coste desproporcionado también responde a la pregunta.

La contribución prevista tiene tres partes: un protocolo temporal verificable, una adaptación compacta de memoria neural y una comparación con ablaciones. La novedad de esa combinación debe justificarse frente al [estado del arte](../references/neural-review.md). El nombre MARS-TITAN no acredita por sí mismo una arquitectura novedosa. La propuesta toma ideas de Titans, pero no se presentará como una reproducción completa de sus resultados.

Quedan fuera del núcleo: operaciones reales, conexión a un bróker, recomendaciones de inversión, generación de estrategias por refuerzo, autoedición de modelos al estilo SEAL y entrenamiento de grandes codificadores multimodales. China/HS300, varias escalas de memoria y kernels propios son extensiones condicionadas a tiempo y evidencia.

## Hipótesis registrables

| Hipótesis | Contraste previsto | Qué permitiría sostenerla |
| --- | --- | --- |
| H1. La memoria aporta señal adicional | Modelo completo frente al mismo codificador sin memoria. | Diferencia de MAE fuera de muestra favorable, intervalo por bloques temporales y consistencia entre ventanas. Informar también tamaño y coste del efecto. |
| H2. La escritura selectiva importa | Sorpresa propuesta frente a escritura uniforme y sorpresa puramente predictiva. | Mejora que no se explique solo por más parámetros, más actualizaciones o más cómputo. |
| H3. Hay complementariedad entre modalidades | Precios frente a precios+texto y modalidades adicionales auditadas. | Contribución sobre el mismo conjunto evaluable y análisis separado del efecto de disponibilidad de datos. |
| H4. La incertidumbre ayuda a abstenerse | Riesgo-cobertura, calibración y simulación con/sin abstención. | Menor error o pérdida a coberturas comparables. Informar operaciones descartadas, costes y periodos sin señal. |

Métrica primaria propuesta: **MAE del retorno residual**. Rank IC medio por sesión será una métrica secundaria prioritaria. Los resultados económicos son secundarios y no sustituyen la pregunta predictiva. La hipótesis principal, la métrica y la familia de comparaciones se congelarán antes de ejecutar el test final. No se fija una mejora porcentual esperada sin piloto que justifique su relevancia práctica.

## Universo y selección del subconjunto

El piloto propone hasta 64 activos estadounidenses y la comparación principal hasta 128, con frecuencia diaria y un único horizonte principal de una sesión. El número final y las fechas se fijarán con la auditoría de cobertura y el tiempo medido, sin seleccionar por rentabilidad futura ni por disponibilidad durante todo el test. La selección se basará en información del periodo inicial y una regla determinista registrada. Conservará altas, bajas, cambios de símbolo y fechas de exclusión cuando existan. Se inventaría toda la copia y se prepara el recorrido por bloques. Ampliar a 256 activos, al universo completo o a China requiere una decisión de presupuesto registrada.

El universo de FinMultiTime no equivale a una lista de constituyentes históricos del S&P 500. Si no se reconstruye la pertenencia temporal, las conclusiones se limitarán explícitamente al universo retrospectivo disponible. No se impondrá como requisito que un activo sobreviva hasta el último día. Un manifiesto recogerá la versión, los archivos utilizados, los hashes, la regla de selección y cada motivo de exclusión. Véase la [ficha inicial](../data/finmultitime-card.md).

## Reloj de decisión y objetivo

Convención inicial para evitar ejecutar al cierre ya observado: construir la predicción después del cierre de la sesión t, cuando los precios de esa sesión estén disponibles, y simular entrada en la apertura de la siguiente sesión y salida en su cierre. Las horas reales y los márgenes de disponibilidad se obtendrán del calendario de mercado, con zona `America/New_York`, y se almacenarán en UTC. Una sesión no equivale a 24 horas naturales.

Para el horizonte principal, el retorno bruto es el retorno simple apertura-cierre de la siguiente sesión. El índice de mercado utilizará el mismo intervalo de negociación. La serie de SPY existe en la copia local, pero hay que comprobar sus ajustes antes de usarla como proxy. Una variante cierre-cierre sería un experimento predictivo distinto. No se mezclarán sus resultados ni se trasladarán automáticamente a la simulación ejecutable.

El objetivo residual por activo i es:

$$y_{i,t}=r^{OC}_{i,t+1}-\hat\alpha_{i,t}-\hat\beta_{i,t}r^{OC}_{m,t+1}.$$

Los coeficientes se estimarán exclusivamente con pares de retornos cuyo cierre ya sea conocido en t. Propuesta inicial: ventana móvil de 252 sesiones y mínimo de 126 observaciones. Descartar los casos sin historial suficiente. El residualizador se calcula según la misma regla temporal para todas las muestras y modelos, sin ajustar coeficientes con el periodo completo. El retorno futuro del mercado forma parte de la **etiqueta**, nunca de las entradas de la predicción.

La extensión sectorial añadiría un factor y su coeficiente con la misma regla. Requiere pertenencia sectorial histórica y una serie de comparación verificable. La etiqueta `Sector` actual del CSV no basta para acreditarlo. Residualizar no demuestra causalidad ni garantiza que la cartera resultante sea neutral al mercado. Se informarán también los retornos brutos y las exposiciones de la simulación.

## Modalidades y disponibilidad

Todas las entradas deben cumplir `available_at <= prediction_at`. El calendario contable, la fecha del texto y el nombre de un archivo son indicios, no pruebas suficientes de disponibilidad. Las reglas completas se especifican en el [contrato de datos](../data/data-contract.md).

Se propone comenzar con precios y texto de disponibilidad defendible. Cuando una noticia solo tiene fecha, se aplica un desplazamiento conservador hasta el cierre de la siguiente sesión posterior a esa fecha. Se estudiará sensibilidad a dos sesiones. Si no se puede verificar su fecha o relevancia para el activo, se excluye o se identifica como una aproximación en un análisis separado.

Las tablas usarán cada hecho y su versión de publicación. Ni el fin del trimestre ni un filing exterior justifican retrospectivamente todas sus cifras. Los gráficos se regenerarán con una ventana que termine en t, evitando usar imágenes semestrales para predecir días interiores. Embeddings, normalizadores, selección de variables y diccionarios se registrarán por versión y corte de entrenamiento. Un codificador preentrenado publicado después del periodo evaluado puede introducir conocimiento retrospectivo: debe declararse y no presentarse como una simulación histórica estricta de disponibilidad del modelo.

El [catálogo macroeconómico](../data/macro-catalog.md) amplía los candidatos por familias. Ninguna fila se admite por aparecer en el catálogo. Se requieren observaciones, versiones y disponibilidad histórica verificables. Una revisión conocida hoy es un evento nuevo y no reemplaza retroactivamente las entradas pasadas. El modelo con macro se compara con la misma arquitectura sin macro sobre muestras comunes.

## Memoria y orden de actualización

La memoria es un estado mutable, distinto de los parámetros compartidos del codificador. Los retornos futuros no pueden entrar en el estado en el momento de producir una predicción. La sorpresa económica se tratará como una puntuación operativa, no como una estimación de un efecto causal identificado.

En cada instante de decisión se procesarán los eventos en este orden:

1. Construir una instantánea común con datos ya disponibles. Consultar el estado previo.
2. Emitir y conservar las predicciones de **todos** los activos de esa sesión con ese estado.
3. Incorporar a la cola de actualización los errores anteriores que hayan madurado, identificados por `label_available_at`.
4. Calcular la puntuación de escritura con información permitida y actualizar el estado para decisiones posteriores.
5. Guardar los eventos aceptados/rechazados y un identificador del estado resultante.

Esta convención es conservadora: una etiqueta conocida en t puede influir a partir de la siguiente decisión. Para las escrituras simultáneas se fija un orden canónico por `(label_available_at, event_id, asset_id)`, independiente del orden de carga. La permutación de activos debe conservar tanto las predicciones actuales como el estado final y las predicciones posteriores. Una agregación conjunta sería una variante distinta que exigiría especificar su reducción. Las actualizaciones autosupervisadas de entradas observadas, si se incluyen, tendrán una variante y un registro separados de la adaptación supervisada por etiquetas maduras.

Cada fold reinicia memoria y optimizador interno. El calentamiento solo puede usar historia previa al comienzo de evaluación y etiquetas maduras. No se hereda el estado del final de otro fold. Se comparará memoria congelada con adaptación online predeterminada. Los hiperparámetros y reglas de actualización permanecerán fijados durante la evaluación.

La [arquitectura candidata](candidate-architecture.md) distingue ese estado persistente de hasta cuatro pasos internos de lectura sobre una instantánea fija. Los pasos internos no escriben nuevos recuerdos ni consultan etiquetas futuras. La política de K se calibra como parte del predictor. Los errores de escritura proceden de predicciones conservadas, no recalculadas posteriormente. La consolidación de parámetros se limita inicialmente al entrenamiento y a reajustes programados anteriores al corte. Cambiar el codificador requiere reconstruir o migrar de forma comprobada los episodios.

## Particiones, ajuste y calibración

Se usará walk-forward expansivo, sujeto a la cobertura real. La configuración inicial plantea al menos tres años de entrenamiento, seis meses de validación, tres de calibración y tres de evaluación por ventana, avanzando tres meses. Son duraciones de diseño, no fechas ya validadas del dataset. El último año elegible se reservará como test final cronológico, sin usarlo para elegir arquitectura, modalidades o costes.

La validación se usa para hiperparámetros, selección de modelo y umbral de escritura. La calibración se reserva para intervalos y abstención. Cada frontera debe purgar ejemplos cuyo intervalo de etiqueta alcance el siguiente tramo. Ningún ejemplo de entrenamiento puede tener `label_available_at` posterior al corte de ajuste. Un margen conservador de una sesión se evaluará para el horizonte principal. Para horizontes mayores debe derivarse del intervalo real de las etiquetas, no de un número fijo heredado.

El modelo final se ajustará con la historia autorizada por el protocolo, se calibrará en el tramo reservado anterior al test y se evaluará una vez sobre el último año. Si se permite actualización de memoria durante ese año, será parte de la política online predefinida, sin selección ni cambios humanos basados en los resultados. Las variantes de adaptación se compararán con esa misma restricción.

```mermaid
flowchart LR
    T[Pasado<br/>ajuste] --> V[Validación<br/>selección]
    V --> C[Calibración<br/>intervalos y abstención]
    C --> E[Ventana futura<br/>evaluación]
    E --> N[Siguiente corte<br/>nuevo ajuste con pasado]
```

La búsqueda se limita inicialmente a diez configuraciones por familia principal y tres semillas de evaluación (`17`, `42`, `123`). Se registrarán todos los intentos, incluidos errores, descartes y cambios manuales. El límite se revisará con el piloto antes del test final. Cualquier reducción se aplicará de forma explicable a todas las familias.

Diez configuraciones es un techo y no una obligación. El [plan de cómputo](../engineering/compute-plan.md) comienza con un cribado menor y reserva recursos para confirmación y análisis. La disponibilidad 24/7 no sustituye una estimación de tiempo. Las ejecuciones prolongadas cumplirán el [contrato de checkpoints](../engineering/checkpoint-recovery.md), con datos, estados, versiones y posición confirmada recuperables.

## Métricas e interpretación

Para predicción: MAE, MSE, dirección del retorno residual, Rank IC por sesión y su distribución. Los empates y ceros tienen una regla explícita. Sesiones con muy pocos activos válidos o predicciones constantes no generan correlaciones válidas y deben contabilizarse, no convertirse en cero silenciosamente. La agregación principal será por sesión para evitar que días con más activos dominen el resultado.

Para incertidumbre: cobertura empírica y anchura de intervalos al 80 % y 95 %, pérdida de cuantiles y curvas riesgo-cobertura. NLL se calculará solo cuando el modelo defina una densidad predictiva evaluable, ya que una lista de cuantiles no la determina. La calibración se evalúa también por régimen, activo y periodo cuando el tamaño lo permite. El núcleo utiliza un calibrador congelado antes de evaluar. ACI, si se incorpora, será una política online separada, con hiperparámetros fijados y actualizaciones por etiquetas maduras. No permite seleccionar umbrales o variantes después de ver el test. La [bibliografía conformal](../references/finance-review.md) no autoriza asumir intercambiabilidad en mercados cambiantes, y la cobertura bajo dependencia es una cuestión empírica del estudio.

La simulación económica ordenará activos por señal al cierre de t y abrirá posiciones en t+1. Como regla inicial, usar cuartiles extremos con exposición larga total +0,5 y corta -0,5: exposición neta cero y bruta uno, aunque beta no necesariamente cero. Los pesos se fijan al abrir y se liquida al cerrar. Deben incluirse costes de ambos lados y todas las operaciones de apertura y cierre. No basta con restar el cambio entre pesos objetivo de días sucesivos si cada día se parte de efectivo. Para una variante de cartera mantenida entre días, se usarían pesos desplazados por rendimientos y otro registro de operaciones.

Los retornos netos se calculan con retornos reales de activos, no sumando residuos como si fueran beneficios. Se estudiarán costes ilustrativos de 0, 5, 10 y 20 puntos básicos **por lado**, separados de préstamo de valores, impacto y deslizamiento. Los valores son escenarios de sensibilidad, no estimaciones observadas de ejecución. Si faltan disponibilidad de préstamo o precios ejecutables, se declara la limitación de la posición corta. La abstención deja capital en efectivo. Se informa su porcentaje y la exposición real, sin renormalizar silenciosamente el riesgo.

Se publicarán retorno acumulado neto, drawdown máximo, rotación, exposición, número de operaciones y Sharpe diario con convención de anualización y análisis de dependencia. La incertidumbre se estimará remuestreando bloques de sesiones que conserven juntos los activos, sobre diferencias pareadas entre modelos. Longitud de bloque y sensibilidad se predefinen con el periodo de desarrollo. Las semillas no se tratarán como nuevos mercados independientes. La búsqueda múltiple se reflejará en los contrastes y en el registro de ensayos. PBO y DSR serán análisis complementarios cuando sus supuestos y datos de entrada estén justificados.

## Reglas para cerrar el estudio

Una afirmación de mejora exige datos y particiones comparables, una magnitud relevante y variabilidad reportada. Un valor p aislado no basta. Si el modelo no supera una referencia, se analizará si el límite procede de la señal, los datos, la capacidad, la actualización o el presupuesto. La discusión diferenciará asociación, contribución de un componente y causalidad económica.

Antes de abrir el test final se cerrarán el [registro de experimentos](experiment-matrix.md), las exclusiones, las fórmulas, el número de ensayos y la política de memoria. Cada resultado de la memoria tendrá un vínculo a su configuración, commit, manifiesto de datos, predicciones y entorno. La documentación actual deja definido este procedimiento. Aún no acredita su ejecución.
