# Límites de la repetición y de la reducción de señales

Fecha de corte: 4 de octubre de 2026. Estas derivaciones delimitan el diseño de atención, replay y predicción de eventos. Son resultados elementales conocidos y comprobaciones propias de sus consecuencias, sin reivindicar novedad matemática. No constituyen una implementación de MARS-TITAN ni resultados sobre mercados.

## Repetir datos cambia la optimización, no los hechos observados

Si D contiene n observaciones y cada una se copia r veces, la pérdida empírica media permanece igual:

$$
\widehat R_{D^{(r)}}(\theta)=\frac{1}{rn}\sum_{i=1}^{n}\sum_{j=1}^{r}
\ell_i(\theta)=\widehat R_D(\theta).
$$

Más pasos pueden acercar el optimizador a un mínimo de esa pérdida o reducir interferencia con otros ejemplos. No añaden información externa. Bajo el supuesto ilustrativo de n observaciones independientes con varianza común $\sigma^2$, la varianza de su media es $\sigma^2/n$, incluso después de duplicarlas. Tratar las nr filas como independientes produciría incorrectamente $\sigma^2/(nr)$.

Con n grupos independientes de r observaciones, correlación común intragrupo $\rho$ y la misma varianza, se obtiene

$$
\operatorname{Var}(\bar Y)=\frac{\sigma^2}{nr}\,[1+(r-1)\rho].
$$

Las copias exactas tienen $\rho=1$. Esta cuenta no estima el tamaño efectivo de una muestra financiera dependiente. Tampoco convierte activos, ventanas solapadas o semillas de entrenamiento en nuevas historias económicas.

Un ejemplo refuta que la mejora por repetición sea monótona. La función verdadera vale cero, pero se observa una etiqueta contaminada $\varepsilon\ne0$. Ajustar repetidamente una pérdida cuadrática desde $\theta_0=0$, con $0<\eta<1$, da

$$
\theta_{k+1}=(1-\eta)\theta_k+\eta\varepsilon,
\qquad
\theta_k=\varepsilon[1-(1-\eta)^k].
$$

El error sobre esa etiqueta disminuye, mientras el MAE frente a la función verdadera aumenta hasta $|\varepsilon|$. No describe una campaña real ni dice que todas las etiquetas sean ruido. Justifica contrastar ajuste, retención y error futuro por separado.

## Seleccionar qué repetir cambia la distribución de aprendizaje

Sea un búfer elegible de N episodios y $a_i\geq0$, con $\sum_i a_i=1$, la distribución objetivo declarada. Puede ser uniforme por fila o equilibrar sesiones. Esas opciones no son equivalentes. Para gradientes $g_i$ evaluados en el mismo estado, el objetivo es $g=\sum_i a_i g_i$.

Si se selecciona un índice con reemplazo según $p_i$, y $p_i>0$ siempre que $a_i>0$, se obtiene la identidad siguiente. Los cocientes se definen solo donde $p_i>0$. Fuera de ese soporte la contribución es cero y no se evalúa una división $0/0$.

$$
\mathbb E_{i\sim p}\left[\frac{a_i}{p_i}g_i\right]=\sum_i a_i g_i.
$$

Es una identidad de muestreo por importancia, relacionada con la corrección de [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952v4). El objetivo a y el soporte deben conocerse. La corrección no recupera episodios que nunca entraron en el búfer, no elimina sesgo de admisión ni corrige por sí sola cambios de distribución. Las prioridades dependientes del estado se tratan como valores fijos durante el sorteo y no se diferencian a través del peso al construir este estimador.

Para otra distribución de probabilidad q, una mezcla con cobertura garantizada,

$$
p_i=(1-\epsilon)q_i+\epsilon a_i,\qquad0<\epsilon\leq1,
$$

cumple $p_i\geq\epsilon a_i$, $a_i/p_i\leq1/\epsilon$ y

$$
\mathbb E_p\left\|\frac{a_i}{p_i}g_i\right\|^2
=\sum_i\frac{a_i^2}{p_i}\|g_i\|^2
\leq\frac1\epsilon\sum_i a_i\|g_i\|^2.
$$

La cota permite controlar la concentración, pero no identifica un valor óptimo de $\epsilon$. Hace falta medir varianza, coste y calidad. Un gradiente insesgado tampoco implica una trayectoria de entrenamiento idéntica. Adam, clipping y estados del optimizador introducen operaciones no lineales.

Recortar el peso a c altera la esperanza. Su sesgo exacto respecto a g es

$$
\mathbb E_p[\min(c,a_i/p_i)g_i]-g
=-\sum_i(a_i-cp_i)_+g_i.
$$

Normalizar pesos con una cantidad aleatoria del mismo minibatch tampoco conserva automáticamente la identidad original. Si se muestrea sin reemplazo, se requieren probabilidades de inclusión o una derivación específica. No basta reutilizar las probabilidades del primer sorteo. Las comparaciones distinguirán priorización sin corregir, priorización corregida y uniforme, porque no responden a la misma pregunta.

## Un gradiente alineado no garantiza mejora tras cualquier paso

Para una pérdida de diagnóstico L con gradiente $\beta$-Lipschitz en el segmento considerado, actualizar con $\theta^+=\theta-\eta g_i$ satisface

$$
L(\theta^+)-L(\theta)
\leq-\eta\nabla L(\theta)^\top g_i
+\frac{\beta\eta^2}{2}\|g_i\|^2.
$$

Un producto escalar positivo indica descenso de primer orden. El término de curvatura puede dominarlo. Para $L(\theta)=\theta^2/2$, $\theta=1$, $g_i=1$ y $\eta=3$, la alineación es positiva pero la pérdida pasa de 0,5 a 2. La cota es exacta en este ejemplo.

El supuesto de suavidad no se aplica automáticamente al MAE ni a cualquier composición con ReLU. Una pérdida sustituta suave puede servir para seleccionar candidatos de replay, pero su mejora no es una garantía para el MAE principal. La sonda que aporta el gradiente debe proceder de historia de entrenamiento autorizada, sin usar el tramo de evaluación para entrenar el selector. Calcular gradientes por candidato, compararlos y conservarlos tiene un coste que debe superar a controles simples para justificarse.

## Menos variables y menos información son cosas distintas

Sean $\mathcal F$ la información de la variante reducida y $\mathcal G$ la de la completa, con $\mathcal F\subseteq\mathcal G$ y $\mathbb E|Y|<\infty$. Al incluir todos los predictores medibles, el riesgo óptimo para MAE cumple

$$
\inf_{g\ \mathcal G\text{-medible}}\mathbb E|Y-g|
\leq
\inf_{f\ \mathcal F\text{-medible}}\mathbb E|Y-f|.
$$

La demostración es inclusión de clases: todo predictor que usa $\mathcal F$ también puede operar con $\mathcal G$ ignorando las entradas adicionales. No implica que una red finita entrenada con más señales vaya a generalizar mejor. La estimación, regularización, capacidad y optimización pueden favorecer una versión reducida.

Si una variable derivada es una función determinista de las entradas conservadas, incluidas su historia, unidades y versiones temporales, no amplía esa información. Sin embargo, puede facilitar el aprendizaje a un modelo limitado. El catálogo del proyecto contiene 70 indicadores `raw` y 70 `derived`. Retener solo el último valor de los 70 primeros no basta para reconstruir tasas interanuales, pendientes u otras transformaciones que requieren historia. Se contarán valores, retardos y bytes, no solo nombres de columnas.

Un contraejemplo muestra por qué no se puede garantizar la misma calidad al quitar información externa. En una nueva observación, sea Z uniforme en {−1,1}, independiente de las entradas X, de los parámetros ya aprendidos y de todo el pasado, con Y = Z. Un predictor que observa Z obtiene MAE cero. Cualquier predictor que solo usa X tiene MAE esperado al menos uno. Más repeticiones, memoria o pasos internos no recuperan ese Z ausente.

La información también puede estar presente indirectamente. Una ablación que retira Z del predictor, pero lo conserva en HMM, router, memoria o calibrador, no prueba que el sistema funcione sin Z. Cada variante necesita una vista de entrada común a todos sus módulos y una declaración separada de información utilizada al entrenar y al inferir.

## Destilación y objetivo de evaluación

Un estudiante puede utilizar menos entradas durante inferencia y haber recibido información adicional mediante su profesor durante entrenamiento. Eso se registra como aprendizaje con información privilegiada, junto con el coste del profesor. No se presenta como entrenamiento sin esas fuentes.

Incluso en un caso ideal, la pérdida de destilación importa. Minimizar error cuadrático frente a un profesor produce su media condicional dadas las entradas del alumno. El MAE requiere una mediana condicional del objetivo. Si Y vale 10 con probabilidad 0,1 y cero con probabilidad 0,9, y el alumno no observa qué caso ocurrirá, la media es 1 y su MAE es 1,8. La mediana cero obtiene MAE 1.

La misma distinción afecta a una mezcla de escenarios: una media ponderada de predicciones por tipo de evento no es en general la mediana de la distribución combinada. Si se modela una distribución conjunta de eventos y retornos, se obtiene primero su marginal predictiva y después el funcional correspondiente a la métrica. Alternativamente, una cabeza de retorno entrenada directamente con el objetivo principal evita imponer esa equivalencia.

## Markov, observaciones duplicadas y confianza

El [filtro HMM propuesto](markov-regimes.md) mantiene $b_t(j)=P(S_t=j\mid x_{1:t})$ mediante

$$
b_t(j)\propto p(x_t\mid S_t=j)\sum_i b_{t-1}(i)P_{ij}.
$$

La actualización utiliza observaciones y parámetros permitidos en ese corte. No demuestra que el mercado sea Markoviano ni que los estados estimados correspondan a categorías económicas verdaderas. La permanencia geométrica de un HMM con transiciones homogéneas es una restricción adicional. Un modelo semi Markov solo se justifica si la duración explícita mejora el contraste.

En un caso estático de dos estados, con prior 1/2 y razón de verosimilitudes 3, una observación produce posterior 3/4. Reutilizar erróneamente cinco copias como si fueran observaciones condicionalmente independientes produce

$$
P(S=1\mid\text{cinco copias contadas como independientes})
=\frac{3^5}{1+3^5}=\frac{243}{244}.
$$

La fuente original solo aporta una observación. Las noticias duplicadas o una misma evidencia repetida por varias rutas no justifican esa concentración. Cinco observaciones realmente independientes bajo el modelo serían otro caso.

El paso de Bayes tampoco es automáticamente no expansivo respecto al prior. Con razón de verosimilitudes L, $f(p)=Lp/(1+(L-1)p)$ tiene derivada $L/[1+(L-1)p]^2$, que puede superar uno. Eso no prueba inestabilidad del algoritmo. Impide atribuir al HMM una garantía general de eliminar ruido o de corregir una emisión mal especificada.

## Calibración no equivale a discriminar errores

Supóngase una población equiprobable con errores binarios e = (0,0,1,1). Predecir probabilidad 1/2 para todos está perfectamente calibrado en esa población, pero no distingue qué casos fallan y tiene Brier 1/4. La predicción ideal (0,0,1,1) también está calibrada y obtiene Brier cero. Es un control ilustrativo con verdad conocida, no una señal disponible al modelo.

Por eso se medirán calibración, discriminación y utilidad de las decisiones por separado. El [estudio de Rahnev](https://pubmed.ncbi.nlm.nih.gov/39814749/) muestra además dependencias entre medidas de metacognición, rendimiento de la tarea y fiabilidad entre sesiones. Para eventos se usarán puntuaciones propias como Brier o log score, con la convención de minimización declarada. [Gneiting y Raftery](https://doi.org/10.1198/016214506000001437), §3, explican su fundamento. No garantizan acierto individual ni validez bajo cualquier cambio de distribución.

## Cómo se decide si una versión reducida conserva calidad

Se define $\Delta=MAE_{reducida}-MAE_{completa}$ sobre sesiones pareadas y un margen $\delta$ fijado antes de evaluar. Una conclusión de no inferioridad requiere que el límite superior del intervalo acordado quede por debajo de $\delta$, además del ahorro de recursos medido. Un resultado no significativo no demuestra equivalencia.

Los intervalos respetarán dependencia temporal y transversal. Los contrastes por familia de eventos y régimen se fijarán antes, con el tratamiento de multiplicidad correspondiente. Los estratos sin evidencia suficiente quedarán inconclusos. Ninguna comprobación finita demuestra calidad máxima ante cualquier acontecimiento futuro.

## Comprobación de los ejemplos

El [programa reproducible](../../reports/research/check_attention_replay_algebra.py) comprueba las identidades y los ejemplos finitos en CPU y FP64. El [registro numérico](../../reports/research/attention-replay-algebra-20261004.json) conserva semillas, tolerancias, versiones y hash del programa. Las demostraciones anteriores dependen de sus supuestos, no de enumerar esos casos. El [informe de verificación](../../reports/research/attention-replay-20261004.md) delimita cobertura, complejidad, mutaciones y pruebas realmente ejecutadas.
