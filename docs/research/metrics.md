# Métricas predictivas y agregación

La evaluación conserva el error por fila y el error por sesión. El primero
da más peso a las fechas con más observaciones. El segundo calcula el error
dentro de cada sesión y promedia después esas medias.

Para una sesión observada $s$, con $n_s$ muestras:

$$
\operatorname{MAE}_s = \frac{1}{n_s}\sum_{i\in s}|\hat y_i-y_i|,
\qquad
\operatorname{MSE}_s = \frac{1}{n_s}\sum_{i\in s}(\hat y_i-y_i)^2.
$$

La métrica primaria prevista para seleccionar las referencias es:

$$
\operatorname{MAE}_{\mathrm{sesiones}}
=\frac{1}{|\mathcal S|}\sum_{s\in\mathcal S}\operatorname{MAE}_s.
$$

Cada sesión se identifica por mercado e instante de decisión. Dos mercados
no se fusionan por coincidir en una fecha. En una comparación conjunta, cada
par observado tiene el mismo peso. También se conservan las medias de cada
mercado. La definición no exige que ambos calendarios tengan el mismo número
de sesiones.

Por ejemplo, los errores absolutos 1 y 3 en una sesión, y 6 en otra, producen
MAE por fila de 10/3 y MAE por sesión de 4. Una prueba integrada con cinco
observaciones en tres sesiones obtiene 0,072 y 0,07 respectivamente. Son
ejemplos de comprobación, no resultados bursátiles.

## Implementación y límites

`SessionErrors` conserva tres acumuladores por sesión. No guarda todos los
errores individuales. Admite lotes de hasta 4.096 observaciones y un presupuesto
predeterminado de 50.000 sesiones. Un lote inválido no modifica el estado ya
confirmado. Se rechazan valores no finitos, desbordamientos, fechas ausentes,
longitudes incompatibles y pérdida de precisión temporal al convertir a
microsegundos. Una colección vacía devuelve métricas indefinidas y su motivo,
nunca un error cero artificial.

Durante la evaluación de cada modelo se crea un acumulador nuevo. No se mezclan
épocas, semillas ni folds. El lector supervisado comprueba las claves y los
recuentos de muestras antes de entregar los lotes. Los resultados informan
`samples`, `session_count`, `session_mae`, `session_mse` y `by_market_session`.
El MAE conserva las unidades del retorno residual y el MSE usa su cuadrado.

La curva de entrenamiento sigue mostrando el error por fila de las predicciones
obtenidas durante las actualizaciones. La evaluación final de entrenamiento y
validación incluye ambas agregaciones. Cambiar la partición física de los
archivos no cambia su definición, salvo el redondeo propio de float64.

Los diagnósticos existentes de `models/baselines/diagnostics.py` calculan además
sesgo, RMSE, cuantiles de error, dirección y correlaciones para sus entradas.
No sustituyen la media por sesión definida aquí. El bootstrap anterior pondera
filas y no debe describirse como un intervalo de esta métrica primaria sin la
adaptación correspondiente.

La pérdida pinball y los intervalos predictivos quedan fuera de este
acumulador y se calculan con el contrato de comparación descrito más abajo. NLL
y ECE no están implementadas. NLL requiere una densidad especificada y ECE
probabilidades de un evento definido. No se deducen esas salidas de una
predicción puntual ni de una lista de cuantiles.

La [verificación de esta agregación](../../reports/resources/session-evaluation-quality.json)
registra ejemplos conocidos, cobertura y mutaciones dirigidas. No presenta esos
ejemplos como precisión obtenida sobre datos de mercado.

## Contrato de comparación por sesión

`evaluation/forecast_panel.py` define `ForecastPanel`, la entrada común de todas
las métricas de comparación. Cada fila contiene identidad (`sample_id`, texto o
entero), mercado declarado, instante de decisión en microsegundos UTC, objetivo
residual, predicción puntual y, si el modelo los emite, cuantiles con sus
niveles. `ForecastPanel.from_arrow` lee el mismo contrato de los Parquet de
predicciones. Un panel corresponde a un modelo, un fold y una semilla.

La construcción valida y no corrige. Fallan los NaN, los infinitos, los enteros
que pierden precisión en float64, las fechas ausentes o con precisión inferior
al microsegundo, los mercados no declarados, las identidades vacías o
duplicadas, los niveles fuera de (0, 1) o no crecientes y los cuantiles
cruzados. Un cruce no se reordena, porque ocultaría un fallo del modelo. El
mensaje indica cuántas filas cruzan.

Las filas se guardan en orden canónico: mercado, instante e identidad. Las
métricas suman siempre en ese orden, de modo que permutar las filas de entrada
o los activos de una sesión produce resultados idénticos bit a bit. La sesión es
el par (mercado, instante). El día de remuestreo es el día natural UTC del
instante. Las sesiones de Estados Unidos y China del mismo día comparten ese día
aunque sus cierres difieran.

`cohort_sha256` resume identidades, mercados, instantes y objetivos, sin las
predicciones. Dos modelos solo se comparan si su huella coincide, es decir, si
evalúan exactamente las mismas filas con los mismos objetivos.

La agregación entre sesiones es explícita en cada resumen
(`market_weighting`). Con `session`, cada par mercado-instante definido pesa lo
mismo. Es la regla primaria ya declarada en este documento. Con `market`, se
promedia dentro de cada mercado y después se da el mismo peso a cada mercado
con al menos una sesión definida. Si se usa, debe declararse antes de evaluar.

Para una métrica por sesión $m_s$ definida en el conjunto $\mathcal D$:

$$
\bar m_{\mathrm{sesión}}=\frac{1}{|\mathcal D|}\sum_{s\in\mathcal D} m_s,
\qquad
\bar m_{\mathrm{mercado}}=\frac{1}{|\mathcal M|}\sum_{k\in\mathcal M}
\frac{1}{|\mathcal D_k|}\sum_{s\in\mathcal D_k} m_s .
$$

Una sesión no definida se cuenta con su motivo y queda fuera de ambos
denominadores. Nunca se sustituye por cero.

## Error puntual: MAE, MSE y RMSE

`score_sessions` calcula $\operatorname{MAE}_s$ y $\operatorname{MSE}_s$ con las
fórmulas del principio. El RMSE resumido es
$\sqrt{\bar{\operatorname{MSE}}}$, la raíz del MSE agregado con la misma
ponderación. No es la media de las raíces por sesión, que sería menor o igual
por la desigualdad de Jensen. El resumen añade el MAE y el MSE por fila como
referencia descriptiva.

El MAE conserva las unidades del retorno residual y se interpreta como el error
típico por activo en una sesión típica. El MSE penaliza más los errores grandes.
Una comparación con la referencia cero indica si el modelo aporta algo sobre no
predecir. Ninguna de estas cifras es un porcentaje de acierto ni una
rentabilidad. Un MAE menor tampoco implica mejor ordenación entre activos.

## Dirección del retorno residual

La convención es `zero_target_excluded_zero_prediction_is_abstention_and_miss`,
la misma de `forecast_reliability.py`. En la sesión $s$:

$$
E_s=\{i: y_i\neq 0\},\quad C_s=\{i\in E_s: \hat y_i\neq 0\},\quad
H_s=\{i\in C_s: \operatorname{sign}\hat y_i=\operatorname{sign}y_i\},
$$

$$
\operatorname{DA}_s=\frac{|H_s|}{|E_s|},\qquad
\operatorname{CDA}_s=\frac{|H_s|}{|C_s|}.
$$

Los objetivos nulos no se juzgan. Una predicción nula es una abstención que
cuenta como fallo en $\operatorname{DA}_s$ y no entra en $\operatorname{CDA}_s$.
Cada cociente solo está definido con denominador positivo. Se informan además
las filas elegibles, las llamadas, los aciertos, los objetivos y predicciones
nulos y la cobertura de llamadas.

Es la métrica más cercana a «cuántas veces acierta». Un 0,5 no es
necesariamente el nivel del azar, porque la proporción de residuos positivos de
una sesión puede alejarse de la mitad. Por eso la comparación se hace frente a
controles con el mismo denominador. El control cero se abstiene siempre y
obtiene $\operatorname{DA}=0$. Acertar el signo de un residuo no equivale a
acertar la subida o bajada bruta del activo ni dice nada sobre el tamaño del
error.

## Correlación de rangos por sesión

El Rank IC de una sesión es la correlación de Pearson entre los rangos medios
de objetivo y predicción dentro de esa sesión, equivalente a
`scipy.stats.spearmanr` con empates promediados:

$$
\rho_s=\frac{\sum_{i\in s}(r_i-\bar r)(\hat r_i-\bar r)}
{\sqrt{\sum_{i\in s}(r_i-\bar r)^2\sum_{i\in s}(\hat r_i-\bar r)^2}},
\qquad \bar r=\frac{n_s+1}{2}.
$$

Los rangos se calculan para todas las sesiones a la vez con dos ordenaciones
estables y sumas por grupo, sin bucles por fila ni por sesión. Una sesión queda
sin valor por `insufficient_assets` si tiene menos activos que el mínimo
declarado (tres por defecto), por `constant_target` o por
`constant_prediction`, en ese orden de prioridad. El resumen da la media sobre
las sesiones definidas, su desviación típica entre sesiones, la fracción de
sesiones positivas y el recuento de cada motivo. La incertidumbre del Rank IC
medio se obtiene con `compare_series` y el contraste `level(modelo)`.

Mide si el modelo ordena bien los activos de una misma decisión, que es lo que
usaría una cartera larga y corta. No mide la escala del error ni la calibración.
Un valor medio positivo con muchas sesiones no definidas describe solo las
sesiones definidas. Las sesiones con pocos activos dan correlaciones muy
ruidosas, por lo que conviene declarar como sensibilidad un mínimo mayor.

## Cuantiles: pérdida pinball y frecuencia por nivel

Para el nivel $\tau$ y el cuantil emitido $q_{i,\tau}$:

$$
\rho_\tau(u)=\max\{\tau u,(\tau-1)u\},\quad u=y_i-q_{i,\tau},\qquad
\operatorname{PL}_{s,\tau}=\frac{1}{n_s}\sum_{i\in s}\rho_\tau(u),
$$

$$
F_{s,\tau}=\frac{1}{n_s}\sum_{i\in s}\mathbf 1\{y_i\le q_{i,\tau}\}.
$$

Se agregan por sesión con la ponderación declarada. `mean_pinball` promedia
los niveles con el mismo peso. Con $\tau=0{,}5$, $\rho_{0,5}(u)=|u|/2$ y la
pérdida es la mitad del MAE de la mediana. La pérdida pinball es una regla de
puntuación propia para el cuantil de nivel $\tau$
([Gneiting y Raftery, 2007](https://doi.org/10.1198/016214506000001437)). Menor es
mejor y solo compara modelos con los mismos niveles.

$F_{\tau}-\tau$ mide la calibración marginal de cada nivel. Con cinco niveles es
el equivalente discreto de un histograma PIT. Una diferencia pequeña no
garantiza calibración condicional por activo, régimen o periodo, y la
dependencia temporal impide asumir garantías de cobertura. Con cinco niveles,
`mean_pinball` no es el CRPS. Tampoco permite calcular NLL, que necesitaría una
densidad declarada.

## Intervalos centrales y afirmaciones de signo

Los pares simétricos $(\tau, 1-\tau)$ forman intervalos centrales de nivel
nominal $1-2\tau$. Con 0,025, 0,1, 0,5, 0,9 y 0,975 son el 80 % y el 95 %.
Por sesión se calculan la cobertura con extremos incluidos y la anchura media:

$$
\operatorname{Cov}_s=\frac{1}{n_s}\sum_{i\in s}\mathbf 1\{q_{i,\tau}\le y_i\le
q_{i,1-\tau}\},\qquad
W_s=\frac{1}{n_s}\sum_{i\in s}(q_{i,1-\tau}-q_{i,\tau}).
$$

Un intervalo que no contiene el cero afirma un signo. `sign_commitment` es la
fracción media de filas con esa afirmación y `committed_sign_error` la fracción
media de afirmaciones con objetivo no nulo cuyo signo falla, sobre las sesiones
con alguna afirmación juzgable. Es la medida más directa de «no alucinar» en
este problema numérico: cuántas veces el modelo se muestra seguro del signo y
se equivoca.

La cobertura debe leerse junto con la anchura. Un intervalo enorme cubre casi
siempre y no informa. Una anchura menor solo es mejor a igual cobertura. Un
error de signo comprometido bajo con muy pocas afirmaciones tiene poco valor, por
eso se informan también las filas afirmadas y fallidas.

## Riesgo-cobertura selectiva

`selective_risk` usa como incertidumbre la anchura de un intervalo central que
el modelo ya emite. No crea otra cabeza ni otra puntuación. Para cada cobertura
$c$, en cada sesión se conservan las $k_s=\lceil c\,n_s\rceil$ filas de menor
anchura, con desempate por el orden canónico. $c$ se convierte a fracción
exacta para que, por ejemplo, $0{,}7\cdot 10$ conserve siete filas. Se calcula
el MAE de las filas conservadas y se agrega por sesión. Todas las sesiones
conservan al menos una fila, así que sus pesos no cambian.

La selección dentro de la sesión usa solo información disponible en la decisión.
El oráculo ordena por el error real y es una cota inferior descriptiva, no un
método. Una curva útil baja al reducir la cobertura y se acerca al oráculo. Si
queda por encima del MAE completo, la anchura no ayuda a abstenerse. Las
comparaciones entre modelos deben hacerse a coberturas iguales, con
`SelectiveRisk.series(c)`. Esta curva no es una regla de negociación, porque un
umbral operativo debe fijarse con calibración y no con la propia evaluación.

Un modelo sin cuantiles no tiene curva y el resumen lo indica. Un radio
constante, como el del calibrador simétrico de `forecast_reliability.py`, da
la misma anchura a todas las filas y no permite ordenar la abstención.

## Comparaciones emparejadas

`evaluation/paired_comparisons.py` recibe una `SessionSeries` por modelo,
obtenida con `SessionScores.series(métrica)` o `SelectiveRisk.series(c)`, y una
familia de contrastes declarada como coeficientes por modelo. Todas las series
deben compartir métrica, mercados y huella de población. Las semillas de un
mismo modelo se promedian sesión a sesión con `SessionSeries.average`. No se
apilan como si fueran sesiones nuevas.

### Delta, mejora relativa e interacción

$$
\Delta=\operatorname{MAE}_{\mathrm{variante}}-\operatorname{MAE}_{\mathrm{base}},
\qquad
\operatorname{Mejora}\,(\%)=100\cdot
\frac{\operatorname{MAE}_{\mathrm{base}}-\operatorname{MAE}_{\mathrm{variante}}}
{\operatorname{MAE}_{\mathrm{base}}},
$$

$$
I=\operatorname{MAE}_{CM}-\operatorname{MAE}_{C}-\operatorname{MAE}_{M}
+\operatorname{MAE}_{B}.
$$

Con pérdidas, $\Delta<0$ y una mejora positiva favorecen a la variante. La
mejora relativa es el cociente de los MAE agregados, no la media de cocientes
por sesión. Solo se calcula para contrastes por pares de una pérdida y queda no
definida, con motivo, si la base vale cero en la estimación o en alguna réplica.
$I<0$ indica que C y M juntos reducen el error más que la suma de sus efectos
separados, e $I>0$ que se solapan o interfieren. `delta`, `interaction` y
`level` construyen estos coeficientes.

Cada contraste se evalúa en las sesiones definidas para todos los modelos que
intervienen y se informan las sesiones excluidas. Con MAE no hay exclusiones.
Con Rank IC, una sesión en la que un modelo da una predicción constante sale del
contraste.

### Remuestreo por bloques de días

Los intervalos usan un bootstrap circular por bloques sobre días UTC, como el
[`CircularBlockBootstrap` de arch](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html).
Cada réplica concatena bloques de $L$ días consecutivos con inicio uniforme y
recorrido circular hasta reunir los $P$ días observados. Cada día arrastra
todas sus sesiones y todos sus activos, y los mismos recuentos se aplican a
todos los modelos y contrastes. Por eso una diferencia constante produce un
intervalo de amplitud nula. No hay bootstrap independiente por fila.

La media de cada réplica se recalcula como cociente de sumas por día y mercado,
de modo que respeta la ponderación declarada también dentro de la réplica. Los
intervalos marginales son los percentiles 2,5 y 97,5 de las réplicas con la
confianza predeterminada del 95 %. El resultado registra
método, unidad, días, sesiones, longitud de bloque, réplicas, semilla, generador
PCG64 y versión de NumPy. La misma familia, semilla y entorno reproduce el
resultado bit a bit. Con otro número de hilos BLAS puede cambiar el último
dígito por el orden de las sumas del producto matricial.

La longitud de bloque no tiene valor por defecto y debe fijarse antes de ver los
resultados. Como propuesta, $L\approx P^{1/3}$ (unos 14 días con 2.500 días de
evaluación) con sensibilidad declarada en `sensitivity_block_lengths`, por
ejemplo 5, 10 y 40. Una evaluación de tres meses tiene unos 63 días, así que
ahí corresponderían bloques de 4 o 5. Si $L\ge P$ no se da intervalo y se
informa el motivo. Los bloques recorren los días observados en orden y no
distinguen un hueco entre ventanas de una continuidad real. Ventanas disjuntas
deben compararse por separado o declararse como una sola serie.

### Familias de afirmaciones

Cuando una llamada contiene varios contrastes que se afirmarán a la vez, se
calcula un intervalo simultáneo por máximo estudentizado del bootstrap
([Montiel Olea y Plagborg-Møller, 2019](https://doi.org/10.1002/jae.2656)):

$$
c_{1-\alpha}=Q_{1-\alpha}\Big(\max_j\frac{|\hat\theta^{*}_j-\hat\theta_j|}{\widehat{\mathrm{se}}_j}\Big),
\qquad \hat\theta_j\pm c_{1-\alpha}\,\widehat{\mathrm{se}}_j .
$$

Aprovecha la correlación entre contrastes porque todos usan las mismas réplicas.
Es menos conservador que Bonferroni y mantiene el nivel conjunto aproximado de
la familia. Las bases auxiliares de la mejora relativa no forman parte de la
familia. La regla propuesta para afirmar un efecto es que su intervalo
simultáneo excluya el cero (`simultaneous_excludes_zero`). Un intervalo que
incluye el cero indica que el resultado no es concluyente, no que los modelos
sean equivalentes. La familia, por ejemplo C, M, CM e I frente a B, debe
declararse antes de evaluar. Esta corrección no cubre la búsqueda previa de
configuraciones, que debe constar en el registro de ensayos.

## Coste medido

El [informe de rendimiento](../../reports/resources/forecast-metrics-benchmark.json)
mide un panel sintético de 2.000.000 filas, 5.000 sesiones, 2.500 días y cinco
cuantiles con `benchmarks/forecast_metrics.py`, NumPy 2.5.3, dos hilos y la CPU
compartida con otras cargas (carga media de 14 a 16). Las medianas de cinco
repeticiones fueron 2,71 s para validar y ordenar el panel, 1,51 s para todas
las métricas por sesión, 1,14 s para la curva riesgo-cobertura de diez niveles y
0,28 s para cuatro contrastes con 2.000 réplicas y cuatro longitudes de bloque.
Los picos de memoria trazada fueron 340, 160 y 126 MiB en esas tres primeras
etapas. Las cifras dependen de la carga concurrente y no son un límite del
equipo.

Ordenar dentro de cada sesión con dos ordenaciones estables, la segunda sobre
`uint16` para que NumPy use radix sort, tardó 0,35 s frente a 0,72 s de
`np.lexsort` con el mismo resultado. Con más de 65.536 sesiones la clave
conserva int64 y la segunda ordenación deja de usar radix sort, con el mismo
orden. La mayor parte del coste restante
del panel es ordenar identidades de texto con Arrow, que no se ha sustituido
porque las alternativas medidas (recuento de distintos más ordenación de tres
claves) no mejoraron.

## Qué no demuestran estas métricas

Ninguna de estas cifras procede todavía de datos de mercado. El bloqueo de
aprendizaje sigue vigente y las pruebas usan valores calculados a mano y datos
sintéticos. Un intervalo bootstrap describe la variabilidad temporal de la
evaluación con la dependencia que capturan los bloques, no la incertidumbre de
la selección de configuraciones ni la de otro periodo. La dirección, el Rank IC y
la cobertura son diagnósticos secundarios. La métrica primaria sigue siendo el
MAE residual por sesión. La [decisión propuesta sobre la cabeza de cuantiles](quantile-head-decision.md)
determina qué modelos tendrán pinball, cobertura y curva riesgo-cobertura.
