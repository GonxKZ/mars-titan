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
no está implementada porque requiere una densidad especificada, que una lista de
cinco cuantiles no determina. El ECE sí se calcula, pero sobre un evento definido
y con una regla declarada antes de los resultados: la [probabilidad implícita de
subida](#probabilidad-implícita-de-subida-brier-y-ece) que se deduce de los
cuantiles. Una predicción puntual no tiene esa probabilidad y no recibe ECE.

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
En los años sin etiquetas residuales de China (antes de 2006), esa ponderación
solo promedia Estados Unidos, así que la regla para las ventanas tempranas debe
fijarse igual para todos los modelos.

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
error. El resumen lo da también en porcentaje (`direction_accuracy_percent` y
`conditional_direction_accuracy_percent`), que es $100\cdot\operatorname{DA}$.

### Precisión y exhaustividad del signo

Con $P_s=\{i: y_i>0\}$, $N_s=\{i: y_i<0\}$, $U_s=\{i\in E_s: \hat y_i>0\}$ y
$D_s=\{i\in E_s: \hat y_i<0\}$:

$$
\operatorname{Prec}^{\uparrow}_s=\frac{|U_s\cap P_s|}{|U_s|},\qquad
\operatorname{Exh}^{\uparrow}_s=\frac{|U_s\cap P_s|}{|P_s|},\qquad
\operatorname{Prec}^{\downarrow}_s=\frac{|D_s\cap N_s|}{|D_s|},\qquad
\operatorname{Exh}^{\downarrow}_s=\frac{|D_s\cap N_s|}{|N_s|}.
$$

Un objetivo exactamente cero no entra en ningún numerador ni denominador,
aunque el modelo afirme un signo para esa fila. Una predicción exactamente cero
no afirma nada, así que no entra en la precisión y cuenta como fallo en la
exhaustividad de la clase de su objetivo. Cada cociente se define solo con
denominador positivo y se promedia entre sesiones como las demás métricas
(`up_precision`, `up_recall`, `down_precision` y `down_recall`). El resumen
conserva los recuentos por filas de objetivos positivos y negativos, llamadas y
aciertos de cada signo. Por ejemplo, con objetivos 1, 2, −1, −2, 0 y 3 y
predicciones 1, −1, −1, 0, 5 y 2, la precisión al alza es 2/2, la exhaustividad
al alza 2/3, la precisión a la baja 1/2 y la exhaustividad a la baja 1/2. La
predicción 5 sobre el objetivo cero no cuenta y la predicción cero sobre −2 es
un fallo de exhaustividad.

Desde la versión 4 de la comparación, `up_precision` y `down_precision` también
son series contrastables. Responden a la pregunta de cuántas veces acierta un
modelo cuando dice que el residuo sube (o baja). Una sesión sin ninguna llamada
de ese signo no tiene precisión, y el contraste emparejado solo usa las sesiones
definidas en todos los brazos que intervienen, con el número de sesiones excluidas
en el informe. La exhaustividad no se contrasta porque un modelo puede subirla
llamando siempre el mismo signo.

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

Sí tiene una relación exacta con dos medidas habituales. Con los pesos canónicos
$w_0=\tfrac12$ y $w_k=\alpha_k/2$ de
[Bracher et al. (2021)](https://doi.org/10.1371/journal.pcbi.1008618), la puntuación de
intervalo ponderada de los $K$ intervalos centrales y la mediana es

$$
\operatorname{WIS}=\frac{1}{K+\tfrac12}\Bigl(\tfrac12\lvert y-m\rvert
+\sum_{k=1}^{K}\tfrac{\alpha_k}{2}\operatorname{IS}_{\alpha_k}\Bigr)
=\frac{1}{K+\tfrac12}\sum_{j=1}^{2K+1}\rho_{\tau_j}(y-q_{\tau_j})
=2\cdot\overline{\operatorname{PL}},
$$

porque $\tfrac12\lvert y-m\rvert=\rho_{0,5}(y-m)$ y cada intervalo cumple la identidad de
la [puntuación de intervalo](#puntuación-de-intervalo). Con los cinco niveles del
proyecto, $K=2$ y la media es la de `mean_pinball`. El mismo doble de la pinball media es
la aproximación del CRPS por cuantiles de `crps_quantile` en scoringrules
([Berrisch y Ziel, 2023](https://arxiv.org/abs/2102.00968)). Las tres cifras resumen los
mismos cinco cuantiles, así que no añaden información a la comparación y ninguna es el
CRPS de una distribución completa.

La [paridad con scoringrules 0.11.0](../../tests/evaluation/test_scores_scoringrules_parity.py)
compara `pinball_loss`, la pinball por nivel de `score_sessions` y la de la
reimplementación independiente con `quantile_score`, y la puntuación de intervalo con
`interval_score`, también sobre intervalos corregidos por la CQR estática. Cubre
empates entre cuantiles, objetivos sobre un cuantil o un extremo, filas muy lejos del
intervalo y objetivos de hasta 3·10⁷. La diferencia máxima fue 3,5·10⁻¹⁸ en la pinball
de entrenamiento y 0 exacto en las demás, frente a una tolerancia declarada de
10⁻¹² + 10⁻¹⁴ |b|. La relación de arriba se cumple con 8,9·10⁻¹⁶. La
`weighted_interval_score` de esa versión con el motor de NumPy suma $w_0\,m$ en lugar de
$w_0\lvert y-m\rvert$ ([frazane/scoringrules#140](https://github.com/frazane/scoringrules/issues/140)),
así que la prueba compone la WIS con `interval_score` y fija ese defecto para detectar
cuándo se corrige.

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

### Error de calibración y sobreconfianza

`quantiles.calibration` resume la calibración marginal del panel con la media y
el máximo de $|F_\tau-\tau|$ entre niveles, la media de
$|\overline{\operatorname{Cov}}-(1-2\tau)|$ entre intervalos centrales y la lista
`undercovered_intervals` de intervalos cuya cobertura agregada queda por debajo
de la nominal. Esta lista es descriptiva y no tiene en cuenta la incertidumbre.

Para juzgar la sobreconfianza con incertidumbre temporal se usa la serie
`coverage_error@0.8` (o `@0.95`), que vale
$\operatorname{Cov}_s-0{,}8$ en cada sesión. Su media con `level` en
`compare_series` da un intervalo por bloques. Si el intervalo simultáneo queda
entero por debajo de cero, el modelo cubre menos de lo que promete y se
clasifica como `undercovers`. Si queda por encima, los intervalos son
conservadores (`overcovers`). Si contiene el cero, el resultado es
`inconclusive`. Esta es la forma medible de «no alucinar» con intervalos: la
cobertura prometida frente a la observada, junto con el error de signo
comprometido descrito arriba. Una cobertura correcta con intervalos muy anchos
no informa, por eso se lee siempre con la anchura.

### Puntuación de intervalo

La cobertura y la anchura se leen juntas, pero no se ordenan con un único número.
La puntuación de intervalo de [Gneiting y Raftery (2007)](https://doi.org/10.1198/016214506000001437)
las combina. Para el intervalo central $[l,u]=[q_{\tau},q_{1-\tau}]$ con
$\alpha=2\tau$:

$$
\operatorname{IS}_\alpha(l,u;y)=(u-l)+\frac{2}{\alpha}(l-y)^{+}+\frac{2}{\alpha}(y-u)^{+},
\qquad
\operatorname{IS}_s=\frac{1}{n_s}\sum_{i\in s}\operatorname{IS}_\alpha(l_i,u_i;y_i).
$$

Es una pérdida propia para el par de cuantiles: menor es mejor y un intervalo
demasiado estrecho paga cada salida con el factor $2/\alpha$, que vale 10 en el
intervalo del 80 % ($\tau=0{,}1$) y 40 en el del 95 % ($\tau=0{,}025$). Cumple
$\tfrac{\alpha}{2}\operatorname{IS}_\alpha=\rho_\tau(y-l)+\rho_{1-\tau}(y-u)$, que las
pruebas comprueban fila a fila. `interval_score` se informa con cada intervalo
del resumen y `interval_score@0.8` y `interval_score@0.95` son series por sesión
que la comparación contrasta como pérdidas, igual que el MAE.

### Probabilidad implícita de subida, Brier y ECE

Los cinco cuantiles definen una función de distribución por tramos. La regla
`piecewise_linear_cdf_at_zero_flat_beyond_extreme_levels_v1`, declarada el 9 de
octubre de 2026 antes de cualquier resultado, interpola linealmente entre los
puntos $(q_j,\tau_j)$ y la deja plana por debajo de $q_{0{,}025}$ y por encima de
$q_{0{,}975}$. Con un empate entre cuantiles se toma el valor continuo por la
derecha. La probabilidad de subida es $p_i=1-\hat F_i(0)$ y queda siempre entre
0,025 y 0,975. La regla no supone una forma de las colas y por eso no afirma más
seguridad que la que dan los niveles extremos.

El evento es $y_i>0$ y solo se juzgan las filas con $y_i\neq 0$, como en la
dirección. Por sesión se informa el Brier $\frac{1}{m_s}\sum_i(p_i-\mathbf 1\{y_i>0\})^2$,
que es una regla propia para probabilidades y la serie `sign_brier` de la
comparación (pérdida). Para la calibración, las filas se reparten en diez
intervalos fijos de probabilidad, $[0;0{,}1)$ hasta $[0{,}9;1]$:

$$
\operatorname{ECE}=\frac{1}{M}\sum_{b=1}^{10}\Bigl|\sum_{i\in b}\mathbf 1\{y_i>0\}-\sum_{i\in b}p_i\Bigr|,
$$

con $M$ las filas juzgables de todas las sesiones. Es la media ponderada por filas
de $|\bar y_b-\bar p_b|$ y cada fila pesa lo mismo. La curva de fiabilidad publica
filas, probabilidad media y frecuencia observada de cada intervalo. El informe
walk-forward da en `sign_reliability` el ECE de cada semilla, el ECE medio de las
semillas, la curva agregada y un intervalo percentil por bloques de días: cada
réplica remuestrea los mismos días para todas las semillas, recalcula el ECE de
cada una y promedia. Se informa en bruto y con el calibrador común de intervalos.

El ECE con intervalos fijos tiene sesgo positivo con pocas filas por intervalo y
depende del número de intervalos. Por eso no entra en las familias de contrastes
y se lee junto al Brier, que sí se contrasta. La probabilidad sale de cuantiles
entrenados con pinball, no de una cabeza de clasificación, así que una mala
calibración del signo puede convivir con buenos cuantiles en el centro.

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
ejemplo 5, 10 y 40. Un año de evaluación tiene unos 252 días y daría bloques
de unos 6. Una evaluación de tres meses tiene unos 63 días y daría bloques de 4
o 5. Si $L\ge P$ no se da intervalo y se informa el motivo. Los bloques
recorren los días observados en orden y no distinguen un hueco entre ventanas
de una continuidad real. Los años de evaluación consecutivos del walk-forward
son contiguos y pueden concatenarse. Ventanas disjuntas deben compararse por
separado o declararse como una sola serie.

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

## Contrastes secundarios de capacidad predictiva

Son análisis secundarios declarados el 10 de octubre de 2026, antes de cualquier
resultado, en la sección `predictive_ability` de las dos comparaciones de la campaña
([separada](../../configs/evaluation/historical-masked-2000-comparison.json) y
[conjunta](../../configs/evaluation/historical-masked-2000-joint-comparison.json)).
`evaluation/predictive_ability.py` los calcula con `arch` 8.0.0 y `statsmodels` 0.15.0,
que la [revisión de bibliotecas](library-review.md) recomendó adoptar
([#493](https://github.com/GonxKZ/mars-titan/issues/493)). Las dos pertenecen al extra
`research` y solo se importan al calcular. Cargar una campaña o ejecutar `rl check` no las
necesita, y la evaluación que declara la sección comprueba que están instaladas antes de
leer ninguna fuente. No sustituyen a las
comparaciones emparejadas, que siguen siendo el contraste principal, y no se usan para
seleccionar modelos, configuraciones ni épocas. Responden a dos preguntas que el contraste
principal no cubre: si alguna variante de una familia supera a su base teniendo en cuenta
que se han probado varias a la vez, y qué brazos no se distinguen del mejor.

La sección es opcional en todas las versiones de la configuración y su presencia no cambia
ninguna otra salida. Se comprobó en procesos separados frente al código de `develop` con
los fixtures de US, CN y US+CN: informe y `sessions.parquet` idénticos, salvo la fecha de
creación, los recursos y la huella del código analizador. La huella SHA-256 de la sección,
serializada con claves ordenadas y sin espacios, es
`1adb4ad3d8b7daccd9fe5390c39966162178912cace114729ca3a4ab76bd47ef`.

### Pérdida diaria

La unidad es el día UTC, la misma que remuestrea el bootstrap principal. Para cada brazo,
las semillas se promedian sesión a sesión con `SessionSeries.average` y la pérdida del día
$t$ es la media de sus sesiones con la ponderación entre mercados declarada:

$$
L_{a,t}=\frac{1}{|S_t|}\sum_{s\in S_t}\ell_{a,s}\quad(\texttt{session}),
\qquad
L_{a,t}=\frac{1}{|M_t|}\sum_{m\in M_t}\frac{1}{|S_{t,m}|}\sum_{s\in S_{t,m}}\ell_{a,s}\quad(\texttt{market}).
$$

Un día solo entra si todos los brazos de la familia tienen alguna sesión definida en él, y
el informe cuenta los excluidos. Con el MAE no hay exclusiones. La media de $L_{a,t}$ pesa
igual cada día, así que en la vista US+CN puede diferir un poco de la media por sesiones
del contraste principal cuando un día tiene sesiones de los dos mercados. Cada familia
necesita al menos `min_days` días (250 en la campaña), que deben superar la longitud de
bloque. El mínimo admitido es 3, porque el umbral del p-valor consistente usa $\log\log T$,
que solo es positivo desde ese tamaño, y Diebold-Mariano con la corrección de Harvey
también lo necesita.

### Contrastes

| Análisis | Fuente | Implementación | Qué devuelve |
| --- | --- | --- | --- |
| Diebold-Mariano con corrección de Harvey | [Diebold y Mariano, 1995](https://doi.org/10.1080/07350015.1995.10524599), [Harvey, Leybourne y Newbold, 1997](https://doi.org/10.1016/S0169-2070(96)00719-4) | `statsmodels.tsa.stattools.diebold_mariano_test` | Estadístico, p-valor bilateral con $t_{T-1}$, retardos, factor de Harvey y p-valor de Holm dentro de la familia |
| SPA | [Hansen, 2005](https://doi.org/10.1198/073500105000000063) | `arch.bootstrap.SPA` | P-valores inferior, consistente y superior con la base como referencia |
| Reality Check | [White, 2000](https://doi.org/10.1111/1468-0262.00152) | P-valor superior del mismo SPA | P-valor |
| StepM | [Romano y Wolf, 2005](https://doi.org/10.1111/j.1468-0262.2005.00615.x) | Descenso propio sobre la API pública de `arch.bootstrap.SPA` | Variantes que superan a la base con un error de familia del 5 % |
| MCS | [Hansen, Lunde y Nason, 2011](https://doi.org/10.3982/ECTA5771) | `arch.bootstrap.MCS`, estadístico $T_R$ | Brazos incluidos con tamaño 0,10, p-valor de cada brazo y orden de eliminación |
| Longitud de bloque | [Politis y White, 2004](https://doi.org/10.1081/ETC-120028836), [corrección de 2009](https://doi.org/10.1080/07474930802459016) | `arch.bootstrap.optimal_block_length` | Longitudes circular y estacionaria, solo como diagnóstico |

El diferencial de Diebold-Mariano es $d_t=L_{\mathrm{variante},t}-L_{\mathrm{base},t}$, de
modo que un estadístico positivo indica que la variante pierde más. El objetivo madura en
una sesión, así que $h=1$, y la varianza de largo plazo usa el núcleo de Bartlett con
$\max(h-1,\lceil T^{1/3}\rceil)$ retardos, la regla por defecto de `statsmodels`, que aquí
se pasa de forma explícita y con la raíz calculada en aritmética entera. Con la corrección
de Harvey,

$$
\mathrm{DM}^{\mathrm{HLN}}=\sqrt{\frac{T+1-2h+h(h-1)/T}{T}}\,
\frac{\bar d}{\sqrt{\widehat{\mathrm{LRV}}/T}}\sim t_{T-1}.
$$

El SPA, el Reality Check, el StepM y el MCS remuestrean con el bootstrap circular por
bloques de la comparación: la misma longitud (16 días), las mismas réplicas (2.000) y la
misma semilla. Con la misma semilla, `arch` sortea exactamente los mismos bloques de días
que `paired_comparisons`. [`test_arch_bootstrap_parity.py`](../../tests/evaluation/test_arch_bootstrap_parity.py)
lo comprueba índice a índice en seis formas, también sorteando por tandas como el
contraste principal y en las réplicas internas del MCS.

Qué análisis recibe cada familia depende de sus contrastes:

| Familia | Diebold-Mariano y longitud de bloque | SPA, Reality Check y StepM | MCS |
| --- | --- | --- | --- |
| Delta, con una base y varias variantes | Cada variante frente a la base | Base como referencia | Base y variantes |
| Factorial | Cada celda frente a B, sin la interacción | B como referencia | Las cuatro celdas |
| Niveles | No | No | Todos sus brazos |
| Pares con bases distintas, como `joint_vs_separate` | Cada par | No hay referencia común | No |

Los p-valores no se corrigen entre familias, igual que en el contraste principal. Cada
familia responde a una pregunta declarada.

### Detalles de `arch` 8.0.0

La lectura del código de la versión fijada mostró tres detalles que cambian lo que se puede
afirmar. Las pruebas los fijan, de modo que una actualización de `arch` que los cambie
obliga a revisar la declaración.

- **El SPA no estudentiza.** `SPA` acepta `studentize=True`, pero la versión 8.0.0 compara
  la media de cada diferencial sin dividir por su desviación. La corrección llegó al
  repositorio después de la versión (bashtage/arch [#871](https://github.com/bashtage/arch/issues/871))
  y su revisión sigue abierta ([#879](https://github.com/bashtage/arch/issues/879),
  [#881](https://github.com/bashtage/arch/issues/881)). La sección declara por eso el
  estadístico sin estudentizar y el código pasa `studentize=False`. La varianza de cada
  diferencial, con el núcleo del bootstrap estacionario, solo interviene en el umbral del
  p-valor consistente. Sin estudentizar, una variante muy variable pesa más en el máximo
  que una estable, y el SPA pierde potencia frente a la versión del artículo.
- **`RealityCheck` es el mismo SPA.** Con el estadístico sin estudentizar y todos los modelos
  recentrados en su media, el p-valor superior es el Reality Check de White.
- **`StepM` falla en un caso.** Si un paso selecciona unos modelos y el siguiente selecciona
  el resto, la versión 8.0.0 intenta un SPA sin modelos y lanza un error
  ([#862](https://github.com/bashtage/arch/issues/862)). El descenso se hace con la API
  pública de `SPA` y la condición de parada de la corrección
  ([#863](https://github.com/bashtage/arch/pull/863)). Una prueba reproduce el fallo con
  una mejora grande y muy variable y otra pequeña y estable, y comprueba que el descenso
  propio selecciona las dos.

El MCS divide por la varianza de la diferencia de cada par. Dos brazos con las mismas
pérdidas diarias la anulan, así que esa familia queda sin MCS y el informe da el motivo.
Lo mismo ocurre con un diferencial constante en Diebold-Mariano o en el SPA.

### Coste

[`benchmarks/predictive_ability_cost.py`](../../benchmarks/predictive_ability_cost.py) mide
la sección completa del ámbito conjunto con los calendarios reales de US (2005 a 2023) y
de CN (2011 a 2023), los 24 brazos y las 13 familias de la comparación conjunta, 2.000
réplicas y bloques de 16 días, sobre pérdidas sintéticas. El
[recibo](../../reports/engineering/predictive-ability-20261010/cost.json) da 31 s de
mediana para las tres vistas (29 y 33 s en las dos repeticiones, con una carga media de 8
a 12 procesos en los 16 hilos de la CPU), con 13 a 14 s en el SPA y el StepM, 12 a 14 s en
el MCS y menos de 0,1 s en Diebold-Mariano y la longitud de bloque, y un pico de 107 MiB en
NumPy. Las dos repeticiones dieron el mismo informe. Frente a los unos 34 minutos que el
[informe de escala](../../reports/engineering/evaluation-scale-20261009/README.md)
extrapola para la comparación conjunta completa es un coste pequeño, todo el cálculo está
en `arch`, y no se ha optimizado ni pasado a C++.

### Qué no permite afirmar

- Diebold-Mariano supone que el diferencial es estacionario. Con modelos que se reestiman
  en cada ventana, su validez con parámetros estimados ([West, 1996](https://doi.org/10.2307/2171956))
  no está garantizada. El contraste de [Giacomini y White (2006)](https://doi.org/10.1111/j.1468-0262.2006.00718.x)
  trata ese caso y no está implementado.
- El MCS identifica un conjunto de brazos que no se distinguen del mejor con la confianza
  declarada. No ordena los brazos incluidos ni dice que sean equivalentes.
- StepM controla el error de familia entre las variantes de una familia, no entre familias,
  vistas ni métricas. La búsqueda previa de configuraciones queda fuera, como en el
  contraste principal.
- La longitud de Politis y White es un diagnóstico. Un valor mayor que el declarado no
  cambia la longitud de ningún contraste, porque elegirla con los datos evaluados sería
  ajustar el bootstrap con el futuro.

## Calibración común de intervalos

`calibration/conformal_quantiles.py` implementa la corrección común de los
modelos con `quantile_head_v1`. Sigue la regresión cuantílica conformalizada
(CQR) de [Romano, Patterson y Candès (2019)](https://arxiv.org/abs/1905.03222v1)
con su puntuación simétrica. Para el intervalo central de nivel nominal
$1-\alpha$ con extremos $q_{\mathrm{bajo}}$ y $q_{\mathrm{alto}}$:

$$
E_i=\max\{q_{\mathrm{bajo},i}-y_i,\; y_i-q_{\mathrm{alto},i}\},\qquad
Q=E_{(k)},\quad k=\lceil (n+1)(1-\alpha)\rceil,
$$

donde $E_{(k)}$ es el $k$-ésimo menor valor de las $n$ puntuaciones del tramo de
calibración. El intervalo calibrado es $[q_{\mathrm{bajo}}-Q,\;q_{\mathrm{alto}}+Q]$.
$Q>0$ ensancha un intervalo sobreconfiado y $Q<0$ estrecha uno demasiado
amplio. El orden $k$ se calcula con fracciones exactas, así que $0{,}8$ y
$0{,}95$ no dependen del redondeo binario. Si $k>n$ o el grupo no alcanza el
mínimo declarado de filas, la corrección queda sin definir con su motivo. Por
ejemplo, con nueve filas $k=\lceil 10\cdot 0{,}95\rceil=10$ y el intervalo del
95 % no se calibra.

La configuración declara las coberturas 0,8 y 0,95, el tramo `calibration`, un
grupo por mercado y el mínimo de filas. Cada mercado recibe su propia
corrección, porque una corrección común podría cubrir de más un mercado y de
menos el otro. El registro guarda niveles, órdenes, correcciones, la cobertura
en calibración sin corregir y corregida, y `coverage_guaranteed=False`. La
garantía de CQR exige intercambiabilidad, que la dependencia temporal no
asegura. Su huella SHA-256 se calcula antes de abrir las predicciones de
evaluación y se conserva en el informe.

La mediana no cambia, de modo que la predicción puntual y el MAE tampoco. Para
conservar el orden de los cinco cuantiles, cada extremo corregido se ensancha
como mucho hasta la mediana y el intervalo del 95 % hasta el del 80 %
(`widen_to_median_and_inner_interval`). Solo se ensancha respecto a la
corrección CQR, nunca se estrecha, así que la cobertura en calibración no baja
de la nominal. El informe cuenta las filas en las que se aplicó esa regla.

La [paridad con MAPIE 1.5.0](../../tests/calibration/test_conformal_mapie_parity.py)
ejecuta `ConformalizedQuantileRegressor` grupo a grupo, con estimadores ya ajustados que
devuelven los cuantiles guardados y `symmetric_correction=True`. Con 400 y 257 filas de
calibración, una corrección positiva y otra negativa, los extremos corregidos y la
mediana coinciden bit a bit. Las dos implementaciones difieren en tamaños concretos,
que la prueba recorre de $n=1$ a $n=120$:

| Caso | Propia | MAPIE |
| --- | --- | --- |
| $k>n$ ($n\le3$ con 0,8 y $n\le18$ con 0,95) | Sin corrección, con su motivo | Rechaza la calibración |
| $k=n$ con $n<1/\alpha$ ($n=4$ con 0,8 y $n=19$ con 0,95) | La puntuación máxima | Rechaza la calibración |
| $(n+1)(1-\alpha)$ entero ($n+1$ múltiplo de 5 con 0,8 o de 20 con 0,95) | $E_{(k)}$ | $E_{(k+1)}$, más conservadora |
| Resto | $E_{(k)}$ | $E_{(k)}$ |

La tercera fila se debe a que MAPIE calcula `np.quantile(E, (1-α)(1+1/n), method="higher")`,
cuyo índice desde cero es $\lceil (1-\alpha)(1+1/n)(n-1)\rceil$. Coincide con $k-1$
salvo cuando $(n+1)(1-\alpha)$ es entero, en cuyo caso toma el estadístico siguiente.
El orden propio es el de Romano, Patterson y Candès (2019), así que la regla del
proyecto no cambia.

## Evaluación walk-forward de la edición desde 2000

`evaluation/walk_forward_comparison.py` aplica estas métricas a las
predicciones de la campaña desde 2000 sin abrir la reserva de 2024. Recibe dos
documentos. La [configuración declarada](../../configs/evaluation/historical-masked-2000-comparison.json)
fija antes de evaluar la política de entradas, los protocolos de cada ámbito
(US, CN y US+CN), las ventanas, los brazos con su salida y semillas, las
métricas, la calibración y las familias de contrastes. Un manifiesto de fuentes,
que se generará cuando existan las predicciones, enlaza cada brazo, semilla y
ventana con sus archivos del tramo de evaluación y, si emite cuantiles, del
tramo de calibración.

La política de entradas siempre se declara. Cada vista de ventana se valida con
`temporal_contracts` bajo esa política, de modo que una vista estricta no pasa
como vista con máscaras ni al revés. Además se rechaza cualquier mezcla entre
brazos: otra política declarada en una fuente, otra vista, otro protocolo u otra
ventana, otra edición (la huella del corpus padre de las vistas), semillas o
ventanas ausentes o sobrantes y filas distintas. La identidad de una fila es
(mercado, activo, instante) dentro de su ventana. Si dos brazos no evalúan las
mismas filas con los mismos objetivos, el error indica cuántas filas solo están
en cada brazo y cuántas tienen otro objetivo.

Cada archivo de evaluación debe contener solo filas del tramo de evaluación que
el protocolo asigna a su ventana, y el de calibración solo filas de su tramo.
Una fila fuera de su tramo se rechaza y no se filtra en silencio. Cualquier fila
con instante igual o posterior al 1 de enero de 2024 se rechaza con un mensaje
propio, y ninguna ventana declarada termina después de esa fecha.

La agregación sigue la definición de este documento:

- Por ventana, cada brazo y semilla se puntúa con `score_sessions` sobre su
  tramo de evaluación.
- Sobre todas las ventanas, las sesiones se unen con `SessionScores.concatenate`
  y cada sesión pesa lo mismo. El resultado coincide con el de un panel único con
  todas las filas, como comprueban las pruebas. Las ventanas de evaluación son
  años consecutivos y no repiten sesiones.
- Por mercado, el ámbito conjunto informa US+CN con la ponderación declarada y
  cada mercado por separado con `select_sessions`.
- Por sesión, `sessions.parquet` conserva los estadísticos de cada sesión, brazo,
  semilla, ventana y variante (cuantiles brutos o calibrados).

Los resúmenes se dan por semilla. Los contrastes promedian primero las semillas
sesión a sesión con `SessionSeries.average` y aplican `compare_series` a cada
familia y métrica declaradas, con la longitud de bloque, réplicas y semilla de
la configuración. Una familia con algún brazo sin cuantiles no recibe pinball y
lo indica. El control cero se construye con las mismas filas y objetivo y
predicción nula. Para cada brazo con cuantiles se informa el error de cobertura
de los intervalos del 80 % y del 95 % con y sin calibración, con su intervalo
por bloques y la diferencia entre ambos.

Los modelos sin cuantiles, Ridge, XGBoost y el control cero, tienen las métricas
de intervalo ausentes con su motivo, nunca a cero. La abstención usa las dos
reglas ya definidas: la predicción nula como abstención (acierto condicionado y
cobertura de llamadas) y el intervalo que excluye el cero como afirmación de
signo (error de signo comprometido y fracción de afirmaciones). No existe una
regla de abstención con umbral declarada y no se crea en esta evaluación.

La configuración de la campaña declara el MAE como métrica primaria,
ponderación por sesión, un mínimo de 30 activos para el Rank IC (con 30 activos
el error típico de una correlación de rangos ronda $1/\sqrt{29}\approx 0{,}19$),
1.000 filas por mercado para calibrar, bloques de 16 días con sensibilidad a 5,
10 y 40, 2.000 réplicas y la semilla 20261009. Los 16 días se aproximan a
$P^{1/3}$ con unos 4.800 días de evaluación en US y unos 3.300 en los otros
ámbitos. Las familias son las referencias frente al control cero, los controles
de Titans-MAC frente a `transformer_direct`, las políticas de escritura M1 a M3
frente a M0, los refinamientos K = 2 y 4 sobre M1, el factorial CM-v1 y el nivel
de cada brazo. Tomar M1 como base de K es una propuesta pendiente de revisión.
M3 todavía no está definida en el código y su brazo exige esa definición antes
de evaluar. La versión 2 de la configuración añade los
[estratos por presencia de modalidades](#estratos-por-presencia-de-modalidades),
un análisis secundario que no cambia nada de lo anterior. La versión 3 añade la
[ablación de modalidades en inferencia](#ablación-de-modalidades-en-inferencia),
otro análisis secundario que tampoco cambia las salidas anteriores.

La versión 4, declarada el 9 de octubre de 2026 antes de cualquier predicción
real, completa la comparación en tres puntos:

- **Métricas contrastadas.** Además del MAE, el MSE, la dirección, el Rank IC y la
  pinball, se contrastan la precisión al alza y a la baja, el Brier del signo y la
  puntuación de los intervalos del 80 % y del 95 %. El informe añade en todas las
  versiones la fiabilidad del signo (`sign_reliability`).
- **Familias arquitectónicas.** GRU frente a GRU episódica (`episodic_gru`), el
  Transformer compacto frente a Titans-MAC `transformer_direct` (`encoder_change`,
  el cambio de codificador con fusión y cabeza comunes), `mac_online` frente a
  MARS-TITAN M0 (`episodic_reader`, el lector episódico con la memoria del núcleo
  activa), M1 frente a CM-v1 B (`cm_v1_base`) y la GRU episódica frente a
  `mac_online` (`core_vs_episodic_gru`). Con el factorial CM-v1 y las familias
  anteriores quedan cubiertos B, B+C, B+M y B+C+M, MARS-TITAN con las
  ampliaciones apagadas (M0) frente a Titans-MAC y el núcleo frente a la GRU
  episódica. Cada familia lleva su propia corrección por máximo estudentizado.
- **Cartera.** La sección `long_short` declara la [cartera larga y corta por
  cuartiles](long-short-portfolio.md), un análisis financiero secundario que
  calcula `long_short_comparison` con las mismas fuentes.

El 10 de octubre de 2026, también antes de cualquier predicción real, la versión 4
añade el [control en línea del Transformer](../engineering/transformer-online-control.md)
(`transformer_compact_online`, #443) y dos familias. `online_learning` contrasta el
control con el Transformer compacto congelado y `memory_vs_online_learning` contrasta
MARS-TITAN M1 con el control. El control recibe las mismas etiquetas maduras que el
banco de M1 en los mismos instantes, así que la segunda familia comprueba si la mejora
de MARS-TITAN se explica solo por seguir aprendiendo. El brazo entra también en la
familia de niveles, lo que añade un nivel a su corrección por máximo estudentizado.

La versión 5 añade a la versión 4 el [diseño conjunto](walk-forward-2000.md#comparación-con-los-controles-separados)
de la campaña A v2 (`joint_design`). Un mercado solo cuenta en las ventanas en las que es
elegible y los controles separados de US y CN se comparan con el brazo conjunto
restringido a las filas de su mercado. Las métricas, la fiabilidad del signo y la cartera
aplican las mismas exclusiones. El control en línea y sus dos familias se evalúan en el ámbito
conjunto, porque los ámbitos de un mercado solo comparan los tres controles separados con
su brazo conjunto.

Las semillas se agregan así. La comparación solo lee el caso elegido de cada
brazo, que la campaña A repite con las semillas 42, 43 y 44. Cada semilla tiene
su resumen y los contrastes, la fiabilidad del signo y la cartera usan la media
de las semillas sesión a sesión. La incertidumbre sale solo del remuestreo de
días y una semilla nunca cuenta como una sesión más. Un brazo determinista con
una semilla, como Ridge, entra con su serie tal cual. Los casos de búsqueda, que
solo usan la semilla 42, no entran en la comparación y quedan en los recibos de
selección y el registro de ensayos.

Con el diseño conjunto de la campaña A, el contraste del modelo conjunto frente al
separado en cada mercado y la exclusión de las métricas chinas anteriores a 2011
se declaran en la comparación conjunta de `feat/campaign-a-joint-design` (#363),
que se construye sobre esta versión. No forman parte de este archivo.

### Brazos postentrenados

La [declaración de la comparación postentrenada](../../configs/posttraining/historical-masked-adapter-comparison-a.json)
no enumera brazos. `posttraining/stage_comparison.py` los deriva del plan de la
etapa de adaptadores y forma una comparación por padre y ámbito con el padre
congelado, la continuación completa y los brazos adaptados de la matriz para su
familia. Así una familia nueva de la matriz entra sin reescribir nada. En el
[walk-forward por etapas](../engineering/masked-posttraining.md#etapa-por-ventana-de-la-campaña)
de A, el padre congelado es el trabajo `frozen` de la etapa, que aplica a la
ventana k el estado elegido por la base en k-1, y el brazo base reentrenado en k
queda como nivel fuera de las familias. Como la primera ventana de cada ámbito
no tiene postentrenamiento, la comparación empieza en la segunda. Las familias
declaradas son `versus_frozen_parent` (adaptados y continuación menos el padre
congelado) y `versus_full_continuation` (adaptados menos la continuación), más el
nivel de cada brazo. Todo lo demás se hereda de la comparación de la campaña:
protocolos, métricas, calibración común, remuestreo y secciones secundarias. Hoy
salen nueve padres: las tres redes recurrentes y DLinear con once brazos, el
Transformer con diecisiete porque la matriz le da puntos de lectura, y los cuatro
brazos de Titans-MAC con entre siete y catorce. La validación de los recibos por
etapas elige el predictor de la cadena y no entra en esta comparación.

La [declaración de A v2](../../configs/posttraining/historical-masked-adapter-comparison-a-v2.json),
fijada el 10 de octubre antes de cualquier resultado, añade dos papeles. `chain` es
el predictor de la cadena (`<brazo>__chain`), cuyas predicciones de cada ventana y
semilla son las del trabajo que eligió su `selection.json`, y `base_retrain` es el
brazo base reentrenado en la ventana. Las familias `versus_frozen_parent` y
`versus_full_continuation` incluyen la cadena como variante, y `versus_base_retrain`
contrasta la cadena con el reentreno, la comparación que el diseño por etapas
informa aparte. Si la campaña declara `walk_forward_stages`, la declaración debe
contrastar esos dos papeles. Salen veinte padres en el ámbito conjunto, los 22
brazos de la etapa salvo Ridge y XGBoost. Cada comparación conserva la
elegibilidad por mercado del modelo conjunto, así que China solo entra en
calibración y métricas desde `fold-006`, y deja fuera los controles separados,
que solo actúan en los ámbitos de un mercado.

El manifiesto de fuentes de un padre une las predicciones del padre, leídas del
manifiesto ya validado de la campaña, con los recibos confirmados de la etapa. Se
rechaza una etapa con otra declaración, otras vistas o un recibo de otra ejecución,
vista o identidad, y la comparación exige las mismas filas y objetivos en todos los
brazos. Cada padre es un análisis secundario propio, sin corrección entre padres,
y no sirve para elegir la arquitectura base. Sin ablación de modalidades
conectada para estos brazos, su sección queda pendiente en el informe.

Con la [retención v2](training-campaign-2000.md#retención-v2-ventana-a-ventana), las
filas de calibración y evaluación del modelo base reentrenado se liberan al terminar
cada ventana, y la comparación de cada padre las lee. Por eso la fase de agregados del
recorrido guarda también, para cada padre con trabajos de la etapa en la ventana, el
manifiesto de fuentes de esa ventana (`sources/windows/<ventana>/<ámbito>/<padre>.json`
en la salida de la etapa) y sus agregados por sesión
(`retention/aggregates/posttraining/<padre>/<ámbito>/<ventana>.npz`), puntuados con la
configuración del padre limitada a la ventana, que conserva su huella. Con `--aggregates`,
`stage_comparison evaluate` lee esos agregados y no abre ninguna fila. Las pruebas
comprueban que el informe y la tabla por sesión salen idénticos a los que leen las
filas, intervalos del bootstrap por bloques incluidos, con la misma semilla y las
sesiones de dos ventanas. No hace falta ninguna tolerancia. La cartera de un padre lee
filas y no se combina con agregados.

Falta conectar los productores de predicciones. Las referencias neuronales con
retención `heldout_full_train_sessions_v1` ya escriben archivos por tramo con
`sample_id`, `asset_id`, mercado, instante, objetivo, predicción y, con
`quantile_head_v1`, sus cinco columnas. El entrenador cronológico de Titans-MAC
todavía no exporta predicciones por fila y la GRU episódica, los tabulares y
CM-v1 necesitan el mismo formato. El manifiesto de fuentes se generará a partir
de sus recibos cuando existan.

## Estratos por presencia de modalidades

Es un análisis secundario y descriptivo, declarado el 9 de octubre de 2026 antes
de cualquier resultado. No se usa para seleccionar modelos, configuraciones ni
épocas, y no cambia la conclusión que se extraiga de la métrica principal, que
sigue siendo el MAE residual por sesión de toda la población evaluada. Responde a
otra pregunta: cómo rinde cada brazo según las modalidades que tenía cada muestra,
sobre todo en la subpoblación con las cuatro modalidades (precios y gráficos,
noticias, fundamentales y macro).

En la edición v3 desde 2000, con 17.076.024 muestras de 5.023 activos, precios,
gráficos y macro están presentes en todas las filas. Los fundamentales cubren el
34,4 % del total (40,6 % en EE. UU. y 0,8 % en China) y las noticias el 17,9 %
(18,6 % y 13,7 %). Por eso los estratos solo distinguen noticias y fundamentales:

| Estrato | Noticias | Fundamentales | Modalidades de la muestra |
| --- | --- | --- | --- |
| `news_and_fundamentals` | Sí | Sí | Las cuatro: precios y gráficos, noticias, fundamentales y macro |
| `news_only` | Sí | No | Precios y gráficos, noticias y macro |
| `fundamentals_only` | No | Sí | Precios y gráficos, fundamentales y macro |
| `neither` | No | No | Precios y gráficos y macro |

`news_and_fundamentals` es el estrato de interés declarado (`focus`). Los cuatro
estratos forman una partición de las filas evaluadas, de modo que cada fila cae
exactamente en uno.

### Declaración

La sección `modality_strata` de la versión 2 de la [configuración](../../configs/evaluation/historical-masked-2000-comparison.json)
fija el estatus (`secondary_descriptive`), el uso permitido, la fuente de la
presencia, las tres modalidades que se dan por presentes, los cuatro patrones, el
estrato de interés, las métricas de los contrastes (solo el MAE), la prohibición
de recalibrar, los umbrales y la corrección por comparaciones múltiples. Su huella
SHA-256 al declararla es
`378a5cfc8640f9b91b2ee739c328cca300422f22799591248fba1e0b74137c27`. El cargador
rechaza cualquier otro valor de esos campos, una sección en la versión 1, una
versión 2 sin sección y la sección con la política estricta. Una configuración de
versión 1 produce las mismas salidas que antes de este cambio. Se comprobó en
procesos separados frente al código anterior con los fixtures de US, CN y US+CN:
informe y `sessions.parquet` idénticos, salvo la fecha de creación, los recursos y
la huella del código analizador, que cambian por definición.

### Origen de los bits de presencia

Para cada ventana, `evaluation/modality_strata.py::view_presence` abre su vista
con `CorpusDataset`, que comprueba las huellas de muestras, etiquetas y precios
frente al manifiesto de la vista. Toma las filas que las etiquetas asignan al
tramo de evaluación y lee la columna `presence` de `samples.parquet` en la
posición `sample_row` de cada etiqueta. Comprueba la forma de cinco booleanos, que
la presencia de noticias coincida con `news_count` y que el instante de la
etiqueta sea el de la muestra. No vuelve a leer los vectores, cuya coherencia con
las máscaras ya se verificó al preparar la edición y que las huellas protegen.

La tabla resultante (activo, mercado, instante y objetivo) pasa por la misma
comprobación que dos brazos, `_same_rows`: debe tener exactamente las filas y los
objetivos del primer brazo, y el error cuenta cuántas filas sobran, faltan o
cambian de objetivo. Solo entonces se alinea cada fila con el orden canónico del
panel. Si alguna fila evaluada no tiene precios, gráficos y macro, la declaración
deja de describir la población. La sección entera queda `not_estimable` con el
número de filas afectadas en cada ventana y el resto del informe no cambia.

### Métricas por estrato

- **MAE por sesión.** En el estrato $k$, el error de la sesión usa solo sus filas
  de $k$, y las sesiones sin filas de $k$ no entran:
  $\operatorname{MAE}_{s,k}=\frac{1}{n_{s,k}}\sum_{i\in s\cap k}|\hat y_i-y_i|$.
  Se agrega con la ponderación declarada (`session`), uniendo ventanas, para el
  ámbito y para cada mercado. Es `score_sessions` aplicado al subconjunto de filas
  del panel ya validado.
- **Relación con la métrica principal.** Dentro de cada sesión,
  $\operatorname{MAE}_s=\sum_k \frac{n_{s,k}}{n_s}\operatorname{MAE}_{s,k}$, y el
  MAE por filas es la media de los estratos con pesos $N_k/N$. El MAE por sesión
  agregado solo es la media de los estratos ponderada por sus sesiones cuando cada
  sesión pertenece a un único estrato. Las pruebas comprueban las tres
  identidades. En general, un estrato pesa en la métrica principal según la
  fracción de filas que ocupa en cada sesión, no según su número de sesiones.
- **Contrastes.** Las familias, la longitud de bloque, las réplicas, la semilla y
  la sensibilidad son las de la comparación principal, aplicadas al MAE del
  estrato. Las semillas se promedian sesión a sesión como en la ruta principal.
- **Intervalos.** Cobertura y anchura de los intervalos del 80 % y del 95 % con el
  calibrador común de cada ventana, ajustado una vez por mercado con todas las
  filas de calibración. Los cuantiles calibrados del panel completo se restringen
  a las filas del estrato. Ningún estrato vuelve a ajustar el calibrador, y las
  pruebas cuentan el mismo número de ajustes con y sin estratos. Se informa además
  el error de cobertura con su intervalo por bloques, en bruto y calibrado, como en
  la sección principal.

### Umbral y celdas no estimables

Antes de ver ningún resultado se fijan `min_rows=1000` y `min_sessions=50`. Una
celda se informa si alcanza los dos mínimos. Las celdas son cada ventana y
mercado, el ámbito completo y cada mercado. Por debajo del umbral la celda aparece
con `estimable=false`, sus filas, sus sesiones y el motivo con el umbral, y sus
métricas, contrastes y coberturas quedan a `null` con ese motivo. Nunca se omite.
Una ventana por debajo del umbral sigue aportando sus filas a la celda del ámbito,
que se define sobre todas las ventanas. Las 1.000 filas coinciden con el mínimo
de la calibración por mercado y las 50 sesiones superan el bloque más largo de la
sensibilidad (40 días). Con un 0,8 % de fundamentales en China, es de esperar que
muchas celdas chinas con fundamentales queden no estimables. Se informará así.

### Comparaciones múltiples

`compare_series` ya da intervalos simultáneos por máximo estudentizado dentro de
cada familia. Entre estratos y ámbitos se aplica además Bonferroni sobre el número
de celdas declarado, cuatro estratos por el número de ámbitos: 12 en US+CN y 4 en
US o en CN. La confianza de contrastes y coberturas es $1-0{,}05/12\approx
0{,}99583$ en el ámbito conjunto y $1-0{,}05/4=0{,}9875$ en los demás. El
bootstrap de un estrato remuestrea solo los días con sesiones de ese estrato, así
que con un estrato disperso un bloque abarca más tiempo de calendario.

### Informe

El informe añade la sección `modality_strata` con `declaration`, `status`,
`presence` (filas y filas incompletas por ventana), `recalibrated=false`,
`multiplicity`, `population` (patrón, filas, sesiones, fracción de filas y
estimabilidad por ámbito y por ventana y mercado), `arms` (MAE por ámbito con los
intervalos calibrados y MAE por ventana y mercado, para cada brazo, semilla y
estrato), `contrasts` e `interval_calibration`. `sessions.parquet` no cambia y
`analysis_source_sha256` añade este módulo y `training/corpus_inputs.py`.

### Coste

Con una ventana sintética de 625.000 filas de evaluación (2.500 activos y 250
sesiones) y 155.000 de calibración, el control cero, un brazo puntual y uno con
cuantiles, dos hilos y una carga media cercana a 3, cuatro repeticiones tardaron
1,93 s sin la sección y entre 3,8 y 4,2 s con ella. El pico de memoria pasó de
1,06 a 1,09 GiB. La mayor parte del aumento es volver a ordenar y puntuar los
subpaneles, incluido el Rank IC por sesión, que los estratos no usan. El resto es
sobre todo construir el panel de presencia y alinear sus filas. Esta medida
sustituye la lectura de la vista por la tabla ya generada. Leer una vista real
comprueba las huellas de todos sus archivos y no se ha medido sobre la edición.

### Qué no permite afirmar

Las diferencias entre estratos describen subpoblaciones distintas, no el efecto
de añadir una modalidad. Tener noticias o fundamentales se asocia al tamaño del
activo, al mercado y al periodo, así que un MAE menor en
`news_and_fundamentals` no demuestra que esas modalidades lo reduzcan. Para eso
harían falta controles con la misma fila y la modalidad enmascarada. Los
contrastes entre brazos dentro de un estrato sí son emparejados, porque todos los
brazos evalúan las mismas filas.

## Ablación de modalidades en inferencia

Es un análisis secundario y descriptivo, declarado el 9 de octubre de 2026 antes
de cualquier resultado. Los estratos comparan subpoblaciones distintas. La
ablación compara cada fila consigo misma: el estado elegido de cada brazo,
semilla y ventana vuelve a predecir la evaluación con noticias, fundamentales o
ambos leídos como ausentes, sin reentrenar ni recalibrar. Mide cuánto depende
cada brazo de esas modalidades en las filas que las tenían. No se usa para
seleccionar modelos, configuraciones ni épocas, y no cambia la conclusión que se
extraiga del MAE residual por sesión de toda la población.

### Declaración

La sección `modality_ablation` de la versión 3 de la
[configuración](../../configs/evaluation/historical-masked-2000-comparison.json)
fija el estatus (`secondary_descriptive`), el uso permitido, las tres variantes
(`mask_news`, `mask_fundamentals` y `mask_news_and_fundamentals`), el
enmascaramiento, la causa de ausencia (`modality_ablation`), el estado de
partida, la política de memoria, el tramo (`evaluation`), las filas, la métrica,
la prohibición de recalibrar, los umbrales, la corrección por comparaciones
múltiples y lo que el análisis no mide. Su huella SHA-256 al declararla es
`c942946e1c2bcbd3d2e700cb0cdc0932b5451b4487ccf4bb57dfe97c2eff61c1`. El cargador
rechaza cualquier otro valor de esos campos y una versión 3 sin la sección. Las
configuraciones de versión 1 y 2 producen las mismas salidas que antes de este
cambio. Se comprobó en procesos separados frente al código anterior con 17
huellas idénticas: lotes del lector y observaciones de la ruta normal, informe y
`sessions.parquet` de las versiones 1 y 2, traslados neuronal y tabular y las
predicciones de una ventana de Titans-MAC.

### Enmascaramiento

`data/modality_ablation.py` escribe en la tabla de muestras lo mismo que tiene
una ausencia real según el contrato de máscaras: bit de presencia falso, vector
con el relleno de ausencia (`missing_fill`, cero en valores, máscaras y edades),
disponibilidad nula y la causa `modality_ablation` en `missing_reasons`. Las
noticias pasan además a cero eventos, porque el lector exige que presencia y
recuento coincidan. Las filas que ya carecían de la modalidad conservan su causa
original. Después la tabla pasa por las mismas comprobaciones que cualquier
muestra. La fila original se valida antes de enmascararla, así que una fila
incoherente no queda oculta.

`CorpusDataset(..., modality_ablation=...)` aplica la variante en
`_sample_group`, el único punto de decodificación de los lotes supervisados, de
las observaciones y del índice de observaciones de los modelos con memoria. La
identidad de la vista no cambia y la ablación se declara en un campo propio de
cada recibo. Las pruebas comparan la lectura enmascarada con un corpus generado
con la modalidad ausente de verdad y obtienen los mismos lotes, observaciones y
eventos del índice, con y sin bloques. Sin el parámetro, la lectura no cambia.

### Estado, memoria y calibración

- **Estado.** En una ventana reentrenada se usa el estado elegido en ella y en una
  ventana trasladada de la variante B, el del ancla que la campaña base traslada.
  Se reutiliza el traslado de cada familia (referencias neuronales y tabulares,
  GRU candidata, Titans-MAC, MARS-TITAN y CM-v1) con el parámetro
  `modality_ablation`. Solo se predice la evaluación. La propia ventana del ancla
  solo se admite con la ablación, porque su estado se eligió con la validación,
  anterior a la calibración y a la evaluación.
- **Memoria.** Titans-MAC, MARS-TITAN, CM-v1 y la GRU candidata tienen estado en
  línea. Su predicción enmascarada recorre el calentamiento y el tramo con las
  mismas entradas ablacionadas y empieza con la memoria inicial, igual que la
  predicción original. La pregunta es qué ocurre si la modalidad no existe en
  todo lo que el modelo observa. Enmascarar solo las filas medidas mezclaría dos
  regímenes de entrada en la misma memoria y no correspondería a ninguna ausencia
  real. Por eso, en estos modelos, una fila sin la modalidad puede cambiar de
  predicción. El informe cuenta esas filas por ventana (`unaffected_changed`) y no
  las usa en la métrica. En un modelo sin memoria ese recuento debe ser cero y las
  pruebas lo comprueban. Con pesos iniciales y un optimizador que no modifica
  pesos, la predicción enmascarada de Titans-MAC coincide bit a bit con la del
  mismo estado sobre un corpus sin noticias ni fundamentales. Con el enlace
  episódico compilado, los traslados enmascarados de MARS-TITAN y de la GRU
  candidata también coinciden con los de un corpus sin esas modalidades.
- **Calibración.** Los cuantiles enmascarados se corrigen con el calibrador común
  de la ventana, ajustado una vez con las predicciones originales de calibración.
  Nunca se vuelve a ajustar y las pruebas cuentan el mismo número de ajustes con y
  sin ablación.

### Métrica, filas e incertidumbre

Las filas afectadas por una variante son las filas evaluadas con al menos una de
sus modalidades presente, leídas de la vista como en los estratos. Son las únicas
cuya entrada cambia. La métrica es la diferencia emparejada del MAE por sesión en
esas filas,
$\Delta=\operatorname{MAE}^{\text{enmascarado}}-\operatorname{MAE}^{\text{original}}$,
con la ponderación declarada, uniendo ventanas, para el ámbito y para cada
mercado. Un valor positivo indica que el error del brazo crece sin la modalidad.
Se informan también la cobertura y la anchura de los intervalos del 80 % y del
95 % de ambas predicciones con el mismo calibrador, y los recuentos de filas
afectadas y no afectadas por ventana y mercado.

Para cada variante y vista, los contrastes forman una familia sobre los brazos
con `compare_series` y el contraste enmascarado menos original. Las semillas se
promedian sesión a sesión y el bloque, las réplicas, la semilla y la sensibilidad
son los de la comparación principal. Entre variantes y vistas se aplica
Bonferroni sobre tres variantes por el número de vistas: 9 celdas en US+CN, con
confianza $1-0{,}05/9\approx 0{,}99444$, y 3 en US o en CN, con $1-0{,}05/3\approx
0{,}98333$. Los umbrales son los de los estratos, 1.000 filas y 50 sesiones. Por
debajo, la celda aparece con su motivo y sin métricas.

### Etapa, recuento y coste

`training/modality_ablation_stage.py` es una etapa posterior de la campaña
(`LATER_STAGES`) que se ejecuta con `scripts/run_masked_campaign.py ablation
check|run|sources`. Parte de una campaña base confirmada, comprueba la protección
del aprendizaje antes de crear salidas y antes de cada trabajo pendiente, e
instala durante la ejecución un gancho global que rechaza cualquier paso de un
optimizador de PyTorch. Cada trabajo confirma su identidad, el estado de partida
y unas filas y objetivos iguales a los de la campaña base en esa ventana.
`sources` publica el manifiesto que lee la comparación con `--ablation-sources`.
Sin ese manifiesto, la sección queda `not_computed` y el resto del informe no
cambia.

Las etapas declaradas para A y B prevén 4.185 predicciones cada una, 1.395 por
variante: 31 pares de brazo y semilla (cinco referencias neuronales y cuatro
brazos de Titans-MAC con tres semillas, XGBoost con tres y Ridge con una) en 19
ventanas de US, 13 de CN y 13 de US+CN. B cuesta lo mismo que A porque cada
ventana predice con el estado que la campaña usa en ella. El comando `throughput`
acepta `--ablation-stage` y estima sus horas con los caudales de inferencia ya
medidos, con la evaluación y su calentamiento en las familias cronológicas. Los
tabulares quedan sin estimar. No se ha medido el coste del análisis sobre la
edición real.

### Qué no mide

- No mide un efecto causal económico. Enmascarar cambia la entrada del modelo, no
  la información disponible en el mercado.
- Dependencia no es utilidad. Un brazo puede cambiar mucho su predicción sin la
  modalidad y no ganar precisión con ella, o al revés.
- No equivale a entrenar sin la modalidad. El estado se ajustó con ella y no se
  ha adaptado a su ausencia, mientras que un modelo entrenado sin ella podría
  compensarla con otras entradas.
- Una fila enmascarada combina rasgos poco frecuentes en el ajuste, como un activo
  grande sin noticias. El modelo puede extrapolar en esas combinaciones.
- Las filas con noticias o fundamentales no son una muestra al azar. Las
  diferencias entre variantes describen poblaciones distintas, igual que los
  estratos.

## Matriz de comparaciones y atribución por componentes

La [matriz de la campaña A](../../configs/evaluation/comparison-matrix-a.json) se
declaró el 10 de octubre de 2026, antes de cualquier resultado. Fija todas las
comparaciones que se medirán además de las familias de la comparación walk-forward:
entre familias, cada variante de MARS-TITAN frente a cada referencia, el núcleo
Titans-MAC frente a una implementación pública de referencia, la cadena por etapas
frente al reentreno, al padre trasladado y a la continuación, y el Transformer en línea
como control de «seguir aprendiendo». Añade la atribución por componentes de dos
linajes, Titans y CM-v1. La [tabla del protocolo](protocol.md#qué-pregunta-responde-cada-comparación)
resume en lenguaje llano qué pregunta responde cada bloque.
`evaluation/comparison_matrix.py` valida la declaración y compila 47 familias con 362
contrastes. `check` los cuenta sin leer datos y `missing` calcula los brazos que faltan.

### Declaración

Cada pregunta tiene un tipo. `pairwise` compara todos los pares de un grupo, `against`
compara cada miembro de un grupo con cada referencia de otro y `pairs` declara pares
[base, variante], con la plantilla `{arm}` repetida para cada miembro de un grupo. Cada
contraste es variante menos base, como `delta`. Un brazo debe ser de la comparación de la
campaña, condicionado (`transformer_compact_online`, `titans_reference_mac` y los brazos
de integración que aún no están en `develop`), derivado (`<brazo>__chain`,
`<brazo>__frozen_parent` y `<brazo>__full_continuation`) o candidato de atribución. Un
nombre desconocido se rechaza al cargar. Los condicionados y derivados tienen su propio
plan y su condición. Los candidatos no forman parte de ningún plan.

La corrección múltiple es la del resto del proyecto. Cada familia es una unidad con su
máximo estudentizado y no hay corrección entre familias. Las preguntas con plantilla
forman una familia por miembro, igual que la comparación postentrenada forma una por
padre. Ninguna familia supera los 64 contrastes de `compare_series`.

### Atribución por componentes

Un linaje declara componentes binarios, sus dependencias estructurales y el conjunto de
componentes de cada brazo con nombre (`evaluation/component_attribution.py`). El linaje
Titans va del Transformer compacto a M3: recorrido directo de Titans, atención MAC,
memoria persistente con lectura y puerta, actualización en inferencia, lector sin
contenido (M0), contenido del banco (M1), escritura por error maduro (M2) y anomalía con
relevancia (M3). K = 2 y 4, los episodios de la primera lectura y las dos variantes de B6
son componentes fuera de la escalera. El linaje CM-v1 tiene C y M sobre su B. Con $v(S)$
la métrica media del brazo que activa el conjunto $S$:

- **Escalera acumulada**: $v(S_{i+1})-v(S_i)$ en el orden declarado y el total
  $v(S_n)-v(S_0)$, que es exactamente la suma de los pasos. Cada paso depende del orden.
- **Dejar uno fuera**: $v(C)-v(C\setminus D(c))$, con $C$ el conjunto completo y $D(c)$
  el componente y todo lo que depende de él. El informe dice qué retira cada contraste.
- **Efectos condicionados**: $v(S\cup\{c\})-v(S)$ para cada par de brazos con nombre que
  solo difiere en $c$. Es la respuesta directa a cuánto aporta una parte según lo demás.
- **Interacción** en un contexto declarado:
  $v(S+a+b)-v(S+a)-v(S+b)+v(S)$, la de CM-v1 con $S=B$.
- **Shapley** de un juego con jugadores $P$ y contexto $S$:

$$
\phi_i=\sum_{T\subseteq P\setminus\{i\}}\frac{|T|!\,(n-|T|-1)!}{n!}
\big[v(S\cup T\cup\{i\})-v(S\cup T)\big],
\qquad \sum_i\phi_i=v(S\cup P)-v(S).
$$

  Solo se calcula si cada coalición respeta las dependencias. Con dos jugadores es la
  media de los dos efectos condicionados. En el juego de los ocho componentes del linaje
  Titans, 236 de las 256 coaliciones activan un componente sin sus dependencias (por
  ejemplo, actualizar una memoria que no se lee). Ese valor no existe y el informe lo
  declara como limitación en lugar de aproximarlo.

Todas estas cantidades son combinaciones lineales de brazos, así que se estiman con
`compare_series` sobre las mismas sesiones y los mismos días remuestreados. Un conjunto
sin brazo deja su contraste pendiente con lo que falta.

### Vistas de métrica

La evaluación no vuelve a puntuar predicciones. `evaluation/session_table_contrasts.py`
lee las tablas por sesión que publican la comparación walk-forward y la cartera, con su
huella, y reconstruye las mismas series. Las pruebas comprueban que un contraste de la
matriz coincide bit a bit con el mismo contraste del informe que publicó la tabla, en
todas las métricas, en el conjunto y en cada mercado. Un brazo que aparezca en dos
informes debe tener las mismas sesiones y valores, y todos los informes deben compartir
mercados, edición y vistas.

| Vista | Fuente | Métricas |
| --- | --- | --- |
| `forecast` | `sessions.parquet` de la comparación, cuantiles en bruto | MAE, MSE, dirección, Rank IC, pinball, precisión por lado, Brier y ECE del signo, puntuación de intervalo |
| `forecast_calibrated` | La misma tabla con la calibración común | Pinball, Brier y ECE del signo, puntuación de intervalo |
| `portfolio` | `sessions.parquet` de la cartera | Los siete estadísticos de la cartera por coste y mercado |
| `policies` | Tabla por sesión de la etapa de políticas | Pendiente de declarar (#137) |
| `architecture_diagnostics` | Tabla por sesión de los diagnósticos | Pendiente de declarar: retención, regímenes, maduración y Jacobiano |

El ECE no es una media por sesión. Su contraste calcula en cada réplica el ECE de cada
semilla con los días remuestreados, promedia las semillas y combina los brazos con los
coeficientes del contraste. Usa los mismos días y réplicas que la fiabilidad del informe
walk-forward, de modo que el nivel de un brazo reproduce su intervalo. La cartera
remuestrea sesiones de cada mercado en orden, como su informe, y reutiliza sus contrastes.

Las vistas de políticas y diagnósticos son el punto de conexión de otras etapas. Leen una
tabla larga con `arm`, `seed`, `market`, `prediction_at`, `metric` y `value`, donde un
valor nulo es una sesión no definida. Cada métrica se declara en la matriz como pérdida
(no negativa, menor es mejor) o ganancia antes de ver la tabla. Una métrica sin declarar
se rechaza. Solo admiten medias por sesión. Un estadístico de recorrido, como el Sharpe
de una política, necesita la vía de la cartera.

### Fuentes y publicación con la campaña

`session_table_contrasts.write_sources` publica el manifiesto de fuentes de la matriz a
partir de los informes ya escritos, con rutas relativas a su carpeta y la huella de cada
archivo. Lo escribe primero como candidato y solo lo deja visible si `load_sources` lo
acepta, así que un informe alterado, incompleto o de otro ámbito no deja manifiesto. La
[matriz de A v2](../../configs/evaluation/comparison-matrix-a-v2.json) es la de A sobre la
comparación conjunta de A v2. Los siete modelos de integración que esa comparación todavía
no declara (`mars_titan_b6`, `mars_titan_b6_bias`, `mars_titan_m1_k4_first_read` y los
cuatro del régimen observable y su calendario, `mars_titan_b6_regime`,
`mars_titan_b6_calendar`, `mars_titan_b6_regime_banks` y `mars_titan_b6_calendar_banks`)
quedan condicionados a [#437](https://github.com/GonxKZ/mars-titan/issues/437) y sus
contrastes, pendientes. Las dos matrices compilan las mismas 47 familias y 362 contrastes.

Cada campaña declara su paso final antes de ver resultados
([A](../../configs/evaluation/historical-masked-publication-a.json) y
[A v2](../../configs/evaluation/historical-masked-publication-a-v2.json)): la campaña, su
matriz y la comparación de su etapa de adaptadores. `training/campaign_publication.py`
exige que la matriz lea la comparación de la campaña con la misma huella y que la
comparación postentrenada derive de una etapa de esa misma campaña, así que una matriz
declarada para otra campaña no llega a evaluarse. El paso final publica, por ámbito, las
fuentes de la campaña, la comparación walk-forward, la cartera si se declara, el
manifiesto de la matriz y la matriz, y después las fuentes y la comparación de cada padre
de la etapa. Con los agregados de la retención v2 no abre ninguna fila. Todo se escribe en
una carpeta provisional que solo toma el nombre del destino al terminar, con un recibo que
guarda la huella de cada archivo. Un fallo deja la carpeta provisional marcada y la
siguiente ejecución la descarta y repite.

### Coste por hora GPU

Si las fuentes incluyen un documento de horas por brazo, cada efecto lleva su versión por
hora. Las horas de un brazo se acumulan con las de sus padres, cada antecesor una vez,
porque un lector no existe sin su padre Titans-MAC. Con los mismos coeficientes del
contraste, $\Delta h=\sum_a w_a H_a$ y

$$
\text{mejora por hora}=\frac{s\,\hat\theta}{\Delta h},
$$

con $s=-1$ si menor es mejor y $s=+1$ si mayor es mejor. El intervalo simultáneo se
divide por la misma constante, porque las horas se tratan como medidas. Si $\Delta h\le 0$
la variante no cuesta más y no se calcula el cociente. Un nivel no tiene coste propio.
El documento declara si las horas son medidas o proyectadas, y el informe lo repite. Hoy
solo existe la proyección de la campaña A v2. Las horas medidas saldrán de los recibos de
la campaña, una conversión que todavía no está escrita.

### Brazos que faltan

`missing` cruza cada contraste con la clase de sus brazos. Con la declaración actual, 219
contrastes solo usan brazos de la campaña, 112 esperan brazos condicionados o derivados y
25 necesitan alguno de los ocho candidatos. Eran 183 y 148 antes de que la comparación
declarada incluyera el control en línea y los tres brazos de integración de MARS-TITAN.
Ningún contraste queda sin nombre. El
[informe de brazos que faltan](../../reports/engineering/component-attribution-20261010/README.md)
da el coste estimado de cada candidato con las horas proyectadas, lo que desbloquea por
sí solo, los lotes que solo sirven juntos y una prioridad calculada. Ningún candidato se
declara en el plan de la campaña desde aquí.

### Qué no permite afirmar

Un paso de la escalera mide el componente después de los anteriores, no su efecto en
general. Dejar uno fuera retira también lo que depende del componente. Un efecto
condicionado vale para su contexto. Shapley reparte una diferencia según una regla de
simetría, no identifica un mecanismo. Ninguna de estas cantidades es un efecto causal
económico. Todas son diferencias de error entre brazos ajustados con las mismas filas.
Con brazos pendientes, una familia se evalúa con los contrastes disponibles y su tamaño
cambia cuando llegan los demás. El informe lo deja escrito para que no se elija la
familia después de ver resultados.

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

La evaluación walk-forward se midió con una ventana anual sintética de 625.000
filas de evaluación (2.500 activos y 250 sesiones) y 155.000 de calibración, un
brazo puntual, uno con cuantiles calibrados y el control cero. Con dos hilos y
la CPU compartida (carga media cercana a 14), tres repeticiones tardaron 4,7,
2,8 y 2,8 s, incluidas lectura, huellas, validación, calibración, puntuación,
contrastes con 2.000 réplicas y el informe. El pico de memoria del proceso,
que también generó los datos, fue de 1,07 GiB. Con unos 1.250 pares de brazo,
semilla y ventana en el ámbito US, la extrapolación lineal ronda la media hora
en un proceso. Es una estimación, no una medida de la campaña real, y no
justifica por ahora otra implementación.

La versión 4 se midió después a la escala del ámbito US de la campaña A, con datos
sintéticos de las mismas formas: 19 ventanas, 12.826.460 filas de evaluación,
3.087.269 de calibración, 23 brazos y 64 series de brazo y semilla. La comparación
sin estratos tardó 1.678 s con un pico de 1,90 GiB y la cartera larga y corta 860 s
con 1,96 GiB, en un proceso con dos hilos y la CPU compartida (carga media de 14 a
18). El tiempo crece casi linealmente con las filas leídas, unos 2 µs por fila y
serie. El [informe de escala](../../reports/engineering/evaluation-scale-20261009/README.md)
recoge las cuatro medidas, sus condiciones y la extrapolación al diseño conjunto.

La matriz de comparaciones se midió con `benchmarks/comparison_matrix.py` sobre una tabla
por sesión sintética del ámbito US con las mismas sesiones, brazos y semillas que la
campaña A (19 ventanas, 597.625 filas de sesión y 65 series). Evaluar los 183 contrastes
estimables que tenía entonces, en 20 familias, con las vistas en bruto y calibrada y el
ECE del signo, tardó 152 s con un pico de 1,44 GiB, dos hilos y la CPU compartida (carga
media cercana a 23).
Alrededor del 60 % del tiempo se va en generar los índices del remuestreo por bloques, que
cada familia repite con la misma semilla. Reutilizarlos ahorraría uno o dos minutos por
evaluación, poco frente al resto de la evaluación, y no se ha hecho. El
[informe de brazos que faltan](../../reports/engineering/component-attribution-20261010/README.md)
recoge la medida y el perfil.

## Qué no demuestran estas métricas

Ninguna de estas cifras procede todavía de datos de mercado. El bloqueo de
aprendizaje sigue vigente y las pruebas usan valores calculados a mano y datos
sintéticos. Un intervalo bootstrap describe la variabilidad temporal de la
evaluación con la dependencia que capturan los bloques, no la incertidumbre de
la selección de configuraciones ni la de otro periodo. La dirección, el Rank IC y
la cobertura son diagnósticos secundarios. La métrica primaria sigue siendo el
MAE residual por sesión. La [decisión sobre la cabeza de cuantiles](quantile-head-decision.md)
adoptó una cabeza común para las arquitecturas neuronales, [implementada](../engineering/quantile-head.md)
con columnas que este panel acepta. Ridge y XGBoost siguen siendo escalares.
