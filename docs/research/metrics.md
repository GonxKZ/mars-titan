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
de evaluar.

Falta conectar los productores de predicciones. Las referencias neuronales con
retención `heldout_full_train_sessions_v1` ya escriben archivos por tramo con
`sample_id`, `asset_id`, mercado, instante, objetivo, predicción y, con
`quantile_head_v1`, sus cinco columnas. El entrenador cronológico de Titans-MAC
todavía no exporta predicciones por fila y la GRU episódica, los tabulares y
CM-v1 necesitan el mismo formato. El manifiesto de fuentes se generará a partir
de sus recibos cuando existan.

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
