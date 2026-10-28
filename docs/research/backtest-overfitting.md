# Sobreajuste del backtest y Sharpe deflactado

Estado: declarado el 29 de octubre de 2026 antes de ejecutar la cartera con predicciones de
la campaña. Está implementado en `evaluation/backtest_overfitting.py` y comprobado con el
ejemplo publicado del DSR, con un cálculo de CSCV por fuerza bruta y con casos de respuesta
conocida. No se ha ejecutado sobre ningún informe real. El bloqueo de aprendizaje sigue
vigente y la reserva de 2024 sigue cerrada.

## Qué elección mide

La comparación elige modelos con el MAE residual de validación, y la
[declaración del 13 de octubre](../../configs/evaluation/historical-masked-2000-selection-effect.json)
ya explica por qué CSCV y el DSR no describen esa elección. Este análisis mide otra: la
afirmación de que un modelo tiene la mejor [cartera larga y corta](long-short-portfolio.md)
de un mercado y un coste, hecha después de ver el Sharpe de todos sobre las mismas sesiones.
Si el informe final hace esa afirmación, el PBO dice con qué frecuencia el modelo elegido en
una mitad de las sesiones queda por debajo de la mediana en la otra mitad, y el DSR dice qué
parte de su Sharpe resiste al número de modelos que se han mirado.

Todas las decisiones están en la
[declaración](../../configs/evaluation/backtest-overfitting-v1.json), fijada antes de
ejecutar. Como advierte el artículo del PBO, CSCV sirve para evaluar un procedimiento de
selección y no debe convertirse en el objetivo de una búsqueda. Aquí no elige nada.

## Fuentes y fórmulas

Las fórmulas son las de Bailey y López de Prado (2014), *The Deflated Sharpe Ratio*
(`bailey2014dsr`), y Bailey, Borwein, López de Prado y Zhu (2017), *The Probability of
Backtest Overfitting* (`bailey2017pbo`), sin cambios. Las entradas están en la
[bibliografía financiera](../references/finance.bib).

El Sharpe de cada serie es por sesión y sin anualizar, $\widehat{SR}=\bar r/s$ con la
desviación típica de ddof = 1. La asimetría $\hat\gamma_3$ y la curtosis $\hat\gamma_4$ son
los momentos de la muestra sin corrección y sin restar 3, como en el código de López de
Prado. El Sharpe probabilístico frente a un umbral $SR^*$ con $T$ sesiones es

$$
\widehat{PSR}(SR^*)=Z\!\left[\frac{(\widehat{SR}-SR^*)\sqrt{T-1}}
{\sqrt{1-\hat\gamma_3\,\widehat{SR}+\frac{\hat\gamma_4-1}{4}\,\widehat{SR}^2}}\right].
$$

El umbral del DSR es el máximo esperado de $N$ ensayos con Sharpe verdadero nulo y varianza
$V$ entre ellos, con $\gamma$ la constante de Euler y Mascheroni:

$$
SR_0=\sqrt{V}\left((1-\gamma)\,Z^{-1}\!\left[1-\tfrac1N\right]+\gamma\,Z^{-1}\!\left[1-\tfrac1{Ne}\right]\right),
\qquad DSR=\widehat{PSR}(SR_0).
$$

CSCV parte las $T$ sesiones en $S$ bloques contiguos iguales. En cada una de las
$\binom{S}{S/2}$ mitades se elige el modelo $n^*$ de mayor Sharpe dentro, se calcula el
rango $\bar r_{n^*}$ de su Sharpe fuera entre los $N$ modelos y se toman
$\bar\omega_c=\bar r_{n^*}/(N+1)$ y $\lambda_c=\ln\bigl(\bar\omega_c/(1-\bar\omega_c)\bigr)$. El
PBO es la fracción de mitades con $\lambda_c\le 0$, que estima la integral
$\int_{-\infty}^{0}f(\lambda)\,d\lambda$ del artículo e incluye por tanto el empate con la
mediana. El informe publica además la regresión del Sharpe fuera del elegido sobre su Sharpe
dentro (la degradación), la probabilidad de que el elegido pierda fuera, los cuantiles de
$\lambda$ y cuántas veces se elige cada modelo.

El Sharpe de cada mitad se obtiene de las medias y los momentos centrados de cada bloque,
sin volver a recorrer las filas. La dispersión de las medias se escribe como
$\sum_b m_b^2-k\mu^2$ sobre medias ya centradas en la de toda la serie, así que cada lote de
mitades se reduce a tres productos de matrices. Las pruebas comparan este cálculo con el que
recorre las filas de cada mitad con una tolerancia relativa de $10^{-12}$.

## Decisiones declaradas

| Decisión | Valor | Motivo |
|---|---|---|
| Fuente | Un informe `long_short_comparison` terminado, sin abrir la reserva, cuyas sesiones coinciden con la huella del informe | Lo que no corresponde al informe no se analiza |
| Estrategias | Los modelos de la comparación, con las semillas promediadas sesión a sesión | El protocolo no trata las semillas como mercados independientes |
| Mercados y costes | Cada mercado y cada coste declarado por separado | Estados Unidos y China nunca se mezclan en una cartera |
| Bloques | $S=16$, con 12.870 mitades | El artículo propone 16 por defecto y estima un error típico de la fracción menor que 0,0045. El texto escribe 12.780 mitades, pero $\binom{16}{8}=12.870$ |
| Recorte | Se descartan las $T \bmod 16$ sesiones más antiguas | Los bloques tienen que ser iguales y nada se rellena. El informe dice cuántas |
| Empates | El primer máximo dentro, en el orden alfabético de los modelos, y el rango medio fuera | El resultado depende solo de los datos |
| Carteras constantes | Un modelo cuya cartera no varía en las sesiones analizadas queda fuera del PBO y del DSR y el informe lo nombra | El control cero no abre posiciones y su Sharpe no existe. Si un modelo incluido no varía en alguna mitad, el análisis se detiene en lugar de dar un Sharpe por convenio |
| Ensayos del DSR | Los modelos de la comparación con cartera variable y, como sensibilidad, 28 y 100 | 28 son las entradas del catálogo de la [criba](../../configs/experiments/variant-screening-v1.json) y 100 el ejemplo del artículo del DSR. El catálogo de la segunda ola se añadirá con otra versión de la declaración antes de ejecutarla |
| Varianza $V$ | La varianza de los Sharpe de los modelos de la comparación, con ddof = 1 | Es la única dispersión observada. Se usa igual en toda la sensibilidad |

## Límites

- Las mitades de CSCV mezclan bloques de fechas distintas. No respetan el orden temporal del
  walk-forward ni separan cambios de régimen, así que el análisis complementa el contraste
  por bloques de la comparación y no lo sustituye.
- Los modelos comparten filas, padre y datos, así que no son ensayos independientes. El
  número de ensayos se informa con su sensibilidad en lugar de estimar un número efectivo
  por agrupación.
- Con $N$ modelos el rango relativo solo toma $N$ valores. Con pocos modelos el PBO es una
  estimación gruesa, y el artículo pide $N$ mucho mayor que 10 para distinguir valores por
  debajo de 1/10.
- La varianza de la sensibilidad sale de los modelos de la comparación y no de las variantes
  de la criba, cuyas carteras no se calculan.
- Un PBO bajo no prueba rentabilidad futura y un PBO alto no impide que todos los modelos
  tengan habilidad parecida. Ninguno de los dos corrige fugas ni costes mal calculados.
- No se calcula el cuarto análisis del artículo, la dominancia estocástica, porque la
  declaración no lo usa.

## Comprobación técnica

- **DSR publicado.** Con el ejemplo de Bailey y López de Prado (2014), un Sharpe anual de 2,5
  sobre 1.250 sesiones de 250 por año, una varianza anual de 1/2 entre 100 ensayos, asimetría
  −3 y curtosis 10, el módulo da $SR_0=0{,}1132$ por sesión y $DSR=0{,}9004$, las cifras del
  artículo. Con 46 ensayos da 0,9505 y, con rendimientos normales, 0,95 con 88 ensayos, las
  otras dos cifras del ejemplo.
- **PBO publicado.** Los ejemplos del artículo del PBO, con un 55 % y un 13 % en la
  estrategia estacional, salen de sorteos sin semilla publicada y no se pueden reproducir
  cifra a cifra. La CSCV se comprueba contra un cálculo por fuerza bruta de la definición,
  con 48 sesiones, cinco modelos (dos idénticos para forzar empates) y 70 mitades: coinciden
  el elegido, los Sharpe dentro y fuera, los logits, el PBO, la probabilidad de pérdida y la
  regresión.
- **Casos de respuesta conocida.** Un modelo que gana en todos los bloques da PBO 0 sin
  pérdidas, una elección que se invierte fuera da PBO 1 y los modelos sin habilidad dan de
  media entre 0,4 y 0,65 en doce muestras de ruido. Una sola muestra de ruido varía mucho.
- **Fuente.** Se rechazan un informe sin terminar, uno que abre la reserva, sesiones que no
  coinciden con su huella, mercados con modelos distintos, semillas a las que les falta una
  sesión, declaraciones que no describen este análisis o con un número de ensayos sin
  procedencia y una salida existente o dentro de la fuente. Un control sin semilla que no
  abre posiciones queda fuera con su nombre, y el análisis se detiene si solo queda un modelo
  con cartera variable. El orden de las filas del archivo no cambia el resultado.

Las mutaciones dirigidas, la cobertura y el coste se publican con su recibo cuando terminan.
