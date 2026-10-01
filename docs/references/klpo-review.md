# KLPO y su encaje en las referencias predictivas

KLPO optimiza políticas a partir de recompensas terminales y probabilidades
registradas durante la generación. La versión revisada es la del
[commit 30c0ae8c](https://github.com/yifanzhang-pro/KLPO/tree/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696),
del 21 de septiembre de 2026. El informe se publicó el 18 de septiembre y se
revisó el día 20. El proyecto incorpora una implementación matemática independiente
para una política predictiva de una decisión. No se ha instalado el paquete
original ni se han reproducido sus experimentos con modelos de lenguaje.

## Mecanismo y evidencia

Si p es la política actual, q el muestreador histórico y R la recompensa
terminal, la ruta predeterminada usa regresión por tokens y una corrección
Monte Carlo. En cada prefijo se obtienen sorteos auxiliares independientes
desde q. El gradiente se construye a partir de

$$
\ell_u=\log p_\theta(a_u\mid s_u)-\log q(a_u\mid s_u),
$$

$$
L_{\mathrm{bp}}=-\frac1B\sum_i\sum_u
\operatorname{sg}(R_i-\beta\ell_{i,u})
\left[\log p_\theta(a_{i,u}\mid s_{i,u})-
\frac1M\sum_j\log p_\theta(v_{i,u,j}\mid s_{i,u})\right].
$$

`sg` impide derivar el coeficiente. La equivalencia en esperanza exige las
condiciones de muestreo del informe. No garantiza mejora en cada actualización,
convergencia de una red ni generalización financiera. La ruta por tokens admite
M igual o mayor que uno. La alternativa por secuencias requiere al menos dos
sorteos y una corrección adicional. Las ecuaciones y sus condiciones están en
el [informe técnico](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/KLPO.pdf)
y la [referencia del algoritmo](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/docs/algorithms.md).

El [README](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/README.md#validation-and-scope)
declara pruebas CPU y señala que el entrenamiento GPU y las pruebas de rendimiento a
escala del informe no están validados. Estas pruebas del proyecto original no se han
ejecutado localmente. La
[guía del backend](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/docs/training.md)
exige actualmente recogida síncrona y una actualización por lote. Las pruebas
locales de la adaptación predictiva no validan ese sistema de entrenamiento.

## Diferencias respecto a la tarea del proyecto

Ridge, boosting, RNN, LSTM, GRU y DLinear producen un número. El
[adaptador residual](../research/predictive-adaptation.md) añade una corrección
lineal y convierte el centro resultante en una distribución discreta de 21
acciones. El padre permanece congelado. Este ajuste no actualiza los pesos
internos del predictor ni se presenta como una nueva arquitectura neuronal.

La implementación también
[valida logaritmos de probabilidades discretas](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/klpo/_validation.py).
Una densidad continua puede superar uno, por lo que no se sustituye sin más
por una cabeza gaussiana. El problema supervisado ya ofrece la etiqueta de
retorno. Para un conjunto finito de predicciones posibles se pueden calcular
directamente sus errores, sin añadir necesariamente la varianza de muestreo
de un estimador de aprendizaje por refuerzo. Esta es una consideración de
diseño del proyecto, no un resultado experimental de KLPO.

## Adaptación de una decisión

Para cada fila, las acciones son retornos de una rejilla fijada con entrenamiento.
La recompensa es $R_a=-|a-y|/s$, donde $s$ se estima solo con entrenamiento.
El muestreador histórico se define como

$$
q_a=(1-\varepsilon)\,\pi_{\mathrm{padre}}(a)+\varepsilon/21,
\qquad \varepsilon=10^{-6}.
$$

La mezcla evita perder soporte por subdesbordamiento. Se calcula con
`logaddexp` y se usa la misma distribución en las acciones, los auxiliares y
la corrección exacta. La política optimizada $p$ conserva la gaussiana discreta
original. Por ello, su centro inicial coincide con el padre, pero $p_0$ no es
exactamente igual a $q$. La exploración uniforme no es una probabilidad mínima
impuesta retrospectivamente a los registros.

Con $h_a=R_a-\beta\log(p_a/q_a)$ y $q$ fija, la ecuación 3.13 se reduce a

$$
F(\theta)=\frac{1}{2\beta}\sum_a q_a
\left(h_a-\sum_bq_bh_b\right)^2.
$$

Su gradiente coincide con la esperanza del sustituto token Full-KL. La variante
MC estima la corrección con 128 sorteos independientes con reemplazo. Se renuevan
las acciones y los auxiliares en cada visita, con generadores separados. Los
duplicados se conservan. El control `klpo_exact` calcula directamente $F$, sin
muestrear acciones ni detener su gradiente. Full-KL y MC sí detienen el coeficiente
$h$ en su sustituto de retropropagación.

Con 21 acciones, la corrección completa evita el muestreo auxiliar. MC se
mantiene para contrastar el estimador del artículo, no como una optimización
presupuesta. El valor de su pérdida sustituta puede ser negativo y no equivale
al de $F$. La igualdad de gradientes en esperanza tampoco implica igualdad de
actualizaciones después de AdamW y del recorte de norma.

La gaussiana discreta con anchura fija tiene un logaritmo afín en el centro salvo
un término común a las acciones. Al centrar $h$, ese término desaparece. En esta
familia, $F$ es cuadrática convexa respecto al centro y a los parámetros del
adaptador lineal. Esta deducción permite un control numérico sencillo. No demuestra
convexidad para una red completa ni que el adaptador pueda representar el óptimo
sin restricciones $p^*\propto q\exp(R/\beta)$.

## Comparación y límites

El [diseño de ejecución](../engineering/klpo-posttraining.md) conserva los controles
REINFORCE, pérdida esperada y MAE. Usa la misma población admitida, normalización,
capacidad y presupuesto. Las etiquetas maduras de entrenamiento permiten conocer
la recompensa de todas las acciones. La validación selecciona el checkpoint,
pero sus etiquetas no se usan para actualizar parámetros. El test permanece cerrado.

Esta es una adaptación predictiva offline. No reproduce negociación secuencial,
costes de cartera ni las evaluaciones agentic del informe. Los
[48 ensayos históricos](../../reports/baselines/reference-variants.md) pertenecen
a otra campaña y no son resultados de KLPO. Las pruebas sintéticas comprueban
álgebra, aprendizaje de una señal definida y recuperación. No se incorporan al
corpus financiero ni demuestran rentabilidad o mejora predictiva real.

La revisión del PDF cubrió las páginas 1 a 11, 16 a 19 y 33 a 36, no una lectura
integral. La copia local tiene 899.179 bytes y SHA-256
`b4029a0eb42d7a05fbd9e4297f022a16d5d286dcd5cc4e0a5e4b1ce9d0db5c51`.
El código declara Apache 2.0. El PDF se conserva solo como copia local de
consulta y no se publica en el repositorio.
