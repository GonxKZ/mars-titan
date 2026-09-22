# KLPO y su encaje en las referencias predictivas

KLPO optimiza políticas a partir de recompensas terminales y probabilidades
registradas durante la generación. La versión revisada es la del
[commit 30c0ae8c](https://github.com/yifanzhang-pro/KLPO/tree/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696),
del 21 de septiembre de 2026. El informe se publicó el 18 de septiembre y se
revisó el día 20. No se ha instalado ni ejecutado KLPO en el proyecto.

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
declara pruebas CPU y señala que el entrenamiento GPU y los benchmarks a
escala del informe no están validados. Estas pruebas upstream no se han
ejecutado localmente. La
[guía del backend](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/docs/training.md)
exige actualmente recogida síncrona y una actualización por lote. El coste
en la RTX 4070 de 8 GB no está medido.

## Diferencias respecto a la tarea del proyecto

Ridge, boosting, GRU y DLinear producen un número. No generan una política de
acciones ni las probabilidades del muestreador que necesita KLPO. Su uso exigiría
otra cabeza o un adaptador identificado como variante distinta. No basta con
cambiar el nombre de una función de pérdida.

La implementación también
[valida logaritmos de probabilidades discretas](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/klpo/_validation.py).
Una densidad continua puede superar uno, por lo que no se sustituye sin más
por una cabeza gaussiana. El problema supervisado ya ofrece la etiqueta de
retorno. Para un conjunto finito de predicciones posibles se pueden calcular
directamente sus errores, sin añadir necesariamente la varianza de muestreo
de un estimador de aprendizaje por refuerzo. Esta es una consideración de
diseño del proyecto, no un resultado experimental de KLPO.

Se ha priorizado comparar MSE, L1 y Huber y un ajuste posterior con control de
pasos. Los [48 ensayos ejecutados](../../reports/baselines/reference-variants.md)
no muestran una mejora consistente frente al retorno cero. Su resultado no
permite afirmar que KLPO vaya a funcionar mejor o peor, porque no se ha probado.

Si se activa posteriormente, necesitará una política explícita, recompensa
predefinida, registros del muestreador y etiquetas maduras. El ajuste se hará
solo con entrenamiento. Se conservarán la referencia original y un control
supervisado con el mismo presupuesto. No se usarán validación o test como
recompensas para mejorar retrospectivamente la predicción evaluada.

La revisión del PDF cubrió las páginas 1 a 11, 16 a 19 y 33 a 36, no una lectura
integral. La copia local tiene 899.179 bytes y SHA-256
`b4029a0eb42d7a05fbd9e4297f022a16d5d286dcd5cc4e0a5e4b1ce9d0db5c51`.
El código declara Apache 2.0. El PDF se conserva solo como copia local de
consulta y no se publica en el repositorio.
