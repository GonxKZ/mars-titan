# Seguimiento interno de salidas y límites de iTDS

[Pilzak, Pennington y Thivierge (2026)](https://doi.org/10.1038/s41467-026-70920-3) estudian una señal que sigue la salida de la red e interviene en su ajuste. Se revisaron las ecuaciones 1–7 y 12–22, los resultados iniciales y la disponibilidad de código. El modelo principal ajusta las conexiones de salida de un reservorio. El artículo también presenta otras arquitecturas. Sus resultados no son una evaluación de retornos financieros ni del núcleo Titans de este proyecto.

La inspección del [código de los autores, Figshare v1](https://doi.org/10.6084/m9.figshare.31146457.v1), distingue procedimientos. `reservoir_class1.m` combina descenso supervisado con descenso de la discrepancia respecto a una salida seguida mediante media exponencial. `reservoir1.m` reconstruye pesos con pseudoinversa y otra expresión de realimentación. `BPTT1.m` aprende un nodo sigmoide auxiliar. `cifar10.py` aprende una cabeza predictiva y añade el gradiente de una divergencia KL solo a la última capa del modelo principal. Por tanto, estos ejemplos no son cuatro implementaciones idénticas de una única recurrencia.

Se leyeron completos esos cuatro archivos y `README.txt`. La versión publicada contiene ejemplos parciales y declara licencia Apache 2.0. No se ejecutó código externo ni se reprodujeron sus experimentos. Las huellas y el alcance de lectura se registran en el [recibo de fuentes](../../reports/research/local-capture-20261020/itds-sources.json).

Hay que fijar signo, orden y escala temporal antes de trasladar una regla. El coeficiente multiplicativo de seguimiento del ejemplo de clasificación no se sustituye directamente por la constante de la ecuación diferencial del artículo. Tampoco una cabeza aprendida que imita logits equivale a promediar pesos. Esta revisión posterior no modifica el protocolo ni los modelos del contraste de captura local.

## Derivación propia para un caso escalar

Para estudiar la regla de descenso con seguimiento, se fija una característica escalar igual a uno, una etiqueta constante z y una salida y=w. Sean a>0 el paso, h>0 la intensidad supervisada, s≥0 la intensidad de seguimiento y 0<b≤1 la tasa de la media. Se actualizan ambas variables con sus valores anteriores:

```text
w_next = w − a·[h·(w−z) + s·(w−u)]
u_next = (1−b)·u + b·w
```

Al restar el equilibrio w=u=z, la matriz de transición es:

```text
A = [[1−a·(h+s), a·s],
     [b,           1−b]]
```

Con T=tr(A) y D=det(A), el polinomio característico es `l²−T·l+D`. La transformación `l=(1+r)/(1−r)` lo convierte, tras multiplicar por `(1−r)²`, en:

```text
(1+T+D)·r² + 2·(1−D)·r + (1−T+D)
```

Las raíces quedan dentro del disco unidad cuando los tres coeficientes son positivos. En el dominio declarado, `1−T+D=a·h·b>0` y `1−D=b+a·(h+s)−a·h·b>0`. La condición restante se reduce a:

```text
a·[h·(2−b)+2·s] < 4−2·b
```

Así, añadir seguimiento no estabiliza cualquier paso. Por ejemplo, con a=1, h=1 y b=1, el caso s=0 tiene radio espectral cero. Con s=1, la matriz es `[[-1,1],[1,0]]` y su radio es `(1+sqrt(5))/2`, mayor que uno. Es un contraejemplo a una garantía universal, no una reproducción de los ajustes del artículo.

Esta derivación requiere característica y etiqueta fijas y una actualización simultánea concreta. No prueba estabilidad de productos variables, capas no lineales, gradientes de todo el predictor o datos financieros. Tampoco caracteriza por sí sola los ejemplos de pseudoinversa, BPTT o KL.

Para valorar otro componente en MARS-TITAN habrá que fijar qué salida sigue, qué parámetros reciben su gradiente y qué estado conserva. Un seguimiento que use objetivos financieros solo puede incorporarlos cuando maduren. Un control con igual arquitectura y coeficiente cero, otro de regularización sencilla y residual cero ayudarían a separar su aportación del cambio de pérdida o del suavizado. No se ha integrado otro módulo ni medido una mejora financiera de iTDS en esta revisión.
