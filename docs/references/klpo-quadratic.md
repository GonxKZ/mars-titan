# Términos cuadráticos del KLPO predictivo

[`quadratic_terms`](../../src/mars_titan/models/klpo_quadratic.py) devuelve tres
coeficientes FP64 por fila, `constant`, `linear` y `curvature`, para el centro
predictivo c en sus unidades originales:

\[
F(c)=C+Lc+\tfrac12 Kc^2.
\]

Es una descomposición del objetivo `klpo_exact` de la familia gaussiana discreta
con anchura fija. No resuelve un sistema de regresión, no actualiza parámetros
y no ejecuta el entrenador. Su contrato es `klpo_exact_center_quadratic_v1`.
El módulo anterior `models/klpo.py` y sus rutas permanecen separados.

La función recibe `logq` y recompensas con forma `[B,A]`, una rejilla creciente
de A acciones, `scale > 0` y `beta > 0`. Admite FP32 y FP64 en el mismo
dispositivo CPU o CUDA. Reutiliza la validación de KLPO para la normalización y
el soporte positivo de q. Los datos, las recompensas y la rejilla se desacoplan
del grafo. El cálculo devuelve coeficientes FP64, sin muestreo ni cambios de RNG.

Se admiten hasta 4.096 filas y acciones por eje y 262.144 elementos en su
producto. El límite se comprueba antes de promover los datos. Permite dividir
la población en lotes, sin recortarla. Acota el cálculo de momentos y sus
temporales, pero no representa un límite del RSS del proceso. Los valores o
resultados no finitos y las disposiciones dispersas se rechazan.

Para obtener los coeficientes, sean z=a/s y
b(a)=R(a)+β log q(a)+β z²/2. La normalización de la gaussiana es común a todas
las acciones y desaparece al centrar bajo q. Por tanto,

\[
C=\frac{\operatorname{Var}_q(b)}{2\beta},\qquad
L=-\frac{\operatorname{Cov}_q(z,b)}{s},\qquad
K=\frac{\beta\operatorname{Var}_q(z)}{s^2}.
\]

Esta identidad corresponde a aritmética real con q, escala y rejilla fijas.
La evaluación del polinomio puede sufrir cancelación. La pérdida centrada de
`token_loss` sigue siendo la referencia numérica del objetivo. Los coeficientes
no garantizan rango completo del sistema asociado a un adaptador lineal ni
convexidad de un backbone entrenable. Resolver y aplicar los coeficientes
constituiría un ajuste distinto de estas comprobaciones algebraicas.

La Hessiana respecto al centro es `beta * Var_q(a) / scale**4`. Se contrasta
con la varianza de `token_loss(..., mode="klpo_exact")`, cuyo h conserva el
grafo. No es una comprobación de segundas derivadas de los sustitutos Full-KL
o MC, que desacoplan h. La igualdad del gradiente en esperanza no implica
igualdad de las derivadas superiores de esos grafos.

La referencia conserva el
[commit 30c0ae8c de KLPO](https://github.com/yifanzhang-pro/KLPO/tree/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696)
documentado en la [revisión original](klpo-review.md). La publicación posterior
[arXiv 2610.08963v1](https://arxiv.org/abs/2610.08963v1) y el
[commit 304e5ac7](https://github.com/yifanzhang-pro/KLPO/tree/304e5ac7ca573d45a0e42e6eaea95203260402bc)
tienen identidad separada. Sus defaults de beta y número de auxiliares no se
trasladan a las configuraciones existentes. En particular, este objetivo exige
beta positiva y no implementa el límite beta cero de los sustitutos nuevos.

Las pruebas de [la descomposición](../../tests/models/test_klpo_quadratic.py)
contrastan una fórmula escalar conocida, la pérdida enumerada, gradientes,
Hessianos y diferencias finitas. También comprueban tipos, soporte, presupuesto,
filas, invariancia a recompensas constantes y ausencia de cambios en los datos.
No contienen optimizadores ni ajuste de señales. La comparación científica de
[#128](https://github.com/GonxKZ/mars-titan/issues/128) sigue pendiente de la
edición histórica desde 2000 preparada y verificada. La reserva final permanece
cerrada.

En la comprobación inicial pasaron 22 casos nuevos y 15 pruebas algebraicas de
la ruta anterior. Se excluyó el caso antiguo que ejecuta un ajuste. Cinco
mutaciones dirigidas de los coeficientes, el gradiente de la rejilla y el límite
de elementos fueron detectadas. La cobertura de sentencias del módulo es
32/33 con coverage.py 7.16.2. Radon 6.0.1 da CCN 21 para la función y CRAP 21,04,
calculado con su cobertura de sentencias. Son diagnósticos de estas pruebas.

Una medición CPU de cinco repeticiones tras un calentamiento obtuvo medianas
de 0,153, 0,296 y 2,582 ms para 1, 256 y 4.096 filas de 21 acciones. El caso
numérico de 4.096×64 elementos tardó 9,946 ms. No cambia la rejilla del modelo.
Se usaron dos hilos y PyTorch 2.14.0. El pico de RSS del proceso fue 538.100 KiB,
incluidas importación y entradas. No es memoria incremental ni una comparación
de velocidad entre algoritmos.

La comprobación CUDA posterior usó dos fixtures `[3,21]`, uno FP32 y otro FP64,
en una RTX 4070 Laptop con PyTorch 2.14.0+cu130. Se compararon coeficientes,
valores, gradientes y Hessianas cuadráticas con CPU, y valor y gradiente con
la varianza exacta en CUDA. El mayor error absoluto de los coeficientes y
Hessianas fue 1,14×10⁻¹³. Cinco combinaciones incompatibles de tipo o dispositivo
se rechazaron. El pico Torch fue 13.312 bytes asignados y 2 MiB reservados, con
límite de allocator de 64 MiB. Esos contadores excluyen el contexto CUDA.
Las entradas y los RNG quedaron iguales. No se midió rendimiento CUDA.

## El objetivo y el error predictivo no son equivalentes

Un caso algebraico de tres acciones `[-1,0,1]` permite comprobar ese límite.
Sea q la gaussiana discreta simétrica de centro 0 y anchura 1, mezclada con
masa uniforme de 10⁻⁶. Para `y=0.2`, `R(a)=-abs(a-y)` y `beta=0.1`,
`Cov_q(R,a)/Var_q(a)=0.2`. Los términos pares de b no aportan covarianza con a,
por lo que el centro estacionario es 2. Las funciones reales `quadratic_terms`
y `token_loss` confirman `F(2)<F(0)`, mientras el error absoluto del centro
pasa de 0,2 a 1,8. Solo se evalúan esas constantes. No se resuelve ni aplica
un ajuste.

Ese ejemplo usa tres acciones, no cambia la rejilla de 21 del adaptador. La
[salida del adaptador](../../src/mars_titan/models/predictive_adaptation.py)
es el centro continuo, pero la
[evaluación vigente](../../src/mars_titan/posttraining/evaluation.py)
publica la mediana discreta como `prediction`. Otra prueba con 21 acciones
equiespaciadas en `[-1,1]`, anchura 1 y centros fijos 0 y 2 obtiene medianas
0 y 0,6. El objetivo exacto también baja y los errores de las medianas frente
a 0,2 pasan de 0,2 a 0,4. Esa segunda comprobación no afirma que 2 sea el centro
estacionario para la rejilla de 21 acciones.

El ejemplo no refuta el objetivo del artículo. La familia gaussiana con
anchura fija no representa libremente la distribución de Gibbs. Minimizar
esta varianza dentro de esa familia no garantiza mejorar el MAE del centro
ni el de la salida discreta. No se extrapola el caso a datos financieros.
Las dos regresiones añadidas pasan junto a las 37 pruebas anteriores, sin
ejecutar el caso antiguo que ajusta parámetros.

Las huellas, perfiles y límites de las comprobaciones quedan en el
[recibo técnico](../../reports/engineering/rl-objectives-verification-20261008.json).
La revisión independiente contrastó además los coeficientes con aritmética
Decimal de 70 dígitos y comprobó la diferencia entre las derivadas superiores
del objetivo exacto y las de sus sustitutos desacoplados.
