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
de velocidad entre algoritmos. La comprobación CUDA queda pendiente de una
ventana coordinada y deberá completarse antes de integrar la ruta.
