# Condiciones matemáticas de memoria, contexto y actualización

Revisión del 4 de octubre de 2026. Este documento desarrolla condiciones y contraejemplos para el [diseño ampliado](neuroarchitecture-review.md). No implementa el candidato ni demuestra utilidad financiera. Las derivaciones elementales son comprobaciones del diseño, sin reivindicar teoremas nuevos.

## Qué información puede utilizar una decisión

Sea $\mathcal F_t$ la información permitida al emitir la predicción. Incluye observaciones publicadas, transformaciones ajustadas con pasado, procedencia del preentrenamiento, parámetros y estado inicial. La causalidad temporal exige que $\hat y_t$ sea medible respecto a $\mathcal F_t$. No basta con que el archivo de precios termine en t si el codificador o los fundamentales contienen información posterior.

Si el estado inicial cumple esa condición y cada transición recibe solo información permitida, la propiedad se conserva por inducción. Las etiquetas pendientes permanecen fuera de la memoria recuperable. Todos los activos de una sesión leen una instantánea común. Una etiqueta publicada durante el procesamiento solo afecta a la siguiente instantánea, según el contrato declarado. Esto no demuestra que las fechas reales del corpus sean correctas.

Repetir K veces una transformación determinista de la misma entrada no añade observaciones externas. Puede mejorar la aproximación realizada por la red. No permite deducir que más pasos internos reducen necesariamente el error irreducible.

## Regla delta con olvido escalar

La referencia asociativa del [candidato](candidate-architecture.md) propone, con $A\in\mathbb R^{d\times m}$, clave $k\in\mathbb R^d$ y valor $v\in\mathbb R^m$,

$$
A^+=(1-\lambda)A+\eta k(v-A^\top k)^\top.
$$

Para dos estados sometidos a las mismas entradas y coeficientes, su diferencia D cumple

$$
D^+=\big[(1-\lambda)I-\eta kk^\top\big]D.
$$

La matriz tiene autovalor $1-\lambda-\eta\|k\|^2$ en la dirección de k y $1-\lambda$ en las direcciones ortogonales. Para d ≥ 2, su norma espectral es

$$
q=\max\{|1-\lambda|,\ |1-\lambda-\eta\|k\|^2|\}.
$$

Para d = 1, esa expresión sigue siendo una cota, aunque el espacio ortogonal no existe. Con $0\leq\lambda\leq1$, una condición suficiente de no expansión es

$$
0\leq\eta\|k\|^2\leq2-\lambda.
$$

El extremo superior puede producir un autovalor −1 y oscilaciones. Para garantizar contracción uniforme en cada actualización basta imponer $q_t\leq\bar q<1$. La condición $\eta<2$ de una regla sin olvido no se puede trasladar sin cambios: con clave unitaria, λ = 0,1 y η = 1,95 se obtiene q = 1,05.

Esta deducción corresponde a la fórmula escrita arriba. Otras reglas aplican el olvido antes de calcular el residual y tienen otro operador. Tampoco cubre puertas que cambian al cambiar A, porque entonces se comparan transiciones distintas. Una cota sobre diferencias no basta para acotar el estado absoluto. Si además $\|\eta_t k_t v_t^\top\|_F\leq M$,

$$
\|A_t\|_F\leq\bar q^t\|A_0\|_F+
M\frac{1-\bar q^t}{1-\bar q}.
$$

## Puertas por canal: acotarlas no basta

[Gated DeltaNet-2, v1, §3.1–3.2](https://arxiv.org/html/2605.22791v1) separa borrado y escritura. En notación clave por valor, una parte del operador de diferencias es $B=I-k(b\odot k)^\top$. Para b variable por canal no es necesariamente simétrico.

Un contraejemplo interior al intervalo de una sigmoide usa

$$
k=\frac1{\sqrt2}(1,1)^\top,\quad b=(0.9,0.1)^\top,\quad
T=0.95B=
\begin{pmatrix}0.5225&-0.0475\\-0.4275&0.9025\end{pmatrix}.
$$

Su radio espectral es 0,95 y su norma espectral es aproximadamente 1,04155. Para $x=(1,-2)^\top$,

$$
\frac{\|Tx\|_2^2}{\|x\|_2^2}=1.0730725>1.
$$

Por tanto, puertas positivas menores que uno y retención escalar menor que uno no garantizan no expansión euclídea en cada paso. El ejemplo no demuestra divergencia: al repetir esta matriz fija, $T^n\to0$. Tampoco refuta los resultados experimentales del artículo. Obliga a separar estabilidad espectral, amplificación transitoria y productos de operadores variables antes de atribuir una garantía a una adaptación propia.

## Actualización proximal de una cohorte

Como variante separada, se puede estudiar una escritura conjunta de las etiquetas maduras de una cohorte. Se fijan pesos $w_j\geq0$, $\sum_jw_j=1$, claves $\|k_j\|\leq1$ y valores disponibles. K tiene filas $\sqrt{w_j}k_j^\top$ y V filas $\sqrt{w_j}v_j^\top$. Se define

$$
A^+=\arg\min_Z\left\{
\frac12\|Z-\rho A\|_F^2+\frac\eta2\|KZ-V\|_F^2\right\},
\qquad0\leq\rho\leq1,\quad\eta\geq0.
$$

Es un problema cuadrático proximal conocido, no un algoritmo reivindicado como nuevo. [Parikh y Boyd](https://web.stanford.edu/~boyd/papers/prox_algs.html), §1.1–1.2 y §2.3 de la copia de los autores, presentan el operador proximal y sus propiedades. Aquí la aplicación a una cohorte concreta se deriva explícitamente.

Con $G=K^\top K$ y $H=K^\top V$, anular el gradiente da

$$
(I+\eta G)A^+=\rho A+\eta H.
$$

La matriz del sistema es definida positiva para todo η finito no negativo, incluso con claves de rango deficiente. Hay una solución única. Se resolvería mediante una biblioteca BLAS/LAPACK y factorización de Cholesky, sin formar la inversa explícita.

Para dos estados con los mismos K, V, ρ y η,

$$
\|D^+\|_F\leq
\frac{\rho}{1+\eta\lambda_{\min}(G)}\|D\|_F
\leq\rho\|D\|_F.
$$

Si ρ = 1 y G tiene rango deficiente no se obtiene contracción estricta en todas las direcciones. Como $\operatorname{tr}G=\sum_jw_j\|k_j\|^2\leq1$,

$$
\kappa_2(I+\eta G)\leq1+\eta.
$$

Es una cota de condición, no permiso para usar η arbitrariamente grande en precisión reducida. Si también $\|v_j\|\leq M$, $\eta_t\leq\eta_{max}$ y $\rho_t\leq\bar\rho<1$, entonces $\|H_t\|_F\leq M$ y

$$
\|A_t\|_F\leq\bar\rho^t\|A_0\|_F+
\eta_{max}M\frac{1-\bar\rho^t}{1-\bar\rho}.
$$

Las condiciones deben volver a analizarse si claves, pesos o puertas dependen del estado comparado. No son una prueba de estabilidad del codificador, del entrenamiento completo ni del precio de mercado.

La suma de Gram y términos cruzados es invariante a la permutación de activos en aritmética exacta. En coma flotante el orden de reducción puede variar. Para dividir una cohorte en microlotes se conservan los pesos globales, se acumulan G y H y se resuelve una sola vez desde la instantánea anterior. Resolver cada microlote, renormalizar sus pesos o aplicar ρ repetidamente cambia el algoritmo. Con cohorte vacía se propone conservar A sin cambios. Aplicar olvido solo por el paso del reloj sería otra variante.

Tampoco es una sustitución numéricamente idéntica de la regla explícita. Para η pequeño,

$$
A^+=\rho A+\eta(H-\rho GA)+O(\eta^2),
$$

mientras que la regla explícita análoga resta $\eta GA$. Coinciden en ese término solo cuando ρ = 1. Su comparación necesita un nuevo identificador experimental.

Para b filas, d dimensiones de clave y m de valor, el coste denso es del orden de $bd^2+bdm+d^3+d^2m$. Hay que contar la matriz anterior, G, H, lado derecho, factor, solución y espacio de trabajo. La escritura rank-one es más barata por evento. Se conservaría la alternativa proximal solo si mejora estabilidad o error con un coste total aceptable. El [SIFt-RLS de Lai y Bernstein](https://arxiv.org/abs/2404.10844v1), §4, aporta otro control de olvido direccional. Sus condiciones sobre regresores y estimación de parámetros fijos sin ruido no se trasladan a una red financiera completa.

## Sorpresa, ruido e información contextual

Un error grande no identifica por sí solo un cambio aprendible. En el modelo escalar de Kalman con varianza previa P, ruido de proceso Q y ruido de observación R, la innovación tiene varianza $S=P+Q+R$ y la ganancia es $K_g=(P+Q)/S$. Con P = 1, los pares (Q,R) = (3,1) y (1,3) tienen S = 5, pero ganancias 0,8 y 0,4. La misma escala de innovación admite actualizaciones distintas.

Este cálculo ilustra la separación estudiada en humanos por [Piray y Daw, 2024](https://www.nature.com/articles/s41467-024-53459-z). No identifica Q y R a partir de retornos ni demuestra que la volatilidad financiera equivalga a cambio de una dinámica latente. La selección por error debe contrastarse con referencias de ruido, cambio y escritura uniforme.

[Zhou y colaboradores, v1, §2.2–2.4](https://arxiv.org/html/2608.25128v1) proponen un diagnóstico de contexto basado en información mutua y residualización. Dos pasos requieren cautela: no rechazar una hipótesis nula no demuestra independencia, e información del residual lineal no equivale en general a información condicional.

Sean X y C independientes y uniformes en {−1,1}, con Y = XC. La regresión lineal de Y sobre X tiene pendiente e intercepto cero, por lo que su residual es Y. Sin embargo,

$$
I(C;Y)=0,\qquad I(C;Y\mid X)=1\ \text{bit}.
$$

Conocer X convierte Y en una función biyectiva de C. Sin X, Y es independiente de C. Es un contraejemplo a esa equivalencia general, no a los resultados medidos en los datasets del artículo. El diagnóstico no puede justificar eliminar modalidades obligatorias ni declarar inútil el contexto financiero antes de probarlo bajo el objetivo del proyecto.

## Mezclas, feedback tardío y pérdidas

Si dos rutas generan y registran predicciones antes de conocer y, al madurar la etiqueta se pueden calcular ambos errores. Para predicción pasiva esto es información completa con retraso, no feedback de bandit. El resultado de una cartera con acciones que cambian el estado requiere otra formulación.

Una mezcla convexa $\hat y=\sum_jp_j\hat y_j$ satisface

$$
|y-\hat y|\leq\sum_jp_j|y-\hat y_j|.
$$

Esto no garantiza superar a la mejor ruta en cada sesión. [Freund y Schapire](https://www.schapire.net/papers/FreundSc95.pdf), figura 1 y §2, estudian actualizaciones multiplicativas con pérdidas acotadas. Usar $\ell=\min(1,|y-\hat y|/c)$, con escala c fijada antes de evaluar, permite estudiar ese objetivo auxiliar, pero una garantía para él no equivale a una garantía para el MAE original.

La pérdida recortada tampoco conserva la desigualdad de convexidad anterior. Con y = 0, predicciones 0 y 4, mezcla por mitades y recorte a 1, la pérdida de la mezcla es 1 y la mezcla de pérdidas es 0,5. Se mantendrán explícitos el objetivo de adaptación y la métrica principal.

Los pesos solo se actualizan al madurar las etiquetas, con predicciones realmente emitidas, cohortes y denominadores comunes. No se aplica una cota sin retrasos a esa secuencia. [Joulani y colaboradores](https://proceedings.mlr.press/v28/joulani13.html) analizan esa diferencia. [Ryabchenko y colaboradores](https://proceedings.mlr.press/v291/ryabchenko25a.html) estudian además capacidad finita bajo condiciones concretas sobre pérdidas y demoras. Ninguno de esos resultados autoriza a borrar selectivamente errores pendientes para ahorrar memoria. La cola deberá detener la admisión o registrar una política de descarte predefinida, con cobertura y coste.

## Comprobación reproducible y límites

El [programa de comprobación](../../reports/research/check_memory_algebra.py) utiliza NumPy en CPU y FP64. El [registro ejecutado](../../reports/research/memory-algebra-20261004.json) contiene versión, semilla, hash del programa, tolerancias y valores obtenidos. No carga modelos, datasets ni CUDA.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  reports/research/check_memory_algebra.py
```

Se comprobaron 360 reglas escalares y 162 sistemas proximales con dimensiones de clave 2, 8 y 32, cohortes de 1 y 17 filas, tres valores de η y tres de ρ. El error máximo de la fórmula de norma escalar fue 6,66 × 10⁻¹⁶. El residuo relativo escalado máximo del sistema proximal fue 3,14 × 10⁻¹⁶ y la diferencia máxima por permutación o acumulación por bloques fue 3,52 × 10⁻¹⁴. Son cifras de comprobación finita, no benchmarks ni una prueba por enumeración de las cotas.

Una futura implementación requeriría además pruebas de claves nulas, pesos concentrados, cohortes vacías, NaN/Inf, recuperación de una cohorte parcial sin aplicar dos veces el olvido, precisión reducida y paridad CPU/CUDA. No se presentan esas pruebas futuras como ejecutadas.

El [informe de verificación](../../reports/research/neuroarchitecture-20261004.md) registra las comprobaciones de biblioteca, cobertura, complejidad y mutación del programa algebraico, con su alcance y limitaciones.
