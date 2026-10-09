# Alcance matemático de C y M

Para una matriz compleja cuadrada A, el radio espectral es `max |λ(A)|`, el radio numérico es `w(A) = max_{||x||₂=1} |x*Ax|` y la norma espectral es `||A||₂ = max_{||x||₂=1} ||Ax||₂`. En general son cantidades diferentes. Una matriz normal tiene radio numérico igual a su norma espectral. Una matriz nilpotente puede tener radio espectral cero y radio numérico positivo.

## Rejilla angular y corrección

Definir, para `θ_l = 2πl/m`,

\[
H(\theta)=\frac{e^{-i\theta}A+e^{i\theta}A^*}{2},\qquad
w_m(A)=\max_{0\le l<m}\lambda_{\max}(H(\theta_l)).
\]

La expresión del radio numérico mediante la función soporte da `w_m(A) ≤ w(A)`. Todo ángulo dista como máximo `π/m` de un nodo. Se tiene

\[
\|H(\theta)-H(\phi)\|_2
\le 2\|A\|_2\sin(|\theta-\phi|/2).
\]

La perturbación de autovalores hermíticos y `||A||₂ ≤ ||A||F` dan

\[
w(A)\le w_m(A)+2\|A\|_F\sin(\pi/(2m)).
\]

Esta es la corrección implementada. La derivación se utiliza en aritmética exacta. El cálculo ordinario de autovalores y normas introduce errores no acotados por el contrato de esta API. Por eso el resultado se llama `angular_corrected_estimate`.

El coste del muestreo es aproximadamente `O(m n³)` por matriz. Los bloques reducen los temporales del recorrido directo a `O(b n²)`. Autograd puede retener tensores de todos los ángulos, de orden `O(m n²)`. Reducir el bloque no elimina ese coste. La implementación utiliza [`torch.linalg.eigvalsh`](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigvalsh.html) sobre matrices hermíticas, sin un solver propio.

## Matriz fija y productos variables

Para A fija, las desigualdades clásicas `w(A^j) ≤ w(A)^j` y `||T||₂ ≤ 2w(T)` implican `||A^j||₂ ≤ 2ρ^j` cuando `w(A) ≤ ρ < 1`. No es una contribución atribuida al manuscrito 325. Pintu Bhunia recoge ambas desigualdades en la introducción, expresiones (1.1) y (1.2), de [Power numerical radius inequalities from an extension of Buzano's inequality](https://arxiv.org/html/2305.17657v1).

En una recurrencia forzada `h_{t+1}=Ah_t+u_t`, con `c₀=1` y `c_j=2ρ^j` para `j≥1`,

\[
\|h_t\|\le c_t\|h_0\|+
\sum_{s=0}^{t-1}c_{t-1-s}\|u_s\|.
\]

No desaparece la contribución de las entradas. La condición puntual tampoco controla una secuencia variable. El contraejemplo probado es

\[
A_1=\begin{pmatrix}0&1.5\\0&0\end{pmatrix},\quad
A_2=\begin{pmatrix}0&0\\1.5&0\end{pmatrix}.
\]

Ambas matrices tienen radio espectral cero, radio numérico 0,75 y norma espectral 1,5. El producto `A₂A₁ = diag(0, 2.25)` crece al elevarlo a potencias. Penalizar separadamente cada radio numérico no demuestra estabilidad del producto.

Una condición distinta sería una matriz P positiva definida común con `A_t* P A_t ≤ ρ² P` para todas las actualizaciones admisibles. Eso permite una contracción en norma P y una conversión euclídea que depende de `sqrt(cond(P))`. No basta una P diferente en cada paso o una muestra finita de Jacobianos para establecer esa condición global.

En una actualización residual `z' = z + η f(z,x)`, el operador local completo es `I + η D_z f`, incluidas las puertas y la atención. El refinamiento del lector episódico tiene esa forma, con `η = σ(s)` y `f_k(z) = tanh(W[z, base, read(z), presencia] + b)`. `refinement_jacobians` calcula cada `J_k = I + σ(s) D_z f_k` completo, incluida la derivada de la lectura respecto a la consulta, con la base y los episodios elegidos fijos. La elección de episodios es constante a trozos y la derivada solo vale lejos de los empates del orden de vecinos. Para MAC, `fast_state_jacobian` deriva la transición rápida completa de pesos y momentum, y C penaliza `RᵀJR` de esa transición, no el refinamiento. Ningún peso interno se identifica automáticamente con estos operadores.

[`operator_dynamics.py`](../../../src/mars_titan/cm/operator_dynamics.py) separa en el código las lecturas de esta sección. `fixed_operator_powers` compara `||A^j||₂` medido con `2ŵ^j`, donde ŵ es la estimación corregida de `w(A)`, sin certificar el redondeo. `variable_products` mide normas y radios espectrales de los productos ordenados `A_t ⋯ A_1` y marca el caso del contraejemplo, con todas las estimaciones puntuales por debajo del umbral y algún producto por encima. Una muestra finita de productos describe esa trayectoria y no prueba la condición común anterior.

## Objetivo de representantes

Para clientes E, candidatos F, representación φ y capacidad positiva k,

\[
J(S)=\sum_{e\in E}\min_{c\in S}d(\phi(e),\phi(c)),
\qquad \varnothing\ne S\subseteq F,\quad |S|\le k.
\]

Cuando no existen restricciones adicionales, añadir representantes no aumenta J. La enumeración puede utilizar `min(k, |F|)` candidatos. Los duplicados geométricos mantienen su multiplicidad como clientes. Sobre los IDs definen una pseudométrica, por lo que no satisfacen literalmente el requisito de separación de una métrica del comparador 125.

La distancia euclídea incluye raíz. Su cuadrado y `1-cosine` no se admiten como métricas. Para usar L1 entera a partir de representaciones continuas, debe cuantizarse cada coordenada una vez con una escala y regla comunes. La API no redondea distancias individuales ni elige una cuantización.

La heurística implementada no reproduce el algoritmo del manuscrito [The Approximation Threshold for Metric k-Median](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-Approximation-Threshold-for-Metric-k-Median-September-24-2026/main.pdf). No recibe su factor `1+2/e+ε`. La comparación con el oráculo usa diferencias de coste. Si el óptimo es cero, no se calcula un ratio. Reducir J no implica reducir el error predictivo.
