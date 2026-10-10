# Contracción certificada de la memoria lineal de Titans

Estado: derivación propia comprobada con aritmética racional exacta y con pruebas numéricas en CPU. No es un resultado predictivo ni sustituye la evaluación con datos. Cubre la memoria lineal de una capa con claves unitarias. La memoria de dos capas con LayerNorm residual de la receta de campaña queda fuera del teorema y solo hereda la cota de escritura acotada de la sección 6.

El programa [`check_titans_memory_certificate.py`](../../reports/research/check_titans_memory_certificate.py) reproduce todas las comprobaciones y su [recibo del 10 de octubre de 2026](../../reports/research/titans-memory-certificate-20261010.json) conserva los valores. No carga datos financieros, no crea optimizadores y se ejecutó con `CUDA_VISIBLE_DEVICES=-1`. La propuesta que usa este resultado es PT1 en las [propuestas posteriores a Titans](post-titans-proposals.md).

## 1. Planteamiento

La referencia es Titans (Behrouz et al., 2025), ecuaciones 1 a 3 de las actas: sorpresa con momentum, pérdida asociativa y olvido. La notación sigue el [núcleo de memoria](../engineering/titans-memory-core.md). Con una sola matriz $W\in\mathbb{R}^{D\times D}$, clave unitaria $k=W_Kx/\lVert W_Kx\rVert$, valor $v=W_Vx$ y pérdida $\ell=\lVert Wk-v\rVert^2$, el gradiente es $2(Wk-v)k^\top$. Cada fila de salida $w$ (con su momentum $s$ y su valor $v_o$) evoluciona de forma independiente:

$$
s_t=\eta_t s_{t-1}-2\theta_t\,(w_{t-1}^\top k_t-v_{o,t})\,k_t,\qquad
w_t=(1-\alpha_{o,t})\,w_{t-1}+s_t .
$$

El olvido $\alpha$ es vectorial por filas en la implementación y $\eta$, $\theta$ son escalares por flujo. Con $z=(w;s)\in\mathbb{R}^{2D}$ la actualización es afín, $z_t=T_tz_{t-1}+b_t$, con

$$
T=\begin{pmatrix}(1-\alpha)I-2\theta kk^\top & \eta I\\ -2\theta kk^\top & \eta I\end{pmatrix},
\qquad b=2\theta v_o\begin{pmatrix}k\\ k\end{pmatrix}.
$$

La comprobación `code_transition` del recibo contrasta esta expresión con `NeuralMemory` de una capa en FP64. En 156 filas de 12 configuraciones la diferencia máxima fue $2{,}2\times10^{-16}$.

## 2. Reducción a bloques 2 × 2

**Lema 1 (derivación propia).** Sea $\Pi=kk^\top$ y $\Pi_\perp=I-\Pi$. Entonces

$$
T=B_\perp\otimes\Pi_\perp+B_\parallel\otimes\Pi,\qquad
B_\parallel(\alpha,\eta,\theta)=\begin{pmatrix}1-\alpha-2\theta & \eta\\ -2\theta & \eta\end{pmatrix},\qquad
B_\perp=B_\parallel(\alpha,\eta,0).
$$

*Demostración.* $I=\Pi+\Pi_\perp$ y $kk^\top=\Pi$. Sustituyendo en cada bloque de $T$ y agrupando por proyector se obtiene la expresión. ∎

Para $P=P_0\otimes I$ con $P_0\in\mathbb{R}^{2\times2}$ simétrica, los términos cruzados contienen $\Pi_\perp\Pi=0$ y desaparecen:

$$
T^\top PT=B_\perp^\top P_0B_\perp\otimes\Pi_\perp+B_\parallel^\top P_0B_\parallel\otimes\Pi .
$$

La comprobación `decomposition` lo contrasta en dimensiones 1, 2, 3, 8 y 32 con un error máximo de $1{,}8\times10^{-15}$.

## 3. Estabilidad de cada paso

**Lema 2 (derivación propia con el criterio de Jury).** El polinomio característico de $B_\parallel$ es $\lambda^2-(1-\alpha-2\theta+\eta)\lambda+\eta(1-\alpha)$. Con $\alpha\in(0,1]$ y $\eta\in[0,1)$, $B_\parallel$ tiene radio espectral menor que uno si y solo si

$$
\theta<\tfrac12(2-\alpha)(1+\eta).
$$

Las otras dos condiciones de Jury, $|\eta(1-\alpha)|<1$ y $\alpha(1-\eta)+2\theta>0$, se cumplen siempre en ese dominio. Los autovalores de $B_\perp$ son $1-\alpha$ y $\eta$. Como $\theta\le\theta_{max}=0{,}1$, cada transición de la implementación es estable por separado siempre que $\alpha>0$ y $\eta<1$. Las sigmoides garantizan ambas cosas, aunque sin margen uniforme.

## 4. La estabilidad de cada paso no basta

**Proposición 3 (contraejemplo propio, verificado en racionales).** En la caja $\alpha\in[27/10000,1]$, $\eta\in[0,9/10]$, $\theta\in[0,1/10]$ existe una secuencia periódica de ocho transiciones, todas con radio espectral menor que uno según Jury y con espectro real no negativo, cuyo producto tiene radio espectral mayor que 1,717.

La secuencia del recibo usa una sola dirección de clave $k=(1,0)$. Alterna pasos sin escritura ($\theta=0$, $\eta=0{,}9$, $\alpha=0{,}0027$), un paso con escritura y sin momentum y un paso con olvido completo ($\alpha=1$, $\eta=0{,}9$, $\theta=0{,}1$). La divergencia procede de que $B_\parallel(\alpha,\eta,\theta)$ y $B_\parallel(\alpha,\eta,0)$ no conmutan. El programa calcula $\operatorname{tr}(N^{64})$ con enteros exactos para $M=N/d$ y comprueba $|\operatorname{tr}(N^{64})|>4d^{64}$. Como $|\operatorname{tr}(M^n)|\le4\rho(M)^n$, eso demuestra $\rho(M)>1$ sin redondeo. La cota inferior certificada es $\rho(M)\ge1{,}717$ y el valor flotante es 1,755, un crecimiento de 1,073 por paso.

Consecuencias. Restringir el espectro de cada paso, como hace MDN con su restricción de cuadrantes en otra parametrización (Huang et al., 2026), no garantiza por sí solo trayectorias acotadas cuando las puertas cambian con la entrada. El fenómeno es conocido en sistemas conmutados (Liberzon, 2003) y en el estudio del radio espectral conjunto (Jungers, 2009). Cao et al. (2026) lo señalan para SSM selectivos. El contraejemplo no afirma que esa secuencia aparezca en datos financieros. Afirma que la parametrización actual la permite.

## 5. Certificado común para cualquier secuencia de claves y puertas

Sea la caja $\mathcal{B}=[\alpha_{lo},1]\times[0,\eta_{hi}]\times[0,\theta_{hi}]$ y sean $V(\mathcal{B})$ sus ocho vértices.

**Teorema 4 (derivación propia).** Sea $D\ge2$. Existe una función de Lyapunov cuadrática común $P\succ0$ con $T^\top PT\preceq\rho^2P$ para todas las claves unitarias y todas las puertas de $\mathcal{B}$ si y solo si existe $P_0\succ0$ de tamaño 2 × 2 con

$$
B_\parallel(\alpha,\eta,\theta)^\top P_0\,B_\parallel(\alpha,\eta,\theta)\preceq\rho^2P_0\quad\text{para todo }(\alpha,\eta,\theta)\in V(\mathcal{B}).
$$

En ese caso, para cualquier secuencia de claves y puertas en $\mathcal{B}$, la parte homogénea cumple $\lVert z_t\rVert_{P_0\otimes I}\le\rho^t\lVert z_0\rVert_{P_0\otimes I}$ y por tanto $\lVert z_t\rVert\le\sqrt{\kappa(P_0)}\,\rho^t\lVert z_0\rVert$.

*Suficiencia.* Por complemento de Schur, $B^\top P_0B\preceq\rho^2P_0$ equivale a

$$
\begin{pmatrix}\rho^2P_0 & B^\top P_0\\ P_0B & P_0\end{pmatrix}\succeq0,
$$

que es afín en $B$. $B_\parallel$ es afín en $(\alpha,\eta,\theta)$ y cada punto de la caja es combinación convexa de sus vértices con la misma combinación de imágenes. La suma convexa de matrices semidefinidas es semidefinida, así que la desigualdad vale en toda la caja, incluido $\theta=0$, que da $B_\perp$. Por el Lema 1 y porque $X\preceq Y$ implica $X\otimes\Pi\preceq Y\otimes\Pi$ para $\Pi\succeq0$, $T^\top(P_0\otimes I)T\preceq\rho^2(P_0\otimes I)$. Las filas con olvidos distintos son sistemas independientes con la misma cota. ∎

*Necesidad.* Si $P$ sirve para todas las claves, también sirve $(I_2\otimes Q)^\top P(I_2\otimes Q)$ para cada matriz ortogonal $Q$, porque el conjunto de transiciones es invariante al rotar la clave. Promediando con la medida de Haar del grupo ortogonal se obtiene $\bar P\succ0$ que conmuta con todo $I_2\otimes Q$ y sigue cumpliendo la desigualdad por convexidad. El grupo ortogonal actúa de forma irreducible en $\mathbb{R}^D$ y su conmutante son los múltiplos de la identidad (lema de Schur en su versión real), así que $\bar P=P_0\otimes I$. Con $D\ge2$ las direcciones de $\Pi$ y $\Pi_\perp$ existen a la vez y el Lema 1 exige la desigualdad para ambos bloques. ∎

La técnica de funciones de Lyapunov cuadráticas comunes y LMI en vértices es estándar (Boyd et al., 1994). La parte propia es la reducción exacta de la memoria de Titans a dos bloques 2 × 2 para claves arbitrarias, la condición necesaria y suficiente que resulta, las cajas certificadas y su uso para parametrizar las puertas. La búsqueda de precedentes está en el [registro de novedad](novelty-ledger.md#búsqueda-de-precedentes-del-9-y-10-de-octubre-de-2026).

## 6. Cajas certificadas

Los certificados se encontraron con una búsqueda flotante de $P_0=\begin{pmatrix}1&p\\p&q\end{pmatrix}$ y después se redondearon a fracciones. La aceptación es exacta: el programa comprueba con `fractions.Fraction` que $P_0\succ0$ y que $\rho^2P_0-B^\top P_0B$ tiene menores principales no negativos en los ocho vértices.

| Caja | $\alpha_{lo}$ | $\eta_{hi}$ | $\theta_{hi}$ | $\rho$ exacto | $p$ | $q$ | Semivida de la contracción |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 27/10000 | 3/10 | 1/10 | 999/1000 | 91/500 | 39/4 | 693 pasos |
| B2 | 1/100 | 1/2 | 1/10 | 49861/50000 | 37/100 | 481/50 | 249 pasos |
| PT1 | 1/500 | 3/10 | 1/10 | 9997/10000 | 181/1000 | 49/5 | 2.310 pasos |

La caja PT1 es la que usa la propuesta implementable. El número de condición de $P_0$ está entre 9,79 y 9,84 en las tres. En 600 secuencias aleatorias de 32 pasos dentro de B1, con vértices y puntos interiores y dimensiones 2, 4 y 8, la norma inducida por paso nunca superó $\rho$ (cociente máximo 0,999998).

La frontera siguiente es una estimación flotante del menor $\alpha_{lo}$ con certificado cuadrático para cada $\eta_{hi}$, con $\theta_{hi}=0{,}1$. Es una condición suficiente. Fuera de ella podría existir estabilidad con una función de Lyapunov no cuadrática.

| $\eta_{hi}$ | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 | 0,9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $\alpha_{lo}$ mínimo | 0,00016 | 0,00069 | 0,0017 | 0,0036 | 0,0074 | 0,018 | 0,054 | 0,12 | 0,23 |
| Semivida máxima (pasos) | 4.219 | 1.004 | 403 | 190 | 94 | 39 | 13 | 5 | 3 |

La receta de campaña inicia las puertas en $\alpha\approx0{,}0027$ (semivida de 256), $\eta=0{,}5$ y $\theta=0{,}05$. Ese punto queda fuera de la frontera para $\eta_{hi}=0{,}5$, que exige $\alpha\ge0{,}0074$. Además las sigmoides no acotan $\alpha$ lejos de cero ni $\eta$ lejos de uno, así que la receta actual no tiene certificado. Para $\eta_{hi}=0{,}5$ y $\alpha_{lo}=0{,}0027$ no hemos encontrado ni certificado ni contraejemplo. La pregunta queda abierta.

## 7. Escritura acotada y contracción incremental

**Proposición 5 (resultado conocido en otra forma).** Sea $\psi_\delta(r)=\max(-\delta,\min(\delta,r))$ la derivada de la pérdida de Huber (dividida por dos) y supongamos la escritura $s_t=\eta_ts_{t-1}-2\theta_t\psi_\delta(w_{t-1}^\top k_t-v_{o,t})k_t$ con puertas en $[\alpha_{lo},1]\times[0,\eta_{hi}]\times[0,\theta_{hi}]$ y $\eta_{hi}<1$. Entonces, para cualquier secuencia de claves, valores y puertas,

$$
\lVert s_t\rVert\le\max\!\Big(\lVert s_0\rVert,\frac{2\theta_{hi}\delta}{1-\eta_{hi}}\Big)=:\bar s,\qquad
\lVert w_t\rVert\le\max\!\Big(\lVert w_0\rVert,\frac{\bar s}{\alpha_{lo}}\Big).
$$

*Demostración.* $\lVert s_t\rVert\le\eta_{hi}\lVert s_{t-1}\rVert+2\theta_{hi}\delta$ y $\lVert w_t\rVert\le(1-\alpha_{lo})\lVert w_{t-1}\rVert+\bar s$. Ambas recurrencias escalares quedan acotadas por su punto fijo o por su valor inicial. ∎

La pérdida de Huber como sesgo atencional procede de la variante Yaad de MIRAS (Behrouz et al., 2026). Que el olvido con actualización acotada mantenga acotados los parámetros es la idea de la modificación σ del control adaptativo robusto (Ioannou y Kokotovic, 1984). No reivindicamos esta proposición. El recibo la comprueba con valores t de Student de 1,5 grados y saltos de factor mil en el 1 % de pasos. La norma máxima observada fue el 0,4 % de la cota en su forma sumada. La cota no depende de la profundidad si en lugar de recortar el residuo se recorta la norma de cada fila del gradiente interno a $G$, porque solo usa $\lVert\Delta s\rVert\le\theta_{hi}G$ por fila. Con una capa y clave unitaria la fila del gradiente es $2\,r\,k$, así que recortar su norma a $G=2\delta$ coincide exactamente con Huber.

**Teorema 6 (derivación propia).** Si las puertas están en una caja $\mathcal{B}$ certificada con $(P_0,\rho)$ y la escritura usa $\psi_\delta$, dos trayectorias con las mismas claves, valores y puertas cumplen $\lVert z_t-z_t'\rVert_{P_0\otimes I}\le\rho^t\lVert z_0-z_0'\rVert_{P_0\otimes I}$.

*Demostración.* $\psi_\delta$ es monótona y 1-Lipschitz, así que $\psi_\delta(r)-\psi_\delta(r')=c\,(r-r')$ con $c\in[0,1]$. La diferencia sigue la transición homogénea con paso $c\theta\in[0,\theta_{hi}]$, que pertenece a $\mathcal{B}$. Se aplica el Teorema 4 paso a paso. ∎

La consecuencia práctica es que la memoria olvida su estado inicial y cualquier perturbación pasada a un ritmo certificado, y que dos copias del mismo flujo recuperadas desde estados distintos convergen. Es la propiedad de contracción en el sentido de Lohmiller y Slotine (1998). En 200 pares de trayectorias de 400 pasos dentro de la caja PT1, con recorte $\delta=0{,}5$ y valores de colas pesadas, la distancia nunca superó $\rho^t$ veces la inicial (cociente máximo 0,81).

## 8. Lo que no queda demostrado

- La memoria de la campaña tiene dos capas con GELU y LayerNorm residual. El Teorema 4 no la cubre. La Proposición 5, con recorte por filas del gradiente interno, sí acota su estado. Su contracción solo puede medirse con el Jacobiano completo que ya calcula `transition_jacobian.py`, y la [especificación de diagnósticos](memory-diagnostics.md) lo pide sobre datos reales cuando se levante el bloqueo.
- El entrenamiento exterior por retropropagación truncada no está cubierto. La contracción del estado rápido no implica que el gradiente exterior esté acotado ni que el ajuste converja.
- Ninguna de estas propiedades implica menor error predictivo. PT1 se evalúa como no inferioridad con un control emparejado.
- No se ha ejecutado Lean. Los Lemas 1 y 2 y la extensión de vértices a caja son formalizables con Mathlib (matrices, `PosSemidef`, producto de Kronecker) con un esfuerzo moderado. Los certificados racionales concretos ya son una comprobación exacta, pero dependen de esos lemas. La necesidad del Teorema 4 usa la medida de Haar y su formalización sería más costosa. Queda como trabajo pendiente y no se presenta como hecho.

## Referencias

- Behrouz, A., Razaviyayn, M., Zhong, P. y Mirrokni, V. (2026). It's all connected: A journey through test-time memorization, attentional bias, retention, and online optimization. *International Conference on Learning Representations*.
- Behrouz, A., Zhong, P. y Mirrokni, V. (2025). Titans: Learning to memorize at test time. *Advances in Neural Information Processing Systems, 38*.
- Boyd, S., El Ghaoui, L., Feron, E. y Balakrishnan, V. (1994). *Linear matrix inequalities in system and control theory*. SIAM. https://doi.org/10.1137/1.9781611970777
- Cao, L., Liu, W., Chen, Z. y Qin, Y. (2026). A control-theoretic view of Mamba on stability and robustness. *Proceedings of the 43rd International Conference on Machine Learning*, PMLR 306, 11350–11364.
- Huang, Y., Liu, X., Huang, H., Lin, X., Liu, Z., Chu, X., Xie, Z. y Cheng, B. (2026). MDN: Parallelizing stepwise momentum for delta linear attention. *Proceedings of the 43rd International Conference on Machine Learning*, PMLR 306, 47153–47177.
- Ioannou, P. A. y Kokotovic, P. V. (1984). Instability analysis and improvement of robustness of adaptive control. *Automatica, 20*(5), 583–594. https://doi.org/10.1016/0005-1098(84)90009-8
- Jungers, R. (2009). *The joint spectral radius: Theory and applications*. Springer. https://doi.org/10.1007/978-3-540-95980-9
- Liberzon, D. (2003). *Switching in systems and control*. Birkhäuser. https://doi.org/10.1007/978-1-4612-0017-8
- Lohmiller, W. y Slotine, J.-J. E. (1998). On contraction analysis for non-linear systems. *Automatica, 34*(6), 683–696. https://doi.org/10.1016/S0005-1098(98)00019-3
