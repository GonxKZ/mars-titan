# Alcance matemático de la política de rutas

[RoutePolicy](../../src/mars_titan/models/titans/refinement_policy.py) utiliza nueve
rasgos disponibles después de K1 y estima cinco ganancias firmadas de MAE. K1 tiene
utilidad incremental cero. Los parámetros de la política quedan congelados durante
evaluación. Esta formulación describe la regla existente y una cota condicional
elemental, sin atribuirle novedad matemática.

## Rangos por rasgo

La comprobación implementada es

\[
B=\prod_{j=1}^{9}[m_j,M_j],\qquad
m_j=\min_{i\in\mathrm{ajuste}}z_{ij},\quad
M_j=\max_{i\in\mathrm{ajuste}}z_{ij}.
\]

`support_min` y `support_max` son los extremos marginales de esta caja. No
identifican el soporte conjunto, la envolvente convexa ni una región calibrada
de alta densidad. Dos observaciones (0,0) y (1,1) producen la caja [0,1]², que
también acepta (0,1), una combinación no observada y fuera del segmento entre
ambas. Los rasgos del lector tienen dependencias adicionales, incluida una
diferencia que se obtiene restando dos de sus componentes.

Fuera de B se devuelve K1 y se evita evaluar la regresión sobre esos valores.
Dentro de B siguen siendo posibles combinaciones nuevas y cambios de distribución.
La pertenencia a B no acredita que una continuación vaya a ayudar. Esta regla
no es un detector OOD calibrado.

## Utilidad y decisión

Sea 0 la ruta K1 y r una de las cinco continuaciones. Para una distribución de
referencia y rasgos z, se define

\[
g_r(z)=\mathbb E[|Y-\widehat Y_0|-|Y-\widehat Y_r|\mid Z=z],
\qquad g_0(z)=0.
\]

El ajuste ridge aproxima estas ganancias mediante pérdidas observadas maduras
y pesos que dan la misma contribución total a cada sesión. No garantiza representar
exactamente esa esperanza condicional ni que la distribución futura coincida con
la de ajuste.

Con costes incrementales c_r en unidades de MAE, c₀=0 y margen η=`min_gain`,
la utilidad utilizada es

\[
v_0(z)=\widehat v_0(z)=0,\qquad
v_r(z)=g_r(z)-c_r-\eta,\qquad
\widehat v_r(z)=\widehat g_r(z)-c_r-\eta.
\]

Para z∈B se escoge el primer máximo de \(\widehat v_r(z)\) según el orden declarado.
El margen η es una penalización fija por continuar. No es una probabilidad,
un intervalo de confianza ni una cota del error de ridge.

## Cota condicional

Si se aportase una cota uniforme externa, con δ≥0,

\[
\max_r|\widehat v_r(z)-v_r(z)|\leq\delta,
\]

entonces, para cualquier alternativa a y la ruta elegida \(\widehat r\),

\[
v_a-v_{\widehat r}
=(v_a-\widehat v_a)
+(\widehat v_a-\widehat v_{\widehat r})
+(\widehat v_{\widehat r}-v_{\widehat r})
\leq\delta+0+\delta=2\delta.
\]

La cota compara utilidades penalizadas por costes y margen. Si se compara con
\(u_r=g_r-c_r\), sin descontar η, la cota correspondiente es \(2\delta+\eta\),
porque la diferencia de penalizaciones entre dos rutas es como mucho η.
Si el objetivo de comparación omite costes y margen, sus diferencias entre rutas
también entran en la cota.

Fuera de B se fuerza K1, que no tiene por qué maximizar las estimaciones. Por
ello, la cota anterior no se aplica frente a todas las rutas en esas filas.
Restringir el comparador a K1 daría una diferencia cero, pero no informa sobre
las continuaciones descartadas.

No existe un δ certificado en el ridge actual. El error de ajuste, los límites
marginales y CQR de la predicción residual no proporcionan esta cota uniforme
sobre ganancias condicionales. El lema explicita una hipótesis suficiente para
deducir esa cota. No acredita mejora predictiva ni utilidad financiera realizada.

## Comprobación formal

[RoutePolicyRegret.lean](../../reports/research/lean/RoutePolicyRegret.lean)
comprueba solo la desigualdad algebraica sobre funciones reales de utilidad y
estimación. Recibe como hipótesis el error uniforme y que la ruta elegida
maximiza la estimación. No formaliza la esperanza condicional, el ajuste ridge,
el programa PyTorch ni su aritmética en coma flotante.

El [recibo de verificación](../../reports/research/lean/route-policy-verification-20261018.json)
registra las versiones, huellas, axiomas y el control negativo ejecutado.
Reutiliza Lean 4.35.0 y el mismo pin de Mathlib del proyecto formal existente.
La aceptación del lema no demuestra sus hipótesis para los datos financieros.
