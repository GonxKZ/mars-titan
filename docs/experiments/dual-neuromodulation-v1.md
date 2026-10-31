# Dos señales para las puertas de escritura y olvido, N1

Está implementado el conversor de señales `DualNeuromodulation` y su uso por `FinancialPredictor.prepare`. Utiliza dos entradas numéricas explícitas y el hook `gate_shift` de la memoria existente. La generación de señales financieras maduras, su normalización, el control temporal barajado y su publicación nativa siguen pendientes. Los consumidores que todavía no registran esa fuente rechazan N1 al construir. No se ha entrenado ni evaluado científicamente esta variante.

## Fuente y alcance

[Yu y Dayan (2005), *Uncertainty, Neuromodulation, and Attention*](https://www.gatsby.ucl.ac.uk/~dayan/papers/yud2005.pdf), DOI [10.1016/j.neuron.2005.04.026](https://doi.org/10.1016/j.neuron.2005.04.026), distingue incertidumbre esperada e inesperada y propone funciones de acetilcolina y noradrenalina. Se consultaron el resumen y el planteamiento inicial, no todo el artículo. La reducción de escritura ante incertidumbre esperada financiera es una hipótesis propia. No se deduce directamente de ese modelo biológico ni demuestra que una volatilidad alta sea ruido irrelevante.

Se comprobó la [versión 1 de Titans, sección 3.1, ecuaciones 13 y 14](https://arxiv.org/html/2501.00663v1#S3.SS1). Allí α controla el olvido, η la persistencia del momentum y θ la contribución del gradiente asociativo. La implementación conserva esa distinción. N1 desplaza los logits de θ y α y deja η sin modificar. No sustituye la sorpresa asociativa por error financiero ni cambia la pérdida de memoria. Las fuentes se consultaron el 31 de octubre de 2026.

## Regla implementada

Para cada fila se reciben dos valores normalizados, e y u, en `[0, 1]`. Sus nombres indican incertidumbre esperada e inesperada. El núcleo valida forma, rango y precisión, pero estos valores numéricos no acreditan por sí solos su maduración financiera.

Con a, b y c no negativos y una cota h positiva:

```text
delta_write  = h * tanh(a*u - b*e)
delta_forget = h * tanh(c*u)

theta_N1 = theta_max * sigmoid(z_theta + delta_write)
alpha_N1 = sigmoid(z_alpha + delta_forget)
S_next   = eta*S_previous - theta_N1*gradient_associative
M_next   = (1-alpha_N1)*M_previous + S_next
```

`a`, `b` y `c` corresponden a `write_unexpected`, `write_expected` y `forget_unexpected`. La cota h es `bound`. Los coeficientes se configuran en `[0, 4]` y la cota en `(0, 4]`. Son elecciones del experimento, no constantes obtenidas del artículo. No se han aprendido. Si PT1 utiliza una caja de puertas, el desplazamiento se aplica antes de ella y α conserva su suelo.

Con u creciente y e fija, el desplazamiento de escritura no disminuye. Con e creciente y u fija, no aumenta. El desplazamiento de olvido es no negativo y no depende de e. Las puertas pueden saturarse. Estos signos no prueban que una actualización sea útil o que se conserve la información relevante. Tampoco son una cota del Jacobiano completo ni de la estabilidad de la red.

Señales nulas o tres coeficientes nulos producen desplazamientos exactamente cero. El módulo no añade parámetros ni buffers y no consume RNG. Se reutilizan `NeuralMemory.update`, el momentum y la actualización asociativa existentes. La memoria compartida de PT4 conserva sus propias puertas. K1, K2 y K4 siguen siendo otro eje.

## Contrato del predictor

`FinancialConfig.dual_neuromodulation` admite `None`, una instancia de `DualNeuromodulationConfig` o un mapa completo de sus cuatro campos. Con `None` se conserva la identidad anterior. Al activarlo, la identidad incluye `dual_neuromodulation_v1` y sus coeficientes.

`FinancialPredictor.prepare(..., neuromodulation=signal)` recibe un tensor `[filas, 2]`, ordenado como `[expected, unexpected]`, en FP32 o FP64. Necesita las mismas filas, dispositivo y precisión del predictor. No admite autograd en esas señales ni autocast. Se rechazan NaN, infinitos, valores fuera de rango, entradas ambiguas y más de 8192 filas. No se recortan o sustituyen valores inválidos para continuar.

La señal es obligatoria con N1 y se rechaza sin declararlo. La memoria debe estar en modo `mac_online`. La variante se mantiene separada de `mature_reliability` para evitar sumar dos políticas sin un contraste propio. C sigue rechazándose porque su diagnóstico actual vuelve a calcular la transición sin estas señales. Una señal cero no habilita silenciosamente ese cruce.

El estado rápido continúa perteneciendo al llamante. Los parámetros compartidos y el objetivo predictivo permanecen separados de los valores externos. La configuración del módulo se comprueba al verificar el predictor y se conserva al recuperar su estado.

## Trabajo pendiente para la variante financiera

Falta construir e a partir de una fuente preespecificada, como volatilidad del residuo ya maduro, y u mediante el CUSUM declarado del error financiero maduro y estandarizado. Ese error debe usar la predicción realmente emitida. Faltan ventana, normalización, inicialización, tratamiento de faltantes, publicación del estado y control causal barajado con versiones y pruebas propias. La anchura de cuantiles de la predicción actual solo está disponible después de calcularla y requeriría otro orden explícito de actualización.

La sesión nativa debe usar el mismo snapshot de señales para todas las decisiones de un evento y actualizarlo únicamente con resultados admitidos, según el contrato de publicación. El acumulador de fiabilidad existente no se reutiliza sin cambios: su puntuación es direccional y su recorrido documentado aplica resultados antes de emitir. No equivale al error financiero o al orden requerido por esta variante.

El conversor numérico no permite atribuir causalidad a sus señales. Una volatilidad predecible puede contener información útil que una escritura menor perdería. Un error alto también puede proceder de un episodio aislado, de un predictor mal calibrado o de ruido, sin cambio de régimen. El retraso de maduración limita la reacción. Estos casos deben conservarse como resultados posibles, con la referencia y los controles, sin presuponer una mejora.

## Comprobaciones técnicas

El [recibo](../../reports/engineering/dual-neuromodulation-20261031/verification.json) conserva 423 identificadores de caso distintos. La selección principal pasa 415 casos CPU sin omisiones y excluye 61 CUDA. La ampliación pasa 41 casos y el caso FP32 adicional uno, con solapamiento. No se suman como 457 casos nuevos ni como una ejecución de toda la batería.

Se comprobaron parámetros comunes, predicción y estado rápido neutrales exactos en FP32/FP64, signos de las puertas reales con η idéntica, ausencia de cambios en el estado anterior, recuperación y siguiente predicción, gradiente de la proyección θ, entradas inválidas y rechazo de consumidores sin fuente. Las 18 mutaciones se detectaron con G1/G2 verdes y M1 rojo, sin intentos de optimizador.

El módulo nuevo cubre sus 38 sentencias y 12 ramas. Los cuatro archivos seleccionados cubren 1282 de 1597 sentencias y 556 de 804 ramas. Las catorce funciones modificadas están medidas. El CRAP máximo es 212,82 en el constructor preexistente del consumidor de mecanismos. La cobertura del resto de esas clases es parcial y estas cifras no demuestran ausencia de errores. CUDA y las medidas de coste continúan pendientes. No hay mejora predictiva medida.

La revisión posterior detectó que un entero no representable, como `1 << 2048`, producía `OverflowError` al comprobar finitud antes del rango. Se reprodujo en dos campos y se corrigió el orden. El [recibo de la versión corregida](../../reports/engineering/dual-neuromodulation-20261031/verification-integer-boundary.json) conserva 425 casos CPU correctos sin omisiones y 61 CUDA excluidos. Se repitieron las 18 mutaciones anteriores y se añadió una que restaura ese orden incorrecto. Las 19 se detectaron con G1/G2 verdes y M1 rojo. Los recuentos anteriores no se suman y su evidencia permanece intacta. No cambia la regla ni el comportamiento de coeficientes válidos.
