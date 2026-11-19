# Dos señales para las puertas de escritura y olvido, N1

Está implementado el conversor de señales `DualNeuromodulation` y su uso por `FinancialPredictor.prepare`. Utiliza dos entradas numéricas explícitas y el hook `gate_shift` de la memoria existente. La [fuente madura](dual-neuromodulation-source-v1.md) implementa RMS, CUSUM saturado y control de prefijo pasado. `FrozenFinancialConsumer` y el enlace ordinario de `FinancialSession` registran esa fuente y publican su estado con la sesión nativa. Otros enlaces sin registro propio la rechazan. No se ha entrenado ni evaluado científicamente esta variante.

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

## Integración financiera y límites

La fuente separada implementa e a partir del RMS de residuo maduro y u mediante un CUSUM saturado del error financiero emitido, con escala previa. El enlace convierte los resultados aplicados por el ejecutor en filas `MatureWrite`, utilizando el punto emitido y el objetivo maduro de cada una. Ventana, normalización, inicio, eventos sin resultados, estado y control pasado tienen versión y pruebas propias. La anchura de cuantiles de la predicción actual solo está disponible después de calcularla y requeriría otro orden explícito de actualización.

La sesión obtiene un snapshot común antes de preparar las predicciones y selecciona sus filas por bloque. Después de emitir, actualiza la fuente con todos los resultados aplicados, independientemente de que M0, M1, M2 o M3 los conserven como episodios. El estado anterior y el posterior se publican juntos. La recuperación reconstruye el avance desde el registro de resultados y lo compara con el estado guardado. El acumulador de fiabilidad existente tiene otra puntuación y otro orden de uso, por lo que no se reutiliza como sustituto de esta fuente.

El consumidor necesita `NeuromodulationSourceConfig` si N1 está activo. El snapshot identifica configuración, estado, evento, corte y filas. Solo sus dos señales numéricas entran al modelo. La conexión actual admite el enlace ordinario de Titans. E2 y el consumidor de mecanismos continúan rechazados hasta registrar y comprobar su composición. No se da por probado un cruce con N2 por la existencia de sus componentes separados.

El conversor numérico no permite atribuir causalidad a sus señales. Una volatilidad predecible puede contener información útil que una escritura menor perdería. Un error alto también puede proceder de un episodio aislado, de un predictor mal calibrado o de ruido, sin cambio de régimen. El retraso de maduración limita la reacción. Estos casos deben conservarse como resultados posibles, con la referencia y los controles, sin presuponer una mejora.

## Comprobaciones técnicas

El [recibo](../../reports/engineering/dual-neuromodulation-20261031/verification.json) conserva 423 identificadores de caso distintos. La selección principal pasa 415 casos CPU sin omisiones y excluye 61 CUDA. La ampliación pasa 41 casos y el caso FP32 adicional uno, con solapamiento. No se suman como 457 casos nuevos ni como una ejecución de toda la batería.

Se comprobaron parámetros comunes, predicción y estado rápido neutrales exactos en FP32/FP64, signos de las puertas reales con η idéntica, ausencia de cambios en el estado anterior, recuperación y siguiente predicción, gradiente de la proyección θ, entradas inválidas y rechazo de consumidores sin fuente. Las 18 mutaciones se detectaron con G1/G2 verdes y M1 rojo, sin intentos de optimizador.

El módulo nuevo cubre sus 38 sentencias y 12 ramas. Los cuatro archivos seleccionados cubren 1282 de 1597 sentencias y 556 de 804 ramas. Las catorce funciones modificadas están medidas. El CRAP máximo es 212,82 en el constructor preexistente del consumidor de mecanismos. La cobertura del resto de esas clases es parcial y estas cifras no demuestran ausencia de errores. CUDA y las medidas de coste continúan pendientes. No hay mejora predictiva medida.

La revisión posterior detectó que un entero no representable, como `1 << 2048`, producía `OverflowError` al comprobar finitud antes del rango. Se reprodujo en dos campos y se corrigió el orden. El [recibo de la versión corregida](../../reports/engineering/dual-neuromodulation-20261031/verification-integer-boundary.json) conserva 425 casos CPU correctos sin omisiones y 61 CUDA excluidos. Se repitieron las 18 mutaciones anteriores y se añadió una que restaura ese orden incorrecto. Las 19 se detectaron con G1/G2 verdes y M1 rojo. Los recuentos anteriores no se suman y su evidencia permanece intacta. No cambia la regla ni el comportamiento de coeficientes válidos.

La conexión nativa tiene un [recibo separado](../../reports/engineering/neuromodulation-native-20261101/verification.json), con 317 casos CPU distintos y 18 mutaciones detectadas. Comprueba la fuente ligada a resultados realmente emitidos, el snapshot previo, M0–M3, publicación interrumpida y recuperación. El módulo del enlace cubre sus 42 sentencias y 10 ramas. La ruta CUDA, el coste de esta integración y su efecto predictivo siguen pendientes. Estos casos no se suman a los recibos anteriores como si fueran todos nuevos.
