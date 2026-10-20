# Separación de las señales de mediana y dispersión

Esta variante experimental distingue dos vías de realimentación de la cabeza de cuantiles de MARS-TITAN. Puede impedir que las pérdidas de las colas actualicen la proyección de ubicación y, por separado, que los incrementos de dispersión envíen gradientes al tronco. Los pesos de dispersión siguen aprendiendo. No hay parámetros, generadores aleatorios o estado persistente adicionales. La opción `quantile_routing=None` conserva la referencia y su identidad. No es el valor por defecto de una campaña.

## Motivación y antecedentes

[Guerguiev, Lillicrap y Richards (2017)](https://doi.org/10.7554/eLife.22901) proponen separar señales ascendentes y de realimentación mediante compartimentos dendríticos. Se consultaron resumen, introducción y apartados del modelo con sus ecuaciones 1 y 2. No se reproduce esa dinámica neuronal ni se equiparan esas señales con mediana y dispersión. La analogía utilizada aquí se limita a separar vías de aprendizaje.

[Whittington y Bogacz (2017)](https://doi.org/10.1162/NECO_a_00949), introducción y comienzo de la sección 2, muestran una correspondencia bajo sus condiciones entre errores de codificación predictiva y retropropagación. Esta implementación sigue usando autodiferenciación. No integra su red Hebbiana ni acredita plausibilidad biológica.

También hay antecedentes directos de aprendizaje automático: [Du et al., arXiv v2](https://arxiv.org/abs/1812.02224v2) adaptan pérdidas auxiliares mediante similitud de gradientes y [Yu et al., arXiv v4](https://arxiv.org/abs/2001.06782v4) proponen proyectar gradientes en conflicto. Se consultaron metadatos y resúmenes, sin reproducirlos. La ruta fija implementada aquí no es ninguno de esos algoritmos ni hereda sus garantías. [ORBIT, arXiv v1 de agosto de 2026](https://arxiv.org/abs/2608.13262v1), cuyo resumen y metadatos también se revisaron, utiliza enseñanza entre profundidades con stop-gradient en otro problema. Separar gradientes no es por sí mismo una novedad.

El [diagnóstico de la cabeza original](../../reports/research/quantile-feedback-20261020/README.md) no identificó una causa dominante: las colas se oponen localmente a la mediana en parte de las filas, pero no invierten su gradiente total en las dos instantáneas observadas. La variante es un contraste sobre asignación de señales, no una corrección de un fallo demostrado ni una mejora predictiva asumida.

## Función y derivadas

La cabeza conserva una proyección de cinco filas, la mediana libre y cuatro incrementos softplus. Para unos pesos dados, los cinco valores emitidos son los mismos. `median_from_tails=False` usa una copia sin gradiente de la ubicación al construir las colas. La salida central conserva su vía original. `encoder_from_spreads=False` calcula los incrementos con una representación separada de su grafo. Sus pesos y bias mantienen sus gradientes. Durante inferencia sin gradientes se usa la ruta original, con una sola proyección lineal.

Con ambas opciones falsas, el tronco recibe únicamente el término de mediana de la pérdida común, que equivale a L1/10. Se comprueba también sobre el predictor financiero completo. Los cuantiles auxiliares siguen entrenando sus parámetros. El clipping global y el optimizador pueden acoplar las actualizaciones, por lo que esa propiedad no acredita una trayectoria idéntica a entrenar solo L1.

La intervención define un gradiente sustituto. Aunque los valores de la pérdida no cambien para pesos fijos, su derivada enrutada puede diferir de la diferencia finita de esa función numérica. Las pruebas lo comprueban explícitamente. No se declara descenso de la pérdida conjunta ni convergencia del predictor. Separar las vías puede retirar información útil y perjudicar tanto el MAE como la incertidumbre.

El contraejemplo calculable usa una representación escalar x=0, pesos `(0,10,1,0,0)`, bias `(0,0,1,0,0)` y etiqueta 0,2. El gradiente del término central respecto a x es 0,1, el auxiliar es −0,675 y el conjunto −0,575. La ruta totalmente separada conserva 0,1, mientras la diferencia finita de los mismos valores sigue siendo −0,575. Es una propiedad de este caso técnico y no un resultado financiero.

## Integración y controles

[`QuantileRouting`](../../src/mars_titan/models/quantile_routing.py) valida dos booleanos e identifica el contrato de derivación. [`QuantileHead`](../../src/mars_titan/models/quantile_head.py) conserva los nombres de parámetros y la inicialización. `FinancialConfig` guarda la configuración activa y rechaza su combinación con una cabeza escalar. La ausencia de configuración mantiene el estado previo y el control de ambas vías activadas reproduce la ruta original.

El [catálogo](../../configs/experiments/quantile-routing-v1.json) contiene referencia, control completo, aislamiento de ubicación, aislamiento de dispersión hacia el tronco y combinación. Reutiliza la receta cronológica común del estudio de claves, sin activar su transformación. No altera memoria neuronal, banco episódico, C/M, K, datos o modalidades. Las huellas del productor y del consumidor congelado incluyen el módulo de configuración.

Las pruebas técnicas cubren valores, derivadas parciales declaradas, el gradiente sustituto, recuperación, identidad, configuración inválida, dispositivos y precisiones. Los resultados predictivos requieren el [piloto preespecificado](../experiments/quantile-routing-v1.md). Ese piloto no sustituye el control multisemilla entre salida escalar y cuantiles de #22 ni decide automáticamente la cabeza de la campaña.
