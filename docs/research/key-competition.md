# Centrado y competencia de claves neuronales

Esta ampliación experimental de Titans-MAC transforma las claves y las consultas antes de su normalización L2. No cambia la pérdida asociativa, la memoria persistente, el banco episódico ni el número de refinamientos K. Permanece desactivada sin `memory_key_competition`. El [piloto de siete configuraciones](../../reports/research/key-competition-20261020/development-v1/README.md) registra un empeoramiento de la combinación y ninguna configuración supera al residual cero.

## Motivación y fuentes

Glovaci, Mihály, Vervaeke y Hu estudian la inhibición, la esparsificación y la decorrelación de la actividad de células granulares del giro dentado mediante noradrenalina en [Communications Biology, 2026](https://doi.org/10.1038/s42003-026-09592-0). La inspección del 20 de octubre comprende metadatos, resumen, introducción y resultados iniciales. No reproduce sus experimentos. Una recuperación posterior de la página quedó bloqueada por la redirección de cookies del editor.

El artículo de [Iyer et al. (2022)](https://doi.org/10.3389/fnbot.2022.846219) combina dendritas activas y representaciones dispersas para aprendizaje multitarea. Se revisaron las secciones 3 y 5.1 como antecedente computacional. [Rao y Ballard (1999)](https://doi.org/10.1038/4580) proponen un modelo de codificación predictiva de la corteza visual. De esta fuente se inspeccionaron resumen y metadatos, sin reproducir su modelo. La predicción financiera y el error maduro de MARS-TITAN son problemas diferentes de esos resultados.

La transformación que sigue es una adaptación matemática propia. Centrado, normalización y selección de componentes tienen antecedentes y no se presentan como algoritmos nuevos. No implementa un circuito biológico, no acredita separación de patrones y no demuestra superioridad frente al estado del arte. Los metadatos DOI de las dos fuentes de Nature se contrastaron también con Crossref. No se redistribuyen sus textos o figuras.

## Transformación y puntos de inserción

Para un vector real de dimensión D, la opción de centrado calcula `u = x − mean(x)`. Sin ella, `u = x`. La competencia conserva las r coordenadas de mayor valor absoluto, mantiene sus signos y resuelve los empates por posición creciente. El control fijo conserva las coordenadas `floor(j D / r)`, con `j = 0,…,r−1`. El resultado es `u_mask / max(||u_mask||₂, 10⁻¹²)`. El vector cero devuelve cero.

`KeyCompetition` valida las opciones y declara su identidad en [`key_competition_config.py`](../../src/mars_titan/models/titans/key_competition_config.py). La referencia numérica está en [`key_competition.py`](../../src/mars_titan/models/titans/key_competition.py). Se aplica después de las proyecciones y activaciones existentes en las dos rutas de escritura de `NeuralMemory`, en la consulta de `TitansMAC` y en la consulta de mercado de `SessionContext`. No modifica los valores asociados. Las huellas del ejecutor incluyen los dos módulos nuevos.

La ausencia de configuración conserva la identidad y la ruta históricas. Con todas las componentes y sin centrado se llama directamente al `F.normalize` original para conservar también el grafo de derivación. La configuración activa tiene una identidad diferente incluso cuando constituye ese control numérico. Los checkpoints de una variante no se hacen pasar por los de otra.

El [catálogo](../../configs/experiments/key-competition-v1.json) contiene siete configuraciones: original, todas las componentes, centrado, competencia, centrado con competencia, selección fija y centrado con selección fija. En dimensión 64 se conservan 16 componentes en las cuatro variantes con selección. No se cambia el valor por defecto de ninguna campaña.

## Alcance matemático y objeciones

El centrado es la proyección ortogonal `C = I − 11ᵀ/D`. En aritmética real, `||Cx||₂² = ||x||₂² − D mean(x)²`. Conservar las r mayores componentes absolutas minimiza el error L2 entre aproximaciones con r coordenadas. Son propiedades de esas operaciones, no garantías sobre el MAE financiero. Los cálculos ordinarios en FP32 y FP64 no son certificados numéricos.

El centrado puede borrar información útil. Por ejemplo, `(1,0)` y `(0,−1)` producen el mismo vector centrado normalizado. La selección también puede aumentar similitud: `(1,0.9,0)` y `(1,0,0.9)` producen el mismo vector al conservar una coordenada. Cerca de un empate la selección puede saltar entre coordenadas, por lo que no se declara una cota Lipschitz global. La derivada mantiene fija la máscara seleccionada y el centrado acopla las derivadas de las coordenadas. Se comprueban diferencias finitas lejos de los cambios de selección.

No hay parámetros, generadores aleatorios o estados persistentes adicionales. La ordenación estable cuesta O(D log D) por vector y sus temporales son O(D). La salida sigue siendo densa y los pesos rápidos siguen ocupando O(D²) por capa y flujo. La esparsidad de las entradas no implica ahorro de almacenamiento ni una aceleración del producto denso. El operador completo de actualización incluye estas transformaciones. No se extrapolan cotas de una matriz fija a toda la dinámica.

## Comprobaciones y piloto

Las pruebas cubren ecuaciones manuales, signos, empates, cero, entradas no finitas, overflow, dimensiones incompatibles, strides, gradientes, consultas y ambas escrituras. La paridad del control completo comprende CPU y CUDA, FP32 y FP64, parámetros, gradientes, RNG, recuperación y cuatro actualizaciones AdamW con fixtures. Las [comprobaciones publicadas](../../reports/research/key-competition-20261020/README.md) registran cobertura, CRAP y mutaciones dirigidas con sus denominadores. No sustituyen a una comparación predictiva.

La hipótesis principal es que centrar y seleccionar claves puede reducir interferencia frente al núcleo original. Se contrasta también con la selección fija del mismo tamaño para separar el efecto de competencia del de reducir coordenadas. Un resultado nulo o negativo descarta su promoción. El coste se mide primero con entradas reales precodificadas, pesos iniciales y sin optimizador, usando el ejecutor de costes existente y este catálogo. El test final permanece cerrado y la campaña histórica no se modifica.
