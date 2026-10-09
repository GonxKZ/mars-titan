# Fuentes de los mecanismos CM-v1

La revisión utilizada está fijada a [`adc7f1241b42e322a6451854ab7e4b4c146bf78a`](https://github.com/openai/math/tree/adc7f1241b42e322a6451854ab7e4b4c146bf78a), del 6 de octubre de 2026. La lectura inicial se realizó el día 7. Se inspeccionaron definiciones, enunciados superiores e introducciones de los manuscritos C y M. No se ejecutó Lean ni se revisó toda la cadena de demostración.

El 9 de octubre se observó la revisión [`fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb`](https://github.com/openai/math/tree/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb), del día 8. Los cuatro PDF y los enunciados C/M seleccionados conservan sus bytes. Cambian `formalization.yaml` y el README de comparadores. Esa comparación parcial no acredita que todas las dependencias sean iguales. Se conserva el pin anterior. El [registro de capturas y hashes](../../../reports/engineering/cm-source-audit-20261009.json) identifica ambas revisiones y los archivos comprobados.

## Fuentes S1–S8

| Fuente | Revisión y lectura | Alcance utilizado |
| --- | --- | --- |
| [S1, familia 325](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/325.md) | Pin anterior, comparador y definiciones de `DirectCrouzeix`, declaraciones superiores y dos manuscritos. | Desigualdad completa declarada. No hay reproducción formal local. |
| [S2, familia 125](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/125.md) | Pin anterior, `MetricKMedian`, su modelo y declaración `main_finite_work`, dos manuscritos. | Distinguir el algoritmo publicado de la heurística implementada. |
| [S3, README Lean](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/README.md), [catálogo](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/formalization.yaml) y [comparadores](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/ComparatorChallenges/README.md) | Archivos fijados y contraste con la revisión del 8 de octubre. | Configuración y correspondencias publicadas, sin ejecutar sus herramientas. |
| [S4, Crouzeix–Palencia](https://arxiv.org/abs/1702.00668v1) | arXiv v1, 2 de febrero de 2017. | Antecedente con constante completa `1+sqrt(2)`. |
| [S5, validación Lean](https://lean-lang.org/doc/reference/latest/ValidatingProofs/) | Captura de `latest` con fecha y hash. Contiene un enlace versionado a `4.35.0-rc4`. | Distinguir enunciado, axiomas, kernel y comprobadores externos. No atribuir detalles nuevos a Lean 4.34.1. |
| [S6, familia 140](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/140.md) | Solo documento de alcance, idéntico en los dos pins comparados. | Límites de memoria y muestras bajo hipótesis gaussianas restringidas. |
| [S7, familia 360](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/docs/360.md) | Solo documento de alcance, idéntico en los dos pins comparados. | Transporte óptimo bajo condiciones geométricas. Fuera de CM-v1. |
| [S8, desigualdades del radio numérico](https://arxiv.org/html/2305.17657v1) | arXiv v1, introducción, expresiones (1.1) y (1.2). | La cota de potencias se deduce de propiedades anteriores, sin necesitar S1. |

En S8, la cabecera identifica v1 del 28 de mayo de 2023 y el HTML imprime también 24 de agosto de 2026. Se registra la discrepancia sin atribuirle una causa. La derivación utilizada está en [alcance matemático](mathematical_scope.md).

## Correspondencia inspeccionada

S1 enlaza [A direct proof of the complete Crouzeix inequality](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/A-direct-proof-of-the-complete-Crouzeix-inequality-September-26-2026/paper.pdf), del 26 de septiembre de 2026. Su teorema 1.1 declara la constante 2 para polinomios con coeficientes matriciales. El [comparador](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/ComparatorChallenges/DirectCrouzeix.lean) activa la norma euclídea de operador, usa vectores unitarios y exige dimensiones positivas. No supone normalidad. La solución conserva esas definiciones en `Model.lean` y termina el argumento superior en `CompleteBound.lean`. Esta inspección no valida todos sus imports ni la elaboración efectiva del enunciado.

El segundo manuscrito, [The complete Crouzeix theorem: optimal similarity and a common positive boundary representation](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-complete-Crouzeix-theorem-September-23-2026/paper.pdf), construye una métrica para una matriz y un dominio bajo sus hipótesis. No acredita una métrica común para una sucesión arbitraria de matrices. CM-v1 no depende de esos resultados para calcular la rejilla angular ni reclama estabilidad global del predictor.

Como antecedente, Michel Crouzeix y César Palencia establecen la constante `1+sqrt(2)` en [The numerical range as a spectral set](https://arxiv.org/abs/1702.00668), publicado inicialmente en arXiv el 2 de febrero de 2017. Cambiar una constante teórica en una expresión no modifica los pesos de un modelo. Tampoco demuestra estabilidad de una secuencia de operadores variables.

S2 enlaza [The Approximation Threshold for Metric k-Median](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-Approximation-Threshold-for-Metric-k-Median-September-24-2026/main.pdf), del 24 de septiembre de 2026. Su teorema 1.1 declara una aproximación determinista `1+2/e+ε`, polinómica para ε fijo. El [comparador](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/lean/ComparatorChallenges/MetricKMedian.lean) exige métrica racional finita y `1≤k≤|F|`. La salida debe ser no vacía y estar contenida en los candidatos. Por ello, los valores cero por defecto de `nearest` y `optimum` no convierten una salida vacía en solución válida. La declaración de solución compone programas y un parser hasta `Turing.TM2ComputableInPolyTime`. No se extrajo un ejecutable.

El [manuscrito de recuperación](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Single-Exponential-Recovery-and-Bounded-Price-Strictness-for-Metric-k-Median-September-24-2026/paper.pdf) usa hipótesis adicionales sobre anclajes y proxies. No es el mismo resultado general ni acredita el selector práctico de MARS-TITAN. Sus duplicados y distancias flotantes tampoco heredan automáticamente las hipótesis racionales y separadoras de S2.

El catálogo YAML inspeccionado enumera `KMedianRecovery`, pero no la declaración `MetricKMedian.main_finite_work`. Sí existen la configuración de ese comparador y el módulo de solución. Esta diferencia de inventario no demuestra inexistencia ni invalidez del resultado.

## Verificación formal y niveles de evidencia

El toolchain publicado fija Lean `v4.34.1` y Mathlib `d13f23b723b8a846827a245b89c10fc7d3f11612`. Las configuraciones de los dos comparadores permiten `propext`, `Quot.sound` y `Classical.choice`, axiomas estándar, y mantienen `enable_nanoda: false`. Los `sorry` de los archivos de desafío no son las pruebas de la solución. La búsqueda estática previa en fuentes OAI no sustituye la lista transitiva de axiomas elaborados.

La guía S5 distingue aceptación del kernel, `#print axioms`, comprobación con `lean4checker` y comparación externa del enunciado. La versión `latest` describe la retirada de `Lean.trustCompiler` en 4.35.0. Ese detalle no se atribuye al compilador fijado 4.34.1. `lean`, `lake` y `elan` no aparecen en el PATH comprobado. No se instalaron, compilaron ni ejecutaron. Tampoco se ejecutó un comprobador externo ni se modificó la configuración del sistema.

| Nivel solicitado | Estado real |
| --- | --- |
| 1. El repositorio declara una formalización | Inspeccionado en los documentos de alcance y comparadores fijados. |
| 2. Enunciado formal y correspondencia con el manuscrito | Inspección estática selectiva de definiciones, hipótesis y declaraciones superiores. No auditoría completa de las demostraciones. |
| 3. Comprobación con la configuración publicada | No reproducida. No hay salida de Lean ni lista transitiva de axiomas obtenida localmente. |
| 4. Hipótesis satisfechas en el módulo concreto | No se acreditan las garantías S1/S2 para los módulos actuales. C diagnostica un Jacobiano comprimido y M usa selección práctica de episodios. |
| 5. Validación numérica de la implementación | Pruebas técnicas CPU/CUDA y referencias pequeñas descritas en [resultados](results.md). No son certificados de redondeo ni de estabilidad global. |
| 6. Efecto experimental en MARS-TITAN | No evaluado. No hay mejora predictiva medida y sigue activo el bloqueo histórico. |

S6 supone observaciones gaussianas exactas, una señal unitaria y memoria finita en bits. No proporciona una cota de muestras para este predictor financiero. S7 estudia transporte óptimo en variedades compactas con curvatura MTW débil y condiciones sobre densidades. No introduce un algoritmo de alineación multimodal para CM-v1. No se revisaron sus manuscritos o soluciones y esa extensión permanece fuera del contraste.

## Bibliotecas y atribución

C utiliza operaciones ya disponibles en PyTorch, en particular [`torch.linalg.eigvalsh`](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigvalsh.html), autograd y normas matriciales. M utiliza NumPy y [`numpy.hypot`](https://numpy.org/doc/stable/reference/generated/numpy.hypot.html) para evitar formar cuadrados intermedios en la distancia euclídea. Se revisaron las licencias primarias de [PyTorch 2.14.0](https://github.com/pytorch/pytorch/blob/v2.14.0/LICENSE) y [NumPy 2.5.3](https://github.com/numpy/numpy/blob/v2.5.3/LICENSE.txt). Son dependencias utilizadas por sus APIs. No se copió código de esas bibliotecas, de los manuscritos ni de un paquete de k-medoids.

Las afirmaciones comprobadas por código son las de las pruebas técnicas y sus límites descritos en [resultados](results.md). La equivalencia con un algoritmo publicado, la reproducción formal y la eficacia experimental son comprobaciones diferentes.
