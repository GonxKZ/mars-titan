# Escala de la salida de MAC y memoria residual con LayerNorm

La PR #387 observó en fixtures que la salida de MAC, `y ⊙ M(y)`, quedaba entre 10⁻⁵ y 10⁻² y que el gradiente de la cabeza bajaba de 0,030 a 4,7·10⁻⁴ a lo largo de 4.096 observaciones, aun con el bias declarado de las puertas. Este documento contrasta esa observación con las dos versiones del artículo, describe el componente añadido y conserva sus medidas. No hay entrenamiento ni selección con datos.

## Qué dicen las fuentes

Se revisaron las actas de [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf) y el [preprint arXiv 2501.00663](https://arxiv.org/abs/2501.00663v1).

| Elemento | Fuente | Estado en el proyecto |
| --- | --- | --- |
| Combinación de MAC `o_t = y_t ⊗ M(y_t)` | Ecuación (10) de las actas y (25) del preprint | Implementada como producto de Hadamard sin normalización ni no linealidad adicional |
| Memoria `M(x) = x + LN(W₁σ(W₂x))`, con residual y LayerNorm | Sección 3.3 de las actas | Omitida en v1. Componente desactivable, activo en la receta de la campaña desde 2000 |
| «⊗ puede ser cualquier puerta no lineal», con salidas normalizadas por pesos vectoriales aprendidos y una no linealidad | Sección de MAG, ecuación (13) de las actas y (28) del preprint | No implementada |
| Normalización y puerta con una capa lineal antes de la proyección final | Detalles de arquitectura de ambas versiones | No implementada |
| Expansión 4 de la memoria, convolución tras q, k y v, residual en todos los bloques | Detalles de arquitectura | No implementadas |

La normalización de `⊗` aparece en la descripción de MAG. Aplicarla a MAC sería una interpretación del proyecto, no una lectura directa del artículo. La memoria con residual y LayerNorm sí corresponde a la memoria que comparten las tres variantes de las actas.

## Diagnóstico

Sin residual, la lectura `M(y)` depende solo de los pesos rápidos. Con el bias de las puertas, la memoria no colapsa, pero la norma de esos pesos sigue el equilibrio entre olvido y escritura, de 8 a 0,3 en el flujo fusionado. La lectura cae con ella, de 0,09 a 9·10⁻⁴, y el producto `y ⊙ M(y)` multiplica dos magnitudes pequeñas. El gradiente que llega a la cabeza y a las proyecciones de la memoria baja en la misma proporción. La omisión de la sección 3.3 explica por tanto la escala observada. No hace falta suponer otra causa.

Con `M(x) = x + LN(MLP(x))`, la LayerNorm sin afinidad fija la norma del término aprendido en torno a √D y el residual conserva la consulta. La lectura deja de depender de la norma de los pesos rápidos.

## Componente añadido

`MemoryConfig.residual_layer_norm` y `FinancialConfig.memory_residual_layer_norm` valen `False` por defecto. Con `True`, `NeuralMemory._apply_memory` devuelve `x + layer_norm(MLP(x))` con `eps = 1e-5`, sin pesos ni bias de normalización. La misma función interviene en la lectura y en la pérdida asociativa de la escritura, como en el artículo. La identidad añade `memory_function` con la forma, la fuente, el épsilon y la ausencia de afinidad, y marca `residual` y `layer_norm`. Conserva expansión 1, así que la desviación respecto a la expansión 4 sigue declarada.

Desactivado, el código produce las mismas huellas de configuración, salidas, estados y escrituras que develop en v1 y con `gate_bias`. Las pruebas de [`test_memory_residual_norm.py`](../../tests/models/titans/test_memory_residual_norm.py) comprueban esa paridad con huellas capturadas antes del cambio, la ecuación de lectura, la actualización frente a la ecuación (3) calculada a mano con la pérdida residual, gradcheck en FP64, que una memoria nula lee la consulta, la causalidad y aislamiento de los flujos en MAC y el emparejamiento de los cuatro controles.

## Medidas en fixtures

[`benchmarks/titans_mac_output_scale.py`](../../benchmarks/titans_mac_output_scale.py) reutiliza los flujos de entrada y la derivada exterior de la medida de retención. Usa FP64, D = 64, cuatro flujos y 4.096 observaciones en CPU, sin optimizador. El [recibo](../../reports/engineering/titans-mac-output-scale-20261009.json) conserva todos los puntos.

| Flujo | Memoria | ‖y ⊙ M(y)‖ al inicio | ‖y ⊙ M(y)‖ tras 4.096 | Gradiente de la cabeza, ventanas que terminan en 64 y 4.096 |
| --- | --- | --- | --- | --- |
| Fusionado | `gate_bias` | 1,7·10⁻³ | 1,9·10⁻⁵ | 0,030 → 4,7·10⁻⁴ |
| Fusionado | `gate_bias` + residual y LN | 0,60 | 0,61 | 7,8 → 10,3 |
| Varianza unidad | `gate_bias` | 0,021 | 3,5·10⁻⁴ | 0,12 → 5,3·10⁻³ |
| Varianza unidad | `gate_bias` + residual y LN | 0,77 | 0,81 | 10,3 → 12,5 |
| Recurrente | `gate_bias` | 0,021 | 3,2·10⁻³ | 0,22 → 0,029 |
| Recurrente | `gate_bias` + residual y LN | 0,79 | 0,92 | 9,2 → 9,1 |

La primera fila reproduce la observación de la PR #387. Con residual y LayerNorm la salida mantiene su escala y el gradiente de la proyección de `α` se sitúa en torno a 2·10⁻³, estable a lo largo del flujo. La norma de los pesos rápidos baja de 8 a cerca de 4,5 y no se anula.

La combinación de residual y LayerNorm sin `gate_bias` no es aceptable. La lectura tampoco se anula, pero los gradientes de la proyección de `α` llegan a 118 y los de la consulta a 6,4·10³, con normas de pesos erráticas. Sin bias, la memoria colapsa hacia cero y la LayerNorm amplifica un término cada vez más pequeño.

## Decisión y límites

La propuesta de adoptar `memory_residual_layer_norm: true` junto a `gate_bias` en la campaña desde 2000 quedó decidida en [#27](https://github.com/GonxKZ/mars-titan/issues/27). Como cambia la arquitectura de la memoria, se declaró en una receta nueva, [`chronological-training-historical-masked.json`](../../configs/titans/chronological-training-historical-masked.json), con la memoria residual en los cuatro brazos. Las recetas v1 (`chronological-training.json` y `chronological-training-quantile.json`) conservan la memoria v1 con `gate_bias`, su contenido y sus identidades.

Con residual y LayerNorm, el gradiente de la cabeza al inicio es unas trescientas veces mayor que con v1. La tasa de aprendizaje se declaró con v1, así que la receta de campaña la busca entre dos casos, 10⁻⁴ y 10⁻³, con el mismo número de casos, presupuesto y selección por validación que las referencias neuronales y sin usar el test. El recorte se mantiene en 1,0. La justificación de la rejilla está en el [entrenador cronológico](titans-chronological-trainer.md#receta-de-la-campaña-y-casos-de-búsqueda).

Son medidas técnicas con entradas aleatorias. No indican qué memoria predice mejor ni sustituyen la comparación con datos.

La misma medida se repitió en `cuda:0` el 9 de octubre ([recibo](../../reports/engineering/cuda-checks-20261009/titans-mac-output-scale-cuda.json), [resumen](../../reports/engineering/cuda-checks-20261009/README.md)). Tres ejecuciones CUDA dan los 612 valores idénticos bit a bit. Frente a CPU, en FP64, las escalas de `gate_bias` coinciden hasta 4·10⁻¹⁶ en todo el flujo. Con residual y LayerNorm las trayectorias se separan con el número de pasos. Con `gate_bias` la mayor diferencia relativa de las escalas es 4·10⁻¹⁵ hasta 256 pasos, 7·10⁻¹⁰ en 1.024, 1,7 % en 2.048 y 4,2 % en 4.096, y los gradientes exteriores difieren hasta un 5 %. Sin `gate_bias` la diferencia llega al 31 % en 64 pasos. En estos fixtures la memoria con residual amplifica el redondeo. Por eso no conviene esperar que una ejecución de la receta de campaña en CPU y otra en CUDA sigan la misma trayectoria, aunque partan del mismo estado.
