# Campaña de entrenamiento sobre la edición histórica desde 2000

Revisión del 9 de octubre de 2026. Este documento ordena el recorrido completo desde los datos hasta la comparación final. Describe un plan y el estado de cada etapa. No contiene resultados predictivos: ningún modelo se ha entrenado sobre esta edición y el test de 2024 sigue cerrado.

## Objetivo

Entrenar desde cero todas las arquitecturas comparadas con todos los datos utilizables desde 2000, con las mismas filas, la misma validación temporal y presupuestos comparables. Después se aplican postentrenamientos y políticas de refuerzo sobre esas mismas bases. La pregunta es cuánto cambia el error de cada variante frente a sus referencias, con incertidumbre temporal y costes medidos.

El bloqueo de aprendizaje sigue vigente hasta que la edición esté completa y verificada. Su condición es preparar la edición desde 2000 con ausencias y máscaras explícitas y la misma población para todos los modelos.

## Etapas y estado

| Etapa | Contenido | Estado a 9 de octubre | Tarea |
| --- | --- | --- | --- |
| 1. Edición de entradas | Codificación v3 de precios, noticias, gráficos, fundamentales y 140 posiciones macro con nivel, presencia y antigüedad, más cinco bits de presencia por modalidad | En curso. Más de 750 de unos 5.028 activos elegibles confirmados. El resto se codifica en paralelo con paridad bit a bit comprobada | [#171](https://github.com/GonxKZ/mars-titan/issues/171) |
| 2. Objetivos | Retorno residual apertura-cierre de la sesión siguiente, OLS con 252 sesiones y un mínimo de 126 pares, solo con pares disponibles en la decisión | Código existente. Exige la población completa, así que espera a la etapa 1 | [#15](https://github.com/GonxKZ/mars-titan/issues/15) |
| 3. Verificación de la edición | Conciliación de los 5.676 candidatos, recuentos frente al censo de ventanas, máscaras, disponibilidad no posterior a la decisión, ninguna fila de 2024 | Verificador por prefijos existente. Falta la pasada completa | [#171](https://github.com/GonxKZ/mars-titan/issues/171) |
| 4. Protocolo temporal | Ventanas anuales expansivas desde el primer año con etiquetas, validación interna al final de cada tramo de entrenamiento, evaluación del año siguiente, purga por intervalo real de etiquetas | Diseño y comprobaciones técnicas en el [protocolo v2](walk-forward-2000.md). Falta decidir el presupuesto y preparar las vistas reales | [#363](https://github.com/GonxKZ/mars-titan/issues/363) |
| 5. Entrenamiento base | Referencias, GRU episódica, Transformer compacto, núcleo Titans-MAC, MARS-TITAN con ampliaciones y CM-v1 | Runners en adaptación a la política con máscaras. El entrenador cronológico de Titans está en implementación | [#234](https://github.com/GonxKZ/mars-titan/issues/234), [#23](https://github.com/GonxKZ/mars-titan/issues/23), [#293](https://github.com/GonxKZ/mars-titan/issues/293) |
| 6. Postentrenamiento | Padre congelado, continuación supervisada, corrección residual y adaptadores, solos y combinados | Implementado sobre la edición estricta. Falta la edición con máscaras y la matriz de combinaciones | [#364](https://github.com/GonxKZ/mars-titan/issues/364), [#128](https://github.com/GonxKZ/mars-titan/issues/128) |
| 7. Refuerzo | Variantes de PPO, KLPO prioritario y Double DQN sobre entornos auditados | Controladores implementados sin ejecutar pasos. Auditoría de entornos en curso | [#137](https://github.com/GonxKZ/mars-titan/issues/137), [#365](https://github.com/GonxKZ/mars-titan/issues/365) |
| 8. Evaluación | MAE residual por sesión y métricas secundarias con incertidumbre por bloques | Métricas principales y secundarias implementadas con pruebas técnicas, sin aplicar a esta edición | [#32](https://github.com/GonxKZ/mars-titan/issues/32), [#36](https://github.com/GonxKZ/mars-titan/issues/36) |

## Población y equidad

Todas las arquitecturas leen las mismas filas de la edición. Una fila existe aunque falten noticias, fundamentales o macro, porque la ausencia se marca y se rellena con cero de forma explícita. Ninguna variante puede descartar filas difíciles para mejorar su promedio.

Los bits de presencia forman parte de la entrada de todas las familias. Hoy solo los reciben Titans y la GRU candidata, así que las referencias neuronales y tabulares se están adaptando antes de cualquier comparación. Sin ese cambio, una diferencia de error podría deberse a que un modelo distingue un cero real de una ausencia y otro no.

La comparación estricta, que exige las cuatro modalidades completas y los 140 indicadores, se conserva como control separado. No se mezclan sus filas con las de la edición desde 2000.

## Validación temporal y sobreajuste

Cada ventana entrena con todo el pasado disponible hasta su corte, valida en el tramo final de ese pasado y evalúa el año siguiente. La purga elimina las filas cuya etiqueta madura después del corte. China no tiene etiquetas residuales antes de 2006 porque sus precios y el factor CSI300 empiezan ese año. Por eso las ventanas conjuntas empiezan en 2011, el primer año con tres años de etiquetas maduras en los dos mercados, y ninguna ventana admite un mercado vacío. Las ventanas solo de US evalúan desde 2005 y siguen entrenando con todo el pasado disponible. El [protocolo v2](walk-forward-2000.md) justifica la decisión.

La selección guarda el mejor estado según la validación temporal, con presupuesto fijo de 30 épocas, paciencia y mejora mínima declaradas en el protocolo. Los controles emparejados conservan el mismo número de actualizaciones. Si una parada independiente rompiera esa igualdad, se usa selección del mejor estado con presupuesto fijo. Los checkpoints de recuperación rotan con un límite pequeño y el mejor estado se guarda aparte, según [la política de checkpoints](../engineering/checkpoint-recovery.md).

El test de 2024 no participa en ninguna selección. Se abrirá una sola vez, con la configuración fijada, al final de la campaña.

## Familias entrenadas

| Familia | Identidad | Qué aísla |
| --- | --- | --- |
| Referencias | RNN, LSTM, GRU, DLinear, Transformer compacto, Ridge y XGBoost | Nivel de error sin memoria persistente |
| GRU episódica | Candidato con banco 128×256, referencia independiente | Memoria episódica sobre un codificador recurrente |
| Núcleo Titans-MAC | `transformer_direct`, `mac_disabled`, `mac_frozen`, `mac_online` | Cambio de codificador frente a memoria neuronal |
| MARS-TITAN con ampliaciones | Banco episódico, escritura M0 a M3, refinamientos K=1, 2 y 4, y las modificaciones del [documento de integración](system-integration.md), cada una desactivable | Aportación de cada ampliación, una cada vez |
| CM-v1 | B, B+C, B+M y B+C+M sobre la B elegida por protocolo | Control del radio numérico y consolidación, según [su especificación](../experiments/mars_titan_cm_v1/specification.md) |

No se ejecuta el producto cartesiano de todas las ampliaciones. Primero se fija la base y después se estudia un mecanismo cada vez, como establece el [documento de integración](system-integration.md).

## Postentrenamiento

Cada postentrenamiento parte de un padre seleccionado en la misma ventana y se compara con ese padre congelado. Los adaptadores se colocan solos y en combinaciones de uno, dos o tres puntos de inserción, con el mismo presupuesto de actualizaciones y la misma validación. La corrección residual inicializada a cero se compara con la salida continua del padre, no solo con la mediana de una rejilla. Los objetivos ya derivados se describen en [adaptación predictiva](predictive-adaptation.md).

## Refuerzo

Los entornos consumen únicamente predicciones fuera de muestra del walk-forward. La ejecución usa el precio posterior a la decisión, con costes y deslizamiento declarados y límites de posición. Un agente escrito a mano que intente leer información futura debe fallar o no obtener ventaja. Los resultados sobre entornos sintéticos no se presentan como resultados sobre FinMultiTime. El controlador KLPO terminal se describe en [su documento de ingeniería](../engineering/terminal-klpo-updates.md).

## Métricas

La métrica principal es el MAE residual por sesión. Primero se promedian los activos de un mismo mercado e instante y después se aplica la ponderación temporal y entre mercados. Se registran también MSE y RMSE, acierto de dirección con convención de empates, correlación de rangos por sesión y, cuando la salida lo permita, pérdida pinball y cobertura de cuantiles. La diferencia frente a una referencia se informa como Delta_error = MAE_variante − MAE_base y como porcentaje 100·(MAE_base − MAE_variante)/MAE_base, con intervalos del 95 % por bloques temporales. El detalle está en [métricas](metrics.md).

## Cómputo

La campaña se ejecuta en una RTX 4070 Laptop de 8 GB con el perfil de energía de ahorro, que el equipo necesita para no apagarse por temperatura. En esas condiciones la GPU trabaja a unos 1.305 MHz con limitación térmica. La auditoría de preparación estima unas 85 h por familia, configuración y semilla si se reentrena cada ventana anual completa con 30 épocas. Es una hipótesis basada en caudales de ediciones anteriores. El presupuesto definitivo se fijará con el caudal medido en la primera ventana, antes de lanzar la campaña, y será el mismo para los brazos emparejados.

## Decisiones pendientes antes de entrenar

- Número de ventanas con reentrenamiento completo y tratamiento de las restantes, entre las [alternativas de presupuesto](walk-forward-2000.md#coste-y-alternativas-de-presupuesto) ([#363](https://github.com/GonxKZ/mars-titan/issues/363)).
- Cabeza de cuantiles común o calibración solo secundaria ([#22](https://github.com/GonxKZ/mars-titan/issues/22)), con una [propuesta registrada](quantile-head-decision.md).
- Semillas fijas y margen mínimo relevante de error, registrados antes de ver resultados.
- Política de retención de predicciones y checkpoints según el disco disponible.
