# Ampliación de la investigación de MARS-TITAN

Autor: Gonzalo García Lama. Revisión: 18 de septiembre de 2026.

## Decisión de alcance

La investigación incorpora memoria inspirada en mecanismos de aprendizaje, recurrencia interna, contexto macroeconómico y eficiencia de extremo a extremo. Se mantiene la pregunta comparativa y la restricción del equipo: 32 GB de RAM y RTX 4070 Max-Q de 8 GB. El equipo puede permanecer encendido, pero aún no hay fechas de entrega confirmadas.

La decisión inicial es una comparación principal de hasta **128 activos estadounidenses**, precedida por un piloto de hasta 64. La selección se hará con la información del periodo de desarrollo y con una regla reproducible. El número definitivo depende de cobertura temporal válida y del coste medido. Se conservará un histórico suficientemente amplio para estudiar cambios y revisitas de régimen, sin seleccionar los años según los resultados del modelo.

El inventario cubre toda la copia local y la canalización se diseña para recorrerla por bloques. Ampliar a 256 activos, al universo completo o a China son pruebas de escalabilidad y transferencia condicionadas al presupuesto. Reducir la campaña no permite relajar los controles temporales, perder referencias importantes o sustituir repeticiones por una única ejecución favorable.

Esta decisión integra la preferencia inicial por aprovechar todo el dataset y la autorización posterior de recortarlo para atender al tiempo y las entregas. La [propuesta académica original](../academic/proposal.md) se conserva como antecedente. Los cambios de amplitud y sus implicaciones se revisarán con la dirección cuando se concrete el calendario.

## Líneas incorporadas

| Línea | Resultado documental | Evidencia futura necesaria |
| --- | --- | --- |
| Aprendizaje y memoria biológica | [Revisión de cerebro y aprendizaje continuo](../references/brain-review.md). | Comparar retención, adaptación y reaprendizaje, sin equiparar un módulo a una región cerebral. |
| Memoria y recurrencia | [Arquitectura candidata](candidate-architecture.md). | Ablaciones a igual capacidad y presupuesto, con K = 1, 2 y 4. |
| Estado del arte reciente | [Arquitecturas y finanzas recientes](../references/frontier-review.md). | Contrastar con antecedentes directos antes de afirmar una diferencia original. |
| Eficiencia y DeepSeek | [Revisión de sistemas](../references/systems-review.md) y [lectura de DeepSeek](../references/deepseek-review.md). | Medir qué técnicas funcionan en un predictor compacto y una única GPU Ada. |
| Contexto financiero | [Revisión macrofinanciera](../references/macro-review.md) y [catálogo macro](../data/macro-catalog.md). | Obtener datos con versiones y horas defendibles, después probar su contribución por familias. |
| Cobertura y coste | [Recorrido del dataset](../engineering/full-dataset-training.md) y [latencia](../engineering/latency-budget.md). | Medir caudal, coste total, memoria y colas de latencia en el equipo real. |
| Continuidad del trabajo | [Plan de cómputo](../engineering/compute-plan.md) y [recuperación](../engineering/checkpoint-recovery.md). | Reanudar tras interrupciones sin perder ni adelantar el estado temporal. |
| Aportación investigadora | [Registro de hipótesis y antecedentes](novelty-ledger.md) y [revisión adversarial](adversarial-review.md). | Resultados reproducibles que confirmen o descarten las diferencias propuestas. |

## Qué se entiende por calidad y precisión

El objetivo es mejorar predicción y calibración bajo restricciones explícitas. No se presupone que toda variable, recuerdo o iteración aporte información. La memoria debe conservar patrones útiles y seguir siendo capaz de aprender cuando cambian. La fiabilidad incluye saber cuándo abstenerse y detectar errores de datos, no solo producir un número con mucha confianza.

El estado del arte será un conjunto de referencias con tarea, datos y recursos comparables. No se utilizará como una etiqueta de superioridad antes de experimentar. Una aportación propia puede consistir en una mejora pequeña pero estable, una política de memoria mejor justificada o un resultado negativo que descarte una combinación costosa.

## Condición de avance

La [guía de implementación](../engineering/implementation-guide.md) concreta el papel de cada tecnología y el orden de cierre. El catálogo separa núcleo, apoyo y extensiones. Ni el número de referencias, ni la página de seguimiento, ni añadir otro lenguaje cambian la pregunta principal. La comparación conserva los seis objetivos y la propuesta presentada como criterio de alcance.

Antes de ampliar el modelo se debe completar el recorrido temporal básico, una referencia cero y lineal, la estimación de tiempo y la prueba de recuperación. La comparación principal probará una hipótesis de memoria. Recurrencia, replay paramétrico, destilación, atención dispersa y transferencia entre mercados entrarán mediante decisiones explícitas de continuidad o descarte.

El catálogo de lectura es amplio. La cuadrícula de entrenamiento será deliberadamente más pequeña, para que haya recursos para verificar datos, repetir comparaciones y analizar fallos. Las nuevas tareas se integran en el [tablero](task-board.md) sin dar por realizados modelos o experimentos.
