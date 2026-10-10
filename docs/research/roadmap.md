# Objetivos, hitos y criterios de cierre

El trabajo se organiza en seis objetivos relacionados entre sí. El contrato de datos y el protocolo permiten empezar por referencias sencillas mientras se estudia la memoria. La comparación se cierra antes de redactar conclusiones.

| Hito | Entregable | Criterios de aceptación | Dependencia principal |
| --- | --- | --- | --- |
| O1. Datos multimodales | Manifiesto, ficha de datos, tabla experimental y auditoría temporal. | Procedencia y disponibilidad por modalidad. Exclusiones justificadas. Cobertura real. Prueba contra contaminación futura y revisión de licencias. | Requisitos, acceso y protocolo. |
| O2. Retornos residuales | Definición de etiqueta y residualizador reproducible. | Fechas de entrada/salida precisas. Coeficientes con historia permitida. Comparación bruto/residual. Sensibilidad al mercado y sector si verificable. | O1 y calendario de decisión. |
| O3. Memoria adaptativa | Núcleo Titans-MAC identificable y ampliaciones separadas. | Ecuaciones, gradientes y estados comprobados. Sorpresa asociativa y error financiero maduro separados. Recuperación y comprobación CUDA. | O1/O2 y referencias comparables. |
| O4. Referencias y ablaciones | GRU episódica, Transformer compacto, Titans-MAC y ampliaciones, además de controles sencillos. | Mismas filas, objetivo, cortes y búsqueda registrada. Separar codificador, memoria neuronal, banco episódico y CM-v1. | O1/O2. O3 para las ablaciones finales. |
| O5. Evaluación | Predicciones fuera de muestra e informe verificable. | Walk-forward purgado, calibración separada, costes completos, intervalos por bloques, test final protegido y todas las ejecuciones registradas. | Protocolo fijado y O4. |
| O6. Interpretación e informe | Discusión, limitaciones, conclusiones y presentación de resultados. | Responder O1–O6 con evidencias, incluidos resultados negativos. Revisar referencias y explicar métodos, figuras y limitaciones. | Evidencias de O1–O5. Redacción paralela. |

## Orden práctico

```mermaid
flowchart LR
    R[Requisitos y protocolo] --> D[O1 · Datos]
    D --> T[O2 · Objetivo]
    T --> B[O4 · Referencias iniciales]
    B --> M[O3 · Memoria]
    M --> A[O4 · Ablaciones]
    A --> E[O5 · Evaluación cerrada]
    E --> C[O6 · Conclusiones y documentación]
    R -. escritura continua .-> C
```

Se trabajará en unidades pequeñas y revisables. El [tablero](task-board.md) desarrolla las tareas, dependencias y prioridades. Las fechas se fijarán cuando se concrete el calendario del proyecto. Las duraciones estimadas no son fechas de entrega confirmadas.

## Estado a 9 de octubre de 2026

La tabla resume la evidencia técnica integrada en `develop` hasta `77bf6b9e` para la [campaña desde 2000](training-campaign-2000.md). Ningún hito está cerrado. La edición histórica, sus objetivos residuales y las vistas de la campaña A ya están verificados y se ha elegido la variante A con todas las familias. Los criterios de aceptación exigen ejecutar entrenamientos y evaluaciones que todavía no se han lanzado y que la protección del aprendizaje sigue deteniendo.

| Hito | Evidencia técnica integrada | Pendiente para su criterio de cierre |
| --- | --- | --- |
| O1. Datos multimodales | [Edición v3 con máscaras completa y verificada](../../reports/data/historical-edition-v3-targets-20261009.json): 5.676 candidatos, 5.023 activos codificados y 17.076.024 muestras, con fundamentales en el 34,4 % de las filas y noticias en el 17,9 %. [Precios negociados reconstruidos](../data/unadjusted-prices.md) y declaración del sesgo de supervivencia en el [protocolo](protocol.md#universo-y-selección-del-subconjunto) | Acompañar los resultados con el sesgo de supervivencia declarado |
| O2. Retornos residuales | Objetivos de la edición v3 generados y verificados: 15.560.197 filas de entrenamiento y 1.221.822 de validación, 294.005 exclusiones con motivo y un recálculo independiente de 190.830 etiquetas sin diferencias ([recibo](../../reports/data/historical-edition-v3-targets-20261009.json)). Factores de mercado versionados | Comparar bruto y residual. Los factores de mercado no tienen versiones contemporáneas verificadas |
| O3. Memoria adaptativa | Núcleo Titans-MAC con [puertas con bias](../engineering/titans-gate-initialization.md) y memoria con residual, [entrenador cronológico](../engineering/titans-chronological-trainer.md) [MARS-TITAN con ampliaciones](../engineering/mars-titan-extensions.md) sobre el padre `mac_online` y [factorial de CM-v1](../experiments/mars_titan_cm_v1/factorial.md) con C como penalización y M como retención con centros fijos | [Comprobaciones CUDA](../../reports/engineering/cuda-checks-20261009/README.md) sin pasos de optimizador: 20 de 21 pasan en la repetición y M2 pasa con el enlace de `develop` tras [#418](https://github.com/GonxKZ/mars-titan/pull/418). [M3](../engineering/m3-write-policy.md) definida, con productor y comprobada en CPU y `cuda:0`. Seis ampliaciones sin conexión, B6 sin emitir por ventanas y ningún ajuste ejecutado |
| O4. Referencias y ablaciones | [Referencias con máscaras](../engineering/masked-reference-runners.md), [comparadores tabulares](../engineering/masked-tabular-comparators.md), [GRU episódica por ventana](../engineering/candidate-chronological-trainer.md) y [matriz de adaptadores](../engineering/masked-posttraining.md), con campañas A y B declaradas. Variante A elegida con una declaración ampliada de 4.680 ajustes | Fijar las opciones de memoria, copiar a la configuración de A la GRU episódica, MARS-TITAN y CM-v1, y ejecutar |
| O5. Evaluación | [Protocolo walk-forward v2](walk-forward-2000.md) con las [vistas de la campaña A](walk-forward-2000.md#vistas-de-la-campaña-a) preparadas y verificadas, [métricas por sesión y CQR común](metrics.md), [estratos descriptivos por presencia de modalidades](metrics.md#estratos-por-presencia-de-modalidades) y [etapa de políticas por ventana](training-campaign-2000.md#etapa-de-políticas-por-ventana) sobre [entornos auditados](../engineering/rl-environment-integrity.md), con [ejecutores nativos sobre cintas reconstruidas](../engineering/native-policy-real-tapes.md) comprobados sin aprendizaje | Predicciones reales, compilación de los ejecutores RL con LibTorch CUDA y medida de su rendimiento, y apertura única del test de 2024 |
| O6. Interpretación e informe | Estado técnico recogido en el README, esta documentación y la memoria de trabajo | Resultados que interpretar |

## Alcance mínimo y extensiones

La [campaña previa de referencias](../engineering/comparison-campaign.md) pasa a
recorrer el universo completo admisible. No se limita a las cifras iniciales de
64 o 128 activos descritas debajo. O1, O2 y las referencias de O4 pueden avanzar
en paralelo con O5, siempre que cada entrenamiento use una instantánea de datos
fijada y validada. El desarrollo del candidato no forma parte de esa campaña.

El mínimo científico es una comparación estadounidense reproducible, con piloto de hasta 64 activos y selección principal propuesta de hasta 128. Incluye un horizonte diario, las cuatro modalidades de disponibilidad justificable, contexto macro, referencias de varias familias y una memoria compacta con ablaciones de componentes. En la comparación estricta, una muestra sin alguna modalidad se excluye. No se sustituye por entrenamiento de dos modalidades. La edición histórica desde 2000 es una comparación adicional que conserva las cuatro posiciones de modalidad con máscaras explícitas y las mismas filas en todos los modelos.

La residualización sectorial, un Transformer adicional, HS300, memorias jerárquicas completas y kernels C++/CUDA son extensiones. Su activación exige que los controles temporales y las referencias funcionen, que exista presupuesto medido y que la comparación principal no quede comprometida.

La [ampliación](research-expansion.md) añade estudio cerebral, macroeconomía, recurrencia y antecedentes recientes, con decisiones de continuidad por componente. El inventario completo y la capacidad de leer por bloques permiten crecer sin cargar todo en RAM. Entrenar el universo completo y transferir a otro mercado son extensiones P2. La recuperación mediante checkpoints se verifica antes de la campaña prolongada.

## Cierre de una tarea

Una tarea se cierra con el artefacto enlazado, su comprobación y las limitaciones que permanecen. Un documento de diseño puede cerrar la tarea de diseño. No cierra implementación, experimento ni validación del objetivo. El tablero y los informes deben mantener esa diferencia.
