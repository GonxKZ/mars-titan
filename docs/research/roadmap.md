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

## Estado a 14 de octubre de 2026

La tabla resume la evidencia técnica integrada en `develop` hasta `80598cd9` para la [campaña desde 2000](training-campaign-2000.md). Ningún hito está cerrado. La edición histórica v3.1 y sus objetivos residuales están generados y verificados, y la campaña elegida es A v2, con un modelo conjunto US+CN. Los criterios de aceptación exigen ejecutar entrenamientos y evaluaciones que todavía no se han lanzado y que la protección del aprendizaje sigue deteniendo.

| Hito | Evidencia técnica integrada | Pendiente para su criterio de cierre |
| --- | --- | --- |
| O1. Datos multimodales | [Edición v3.1 con máscaras](../data/edition-v3-1.md#ejecución-del-10-y-11-de-octubre), generada y verificada el 11 de octubre ([recibo](../../reports/data/edition-v3-1-verification-20261011.json)): 5.676 candidatos, 5.023 activos codificados y 18.698.976 muestras, con fundamentales en el 32,5 % de las filas y noticias en el 16,9 %. Conversión exacta de los precios, filas recuperadas por redondeo, ausencias de todo el mercado con máscara y vectores en FP32 estricto. [Precios negociados reconstruidos](../data/unadjusted-prices.md) y declaración del sesgo de supervivencia en el [protocolo](protocol.md#universo-y-selección-del-subconjunto) | Vistas de la campaña A v2 sobre la v3.1 y acompañar los resultados con el sesgo de supervivencia declarado |
| O2. Retornos residuales | Objetivos `targets-v3.1` generados y verificados: 17.142.187 filas de entrenamiento y 1.235.785 de validación, 321.004 exclusiones con motivo y un recálculo independiente de 223.459 etiquetas sin diferencias. Factores de mercado versionados. [Diagnóstico de la etiqueta](../../reports/targets/diagnostics.md) sobre la v3 y [repetición sobre la v3.1](../../reports/targets/diagnostics.md#repetición-sobre-la-v31), con las 18.377.972 etiquetas recalculadas bit a bit, bruto y residual sobre las mismas filas, exposición restante, tres sensibilidades declaradas y desglose por patrón de máscaras. Se conserva la ventana de 252 sesiones sin recortar extremos | Los factores de mercado no tienen versiones contemporáneas verificadas |
| O3. Memoria adaptativa | Núcleo Titans-MAC con [puertas con bias](../engineering/titans-gate-initialization.md) y memoria con residual, [entrenador cronológico](../engineering/titans-chronological-trainer.md), [MARS-TITAN con ampliaciones](../engineering/mars-titan-extensions.md) sobre el padre `mac_online` y [factorial de CM-v1](../experiments/mars_titan_cm_v1/factorial.md) con C como penalización y M como retención con centros fijos | [Comprobaciones CUDA](../../reports/engineering/cuda-checks-20261009/README.md) sin pasos de optimizador: 20 de 21 pasan en la repetición y M2 pasa con el enlace de `develop` tras [#418](https://github.com/GonxKZ/mars-titan/pull/418). [M3](../engineering/m3-write-policy.md) definida, con productor y comprobada en CPU y `cuda:0`. B6 emite en las ventanas walk-forward y la representación de régimen de la campaña es su [enrutamiento por un régimen observable](../engineering/mars-titan-extensions.md#enrutamiento-de-b6-por-régimen), con control de calendario y comparación declarada (#20). Seis ampliaciones siguen sin conexión, entre ellas el HMM filtrado, y no se ha ejecutado ningún ajuste |
| O4. Referencias y ablaciones | [Referencias con máscaras](../engineering/masked-reference-runners.md), [comparadores tabulares](../engineering/masked-tabular-comparators.md), [GRU episódica por ventana](../engineering/candidate-chronological-trainer.md) y [matriz de adaptadores](../engineering/masked-posttraining.md). La [campaña A v2](training-campaign-2000.md#campaña-a-v2-con-modelo-conjunto) declara 2.873 ajustes con las referencias, Titans-MAC, la GRU candidata, MARS-TITAN (incluidos `mars_titan_m1_k4_first_read` y los seis B6) y CM-v1, el mismo presupuesto de búsqueda para todas las familias neuronales, las opciones de memoria medidas y el control en línea `transformer_compact_online` | Repetir sobre la v3.1 la medida de memoria que fija esas opciones y ejecutar |
| O5. Evaluación | [Protocolo walk-forward v2](walk-forward-2000.md) y [protocolo conjunto v3](walk-forward-2000.md#protocolo-conjunto-v3-de-la-campaña-a-v2), [walk-forward por etapas](walk-forward-2000.md#walk-forward-por-etapas-de-la-campaña-a-v2), [métricas por sesión y CQR común](metrics.md), [estratos descriptivos por presencia de modalidades](metrics.md#estratos-por-presencia-de-modalidades) y [etapa de políticas por ventana](training-campaign-2000.md#etapa-de-políticas-por-ventana) sobre [entornos auditados](../engineering/rl-environment-integrity.md), con [ejecutores nativos sobre cintas reconstruidas](../engineering/native-policy-real-tapes.md) comprobados sin aprendizaje y [compilados con LibTorch CUDA](../../reports/engineering/rl-stage-native-20261009/README.md) | Terminar y verificar las vistas de A v2 sobre la v3.1, predicciones reales, medida del coste de la actualización RL y apertura única del test de 2024 |
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
