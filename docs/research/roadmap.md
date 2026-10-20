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

## Estado revisado el 20 de octubre de 2026

La tabla recoge código, datos verificados y estudios exploratorios publicados hasta `997c9eecd`. La edición v3.1 y sus objetivos están generados y verificados. Los pilotos de memoria, refinamientos y plasticidad tienen resultados acotados. Ningún hito científico queda cerrado por ellos. La campaña A v2 del universo completo y el test final de 2024 siguen cerrados.

| Hito | Evidencia publicada | Pendiente para su criterio de cierre |
| --- | --- | --- |
| O1. Datos multimodales | [Edición v3.1 con máscaras](../data/edition-v3-1.md#ejecución-del-10-y-11-de-octubre), generada y verificada el 11 de octubre ([recibo](../../reports/data/edition-v3-1-verification-20261011.json)): 5.676 candidatos, 5.023 activos codificados y 18.698.976 muestras, con fundamentales en el 32,5 % de las filas y noticias en el 16,9 %. Conversión exacta de los precios, filas recuperadas por redondeo, ausencias de todo el mercado con máscara y vectores en FP32 estricto. [Precios negociados reconstruidos](../data/unadjusted-prices.md) y declaración del sesgo de supervivencia en el [protocolo](protocol.md#universo-y-selección-del-subconjunto). [Vistas A v2 verificadas](../../reports/data/campaign-a-v2-views-v31-20261014.json) y [disjunción temporal comprobada](../../reports/data/campaign-a-v2-data-review-20261014.json). | Acompañar los futuros resultados con las limitaciones de procedencia y el sesgo de supervivencia declarado |
| O2. Retornos residuales | Objetivos `targets-v3.1` generados y verificados: 17.142.187 filas de entrenamiento y 1.235.785 de validación, 321.004 exclusiones con motivo y un recálculo independiente de 223.459 etiquetas sin diferencias. Factores de mercado versionados. [Diagnóstico de la etiqueta](../../reports/targets/diagnostics.md) sobre la v3 y [repetición sobre la v3.1](../../reports/targets/diagnostics.md#repetición-sobre-la-v31), con las 18.377.972 etiquetas recalculadas bit a bit, bruto y residual sobre las mismas filas, exposición restante, tres sensibilidades declaradas y desglose por patrón de máscaras. Se conserva la ventana de 252 sesiones sin recortar extremos | Los factores de mercado no tienen versiones contemporáneas verificadas |
| O3. Memoria adaptativa | Núcleo Titans-MAC con [puertas con bias](../engineering/titans-gate-initialization.md) y memoria con residual, [entrenador cronológico](../engineering/titans-chronological-trainer.md), [MARS-TITAN con ampliaciones](../engineering/mars-titan-extensions.md) sobre el padre `mac_online` y [factorial de CM-v1](../experiments/mars_titan_cm_v1/factorial.md) con C como penalización y M como retención con centros fijos. La [repetición CUDA del 14 de octubre](../../reports/engineering/cuda-checks-20261014/README.md) cubre los recorridos declarados sin pasos de optimizador. [M3](../engineering/m3-write-policy.md) definida, con productor y comprobada en CPU y `cuda:0`. B6 emite en las ventanas walk-forward y la representación de régimen de la campaña es su [enrutamiento por un régimen observable](../engineering/mars-titan-extensions.md#enrutamiento-de-b6-por-régimen), con control de calendario y comparación declarada (#20). | La comparación confirmatoria de la memoria sigue pendiente. Los [pilotos de memoria](../../reports/research/memory-microstudy-20261018/README.md) y [plasticidad](../../reports/research/neural-plasticity-20261020/development-v2/README.md) sí han ejecutado ajustes acotados. El HMM filtrado y las demás integraciones pendientes requieren su propia comprobación |
| O4. Referencias y ablaciones | [Referencias con máscaras](../engineering/masked-reference-runners.md), [comparadores tabulares](../engineering/masked-tabular-comparators.md), [GRU episódica por ventana](../engineering/candidate-chronological-trainer.md) y [matriz de adaptadores](../engineering/masked-posttraining.md). La [campaña A v2](training-campaign-2000.md#campaña-a-v2-con-modelo-conjunto) declara 3.177 ajustes con las referencias, Titans-MAC, la GRU candidata, MARS-TITAN (incluidos `mars_titan_m1_k4_first_read` y los ocho B6, dos de ellos con regla kalman) y CM-v1, el mismo presupuesto de búsqueda para todas las familias neuronales, las opciones de memoria medidas y el control en línea `transformer_compact_online` | La [ronda de refinamientos](../../reports/research/refinement-development-20261019/completed-v1/README.md) completó 288 tareas con cohortes de 64 y 256 activos y tres semillas. Faltan los ajustes y costes de las familias y recorridos no cubiertos de la campaña A v2 |
| O5. Evaluación | [Protocolo walk-forward v2](walk-forward-2000.md) y [protocolo conjunto v3](walk-forward-2000.md#protocolo-conjunto-v3-de-la-campaña-a-v2), [walk-forward por etapas](walk-forward-2000.md#walk-forward-por-etapas-de-la-campaña-a-v2), [métricas por sesión y CQR común](metrics.md), [estratos descriptivos por presencia de modalidades](metrics.md#estratos-por-presencia-de-modalidades) y [etapa de políticas por ventana](training-campaign-2000.md#etapa-de-políticas-por-ventana) sobre [entornos auditados](../engineering/rl-environment-integrity.md), con [ejecutores nativos sobre cintas reconstruidas](../engineering/native-policy-real-tapes.md) comprobados sin aprendizaje y [compilados con LibTorch CUDA](../../reports/engineering/rl-stage-native-20261009/README.md) | Completar las predicciones y la comparación de la campaña A v2. Los pilotos ya conservan predicciones de modelos ajustados y CQR. Siguen pendientes las políticas financieras entrenadas y sus costes. El test de 2024 sigue reservado |
| O6. Interpretación e informe | Interpretación exploratoria publicada con los pilotos, incluidos los resultados negativos: ninguno de los 48 selectores ni las once configuraciones del piloto de plasticidad supera al residual cero | Completar la interpretación de la campaña confirmatoria cuando exista |

## Alcance mínimo y extensiones

La [campaña previa de referencias](../engineering/comparison-campaign.md) pasa a
recorrer el universo completo admisible. No se limita a las cifras iniciales de
64 o 128 activos descritas debajo. O1, O2 y las referencias de O4 pueden avanzar
en paralelo con O5, siempre que cada entrenamiento use una instantánea de datos
fijada y validada. El desarrollo del candidato no forma parte de esa campaña.

El diseño inicial proponía una comparación estadounidense con piloto de hasta 64 activos y muestra principal de hasta 128. La campaña A v2 vigente utiliza todo el histórico admisible y un modelo conjunto US+CN, con controles por mercado. La comparación estricta conserva sus filas completas por separado. La edición histórica mantiene las cuatro posiciones de modalidad y el contexto macro con máscaras explícitas, usando las mismas filas en los modelos comparados. La reducción a un subconjunto requeriría otra declaración, no una poda silenciosa de la campaña.

La residualización sectorial, un Transformer adicional, HS300, memorias jerárquicas completas y kernels C++/CUDA son extensiones. Su activación exige que los controles temporales y las referencias funcionen, que exista presupuesto medido y que la comparación principal no quede comprometida.

La [ampliación](research-expansion.md) añade estudio cerebral, macroeconomía, recurrencia y antecedentes recientes, con decisiones de continuidad por componente. El inventario completo y la capacidad de leer por bloques permiten crecer sin cargar todo en RAM. El universo completo admisible forma parte de la campaña elegida. Las transferencias adicionales conservan sus propias hipótesis y controles. La recuperación mediante checkpoints se verifica antes de la campaña prolongada.

## Cierre de una tarea

Una tarea se cierra con el artefacto enlazado, su comprobación y las limitaciones que permanecen. Un documento de diseño puede cerrar la tarea de diseño. No cierra implementación, experimento ni validación del objetivo. El tablero y los informes deben mantener esa diferencia.
