# Objetivos, hitos y criterios de cierre

El trabajo se organiza en seis hitos alineados con la propuesta presentada. No son seis carpetas ni una secuencia rígida de implementación. El contrato de datos y el protocolo permiten iniciar referencias simples mientras se estudia la memoria. La comparación se cierra antes de redactar conclusiones.

| Hito | Entregable | Criterios de aceptación | Dependencia principal |
| --- | --- | --- | --- |
| O1. Datos multimodales | Manifiesto, ficha de datos, tabla experimental y auditoría temporal. | Procedencia y disponibilidad por modalidad. Exclusiones justificadas. Cobertura real. Prueba contra contaminación futura y revisión de licencias. | Requisitos, acceso y protocolo. |
| O2. Retornos residuales | Definición de etiqueta y residualizador reproducible. | Fechas de entrada/salida precisas. Coeficientes con historia permitida. Comparación bruto/residual. Sensibilidad al mercado y sector si verificable. | O1 y calendario de decisión. |
| O3. Memoria adaptativa | Diseño e implementación compacta con incertidumbre. | Lectura y escritura ordenadas. Etiquetas maduras. Estado reiniciable. Prueba numérica y presupuesto medido en CUDA. | O1/O2 y referencia neural simple. |
| O4. Referencias y ablaciones | Modelos comparables y matriz de componentes. | Mismo universo, targets, cortes y búsqueda registrada. Referencias cero/Ridge/boosting/GRU y variantes de memoria. | O1/O2. O3 para las ablaciones finales. |
| O5. Evaluación | Predicciones fuera de muestra e informe verificable. | Walk-forward purgado, calibración separada, costes completos, intervalos por bloques, test final protegido y todas las ejecuciones registradas. | Protocolo fijado y O4. |
| O6. Interpretación y memoria | Discusión, limitaciones, conclusiones y defensa. | Responder O1–O6 con evidencia, resultados negativos incluidos, referencias correctas, requisitos académicos revisados y ensayo de defensa. | Evidencias de O1–O5. Redacción paralela. |

## Orden práctico

```mermaid
flowchart LR
    R[Requisitos y protocolo] --> D[O1 · Datos]
    D --> T[O2 · Objetivo]
    T --> B[O4 · Referencias iniciales]
    B --> M[O3 · Memoria]
    M --> A[O4 · Ablaciones]
    A --> E[O5 · Evaluación cerrada]
    E --> C[O6 · Conclusiones y defensa]
    R -. escritura continua .-> C
```

Se trabajará en unidades pequeñas y revisables. El [tablero](task-board.md) desarrolla las tareas, dependencias y prioridades. Sus fechas se asignarán cuando se conozca el calendario del aula. Las duraciones estimadas son de planificación, no compromisos institucionales.

## Alcance mínimo y extensiones

El mínimo científico es un subconjunto estadounidense, un horizonte diario, precios y texto de disponibilidad justificable, referencias de varias familias y una memoria compacta con ablaciones. Los fundamentales y gráficos deben auditarse y evaluarse cuando sean válidos. Excluirlos requiere evidencia y discusión, no omisión silenciosa del primer objetivo.

La residualización sectorial, un Transformer adicional, HS300, memorias jerárquicas completas y kernels C++/CUDA son extensiones. Su activación exige que los controles temporales y las referencias funcionen, que exista presupuesto medido y que la comparación principal no quede comprometida.

## Cierre de una tarea

Una tarea se cierra con el artefacto enlazado, su comprobación y las limitaciones que permanecen. Un documento de diseño puede cerrar la tarea de diseño. No cierra implementación, experimento ni validación del objetivo. La matriz de rúbrica mantendrá esa diferencia durante todo el proyecto.
