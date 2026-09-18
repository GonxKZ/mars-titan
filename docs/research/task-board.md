# Tablero de trabajo de MARS-TITAN

Autor: Gonzalo García Lama.

El [catálogo de tareas](../../.github/planning/issues.json) contiene 46 tareas con alcance, criterios verificables, evidencias previstas, riesgos, dependencias y relación con la rúbrica. Es la especificación para crear y mantener las issues del [repositorio MARS-TITAN](https://github.com/GonxKZ/mars-titan) y su [tablero de GitHub Projects](https://github.com/users/GonxKZ/projects/4). Los identificadores MT-001 a MT-046 son estables y no equivalen a los números que GitHub asigne.

El catálogo local registra una tarea Done y 45 Todo. Done acredita únicamente la preparación documental de MT-001. En GitHub, MT-001 permanece abierta y En curso hasta verificar la entrega completa. Las demás tareas comienzan Pendiente. El código científico, los datos derivados, los entrenamientos y los resultados descritos son entregables futuros. La preparación del tablero no inicia su implementación.

## Organización por objetivos

| Hito | Resultado necesario para completarlo |
| --- | --- |
| O1. Datos multimodales y disponibilidad temporal | Subconjunto auditado, trazable y documentado, con políticas temporales y limitaciones explícitas. |
| O2. Retornos residuales y etiquetas | Definición y cálculo verificables, coeficientes históricos y disponibilidad de etiquetas diferenciada. |
| O3. Memoria de eventos e incertidumbre | Modelo compacto funcional, actualización temporalmente válida, estado aislado y calibración evaluable. |
| O4. Referencias y ablaciones | Modelos comparables y variantes que permitan estudiar memoria, sorpresa y modalidades. |
| O5. Evaluación temporal y económica | Protocolo cerrado, resultados fuera de muestra, costes e incertidumbre de las diferencias. |
| O6. Análisis, memoria y defensa | Interpretación crítica, recursos medidos, entregas completas, predepósito cerrado y defensa realizada. |

Los hitos agrupan objetivos y no imponen terminar todos los anteriores. MT-030, MT-031 y MT-032 se preparan pronto para habilitar MT-024: la predicción cero y Ridge no dependen de finalizar O3. Las dependencias de cada tarea determinan el orden necesario. La documentación y las entregas académicas avanzan junto con la investigación.

## Flujo del tablero

El flujo del tablero es **Pendiente → En curso → En revisión → Hecho**, con Bloqueado para impedimentos concretos. El catálogo usa Todo y Done como estados de planificación. Los estados efectivos de trabajo se mantienen en GitHub Projects.

- **Pendiente:** tarea definida, todavía no iniciada. Puede seleccionarse cuando sus dependencias estén resueltas y se disponga de datos o decisiones necesarias.
- **En curso:** trabajo delimitado en ejecución, con responsable y evidencia en preparación.
- **En revisión:** entregable disponible y comprobaciones ejecutadas, pendiente de revisar contra sus criterios.
- **Hecho:** todos los criterios están satisfechos y la evidencia está enlazada en la issue. Una implementación sin ejecutar, un borrador sin entregar o una defensa solo ensayada no se consideran completos cuando la tarea requiere esos resultados.
- **Bloqueado:** existe un impedimento registrado junto con la condición necesaria para continuar.

El límite es **dos tareas activas en total**, sumando En curso y En revisión. Los bloqueos se indican con el estado Bloqueado, la etiqueta `bloqueado`, un motivo concreto y la condición de desbloqueo. El trabajo que espera a una respuesta externa conserva su historial y dependencias. Las decisiones del director y las fechas del campus no se sustituyen por plazos inventados.

Una tarea no pasa a Hecho solo porque exista un archivo. Deben comprobarse su contenido, los criterios y, cuando corresponda, la ejecución, la entrega o el acto académico. La eliminación de una modalidad por falta de disponibilidad puede ser una conclusión válida si se documenta conforme al criterio de la tarea. No puede presentarse como una modalidad evaluada.

## Prioridad, tamaño y decisiones

**P0** protege validez temporal, reproducibilidad mínima o requisitos académicos imprescindibles. **P1** completa la comparación, los modelos y la interpretación. **P2** corresponde a una extensión prescindible. MT-039 puede cerrarse con una decisión no-go respaldada por el perfil y no bloquea la memoria ni la defensa.

**S** indica una decisión o entrega breve y acotada. **M** identifica un componente o análisis con verificación propia. **L** corresponde a un conjunto integrado que requiere varias sesiones. Los tamaños son relativos y no equivalen a días ni fechas de compromiso.

Los pesos de rúbrica de cada issue corresponden a los criterios a los que aporta evidencia. No son una distribución del esfuerzo ni puntos que se obtengan al cerrarla. La [matriz académica](../academic/rubric-matrix.md) contiene los ocho pesos oficiales, que suman 100 %, y distingue preparación de cumplimiento demostrado.

## Evidencias por responsabilidad

| Ubicación prevista | Responsabilidad |
| --- | --- |
| `dataset/` | Copia original preservada, fuera de Git. |
| `data/` | Manifiestos, subconjuntos y derivados. Los datos grandes permanecen fuera de Git. |
| `src/mars_titan/data/`, `targets/` | Disponibilidad, uniones, transformaciones y etiquetas. |
| `src/mars_titan/models/`, `memory/`, `calibration/` | Predictores, estado de memoria e incertidumbre. |
| `src/mars_titan/training/`, `evaluation/` | Ejecución temporal, métricas, simulación y trazabilidad. |
| `tests/`, `configs/` | Comprobaciones significativas y configuración declarativa. |
| `experiments/`, `reports/` | Registro de ejecuciones, resultados y análisis. Pesos y artefactos grandes excluidos de Git. |
| `docs/`, `thesis/` | Diseño, fuentes, decisiones, memoria y defensa. |
| `benchmarks/`, `native/` | Solo si el perfil justifica una exploración C++/CUDA optativa. |

Son destinos previstos, no una afirmación de que los módulos científicos existan. Las carpetas se organizan por responsabilidad, nunca por número de fase. Las herramientas existentes de descarga bibliográfica y comprobación documental se mantienen como utilidades. No se añade ingeniería ajena a los objetivos.

## Restricciones comunes de ejecución

- Gestionar Python, dependencias y ejecución con uv. Verificar NVIDIA y PyTorch antes de las cargas acelerables. Usar `cuda:0` y registrar el cumplimiento experimental de 8 GB. Una indisponibilidad debe explicar su causa antes de considerar otra ejecución.
- No usar etiquetas futuras para entrenar, escribir memoria, calcular sorpresa o calibrar. Predecir todos los activos del mismo instante antes de las escrituras correspondientes. Aplicar escrituras maduras en orden estable por `(label_available_at, event_id, asset_id)` y verificar que permutar el lote conserva el estado resultante y las predicciones posteriores. Respetar reinicios, reconstrucción y freeze por fold.
- Tratar noticias con fecha sola mediante lag de una sesión negociable, reconstruir fundamentales por filings y regenerar gráficos con ventanas truncadas. Documentar sesgos de supervivencia y preentrenamiento.
- Reservar la prueba final. Realizar selección, normalización, calibración y elección de umbrales con datos permitidos. Mantener todos los ensayos y resultados negativos.
- Utilizar cuantiles con pinball, cobertura y anchura en el núcleo de incertidumbre. Mantener congelado su calibrador durante test. NLL requiere una densidad especificada. ACI es una opción prequential separada, con política e hiperparámetros fijados previamente y actualizaciones con etiquetas maduras, sin tuning en test.
- Comparar al menos cero, Ridge, boosting, GRU y una referencia de memoria, junto con las ablaciones comprometidas. DLinear es deseable, pero su exclusión justificada no bloquea la evaluación. Las adaptaciones y exclusiones deben quedar visibles.
- Evaluar resultados fuera de muestra y costes de simulación. Residualización y ablaciones no identifican por sí solas causalidad económica ni acreditan rentabilidad futura.
- Seguir las tres entregas, el predepósito sin cambios y las autorizaciones reales descritas en los [requisitos](../academic/requirements.md). La defensa individual tiene máximo de 15 minutos, 10 recomendados, y exige obtener al menos 1,5 de los 3 puntos de exposición para aprobar.
- Utilizar ramas descriptivas y Conventional Commits en inglés para cambios futuros. Cada issue debe cerrar un resultado coherente, con criterios comprobados y sin introducir datos o publicaciones de terceros en Git.

## Catálogo de trabajo

| ID | Tarea | Hito | Prioridad | Tamaño | Estado inicial | Dependencias |
| --- | --- | --- | --- | --- | --- | --- |
| MT-001 | Preparar requisitos, rúbrica y revisión del original | Transversal | P0 | M | Done | Ninguna |
| MT-002 | Confirmar plantilla, calendario y requisitos con la dirección | Transversal | P0 | S | Todo | MT-001 |
| MT-003 | Completar el estado del arte y mantener referencias verificables | O6 | P1 | M | Todo | MT-001 |
| MT-004 | Verificar el entorno científico con uv y CUDA explícita | Transversal | P0 | M | Todo | Ninguna |
| MT-005 | Inventariar FinMultiTime y sus condiciones de uso | O1 | P0 | M | Todo | Ninguna |
| MT-006 | Auditar precios, identificadores y sesgo de supervivencia | O1 | P0 | L | Todo | MT-005 |
| MT-007 | Normalizar noticias con disponibilidad conservadora y deduplicación | O1 | P0 | M | Todo | MT-005 |
| MT-008 | Reconstruir disponibilidad de fundamentales por filings | O1 | P0 | L | Todo | MT-005 |
| MT-009 | Regenerar gráficos con ventanas truncadas al instante de decisión | O1 | P1 | M | Todo | MT-006 |
| MT-010 | Auditar preentrenamiento y versionar representaciones congeladas | O1 | P1 | M | Todo | MT-007, MT-009, MT-004 |
| MT-011 | Construir contratos point-in-time y uniones as-of | O1 | P0 | L | Todo | MT-006, MT-007, MT-008 |
| MT-012 | Congelar el subconjunto experimental y su data card | O1 | P0 | M | Todo | MT-005, MT-006, MT-007, MT-008, MT-011 |
| MT-013 | Definir horizonte, retorno residual y cronología de decisión | O2 | P0 | M | Todo | MT-001 |
| MT-014 | Versionar factores de mercado y sector con su disponibilidad | O2 | P0 | M | Todo | MT-005, MT-006, MT-013 |
| MT-015 | Implementar residuales con coeficientes históricos y etiquetas diferidas | O2 | P0 | L | Todo | MT-011, MT-013, MT-014 |
| MT-016 | Diagnosticar estabilidad y cobertura del objetivo residual | O2 | P1 | M | Todo | MT-012, MT-015 |
| MT-017 | Implementar representación de eventos y fusión común | O3 | P1 | L | Todo | MT-004, MT-011, MT-012 |
| MT-018 | Implementar memoria global mínima y trazable | O3 | P1 | M | Todo | MT-017 |
| MT-019 | Definir sorpresa económica con información ya observada | O3 | P1 | M | Todo | MT-015, MT-018 |
| MT-020 | Evaluar un enrutamiento por regímenes observables | O3 | P1 | M | Todo | MT-018, MT-019 |
| MT-021 | Verificar estado prequential, reinicios y aislamiento por fold | O3 | P0 | L | Todo | MT-015, MT-018 |
| MT-022 | Implementar cuantiles y calibración temporal controlada | O3 | P1 | L | Todo | MT-013, MT-017, MT-021 |
| MT-023 | Integrar MARS-TITAN compacto y verificar un entrenamiento mínimo | O3 | P1 | L | Todo | MT-004, MT-017, MT-018, MT-019, MT-020, MT-021, MT-022 |
| MT-024 | Establecer predicción cero y Ridge como referencias iniciales | O4 | P0 | M | Todo | MT-015, MT-031, MT-032 |
| MT-025 | Añadir un baseline de gradient boosting con presupuesto acotado | O4 | P1 | M | Todo | MT-024 |
| MT-026 | Comparar referencias temporales compactas | O4 | P1 | L | Todo | MT-004, MT-024 |
| MT-027 | Contrastar una referencia de memoria con fidelidad explícita | O4 | P1 | L | Todo | MT-003, MT-018, MT-021, MT-031, MT-032 |
| MT-028 | Fijar la matriz de ablaciones y los controles de capacidad | O4 | P0 | M | Todo | MT-013, MT-017, MT-030 |
| MT-029 | Ejecutar ablaciones de memoria, sorpresa y modalidades | O4 | P1 | L | Todo | MT-010, MT-023, MT-027, MT-028, MT-031, MT-032 |
| MT-030 | Cerrar el protocolo walk-forward y la reserva final | O5 | P0 | M | Todo | MT-013 |
| MT-031 | Construir el runner y registro de experimentos reproducibles | O5 | P0 | L | Todo | MT-004, MT-012, MT-015, MT-030 |
| MT-032 | Verificar métricas predictivas, de ranking y calibración | O5 | P0 | M | Todo | MT-013, MT-030 |
| MT-033 | Implementar simulación long-short con costes y cronología operable | O5 | P1 | L | Todo | MT-006, MT-013, MT-030 |
| MT-034 | Seleccionar configuraciones solo con entrenamiento y validación | O5 | P1 | L | Todo | MT-023, MT-024, MT-025, MT-026, MT-027, MT-028, MT-031, MT-032, MT-033 |
| MT-035 | Ejecutar la comparación final fuera de muestra | O5 | P0 | L | Todo | MT-029, MT-034 |
| MT-036 | Estimar incertidumbre de las diferencias y sensibilidad a selección | O5 | P1 | M | Todo | MT-035 |
| MT-037 | Analizar robustez a costes, regímenes y modalidades ausentes | O5 | P1 | M | Todo | MT-035 |
| MT-038 | Medir recursos y verificar la restricción de 8 GB | O6 | P0 | M | Todo | MT-023, MT-026, MT-027, MT-035 |
| MT-039 | Decidir si compensa una optimización C++/CUDA | O6 | P2 | M | Todo | MT-038 |
| MT-040 | Sintetizar fallos, aportaciones y límites de los seis objetivos | O6 | P1 | M | Todo | MT-016, MT-029, MT-036, MT-037, MT-038 |
| MT-041 | Preparar y entregar el primer borrador académico | O6 | P1 | M | Todo | MT-001, MT-002, MT-003, MT-013, MT-030 |
| MT-042 | Preparar y entregar el segundo borrador con avance comprobado | O6 | P1 | M | Todo | MT-041, MT-012, MT-015, MT-024, MT-023, MT-028 |
| MT-043 | Completar y entregar el tercer borrador de la memoria | O6 | P1 | L | Todo | MT-042, MT-040 |
| MT-044 | Reproducir tablas y figuras y revisar el paquete de evidencias | O6 | P0 | M | Todo | MT-003, MT-035, MT-036, MT-037, MT-038, MT-043 |
| MT-045 | Cerrar predepósito y tramitar el depósito autorizado | O6 | P0 | M | Todo | MT-002, MT-043, MT-044 |
| MT-046 | Preparar y realizar la defensa con dominio de resultados | O6 | P0 | L | Todo | MT-040, MT-044, MT-045 |

La preparación de MT-041 y MT-042 puede requerir adaptar el alcance al calendario real con el director, conservando una descripción veraz del avance. Las fechas se incorporarán desde el campus. El catálogo no declara presentada ninguna entrega ni concedida ninguna autorización.

## Cierre y mantenimiento

Al comenzar una tarea se revisan sus dependencias y la vigencia de sus supuestos. Al cerrarla se enlazan el cambio, las comprobaciones, los artefactos y la decisión correspondiente. Si aparece una necesidad nueva, se incorpora con identificador propio y se explica su relación con O1-O6. No se amplía silenciosamente una tarea en curso.

Las cuestiones de permisos, autoría y uso de herramientas se resuelven conforme a las fuentes académicas y al uso real. No se publican declaraciones ni autorizaciones que no estén acreditadas. La memoria final y la defensa deben reflejar la evidencia conseguida, incluso cuando esta no respalde la superioridad de MARS-TITAN.
