# Tablero de trabajo de MARS-TITAN

Autor: Gonzalo García Lama.

El [catálogo de tareas](../../.github/planning/issues.json) contiene 62 tareas activas con alcance, criterios verificables, evidencias previstas, riesgos, dependencias y relación con la rúbrica. Es la especificación para crear y mantener las issues del [repositorio MARS-TITAN](https://github.com/GonxKZ/mars-titan) y su [tablero de GitHub Projects](https://github.com/users/GonxKZ/projects/4). Los identificadores MT son estables y no equivalen a los números que GitHub asigne. El catálogo activo excluye cinco tareas consolidadas, cuyo historial se conserva en un registro separado.

El catálogo y el tablero registran **una tarea Hecho y 61 Pendiente**. [MT-001](https://github.com/GonxKZ/mars-titan/issues/1) conserva su cierre por la preparación documental verificada. Las otras 61 tareas siguen abiertas. La revisión de las 67 tareas originales consolidó cinco duplicadas, cerradas como no planificadas y retiradas del tablero activo. Ese cierre no acredita trabajo científico. El [mapa remoto](../../.github/planning/remote-map.json) recoge 62 elementos activos, 177 dependencias y la huella del catálogo sincronizado.

## Organización por objetivos

| Hito | Resultado necesario para completarlo |
| --- | --- |
| O1. Datos multimodales y disponibilidad temporal | Inventario completo y muestra principal auditada, trazable y documentada, con políticas temporales y limitaciones explícitas. |
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

**P0** protege validez temporal, reproducibilidad mínima o requisitos académicos imprescindibles. **P1** completa la comparación, los modelos y la interpretación. **P2** corresponde a una extensión prescindible. MT-039, MT-050, MT-053, MT-055, MT-063, MT-064 y MT-067 pueden cerrarse con una decisión no-go respaldada por evidencia y no bloquean la memoria ni la defensa. Una decisión no-go no se presenta como un resultado de un experimento no ejecutado.

**S** indica una decisión o entrega breve y acotada. **M** identifica un componente o análisis con verificación propia. **L** corresponde a un conjunto integrado que requiere varias sesiones. Los tamaños son relativos y no equivalen a días ni fechas de compromiso.

Los pesos de rúbrica de cada issue corresponden a los criterios a los que aporta evidencia. No son una distribución del esfuerzo ni puntos que se obtengan al cerrarla. La [matriz académica](../academic/rubric-matrix.md) contiene los ocho pesos oficiales, que suman 100 %, y distingue preparación de cumplimiento demostrado.

## Alcance principal y ampliaciones condicionadas

Se prioriza una comparación terminada y repetible dentro de las entregas. La propuesta de planificación es un piloto de hasta 64 activos y una muestra principal de 128 activos estadounidenses. MT-012 fija la regla de selección y el manifiesto piloto con información de entrenamiento. MT-060 mide el presupuesto y MT-034 congela la versión definitiva de la muestra principal. Son puntos de partida, no tamaños ya validados ni compromisos de duración.

El inventario recorre toda la copia de Estados Unidos y China, distinguiendo archivo inventariado, inspeccionado y validado. La auditoría detallada se concentra primero en la muestra principal. La [preparación por bloques](../engineering/full-dataset-training.md) permite limitar RAM y evitar ventanas duplicadas. Ni el inventario ni el soporte de lectura acreditan haber entrenado sobre toda la fuente.

| Decisión | Tareas | Condición de continuidad |
| --- | --- | --- |
| Muestra principal y referencias comparables | MT-012, MT-024, MT-025, MT-026, MT-030, MT-034 y MT-035 | Misma población, cortes válidos, presupuesto y repetición suficientes. |
| Memoria candidata y controles | MT-019 y MT-052 | Capacidad y escrituras acotadas, controles simples y utilidad medida. La retención no se confunde con precisión. |
| Recurrencia de uno, dos o cuatro pasos | MT-054 | Mantener K = 1 si el refinamiento no aporta una relación defendible entre error y coste. |
| Indicadores macro priorizados | MT-056 y MT-057 | Admitir solo series con disponibilidad demostrada. Un catálogo amplio no obliga a entrenar con todas. |
| Escalado a 256 activos o al universo completo | MT-064 | Cobertura auditada y horas suficientes después de reservar recursos para el núcleo y las entregas. |
| China y eventual transferencia | MT-058 y MT-064 | Calendarios y normas históricas verificables, resultados separados y presupuesto disponible. |
| Fuentes gratuitas y actualización incremental | MT-066 y MT-067 | Condiciones de uso, versiones inmutables y separación del benchmark congelado. Descargar hoy no acredita disponibilidad histórica. |
| Nuevas familias, replay, destilación y concurrencia | MT-050, MT-053, MT-055 y MT-063 | Pregunta específica, evidencia previa y decisión go/no-go. No ejecutar todas las combinaciones. |

MT-065 define checkpoints coherentes y reanudación dentro del fold antes del runner MT-031 y, por dependencia, antes de la evaluación final. Conserva optimización, RNG, cursor, memoria, colas y revisiones, con pruebas de interrupción y retención acotada. MT-034 utiliza las ablaciones de MT-029 y los controles de compatibilidad y presupuesto de MT-059 y MT-060. No depende de las extensiones P2. Las tareas especializadas de cada mecanismo se ejecutan cuando se active ese mecanismo, con las comprobaciones correspondientes antes del test. Los antecedentes se documentan en MT-003, las pruebas se diseñan en MT-028 y las falsaciones del candidato se ejecutan en MT-029.

## Presupuesto y calendario relativo

El [presupuesto de latencia](../engineering/latency-budget.md) y MT-060 deben aportar tiempos observados de preparación, entrenamiento, inferencia y actualización. La estimación de la campaña incorporará ejemplos, pasadas, folds, semillas, búsqueda, calibración y repetición de fallos. Una segunda escala comprobará la extrapolación del piloto. El tamaño de 109 GiB no determina por sí solo horas de entrenamiento.

El equipo puede permanecer disponible 24/7. Esto permite programar ejecuciones largas con recuperación, pero no acredita rendimiento sostenido a máxima potencia. MT-060 mide temperatura, limitación térmica, carga de CPU y coste de supervisión, y MT-065 verifica recuperación ante interrupciones.

La secuencia de planificación es inventario y contratos, checkpoints verificables, piloto, decisión de muestra y presupuesto, comparación confirmatoria, y cierre de análisis y entregas. No hay fechas académicas comunicadas y no se asignan fechas ni semanas ficticias. Cuando esté disponible el calendario, se fijarán los límites dejando margen explícito para redacción, revisión con el director, predepósito y defensa. Si el presupuesto no alcanza, se reduce primero el número de extensiones y después el tamaño de la muestra, conservando referencias, controles temporales e incertidumbre de los resultados.

## Evidencias por responsabilidad

| Ubicación prevista | Responsabilidad |
| --- | --- |
| `dataset/` | Copia original preservada, fuera de Git. |
| `data/` | Catálogos, manifiestos, muestra principal y derivados. Los datos grandes permanecen fuera de Git. |
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
- Tratar noticias con fecha sola mediante lag de una sesión negociable, reconstruir fundamentales por filings y regenerar gráficos con ventanas truncadas. Las revisiones macro son eventos nuevos y no reescriben instantáneas antiguas. Documentar sesgos de supervivencia y preentrenamiento.
- Reservar la prueba final. Realizar selección, normalización, calibración y elección de umbrales con datos permitidos. Mantener todos los ensayos y resultados negativos.
- Utilizar cuantiles con pinball, cobertura y anchura en el núcleo de incertidumbre. Mantener congelado su calibrador durante test. NLL requiere una densidad especificada. ACI es una opción prequential separada, con política e hiperparámetros fijados previamente y actualizaciones con etiquetas maduras, sin tuning en test.
- Comparar al menos cero, Ridge, boosting, GRU y una referencia de memoria, junto con las ablaciones comprometidas, sobre la misma muestra principal. HistGradientBoosting sirve de control inicial y XGBoost con memoria externa queda previsto para ampliaciones que lo necesiten. DLinear es deseable, pero su exclusión justificada no bloquea la evaluación. Las adaptaciones y exclusiones deben quedar visibles.
- Evaluar resultados fuera de muestra y costes de simulación. Versionar normas por vigencia y no aplicar reglas de 2026 a periodos anteriores sin evidencia. Si se incorpora China, su objetivo predictivo y su viabilidad operativa se justifican por separado. Residualización y ablaciones no identifican por sí solas causalidad económica ni acreditan rentabilidad futura.
- Seguir las tres entregas, el predepósito sin cambios y las autorizaciones reales descritas en los [requisitos](../academic/requirements.md). La defensa individual tiene máximo de 15 minutos, 10 recomendados, y exige obtener al menos 1,5 de los 3 puntos de exposición para aprobar.
- Utilizar ramas descriptivas y Conventional Commits en inglés para cambios futuros. Cada issue debe cerrar un resultado coherente, con criterios comprobados y sin introducir datos o publicaciones de terceros en Git.

## Catálogo de trabajo

| ID | Tarea | Hito | Prioridad | Tamaño | Estado actual | Dependencias |
| --- | --- | --- | --- | --- | --- | --- |
| [MT-001](https://github.com/GonxKZ/mars-titan/issues/1) | Preparar requisitos, rúbrica y revisión del original | Transversal | P0 | M | Hecho | Ninguna |
| [MT-002](https://github.com/GonxKZ/mars-titan/issues/2) | Confirmar requisitos y calendario académico | Transversal | P0 | S | Pendiente | [MT-001](https://github.com/GonxKZ/mars-titan/issues/1) |
| [MT-003](https://github.com/GonxKZ/mars-titan/issues/3) | Completar el estado del arte y las referencias | O6 | P1 | M | Pendiente | [MT-001](https://github.com/GonxKZ/mars-titan/issues/1) |
| [MT-004](https://github.com/GonxKZ/mars-titan/issues/4) | Verificar el entorno con uv y CUDA | Transversal | P0 | M | Pendiente | Ninguna |
| [MT-005](https://github.com/GonxKZ/mars-titan/issues/5) | Inventariar FinMultiTime | O1 | P0 | M | Pendiente | Ninguna |
| [MT-006](https://github.com/GonxKZ/mars-titan/issues/6) | Auditar precios y universo histórico | O1 | P0 | L | Pendiente | [MT-005](https://github.com/GonxKZ/mars-titan/issues/5) |
| [MT-007](https://github.com/GonxKZ/mars-titan/issues/7) | Normalizar noticias y su disponibilidad | O1 | P0 | M | Pendiente | [MT-005](https://github.com/GonxKZ/mars-titan/issues/5) |
| [MT-008](https://github.com/GonxKZ/mars-titan/issues/8) | Reconstruir fundamentales por publicación | O1 | P0 | L | Pendiente | [MT-005](https://github.com/GonxKZ/mars-titan/issues/5) |
| [MT-009](https://github.com/GonxKZ/mars-titan/issues/9) | Generar gráficos con historia disponible | O1 | P1 | M | Pendiente | [MT-006](https://github.com/GonxKZ/mars-titan/issues/6) |
| [MT-010](https://github.com/GonxKZ/mars-titan/issues/10) | Auditar y versionar codificadores congelados | O1 | P1 | M | Pendiente | [MT-007](https://github.com/GonxKZ/mars-titan/issues/7), [MT-009](https://github.com/GonxKZ/mars-titan/issues/9), [MT-004](https://github.com/GonxKZ/mars-titan/issues/4) |
| [MT-011](https://github.com/GonxKZ/mars-titan/issues/11) | Construir el contrato temporal de datos | O1 | P0 | L | Pendiente | [MT-006](https://github.com/GonxKZ/mars-titan/issues/6), [MT-007](https://github.com/GonxKZ/mars-titan/issues/7), [MT-008](https://github.com/GonxKZ/mars-titan/issues/8) |
| [MT-012](https://github.com/GonxKZ/mars-titan/issues/12) | Definir la selección y publicar el piloto | O1 | P0 | M | Pendiente | [MT-005](https://github.com/GonxKZ/mars-titan/issues/5), [MT-006](https://github.com/GonxKZ/mars-titan/issues/6), [MT-007](https://github.com/GonxKZ/mars-titan/issues/7), [MT-008](https://github.com/GonxKZ/mars-titan/issues/8), [MT-011](https://github.com/GonxKZ/mars-titan/issues/11) |
| [MT-013](https://github.com/GonxKZ/mars-titan/issues/13) | Definir objetivo y reloj de decisión | O2 | P0 | M | Pendiente | [MT-001](https://github.com/GonxKZ/mars-titan/issues/1) |
| [MT-014](https://github.com/GonxKZ/mars-titan/issues/14) | Versionar factores de mercado y sector | O2 | P0 | M | Pendiente | [MT-005](https://github.com/GonxKZ/mars-titan/issues/5), [MT-006](https://github.com/GonxKZ/mars-titan/issues/6), [MT-013](https://github.com/GonxKZ/mars-titan/issues/13) |
| [MT-015](https://github.com/GonxKZ/mars-titan/issues/15) | Calcular residuales y maduración de etiquetas | O2 | P0 | L | Pendiente | [MT-011](https://github.com/GonxKZ/mars-titan/issues/11), [MT-013](https://github.com/GonxKZ/mars-titan/issues/13), [MT-014](https://github.com/GonxKZ/mars-titan/issues/14) |
| [MT-016](https://github.com/GonxKZ/mars-titan/issues/16) | Diagnosticar cobertura y estabilidad del residual | O2 | P1 | M | Pendiente | [MT-012](https://github.com/GonxKZ/mars-titan/issues/12), [MT-015](https://github.com/GonxKZ/mars-titan/issues/15) |
| [MT-017](https://github.com/GonxKZ/mars-titan/issues/17) | Implementar codificación y fusión de eventos | O3 | P1 | L | Pendiente | [MT-004](https://github.com/GonxKZ/mars-titan/issues/4), [MT-011](https://github.com/GonxKZ/mars-titan/issues/11), [MT-012](https://github.com/GonxKZ/mars-titan/issues/12) |
| [MT-018](https://github.com/GonxKZ/mars-titan/issues/18) | Implementar memoria global acotada | O3 | P1 | M | Pendiente | [MT-017](https://github.com/GonxKZ/mars-titan/issues/17) |
| [MT-019](https://github.com/GonxKZ/mars-titan/issues/19) | Comparar políticas de escritura | O3 | P1 | L | Pendiente | [MT-015](https://github.com/GonxKZ/mars-titan/issues/15), [MT-018](https://github.com/GonxKZ/mars-titan/issues/18), [MT-021](https://github.com/GonxKZ/mars-titan/issues/21), [MT-028](https://github.com/GonxKZ/mars-titan/issues/28) |
| [MT-020](https://github.com/GonxKZ/mars-titan/issues/20) | Comparar enrutamiento por regímenes | O3 | P1 | M | Pendiente | [MT-018](https://github.com/GonxKZ/mars-titan/issues/18), [MT-019](https://github.com/GonxKZ/mars-titan/issues/19) |
| [MT-021](https://github.com/GonxKZ/mars-titan/issues/21) | Verificar orden temporal y aislamiento | O3 | P0 | L | Pendiente | [MT-015](https://github.com/GonxKZ/mars-titan/issues/15), [MT-018](https://github.com/GonxKZ/mars-titan/issues/18) |
| [MT-022](https://github.com/GonxKZ/mars-titan/issues/22) | Implementar cuantiles y calibración | O3 | P1 | L | Pendiente | [MT-013](https://github.com/GonxKZ/mars-titan/issues/13), [MT-017](https://github.com/GonxKZ/mars-titan/issues/17), [MT-021](https://github.com/GonxKZ/mars-titan/issues/21) |
| [MT-023](https://github.com/GonxKZ/mars-titan/issues/23) | Integrar el modelo y ejecutar un ensayo mínimo | O3 | P1 | L | Pendiente | [MT-004](https://github.com/GonxKZ/mars-titan/issues/4), [MT-017](https://github.com/GonxKZ/mars-titan/issues/17), [MT-018](https://github.com/GonxKZ/mars-titan/issues/18), [MT-019](https://github.com/GonxKZ/mars-titan/issues/19), [MT-020](https://github.com/GonxKZ/mars-titan/issues/20), [MT-021](https://github.com/GonxKZ/mars-titan/issues/21), [MT-022](https://github.com/GonxKZ/mars-titan/issues/22) |
| [MT-024](https://github.com/GonxKZ/mars-titan/issues/24) | Validar cero, Ridge y el cálculo por bloques | O4 | P0 | M | Pendiente | [MT-015](https://github.com/GonxKZ/mars-titan/issues/15), [MT-031](https://github.com/GonxKZ/mars-titan/issues/31), [MT-032](https://github.com/GonxKZ/mars-titan/issues/32) |
| [MT-025](https://github.com/GonxKZ/mars-titan/issues/25) | Evaluar la referencia de boosting | O4 | P1 | M | Pendiente | [MT-024](https://github.com/GonxKZ/mars-titan/issues/24) |
| [MT-026](https://github.com/GonxKZ/mars-titan/issues/26) | Evaluar GRU y decidir sobre DLinear | O4 | P1 | L | Pendiente | [MT-004](https://github.com/GonxKZ/mars-titan/issues/4), [MT-024](https://github.com/GonxKZ/mars-titan/issues/24) |
| [MT-027](https://github.com/GonxKZ/mars-titan/issues/27) | Evaluar una referencia de memoria identificable | O4 | P1 | L | Pendiente | [MT-003](https://github.com/GonxKZ/mars-titan/issues/3), [MT-018](https://github.com/GonxKZ/mars-titan/issues/18), [MT-021](https://github.com/GonxKZ/mars-titan/issues/21), [MT-031](https://github.com/GonxKZ/mars-titan/issues/31), [MT-032](https://github.com/GonxKZ/mars-titan/issues/32) |
| [MT-028](https://github.com/GonxKZ/mars-titan/issues/28) | Diseñar ablaciones y falsaciones | O4 | P0 | M | Pendiente | [MT-013](https://github.com/GonxKZ/mars-titan/issues/13), [MT-017](https://github.com/GonxKZ/mars-titan/issues/17), [MT-030](https://github.com/GonxKZ/mars-titan/issues/30), [MT-003](https://github.com/GonxKZ/mars-titan/issues/3) |
| [MT-029](https://github.com/GonxKZ/mars-titan/issues/29) | Ejecutar ablaciones y falsaciones | O4 | P1 | L | Pendiente | [MT-010](https://github.com/GonxKZ/mars-titan/issues/10), [MT-023](https://github.com/GonxKZ/mars-titan/issues/23), [MT-027](https://github.com/GonxKZ/mars-titan/issues/27), [MT-028](https://github.com/GonxKZ/mars-titan/issues/28), [MT-031](https://github.com/GonxKZ/mars-titan/issues/31), [MT-032](https://github.com/GonxKZ/mars-titan/issues/32) |
| [MT-030](https://github.com/GonxKZ/mars-titan/issues/30) | Fijar el protocolo y la reserva final | O5 | P0 | M | Pendiente | [MT-013](https://github.com/GonxKZ/mars-titan/issues/13) |
| [MT-031](https://github.com/GonxKZ/mars-titan/issues/31) | Implementar ejecución y trazabilidad | O5 | P0 | L | Pendiente | [MT-004](https://github.com/GonxKZ/mars-titan/issues/4), [MT-012](https://github.com/GonxKZ/mars-titan/issues/12), [MT-015](https://github.com/GonxKZ/mars-titan/issues/15), [MT-030](https://github.com/GonxKZ/mars-titan/issues/30), [MT-065](https://github.com/GonxKZ/mars-titan/issues/67) |
| [MT-032](https://github.com/GonxKZ/mars-titan/issues/32) | Verificar métricas y agregaciones | O5 | P0 | M | Pendiente | [MT-013](https://github.com/GonxKZ/mars-titan/issues/13), [MT-030](https://github.com/GonxKZ/mars-titan/issues/30) |
| [MT-033](https://github.com/GonxKZ/mars-titan/issues/33) | Implementar la simulación con costes | O5 | P1 | L | Pendiente | [MT-006](https://github.com/GonxKZ/mars-titan/issues/6), [MT-013](https://github.com/GonxKZ/mars-titan/issues/13), [MT-030](https://github.com/GonxKZ/mars-titan/issues/30) |
| [MT-034](https://github.com/GonxKZ/mars-titan/issues/34) | Fijar muestra principal y finalistas | O5 | P1 | L | Pendiente | [MT-023](https://github.com/GonxKZ/mars-titan/issues/23), [MT-024](https://github.com/GonxKZ/mars-titan/issues/24), [MT-025](https://github.com/GonxKZ/mars-titan/issues/25), [MT-026](https://github.com/GonxKZ/mars-titan/issues/26), [MT-027](https://github.com/GonxKZ/mars-titan/issues/27), [MT-028](https://github.com/GonxKZ/mars-titan/issues/28), [MT-031](https://github.com/GonxKZ/mars-titan/issues/31), [MT-032](https://github.com/GonxKZ/mars-titan/issues/32), [MT-033](https://github.com/GonxKZ/mars-titan/issues/33), [MT-059](https://github.com/GonxKZ/mars-titan/issues/61), [MT-060](https://github.com/GonxKZ/mars-titan/issues/62), [MT-029](https://github.com/GonxKZ/mars-titan/issues/29) |
| [MT-035](https://github.com/GonxKZ/mars-titan/issues/35) | Ejecutar la comparación fuera de muestra | O5 | P0 | L | Pendiente | [MT-029](https://github.com/GonxKZ/mars-titan/issues/29), [MT-034](https://github.com/GonxKZ/mars-titan/issues/34) |
| [MT-036](https://github.com/GonxKZ/mars-titan/issues/36) | Estimar incertidumbre y efecto de la selección | O5 | P1 | M | Pendiente | [MT-035](https://github.com/GonxKZ/mars-titan/issues/35) |
| [MT-037](https://github.com/GonxKZ/mars-titan/issues/37) | Evaluar robustez y abstención | O5 | P1 | L | Pendiente | [MT-035](https://github.com/GonxKZ/mars-titan/issues/35) |
| [MT-038](https://github.com/GonxKZ/mars-titan/issues/38) | Informar recursos de los experimentos finales | O6 | P0 | M | Pendiente | [MT-023](https://github.com/GonxKZ/mars-titan/issues/23), [MT-026](https://github.com/GonxKZ/mars-titan/issues/26), [MT-027](https://github.com/GonxKZ/mars-titan/issues/27), [MT-035](https://github.com/GonxKZ/mars-titan/issues/35), [MT-060](https://github.com/GonxKZ/mars-titan/issues/62) |
| [MT-039](https://github.com/GonxKZ/mars-titan/issues/39) | Decidir sobre optimización nativa | O6 | P2 | M | Pendiente | [MT-038](https://github.com/GonxKZ/mars-titan/issues/38) |
| [MT-040](https://github.com/GonxKZ/mars-titan/issues/40) | Redactar conclusiones de los seis objetivos | O6 | P1 | M | Pendiente | [MT-016](https://github.com/GonxKZ/mars-titan/issues/16), [MT-029](https://github.com/GonxKZ/mars-titan/issues/29), [MT-036](https://github.com/GonxKZ/mars-titan/issues/36), [MT-037](https://github.com/GonxKZ/mars-titan/issues/37), [MT-038](https://github.com/GonxKZ/mars-titan/issues/38) |
| [MT-041](https://github.com/GonxKZ/mars-titan/issues/41) | Preparar y entregar el primer borrador | O6 | P1 | M | Pendiente | [MT-001](https://github.com/GonxKZ/mars-titan/issues/1), [MT-002](https://github.com/GonxKZ/mars-titan/issues/2), [MT-003](https://github.com/GonxKZ/mars-titan/issues/3), [MT-013](https://github.com/GonxKZ/mars-titan/issues/13), [MT-030](https://github.com/GonxKZ/mars-titan/issues/30) |
| [MT-042](https://github.com/GonxKZ/mars-titan/issues/42) | Preparar y entregar el segundo borrador | O6 | P1 | M | Pendiente | [MT-041](https://github.com/GonxKZ/mars-titan/issues/41), [MT-012](https://github.com/GonxKZ/mars-titan/issues/12), [MT-015](https://github.com/GonxKZ/mars-titan/issues/15), [MT-024](https://github.com/GonxKZ/mars-titan/issues/24), [MT-023](https://github.com/GonxKZ/mars-titan/issues/23), [MT-028](https://github.com/GonxKZ/mars-titan/issues/28) |
| [MT-043](https://github.com/GonxKZ/mars-titan/issues/43) | Completar y entregar el tercer borrador | O6 | P1 | L | Pendiente | [MT-042](https://github.com/GonxKZ/mars-titan/issues/42), [MT-040](https://github.com/GonxKZ/mars-titan/issues/40) |
| [MT-044](https://github.com/GonxKZ/mars-titan/issues/44) | Reproducir figuras y revisar las evidencias | O6 | P0 | M | Pendiente | [MT-003](https://github.com/GonxKZ/mars-titan/issues/3), [MT-035](https://github.com/GonxKZ/mars-titan/issues/35), [MT-036](https://github.com/GonxKZ/mars-titan/issues/36), [MT-037](https://github.com/GonxKZ/mars-titan/issues/37), [MT-038](https://github.com/GonxKZ/mars-titan/issues/38), [MT-043](https://github.com/GonxKZ/mars-titan/issues/43) |
| [MT-045](https://github.com/GonxKZ/mars-titan/issues/45) | Cerrar predepósito y tramitar la autorización | O6 | P0 | M | Pendiente | [MT-002](https://github.com/GonxKZ/mars-titan/issues/2), [MT-043](https://github.com/GonxKZ/mars-titan/issues/43), [MT-044](https://github.com/GonxKZ/mars-titan/issues/44) |
| [MT-046](https://github.com/GonxKZ/mars-titan/issues/46) | Preparar y realizar la defensa | O6 | P0 | L | Pendiente | [MT-040](https://github.com/GonxKZ/mars-titan/issues/40), [MT-044](https://github.com/GonxKZ/mars-titan/issues/44), [MT-045](https://github.com/GonxKZ/mars-titan/issues/45) |
| [MT-048](https://github.com/GonxKZ/mars-titan/issues/50) | Preparar datos por bloques y ventanas bajo demanda | O1 | P1 | L | Pendiente | [MT-004](https://github.com/GonxKZ/mars-titan/issues/4), [MT-005](https://github.com/GonxKZ/mars-titan/issues/5), [MT-011](https://github.com/GonxKZ/mars-titan/issues/11), [MT-012](https://github.com/GonxKZ/mars-titan/issues/12) |
| [MT-050](https://github.com/GonxKZ/mars-titan/issues/52) | Contrastar familias secuenciales adicionales | O4 | P2 | L | Pendiente | [MT-026](https://github.com/GonxKZ/mars-titan/issues/26), [MT-028](https://github.com/GonxKZ/mars-titan/issues/28), [MT-048](https://github.com/GonxKZ/mars-titan/issues/50) |
| [MT-052](https://github.com/GonxKZ/mars-titan/issues/54) | Medir retención e interferencia | O3 | P1 | M | Pendiente | [MT-024](https://github.com/GonxKZ/mars-titan/issues/24), [MT-030](https://github.com/GonxKZ/mars-titan/issues/30), [MT-019](https://github.com/GonxKZ/mars-titan/issues/19) |
| [MT-053](https://github.com/GonxKZ/mars-titan/issues/55) | Evaluar consolidación y replay | O3 | P2 | L | Pendiente | [MT-019](https://github.com/GonxKZ/mars-titan/issues/19), [MT-052](https://github.com/GonxKZ/mars-titan/issues/54), [MT-059](https://github.com/GonxKZ/mars-titan/issues/61) |
| [MT-054](https://github.com/GonxKZ/mars-titan/issues/56) | Evaluar pasos recurrentes y parada | O4 | P1 | L | Pendiente | [MT-023](https://github.com/GonxKZ/mars-titan/issues/23), [MT-028](https://github.com/GonxKZ/mars-titan/issues/28), [MT-032](https://github.com/GonxKZ/mars-titan/issues/32) |
| [MT-055](https://github.com/GonxKZ/mars-titan/issues/57) | Evaluar destilación temporalmente válida | O4 | P2 | L | Pendiente | [MT-030](https://github.com/GonxKZ/mars-titan/issues/30), [MT-054](https://github.com/GonxKZ/mars-titan/issues/56), [MT-059](https://github.com/GonxKZ/mars-titan/issues/61) |
| [MT-056](https://github.com/GonxKZ/mars-titan/issues/58) | Auditar indicadores macro y sus versiones | O1 | P1 | L | Pendiente | [MT-003](https://github.com/GonxKZ/mars-titan/issues/3), [MT-005](https://github.com/GonxKZ/mars-titan/issues/5) |
| [MT-057](https://github.com/GonxKZ/mars-titan/issues/59) | Integrar macro con versiones y disponibilidad | O1 | P1 | L | Pendiente | [MT-011](https://github.com/GonxKZ/mars-titan/issues/11), [MT-056](https://github.com/GonxKZ/mars-titan/issues/58) |
| [MT-058](https://github.com/GonxKZ/mars-titan/issues/60) | Versionar calendarios y reglas de mercado | O2 | P1 | L | Pendiente | [MT-005](https://github.com/GonxKZ/mars-titan/issues/5), [MT-013](https://github.com/GonxKZ/mars-titan/issues/13) |
| [MT-059](https://github.com/GonxKZ/mars-titan/issues/61) | Comprobar compatibilidad de las versiones | O3 | P0 | L | Pendiente | [MT-010](https://github.com/GonxKZ/mars-titan/issues/10), [MT-021](https://github.com/GonxKZ/mars-titan/issues/21), [MT-022](https://github.com/GonxKZ/mars-titan/issues/22) |
| [MT-060](https://github.com/GonxKZ/mars-titan/issues/62) | Medir el piloto y fijar el presupuesto | O6 | P0 | L | Pendiente | [MT-004](https://github.com/GonxKZ/mars-titan/issues/4), [MT-023](https://github.com/GonxKZ/mars-titan/issues/23), [MT-026](https://github.com/GonxKZ/mars-titan/issues/26), [MT-027](https://github.com/GonxKZ/mars-titan/issues/27) |
| [MT-063](https://github.com/GonxKZ/mars-titan/issues/65) | Medir concurrencia y antigüedad del estado | O3 | P2 | L | Pendiente | [MT-053](https://github.com/GonxKZ/mars-titan/issues/55), [MT-059](https://github.com/GonxKZ/mars-titan/issues/61), [MT-060](https://github.com/GonxKZ/mars-titan/issues/62) |
| [MT-064](https://github.com/GonxKZ/mars-titan/issues/66) | Evaluar ampliaciones del universo | O4 | P2 | L | Pendiente | [MT-030](https://github.com/GonxKZ/mars-titan/issues/30), [MT-048](https://github.com/GonxKZ/mars-titan/issues/50), [MT-060](https://github.com/GonxKZ/mars-titan/issues/62), [MT-024](https://github.com/GonxKZ/mars-titan/issues/24), [MT-025](https://github.com/GonxKZ/mars-titan/issues/25) |
| [MT-065](https://github.com/GonxKZ/mars-titan/issues/67) | Verificar checkpoints y reanudación | O5 | P0 | L | Pendiente | [MT-004](https://github.com/GonxKZ/mars-titan/issues/4), [MT-030](https://github.com/GonxKZ/mars-titan/issues/30) |
| [MT-066](https://github.com/GonxKZ/mars-titan/issues/68) | Verificar fuentes públicas complementarias | O1 | P1 | M | Pendiente | [MT-005](https://github.com/GonxKZ/mars-titan/issues/5) |
| [MT-067](https://github.com/GonxKZ/mars-titan/issues/69) | Implementar actualizaciones con instantáneas inmutables | O1 | P2 | M | Pendiente | [MT-011](https://github.com/GonxKZ/mars-titan/issues/11), [MT-066](https://github.com/GonxKZ/mars-titan/issues/68) |

La preparación de MT-041 y MT-042 puede requerir adaptar el alcance al calendario real con el director, conservando una descripción veraz del avance. Las fechas se incorporarán desde el campus. El catálogo no declara presentada ninguna entrega ni concedida ninguna autorización.

## Consolidaciones e historial

La [revisión global](backlog-review.md) explica la función de cada tarea y las cinco consolidaciones aplicadas. El [registro de retiradas](../../.github/planning/retired-issues.json) conserva texto anterior, etiquetas, dependencias e identificador del antiguo elemento del proyecto. Las issues siguen accesibles y pueden añadirse de nuevo al tablero si se revisa la decisión.

| Issue retirada | Trabajo gestionado en | Motivo |
| --- | --- | --- |
| [MT-047](https://github.com/GonxKZ/mars-titan/issues/49) | [MT-005](https://github.com/GonxKZ/mars-titan/issues/5) | El inventario completo queda en MT-005. La auditoría por modalidad y la muestra mantienen sus tareas específicas. |
| [MT-049](https://github.com/GonxKZ/mars-titan/issues/51) | [MT-064](https://github.com/GonxKZ/mars-titan/issues/66) | La decisión y ejecución de ampliaciones se concentran en MT-064, reutilizando las referencias de MT-024 y MT-025. |
| [MT-051](https://github.com/GonxKZ/mars-titan/issues/53) | [MT-019](https://github.com/GonxKZ/mars-titan/issues/19) | La selección por sorpresa, relevancia y diversidad se gestiona en MT-019. La capacidad y recuperación permanecen en MT-018. |
| [MT-061](https://github.com/GonxKZ/mars-titan/issues/63) | [MT-037](https://github.com/GonxKZ/mars-titan/issues/37) | Las perturbaciones de entradas y la abstención forman parte del análisis de robustez de MT-037. |
| [MT-062](https://github.com/GonxKZ/mars-titan/issues/64) | [MT-029](https://github.com/GonxKZ/mars-titan/issues/29) | Las falsaciones del candidato se ejecutan con las ablaciones de desarrollo de MT-029, con antecedentes en MT-003 y diseño en MT-028. |

Las retiradas están cerradas con motivo `not_planned` y conservan la etiqueta `duplicate` ya existente en GitHub. No se marcan como experimentos completados ni forman parte de las 62 tareas activas.

## Etiquetas de dominio

El catálogo utiliza 20 etiquetas. Cada tarea activa tiene entre dos y cinco. Se han incorporado `macroeconomia` para indicadores y revisiones, `aprendizaje-continuo` para retención y replay, `checkpoints` para recuperación y `fuentes-externas` para adquisición complementaria. Las etiquetas ajenas al catálogo se conservan. Prioridad, tamaño, hito y estado siguen siendo campos separados.

## Cierre y mantenimiento

Al comenzar una tarea se revisan sus dependencias y la vigencia de sus supuestos. Al cerrarla se enlazan el cambio, las comprobaciones, los artefactos y la decisión correspondiente. Si aparece una necesidad nueva, se incorpora con identificador propio y se explica su relación con O1-O6. No se amplía silenciosamente una tarea en curso.

Las cuestiones de permisos, autoría y uso de herramientas se resuelven conforme a las fuentes académicas y al uso real. No se publican declaraciones ni autorizaciones que no estén acreditadas. La memoria final y la defensa deben reflejar la evidencia conseguida, incluso cuando esta no respalde la superioridad de MARS-TITAN.
