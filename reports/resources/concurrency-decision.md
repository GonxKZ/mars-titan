# Decisión sobre actualización concurrente de estados

El 20 de octubre de 2026 se decide no activar el experimento de doble búfer y actualización concurrente de #65 en el alcance actual. Se conserva la ejecución secuencial y la publicación coherente de estados. Esta decisión no prohíbe el procesamiento por lotes, la precarga acotada ni las optimizaciones internas que mantengan el mismo contrato.

La tarea estaba condicionada a una política de actualización útil y a un incumplimiento medido que el solapamiento pudiera resolver. La comparación de consolidación y replay de #55 sigue pendiente. Los parámetros compartidos permanecen congelados durante la evaluación principal. Un reajuste concurrente de esos parámetros cambiaría el protocolo de ese tramo. La adaptación interna de Titans y la admisión de etiquetas maduras conservan su orden establecido.

La [ronda de refinamientos](../research/refinement-development-20261019/completed-v1/README.md) terminó 288 tareas. Ninguno de sus 48 selectores superó el control residual cero en MAE. Ese resultado no demuestra que ninguna política de memoria pueda ayudar, pero tampoco justifica añadir concurrencia para desplegar esas configuraciones.

Las [medidas de latencia](../engineering/refinement-selected-latency-20261019/README.md) comprobaron doce casos con entradas del corpus codificado. Para K1 y 256 activos, las medianas entre pasadas de p99 fueron 25,153 ms con los gráficos originales y 28,273 ms con la representación geométrica. Con 64 activos fueron 13,576 y 15,622 ms. Estos valores no incluyen producir los embeddings originales y no representan una petición individual ni todo el universo histórico.

El perfil de llamadas señaló preparación y decodificación de observaciones como candidatos a revisión. `_event` acumuló 59,455 s de una fase perfilada de 114,811 s. `_sample_group`, incluido en recorridos anteriores, acumuló 27,193 s. Son tiempos anidados y no se suman. No se ha demostrado que el doble búfer reduzca ese coste.

El objetivo exploratorio de p99 de 5 ms para un activo y el presupuesto completo de todas las señales siguen sin comprobarse. Esta decisión no declara satisfechos todos los límites ni extrapola de 256 activos al universo completo. La medición y el presupuesto pendientes se mantienen en #62.

No se han ejecutado comparaciones entre dos buffers, pruebas de saturación de esa arquitectura ni medidas de su antigüedad. No se publican cifras inventadas para esos casos. Los artefactos de entrenamiento, inferencia y recuperación existentes se conservan. La reserva final permanece cerrada.

La extensión podrá reabrirse cuando una actualización admitida por el protocolo muestre utilidad, su ejecución secuencial incumpla un límite relevante y el perfil sitúe ese trabajo en el recorrido que interesa reducir. Su nueva comparación tendrá que incluir ambos buffers, colas, optimizador, activaciones, transferencias, recuperación y efectos sobre p95/p99, antigüedad, error y cobertura. Una mejora de la media no bastará.

El cierre corresponde a la decisión de no activar esta extensión condicional. No acredita una implementación concurrente ni completa las tareas de consolidación, retención o presupuesto de la campaña.
