# Mecanismos de MARS-TITAN-CM-v1

Esta entrega contiene dos mecanismos matemáticos independientes. C estima el radio numérico de una matriz proporcionada y calcula una penalización. M selecciona representantes reales de un conjunto de candidatos. Las funciones están en [`numerical_radius.py`](../../../src/mars_titan/cm/numerical_radius.py) y [`medoids.py`](../../../src/mars_titan/cm/medoids.py). Sus pruebas están en [`tests/cm`](../../../tests/cm).

El [control local de MAC](mac_local_control.md) conecta C con una compresión del Jacobiano de su transición rápida completa, incluidos pesos y momentum. Admite diagnóstico y penalización heurística y tiene comprobaciones CPU/CUDA. El [consumidor financiero](../../engineering/financial-session-v2.md) conecta núcleo, lectura y banco bajo M0/M1 y admite retenciones configuradas. [M2](../../engineering/mature-error-write-policy.md) añade tres índices de escritura, con la consolidación M desactivada. La GRU episódica conserva su papel de referencia separada y admite la [entrada histórica con máscaras](../../../native/candidate_historical.md). El [protocolo](protocol.md) distingue las composiciones técnicas comprobadas del factorial de aprendizaje pendiente. El [factorial](factorial.md) fija esa nueva identidad de B, con SDPA Math y `fastpath=False` declarado por cada ejecutor. Implementa la penalización C en el ajuste del núcleo, la retención M en el banco del lector y sus ventanas en la campaña, sin ejecutarlos. Los Jacobianos completos de MAC y del refinamiento están en [`transition_jacobian.py`](../../../src/mars_titan/models/titans/transition_jacobian.py) y las lecturas de operador fijo y de productos variables en [`operator_dynamics.py`](../../../src/mars_titan/cm/operator_dynamics.py).

## Contrato de C

`numerical_radius_estimates(matrix, *, grid_size=64, angle_block_size=8, max_estimated_bytes=134217728)` recibe un tensor PyTorch cuadrado, real o complejo, con tipos float32, float64, complex64 o complex128. Admite dimensiones de lote, hasta 16 matrices, orden 1–256 y rejilla de 2–1024 ángulos. Trabaja en complex128 sobre el dispositivo de entrada. La conversión conserva los gradientes.

Devuelve cuatro cantidades por matriz:

- `grid_lower_estimate`, máximo de los autovalores hermíticos muestreados.
- `angular_corrected_estimate`, estimación anterior más la corrección analítica que utiliza la norma de Frobenius.
- `frobenius_norm`, utilizada por esa corrección.
- `spectral_norm`, control independiente de norma de operador.

Los nombres expresan estimaciones de coma flotante. La rejilla y la corrección tienen desigualdades en aritmética exacta, pero la implementación no aporta certificados del error del solver o del redondeo. La [justificación matemática](mathematical_scope.md) distingue esas afirmaciones.

`radius_penalty(estimates, threshold, measure='angular_corrected_estimate')` devuelve la bisagra cuadrática `max(measure - threshold, 0)²`, sin reducir las dimensiones de lote. Admite también la estimación de rejilla y la norma espectral. El umbral debe ser finito y no negativo. La función no modifica pesos ni selecciona una matriz arquitectónica.

La memoria estimada incluye tensores de entrada convertidos, bloques de ángulos y tensores retenidos por autograd. Se comprueba el presupuesto antes del solver. La estimación no incluye toda la memoria interna de LAPACK, CUDA o del proceso. Los máximos de autovalores múltiples y los empates de ángulos pueden ser no diferenciables. Autograd puede devolver una elección finita sin que exista un gradiente único.

## Contrato de M

`select_medoids(client_points, client_ids, candidate_points, candidate_ids, capacity, *, backend='greedy_swap', metric='euclidean', ...)` recibe dos matrices NumPy con la misma dimensión de representación. Clientes y candidatos pueden ser conjuntos diferentes. Los IDs son cadenas no vacías, únicas dentro de cada conjunto y de hasta 256 bytes UTF-8. Un ID compartido debe tener exactamente la misma representación. Las coordenadas no finitas se rechazan.

Los duplicados geométricos con distintos IDs conservan su multiplicidad en el coste. La salida contiene IDs e índices de las filas originales de candidatos. No crea centroides, no combina episodios y no recibe etiquetas. El consumidor debe acreditar que cada cliente y candidato era elegible antes de la consulta correspondiente.

`euclidean` calcula la distancia euclídea con raíz, usando `numpy.hypot`. `l1` suma diferencias absolutas. Las coordenadas enteras L1 se normalizan a int64 y se comprueba el desbordamiento de las distancias antes de calcularlas. El coste agregado utiliza enteros Python. Las coordenadas enteras euclídeas se aceptan solo dentro de ±2⁵³ para evitar pérdidas silenciosas al convertir a float64. No se mezclan coordenadas enteras y flotantes entre ambos conjuntos. La cuantización, cuando exista, debe definirla el consumidor antes de esta API.

Capacidad cero produce error incluso con clientes vacíos. Con capacidad positiva y clientes vacíos se devuelve selección vacía y coste cero. Con clientes y sin candidatos se produce error. Si la capacidad alcanza el número de candidatos se conservan todos. Un presupuesto de trabajo o memoria agotado lanza `MedoidBudgetExceeded`, sin devolver un supuesto óptimo.

El backend `greedy_swap` añade candidatos de forma greedy y aplica intercambios unitarios que reducen estrictamente el coste. Recalcula cada conjunto aceptado. Ordena clientes y candidatos por ID para resolver empates y sumar en un orden reproducible. Devuelve `one_swap_local` si una búsqueda completa no encuentra mejoras o `swap_limit` si termina por el máximo de intercambios. Ese último estado identifica un resultado parcial. Ninguno implica optimalidad global.

El backend `enumeration` evalúa todos los subconjuntos del tamaño efectivo. Devuelve `enumerated`. Para L1 entera, el oráculo es exacto sobre el objetivo entero. Para distancias flotantes, compara los costes calculados en float64. Las sumas usan `math.fsum` dentro y entre bloques canónicos de clientes. Cambiar el tamaño de los bloques puede cambiar el último bit y la elección entre costes casi iguales.

El presupuesto predeterminado limita los buffers propios estimados a 64 MiB, los pares de distancia a 50 millones, las combinaciones a 10.000 y los intercambios a 100. Los bloques máximos solicitados son 4.096 clientes y 16 candidatos. Ambos ejes se reducen si hace falta. Se cuentan copias de coordenadas, índices y metadatos, vecinos primero y segundo, propietarios, bloques y sumas auxiliares. No se construye una matriz global de distancias. La estimación de buffers no equivale al RSS total. La salida registra bytes estimados, tamaño de bloque, pares, evaluaciones del objetivo e intercambios.

Las comprobaciones ejecutadas y las dependencias pendientes constan en [resultados](results.md). La [auditoría de fuentes](source_audit.md) delimita qué resultados publicados respaldan el análisis y cuáles no se han reproducido.

La [retención con centros fijos](anchored_retention.md) amplía la selección mediante un fondo de distancias y candidatos variables restringidos. Mantiene el objetivo sobre todo el conjunto de clientes actual. El banco la ofrece como política `anchored` y el consumidor proporciona codec fijo y publicación conjunta. La existencia de esta ruta no acredita haber entrenado ni evaluado el factorial CM completo.
