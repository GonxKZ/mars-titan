# Lectura episódica posterior a MAC

`models.titans.episodic_readout` implementa un operador puro para refinar el estado de una predicción. Recibe `working_state`, la instantánea de episodios maduros y un contrato explícito. No ejecuta MAC, admite episodios, publica checkpoints ni recorre el corpus. Es un componente para la integración con el consumidor financiero, que sigue pendiente.

## Instantánea y lectura

`EpisodeSnapshot.create` recibe tensores CPU: claves y valores FP32 de 64 coordenadas, labels FP64, IDs y fechas int64. Exige IDs positivos crecientes, claves ya normalizadas por el banco y `0 ≤ available_at ≤ decision_at < maturity_at ≤ cutoff`. El llamante acredita la procedencia de esas fechas. Los metadatos incluyen codec y contexto de cohorte, ambos con identidad SHA256.

La instantánea contiene copias propias sin grafo de hasta 1024 episodios. Mantiene claves FP64 para la selección, valores en la precisión del lector y labels FP64. Claves, valores, labels e IDs se trasladan al dispositivo solicitado. Las tres fechas permanecen en CPU. La creación no renormaliza ni acepta otro formato de claves con las mismas dimensiones.

`export_cpu` y `restore` contrastan el digest de los bytes, configuración, precisión, codec y contexto. La recuperación no reconstruye un grafo de autograd. En cada lectura se comprueban las versiones y propiedades de los tensores sin transferir sus contenidos a CPU. Ese control barato no detecta toda escritura posible mediante `.data`. `verify` y la exportación realizan la comprobación fuerte en las fronteras. Los tensores internos no deben modificarse. Ni la clase congelada ni el contador de versiones constituyen una protección contra manipulación arbitraria de objetos Python.

Cada iteración consulta la misma instantánea completa en el dispositivo. La selección calcula puntuaciones FP64, ordena de forma estable y toma hasta ocho vecinos. Los IDs crecientes resuelven empates exactos a favor del menor ID. Top-k no tiene gradiente. Dentro del conjunto elegido, los pesos `softmax(q·key/temperature)` usan la precisión del modelo y conservan el gradiente de la consulta.

La consulta es una proyección con bias, normalizada mediante escalado por la magnitud máxima y un suelo de norma `1e-12`. La consulta cero devuelve pesos uniformes entre los vecinos elegidos y mantiene gradientes finitos. Los valores proyectados concatenan 64 rasgos estables y el label maduro. Una conversión a FP32 que produzca infinito causa error. No se utiliza el error financiero como característica implícita.

El banco vacío devuelve lectura cero, pesos e IDs vacíos y presencia falsa. Un episodio de valor cero mantiene presencia verdadera. Las lecturas ausentes con relleno no nulo, NaN o infinito se rechazan.

## Refinamiento y controles

Con `z_0 = working_state`, cada paso aplica

```text
q_k = normalize(query_projection(z_k))
r_k = read(snapshot, q_k)
z_(k+1) = z_k + sigmoid(step_logit) * tanh(W[z_k, z_0, r_k, presence] + b)
```

La forma residual procede de la [referencia GRU](../../native/candidate.md). La adaptación conserva la anchura D del núcleo financiero, claves y valores de 64 y su cabeza escalar. No comparte el espacio 128×256 ni la cabeza de cuantiles del candidato GRU. Todas las proyecciones tienen bias, no hay dropout y el paso inicial es 0,1. La temperatura predeterminada es 1. K es 1, 2 o 4 y reselecciona vecinos globalmente en cada iteración. No vuelve a escribir la memoria neuronal de Titans.

`mode="bank"` exige una instantánea, incluso cuando está vacía. `mode="no_bank"` conserva el mismo bloque y recibe lectura cero y presencia falsa. Ese bloque todavía puede modificar el estado. `copy_readout_parameters` empareja sus parámetros mediante una copia y un recibo de hashes, sin transferir episodios. Las cargas ordinarias rechazan otros modos, K, codec o precisión, aunque coincidan las formas.

`apply_episodic_readout(prepared, head, extension=None)` devuelve exactamente el tensor `prepared.point_predictions`, sin repetir la cabeza. Una ampliación activa consume `prepared.working_state` y aplica la cabeza recibida al estado refinado. Esta composición no llama al núcleo ni altera `prepared.next_state`. Omitir la ampliación y conservar un refinador sin banco son controles distintos.

El modo predeterminado devuelve resultados desacoplados. `differentiable=True` conserva el grafo del estado de trabajo actual y los parámetros, pero no añade un grafo a episodios persistidos. Exige que `inference_mode` esté desactivado. Para propagar hasta el núcleo, la preparación original también debe conservar su grafo. No hay un optimizador ni aprendizaje interno en este lector.

M0–M3, la admisión económica, régimen e incertidumbre no se implementan ni se seleccionan en este módulo. C sigue referido a la transición rápida MAC anterior a esta lectura. No controla el refinamiento episódico completo.

## Presupuesto y comprobaciones

El lote admite hasta 256 filas y D hasta 128. La instantánea tiene un presupuesto predeterminado de 2 MiB y máximo de 8 MiB, aplicado a fuente retenida y destino antes de copiar. `usage` desglosa claves, valores, labels, IDs y fechas. Se cuenta por separado una reserva de 1024 bytes para metadatos.

El lector estima puntuaciones, índices, consultas, valores elegidos, estados de K y una reserva para el grafo solicitado. El presupuesto predeterminado es 32 MiB, con máximo de 128 MiB. Se valida antes de proyectar, tanto en `forward` como en `read`. Algunas combinaciones grandes con diferenciación requieren un presupuesto explícito mayor. El snapshot residente y los parámetros tienen cuentas independientes. Estas estimaciones no son cotas del RSS, del asignador o de los buffers internos de biblioteca.

Las pruebas CPU incluyen ecuación manual, gradcheck FP64, gradientes de consulta, proyección, refinamiento y cabeza, reselección de vecinos, empates, máscaras, aislamiento y recuperación. Con un único vecino se comprueba que el gradiente de selección de la consulta es cero. Las pruebas de composición conservan una sola escritura MAC aunque K sea 4. Se conservan las regresiones de consulta cero, puerta no finita, presupuesto y ausencia de grafo.

El [recibo CPU](../../reports/research/episodic-readout-verification-20261008.json) recoge 407 casos, 78 propios, y 12 mutaciones detectadas. La medición técnica separa instantánea, lector y dos grafos de ejemplo. No incluye la sesión completa, disco, MAC ni cabeza. No demuestra utilidad predictiva ni aceleración de la canalización.

La comprobación manual `tests/models/titans/cuda_episodic_readout_check.py` prepara seis casos FP32/FP64 con K1/2/4, gradientes y recuperación. No forma parte de la suite CPU. Su ejecución y la integración del consumidor quedan pendientes. El bloqueo de aprendizaje continúa vigente.
