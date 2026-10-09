# Entornos financieros por lotes y contexto causal

`FinancialBatch` ejecuta hasta 4096 carteras independientes mediante la sesión financiera C++20 existente. Cada cartera conserva su efectivo, posiciones, órdenes, acciones aplicadas y cursor. Las cintas de mercado se comparten como datos de solo lectura. Repetir una cinta permite medir el motor, pero no crea nuevas trayectorias independientes ni aumenta la información del dataset.

El lote admite 1, 2, 4 u 8 trabajadores persistentes. Una llamada recibe las seis acciones financieras ya definidas y devuelve recompensas y máscaras por entorno. Las observaciones forman una matriz contigua `[entorno, característica]`. La paralelización se limita a preparar estados independientes. El controlador confirma todo el lote cuando han terminado correctamente el cálculo contable y las observaciones. Un fallo de cualquier entorno conserva el estado confirmado de todos.

## Uso y límites

Las interfaces públicas están en `native/include/mars_titan/financial_batch.hpp` y `markov_filter.hpp`. El consumidor C++ enlaza `mars_titan_financial`. La compilación pura no necesita Python, Arrow, CUDA ni una nueva biblioteca matemática:

```bash
cd native
cmake --preset native-release -DMARS_TITAN_BUILD_RUNNER=OFF -DMARS_TITAN_BUILD_FINANCIAL=ON
cmake --build --preset native-release
ctest --preset native-release
../build/native/native-release/mars-titan-batch-benchmark --environments 4096 --assets 64 --sessions 128 --workers 4
```

`--reference` ejecuta las mismas acciones con sesiones independientes en secuencia. `--rules cn` declara en la cinta lote de 100, venta del resto impar, banda del 10 % y timbre de venta, y `--capital` fija el capital inicial. Sirven para medir el coste de las [reglas de mercado](native-financial-simulation.md#reglas-de-mercado). Sin ellas, la cinta y el checksum son los anteriores. El ejecutable es un diagnóstico técnico sobre una cinta generada de contabilidad conocida. Sus resultados no son retornos de una estrategia entrenada. Las cifras, paridad y condiciones se recogen en el [informe de medición](../../reports/resources/batched-rl-performance.md).

Los entornos de un lote comparten el orden de activos, la moneda, la partición y el esquema de contexto. Pueden recorrer cintas diferentes y tener distintas fechas de finalización. No se mezclan entrenamiento y validación. Cada cinta conserva las condiciones de admisión histórica del simulador original. Los OHLC reales siguen necesitando procedencia acreditada.

Cada instancia admite un único controlador. Sus llamadas públicas no deben competir entre sí. Los hilos internos trabajan sobre carteras distintas. Quien aporte una cinta tampoco puede modificarla mediante otra referencia mientras exista el lote. Los spans y el resultado de `step` se prestan hasta la siguiente operación que modifique el lote.

El límite configurable de 512 MiB comprueba el volumen previsto de buffers, contexto y estados antes de crear sesiones e hilos. No es una medición ni un límite de RSS del sistema operativo. Las cintas aportadas, sus metadatos, pilas de hilos, sobrecarga del asignador y snapshots retenidos por el consumidor tienen costes adicionales. `reserved_payload_bytes` informa de ese presupuesto de datos, mientras el diagnóstico informa separadamente de `VmHWM` en Linux.

`reset(indices)` reinicia exclusivamente los entornos solicitados y rechaza índices repetidos o inexistentes. No hay reinicio automático al terminar. La observación final sigue disponible para calcular el bootstrap de una truncación. Un cierre ausente produce una recompensa inválida, la ruina es una terminación y el límite de la cinta es una truncación. La pausa administrativa consiste en dejar de llamar a `step` y conservar el snapshot, sin introducir una transición ficticia.

`snapshot` conserva todas las carteras y los identificadores de contexto. `restore` valida todos los candidatos y sus observaciones antes de sustituir una cartera. Esta interfaz es una recuperación en memoria. El [ejecutor PPO nativo](native-ppo.md) añade política, optimizador, RNG, rollout parcial y manifiestos verificados para recuperar un entrenamiento. El lote no crea checkpoints de disco por paso.

## Información externa y estado oculto

Un `ContextTape` opcional declara una huella de origen, nombres únicos, unidades y valores por sesión. Cada característica contiene valor, presencia y fecha de disponibilidad. El lote rechaza valores no finitos, publicaciones posteriores al cierre y ausencias ambiguas. Una ausencia se representa mediante valor cero, máscara cero y fecha cero. En la observación, cada característica añade valor, máscara y antigüedad en días.

La huella es un contrato de procedencia aportado por el productor, como en `MarketTape`. La capa de archivos debe verificarla contra su contenido antes de construir estos objetos. Declarar una huella no autentica una fuente. Las noticias o cuentas simuladas siguen siendo ficticias y no rellenan modalidades reales ausentes.

`MarkovFilter` calcula probabilidades filtradas de un HMM con emisiones gaussianas diagonales. Admite de 1 a 16 estados y de 1 a 64 variables. Recibe parámetros congelados, observaciones con máscaras y fechas, y decisiones en orden temporal estricto. El primer paso condiciona el prior y los siguientes aplican transición y emisión. La normalización logarítmica evita perder colas que pueden volver a ganar probabilidad. El snapshot conserva también los parámetros, cursor y probabilidades logarítmicas.

El filtro no ajusta parámetros, no suaviza con observaciones futuras y no conoce el estado verdadero de un mercado. Aplica una transición por observación, con independencia de la distancia entre fechas. Utiliza diferencias de emisiones factorizadas y rechaza atómicamente cancelaciones mal condicionadas. Una integración debe ajustar transiciones, emisiones y normalizadores exclusivamente con el pasado autorizado de cada partición. Los estados latentes tampoco convierten automáticamente el problema financiero en un proceso de Markov observado. El [diseño de regímenes](../research/markov-regimes.md) define los controles frente a contexto continuo y ausencia de régimen.

## Rollouts y especialistas

`generalized_advantage` admite `[tiempo]` y `[tiempo, entorno]`, calcula en `float64` y mantiene separadas las recurrencias de cada entorno. Una terminación anula el bootstrap y el arrastre. Una truncación solo corta el arrastre entre episodios. Las máscaras deben ser booleanas y recompensas y valores deben ser finitos. No se aplana el lote antes de calcular las ventajas.

El [entrenador PPO C++20](native-ppo.md) ya recoge decisiones por lotes con LibTorch, calcula el bootstrap antes de confirmar la contabilidad y conserva recorridos parciales. Valida el estado inicial y las actualizaciones con una política greedy sobre fuentes sintéticas separadas. Double DQN conserva su implementación previa. Las medidas del motor siguen sin acreditar una aceleración del entrenamiento completo, que incluye inferencia, transferencias, optimizador, validación y persistencia. La campaña GPU activa se ejecuta desde su versión congelada.

El contexto fechado puede entrar en el entrenador mediante los archivos `context.json` y `context.parquet` de cada cinta. El filtro HMM no se conecta automáticamente a la política. Sus probabilidades necesitarían un productor de contexto que respete el corte temporal y acredite sus parámetros y fuentes.

La [revisión de expertos y contexto](../research/contextual-experts.md) propone comparar dos especialistas con un modelo de capacidad equivalente, una media y una mezcla constante. Registra controles para separar el efecto del contexto, la memoria y el coste adicional. El candidato MARS-TITAN, sus especialistas y su posible transferencia siguen pendientes de implementación y validación.
