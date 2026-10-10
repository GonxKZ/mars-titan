# PPO nativo con entornos financieros por lotes

`mars-titan-ppo` ajusta una política financiera en C++20 con autograd y Adam de LibTorch. Recoge decisiones de `FinancialBatch`, calcula ventajas por entorno y guarda el estado necesario para continuar. Las predicciones de entrada están congeladas. El ajuste de la política no modifica el predictor padre ni sus codificadores.

El ejecutable admite fuentes sintéticas de entrenamiento y validación. La comprobación CPU es un diagnóstico limitado a 32 transiciones de ajuste. La ruta de ejecución CUDA exige `cuda:0`, admisión exclusiva y presupuesto de VRAM. Esta herramienta permite comprobar el recorrido de aprendizaje y recuperación, pero no constituye una ejecución del candidato MARS-TITAN ni una comparación científica sobre FinMultiTime.

Esta guía describe el esquema 1, con MLP y réplicas equilibradas de hasta doce fuentes por partición. El [esquema 2 de adaptación](adaptive-rl.md) añade un catálogo de mundos rotatorio, calentamiento sin operaciones, controles de ventana y GRU, memoria episódica, HMM, consolidación auxiliar y Double DQN. Tiene su propio contrato de recuperación, trazas y auditoría final. El esquema 4 ajusta PPO y Double DQN sobre las cintas reconstruidas de la etapa de políticas y se describe con `mars-titan-klpo` en [políticas nativas sobre cintas reconstruidas](native-policy-real-tapes.md).

## Política y recorrido de aprendizaje

`PpoPolicy` utiliza dos capas densas de 64 unidades con activación `tanh`. Una salida compartida produce seis logits de acción y un valor del crítico. Los parámetros y las observaciones son FP32. La estimación de ventajas y el cociente de probabilidades del objetivo recortado se calculan en FP64.

Las decisiones conservan el contrato de la [simulación financiera](persistent-simulation.md):

| Acción | Decisión |
| --- | --- |
| `0` | Conservar posiciones y órdenes pendientes. |
| `1` | Solicitar el paso a efectivo. |
| `2` | Rebalancear con exposición objetivo del 25 %. |
| `3` | Rebalancear con exposición objetivo del 50 %. |
| `4` | Rebalancear con exposición objetivo del 75 %. |
| `5` | Rebalancear con exposición objetivo del 100 %. |

El motor distribuye la exposición entre el cuartil superior de las predicciones positivas, con pesos iguales y desempate por identificador. La política no elige pesos libres para cada empresa. Decide al cierre y la orden se ejecuta en la apertura siguiente, con las restricciones de efectivo, costes y volumen conocido del simulador. La exposición realizada puede diferir de la solicitada.

`PpoTrainer` mantiene el rollout en CPU con forma `[tiempo, entorno, característica]` y realiza inferencia para todo el lote. Durante el ajuste muestrea acciones de la distribución categórica. Conserva observaciones, acciones, probabilidades anteriores, valores, recompensas y máscaras. La MLP calcula GAE en FP64 con ATen en CPU y selecciona las filas válidas antes del traslado. La conversión a FP32 de los objetivos y su normalización siguen en el dispositivo de la política. La ruta recurrente traslada el recorrido y su prefijo. Ambas aplican minibatches con objetivo PPO recortado, pérdida del crítico, entropía y recorte de la norma del gradiente.

Las ventajas se calculan antes de aplanar tiempo y entorno. Una terminación por ruina anula el bootstrap y corta la recurrencia. Una truncación conserva el valor de la observación final y corta la recurrencia entre episodios. Una recompensa inválida se excluye del objetivo. Los reinicios se registran y se aplican después de conservar la observación final. La contabilidad y el bootstrap se validan antes de confirmar cada paso del lote. En la MLP sin contexto ni Double DQN, el paso siguiente muestrea de la salida de ese bootstrap en lugar de repetir el forward, siempre que la observación coincida byte a byte y no haya pasado ningún paso del optimizador. Una pausa, una restauración, una actualización o un fallo descartan esa salida. El muestreo y el estado del generador son los mismos que con el forward repetido.

La [configuración ordinaria](../../configs/simulation/native-ppo.json) fija 16 entornos, 8192 transiciones, recorridos de 1024 transiciones, cuatro épocas y minibatches de 64. Una llamada al entrenador confirma tantas transiciones como entornos contiene el lote. Repetir una cinta crea carteras con decisiones distintas, pero no añade trayectorias de mercado independientes. Se admiten las semillas de política 42, 43 y 44, con 42 en la configuración incluida.

## Fuentes, contexto y observabilidad

Cada fuente contiene `manifest.json` y `market.parquet`. El ejecutor comprueba partición, dominio sintético, huellas, activos, moneda y predictor padre. Entrenamiento y validación deben compartir el esquema. Se rechazan fuentes duplicadas y la reutilización de archivos de mercado o de las semillas del generador declaradas entre ambas particiones. Se admiten hasta doce fuentes por partición y las réplicas de entrenamiento se distribuyen de forma equilibrada.

La observación financiera contiene seis características por activo y dos de cuenta. Puede ampliarse mediante `context.json` y `context.parquet`. El manifiesto de contexto enlaza la huella de la cinta, los nombres y unidades de los campos y los bytes del Parquet. Cada campo añade valor, presencia y antigüedad. Su `available_at` debe ser anterior o igual al cierre en que se decide. Las ausencias usan valor, máscara y fecha cero. Un periodo contable terminado no acredita por sí solo que el dato estuviera publicado.

El lector limita el contexto a 512 campos, 64 MiB de archivo y 128 MiB decodificados. Comprueba el orden de sesiones y campos, los tipos, las máscaras explícitas y los hashes antes de construir el lote. Las noticias, fundamentales o variables macro simuladas siguen siendo ficticias. Estos datos no completan modalidades reales ausentes.

El estado interno permite reanudar la contabilidad sobre una cinta, mientras que la política recibe una observación parcial. Esa recuperabilidad no demuestra que la observación financiera cumpla la propiedad de Markov. En un modelo POMDP, el estado de creencia depende del modelo de transición y observación asumido. La [revisión de observabilidad y contexto](../research/contextual-experts.md) recoge esta distinción y la referencia de Kaelbling, Littman y Cassandra.

El [filtro HMM](batched-rl-environments.md#información-externa-y-estado-oculto) calcula probabilidades filtradas con parámetros congelados. El esquema 1 no lo ejecuta automáticamente. El [esquema 2](adaptive-rl.md#variantes-implementadas) sí lo integra en sus variantes HMM, tras verificar parámetros ajustados exclusivamente con entrenamiento. El [contraste de regímenes](../research/markov-regimes.md) conserva la separación entre filtrado con pasado, suavizado con futuro y una hipótesis de utilidad que todavía debe evaluarse.

## Validación y selección

El estado inicial θ₀ se guarda antes de la primera transición y se evalúa antes de cualquier actualización de Adam. Puede seguir siendo el mejor si ningún ajuste posterior mejora el criterio declarado. La evaluación usa `argmax` de los logits, sin muestreo, y recorre cada fuente de validación una vez. No reinicia un episodio terminado para darle más peso.

La selección prioriza un menor número de ruinas. Si ese número coincide, exige que el crecimiento logarítmico medio neto de costes supere al mejor en más de `min_delta`. La ruina utiliza la penalización configurada al calcular esa media, porque el logaritmo de un patrimonio cero no es finito. La métrica se identifica como `ruin_count_then_mean_log_growth`. El MAE de las predicciones congeladas no interviene en esta selección financiera.

Esa métrica valora la cartera final al cierre sin pagar su venta. Con 10 pb y exposición completa, la diferencia llega a 0,001, diez veces el `min_delta` configurado, y favorece a las políticas que terminan invertidas. La variante declarada `ruin_count_then_mean_liquidated_log_growth` usa el patrimonio tras vender todas las posiciones al último cierre con el coste de la sesión. Las configuraciones existentes conservan la métrica anterior y sus resultados no cambian. Una campaña nueva debe declarar la variante en su configuración, y entonces el mejor estado guarda también `mean_liquidated_log_growth`.

Una evaluación incompleta no tiene puntuación. Sus medias quedan como NaN en lugar del cero anterior, y la selección la rechaza antes de compararla. Las fuentes de ajuste y validación tampoco pueden contener cierres ausentes, porque una transición inválida ocultaría al objetivo la pérdida de la posición.

Solo cuenta una evaluación completa. Una pausa durante la validación conserva la evaluación pendiente. Se valida de nuevo después de una actualización del optimizador, sin repetirla por cada checkpoint administrativo. La configuración fija `min_delta=0.0001`, paciencia de cinco evaluaciones y `early_stopping=false`, por lo que mantiene el presupuesto de ajuste. Activar la parada temprana requiere declarar ese cambio y su efecto sobre la igualdad de actualizaciones de una comparación emparejada.

El mejor estado y el último estado de recuperación tienen funciones distintas. Terminar el presupuesto no sustituye el mejor por los últimos pesos. El test final permanece cerrado y su marca se comprueba en la configuración y en los manifiestos antes de leer el Parquet.

## Compilación y ejecución

La integración LibTorch está definida para Linux, acepta PyTorch `>=2.14,<3` y comprueba la ABI de libstdc++. CMake obtiene las cabeceras y bibliotecas del paquete instalado con `uv`. El programa enlaza LibTorch, Arrow/Parquet y OpenSSL, sin `libpython`. El lanzador Python se ocupa de la admisión y las señales, mientras el proceso C++ ejecuta la inferencia, el backward, el optimizador y la simulación.

Desde la raíz del repositorio:

```bash
uv sync --locked --extra cuda --extra reinforcement
cd native
cmake --preset native-ppo-release
cmake --build --preset native-ppo-release
ctest --preset native-ppo-release -j 1
cd ..
```

El binario queda en `build/native/native-ppo-release/mars-titan-ppo`. Los presets PPO compilan con un trabajo. `ctest --preset native-ppo-release` omite las pruebas con la etiqueta `optimizer-steps`, que aplican pasos de optimizador, y `native-ppo-release-learning` las incluye para cuando se levante el bloqueo de aprendizaje. El backend CUDA reutiliza las bibliotecas del wheel y no activa `nvcc` ni compila kernels propios. `MARS_TITAN_LIBTORCH_ENABLE_CUDA=OFF` permite construir una variante para los diagnósticos CPU. Los perfiles `native-ppo-debug`, `native-ppo-asan-ubsan`, `native-ppo-static-analysis` y `native-ppo-coverage` separan los diagnósticos de Release. MSan y TSan sobre el ejecutable completo requieren un runtime LibTorch instrumentado.

El preparador analítico crea escenarios con semillas distintas para entrenamiento y validación. Usa el mecanismo ficticio conocido, sin cargar un predictor entrenado:

```bash
uv run --no-sync python scripts/prepare_financial_scenarios.py \
  --output data/interim/native-ppo-scenarios-v1

uv run --no-sync python scripts/run_native_ppo.py \
  --config configs/simulation/native-ppo-diagnostic.json \
  --train-tape data/interim/native-ppo-scenarios-v1/known_signal-train-42 \
  --validation-tape data/interim/native-ppo-scenarios-v1/known_signal-validation-1042 \
  --output artifacts/native/ppo-diagnostic-v1 \
  --diagnostic
```

El diagnóstico usa dos entornos y 32 transiciones de ajuste. Para ejecutar el presupuesto ordinario sobre esas fuentes sintéticas, la orden es:

```bash
uv run --no-sync python scripts/run_native_ppo.py \
  --config configs/simulation/native-ppo.json \
  --train-tape data/interim/native-ppo-scenarios-v1/known_signal-train-42 \
  --validation-tape data/interim/native-ppo-scenarios-v1/known_signal-validation-1042 \
  --output artifacts/native/ppo-synthetic-v1
```

El lanzador rechaza el inicio si otra carga científica ocupa la GPU. Conserva un margen de 1 GiB y limita el presupuesto del proceso al menor valor entre 6 GiB, la memoria libre tras ese margen y el 75 % de la capacidad visible. El binario comprueba el bloqueo heredado y configura el allocator antes de crear tensores CUDA. Ese límite del allocator no incluye toda la memoria del contexto o de otras bibliotecas. No hay cambio automático de CUDA a CPU.

La salida debe ser nueva y quedar separada de sus fuentes. `--resume` recupera la misma configuración y salida. `--stop-after` solicita una pausa al alcanzar una barrera del lote. SIGINT y SIGTERM también solicitan una pausa recuperable. Cambiar datos, dispositivo, configuración, versión de LibTorch o identidad de compilación exige otra ejecución.

## Estado persistente y límites de la evidencia

El checkpoint incluye pesos, estado de Adam, RNG de acciones y barajado, rollout parcial, cursores, contabilidad, reinicios pendientes y progreso de selección. `PpoCheckpointStore` verifica los hashes de metadatos, política y rollout antes de entregarlos al lector de LibTorch. La identidad es inmutable y un bloqueo exclusivo protege la salida.

Se conservan dos checkpoints recientes y el mejor si es distinto, hasta tres bundles confirmados. El mejor solo cambia tras una mejora validada. La publicación atómica del índice precede a la limpieza de estados propios expulsados. Si el último bundle está corrupto, puede recuperarse el anterior íntegro y el recibo identifica el cambio. Un índice corrupto se rechaza. Las escrituras interrumpidas reutilizan únicamente archivos que coincidan con su contenido esperado.

El almacén admite 128 MiB por archivo binario y 32 MiB de metadatos, hasta 288 MiB por bundle y 864 MiB de contenido retenido. La publicación puede necesitar un cuarto bundle y un temporal de hasta 128 MiB. El lector de la política aplica además su límite de 64 MiB. Los presupuestos de buffers y archivos no equivalen al RSS total. `run.json` registra transiciones, pasos de Adam, evaluaciones, selección, recibo y recursos. El pico de VRAM queda sin valor cuando no se ha medido.

Las pruebas de política, entrenador, archivos y lanzador contrastan fórmulas controladas, máscaras, recuperación, rechazo de datos futuros, corrupción y admisión de recursos. El [informe de verificación](../../reports/resources/native-ppo-verification.md) registra los casos ejecutados, la cobertura, los sanitizadores y sus límites. Las pruebas CPU pequeñas no acreditan el rendimiento del entrenamiento CUDA. Las [medidas del simulador por lotes](../../reports/resources/batched-rl-performance.md) tampoco miden este recorrido completo.

Esta entrega no añade una mezcla de expertos ni entrena el candidato MARS-TITAN. La utilidad del contexto, la generalización a datos reales y la comparación con otros métodos siguen sujetas al [protocolo de investigación](../research/protocol.md). El éxito de una prueba sintética verifica un comportamiento del programa, no una mejora de predicción o rentabilidad futura.
