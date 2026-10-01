# Simulación de posiciones y comparadores financieros

El módulo `simulation` evalúa reglas financieras sobre predicciones congeladas. No se conecta a un bróker. El error predictivo sigue siendo la medida principal de los modelos. Los resultados de esta simulación se publican en un grupo separado.

`Portfolio` mantiene efectivo por moneda, cantidades, órdenes pendientes, derechos de cobro, costes acumulados y última valoración. Una decisión tomada al cierre se ejecuta en la apertura siguiente. El motor vende antes de comprar y reparte el efectivo disponible entre las compras, respetando lotes y costes. No admite deuda ni posiciones cortas. El límite de participación utiliza el volumen conocido en el cierre de decisión, no el volumen futuro. Es una aproximación experimental de liquidez, no una reconstrucción de la profundidad del mercado.

Los splits modifican cantidades y órdenes. Los dividendos generan un derecho de cobro antes de operar en la fecha exdividendo y pasan a efectivo en la fecha de pago. Ambos tratamientos requieren acciones acreditadas. Una baja sin recuperación requiere una acción explícita. El motor no interpreta una cotización ausente como una pérdida total ni vende a un precio anterior para cerrar la cuenta.

Si falta el cierre de una posición, su patrimonio queda sin valorar y termina la evaluación con `completed=false`. Esta transición no entra en el objetivo de entrenamiento. La ruina conocida termina el episodio y utiliza la penalización fijada de −20, porque el logaritmo de cero no es finito. El resultado financiero de esa ruina conserva un retorno de −1. El final de un periodo es un truncamiento y no liquida la cartera. Una pausa administrativa guarda el estado sin crear una transición.

## Datos y decisiones

`MarketTape` conserva precios OHLCV, calendarios de apertura y cierre y predicciones disponibles antes de cada decisión. Ordena los activos por identificador. La versión persistente es un Parquet con grupos de hasta 16 sesiones y un manifiesto con huellas del contenido. La lectura comprueba integridad y limita el tamaño descomprimido. Las modalidades originales no se duplican en cada transición.

El histórico real exige precios sin ajustar y evidencia explícita de sus ajustes, calendario y acciones corporativas. La auditoría existente no acredita estos extremos para todo el corpus. Por ello, esta entrega solo verifica la simulación sobre datos sintéticos de contabilidad conocida. Una marca de procedencia en un archivo no sustituye la revisión de esa evidencia.

Las seis acciones son conservar órdenes y posiciones, pasar a efectivo y rebalancear al 25 %, 50 %, 75 % o 100 %. Se elige el cuartil superior de las predicciones positivas, con pesos iguales y desempate por identificador. Las cantidades se calculan al cierre. El salto hasta la apertura y los costes pueden hacer que la exposición ejecutada difiera de la solicitada. La política no asigna pesos libres por empresa.

La observación contiene, por activo, predicción, peso, último retorno de cierre, volumen conocido, orden pendiente y máscara. Añade proporciones de efectivo y derechos de cobro. No contiene etiquetas futuras. La recompensa ordinaria es la diferencia del logaritmo del patrimonio neto de costes.

## Comparación y recuperación

`configs/simulation/comparators.json` fija semillas 42, 43 y 44, dos capas de 64 unidades y 8192 decisiones por ajuste. PPO usa recorridos de 128 pasos y cuatro épocas por recorrido. Double DQN mantiene un replay de 4096 observaciones, elige la acción con la red actual y la valora con la red objetivo. El replay tiene un límite adicional de 128 MiB. Las referencias conservan efectivo, mantienen la asignación inicial o rebalancean al 50 %. La evaluación repite los periodos con 0, 10 y 25 puntos básicos por operación.

PPO conserva el bootstrap al truncar y corta la recurrencia de ventajas entre episodios. Double DQN anula el bootstrap al terminar por ruina. Esta distinción sigue el [contrato de Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/). Los algoritmos se basan en [PPO](https://arxiv.org/abs/1707.06347) y [Double DQN](https://arxiv.org/abs/1509.06461).

Cada punto de control incluye redes, optimizador, RNG, replay o recorrido PPO parcial, cursor y estado contable. Se confirma un estado inicial antes de la primera transición. El índice mantiene dos estados íntegros y permite recuperar el anterior si el último archivo está dañado. Cambiar código, fuentes o presupuesto exige otra ejecución. Los resultados de evaluación no seleccionan hiperparámetros ni abren el test.

## Ejecución local

La preparación analítica genera doce escenarios y conserva su procedencia. La referencia fija usa el evento ficticio conocido, no un predictor entrenado.

```bash
uv run --extra reinforcement python scripts/prepare_financial_scenarios.py --output data/interim/financial-scenarios-v1
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --extra reinforcement --extra cuda python scripts/run_financial_comparators.py \
  --train-tape data/interim/financial-scenarios-v1/known_signal-train-42 \
  --validation-tape data/interim/financial-scenarios-v1/known_signal-validation-1042 \
  --output data/interim/financial-comparison-v1
```

Las cargas científicas requieren `cuda:0` y una admisión exclusiva con margen de memoria. Si existe otra carga de cómputo, el ejecutor rechaza el inicio. `--diagnostic` habilita únicamente pruebas técnicas explícitas de CPU con un máximo de 32 decisiones por ajuste y una configuración separada. No representa una comparación científica de los métodos.

`scripts/prepare_financial_scenarios.py` también admite un predictor entrenado. Los argumentos `--parent`, `--ordered`, `--supervision` y `--encoded` deben aparecer juntos. `--parent` recibe el `run.json` completo del padre y `--ordered` el manifiesto del corpus causal ordenado. `--supervision` debe ser la supervisión exacta admitida por ese padre, incluida la vista del brazo cuando corresponda. `--encoded` recibe el manifiesto de codificación al que apunta `configuration.source_manifest_sha256` de esa supervisión.

El preparador comprueba las huellas entre los manifiestos, los recuentos, el contexto y las dimensiones antes de generar escenarios. El contexto de `--config` debe coincidir con el padre. Los codificadores deben conservar exactamente `FrozenEncoders.spec`. Las noticias, gráficos, fundamentales y macro utilizan dimensiones 384, 512, 45 y 420. La admisión mediante `GpuLease` precede a la carga del padre y los codificadores. Comprueba los recursos antes de cada cohorte y al terminar la inferencia. Este modo no ofrece una alternativa CPU en la CLI.

Cada mundo pasa por `EncodedWorld` y después por `MarketTape.from_world`, con `parent.predict` y la huella de su checkpoint. También se calcula la última observación sin solicitar etiquetas inexistentes. El padre permanece congelado y sus archivos originales no se modifican. La salida sigue conteniendo precios y predicciones en el mismo formato Parquet. Su dominio continúa siendo sintético aunque las predicciones procedan de un padre entrenado con datos reales. Los escenarios de validación deben tener semillas distintas de los de entrenamiento.

`--resume` exige un índice de preparación con la misma configuración, padre, datos, codificadores y código. Los mundos completos se recalculan y se comparan por su huella antes de reutilizarlos, sin reescribir sus archivos. Un directorio parcial sin manifiesto confirmado se rechaza. Las salidas anteriores sin la identidad de preparación requieren una carpeta nueva. La recuperación se hace por mundos completos, no en mitad de una cohorte. Sin los cuatro argumentos del padre se conserva la referencia analítica y no se importa PyTorch.

La integración se comprueba con una referencia DLinear de prueba, un checkpoint serializado y codificadores pequeños explícitos en CPU. Estas pruebas verifican contratos y recuperación. La preparación con pesos y codificadores reales en CUDA sigue pendiente. No acredita el histórico OHLC ni añade resultados financieros sobre datos reales.

Los informes incluyen retorno neto, caída máxima, costes, rotación respecto al capital inicial, pasos y motivo de resultado incompleto. No convierten retornos residuales en beneficios. El estado final conserva las posiciones y órdenes pendientes para inspección y recuperación.
