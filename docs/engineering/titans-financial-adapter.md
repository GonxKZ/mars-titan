# Adaptador financiero con estado por flujo

`models.titans.financial` compone el Transformer de precios, las proyecciones multimodales, la fusión, la cabeza escalar y el núcleo MAC existentes. Prepara una decisión por activo y devuelve una propuesta de estado. No confirma una sesión ni recorre el corpus. Es la primera pieza de integración financiera y no representa todavía MARS-TITAN con sus ampliaciones completas.

## Token y controles

La ventana cercana de precios pertenece al Transformer común y conserva el contexto declarado por el corpus. En la edición histórica son 64 sesiones. La fusión produce un token por nueva decisión y MAC recibe `C=1`. Recodificar ventanas solapadas no vuelve a escribir sus 64 observaciones en la memoria neuronal.

| `variant` | Cálculo después de la fusión | Escrituras asociativas |
| --- | --- | --- |
| `transformer_direct` | Cabeza escalar existente | Ninguna |
| `mac_disabled` | Atención MAC sin prefijos ni gate, seguida de la misma cabeza | Ninguna |
| `mac_frozen` | MAC con P, lectura y gate, seguido de la misma cabeza | Ninguna |
| `mac_online` | MAC completo seguido de la misma cabeza | Una por decisión y flujo |

`FinancialConfig(gate_bias=None)` construye la memoria con la identidad v1 de las puertas, sin bias, y conserva las huellas anteriores. Con `gate_bias` los tres controles MAC usan la [inicialización con bias declarado](titans-gate-initialization.md). Una receta JSON puede declararla con sus tres valores. La identidad registra `memory_gate_bias` en las cuatro variantes, también en `transformer_direct`, para que el emparejamiento desde `mac_online` compare la misma configuración.

La atención del control desactivado recibe un único token. Ese control mide el bloque añadido, sin constituir otro codificador temporal de 64 posiciones. La construcción utiliza `MultimodalReference` para obtener los componentes comunes. Sus rutas originales no se modifican. Todos los controles nuevos usan dropout cero.

`copy_paired_parameters(source, target)` copia explícitamente los parámetros compatibles y devuelve un recibo con sus nombres, formas y huellas de origen y destino. La fuente `mac_online` permite emparejar los cuatro controles. El recibo distingue parámetros copiados de los que solo se inicializan. No transfiere cursores, momentum ni pesos rápidos de un flujo. Las cargas ordinarias siguen rechazando contratos diferentes. Esta copia se realiza antes del recorrido y no reanuda un grafo diferenciable existente.

## Frontera de entrada

`FinancialInputSpec` conserva huellas de fuente y vista, representación, catálogos ordenados, dimensiones y política de entradas. Se construye a partir de la identidad que ya valida el lector. No acredita por sí sola la admisión de una edición. La política predeterminada sigue siendo `strict_inputs_v1` y la edición histórica exige `historical_masked_2000_v1` explícita.

`DecisionBatch.from_corpus(batch, specification, device=..., dtype=...)` recibe un lote NumPy del corpus. Valida los tipos originales float32, formas, finitud, presencias y rellenos antes de convertir a tensores. El lector y esta frontera CPU utilizan `validate_historical_vectors`, en `data.input_policy`. La validación Arrow de presencia y el contraste con `news_count` permanecen en el lector.

La función `validated_cpu_batch` expone la misma copia verificada como `CPUDecisionBatch`, con matrices NumPy respaldadas por bytes inmutables y una huella de contenido. `setflags(write=True)` no puede reactivar la escritura sobre esos buffers. `verify()` contrasta además formas, tipos, strides y digest antes de compartir la vista. Un codec separado debe comprobarla antes de consumirla. `DecisionBatch.from_validated` realiza esa comprobación antes de trasladar tensores al dispositivo. `from_corpus` compone ambas operaciones. No se duplican las reglas ni se añaden campos de supervisión a esa vista.

Las modalidades ausentes se anulan después de sus proyecciones con bias. Las cinco presencias se añaden a la fusión en orden precios, noticias, gráficos, fundamentales y macro. El primer bloque de fusión histórico tiene `5H+5` entradas. Los triples contables y macro conservan valores, máscaras por concepto y edades. Un cero observado conserva su máscara y no equivale a una ausencia. La ruta estricta conserva la fusión de `5H` entradas.

El lote de decisión contiene únicamente entradas, presencias, IDs de muestra y flujo, corte, disponibilidad e identidad. No copia ni consulta `target` o `target_available_at`. Requiere un único corte por lote, flujos únicos y disponibilidad no posterior a ese corte. La reserva desde 2024 se rechaza. Los tensores del lote se consideran de solo lectura. El control de versiones detecta modificaciones ordinarias posteriores a la adaptación, pero no es un certificado criptográfico frente a escrituras externas mediante `.data` o almacenamiento compartido manipulado.

`CorpusDataset.batches` puede mezclar activos, grupos y filas. No debe usarse como si ya fuese un flujo cronológico. El futuro coordinador tendrá que reutilizar sus validaciones, separar observaciones sin objetivo y agrupar las decisiones por corte. No se ha añadido otro lector ni un materializador de objetivos.

## Preparación y estado

```python
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor

config = FinancialConfig(input_spec, variant="mac_online")
predictor = FinancialPredictor(config, device="cpu")
state = predictor.initial_state(decisions.flow_ids)
prepared = predictor.prepare(decisions, state)
```

`input_spec` y `decisions` corresponden a la especificación y al lote verificados en la frontera anterior. `PreparedDecisions` contiene `point_predictions`, `detached_tokens`, `next_state`, `representation_id` e `input_digest`. El token está desacoplado del grafo y su identidad incluye configuración y parámetros. No es una clave episódica estable frente a cambios de los pesos de fusión. Una selección de filas deriva su digest del lote verificado y de los índices elegidos, sin leer de vuelta los tensores desde GPU.

`working_state` expone la representación posterior a MAC y anterior a la cabeza, sin volver a ejecutar la transición. Conserva el grafo solo con `differentiable=True`. Un consumidor episódico puede refinar esa representación después de MAC, sin utilizarla como clave estable del codec ni repetir escrituras asociativas. `point_predictions` sigue siendo la predicción base.

`FinancialState` enlaza configuración, parámetros, flujos, último ID y corte, contador de observaciones y `MACState` cuando corresponde. Entradas y estado deben tener los mismos flujos en el mismo orden. `DecisionBatch.select` y `select_state` permiten aplicar una selección explícita a ambos. El predictor rechaza repeticiones, orden inverso y cambios de identidad. Los controles sin escritura también avanzan el cursor de observaciones. El estado recibido permanece intacto.

`prepare(..., differentiable=True)` conserva el camino del objetivo externo por la actualización asociativa. El modo predeterminado devuelve predicción y estado sin grafo persistente. `torch.no_grad()` es compatible con la actualización interna. `torch.inference_mode()` se rechaza en esta interfaz. No se ejecuta un optimizador ni se utiliza un error financiero para calcular la sorpresa asociativa.

Los parámetros deben permanecer estables durante el flujo, salvo mediante la carga, la copia identificada o el paso del [entrenador cronológico](titans-chronological-trainer.md), que vuelve a sellar su huella después de cada actualización. Dentro de `prepare` se comprueban versiones de tensores, sin convertir parámetros o entradas CUDA a NumPy/CPU. Ese control barato no detecta cualquier modificación posible. `verify_parameter_identity`, la creación de estado y las fronteras de exportación y recuperación contrastan además las huellas de bytes. Debe confirmarse esa comprobación antes de publicar una propuesta de sesión. Una modificación no autorizada mediante `.data` puede eludir el contador y se rechaza en esa frontera fuerte.

## Recuperación y presupuestos

La recuperación exige guardar tanto `predictor.state_dict()` como `predictor.export_state(state)`. Primero se restauran los parámetros y después el estado rápido. El formato conserva configuración, contratos internos, precisión y modo train/eval. Guardar solo pesos o solo memoria no reconstruye la siguiente decisión. La recuperación no reconstruye un grafo de autograd pendiente.

Los contadores de observaciones son int64 en CPU. Los pesos rápidos, momentum y contadores internos de MAC usan el dispositivo del módulo. Al trasladar un archivo entre dispositivos, debe conservarse esa separación. La API recibe diccionarios ya cargados y no incorpora un cargador de archivos arbitrarios ni publicación atómica propia.

`max_batch` admite hasta 256 flujos por llamada. `max_state_bytes`, predeterminado a 64 MiB y limitado a 256 MiB, acota el almacenamiento retenido por los tensores y los metadatos canónicos del estado. `state_usage` informa del coste lógico de tensores por flujo, del almacenamiento real retenido, los metadatos y el total. Con anchura 32 y float64, un control MAC necesita 32.784 bytes de tensores por flujo. La cuenta incluye dos matrices rápidas, dos de momentum y ambos contadores. Las vistas que retienen un almacenamiento mayor se contabilizan por ese almacenamiento.

Estas cantidades no son el RSS ni el tamaño de un archivo de PyTorch. El presupuesto agregado de miles de activos y la retención de archivos pertenecen al coordinador. Superar un límite causa error, sin truncar la población ni reiniciar memorias silenciosamente.

El [puente de estados por bloques](financial-state-blocks.md) añade exportación CPU, recuperación con dispositivo explícito y reunión de hasta 256 flujos desde payloads confirmados. Un índice de referencias permite conservar un censo mayor sin formar un único estado tensorial. La comprobación del contenido de los artefactos y la retención de referencias anidadas siguen correspondiendo al coordinador.

El adaptador base no consulta el banco y conserva K=1. El [consumidor financiero](financial-session-v2.md) añade el codec fijo, una instantánea por sesión y la publicación conjunta de predicciones, estados, pendientes y banco. El lector externo admite K=1, 2 y 4 sin multiplicar las escrituras de MAC. M0/M1 están integrados. [M2](mature-error-write-policy.md) compone tres índices 50/25/25 con error de la emisión original, pruebas CPU y una comprobación CUDA focal FP32/FP64, K=1, B_mem=4 y C apagado. [M3](m3-write-policy.md) reutiliza esos índices con una puntuación compuesta y solo tiene pruebas CPU. La consolidación M sobre M2 sigue pendiente.

La opción [`local_control`](../experiments/mars_titan_cm_v1/mac_local_control.md) añade C con identidad separada. `None` conserva la ruta anterior. Una configuración explícita, incluido el modo `disabled`, usa SDPA Math durante la preparación, que abarca el codificador de precios y MAC. Por sí sola no desactiva la ruta fusionada del Transformer. El consumidor congelado declara además `fastpath=False` para todos sus controles. La selección de C se fija sobre el grupo lógico y se pasa a los bloques físicos mediante `control_selection` y `control_context_id`. El resultado opcional contiene el diagnóstico o la contribución de penalización, sin publicar estado adicional. La retención M se configura en el banco externo. La comparación factorial necesita registrar todos estos ajustes en B.

La comprobación del adaptador anterior a la opción C pasó 330 pruebas CPU, incluidos los lectores históricos. Quince mutaciones dirigidas fueron detectadas. La revisión corrigió dos defectos: la reactivación de escritura en la copia CPU sin actualizar su huella y la interpretación multidimensional de una selección de filas recibida como tupla.

Los cuatro controles se han contrastado en CUDA con FP32 y FP64. Los fixtures tienen dos flujos, ventanas de 64 sesiones, anchura 32 y modalidades reducidas. La mayor diferencia de salida CPU/CUDA es 3,35 × 10⁻⁸ en FP32 y 2,08 × 10⁻¹⁷ en FP64. Los gradientes respetan las tolerancias declaradas y la recuperación de la siguiente predicción y actualización es exacta. Los RNG CPU/CUDA permanecen iguales. El [recibo original](../../reports/research/titans-financial-adapter-verification-20261008.json) conserva ese alcance. La [verificación posterior del consumidor](../../reports/engineering/financial-session-completion-20261008.json) comprueba la composición congelada y su contrato numérico, con fallos y límites registrados. La ejecución científica sobre el corpus completo y las restantes ampliaciones siguen pendientes.
