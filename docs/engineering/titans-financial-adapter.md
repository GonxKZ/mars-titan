# Adaptador financiero con estado por flujo

`models.titans.financial` compone el Transformer de precios, las proyecciones multimodales, la fusión, la cabeza escalar y el núcleo MAC existentes. Prepara una decisión por activo y devuelve una propuesta de estado. No confirma una sesión ni recorre el corpus. Es la primera pieza de integración financiera y no representa todavía MARS-TITAN con sus ampliaciones completas.

## Validación por componentes

`FinancialConfig.__post_init__` conserva el orden de las comprobaciones y delega sus bloques en métodos privados de la misma clase. La entrada, la cabeza y la disposición se comprueban antes de las puertas. La estabilidad se normaliza antes de calcular los logits iniciales de `gate_bias`. La captura depende de la cascada ya declarada. Después se comprueban contexto, grupos, modulación y operadores regionales, seguidos por lectores, objetivos, mezclas de sesión, cabezas auxiliares, fiabilidad y límites. Las exclusiones permanecen en la posición que determina qué error aparece primero.

El [recibo del 31 de octubre](../../reports/engineering/config-validation-20261031/verification.json) compara la misma selección de 1.284 casos CPU antes y después, sin omisiones ni pasos de optimizador. La complejidad de `__post_init__` pasa de 132 a 1 y los 18 métodos nuevos tienen un máximo de 15. El CRAP máximo de los métodos de la clase pasa de 198,7389 a 33,0317. El máximo posterior pertenece a `identity`, que mantiene su cuerpo y su complejidad 33. Las condiciones de validación no se han eliminado y estas cifras no miden rendimiento ni calidad predictiva. El recibo registra las convenciones de coverage.py y radon.

La expansión de las llamadas recupera el AST de las sentencias anteriores y comprueba que los locales permanecen dentro de sus bloques. Una matriz de 712 casos conserva 162 configuraciones aceptadas y 550 rechazos con sus tipos y mensajes. Es una comparación acotada, no una prueba para toda entrada posible. Los fixtures de los cuatro controles con dos políticas de entrada conservan pesos, identidades, salidas y estados bit a bit en CPU y FP64. Cada configuración recorre tres grupos de dos flujos y recupera su estado antes del siguiente grupo. No se ha utilizado el corpus real ni se ha ajustado el predictor.

Los 63 casos de contratos y componentes pasan y las 33 mutaciones dirigidas se detectan con G1 verde, M1 rojo y G2 verde, sin optimizador. Las selecciones se solapan y no se suman como casos distintos. Estas mutaciones comprueban conexión, precedencia, normalización, límites y exclusiones. No son una puntuación de mutación global. Los campos, sus valores por defecto y las identidades comprobadas se conservan. El código mantiene su procedencia nueva y los manifiestos no deben reutilizar huellas de otra ejecución.

La regresión de modelos y consumidores terminó el 31 de octubre. La unión de 349 módulos importadores y 120 módulos de modelos contiene 393 archivos, sin duplicarlos. En 50 tandas pasan 8527 casos y se omiten 111, sin fallos ni errores. Son 72 omisiones por el bloqueo de aprendizaje, 30 por el ejecutable PPO ausente y nueve por la biblioteca de simulación ausente. Las fuentes y la protección conservan sus huellas al terminar. La nueva organización solo cambia validación Python y no acredita una comprobación nueva de los operadores CUDA. La verificación general de CUDA continúa separada en #816. El aprendizaje y el test final siguen bloqueados.

```bash
uv --no-cache run --no-sync pytest -q -p no:cacheprovider \
  tests/models/titans/test_financial_config_validation.py \
  tests/models/titans/test_financial_config_components.py
```

## Token y controles

La ventana cercana de precios pertenece al Transformer común y conserva el contexto declarado por el corpus. En la edición histórica son 64 sesiones. La fusión produce un token por nueva decisión y MAC recibe `C=1`. Recodificar ventanas solapadas no vuelve a escribir sus 64 observaciones en la memoria neuronal.

| `variant` | Cálculo después de la fusión | Escrituras asociativas |
| --- | --- | --- |
| `transformer_direct` | Cabeza escalar existente | Ninguna |
| `mac_disabled` | Atención MAC sin prefijos ni gate, seguida de la misma cabeza | Ninguna |
| `mac_frozen` | MAC con P, lectura y gate, seguido de la misma cabeza | Ninguna |
| `mac_online` | MAC completo seguido de la misma cabeza | Una por decisión y flujo |

`FinancialConfig(gate_bias=None)` construye la memoria con la identidad v1 de las puertas, sin bias, y conserva las huellas anteriores. Con `gate_bias` los tres controles MAC usan la [inicialización con bias declarado](titans-gate-initialization.md). Una receta JSON puede declararla con sus tres valores. La identidad registra `memory_gate_bias` en las cuatro variantes, también en `transformer_direct`, para que el emparejamiento desde `mac_online` compare la misma configuración.

`FinancialConfig(memory_projections="linear_v1")` conserva las proyecciones lineales anteriores. Con `titans_mac_paper_projections_v2` la memoria usa SiLU, la convolución causal de núcleo 4 y la norma L2 de la [sección 4.4 del artículo](titans-memory-core.md#proyecciones-de-la-sección-44). El estado por flujo añade entonces las ventanas de la convolución y la identidad registra `memory_projections` en las cuatro variantes, con el mismo criterio que `memory_gate_bias`.

La atención del control desactivado recibe un único token. Ese control mide el bloque añadido, sin constituir otro codificador temporal de 64 posiciones. La construcción utiliza `MultimodalReference` para obtener los componentes comunes. Sus rutas originales no se modifican. Todos los controles nuevos usan dropout cero.

`copy_paired_parameters(source, target)` copia explícitamente los parámetros compatibles y devuelve un recibo con sus nombres, formas y huellas de origen y destino. La fuente `mac_online` permite emparejar los cuatro controles. El recibo distingue parámetros copiados de los que solo se inicializan. No transfiere cursores, momentum ni pesos rápidos de un flujo. Las cargas ordinarias siguen rechazando contratos diferentes. Esta copia se realiza antes del recorrido y no reanuda un grafo diferenciable existente.

## Frontera de entrada

`FinancialInputSpec` conserva huellas de fuente y vista, representación, catálogos ordenados, dimensiones y política de entradas. Se construye a partir de la identidad que ya valida el lector. No acredita por sí sola la admisión de una edición. La política predeterminada sigue siendo `strict_inputs_v1` y la edición histórica exige `historical_masked_2000_v1` explícita.

`DecisionBatch.from_corpus(batch, specification, device=..., dtype=...)` recibe un lote NumPy del corpus. Valida los tipos originales float32, formas, finitud, presencias y rellenos antes de convertir a tensores. El lector y esta frontera CPU utilizan `validate_historical_vectors`, en `data.input_policy`. La validación Arrow de presencia y el contraste con `news_count` permanecen en el lector.

La función `validated_cpu_batch` expone la misma copia verificada como `CPUDecisionBatch`, con matrices NumPy respaldadas por bytes inmutables y una huella de contenido. Cada entrada se copia a su objeto `bytes` antes de validarla, así que lo validado es lo que se conserva. `setflags(write=True)` no puede reactivar la escritura sobre esos buffers. La huella SHA-256 se calcula una sola vez, al crear la vista. `verify()` exige después los mismos arreglos de solo lectura, con sus formas, tipos y strides, y la huella guardada, sin volver a recorrer los bytes. Como el contenido de un objeto `bytes` no puede cambiar, sustituir un arreglo, aunque tenga los mismos valores, cambiar su forma o volver a escribir en él se rechaza. Un codec separado debe comprobarla antes de consumirla. `DecisionBatch.from_validated` realiza esa comprobación antes de trasladar tensores al dispositivo. `from_corpus` compone ambas operaciones. No se duplican las reglas ni se añaden campos de supervisión a esa vista.

Las modalidades ausentes se anulan después de sus proyecciones con bias. Las cinco presencias se añaden a la fusión en orden precios, noticias, gráficos, fundamentales y macro. El primer bloque de fusión histórico tiene `5H+5` entradas. Los triples contables y macro conservan valores, máscaras por concepto y edades. Un cero observado conserva su máscara y no equivale a una ausencia. La ruta estricta conserva la fusión de `5H` entradas.

El lote de decisión contiene únicamente entradas, presencias, IDs de muestra y flujo, corte, disponibilidad e identidad. No copia ni consulta `target` o `target_available_at`. Requiere un único corte por lote, flujos únicos y disponibilidad no posterior a ese corte. La reserva desde 2024 se rechaza. Los tensores del lote se consideran de solo lectura. El control de versiones detecta modificaciones ordinarias posteriores a la adaptación, pero no es un certificado criptográfico frente a escrituras externas mediante `.data` o almacenamiento compartido manipulado.

`CorpusDataset.batches` puede mezclar activos, grupos y filas. No se utiliza directamente como un flujo cronológico. El [consumidor financiero](financial-session-v2.md) y el [entrenador cronológico](titans-chronological-trainer.md) separan observaciones y etiquetas y agrupan las decisiones por corte, reutilizando las validaciones comunes. El adaptador no añade otro materializador de objetivos.

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

Las comprobaciones de finitud de `prepare`, de la lectura episódica y de los entrenadores pasan por `state.check_finite`. En CUDA, dentro de `deferred_checks`, cada una deja en el dispositivo la norma infinito del tensor, una sola reducción que es finita si y solo si lo son todos sus valores, y el tramo lee todas las comprobaciones con una copia al anfitrión al terminar. El error es el mismo `ValueError` de la primera comprobación fallida en orden de programa. El entrenador cronológico abre un tramo por evento y otro por cada grupo de flujos de la repetición. Los lectores de MARS-TITAN abren uno por evento y otro por la repetición de cada tramo. Todos se resuelven antes de cualquier paso, admisión o checkpoint. En CPU la comprobación sigue siendo inmediata. La igualdad de bits con la versión anterior, las mutaciones y el trabajo de anfitrión retirado están en el [informe del recorrido por evento](../../reports/engineering/titans-event-path-20261013/README.md). Con eventos reales, el caudal de `mac_online` pasó de 13.583 y 14.379 a 19.534 y 19.319 filas/s. El mismo informe recoge los perfiles que descartan capturar CUDA Graphs por forma exacta y fusionar la actualización de memoria en un núcleo propio.

## Recuperación y presupuestos

La recuperación exige guardar tanto `predictor.state_dict()` como `predictor.export_state(state)`. Primero se restauran los parámetros y después el estado rápido. El formato conserva configuración, contratos internos, precisión y modo train/eval. Guardar solo pesos o solo memoria no reconstruye la siguiente decisión. La recuperación no reconstruye un grafo de autograd pendiente.

Los contadores de observaciones son int64 en CPU. Los pesos rápidos, momentum y contadores internos de MAC usan el dispositivo del módulo. Al trasladar un archivo entre dispositivos, debe conservarse esa separación. La API recibe diccionarios ya cargados y no incorpora un cargador de archivos arbitrarios ni publicación atómica propia.

`max_batch` usa 256 flujos por llamada por defecto y admite un máximo de 4.096. `max_state_bytes`, predeterminado a 64 MiB y limitado a 1 GiB, acota el almacenamiento retenido por los tensores y los metadatos canónicos del estado. `state_usage` informa del coste lógico de tensores por flujo, del almacenamiento real retenido, los metadatos y el total. Con anchura 32 y float64, un control MAC necesita 32.784 bytes de tensores por flujo. La cuenta incluye dos matrices rápidas, dos de momentum y ambos contadores. Las vistas que retienen un almacenamiento mayor se contabilizan por ese almacenamiento.

Estas cantidades no son el RSS ni el tamaño de un archivo de PyTorch. El presupuesto agregado de miles de activos y la retención de archivos pertenecen al coordinador. Superar un límite causa error, sin truncar la población ni reiniciar memorias silenciosamente.

El [recuento de metadatos](../../reports/engineering/state-metadata-sizing-20261018/README.md) reutiliza únicamente el tamaño de la configuración invariable por modo train/eval. Serializa los campos variables en cada llamada y conserva el número exacto de bytes y el límite. La caché contiene como máximo dos enteros y se invalida al volver a sellar parámetros. Las comprobaciones incluyen escapes JSON, cambio de modo, carga, límites exactos y recuperación CPU/CUDA.

El [puente de estados por bloques](financial-state-blocks.md) añade exportación CPU, recuperación con dispositivo explícito y reunión de hasta 256 flujos desde payloads confirmados. Un índice de referencias permite conservar un censo mayor sin formar un único estado tensorial. La comprobación del contenido de los artefactos y la retención de referencias anidadas siguen correspondiendo al coordinador.

El adaptador base no consulta el banco y conserva K=1. El [consumidor financiero](financial-session-v2.md) añade el codec fijo, una instantánea por sesión y la publicación conjunta de predicciones, estados, pendientes y banco. El lector externo admite K=1, 2 y 4 sin multiplicar las escrituras de MAC. M0/M1 están integrados. [M2](mature-error-write-policy.md) compone tres índices 50/25/25 con error de la emisión original, pruebas CPU y una comprobación CUDA focal FP32/FP64, K=1, B_mem=4 y C apagado. [M3](m3-write-policy.md) reutiliza esos índices con una puntuación compuesta, con pruebas CPU y comprobaciones CUDA superadas el 9 de octubre. La consolidación M sobre M2 sigue pendiente.

La opción [`local_control`](../experiments/mars_titan_cm_v1/mac_local_control.md) añade C con identidad separada. `None` conserva la ruta anterior. Una configuración explícita, incluido el modo `disabled`, usa SDPA Math durante la preparación, que abarca el codificador de precios y MAC. Por sí sola no desactiva la ruta fusionada del Transformer. El consumidor congelado declara además `fastpath=False` para todos sus controles. La selección de C se fija sobre el grupo lógico y se pasa a los bloques físicos mediante `control_selection` y `control_context_id`. El resultado opcional contiene el diagnóstico o la contribución de penalización, sin publicar estado adicional. La retención M se configura en el banco externo. La comparación factorial necesita registrar todos estos ajustes en B.

La comprobación del adaptador anterior a la opción C pasó 330 pruebas CPU, incluidos los lectores históricos. Quince mutaciones dirigidas fueron detectadas. La revisión corrigió dos defectos: la reactivación de escritura en la copia CPU sin actualizar su huella y la interpretación multidimensional de una selección de filas recibida como tupla.

Los cuatro controles se han contrastado en CUDA con FP32 y FP64. Los fixtures tienen dos flujos, ventanas de 64 sesiones, anchura 32 y modalidades reducidas. La mayor diferencia de salida CPU/CUDA es 3,35 × 10⁻⁸ en FP32 y 2,08 × 10⁻¹⁷ en FP64. Los gradientes respetan las tolerancias declaradas y la recuperación de la siguiente predicción y actualización es exacta. Los RNG CPU/CUDA permanecen iguales. El [recibo original](../../reports/research/titans-financial-adapter-verification-20261008.json) conserva ese alcance. La [verificación posterior del consumidor](../../reports/engineering/financial-session-completion-20261008.json) comprueba la composición congelada y su contrato numérico, con fallos y límites registrados. La ejecución científica sobre el corpus completo y las restantes ampliaciones siguen pendientes.
