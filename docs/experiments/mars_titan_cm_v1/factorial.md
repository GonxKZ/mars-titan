# Factorial CM-v1 sobre Titans-MAC

Este documento describe la implementación del factorial B, B+C, B+M y B+C+M sobre el núcleo Titans-MAC integrado. Nada se ha ejecutado con datos. El bloqueo de la edición histórica desde 2000 sigue vigente y las pruebas recorren los bucles hasta el paso del optimizador con un registrador que no modifica pesos. Las comprobaciones son técnicas y no aportan ningún resultado predictivo.

La declaración está en [`configs/titans/cm-v1-factorial.json`](../../../configs/titans/cm-v1-factorial.json) y el código en [`training/cm_v1_factorial.py`](../../../src/mars_titan/training/cm_v1_factorial.py).

## B y los cuatro brazos

B es la referencia del factorial y tiene identidad propia. Se compone de dos partes:

1. Un núcleo Titans-MAC `mac_online` ajustado con la receta del núcleo de la campaña (`chronological-training-historical-masked.json`, con memoria residual normalizada y `gate_bias`). Lleva el control local C en modo `disabled`, así que usa SDPA Math y la misma base de proyección que B+C, pero no mide nada.
2. Un lector episódico M1 con K = 1 y selección por paso, ajustado sobre ese núcleo congelado con la receta del lector (`episodic-readout-historical-masked.json`) y retención `reservoir`.

La receta del núcleo entra como parámetro de la declaración y su huella forma parte de la identidad de cada brazo. C solo cambia el ajuste del núcleo y M solo la retención del banco del lector. Por eso hay dos núcleos auxiliares y cuatro brazos:

| Brazo | Núcleo | Control C del núcleo | Retención del lector |
| --- | --- | --- | --- |
| `cm_v1_b` | `cm_v1_core_b` | `disabled` | `reservoir` |
| `cm_v1_bc` | `cm_v1_core_c` | `penalty` | `reservoir` |
| `cm_v1_bm` | `cm_v1_core_b` | `disabled` | `anchored` |
| `cm_v1_bcm` | `cm_v1_core_c` | `penalty` | `anchored` |

B y B+M comparten el mismo núcleo elegido, así que su diferencia se reduce al factor M. B+C y B+C+M comparten el suyo. Los dos núcleos parten de los mismos parámetros iniciales con la misma semilla y tienen la misma base de C, que se genera con su propio generador. Los cuatro lectores parten también de los mismos parámetros. Todos los brazos usan los mismos casos de búsqueda (tasas 1e-4 y 1e-3), las mismas 30 épocas con presupuesto fijo y la misma selección por `session_mae` de validación.

Valores declarados antes de ejecutar, sin ajuste con datos:

- C: rango 4, frecuencia 16, semilla 91, 64 ángulos, umbral 1, peso 0,1, dos flujos medidos por evento y 128 MiB de presupuesto. Con dimensión 64 en FP32 la estimación del control es 72 MiB para dos flujos.
- M: retención con centros fijos con frontera 8, ocho candidatos nuevos y ocho intercambios, con la capacidad (1.024) y la semilla (73) del banco de B.

`load_declaration` rechaza cualquier cambio de B, de los brazos, de los núcleos o de los campos de los factores. Un cambio de valores exige otra identidad de los cuatro brazos.

## C durante el ajuste del núcleo

`ChronologicalTrainer` admite ahora el control local en dos modos. `disabled` identifica B y conserva el objetivo de la tarea. `penalty` suma al objetivo del tramo la media de los términos de sus grupos lógicos medidos:

```text
objetivo = pérdida media de las etiquetas maduras del tramo + media_g P_g
P_g = peso · Σ_{flujos medidos de g} max(ŵ(RᵀJR) − umbral, 0)² / n_seleccionados(g)
```

El grupo lógico es el evento completo, con todos sus bloques físicos. La selección se fija antes de dividirlo y su contexto identifica parámetros, índice, tramo e instante, así que dividir en bloques o reanudar no cambia los flujos medidos. El término conserva el gradiente del token actual y de los parámetros de MAC, y el estado rápido medido se desacopla de la historia. Si un tramo termina sin etiquetas maduras no hay paso y su término se descarta (`control_groups_discarded`). C no añade actualizaciones, de modo que B y B+C tienen el mismo número de pasos. El modo `diagnostic` y la combinación de la penalización con `accumulation_rows` se rechazan. Sin control local, la identidad del entrenador conserva literalmente su forma anterior.

Al predecir con el núcleo congelado, C no interviene. Un núcleo ajustado con la penalización se copia a un gemelo `disabled` con la misma base mediante `copy_paired_parameters`, y su emisión es la misma.

## M en el banco del lector

M usa la [retención con centros fijos](anchored_retention.md) del banco M1. Cada selección parte del banco confirmado y de las etiquetas maduras del evento, conserva episodios reales con su clave, valor, etiqueta y procedencia y no supera la capacidad. Las etiquetas pendientes nunca entran en el conjunto de clientes. Cuando el banco no se desborda, la retención con centros fijos conserva todos los episodios y el lector emite lo mismo que con `reservoir`.

## Jacobianos completos y tres lecturas separadas

[`transition_jacobian.py`](../../../src/mars_titan/models/titans/transition_jacobian.py) calcula J denso en dimensiones pequeñas (orden hasta 256):

- `fast_state_jacobian` deriva la transición rápida completa de MAC, `z' = F(z, x)` con pesos rápidos y momentum, con SDPA Math. `fast_state_trajectory` da los J_t de una secuencia de tokens de un flujo. El control C comparte con este módulo la definición de F.
- `refinement_jacobians` da `J_k = ∂z_{k+1}/∂z_k` de cada refinamiento del lector, con la base y los episodios elegidos fijos. Es `I + η D_z f_k`, con η = σ(s) y `f_k = tanh(W[z_k, base, read(z_k), presencia] + b)`. Incluye la derivada de la lectura respecto a la consulta y de la tangente hiperbólica. Los episodios elegidos son constantes a trozos y la derivada vale lejos de empates del orden de vecinos.

[`operator_dynamics.py`](../../../src/mars_titan/cm/operator_dynamics.py) separa tres lecturas que antes podían confundirse:

| Lectura | Función | Qué afirma |
| --- | --- | --- |
| Diagnóstico heurístico | `numerical_radius_estimates`, `radius_penalty` | Estimación puntual de cada operador. Es la que penaliza C |
| Operador fijo | `fixed_operator_powers` | Normas de `A^j` frente a `2 ŵ^j`, la expresión de `w(A^j) ≤ w(A)^j` y `‖T‖₂ ≤ 2 w(T)` en aritmética exacta |
| Dinámica variable | `variable_products` | Normas y radios espectrales medidos de `A_t ⋯ A_1`, sin cota |

El contraejemplo de la issue pasa por las tres. A1 y A2 tienen radio numérico 0,75 y cumplen por separado la expresión del operador fijo, pero sus productos alternados crecen como 2,25ᵏ. `variable_products` lo marca con `pointwise_below_but_product_above`. Ninguna de estas lecturas certifica la estabilidad de la red completa.

## Ventanas, traslado y campaña

`run_cm_v1_core_window` ajusta un núcleo con `run_titans_window`, que acepta ahora `local_control` solo en `mac_online` y lo registra en la petición. `run_cm_v1_window` ajusta el lector del brazo sobre el núcleo elegido y comprueba que la petición del padre declara exactamente el control C del brazo. `carry_cm_v1` traslada núcleo y lector elegidos a una ventana posterior. Las tres comprueban la protección del aprendizaje antes de leer fuentes.

La ventana del lector es común con MARS-TITAN. `ReadoutFamily` reúne lo que distingue a cada familia: control esperado en el padre, variante, retención e identidad. MARS-TITAN rechaza ahora cualquier padre con control C.

La sección opcional `cm_v1` de la campaña solo declara la ruta de la declaración y la semilla de búsqueda. Los dos núcleos son trabajos auxiliares: se ajustan en las ventanas reentrenadas con sus casos y finalistas, no tienen traslado y no publican recibo de ventana. Cada búsqueda de un brazo depende de las búsquedas de su núcleo en la ventana y cada finalista del finalista del núcleo con su semilla. Los ejecutores declaran `fastpath=False` mientras dura cada trabajo. Las campañas A y B no declaran todavía la sección. Declararla añadiría 1.080 ajustes en A (360 de núcleos) y 408 ajustes con 336 traslados en B (136 de núcleos).

## Comprobaciones

Todas en CPU, con `CUDA_VISIBLE_DEVICES=-1`, dos hilos y pruebas por archivo.

| Archivo | Qué comprueba |
| --- | --- |
| `tests/training/test_financial_run_control.py` | Parámetros y base emparejados, identidad, rechazos, paridad exacta con B sin flujos medidos, linealidad del gradiente en el peso sin recorte, la cabeza fuera del camino de C, grupo lógico frente a bloque físico, validación igual a B y reanudación exacta |
| `tests/cm/test_operator_dynamics.py` | A1/A2, expresión del operador fijo, productos frente a su cálculo explícito y rechazos |
| `tests/models/titans/test_transition_jacobian.py` | `RᵀJR` igual a la compresión de C, J frente a diferencias centrales en FP64, trayectoria, `I + η D_z f` en los dos modos de selección y sin banco |
| `tests/training/test_cm_v1_factorial.py` | Declaración, presupuesto de C, núcleos emparejados con el mismo número de pasos, cada factor solo donde se declara, control del padre, recuperación de un brazo completo, rechazo de una declaración cambiada y retención con episodios reales dentro de la capacidad |
| `tests/memory/test_mars_titan_session_parity.py` | El recorrido cronológico del lector emite lo mismo que `FinancialSession` también con centros fijos |
| `tests/training/test_cm_v1_campaign.py` | Plan, dependencias, auxiliares sin traslado ni recibo y campaña B reducida con los ejecutores reales hasta la tercera ventana |

Con el registrador no hay pasos, así que el núcleo de C conserva los parámetros de B. La campaña reducida comprueba entonces que B y B+C, y B+M y B+C+M, emiten exactamente las mismas filas en cada ventana. Es la paridad del factorial con C sin efecto, no una medida de su efecto.

## Comprobaciones CUDA pendientes

No se ha usado la GPU. `tests/training/cuda_cm_v1_control_check.py` compara en CPU y `cuda:0` un ajuste del núcleo sin pasos y su validación con C `disabled` y `penalty`, en FP32 y FP64. Su lógica se ensayó en CPU con `MARS_TITAN_CM_CONTROL_CHECK_DEVICE=cpu`, lo que no acredita CUDA:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  MARS_TITAN_CM_CONTROL_CHECK_REPORT=$PWD/cm-v1-control-cuda.json \
  UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA> \
  uv run --no-sync python -m pytest -q tests/training/cuda_cm_v1_control_check.py
```

El lector de los brazos usa la comprobación CUDA de MARS-TITAN (`tests/training/cuda_mars_titan_run_check.py`). La retención con centros fijos se calcula en CPU por diseño.

## Pendiente

- Ajustar y comparar los cuatro brazos cuando la edición histórica desde 2000 esté verificada.
- Medir memoria y caudal de la penalización C y del lector en `cuda:0` antes de declarar la sección en A y B. La orden de medición de la campaña no incluye todavía CM-v1.
- Una condición de contracción común a todos los Jacobianos admisibles, si se quiere una garantía para productos variables. Ninguna lectura actual la aporta.
- La comparación con MAE residual por sesión, diferencias emparejadas, incertidumbre por bloques y la interacción `MAE_CM − MAE_C − MAE_M + MAE_B`, descritas en el [protocolo](protocol.md).
