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

El grupo lógico es el evento completo, con todos sus bloques físicos. La selección se fija antes de dividirlo y su contexto identifica parámetros, índice, tramo e instante, así que dividir en bloques o reanudar no cambia los flujos medidos. El término conserva el gradiente del token actual y de los parámetros de MAC, y el estado rápido medido se desacopla de la historia. Si un tramo termina sin etiquetas maduras no hay paso y su término se descarta (`control_groups_discarded`). C no añade actualizaciones, de modo que B y B+C tienen el mismo número de pasos. El modo `diagnostic` se rechaza en el ajuste, con acumulación o sin ella. Sin control local, la identidad del entrenador conserva literalmente su forma anterior.

Al predecir con el núcleo congelado, C no interviene. Un núcleo ajustado con la penalización se copia a un gemelo `disabled` con la misma base mediante `copy_paired_parameters`, y su emisión es la misma.

### Acumulación por bloques con C

La receta del núcleo es la de Titans-MAC, que prevé `accumulation_rows=128` si el tramo completo de `mac_online` no cabe en los 8 GB de la GPU. Con acumulación, el entrenador emite cada bloque con el mismo cálculo, corta su grafo y, al actualizar, repite el tramo por bloques de flujos desde su estado inicial. La primera versión de C rechazaba esa combinación en el entrenador, y el plan de la campaña rechazaba la sección `cm_v1` si la receta fijaba la opción. Para admitirla había que comprobar que el gradiente del objetivo completo se puede repartir entre bloques sin cambiar su valor.

Sea un tramo con N etiquetas maduras de pérdida ℓ_i y G grupos medidos. Cada grupo g es un evento y su selección S_g se fija sobre todos sus flujos:

```text
J(θ)   = (1/N) Σ_i ℓ_i(θ) + (1/G) Σ_g P_g(θ)
P_g(θ) = (peso / |S_g|) Σ_{f ∈ S_g} h_{f,g}(θ)
h_{f,g}(θ) = max(ŵ(Rᵀ ∂F/∂z(z̄_{f,g}, x_{f,g}(θ), θ) R) − umbral, 0)²
```

z̄_{f,g} es el valor desacoplado del estado rápido del flujo f al llegar al evento g y x_{f,g}(θ) su token actual. El término se descompone por flujos por tres razones:

1. h_{f,g} solo depende de θ, de las entradas del flujo en ese evento y del valor z̄_{f,g}. El estado medido se copia sin grafo y el token sale de codificadores y fusión que actúan fila a fila, sin normalización por lote y con dropout nulo. El operador se calcula sobre un estado de un solo flujo, la atención de MAC no mezcla flujos y el radio se estima matriz a matriz.
2. |S_g| y G no dependen del bloque. La selección se fija con los contadores de todos los flujos del evento antes de dividirlo, y G es el número de grupos medidos que registra la emisión.
3. θ no cambia dentro del tramo y z̄_{f,g} solo depende del estado del flujo al empezar el tramo y de sus propias entradas, que la repetición recorre en el orden original.

Entonces, para cualquier partición B_1, …, B_K de los flujos que tienen etiquetas o términos medidos en el tramo,

```text
J(θ) = Σ_k [ (N_k/N) L_k(θ) + (1/G) Σ_g (peso / |S_g|) Σ_{f ∈ S_g ∩ B_k} h_{f,g}(θ) ]
```

con L_k la pérdida media de las N_k etiquetas del bloque k. El gradiente es lineal, así que ∇J es la suma de los gradientes de los bloques. La media de la tarea se reparte por la fracción de etiquetas, como en la acumulación sin C. Cada evento ya divide su término entre |S_g| cuando se calcula por bloques físicos, de modo que la única corrección es dividir la suma de los términos del bloque entre G, el número de grupos del tramo, y no entre los términos que caen en ese bloque. Un flujo medido sin etiquetas en el tramo también tiene que repetirse, aunque solo aporte su término de C.

La implementación sigue esa forma. Cada lote del tramo guarda el plan de C de su evento. La emisión cuenta el término sin conservar su grafo, que retendría las activaciones del lote, y el tramo guarda solo los flujos medidos de cada grupo. `_replay` añade esos flujos a los que tienen etiquetas, repite cada lote con su plan y suma al objetivo del bloque sus términos divididos por G. Al terminar exige que la repetición haya recalculado exactamente los flujos medidos al emitir. La identidad del entrenador añade `penalty_accumulation` solo en esta combinación, así que ninguna identidad anterior cambia.

La igualdad no es bit a bit en general, porque cambian el orden de las sumas y el reparto de la división. En float64 y CPU, con tres flujos, dos configuraciones de C y bloques de 1, 2 (que deja un bloque incompleto) y 3 flujos, la mayor diferencia de los gradientes registrados frente al tramo completo fue 1,1·10⁻¹⁶ en absoluto y 2·10⁻¹⁴ relativa a la mayor magnitud de cada tensor. Con un único bloque coincidieron bit a bit en ese fixture. La prueba declara rtol 10⁻¹⁰ y atol 10⁻¹³, las mismas que la acumulación sin C.

El coste añadido no está medido en `cuda:0`. Cada término de C se calcula dos veces, al emitir para los contadores y al repetir con grafo, igual que el forward de la tarea. Los flujos medidos sin etiquetas añaden su recorrido del tramo a la repetición, aunque con dos flujos medidos por evento y casi todos los flujos con etiquetas en cada tramo deberían ser pocos. En el fixture técnico, el mayor grafo guardado al retropropagar bajó de 6,6 MB con el tramo completo a 2,2 MB con bloques de un flujo, con tres flujos medidos por evento.

La repetición también vuelve a validar el plan del evento en cada `prepare`, que lo valida dos veces. En CPU y con 4.202 flujos por evento, cada validación de un bloque de 128 flujos costó unos 6,2 ms. La emisión ya paga ese coste en cada lote y la repetición lo paga en cada par de bloque y lote con flujos. La medida en `cuda:0` dirá si pesa lo bastante como para validar el plan una sola vez por evento.

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

La sección opcional `cm_v1` de la campaña solo declara la ruta de la declaración y la semilla de búsqueda. Los dos núcleos son trabajos auxiliares: se ajustan en las ventanas reentrenadas con sus casos y finalistas, no tienen traslado y no publican recibo de ventana. Cada búsqueda de un brazo depende de las búsquedas de su núcleo en la ventana y cada finalista del finalista del núcleo con su semilla. Los ejecutores declaran `fastpath=False` mientras dura cada trabajo. Los nombres de los cuatro brazos son los de la comparación declarada y no cambian. La etapa de políticas solo puede usar como predictores esos brazos, nunca los núcleos auxiliares. Las configuraciones A y B no declaran todavía la sección. La campaña elegida, la A, la incluye en su [declaración ampliada](../../research/training-campaign-2000.md#declaración-preparada-de-las-familias-pendientes), donde añade 1.080 ajustes (360 de núcleos). En B añadiría 408 ajustes con 336 traslados (136 de núcleos).

## Comprobaciones

Todas en CPU, con `CUDA_VISIBLE_DEVICES=-1`, dos hilos y pruebas por archivo.

| Archivo | Qué comprueba |
| --- | --- |
| `tests/training/test_financial_run_control.py` | Parámetros y base emparejados, identidad, rechazos, paridad exacta con B sin flujos medidos con acumulación y sin ella, objetivo de cada tramo igual a la tarea más la media de sus grupos, frecuencia según el contador de cada flujo, términos descartados sin paso, linealidad del gradiente en el peso sin recorte, la cabeza fuera del camino de C, grupo lógico frente a bloque físico, validación igual a B y reanudación exacta con acumulación y sin ella. Con `accumulation_rows` de 1, 2 y 3, gradientes iguales al tramo completo en float64 con un flujo medido sin etiquetas en algún tramo, contadores iguales bit a bit, término emitido sin grafo y menor grafo vivo |
| `tests/cm/test_operator_dynamics.py` | A1/A2, expresión del operador fijo, productos frente a su cálculo explícito, un operador expansivo que no se confunde con el contraejemplo y rechazos |
| `tests/models/titans/test_transition_jacobian.py` | `RᵀJR` igual a la compresión de C, J frente a diferencias centrales en FP64, trayectoria, `I + η D_z f` en los dos modos de selección y sin banco, y `first_read` con el episodio del primer paso donde una nueva búsqueda elegiría otro |
| `tests/training/test_cm_v1_factorial.py` | Declaración, presupuesto de C, núcleos emparejados con el mismo número de pasos, cada factor solo donde se declara, control del padre, recuperación de un brazo completo, rechazo de una declaración cambiada al ajustar y al trasladar, y retención con episodios reales dentro de la capacidad |
| `tests/memory/test_mars_titan_session_parity.py` | El recorrido cronológico del lector emite lo mismo que `FinancialSession` también con centros fijos |
| `tests/training/test_cm_v1_campaign.py` | Plan, dependencias, salida de los auxiliares, auxiliares sin traslado ni recibo ni papel de predictor en la etapa de políticas y campaña B reducida con los ejecutores reales hasta la tercera ventana |

La [mutación dirigida](results.md#factorial-sobre-titans-mac) cubrió el objetivo de C, la selección por evento, los contadores, los rechazos, el padre, el gemelo disabled, la retención, la campaña y las lecturas del operador. Para la acumulación con C se aplicaron de uno en uno siete defectos: omitir los flujos medidos sin etiquetas con la comprobación final y sin ella, no dividir entre G, dividir entre los términos del bloque, conservar el grafo del término emitido, no contar los grupos del objetivo al repetir y duplicar el peso de la tarea con C. Los siete hicieron fallar alguna prueba.

Antes de cambiar el entrenador se guardaron las huellas de los gradientes registrados, la auditoría y el historial de ocho recorridos con el código de `develop`: `mac_online` sin control, B con C `disabled` sin acumulación y con bloques de 1 y 2 flujos, y la penalización sin acumulación con truncamiento 3 y 1. Con el cambio, las ocho huellas coinciden bit a bit.

Con el registrador no hay pasos, así que el núcleo de C conserva los parámetros de B. La campaña reducida comprueba entonces que B y B+C, y B+M y B+C+M, emiten exactamente las mismas filas en cada ventana. Es la paridad del factorial con C sin efecto, no una medida de su efecto.

## Comprobaciones CUDA

`tests/training/cuda_cm_v1_control_check.py` compara en CPU y `cuda:0` un ajuste del núcleo sin pasos y su validación con C `disabled` y `penalty`, en FP32 y FP64, sin acumulación y con bloques de dos flujos. Sus ocho casos pasaron en `cuda:0` el 9 de octubre ([recibo](../../../reports/engineering/cuda-checks-20261009/cm-v1-control-cuda.json)), después de inicializar CUDA antes de reiniciar el pico de memoria, que antes los hacía fallar sin comparar nada. Con estos fixtures pequeños el recorrido tarda más en CUDA que en CPU porque lo dominan los lanzamientos, así que esos tiempos no estiman la campaña. La orden:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  MARS_TITAN_CM_CONTROL_CHECK_REPORT=$PWD/cm-v1-control-cuda.json \
  UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA> \
  uv run --no-sync python -m pytest -q tests/training/cuda_cm_v1_control_check.py
```

El lector de los brazos usa la comprobación CUDA de MARS-TITAN (`tests/training/cuda_mars_titan_run_check.py`). La retención con centros fijos se calcula en CPU por diseño.

La memoria y el caudal de los dos núcleos con `accumulation_rows` en `null` y en 128 pueden medirse con la [orden de caudal de la campaña](../../research/training-campaign-2000.md#medición-de-caudal) y la declaración preparada (`--extensions`), que necesita las vistas reales. Los núcleos comparten la receta de `mac_online`, cuya estimación en `cuda:0` es de 14,74 GiB por tramo con 5.023 flujos sin acumulación y 1,24 GiB con `accumulation_rows=128` ([resumen](../../../reports/engineering/cuda-checks-20261009/README.md)).

## Pendiente

- Ajustar y comparar los cuatro brazos en la campaña A. La edición histórica desde 2000 y sus objetivos ya están verificados, pero el bloqueo de aprendizaje sigue activo.
- Copiar la sección a la configuración de A con la opción de memoria que se fije para la receta de Titans-MAC. El caudal de la penalización C y del lector en `cuda:0` sigue sin medir. La orden de medición recorre ya los dos núcleos con `accumulation_rows` en `null` y en 128 y el lector de cada brazo.
- Una condición de contracción común a todos los Jacobianos admisibles, si se quiere una garantía para productos variables. Ninguna lectura actual la aporta.
- La comparación con MAE residual por sesión, diferencias emparejadas, incertidumbre por bloques y la interacción `MAE_CM − MAE_C − MAE_M + MAE_B`, descritas en el [protocolo](protocol.md).
