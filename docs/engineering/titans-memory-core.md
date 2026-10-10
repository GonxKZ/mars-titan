# Memoria neuronal y MAC con estado explícito

`mars_titan.models.titans` implementa una referencia técnica de memoria neuronal y un bloque MAC. Recibe tensores ya preparados y devuelve una salida al cierre de cada segmento. No incorpora un predictor financiero, banco episódico, etiquetas maduras ni extensiones K o C/M. Las pruebas usan fixtures y actualizaciones asociativas internas, sin optimizador externo.

## Ecuaciones y decisiones de la adaptación

La referencia científica es [Titans: Learning to Memorize at Test Time, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf), de Ali Behrouz, Peilin Zhong y Vahab Mirrokni. La sorpresa con momentum aparece en la ecuación 1, la pérdida asociativa en la 2, el olvido en la 3 y MAC en las ecuaciones 7 a 10. El [preprint arXiv v1](https://arxiv.org/abs/2501.00663v1) presenta MAC en las ecuaciones 21 a 25.

Para cada flujo y observación `x`, el núcleo proyecta `k = W_K x` y `v = W_V x`. La memoria tiene una o dos matrices cuadradas `D × D`, sin bias. Con dos capas aplica GELU exacta entre ambas. Esta adaptación compacta no reproduce la expansión, el residual ni LayerNorm descritos en la sección 3.3 de la versión final. La memoria `x + LN(MLP(x))` existe ahora como [componente desactivable con identidad propia](titans-mac-output-scale.md). La expansión sigue sin reproducirse. La normalización L2 de claves y queries, con `eps=1e-12`, se identifica en configuración y puede desactivarse. El preprint aplica además SiLU al calcular consultas, claves y valores y una convolución causal tras cada proyección (sección 4.4). La [revisión de implementaciones públicas](../research/titans-reference-implementations.md) detectó que la omisión de SiLU no estaba declarada y conserva la paridad numérica de esta memoria con `lucidrains/titans-pytorch`. Ambos componentes existen ahora como opciones desactivables con la identidad `titans_mac_paper_projections_v2`, descrita más abajo en [proyecciones de la sección 4.4](#proyecciones-de-la-sección-44). Sin ellas el núcleo es el mismo, bit a bit, que antes de añadirlas.

Se usa la pérdida `ℓ = ||M(k) − v||²`, sumada sobre componentes. Cada token actualiza sus pesos actuales de forma secuencial:

```text
α = sigmoid(A x)                 forma [B, D, 1]
η = sigmoid(e x)                 forma [B, 1, 1]
θ = theta_max · sigmoid(t x)     forma [B, 1, 1]
S_t = η · S_(t−1) − θ · ∂ℓ/∂W
W_t = (1 − α) · W_(t−1) + S_t
```

`α` se aplica por filas de salida a cada matriz. Es una decisión explícita para interpretar el olvido vectorial de la versión final. En arXiv v1 el olvido se presenta como escalar. Las tres tasas dependen de la entrada. En la identidad v1 sus proyecciones no tienen bias y `α` empieza cerca de 0,5. La opción `gate_bias` añade bias declarados con una identidad nueva y conserva v1 como valor por defecto. Su análisis está en [inicialización de las puertas](titans-gate-initialization.md). Matemáticamente `α` y `η` pertenecen a `(0, 1)` y `θ` a `(0, theta_max)`, con `0 < theta_max ≤ 1`. En coma flotante la sigmoide puede saturar en los extremos. Se rechazan NaN e infinitos antes de aplicar la sigmoide. No se promedian gradientes entre flujos ni se congelan en el inicio de un chunk.

Con una sola matriz, la derivada de referencia es `2 (Wk − v) kᵀ`. La reducción sumada conserva este factor 2. Los pesos iniciales son parámetros lentos aprendibles. Cada flujo recibe una copia independiente, momentum cero y contador cero. La inicialización usa semillas locales identificadas y restaura el RNG global de CPU.

## Derivada interna y objetivo externo

La atención puede producir una observación `y(W_prev)`. El gradiente interno requiere la derivada parcial respecto a los pesos con `y` fija. Derivar directamente la pérdida con respecto al mismo tensor `W_prev` incluiría también el camino por `y` y cambiaría la regla de actualización.

La implementación crea `U = W_prev.clone()` y calcula `g = ∂ℓ(U, y)/∂U`. `y` no depende de esa nueva variable interna. La copia conserva su conexión diferenciable con `W_prev`, igual que `y`. Al derivar después un objetivo externo a través de `W_next`, se conservan ambos caminos y las derivadas del gradiente interno. Desacoplar `y` corregiría la derivada parcial a costa de perder parte del gradiente externo.

`update(..., differentiable=True)` utiliza `autograd.grad(..., create_graph=True)`. No llama a `backward`, no acumula `.grad` en parámetros compartidos y no cambia los pesos recibidos. El modo `False` devuelve pesos, momentum y pasos sin grafo persistente. Puede funcionar dentro de `torch.no_grad()` porque habilita localmente la derivación asociativa. `torch.inference_mode()` es incompatible con la actualización y se rechaza. La API explícita decide esta política, no `module.train()` ni `module.eval()`.

## Orden de MAC y disponibilidad del segmento

MAC lee `h = M_prev(q(S))` sin escritura, concatena `[P, h, S]` y aplica atención triangular. `P` contiene parámetros aprendibles independientes de la entrada. La ecuación 7 del PDF final imprime `[P, S, h]`, mientras su figura 4a y la ecuación 22 de arXiv v1 colocan `h` antes de `S`. El contrato implementado fija `[P, h, S]` y registra esa discrepancia.

Solo las salidas de atención alineadas con las posiciones de `S` actualizan la memoria. El núcleo devuelve `y_last ⊙ M_next(y_last)`, con producto Hadamard explícito. La selección de posiciones escritas y la interpretación Hadamard del gate son decisiones identificadas de esta adaptación.

Para segmentos con más de un token, una máscara triangular sobre la concatenación no basta para afirmar causalidad por token. La consulta de `S_i` puede leer un `h_j` calculado desde un `S_j` posterior dentro del mismo segmento. La API entrega únicamente una salida `[B,D]` al cierre, cuando todo `S` está disponible. Un segmento futuro no participa en una salida ya emitida. El llamante debe agrupar exclusivamente entradas disponibles antes de ese cierre. Las fechas y la admisión de datos pertenecen al contrato del consumidor, fuera de este bloque.

Los controles se identifican en `memory_mode`:

- `online` lee la memoria anterior, actualiza con las posiciones de `S` y aplica el gate.
- `frozen` usa la misma lectura, atención y gate, conservando pesos, momentum y pasos.
- `disabled` aplica los mismos parámetros de atención a `S`, sin prefijos ni gate de memoria. Conserva una copia del estado sin cambios.

`persistent_tokens=0` elimina solo `P`. Los controles conservan los parámetros declarados para admitir comparaciones con los mismos pesos de atención. Esto no acredita un ahorro de memoria o tiempo.

## Proyecciones de la sección 4.4

La sección 4.4 del [preprint arXiv v1](https://arxiv.org/abs/2501.00663v1) describe tres detalles de las proyecciones que el núcleo no tenía completos. Usa SiLU como no linealidad al calcular consultas, claves y valores, normaliza consultas y claves con la norma L2 e incorpora una convolución 1D separable en profundidad tras cada proyección, siguiendo a modelos recurrentes lineales recientes. Cita Mamba-2 (Gu y Dao, 2024) y Gated DeltaNet. [#449](https://github.com/GonxKZ/mars-titan/issues/449) los implementa como componentes desactivables, porque la campaña desde 2000 vuelve a entrenar todos los modelos y el Titans-MAC que compare debe seguir el artículo también en este punto.

| Componente | Campo de configuración | Entrada en la identidad | Desactivado |
| --- | --- | --- | --- |
| SiLU sobre q, k y v | `MemoryConfig.qkv_silu` | `qkv_activation` | `False`, valor por defecto |
| Convolución causal en profundidad | `MemoryConfig.qkv_convolution`, el tamaño K del núcleo | `qkv_convolution` | `0`, valor por defecto |
| Norma L2 de q y k | `MemoryConfig.normalize_qk`, que ya existía | `normalize_qk` | `False`. El valor por defecto sigue siendo `True` |

La norma L2 ya estaba disponible con la misma semántica, así que se reutiliza el campo existente en lugar de duplicarlo. Las dos entradas nuevas solo aparecen en la identidad cuando se activan, igual que `gate_bias` y la memoria residual. Por eso las huellas anteriores no cambian. Con SiLU, núcleo 4 y L2 activos a la vez, la identidad añade además `projections: titans_mac_paper_projections_v2`. En el predictor financiero el conmutador es `FinancialConfig.memory_projections`, con `linear_v1` por defecto y `titans_mac_paper_projections_v2` para el núcleo del artículo. Como las demás opciones de memoria, entra en la identidad de las cuatro variantes para que el emparejamiento desde `mac_online` compare la misma configuración.

### Ecuaciones implementadas

Para un flujo, sea `y_t` la salida de atención de una posición escrita y `x_t` el token del segmento. Con los tres componentes activos:

```text
p_t = W_K y_t                                         proyección lineal sin bias
u_t[c] = Σ_{j=0..K−1} w_K[c, j] · p_{t−K+1+j}[c]      convolución causal por canal, sin bias
k_t = normalize(SiLU(u_t)),  eps = 1e-12
v_t = SiLU(conv_V(W_V y_t))
q_t = normalize(SiLU(conv_Q(W_Q x_t)))
```

Las entradas anteriores al primer token del flujo valen cero, lo que equivale al relleno por la izquierda de `ShortConvolution`. El orden es proyección lineal, convolución, SiLU y L2, el mismo que la capa `GatedDeltaNet` de flash-linear-attention, que aplica `ShortConvolution(activation='silu')` y normaliza q y k dentro del kernel. Los valores no se normalizan. Las tasas `α`, `η` y `θ` siguen calculándose desde `y_t` sin convolución ni SiLU, porque la sección 4.4 se refiere a consultas, claves y valores. La lectura final `y ⊙ M_t(y)` de la ecuación 25 no tiene proyección y tampoco cambia.

Interpretamos «separable en profundidad» como la factorización habitual en estos modelos. La proyección lineal hace de parte puntual y la convolución por canal de parte en profundidad, sin otra convolución 1 × 1 añadida. La atención softmax de MAC conserva las proyecciones estándar de `nn.MultiheadAttention`. El texto dice que se usa SiLU «para calcular consultas, claves y valores» sin precisar si incluye las de la atención, y aquí se aplica a las de la memoria, que es el componente recurrente al que se refiere la comparación con Gated DeltaNet. Es una decisión de lectura que queda declarada.

### Decisiones propias

El artículo no fija el tamaño del núcleo. Se adopta K = 4 como decisión propia, tomando la convención de la capa `GatedDeltaNet` de [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) en el commit `855e5026` (`fla/layers/gated_deltanet.py`), que usa por defecto `conv_size=4` y `conv_bias=False` con `ShortConvolution`. Gated DeltaNet ([Yang, Kautz y Hatamizadeh, 2024](https://arxiv.org/abs/2412.06464)) es uno de los dos modelos que el artículo cita al introducir la convolución. Los pesos tienen la disposición `[D, 1, K]` de `nn.Conv1d` en profundidad y su inicialización por defecto, `U(−1/√K, 1/√K)`, que hereda `ShortConvolution`. El modelo completo de flash-linear-attention vuelve a inicializar esas capas con una normal de media 0 y desviación típica 0,02. Aquí se conserva la inicialización de la capa, coherente con las proyecciones lineales del núcleo, que también mantienen la de PyTorch. Los tres núcleos se sortean con la semilla del módulo después de todos los parámetros anteriores. Con la misma semilla, los demás parámetros toman exactamente los valores de la identidad anterior, de modo que la comparación pareada solo difiere en las proyecciones.

### Estado de la convolución

Cada ventana guarda las K − 1 proyecciones lineales anteriores del flujo, antes de la convolución. `NeuralMemoryState.convolution` contiene las de claves y valores y `MACState.convolution` la de consultas, porque la proyección de consultas pertenece a MAC. Sin convolución ambas tuplas están vacías y el estado es el anterior. Las ventanas se crean a cero, se arrastran entre segmentos y llamadas, se validan como el resto del estado (forma `[B, K − 1, D]`, tipo, dispositivo, contigüidad, valores finitos y almacenamiento propio) y entran en el presupuesto de bytes. Cada fila es un flujo, así que la ventana de un activo no ve nunca a otro. Un estado nuevo, por ejemplo al empezar cada pasada de la ventana walk-forward, empieza con ventanas a cero igual que con pesos rápidos iniciales.

En `frozen` la consulta se sigue calculando y su ventana avanza, mientras que las de claves y valores no cambian porque no hay escrituras. En `disabled` no se calcula ninguna proyección de la memoria y las tres ventanas se copian sin cambios. En el predictor financiero cada decisión es un segmento de un token, así que la ventana de un activo contiene las proyecciones de sus K − 1 decisiones anteriores, que son sesiones ya observadas.

`export_state`, `restore_state`, `export_state_cpu`, `gather_state`, `select_state` y la división y unión de estados del entrenador cronológico conservan las ventanas. Los payloads añaden el campo `convolution` en la memoria y en MAC solo cuando la convolución está activa, de modo que los payloads anteriores mantienen exactamente sus campos. El estado por flujo crece en `3 (K − 1) D` elementos. Con `D = 64`, K = 4 y FP32 son 2.304 bytes sobre los 65.552 anteriores, un 3,5 %.

Cada posición se calcula con sumas elemento a elemento en un orden fijo, `((w₀p₀ + w₁p₁) + w₂p₂) + w₃p₃`, y las claves y valores se proyectan token a token como antes. Por eso procesar una secuencia de una vez o en trozos da los mismos bits. Para segmentos de más de un token, la proyección de consultas de todo el segmento puede depender del tamaño del segmento en el último bit, igual que en el núcleo anterior. En el predictor financiero los segmentos tienen un token.

### Control C y Jacobiano

Con convolución, la transición rápida depende también de las ventanas. `fast_state_point` añade a pesos y momentum las ventanas de claves, valores y consultas, en ese orden, y `fast_state_dimension` da el orden de `z`, `2 · depth · D² + 3 (K − 1) D`. Así el control C y el Jacobiano denso siguen midiendo la transición completa y no un bloque. El bloque de la ventana de consultas es un desplazamiento nilpotente, porque su entrada nueva `W_Q x` no depende de `z`. Las ventanas de claves y valores sí dependen de los pesos a través de `y`. El contrato de C añade `fast_state_layout` solo con convolución y su base cambia de dimensión, de 16.384 a 16.960 con `D = 64`. Un control C sobre el núcleo nuevo es, por tanto, otra identidad de C.

### Correspondencia con módulos y pruebas

| Ecuación o propiedad | Módulo | Prueba en `tests/models/titans/test_paper_projections.py` |
| --- | --- | --- |
| `u_t` como convolución causal de `nn.Conv1d` | `causal_convolution.py` | `test_convolution_matches_causal_conv1d_and_gives_the_same_bits_in_chunks` |
| `k_t`, `v_t` y la actualización asociativa | `NeuralMemory.update` y `project` | `test_memory_update_writes_silu_of_the_causal_convolution_with_l2_keys` |
| `q_t` y la lectura de MAC | `TitansMAC.forward` | `test_mac_reads_with_the_l2_of_silu_of_the_convolved_query` |
| Componentes apagados idénticos al núcleo anterior | Todo el núcleo y el predictor | `test_disabled_projections_keep_the_develop_core_bit_for_bit` y `..._financial_predictor_bit_for_bit` |
| Mismos bits de una vez y en trozos | `NeuralMemory.update` | `test_memory_update_in_chunks_gives_the_same_bits_as_one_call` |
| El futuro no cambia el pasado | Convolución y MAC | `test_convolution_matches_causal_conv1d...` y `test_future_segments_do_not_change_past_outputs_or_windows` |
| Restaurar a mitad reproduce los bits | `export_state`, `restore_state`, `export_state_cpu` | `test_restoring_the_mac_state_mid_sequence_reproduces_the_same_bits` y `test_financial_restore_mid_sequence_reproduces_the_same_bits` |
| Aislamiento por flujo | Estado por filas | `test_windows_stay_isolated_per_flow` y `test_financial_selection_gathering_and_trainer_rows_carry_each_flow_window` |
| Ventanas según el modo | `TitansMAC.forward` | `test_windows_only_advance_for_the_projections_each_mode_computes` |
| Jacobiano con ventanas | `transition_jacobian.py`, `local_control.py` | `test_dense_jacobian_with_windows_matches_differences_and_shifts_the_query_window` y `test_local_control_projects_the_jacobian_with_windows` |
| Punto de control del entrenador | `export_state_cpu`, `torch.save` con `weights_only`, `restore_state` y `_split` | `test_trainer_checkpoint_round_trip_keeps_the_windows_of_every_flow` |
| Consumidor congelado de MARS-TITAN | `frozen_financial.py`, que admite el módulo nuevo y lo incluye en su identidad de implementación | `test_frozen_consumer_admits_the_convolution_and_reproduces_the_predictor` |

La prueba de paridad compara huellas SHA-256 de salidas, estados, gradientes de medida, parámetros e identidades en FP32 estricto, capturadas en `develop` 030e7b31 antes del cambio. Cubre MAC en los tres modos, sin L2, con memoria residual y `gate_bias`, y el predictor financiero en sus tres variantes MAC, también con la memoria de la receta de la campaña. Diecisiete mutaciones dirigidas de la lógica nueva se ejecutaron contra estas pruebas. Dieciséis fallaron como se esperaba. La que sobrevivió quitaba un `detach` redundante en MAC, porque esa rama ya se calcula sin grafo, y el `detach` se eliminó.

### Desviaciones que siguen

- La memoria mantiene expansión 1, con matrices `D × D`.
- La actualización es secuencial por token. No se implementa la forma paralela por bloques de la sección 3.2 (ecuaciones 16 a 18).
- La salida de MAC es el producto Hadamard `y ⊙ M(y)`. No se implementan la normalización y la puerta con una capa lineal antes de la proyección de salida que la misma sección 4.4 menciona, ni conexiones residuales alrededor del bloque.
- `α` se aplica por filas de salida, como se explica arriba.
- La atención softmax no usa SiLU ni convolución, por la lectura declarada del texto.
- El tamaño del núcleo y la inicialización de la convolución son decisiones propias.

### Paridad con implementaciones públicas

El [arnés de paridad](../../benchmarks/titans_reference_parity.py) con `lucidrains/titans-pytorch@1d40c445` se volvió a ejecutar en CPU sobre esta rama, con la configuración antigua. Todas las cifras coinciden exactamente con el [recibo original](../../reports/engineering/titans-reference-parity-20261009/parity-cpu.json). Solo cambian los hashes de los archivos del núcleo ([recibo](../../reports/engineering/titans-paper-projections-20261010/parity-cpu-previous-core.json)). La configuración nueva no tiene equivalente en lucidrains. Su memoria admite una activación opcional en las proyecciones, que por defecto no existe y que su transformador MAC no fija, pero no tiene convolución y normaliza q y k con un RMSNorm por cabeza en lugar de L2. La operación Titans de flash-linear-attention recibe q, k y v ya calculados y queda fuera de esta comparación. La misma ejecución en `cuda:0` también reproduce el recibo original, salvo el máximo del asignador.

### Coste medido sin entrenar

Con la receta de la campaña, 128 flujos y 8 instantes de entradas reales de la vista `US+CN/fold-012`, en `cuda:0` y FP32 estricto, el núcleo nuevo multiplica en `mac_online` la mediana del forward con grafo por 1,16 (de 90,4 a 105,1 ms), la del backward por 1,19 (de 57,1 a 67,7 ms) y la de la inferencia por 1,14. El pico asignado pasa de 433,1 a 437,1 MiB. No se construyó ningún optimizador. El [recibo y su descripción](../../reports/engineering/titans-paper-projections-20261010/README.md) recogen las dos fases de la medida, `mac_frozen`, los percentiles y las condiciones de carga, que no permiten leer las cifras como tiempos absolutos.

## API, recuperación y límites

`NeuralMemory.initial_state(B)` crea `NeuralMemoryState`. `read(query, state)` recibe `[B,T,D]` y devuelve otra lectura de esa forma. `update(observed, state)` devuelve un estado nuevo. `TitansMAC.initial_state(B)` envuelve la memoria en `MACState` y `forward(segment, state)` devuelve `(output, next_state)`. La salida de evaluación está desacoplada. Las filas del estado representan flujos. Cualquier permutación debe aplicarse conjuntamente a entradas, pesos, momentum y pasos.

El estado conserva tuplas de pesos y momentum `[B,D,D]`, pasos `int64[B]` y una huella del contrato. Las lecturas, actualizaciones y copias de recuperación no modifican el estado recibido. Se rechazan tensores de estado que compartan almacenamiento, aunque sean vistas disjuntas. Los tensores deben ser contiguos y finitos. Las entradas y el estado deben tener el tipo y dispositivo del módulo. La referencia admite float32 y float64 y rechaza autocast.

`export_state` y `restore_state` copian tensores y validan la versión y la configuración completa. El `state_dict` del módulo también conserva su contrato y rechaza configuraciones incompatibles aunque coincidan dimensiones, incluso dentro de un módulo padre. Recuperar la siguiente llamada exige conservar tanto esos parámetros como el estado rápido. La exportación corta el grafo de autograd y no reanuda una derivación exterior pendiente. La API recibe un diccionario ya cargado, no valida ni limita archivos arbitrarios de `torch.load`.

Los límites predeterminados de memoria son 256 flujos, 64 tokens por llamada y 64 MiB de estado. La configuración admite como máximo `D=512`, dos capas, 256 tokens y 256 MiB. MAC limita por separado el segmento, hasta el máximo de la memoria, y los elementos `B × heads × (|P| + 2|S|)²` de atención, con 8.388.608 por defecto y máximo 67.108.864. El control desactivado usa `|S|` en este cálculo.

El presupuesto lógico del estado es `B × (2 × depth × D² × bytes_por_elemento + 8)`, más `B × 3 (K − 1) D × bytes_por_elemento` con convolución. Ese término cuenta también la ventana de consultas, aunque viaje en `MACState`. También se mide el almacenamiento retenido por los tensores para rechazar vistas pequeñas que retengan bloques mayores. Los límites se comprueban antes de copiar el estado o calcular atención. No equivalen al RSS ni acotan el grafo exterior que un llamante conserve entre llamadas diferenciables. El consumidor debe fijar una política explícita de duración de ese grafo.

Las 61 pruebas CPU contrastan la ecuación lineal, momentum, olvido por filas, gradientes externos, `gradcheck`, aislamiento entre flujos, lectura posterior, recuperación, máscaras de atención y entradas inválidas. La revisión corrigió la dependencia accidental del dispositivo ambiental durante la inicialización. Se mantiene la misma configuración y los mismos pesos con la CPU solicitada de forma explícita.

También se ejecutaron comprobaciones CUDA en la RTX 4070 Laptop con FP64 `[2,2,4]` y FP32 `[2,8,64]`. Contrastan salidas, gradientes y recuperación, con el asignador Torch limitado a 512 MiB. Los picos observados fueron 17,10 y 19,76 MB asignados, respectivamente. El [recibo](../../reports/research/titans-memory-core-verification-20261008.json) conserva tolerancias y diferencias reales, además de la cobertura y las doce mutaciones de la revisión anterior al arreglo de inicialización. La carga externa permaneció activa, por lo que estos casos no constituyen un benchmark aislado. La integración financiera y el estudio experimental siguen pendientes.
