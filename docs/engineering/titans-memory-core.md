# Memoria neuronal y MAC con estado explícito

`mars_titan.models.titans` implementa una referencia técnica de memoria neuronal y un bloque MAC. Recibe tensores ya preparados y devuelve una salida al cierre de cada segmento. No incorpora un predictor financiero, banco episódico, etiquetas maduras ni extensiones K o C/M. Las pruebas usan fixtures y actualizaciones asociativas internas, sin optimizador externo.

## Ecuaciones y decisiones de la adaptación

La referencia científica es [Titans: Learning to Memorize at Test Time, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf), de Ali Behrouz, Peilin Zhong y Vahab Mirrokni. La sorpresa con momentum aparece en la ecuación 1, la pérdida asociativa en la 2, el olvido en la 3 y MAC en las ecuaciones 7 a 10. El [preprint arXiv v1](https://arxiv.org/abs/2501.00663v1) presenta MAC en las ecuaciones 21 a 25.

Para cada flujo y observación `x`, el núcleo proyecta `k = W_K x` y `v = W_V x`. La memoria tiene una o dos matrices cuadradas `D × D`, sin bias. Con dos capas aplica GELU exacta entre ambas. Esta adaptación compacta no reproduce la expansión, el residual ni LayerNorm descritos en la sección 3.3 de la versión final. La normalización L2 de claves y queries, con `eps=1e-12`, se identifica en configuración y puede desactivarse.

Se usa la pérdida `ℓ = ||M(k) − v||²`, sumada sobre componentes. Cada token actualiza sus pesos actuales de forma secuencial:

```text
α = sigmoid(A x)                 forma [B, D, 1]
η = sigmoid(e x)                 forma [B, 1, 1]
θ = theta_max · sigmoid(t x)     forma [B, 1, 1]
S_t = η · S_(t−1) − θ · ∂ℓ/∂W
W_t = (1 − α) · W_(t−1) + S_t
```

`α` se aplica por filas de salida a cada matriz. Es una decisión explícita para interpretar el olvido vectorial de la versión final. En arXiv v1 el olvido se presenta como escalar. Las tres tasas dependen de la entrada, sin bias. Matemáticamente `α` y `η` pertenecen a `(0, 1)` y `θ` a `(0, theta_max)`, con `0 < theta_max ≤ 1`. En coma flotante la sigmoide puede saturar en los extremos. Se rechazan NaN e infinitos antes de aplicar la sigmoide. No se promedian gradientes entre flujos ni se congelan en el inicio de un chunk.

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

## API, recuperación y límites

`NeuralMemory.initial_state(B)` crea `NeuralMemoryState`. `read(query, state)` recibe `[B,T,D]` y devuelve otra lectura de esa forma. `update(observed, state)` devuelve un estado nuevo. `TitansMAC.initial_state(B)` envuelve la memoria en `MACState` y `forward(segment, state)` devuelve `(output, next_state)`. La salida de evaluación está desacoplada. Las filas del estado representan flujos. Cualquier permutación debe aplicarse conjuntamente a entradas, pesos, momentum y pasos.

El estado conserva tuplas de pesos y momentum `[B,D,D]`, pasos `int64[B]` y una huella del contrato. Las lecturas, actualizaciones y copias de recuperación no modifican el estado recibido. Se rechazan tensores de estado que compartan almacenamiento, aunque sean vistas disjuntas. Los tensores deben ser contiguos y finitos. Las entradas y el estado deben tener el tipo y dispositivo del módulo. La referencia admite float32 y float64 y rechaza autocast.

`export_state` y `restore_state` copian tensores y validan la versión y la configuración completa. El `state_dict` del módulo también conserva su contrato y rechaza configuraciones incompatibles aunque coincidan dimensiones, incluso dentro de un módulo padre. Recuperar la siguiente llamada exige conservar tanto esos parámetros como el estado rápido. La exportación corta el grafo de autograd y no reanuda una derivación exterior pendiente. La API recibe un diccionario ya cargado, no valida ni limita archivos arbitrarios de `torch.load`.

Los límites predeterminados de memoria son 256 flujos, 64 tokens por llamada y 64 MiB de estado. La configuración admite como máximo `D=512`, dos capas, 256 tokens y 256 MiB. MAC limita por separado el segmento, hasta el máximo de la memoria, y los elementos `B × heads × (|P| + 2|S|)²` de atención, con 8.388.608 por defecto y máximo 67.108.864. El control desactivado usa `|S|` en este cálculo.

El presupuesto lógico del estado es `B × (2 × depth × D² × bytes_por_elemento + 8)`. También se mide el almacenamiento retenido por los tensores para rechazar vistas pequeñas que retengan bloques mayores. Los límites se comprueban antes de copiar el estado o calcular atención. No equivalen al RSS ni acotan el grafo exterior que un llamante conserve entre llamadas diferenciables. El consumidor debe fijar una política explícita de duración de ese grafo.

Las pruebas CPU FP64 contrastan la ecuación lineal, momentum, olvido por filas, gradientes externos, `gradcheck`, aislamiento entre flujos, lectura posterior, recuperación, máscaras de atención y entradas inválidas. Son comprobaciones técnicas del núcleo. La integración con el predictor, los experimentos y la verificación CUDA quedan fuera de esta entrega.
