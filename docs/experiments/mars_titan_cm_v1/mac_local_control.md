# Diagnóstico local del estado rápido de MAC

[`MACProjectionControl`](../../../src/mars_titan/models/titans/local_control.py) estudia una compresión del Jacobiano de la transición rápida real de Titans-MAC. Está desactivado por defecto. El modo de penalización produce un término opcional y no ejecuta un optimizador.

El estado por flujo es `z = (vec W₁, …, vec W_L, vec m₁, …, vec m_L)`, de dimensión `2LD²`. Incluye pesos rápidos y momentum. Para un token x, `F(z,x)` contiene la lectura de memoria previa, atención, tasas dependientes de su salida y actualización asociativa. La derivada interna mantiene fija la observación frente a su variable local. La derivada exterior conserva la dependencia de esa observación respecto a los pesos previos.

Los contadores son metadatos discretos. La puerta Hadamard final afecta a la predicción, pero no realimenta la transición rápida actual. No se añade una coordenada artificial para esa salida ni para la cabeza financiera. Un banco que refine la salida después de MAC tampoco entra en este operador. Si cambia esa dependencia, deberá revisarse la definición de F.

## Operador comprimido

El control obtiene `A = RᵀJR`, con `J = ∂F/∂z`, mediante productos Jacobiano-vector. R es una base fija generada en CPU, con QR FP64, generador independiente y diagonal positiva del factor triangular. Después se convierte a la precisión del núcleo. Una conversión explícita de dtype reconstruye la base desde esa definición, sin acumular redondeos de conversiones intermedias.

En aritmética exacta y con columnas ortonormales, `u*Au = (Ru)*J(Ru)` para cada u unitario. Por tanto, `W(A) ⊆ W(J)` y `w(A) ≤ w(J)`. En coma flotante, la ortogonalidad y las estimaciones son aproximadas. La [corrección angular](mathematical_scope.md) solo corresponde a A. No es una cota superior del radio de J completo.

La compresión puede omitir direcciones expansivas. Para `J = diag(0.2,0.2,2)` y R formado por las dos primeras coordenadas, el radio comprimido es 0,2 y el completo es 2. Tampoco representa las potencias del operador: para `J = [[0,2],[0.6,0]]` y `R=e₁`, `A=0`, pero `RᵀJ²R=1.2`. No se atribuyen estabilidad global ni retención a esta medida.

## Configuración y uso

`MACProjectionConfig` identifica modo disabled/diagnostic/penalty, rango, frecuencia, semilla, rejilla, umbral, peso, máximo de flujos y bytes estimados. Admite MAC online, dimensión hasta 64, un token por transición, rango 1–4 y 2–128 ángulos. La penalización exige peso positivo. Los otros modos exigen peso cero.

El control usa [`autograd.functional.jvp`](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.functional.jvp.html) y el backend SDPA Math. J incluye derivadas de la actualización asociativa y derivar la penalización puede necesitar terceras derivadas de su pérdida. El backend CPU predeterminado de la instalación comprobada no admite la derivada de su atención fusionada que necesita esta ruta. El consumidor del nuevo factorial debe usar Math también para su B, con identidad propia.

La selección se fija sobre el grupo lógico completo antes de dividirlo en lotes físicos:

```python
selection = control.select_flows(flow_ids, observed_steps, context_id=context_id)
result = control(
    mac,
    segment,
    state,
    flow_ids=block_flow_ids,
    observed_steps=block_observed_steps,
    selection=selection,
    context_id=context_id,
    differentiable=True,
)
```

`context_id` es la huella SHA-256 que el coordinador calcula desde el checkpoint confirmado y el manifiesto del grupo lógico. El control comprueba su formato y coincidencia, no inspecciona esos artefactos externos. El plan conserva también la identidad del control y los pares ID/contador canónicos. Rechaza un contexto distinto, otro contrato, contadores obsoletos y bloques ajenos. El caso desactivado no exige selección.

Un flujo es elegible cuando su siguiente contador de observaciones es múltiplo de `frequency`. Se eligen los primeros IDs canónicos hasta `max_flows`. Esta selección determinista no pretende ser una muestra aleatoria. Los grupos admiten hasta 4.096 IDs ASCII únicos de hasta 128 caracteres. Cada bloque mide únicamente la intersección con los IDs seleccionados del grupo.

El resultado contiene operadores, estimaciones, IDs medidos, índices en el bloque, identidad de selección y número de reevaluaciones de F. La penalización del bloque es `weight * sum(max(estimate-threshold,0)²) / n_selected_group`. Sumar contribuciones de los bloques conserva la media del grupo. Un bloque sin mediciones devuelve `None`, sin presentar una ausencia como cero observado.

El modo diagnostic devuelve resultados desacoplados. El modo penalty conserva el gradiente solo con `differentiable=True`. Con False devuelve el valor numérico sin grafo. La instantánea z se desacopla del historial en ambos casos. En la penalización diferenciable se conserva el camino del token actual. Las reevaluaciones usan copias y no publican otra propuesta de memoria. El llamante sigue siendo responsable de confirmar una sola transición ordinaria por decisión.

## Presupuesto y recuperación

El plan conserva una estimación para todos los flujos seleccionados del grupo. Se comprueba antes del cálculo y no se reinicia al dividirlo. El coordinador registra ese presupuesto una vez por identidad de selección y suma las reevaluaciones efectivas de los bloques.

La estimación incluye generación de base, pesos y momentum temporales, grafos JVP, atención y cálculo espectral pequeño. Los factores de reserva se contrastan con eventos de almacenamiento guardados por autograd. Esos eventos cuentan repeticiones y almacenamiento compartido, no miden memoria viva simultánea. La estimación no certifica RSS, espacios internos de biblioteca ni el grafo previo del backbone. La precisión, el backend y la versión del cálculo forman parte del contrato.

`state_dict` conserva la base y su huella. La carga directa o anidada debe comprobar el contrato antes de copiar, incluso con `strict=False`. Cambiar modo, frecuencia, configuración MAC, precisión o bytes de la base invalida la carga. Las firmas de tensores detectan modificaciones ordinarias durante el uso. `verify_basis` y el guardado verifican los bytes. Una modificación mediante `.data` puede eludir el contador de versiones y requiere esa frontera fuerte.

## Integración y comprobaciones

`FinancialPredictor(..., local_control=None)` conserva la identidad y el recorrido anteriores. Una configuración explícita añade la base y el contrato al modelo y aplica Math a toda la preparación, incluido B con modo disabled. Solo se admite `mac_online`. El emparejamiento copia parámetros comunes y exige la misma base y configuración numérica, salvo modo y peso. Las cargas ordinarias siguen rechazando modos diferentes.

Cada `prepare` recibe `control_selection` y `control_context_id` cuando C está activo. Valida el plan antes de codificar. Después prepara una sola transición ordinaria y evalúa el diagnóstico sobre copias del estado previo. `PreparedDecisions.local_control` devuelve su resultado. `working_state` expone la salida anterior a la cabeza para la lectura episódica posterior, sin repetir MAC. El modo desactivado no necesita selección.

El 8 de octubre de 2026 pasaron 288 pruebas CPU de Titans y C/M, incluidas 60 nuevas. El diagnóstico coincide con el Jacobiano denso pequeño, conserva el gradiente hacia la fusión y el backbone, desacopla la historia y conserva la selección y el denominador al dividir un grupo. Doce mutaciones dirigidas fueron detectadas con las 48 primeras pruebas focales. Una comparación independiente con la revisión anterior obtuvo igualdad exacta en 32 pares de predicción y estado, con cuatro controles, dos precisiones y modos train/eval, sin C explícito.

El coste técnico siguiente corresponde a dos flujos, ventanas de 64 sesiones, una capa de Transformer, cuatro prefijos persistentes, cuatro cabezas, rango 4, 64 ángulos y un flujo seleccionado. Se midieron selección, prepare y gradiente a los parámetros conectados. El objetivo del fixture es la suma cuadrática de salidas más el término opcional, sin objetivos financieros ni optimizador. Se utilizó CPU FP64, dos hilos, un calentamiento y tres repeticiones, con carga ajena sin aislar.

| Dimensión | B con Math | Diagnóstico y gradiente de la salida | Penalización y gradiente conjunto |
| ---: | ---: | ---: | ---: |
| 32 | 9,56 ms | 30,21 ms | 49,25 ms |
| 64 | 12,40 ms | 35,77 ms | 51,01 ms |

Son medianas internas del fixture, sin conversión del lote, creación del modelo o guardado. El proceso alcanzó 690.569.216 bytes de RSS, incluidas importaciones y comprobaciones de paridad. El presupuesto estimado de C fue 20.889.600 y 75.571.200 bytes para las dos dimensiones. No es una medida incremental de RAM. No se han medido energía, una sesión financiera real ni efecto predictivo. La comprobación CUDA de esta opción sigue pendiente.
