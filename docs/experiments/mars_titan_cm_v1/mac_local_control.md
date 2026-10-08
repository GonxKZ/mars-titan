# Diagnóstico local del estado rápido de MAC

[`MACProjectionControl`](../../../src/mars_titan/models/titans/local_control.py) estudia una compresión del Jacobiano de la transición rápida real de Titans-MAC. Está desactivado por defecto. El modo de penalización produce un término opcional y no ejecuta un optimizador.

El estado por flujo es `z = (vec W₁, …, vec W_L, vec m₁, …, vec m_L)`, de dimensión `2LD²`. Incluye pesos rápidos y momentum. Para un token x, `F(z,x)` contiene la lectura de memoria previa, atención, tasas dependientes de su salida y actualización asociativa. La derivada interna mantiene fija la observación frente a su variable local. La derivada exterior conserva la dependencia de esa observación respecto a los pesos previos.

Los contadores son metadatos discretos. La puerta Hadamard final afecta a la predicción, pero no realimenta la transición rápida actual. No se añade una coordenada artificial para esa salida ni para la cabeza financiera. Un banco que refine la salida después de MAC tampoco entra en este operador. Si cambia esa dependencia, deberá revisarse la definición de F.

## Operador comprimido

El control obtiene `A = RᵀJR`, con `J = ∂F/∂z`, mediante productos Jacobiano-vector. R es una base fija generada en CPU, con QR FP64, generador independiente y diagonal de R triangular positiva. Después se convierte a la precisión del núcleo. Una conversión explícita de dtype reconstruye la base desde esa definición, sin acumular redondeos de conversiones intermedias.

En aritmética exacta y con columnas ortonormales, `u*Au = (Ru)*J(Ru)` para cada u unitario. Por tanto, `W(A) ⊆ W(J)` y `w(A) ≤ w(J)`. En coma flotante, la ortogonalidad y las estimaciones son aproximadas. La [corrección angular](mathematical_scope.md) solo corresponde a A. No es una cota superior del radio de J completo.

La compresión puede omitir direcciones expansivas. Para `J = diag(0.2,0.2,2)` y R formado por las dos primeras coordenadas, el radio comprimido es 0,2 y el completo es 2. Tampoco representa las potencias del operador: para `J = [[0,2],[0.6,0]]` y `R=e₁`, `A=0`, pero `RᵀJ²R=1.2`. No se atribuyen estabilidad global ni retención a esta medida.

## Configuración y uso

`MACProjectionConfig` identifica modo disabled/diagnostic/penalty, rango, frecuencia, semilla, rejilla, umbral, peso, máximo de flujos y bytes estimados. Admite MAC online, dimensión hasta 64, un token por transición, rango 1–4 y 2–128 ángulos. La penalización exige peso positivo. Los otros modos exigen peso cero.

El control usa [`autograd.functional.jvp`](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.functional.jvp.html) y el backend SDPA Math. J incluye derivadas de la actualización asociativa y derivar la penalización puede necesitar terceras derivadas de su pérdida. El backend CPU predeterminado de la instalación comprobada no admite la derivada de su atención fusionada que necesita esta ruta. El consumidor del nuevo factorial debe usar Math también para su B, con identidad propia.

La selección se fija sobre el grupo lógico completo antes de dividirlo en lotes físicos:

```python
selection = control.select_flows(flow_ids, observed_steps, context_id=context_id)
result = control(
    mac, segment, state,
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

Las pruebas utilizan fixtures técnicos CPU. Comprueban el Jacobiano denso pequeño, gradientes del token actual, desacoplamiento de la historia, selección y partición, presupuestos, recuperación, RNG y errores de entrada. No acreditan CUDA, una sesión financiera real ni un efecto predictivo.
