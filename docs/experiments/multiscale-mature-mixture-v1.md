# Mezcla de correctores con distintas semividas

`multiscale_mature_mixture_v1` (E1 de [#750](https://github.com/GonxKZ/mars-titan/issues/750)) combina correctores que olvidan a distintas velocidades. Los pesos de la combinación solo cambian al recibir etiquetas maduras. Es un factor opcional de `MatureCorrection`, desactivado en la campaña histórica, con [configuración propia](../../configs/experiments/multiscale-mature-mixture-v1.json) y una [receta con olvido discreto cero](../../configs/titans/mature-correction-calendar-scales.json).

E1 no es [PT5](../engineering/multiscale-memory.md). PT5 modifica los pesos rápidos de Titans con una cascada o un banco. E1 conserva el padre congelado y añade expertos de corrección financiera madura. Puede ampliar B6, las áreas B1, el puente B2 y las claves de roles B3. La comparación declarada conserva una referencia B6 con su retención original.

## Decaimiento y lectura

El código está en [multiscale_mixture.py](../../src/mars_titan/memory/multiscale_mixture.py). Cada experto empieza con los mismos coeficientes nulos y recibe las mismas claves y valores maduros. La configuración principal fija H = (7, 91, 365) días de 24 horas en UTC. Son intervalos elegidos para desarrollo, no constantes del artículo ni semanas de cinco velas exactas.

Si τ es el último corte en que se publicaron escrituras maduras, la lectura en t calcula:

```text
d_e(t) = 2 ** (−(t − τ) / (H_e × 86.400.000.000))
memoria_e_temporal = decay_coefficients(memoria_e, d_e(t))
q_e(t) = memoria_e_temporal.read(clave)
corrección(t) = Σ_e softmax(logpesos)_e × q_e(t)
```

Los cortes se expresan en microsegundos. La operación escala todos los coeficientes del backend, incluidos los mapas T de B2 cuando están presentes. No cambia sus contadores ni sus cursores. No toca los parámetros compartidos ni los pesos rápidos del padre.

La lectura prepara copias temporales y deja intacto el estado confirmado. Otra consulta intermedia no cambia el reloj ni introduce redondeos acumulados en una predicción posterior. Solo una escritura no vacía publica otro reloj. Las consultas anteriores a ese reloj se rechazan. Un decaimiento muy largo puede llegar a cero en FP64, caso comprobado expresamente.

El backend debe usar delta o proximal y λ = 0. El olvido de calendario se aplica antes de la escritura y la regla original actualiza esos coeficientes con la misma η. Así se evita sumar otro olvido por fila o por cohorte. La semivida describe la parte homogénea. Las escrituras y sus soluciones proximales también cambian el estado, por lo que H no garantiza retención de información ni una semivida de la predicción. La memoria kalman y el control kernel no se convierten en expertos de E1.

## Pesos de combinación y predicciones guardadas

La sesión emite primero con el estado anterior. Guarda la predicción del núcleo, la clave, la ruta si existe y el vector de componentes q calculado en esa decisión. Los componentes son predicciones auxiliares de la mezcla, no operaciones financieras independientes. Al madurar una etiqueta `y`, se calcula `v = y − núcleo guardado`. El error de cada componente se obtiene de aquel q guardado, nunca de una lectura nueva de la memoria.

La regla principal utiliza Hedge en forma logarítmica:

```text
ℓ_e = media_canónica_de_la_publicación(min(|q_e_guardado − v| / L, 1))
z_e_nuevo = z_e − η_mezcla × ℓ_e
z_nuevo = z_nuevo − max(z_nuevo)
```

L = 0,02 y η_mezcla = 1 están fijados en la configuración inicial. El recorte afecta solo al ajuste de la mezcla. El MAE y las predicciones no se recortan. Se rechazan diferencias no finitas antes de aplicar el recorte. Se conserva un vector de logpesos finitos con máximo cero y la lectura calcula sus pesos con softmax. Un peso puede redondearse a cero, no se garantiza que todos los expertos sigan teniendo una influencia apreciable.

La pérdida interna promedia los resultados de la cohorte publicada en orden canónico. No sustituye el MAE por sesión, la ponderación entre mercados ni el criterio de la criba. Los retrasos o distintas políticas de entrega pueden cambiar las cohortes. Su calendario debe ser común a las configuraciones comparadas.

El antecedente del ajuste de pesos es Freund y Schapire, figura 1 y sección 2 de *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*. El manuscrito consultado está fechado el 19 de diciembre de 1996 y señala la publicación de 1997. Su formulación usa pérdidas acotadas recibidas después de elegir la distribución de cada ronda. [PDF de los autores](https://www.schapire.net/papers/FreundSc95.pdf), [DOI](https://doi.org/10.1006/jcss.1997.1504).

La adaptación financiera es propia. No se atribuye directamente la cota del artículo a estas predicciones retrasadas. Además, la pérdida recortada no es convexa. Con expertos −100 y +100, etiqueta +100, pesos iguales y recorte en 1, la pérdida recortada de la media es 1 y la media de pérdidas recortadas es 0,5. La prueba conserva este contraejemplo. La actualización puede empeorar frente a una mezcla uniforme o tardar en abandonar un experto que funcionó bien en otro periodo.

## Persistencia, presupuesto y controles

El estado conserva los expertos, los logpesos, el reloj, el número de publicaciones, las escrituras y sus cursores. Las huellas incluyen los coeficientes y los metadatos. La recuperación rechaza otra configuración, campos ausentes o huellas incompatibles. `FinancialSession` conserva además los componentes pendientes y valida su tipo FP64, forma y alineamiento con las predicciones. Las censuras y liquidaciones retiran las mismas filas de todos esos campos.

Cada componente pendiente añade 8 bytes por decisión. Con tres expertos son 24 bytes, además de los campos existentes y de la ruta de B3 cuando se use. Las tablas de roles se comparten entre expertos. Los coeficientes e índices por área pertenecen a cada copia. La sesión comprueba que los tensores y la cola máxima declarada caben en el artefacto de 64 MiB, con la reserva de metadatos de su productor, antes de aceptar el modelo.

La configuración contiene once variantes y la referencia con retención histórica, con dos casos comunes de η por receta. Los controles principales son los mismos tres expertos con pesos uniformes, una única escala de 91 días y tres copias de esa escala. Las tres copias igualan coeficientes y componentes pendientes, no necesariamente tiempo. Hay parejas adaptativa/uniforme al combinar E1 con B1, B2 y B3. Ningún identificador histórico se sustituye por E1.

Los módulos existentes exponen `decay_coefficients` y conservan su comportamiento previo cuando no se usa E1. La ventana repite cada tramo desde su inicio al recuperarse, según el contrato de B6. La sesión financiera tiene recuperación por generación, incluido el vector de componentes de cada decisión pendiente. No se reconstruye ese vector con información posterior.

El [esquema](../diagrams/variants/e1-multiscale-mature-mixture.png) separa la lectura, la cola pendiente y la actualización posterior. No hay comparación predictiva ejecutada de E1 ni una diferencia de MAE que informar.

## Comprobaciones y coste del 26 de octubre de 2026

La [verificación](../../reports/engineering/multiscale-mature-mixture-20261026/verification.json) registra 264 pruebas CPU distintas y 17 casos adicionales de la guarda de aprendizaje. Incluye una recurrencia independiente, gradcheck, recuperación nativa, alineamiento del registro pendiente, orden de activos, etiquetas futuras, valores no finitos y límites del artefacto. Los quince casos sin E1 coinciden bit a bit con la referencia 3e89cc5df. Quedan cinco casos de interoperabilidad CUDA pendientes en el conjunto B1/B2/B3/E1. No se ha ejecutado la suite completa del proyecto.

El módulo nuevo cubre 153/153 sentencias y 50/50 ramas con coverage.py 7.16.1. El CRAP máximo por función es 30, calculado con Radon 6.0.1 y cobertura de sentencias. La [integración](../../reports/engineering/multiscale-mature-mixture-20261026/quality-integration.json) informa por separado 1.484/2.019 sentencias y 480/762 ramas. Diez [mutaciones dirigidas](../../reports/engineering/multiscale-mature-mixture-20261026/mutation-results.json) fueron detectadas con referencia verde antes y después, sin intentos de pasos de optimizador. Estas cifras no demuestran ausencia de errores ni una cobertura global completa.

Las dos pasadas del [benchmark](../../benchmarks/multiscale_mixture_cost.py) usan 64, 256 y 1.024 filas, cinco ciclos de calentamiento y cuarenta medidas por caso, con dos hilos CPU y FP64 en el corrector. Incluyen el codec, la lectura y una escritura madura desde un estado fijo. Las rutas de régimen ya vienen dadas. Se excluyen padre, almacenamiento, cálculo del régimen y colas. La [primera pasada](../../reports/engineering/multiscale-mature-mixture-20261026/benchmark-1.json) y la [segunda](../../reports/engineering/multiscale-mature-mixture-20261026/benchmark-2.json) conservan todas las muestras, percentiles y huellas.

| Configuración, 256 filas | Mediana 1 (ms) | Mediana 2 (ms) | Bytes numéricos residentes |
| --- | ---: | ---: | ---: |
| B6 con retención histórica | 0,703 | 0,623 | 512 |
| Una memoria sin olvido ni mezcla | 0,697 | 0,645 | 512 |
| E1 adaptativa, tres escalas | 1,986 | 2,321 | 1.560 |
| E1 uniforme, tres escalas | 2,063 | 4,552 | 1.560 |
| B1+B2+B3+E1 adaptativa | 49,617 | 50,206 | 1.593.880 |

La dispersión entre pasadas impide interpretar pequeñas diferencias como una ventaja. E1 añade coste frente a B6 y todavía no hay evidencia de que lo compense. Los procesos completos tardaron 75,94 y 90,74 segundos, con picos RSS de 749.980 y 749.776 KiB. Esos picos incluyen importaciones y todas las configuraciones, no son memoria atribuible a una sola variante. Los bytes de la tabla excluyen colas, objetos Python y temporales. No se ha medido energía ni VRAM. No se ha sustituido esta referencia por un kernel propio.
