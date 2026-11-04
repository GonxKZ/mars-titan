# Parada residual del puente episódico

Esta variante experimental elige por predicción un prefijo de la recurrencia regional. Conserva el episodio canónico, sus IDs, las consultas regionales, los parámetros del puente y la actualización MAC. Cada región sigue atendiendo y transformando la misma evidencia con sus propios pesos. La parada no añade otra memoria ni demuestra especialización de las regiones.

La opción se expresa mediante `BridgeRecurrenceConfig.stopping`. Su valor por defecto es `None`. Al activarla se obtiene una identidad distinta, con el contrato `shared_evidence_bridge_adaptive_v1`. Una intervención fija conserva su envoltura y declara este contrato como `base_kind`. La jerarquía L permanece separada y su composición con la parada se rechaza.

## Transición y decisión

Sean $s_i^j$ los estados de la región receptora $i$, $G$ el grafo normalizado, $W_i$ sus pesos reutilizados y $\gamma$ la ganancia del puente. Para regiones presentes se ejecuta

\[
s_i^{j+1}=(1-\gamma)s_i^j+
\gamma\tanh\left(\left(\sum_k G_{ik}s_k^j\right)W_i\right).
\]

Las regiones ausentes y sus aristas se anulan. El grafo, los pesos y la evidencia permanecen fijos durante una lectura. J cuenta estas vueltas. K cuenta las lecturas episódicas y no las actualizaciones de memoria neuronal de Titans.

La decisión utiliza los estados representables anteriores y actuales, separados de autograd y convertidos a FP64. Si P aplica la máscara de regiones presentes, se calcula

\[
\delta_j=\|Ps^j-Ps^{j-1}\|_\infty,
\qquad
q_j=\max(\|Ps^{j-1}\|_\infty,\|Ps^j\|_\infty).
\]

Se acepta el primer prefijo admisible cuyo cambio satisface \(\delta_j\le a+r q_j\). J máximo pertenece a {1, 2, 4}. No hay salida en J3. Si ningún prefijo cumple el criterio, se devuelve J máximo con `criterion_met=False`.

Con tolerancia relativa cero se compara directamente el cambio absoluto. Así no se pierde una diferencia pequeña al dividirla por otra coordenada grande. La implementación aplica la máscara antes de restar, rechaza entradas e intermedios ejecutados no finitos y comprueba el desbordamiento de la diferencia FP64. Para tolerancia relativa positiva evita formar un producto de umbral desbordado. Estos cálculos ordinarios en coma flotante no son certificados numéricos.

`adaptive_reference_full` calcula J máximo y selecciona el prefijo. `adaptive_compact` continúa únicamente con las filas pendientes. Ambos devuelven valores con autograd y tres tensores detached por fila: `selected_steps`, `evaluated_steps` y `criterion_met`. La ruta densa comprueba también intermedios posteriores a una salida seleccionada. Puede rechazar un desbordamiento que la compacta no llega a ejecutar. Su comparación necesita un desenrollado completo finito y margen respecto al umbral.

Un snapshot vacío devuelve cero vueltas y criterio falso. Un pool poblado con todas las regiones retiradas conserva sus episodios, produce valores cero y elige J1 con criterio verdadero. El control `no_bank` no recibe un snapshot implícito.

## Gradientes, límites y antecedentes

El backward deriva el prefijo seleccionado. La decisión discreta no tiene un gradiente global garantizado y en una frontera puede diferir de la derivada de la función completa, incluso cuando esta existe. Un residual pequeño tampoco prueba estabilidad. Con una región escalar, \(W=2\), \(\gamma=1/4\) y estado inicial cero, todas las vueltas permanecen en cero, pero la derivada de cada transición vale 1,25. La prueba correspondiente conserva ese contraejemplo.

El umbral depende de las coordenadas de la representación. Cambiar su escala puede cambiar las salidas tempranas. Debe fijarse en desarrollo y contrastarse con los mismos embeddings, pesos, datos y presupuestos. No se deduce de él una cota de error financiero, retención de información o convergencia.

ACT aprende cuántos pasos computar y PonderNet aprende una política de parada mediante su propio objetivo. Son antecedentes del cálculo adaptativo. Este criterio determinista del residual no reproduce ninguno de los dos procedimientos. Las revisiones consultadas son [ACT v6, 21 de febrero de 2017](https://arxiv.org/abs/1603.08983v6) y [PonderNet v2, 2 de septiembre de 2021](https://arxiv.org/abs/2107.05407v2), verificadas el 23 de octubre de 2026.

[SEAL v2, 18 de septiembre de 2025](https://arxiv.org/abs/2506.10943v2), genera datos y directivas para adaptar pesos y aprende propuestas a partir del rendimiento posterior. La parada del puente no genera datos, edita código ni ajusta pesos. El [contraste de configuraciones registradas](../engineering/registered-trajectories.md) mantiene otro alcance y no equivale a implementar SEAL.

## Integración y contabilidad

`bridge_recurrence.py` contiene la transición y los dos backends. `evidence_bridge.py` devuelve `AdaptiveBridgeRead` y `EpisodicReadout` mantiene las lecturas existentes. Los consumidores conservan las emisiones y sus snapshots. Los recibos registran las decisiones por lectura y el archivo nativo permite recuperar el estado confirmado.

Se distinguen tres cuentas. `bridge_initial_views` cuenta las vistas iniciales, `bridge_selected_node_updates` cuenta R por la suma de profundidades elegidas y `bridge_node_updates` cuenta R por la suma de vueltas calculadas. Ninguna de ellas es una medida de FLOPs. Una referencia densa que selecciona J1 y calcula J4 conserva el coste de J4.

La admisión reserva J máximo antes de MAC para toda la colección y su emisión. La cota conservadora sigue siendo 38 lecturas por fila. Compartir los prefijos de seis rutas permite ejecutar nueve lecturas por contexto, pero no devuelve cuota. La colección activa archiva esas lecturas reales y reconstruye los agregados al restaurarse. No suma de nuevo los prefijos comunes de las seis salidas. El paquete conserva su límite de 48 MiB y una cota previa de los metadatos nuevos.

El validador de recibos comprueba tipos, prefijos, backend, esquema y sumas. No autentica por sí solo el residual calculado ni un paquete cuyos descriptor y lecturas se hayan sustituido de forma coherente. La procedencia depende también del productor, sus fuentes y el registro nativo.

Al desactivar la opción se conservan la configuración estructural, los parámetros y la lectura de checkpoints del módulo. Los sellos de implementación registran el código nuevo. Una sesión histórica ligada a fuentes concretas requiere esas fuentes y no debe recuperarse falsificando su identidad.

## Declaración en la criba

La [declaración](../../configs/experiments/adaptive-bridge-depth-v1.json) registra D2 en la criba de [#746](https://github.com/GonxKZ/mars-titan/issues/746) con tres lectores sobre el mismo padre, M1, tres vistas en anillo, ganancia 0,25 y la misma semilla. Los tres tienen los mismos parámetros.

| Configuración | Vueltas calculadas | Salida emitida |
| --- | ---: | --- |
| `adaptive_j4` | 4 | El primer prefijo J1, J2 o J4 que cumple la parada |
| `fixed_j4` | 4 | J4 |
| `fixed_j1` | 1 | J1 |

El contraste principal enfrenta `adaptive_j4` con `fixed_j4`. Con `adaptive_reference_full`, la lectura adaptativa calcula siempre J4, así que los dos lectores reservan y ejecutan el mismo trabajo y solo difieren en el prefijo que emiten y derivan. El segundo contraste, `fixed_j4` frente a `fixed_j1`, mide la profundidad y solo se publica. `fixed_j4` es la misma definición que `flat_k1_j4_final` de D1, así que el motor la ejecuta una vez para las dos entradas.

La tolerancia es relativa, $r = 0{,}05$ con $a = 0$, y quedó fijada al declarar la entrada, antes de abrir ningún dato. Con $a = 0$, el criterio compara el cambio de una vuelta con el tamaño del propio estado y no necesita un umbral en las unidades de la representación. La trayectoria sí depende de esa escala a través de tanh, así que el valor no se traslada a otras representaciones. Con 0,05, la lectura se detiene cuando una vuelta no mueve ninguna coordenada presente más de un 5 % de la mayor de ellas. Es un valor previo, sin ajuste ni justificación empírica. Si el criterio casi nunca se cumple, la variante coincide con `fixed_j4`, y si se cumple siempre en la primera vuelta emite J1 con el cálculo de J4. Los recibos guardan la profundidad elegida en cada lectura, así que el análisis puede publicar su distribución junto al contraste.

D2 se descarta si la parada no mejora el IC por sesión de `fixed_j4` con la regla de decisión de la criba.

## Comprobaciones y evaluación pendiente

Las pruebas técnicas cubren ecuaciones, gradientes de la rama elegida, máscaras, tolerancias extremas, desbordamientos, RNG, checkpoints, permutación, bloques, conservación de IDs, recuperación y coste de prefijos compartidos. Utilizan fixtures identificados, sin aprendizaje financiero. La mutación dirigida, la cobertura y CRAP son comprobaciones parciales, sin acreditar ausencia de errores.

La variante permanece desactivada por defecto. Su integración técnica no demuestra una mejora de MAE ni una reducción de latencia. El contraste de coste y la evaluación predictiva necesitan sus propios resultados. La campaña completa y el test final de 2024 siguen cerrados.
