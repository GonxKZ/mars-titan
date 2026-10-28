# Atención axial en el tiempo y entre regiones

`axial_time_region_v1` es la variante C2 del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#751](https://github.com/GonxKZ/mars-titan/issues/751)). Cada flujo guarda en su estado los bloques de fuente de sus últimas emisiones y, antes de la fusión, atiende primero a lo largo del tiempo de cada región y después entre las regiones de la emisión actual. Está implementada en `src/mars_titan/models/titans/axial_time_region.py` y declarada en [su configuración](../../configs/experiments/axial-time-region-v1.json), junto a un control con los mismos parámetros que solo atiende al tiempo. Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y alcance

En `mac_online` cada flujo emite un token por instante y el pasado solo llega a través de la memoria neuronal de MAC, que comprime todas las emisiones anteriores en sus pesos rápidos. Las cinco fuentes (precios, noticias, gráficos, fundamentales y macro) se mezclan en la fusión de cada emisión sin ver cómo evolucionó cada una. C2 pregunta si leer explícitamente las últimas emisiones de cada fuente y repartir después lo leído entre fuentes mejora la predicción frente a leer solo el tiempo y frente a mezclar fuentes sin tiempo, que es lo que hace C1.

La atención axial de [Ho et al. (2019)](https://arxiv.org/abs/1912.12180v1) atiende a lo largo de un solo eje del tensor cada vez. Su sección 3.1 (página 4) define la versión enmascarada, en la que la salida de una posición solo depende de las anteriores y de ella misma, y explica que dos capas sobre ejes distintos dan un campo receptivo que cubre todo el tensor. [MASTER (Li et al., 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/27767) aplica la misma idea a acciones: agrega primero dentro de cada acción a lo largo de su ventana y después entre acciones en cada instante (páginas 164 y 165), con una ventana de τ = 8 sesiones (página 166). En C2 el segundo eje son las regiones de un mismo activo, no los activos de una sesión. La atención entre activos es C5.

## Historial del flujo

Con C2, el estado de cada flujo lleva un `RegionHistory` con los bloques [W − 1, 5, H] de sus W − 1 emisiones anteriores y su presencia booleana. Cada emisión desplaza el historial una posición y añade sus propios bloques de entrada, con las fuentes ausentes a cero. Al empezar, ninguna posición está presente. Como el cursor del flujo exige instantes estrictamente crecientes, el historial solo contiene emisiones anteriores del mismo flujo y la variante no lee nada posterior a la emisión.

El historial avanza una posición por emisión del flujo y no por día de calendario. Si un activo no tiene fila en una sesión, su historial no cambia, así que el desfase ℓ cuenta emisiones. Se guardan los bloques de entrada y no las salidas de C2, de modo que el historial no depende de sus propios parámetros. Durante el ajuste, el historial conserva el grafo de los bloques que guarda hasta que el recorrido corta el del estado de MAC, en los mismos puntos.

El historial se exporta, se restaura y se reúne por filas con el resto del estado, y entra en el recuento de bytes. Con W = 8 y H = 64 en FP32 ocupa 8.995 bytes por flujo. Al decodificar un estado se rechaza un historial con posiciones presentes anteriores a la primera emisión del flujo.

## Ecuaciones ejecutadas

Sea $S_{w,r}$ la secuencia de W bloques de la región r, con la emisión actual en la posición W − 1, $m_{w,r}$ su presencia, $z_r$ el bloque actual, $\rho_r$ un embedding de región y $\ell_k$ un embedding de desfase k. Con cabezas de dimensión $d_h$,

$$
x_{w,r}=S_{w,r}+\rho_r+\ell_{W-1-w},\qquad n=\operatorname{LayerNorm}_1(x),
$$

$$
u_r=z_r+O_1\Bigl[\textstyle\sum_{w\in A_r}\operatorname{softmax}_{w}\bigl(q_r\cdot k_{w,r}/\sqrt{d_h}\bigr)\,v_{w,r}\Bigr]_h,
$$

con $q_r=W_q n_{W-1,r}$, $k_{w,r}=W_k n_{w,r}$, $v_{w,r}=W_v n_{w,r}$ y $A_r$ las posiciones presentes de la región más la actual. Es la fila W − 1 de la atención con máscara causal, la única cuya salida se usa. La segunda capa reparte entre regiones:

$$
y_r=\operatorname{LayerNorm}_2(u_r+\rho_r),\qquad
\text{salida}_r=u_r+O_2\Bigl[\textstyle\sum_{s\in B_r}\operatorname{softmax}_{s}\bigl(q'_r\cdot k'_s/\sqrt{d_h}\bigr)\,v'_s\Bigr]_h,
$$

con $q'$, $k'$ y $v'$ de $y$ y $B_r$ las regiones presentes de la emisión más la propia r. Una región ausente en la emisión conserva su bloque y no entra como clave en ninguna región. Las posiciones ausentes del pasado no entran en ninguna atención. Admitir siempre la consulta propia evita un softmax vacío y su salida se descarta si la región falta.

$O_1$ y $O_2$ empiezan en cero, así que al construirse la salida es $z$ y el predictor emite exactamente lo mismo que su referencia en cada paso. El gradiente de las dos proyecciones no es cero con esa inicialización, así que las dos aprenden. La semilla de C2 se deriva de la del predictor con SHA-256 y no consume el generador global. Los embeddings empiezan con desviación 0,02, pequeños frente a los bloques, para que la atención dependa al principio sobre todo del contenido.

## Control y contrastes

El control `time_only_attention_v1` tiene exactamente los mismos parámetros, con la misma semilla y la misma inicialización. Su segunda capa vuelve a leer el tiempo de cada región en lugar de mezclar regiones: la consulta sale de $\operatorname{LayerNorm}_2(u_r+\rho_r+\ell_0)$ y las claves y valores de la misma secuencia con la posición actual sustituida por $u_r+\rho_r+\ell_0$. Así ninguna región del control recibe información de otra antes de la fusión, y la diferencia con la variante es solo el eje de la segunda capa.

| Contraste | Variante | Control | Pregunta |
| --- | --- | --- | --- |
| Eje de región tras el tiempo | `axial_time_region` | `time_only_control` | ¿Aporta mezclar fuentes después de leer su pasado? |
| Dos lecturas del pasado | `time_only_control` | Referencia | ¿Aporta leer explícitamente las últimas emisiones? |

En la criba, C2 se compara también con C1 (`talking_heads`), que mezcla regiones sin eje temporal con el mismo presupuesto de ajuste. C2 se descarta si no mejora a su control de solo tiempo con la regla de la criba. Con H = 64, los dos tipos añaden 33.856 parámetros. La ventana W = 8 y las cuatro cabezas se fijaron al declarar la variante, antes de abrir datos: la ventana repite la τ = 8 de MASTER y las cabezas son las de C5 y de la atención de MAC.

## Requisitos de ejecución

C2 ocupa la modulación de la fusión y no se combina con el operador regional, C1, C6 ni PT6, y solo se declara sobre `mac_online`. Como necesita el historial de cada flujo, los tokens fila a fila de `fused_tokens` la rechazan y no se combina con las cabezas auxiliares de F5, que los usan. El consumidor congelado, sus cargadores del recorrido y Dust la rechazan hasta que registren el historial. La fuente del módulo entra en la identidad del entrenador solo cuando está activa.

## Correspondencia con el código y las pruebas

| Elemento | Código | Prueba |
| --- | --- | --- |
| Ecuaciones completas de los dos tipos | `AxialTimeRegion.forward` | `test_the_layer_matches_a_reference_written_position_by_position`, con una referencia flujo a flujo, región a región y cabeza a cabeza en FP64 |
| Máscara causal y posiciones ausentes | `allowed` | `test_absent_positions_of_the_past_are_never_read` |
| Región ausente intacta y sin mensajes | salida enmascarada | `test_an_absent_region_keeps_its_block_and_sends_nothing` |
| Solo la variante mezcla regiones | segunda capa | `test_only_the_axial_layer_mixes_regions` |
| Paridad al construirse en cada paso | $O_1=O_2=0$, semilla derivada | `test_each_kind_starts_as_the_reference_at_every_step` |
| Mismos parámetros en variante y control | construcción común | `test_the_variant_and_its_control_share_parameters_and_initialization` |
| Gradientes hasta los parámetros y el pasado presente | autograd | `test_gradients_reach_every_parameter_and_the_stored_history` |
| Historial de las W − 1 últimas emisiones | `RegionHistory`, `_prepare` | `test_the_history_keeps_the_last_window_minus_one_emissions` |
| Exportación, restauración, reunión y bytes | `export_state`, `financial_blocks` | `test_the_state_round_trips_through_export_restore_and_gather` |
| Corte del grafo y reunión en el motor | `_detached`, `FlowStates.gather` | `test_the_run_engine_detaches_and_gathers_the_history` |

## Desviaciones respecto a las fuentes

| Fuente | C2 | Motivo |
| --- | --- | --- |
| Ho et al. usan bloques Transformer completos con capa feed-forward | Dos atenciones residuales con normalización previa y proyecciones de salida a cero | Paridad exacta con la referencia al construirse |
| Ho et al. calculan todas las posiciones de cada eje | Solo la posición actual | Es la única salida que se usa, y la máscara causal la hace exacta |
| MASTER codifica la ventana con un Transformer bidireccional y posiciones sinusoidales | Lectura causal desde la emisión actual con desfases aprendidos | En línea solo existe la emisión actual. Las anteriores ya se emitieron |
| MASTER agrega entre acciones en cada instante | Agregación entre las regiones de un activo en la emisión actual | La mezcla entre activos es C5 |
| Un Transformer de dos capas leería en la segunda las salidas de la primera en cada posición | El control de solo tiempo relee los bloques guardados y sustituye solo la posición actual | El historial guarda entradas, no salidas, para no depender de los parámetros de C2 |
| Ventanas de días de calendario | Ventana de emisiones del flujo | El estado avanza con cada emisión del activo |

## Comprobación técnica

Pendiente de la verificación final.

## Coste

Pendiente de la medida.

## Pendiente

Faltan la medida CUDA, el registro en el consumidor congelado, su entrada en la criba y la comparación predictiva con las mismas filas, cortes y presupuesto. El aprendizaje sigue bloqueado y la variante no se activa por defecto.
