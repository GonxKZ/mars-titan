# Red de redes holográfica

`holographic_network_v1` (H1) es la pieza central de la segunda ola del estudio holográfico ([#782](https://github.com/GonxKZ/mars-titan/issues/782), [#783](https://github.com/GonxKZ/mars-titan/issues/783)). Cada bloque de la fusión se lee a la vez desde varias áreas con pesos distintos, sin copiar el dato, y las áreas se comunican por un puente. Está implementada en `src/mars_titan/models/titans/holographic_network.py` y declarada en [su configuración](../../configs/experiments/holographic-network-v1.json), junto a sus ablaciones H2 y H3. Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y alcance

En la referencia `mac_online`, cada modalidad llega a la fusión como un único vector. Los fundamentales mezclan conceptos de balance con ratios de empresa y el macro reúne doce familias con calendarios y dinámicas distintas, de la inflación a las cadenas de suministro. H1 pregunta si leer cada familia por separado, desde varias áreas a la vez y con comunicación entre ellas, mejora la predicción frente a la referencia con los mismos parámetros compartidos.

Este documento cubre el nivel de la fila. La issue declara un factorial de 1, 2 o 3 niveles por 5, 19 o 38 áreas. Los niveles 2 (grupos de activos de la sesión) y 3 (régimen en el tiempo) necesitan los grupos de [H4](asset-group-membership-v1.md), ya implementados como variante propia, y un estado de régimen que todavía no existe. Mientras no se integren, la configuración rechaza `levels` distinto de 1. Las [predicciones por área](#predicciones-por-área) son un campo opcional de H1 con sus propios controles.

## Bloques y áreas

Con 5 áreas, cada bloque es la representación de una modalidad. Con 19 o 38, los fundamentales y el macro se parten por familias. El bloque de una familia F es la salida del codificador de su modalidad cuando solo ve las columnas de esa familia,

$$
z_F=\operatorname{SiLU}\bigl(W_{:,F}\,x_F+b\bigr),
$$

con los mismos $W$ y $b$ de la referencia, así que los bloques no añaden parámetros. Las columnas de cada concepto son su valor, su máscara de observación y el logaritmo de su edad, la disposición `values_observed_log_age` de la edición. Precios, noticias y gráficos llegan como un único vector, porque sus codificadores no separan canales.

| Familia | Elementos en la edición v3.1 | Origen de la regla |
| --- | ---: | --- |
| Balance | 19 conceptos (8 en USD, 8 en CAD y 3 en CNY) | Catálogo contable conjunto |
| Ratios de empresa | 7 | Factores de empresa |
| Macro por categoría | 23, 22, 14, 5, 15, 25, 8, 10, 8, 5, 3 y 2 | Columna `category` de `data/catalogs/macro-indicators.csv` |

Las categorías macro son, en ese orden, inflación, empleo, actividad, vivienda, cuentas nacionales, tipos, liquidez, crédito, divisas, materias primas, condiciones financieras y cadenas de suministro. Las 19 áreas son tres de precios, noticias, gráficos y las 14 familias, sobre 17 bloques. Las tres de precios llevan nombres orientativos (tendencia, rango y volatilidad, volumen y liquidez), pero leen el mismo bloque y solo se distinguen por su embedding, su ajuste y su pertenencia aprendida. Nada fuerza esa especialización. Las 38 parten cada una de las 19 en dos con una regla fijada en el código y en la identidad: cada familia se divide en sus valores y en su observación (máscara y edad), y cada vector único se desdobla en dos vistas. Quedan 31 bloques. La configuración falla al declararse si la edición no trae las familias, por ejemplo con la política estricta sin máscaras o con un indicador sin categoría.

## Ecuaciones ejecutadas

**Pertenencia.** Cada bloque b se reparte entre las áreas a con

$$
p_{ab}=\operatorname{softmax}_a\bigl(\pi_{ab}+\Delta_{ab}\bigr),
$$

donde $\pi$ es el logaritmo del peso del catálogo. Con fuga $\ell=0{,}1$, las $k_b$ áreas de la familia del bloque reciben $(1-\ell)/k_b$ cada una y las demás $\ell/(A-k_b)$. $\Delta$ empieza en cero. Los pesos se calculan en FP64 y el tipo de trabajo los redondea una sola vez.

**Áreas.** Con $m_a=\sum_b p_{ab}\,z_b$, el estado inicial del área a es

$$
h_a=\operatorname{LayerNorm}\bigl(m_a+U_aV_a^{\top}m_a+e_a\bigr),
$$

así que el área lee cada bloque con $(I+U_aV_a^{\top})\,z_b$. $U_a$ empieza en cero, como en LoRA ([Hu et al., 2022](https://arxiv.org/abs/2106.09685)), y $e_a$ es un embedding del área sin identidad de activo.

**Grupos ocultos.** Cada ranura s mezcla los bloques presentes con $d_{sb}=\operatorname{softmax}_b\bigl(k_s\cdot\operatorname{LayerNorm}(z_b)/\sqrt{H}\bigr)$ y empieza en $\operatorname{LayerNorm}(\sum_b d_{sb}z_b+f_s)$. La lectura de vuelta usa $c_{bs}=\operatorname{softmax}_s$ de los mismos logits, como el despacho y la combinación de Soft MoE ([Puigcerver et al., 2024](https://arxiv.org/abs/2308.00951)).

**Puente.** Las áreas y las ranuras son los nodos de J = 2 ciclos con pasos locales de atención enmascarada y un resumen por ciclo, el calendario de `RegionalFusion`. Las áreas se unen en anillo, en anillo con cuerdas a un cuarto de su número o de forma completa, y cada ranura se une con todos los nodos. Como en `RegionalFusion`, un área cuyo bloque falta empieza en cero, no envía mensajes y no entra en el resumen, que promedia los nodos presentes.

**Lectura.** Con $\Delta_n=\text{local}_n-\text{inicial}_n$ y el contexto $c$ del resumen,

$$
\delta_b=\sum_a p_{ab}\,\Delta_a+\sum_s c_{bs}\,\Delta_s+c,
\qquad
\text{salida}_m=\text{bloque}_m+W_o\,\frac{1}{|m|}\sum_{b\in m}\delta_b
$$

en las modalidades presentes, donde $|m|$ es el número de bloques de la modalidad m. Una fila sin ningún bloque presente conserva sus bloques, aunque en la edición no ocurre porque los precios y los gráficos siempre están. Con la media, la escala de la corrección no depende de en cuántos bloques se parte una modalidad, así que 5, 19 y 38 áreas son comparables. $W_o$ empieza en cero y el predictor emite al construirse exactamente lo mismo que la referencia. La semilla de la red se deriva de la del predictor con SHA-256 y no consume el generador global. El estado de trabajo se reinicia en cada llamada, cada fila se calcula sola y el forward no actualiza parámetros.

## Ablaciones y control

- **H2 `catalog_membership_only_v1`** fija $\Delta=0$ y conserva el resto. Separa la pertenencia aprendida de la del catálogo.
- **H3 `hidden_groups_only_v1`** conserva solo las ranuras, con topología completa. En la declaración tiene 23 ranuras, los mismos nodos que H1 con 19 áreas y 4 ranuras, para que el puente haga el mismo trabajo. Separa las áreas del catálogo de unos grupos aprendidos sin estructura previa.
- **La referencia** es el mismo predictor sin la red. Sus parámetros compartidos son idénticos bit a bit en todos los modelos, y las ranuras, el puente y la lectura salen de la misma semilla en H1, H2 y H3.

Ningún modelo iguala exactamente los parámetros de otro. Con H = 64, la red añade:

| Modelo | Parámetros añadidos |
| --- | ---: |
| H1 con 5 áreas | 45.017 |
| H1 con 19 áreas | 53.379 |
| H1 con 38 áreas | 65.178 |
| H2 con 19 áreas | 53.056 |
| H3 con 23 ranuras | 44.416 |

Si H1 supera a la referencia pero no a H3, la ganancia no se puede atribuir a las áreas del catálogo.

## Predicciones por área

Con `area_predictions` declarado, cada área vota a partir de su estado final $s_{ra}$ en la fila r, después de los J ciclos del puente. Con $P_r$ las áreas presentes de la fila,

$$
\delta_{ra}=w_a\cdot\operatorname{LayerNorm}(s_{ra})+c_a,
\qquad
\omega_{ra}=\frac{e^{\ell_a}}{\sum_{a'\in P_r}e^{\ell_{a'}}}\quad(a\in P_r).
$$

La predicción del área es $\hat y_{ra}=\operatorname{sg}(\hat y^{\text{ref}}_r)+\delta_{ra}$, donde $\hat y^{\text{ref}}$ es la cabeza escalar sobre la representación con H1 y $\operatorname{sg}$ corta el gradiente. La salida del predictor pasa a ser

$$
\hat y_r=\hat y^{\text{ref}}_r+\sum_{a\in P_r}\omega_{ra}\,\delta_{ra}.
$$

Un área sin bloques presentes no vota y una fila sin áreas presentes no se desplaza. $w$, $c$ y $\ell$ empiezan en cero, así que al construirse los votos son nulos, la lectura reparte por igual y el predictor emite exactamente lo mismo que H1 sin votos.

El objetivo de cada tramo suma a la tarea el término

$$
\lambda\,\frac{1}{N}\sum_{r}\frac{1}{|P_r|}\sum_{a\in P_r}L\bigl(\hat y_{ra},y_r\bigr),
$$

con la misma pérdida puntual L de la receta (MAE, MSE o Huber) y $\lambda$ igual a `area_predictions`. Como la referencia entra sin grafo, la cabeza solo aprende de la tarea. Los estados de las áreas sí reciben el gradiente de su pérdida, que llega a los votos, al puente y a la pertenencia de H1 y a los codificadores de los bloques, pero no a la cabeza, a la fusión, a Titans-MAC ni a $W_o$. Es decir, el término cambia también las representaciones que comparte la tarea. La media por fila permite acumular el gradiente por bloques de flujos con la misma fracción de etiquetas que la tarea, y el contador `area_loss_sum` suma el término ponderado de cada tramo.

El recorte de gradientes es global e incluye los parámetros de los votos. Mientras los votos conservan su inicialización a cero, con $\lambda=0$ y sin recorte los gradientes compartidos coinciden bit a bit con los de H1. En esa condición, el recorte puede cambiar el factor positivo común a todo el paso. Una vez que cambian los votos, también cambia la tarea emitida y los gradientes compartidos pueden diferir aunque $\lambda=0$. La prueba del contador de gradientes conserva los pesos en cero y no demuestra paridad después de aprender. Las predicciones por área exigen la cabeza escalar sin objetivo de salida de #749, porque cada área aprende con la pérdida puntual de la receta, y H3 rechaza el campo porque no tiene áreas del catálogo. P1 y las predicciones por área también se rechazan juntos hasta definir sus controles y la interacción de sus pérdidas y estados. El campo con peso cero mantiene la lectura de votos y requiere el mismo rechazo.

| Contraste | Variante | Control | Qué separa |
| --- | --- | --- | --- |
| Predicciones por área | H1 con 19 áreas y $\lambda=0{,}5$ | H1 con 19 áreas | Los votos con su pérdida frente a H1 |
| Pérdida por área | H1 con 19 áreas y $\lambda=0{,}5$ | H1 con 19 áreas y $\lambda=0$ | La pérdida de cada área con la misma lectura y los mismos parámetros |

El peso 0,5 se fijó antes de ejecutar y no se ajusta. Con $\lambda=0$ las áreas siguen votando y la lectura sigue aprendiendo, pero solo a través de la salida. Los votos añaden 1.382 parámetros con H = 64 y 19 áreas: la normalización, $w$, $c$ y $\ell$.

El diseño combina dos antecedentes. En el [Thousand Brains Project](https://arxiv.org/abs/2412.18354) (Clay et al., 2024), cada módulo de aprendizaje puede reconocer objetos por sí solo y los módulos se envían por conexiones laterales sus hipótesis de objeto y pose hasta un consenso. En [Deeply-Supervised Nets](https://proceedings.mlr.press/v38/lee15a.html) (Lee et al., 2015), cada capa oculta tiene un clasificador con su propia pérdida de acompañamiento, que los autores interpretan como una regularización. La objeción principal la recogen [Du et al. (2018)](https://arxiv.org/abs/1812.02224): no siempre se sabe si una pérdida auxiliar ayuda a la tarea principal ni cuándo empieza a perjudicarla. Aquí no se adapta $\lambda$ durante el ajuste, y el contraste con $\lambda=0$ y con H1 mide si el término ayuda, perjudica o no cambia nada.

## Declaración

La [configuración](../../configs/experiments/holographic-network-v1.json) parte de la receta histórica con secuencia, cabeza escalar y MAE, con `default_enabled` y `final_test_access` en falso. Declara siete contrastes:

| Contraste | Variante | Control |
| --- | --- | --- |
| Red holográfica | H1 con 19 áreas | Referencia |
| Familias del catálogo | H1 con 19 áreas | H1 con 5 áreas |
| Áreas partidas | H1 con 38 áreas | H1 con 19 áreas |
| Pertenencia aprendida | H1 con 19 áreas | H2 con 19 áreas |
| Áreas frente a grupos ocultos | H1 con 19 áreas | H3 con 23 ranuras |
| Predicciones por área | H1 con 19 áreas y $\lambda=0{,}5$ | H1 con 19 áreas |
| Pérdida por área | H1 con 19 áreas y $\lambda=0{,}5$ | H1 con 19 áreas y $\lambda=0$ |

La topología es el anillo en todos los modelos. El mundo pequeño y la topología completa de las áreas existen como opciones, pero no se criban. El registro de estos modelos en la criba de la segunda ola queda para su protocolo.

## Correspondencia con el código y las pruebas

| Elemento | Código | Prueba |
| --- | --- | --- |
| Disposición de 5, 19 y 38 áreas | `area_layout` | `test_the_layouts_split_every_family_column_once`, `test_the_five_areas_are_the_modalities` |
| Bloques por familia sin parámetros nuevos | `HolographicNetwork._source_blocks` | `test_family_blocks_are_the_encoder_on_the_family_columns` |
| Pertenencia del catálogo con fuga | `log_catalog` | `test_the_catalog_membership_sums_to_one_with_its_leakage` |
| Ecuaciones completas | `HolographicNetwork.forward` | `test_the_module_matches_a_reference_written_area_by_area`, con una referencia fila a fila y nodo a nodo en FP64 |
| Nodos ausentes sin mensajes | máscara del puente | `test_an_absent_area_starts_at_zero_and_sends_no_messages` |
| Modalidad ausente intacta | salida enmascarada | `test_an_absent_modality_keeps_its_block_and_sends_nothing` |
| Paridad al construirse en FP32 y FP64 | $W_o=0$, semilla derivada | `test_each_kind_starts_as_the_reference_predictor`, `test_the_global_generator_is_not_consumed` |
| Gradientes hasta Δ, U, V, ranuras y puente | autograd | `test_gradients_reach_every_parameter_once_the_readout_moves` |
| Topologías | `area_edges`, `node_edges` | `test_the_topologies_of_the_areas` |
| Recuperación del estado | `get_extra_state` | `test_the_state_dict_round_trips_and_another_declaration_is_refused` |
| Votos, lectura y fila sin áreas | `HolographicNetwork.area_votes` | `test_the_votes_follow_their_equation_row_by_row`, `test_a_row_without_present_areas_keeps_its_output_and_a_finite_gradient` |
| Paridad al construirse y salida con los votos | `FinancialPredictor._voting_fuse` | `test_the_votes_start_at_zero_and_the_predictor_emits_as_h1`, `test_the_output_is_the_reference_plus_the_weighted_votes` |
| Alcance de la pérdida por área | referencia sin grafo | `test_the_area_loss_reaches_the_areas_and_the_encoders_but_not_the_output_path` |
| Pérdida en el entrenador, acumulación y recorte | `ChronologicalInference._area_loss`, `_area_term` | `tests/training/test_financial_run_area_predictions.py` |

## Desviaciones respecto a las fuentes

| Fuente | H1 | Motivo |
| --- | --- | --- |
| MaGNet ([Tan et al., 2025](https://arxiv.org/abs/2511.00085)) asigna los activos a hiperaristas de forma blanda, con pesos basados en la divergencia de Jensen-Shannon | La pertenencia parte del catálogo y aprende una corrección Δ desde cero | Las familias quedan fijadas antes de ver datos y la ablación H2 mide qué aporta Δ |
| Soft MoE normaliza X y Φ con L2 y una escala aprendida | LayerNorm de los bloques y escala $1/\sqrt{H}$ | El artículo indica que con dimensiones pequeñas esa normalización apenas influye, y aquí H = 64 |
| Soft MoE aplica un experto MLP a cada ranura | Las ranuras son nodos del puente compartido | El cálculo por ranura es el del puente y no un experto propio |
| LoRA suma $BA$ a los pesos de una capa | El ajuste $(I+U_aV_a^{\top})$ se compone tras el codificador compartido | El codificador de la referencia ya es el proyector $W_b$ de la issue y no se duplica |
| La issue pide reutilizar el operador regional | Mismo calendario con una implementación propia para N nodos | `RegionalFusion` está fijada a cinco nodos y su fuente forma parte de identidades ya registradas |
| En el Thousand Brains Project cada módulo puede aprender un modelo completo y los módulos votan hipótesis hasta un consenso | Cada área vota una corrección escalar sobre la referencia compartida, combinada con una media ponderada aprendida y sin iterar | La comunicación lateral ya la hace el puente, y una corrección parte de la escala del objetivo y conserva la paridad al construirse |
| Deeply-Supervised Nets añade una pérdida por capa oculta, con pesos $\alpha_m$ y un umbral $\gamma$ o un decaimiento que la anula durante el ajuste | Una pérdida por área del mismo nivel con un peso $\lambda$ fijo | Con $\lambda$ fijo la comparación con $\lambda=0$ es limpia, y anular el término por umbral añadiría un hiperparámetro elegido con los datos |

## Comprobación técnica

La comprobación del 28 de octubre se hizo en CPU, sin pasos de optimizador y con la GPU ocupada por otras cargas.

- Las 91 pruebas de H1, H2, H3 y de su medida de coste pasan con medida de cobertura. Con coverage 7.16.1 y radon 6.0.1, `holographic_network.py` cubre 327/327 sentencias y 114/114 ramas. El CRAP máximo es 33, en `area_layout`, seguido de `_checked` (28), el constructor (19) y la validación de la configuración (15). Con toda la cobertura, ese valor coincide con la complejidad ciclomática y señala dónde convendría simplificar, no un fallo.
- La regresión de los módulos que importan los caminos cambiados pasa: 4.630 casos, 4.613 en verde y 17 omitidos. 15 omisiones son la protección del aprendizaje y 2 los binarios nativos `mars-titan-ppo` y `libmars_titan_simulation`, que no están compilados.
- Diecinueve mutaciones dirigidas se detectan con los controles en verde antes y después, en una copia aislada del árbol final. Cubren ignorar la corrección Δ de la pertenencia, que un área sin bloques presentes envíe mensajes, sumar los bloques de una modalidad en la lectura en lugar de promediarlos, duplicar el contexto que vuelve a los bloques, promediar el resumen también sobre los nodos ausentes, despachar a las ranuras desde bloques ausentes, invertir la fuga del catálogo dentro de su familia, leer las columnas de una familia en un solo canal, conservar una lectura aleatoria, consumir el generador global, escribir en una modalidad ausente, conservar el bloque de una familia ausente, quitar las cuerdas del mundo pequeño, admitir dos niveles, aceptar valores de la configuración sin comprobar su tipo, admitir H1 fuera de `mac_online`, dejar el catálogo macro fuera de la identidad del entrenador y que el consumidor congelado o un padre con H1 lleguen a aceptarla.
- La primera pasada de mutaciones fue inconclusa porque la copia aislada no incluía los catálogos versionados de `data/` y las pruebas no llegaban a recogerse. Se corrigió el ejecutor y se repitió completa.
- Tras corregir la prosa de la medida de coste, sus 91 pruebas dirigidas vuelven a pasar sobre el árbol final.
- Tras integrar develop con la enmienda de la época común ([#798](https://github.com/GonxKZ/mars-titan/pull/798)) y los árbitros en línea ([#797](https://github.com/GonxKZ/mars-titan/pull/797)), pasan 302 casos de H1, C5, la identidad del entrenador, la época común, la selección y la recuperación, y la protección del aprendizaje omite otros 25.

El [recibo de verificación](../../reports/engineering/holographic-network-20261028/verification.json) conserva los recuentos y las huellas de los informes JUnit, de la cobertura, de las mutaciones y del enlace nativo.

La [verificación de la integración del 30 de octubre](../../reports/engineering/area-predictions-20261030/verification.json) registra 899 casos CPU aprobados sin omisiones, con 59 casos CUDA excluidos de esa selección. Después se aprobaron diez casos adicionales, seis del módulo en FP32 y cuatro de la configuración del estado plástico. La integración posterior con develop pasa otros 80 casos de análisis, motor y compatibilidad. Estas selecciones se solapan y no se suman como un único total. Los seis casos CUDA de la matriz nueva están preparados pero no se han ejecutado.

Se detectan 44 mutaciones dirigidas de los votos y su integración con P1, más una de la exigencia de configuración tipada del estado plástico. Dos mutaciones adicionales actúan sobre la prueba de recuperación, para exigir que cargue el checkpoint y use votos no nulos. Se informan aparte de las 45 mutaciones de producción. Todos los casos detectados tienen G1 verde, M1 rojo y G2 verde, sin pasos de optimizador. No es una puntuación de mutación global.

La cobertura acumulada de doce módulos suma 3.156/3.911 sentencias y 1.224/1.622 ramas. Se midieron 76 funciones de producción cambiadas. El CRAP máximo es 942,1246 en `FinancialConfig.__post_init__`, cuya complejidad ciclomática es 139. La separación por componentes sigue pendiente en #801. Tres líneas cambiadas sin ejecutar corresponden a contadores de memoria CUDA del benchmark. El recibo conserva las convenciones de coverage.py y radon y las huellas de las fuentes, que no cambiaron al incorporar develop.

El grafo estático identifica 394 módulos importadores. Las selecciones anteriores cubren 41 y la regresión de los otros 353 está encolada detrás de #747, por tandas de hasta ocho módulos y dos trabajadores. Faltan las comprobaciones y el coste CUDA de #816 y la integración de P1 (#818). No se ha ejecutado una comparación predictiva de estos votos ni se afirma una mejora. El aprendizaje y el test final permanecen bloqueados.

## Coste

La medida CPU del 28 de octubre llama solo a `HolographicNetwork` con las formas de la edición, en FP32 y con H = 64, dos hilos, tres calentamientos y veinte repeticiones en cada uno de los dos órdenes de los modelos. Los bloques y los codificadores de fundamentales y macro son los de la referencia, con noticias, fundamentales y macro ausentes en parte de las filas. El equipo tenía otras cargas, así que los percentiles son descriptivos. `forward` es la inferencia sin gradiente y `backward` añade la propagación de una suma de cuadrados hasta los parámetros de la red y de los codificadores, sin optimizador. El [recibo](../../reports/engineering/holographic-network-20261028/cost-cpu.json) conserva todas las muestras.

| Forward | 64 filas | 256 filas | 1.024 filas |
| --- | ---: | ---: | ---: |
| H1 con 5 áreas | 13,8 ms (16,4) | 28,5 ms (34,6) | 119,6 ms (144,0) |
| H1 con 19 áreas | 19,8 ms (24,5) | 81,4 ms (120,2) | 423,6 ms (470,1) |
| H1 con 38 áreas | 42,8 ms (58,6) | 132,5 ms (219,6) | 831,9 ms (1.132,6) |
| H2 con 19 áreas | 21,4 ms (24,8) | 78,6 ms (118,6) | 452,8 ms (529,8) |
| H3 con 23 ranuras | 18,0 ms (21,6) | 105,3 ms (119,9) | 459,4 ms (618,2) |

| Backward | 64 filas | 256 filas | 1.024 filas |
| --- | ---: | ---: | ---: |
| H1 con 5 áreas | 28,5 ms (31,3) | 78,9 ms (89,1) | 340,2 ms (399,0) |
| H1 con 19 áreas | 58,1 ms (66,8) | 174,0 ms (235,7) | 841,8 ms (1.069,0) |
| H1 con 38 áreas | 92,2 ms (139,8) | 338,7 ms (518,2) | 1.648,7 ms (2.296,5) |
| H2 con 19 áreas | 62,9 ms (69,5) | 185,2 ms (264,8) | 838,2 ms (947,4) |
| H3 con 23 ranuras | 57,3 ms (65,4) | 187,7 ms (274,1) | 763,5 ms (952,0) |

Cada celda da la mediana y, entre paréntesis, el percentil 95 de las cuarenta muestras. Con 1.024 filas, 38 áreas cuestan unas siete veces el forward de 5 áreas, y H2 y H3 cuestan lo mismo que H1 con 19 áreas, como se buscaba al igualar los nodos. Es mucho más que la atención entre activos C5, que tarda alrededor de un milisegundo con 256 filas, aunque las dos medidas no comparten proceso ni número de hilos. No se han medido la memoria, la VRAM ni la energía, y queda pendiente medir y perfilar en CUDA antes de decidir si hace falta optimizar el módulo.

```bash
CUDA_VISIBLE_DEVICES=-1 uv run python benchmarks/holographic_network_cost.py --run \
  --output "${MARS_HOLOGRAPHIC_COST:?}"
```

La medida del 30 de octubre añade los dos modelos de 19 áreas que votan, con λ = 0,5 y con λ = 0. En esta medida hacen el mismo cálculo, porque λ solo entra en la pérdida del recorrido, así que la diferencia entre ellos mide el ruido. Los votos añaden 1.382 parámetros a los 53.379 de H1 con 19 áreas. Las condiciones son las de la medida anterior, con el [recibo](../../reports/engineering/area-predictions-20261030/cost-cpu.json) del commit `918100ca3`.

| Pasada | Filas | H1 con 19 áreas | Votos con λ = 0,5 | Votos con λ = 0 |
| --- | ---: | ---: | ---: | ---: |
| Forward | 64 | 14,0 ms | 14,5 ms | 13,5 ms |
| Forward | 256 | 56,1 ms | 53,6 ms | 44,4 ms |
| Forward | 1.024 | 467,4 ms | 347,1 ms | 444,9 ms |
| Backward | 64 | 37,7 ms | 44,0 ms | 38,9 ms |
| Backward | 256 | 143,8 ms | 218,9 ms | 176,8 ms |
| Backward | 1.024 | 664,1 ms | 489,1 ms | 692,8 ms |

Son medianas. La media de carga del equipo estuvo entre 11 y 15 durante la medida. Las relaciones con H1 van de 0,74 a 1,52 según el tamaño y la pasada, cambian de signo entre tamaños y los dos modelos que calculan lo mismo difieren hasta un 30 %. Con ese ruido la medida no permite atribuir un coste a los votos. Por fila añaden una normalización de H componentes, un producto escalar por área presente y una softmax sobre las áreas presentes, poco frente a las rondas de la red. La medida se repetirá con la máquina libre.

## Pendiente

Faltan los niveles 2 y 3, H5 a H7, `membership_embeddings_v1`, la medida CPU de los votos con la máquina libre, la medida CUDA de [#816](https://github.com/GonxKZ/mars-titan/issues/816), el registro en el consumidor congelado y la comparación predictiva con las mismas filas, cortes y presupuesto. El acuerdo entre áreas de #787 leerá las predicciones por área cuando se integre en la criba. El aprendizaje sigue bloqueado y la variante no se activa por defecto.
