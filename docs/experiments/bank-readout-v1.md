# Lectura contextual de bancos históricos

`MARS-TITAN-BANK-READ-v1` añade una lectura opcional de los tres bancos neuronales de PT5. Conserva el núcleo de Titans-MAC y el predictor anterior. La opción `memory_bank_readout=None` mantiene el cálculo y el formato anteriores. No se ha entrenado esta variante ni hay una mejora predictiva medida.

Los bancos contienen pesos rápidos de redes asociativas por flujo. Sus asociaciones proceden de claves y valores proyectados de las observaciones admitidas. No son episodios con etiquetas financieras ni las cinco fuentes del grafo regional. `RegionalFusion` sigue siendo memoria de trabajo que se reinicia por predicción. Ambos mecanismos pueden combinarse, con controles separados.

## Lectura y escritura ejecutadas

Sean W₁, W₂ y W₃ las tuplas de matrices de los bancos y M(q,Wᵣ) la red de memoria completa del banco r. Con dos capas incluye GELU entre ellas. Si se declara la versión residual del artículo, cada red aplica además q + LN(MLP(q)). Primero se evalúa cada red y después se mezclan sus salidas:

$$
p_r(q)=\operatorname{softmax}_{r\in A}\left(\frac{\nu(q)^\mathsf{T}\nu(a_r)}{T}\right),\qquad
R(q,W)=\sum_{r\in A}p_r(q)M(q,W_r).
$$

A contiene los bancos disponibles para leer. Las direcciones aᵣ son parámetros compartidos entrenables y T es un hiperparámetro fijo durante el tramo. ν escala el vector por su máximo absoluto, acotado por debajo por uno, y después lo divide por su norma con un suelo de 10⁻¹². Esta forma evita desbordamientos innecesarios en la suma de cuadrados. No transforma una consulta fuera de distribución en información fiable.

La ruta uniforme calcula los logits y los anula. Sus direcciones permanecen almacenadas, pero reciben gradiente cero. La ruta constante usa una dirección de consulta fija y permite aprender preferencias globales entre bancos. La ruta por consulta permite que los pesos cambien con la entrada. Los tres controles tienen los mismos parámetros y estados almacenados, con distinta capacidad efectiva.

Si A está vacío, la lectura es cero. Se siguen comprobando todos los resultados para detectar NaN o infinito. La máscara cambia la lectura, no borra matrices ni detiene las escrituras. Es un diagnóstico de disponibilidad. Borrar un banco, reinicializarlo o interrumpir sus actualizaciones son otras intervenciones y no se consideran equivalentes.

En una red no lineal, mezclar sus salidas no equivale a aplicar una red de pesos medios. Por eso `readable_weights` rechaza proporcionar una supuesta red equivalente cuando este lector está activo. La escritura utiliza la misma función R:

$$
\ell(\Delta)=\left\|R\left(k,\{W_r+\Delta\}_{r=1}^3\right)-v\right\|_2^2,\qquad
G=\left.\nabla_\Delta\ell(\Delta)\right|_{\Delta=0}
=\sum_r\nabla_{W_r}\ell.
$$

La suma se entiende por capa, manteniendo fija la observación en esa derivada interna. Δ es una coordenada de traslación compartida. No son tres pasos de gradiente independientes ni una proyección ortogonal del gradiente dividida entre tres. Con lectura compartida se promedian las pérdidas de las observaciones de la cohorte, como en el mecanismo anterior.

$$
S_t=\eta_t S_{t-1}-\theta_t G_t,\qquad
W_{r,t}=d_{r,t}W_{r,t-1}+S_t.
$$

Se reutiliza el momentum existente. En `ema_bank`, d₁ = 1 − α depende de la entrada y los otros dos olvidos son los de PT5. Los tres bancos reciben la misma escritura en cada observación. Esto no son tres frecuencias de actualización diferentes. Tampoco se garantiza que el primer banco olvide más deprisa que los otros. Por eso se identifican como `level_1`, `level_2` y `level_3`.

`replica_bank` conserva la misma cantidad de matrices y aplica el mismo α y la misma escritura a todas. Si parten de copias idénticas, las matrices siguen siendo idénticas. Es un control de redundancia simple. El acoplamiento no se utiliza en esa dinámica y su identidad lo declara. Una recuperación resistente a deshabilitar una copia puede explicarse por esa redundancia, sin atribuirla a especialización neuronal.

## Estado, causalidad y límites matemáticos

Las direcciones pertenecen a los parámetros compartidos del ajuste externo. Las matrices, niveles, momentum y cursores son estado rápido por flujo. Los tokens persistentes de Titans conservan su función anterior. El banco episódico y la cola financiera siguen separados. Ninguna etiqueta futura interviene en G. El error financiero maduro continúa calculándose con la predicción emitida y su disponibilidad real.

Las consultas de una fila no leen las de otros activos. El coste de lectura es O(R B T L D²), con R = 3 y profundidad L de una o dos capas. No se forma una matriz entre todos los activos o episodios. El estado persistente usa las mismas matrices de PT5. Se añaden 3D parámetros y tres booleanos compartidos. Las activaciones y gradientes de las tres redes aumentan el coste transitorio y necesitan medidas aparte.

El caso lineal escalar permite comprobar el Jacobiano de la actualización completa. Para clave fija k, pesos de lectura a, olvidos d y c = 2θk², el estado ampliado es (w₁,w₂,w₃,S):

$$
A_{\mathrm{local}}=
\begin{bmatrix}
d_1-ca_1&-ca_2&-ca_3&\eta\\
-ca_1&d_2-ca_2&-ca_3&\eta\\
-ca_1&-ca_2&d_3-ca_3&\eta\\
-ca_1&-ca_2&-ca_3&\eta
\end{bmatrix}.
$$

El término forzado 2θvk se suma a las cuatro componentes. Las pruebas comparan esta matriz con autograd, incluida una entrada sin normalización para la que su radio espectral supera uno aunque cada olvido individual sea menor que uno. Controlar solo esos olvidos sería insuficiente.

Este cálculo mantiene fija la entrada y usa memoria lineal. No incluye la dependencia de la atención o de una consulta respecto al estado en el predictor completo. No demuestra estabilidad global, retención semántica ni generalización. No es una formalización reproducida en Lean. La combinación con C local, PT1, plasticidad o cascada se rechaza hasta definir y verificar sus propios contrastes.

## Controles e integración

La [declaración de brazos](../../configs/experiments/bank-readout-v1.json) incluye la lectura anterior de matrices medias, mezclas de funciones uniforme, constante y por consulta, réplicas y dos combinaciones con el mismo grafo regional. Reutiliza la [receta cronológica](../../configs/experiments/bank-readout-recipe-v1.json). No duplica la canalización de entrenamiento. K = 1 sigue siendo el contraste principal y no cuenta bancos ni escrituras internas.

El consumidor congelado registra fuente, configuración y parámetros del módulo. Se valida el checkpoint del lector antes de copiar parámetros de sus padres. La recuperación conserva todas las matrices históricas y reproduce la siguiente lectura y actualización. Los contadores de la sesión distinguen evaluaciones de funciones, consultas de rutas y matrices actualizadas. No son FLOPs, lanzamientos CUDA ni pasos de optimizador.

Los contrastes científicos necesitan las mismas filas históricas con máscaras, cortes, estados iniciales y presupuestos pertinentes. La media de matrices almacena menos parámetros de lectura. La referencia original de MARS-TITAN y CM-v1 conservan sus identidades. La variante nueva no redefine B ni se activa por defecto. El bloqueo del aprendizaje y del test final continúa vigente.

## Antecedentes y criterios de descarte

[Roy y colaboradores (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8980018/) observaron participación y reactivación conjunta de engramas de una memoria contextual en varias regiones de ratones. No establecen copias completas en todas ellas ni inmunidad ante cualquier lesión. Aquí esa observación motiva un contraste de recuperación parcial, sin asignar funciones anatómicas a matrices.

[Benna y Fusi (2016)](https://doi.org/10.1038/nn.4401) estudian procesos sinápticos acoplados con distintas escalas. PT5 conserva su contraste de cascada. Este lector emplea el banco EMA como alternativa independiente y no reproduce esos procesos biológicos.

[Nested Learning, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/4309616aaed8e848009bc4a7ef73b493-Paper-Conference.pdf), apéndice C, ecuación (42), ya contempla agregar memorias independientes, incluida una suma aprendida. No se reivindica novedad por combinar bancos. La escritura mediante traslación común es una decisión explícita de esta adaptación y difiere de optimizar cada memoria por separado.

[Song y colaboradores, versión 1 de septiembre de 2026](https://arxiv.org/html/2609.37791v1), separan modulación de memoria de trabajo y recuperación episódica contextual. Su tabla S1 distingue alineación cerebral y predicción. La modulación de salida tiene mayor correlación de prueba que la modulación recurrente de bajo rango. El parecido con respuestas cerebrales no garantiza el mejor predictor. Nuestro módulo no implementa su inferencia bayesiana de contexto ni convierte las etiquetas de una historia en regímenes financieros.

La hipótesis de utilidad se descarta si la ruta por consulta no supera a las mezclas uniforme y constante bajo el protocolo temporal, o si el coste no compensa su efecto observado. La hipótesis de cooperación se contrasta frente a réplicas con iguales matrices. Un beneficio en un ejemplo sintético, una máscara o una distancia de representaciones no acredita mejora de MAE. Las comprobaciones técnicas y las medidas se documentan por separado de cualquier experimento predictivo.
