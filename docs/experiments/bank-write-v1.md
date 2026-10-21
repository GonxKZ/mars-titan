# Escritura por banco neuronal

`MARS-TITAN-BANK-WRITE-v1` contrasta cómo se asigna el gradiente asociativo a los tres bancos de [BANK-READ-v1](bank-readout-v1.md). Conserva Titans-MAC, sus proyecciones y su pérdida interna. Añade dos reglas opcionales con el mismo almacenamiento de pesos y momentum. No se ha entrenado la variante ni se ha medido su utilidad financiera.

Los bancos son redes de memoria con pesos rápidos por flujo. No son episodios financieros, regiones anatómicas ni modalidades. El grafo regional conserva cinco representaciones de entrada y reinicia su estado de trabajo por predicción. El banco episódico sigue separado, con sus identidades y etiquetas maduras. K cuenta refinamientos episódicos y no bancos, capas de memoria o pasos internos de escritura.

## Regla ejecutada y control

La lectura conserva la mezcla de funciones completas, incluidas GELU y la opción residual del núcleo:

$$
R(k,W)=\sum_{r=1}^{3}a_r(k)M(k,W_r),\qquad
L(W;k,v)=\|R(k,W)-v\|_2^2,\qquad
G_r=\partial L/\partial W_r.
$$

La derivada interna mantiene fijas la observación, sus claves y valores y la ruta que estos producen. Si se pide diferenciación externa, se conserva el grafo de sus productores para derivar a través de esa escritura. No se usan etiquetas financieras en esta pérdida asociativa.

`per_bank` conserva cada parcial. `mean_partial` las proyecta sobre el subespacio de gradientes iguales entre bancos:

$$
\bar G=\tfrac13\sum_rG_r,\qquad
\widetilde G_r=G_r\quad\text{o}\quad\widetilde G_r=\bar G.
$$

Esta media es la proyección ortogonal para el producto de Frobenius entre las tres tuplas de matrices. Con P = 11ᵀ/3, Pᵀ = P y P² = P. Por tanto, la descomposición G = PG + (I − P)G separa las componentes común y diferencial y satisface la identidad de Pitágoras. Es una propiedad algebraica, no una certificación de la aritmética de coma flotante ni una garantía predictiva.

En cada capa se aplica:

$$
S_{r,t}=\eta_tS_{r,t-1}-\theta_t\widetilde G_{r,t},\qquad
W_{r,t}=d_{r,t}W_{r,t-1}+S_{r,t}.
$$

Las puertas α, η y θ siguen dependiendo de la entrada como en el núcleo. En el banco EMA, d₁ = 1 − α y los otros dos olvidos son los de PT5. El control `replica_bank` usa dᵣ = 1 − α en los tres. Cada banco tiene su propio momentum almacenado, en orden banco y después capa. No se añaden optimizadores externos ni parámetros compartidos respecto al mismo lector.

La regla anterior `common_translation` se conserva literalmente como valor por defecto. Suma las parciales y escribe un único momentum en los tres bancos. Su escala de paso es distinta de la proyección media. No se presenta como un control equivalente ni se multiplica el paso de `per_bank` por tres para igualar retrospectivamente una trayectoria. El contraste principal nuevo es `mean_partial` frente a `per_bank`. No redefine el B histórico ni activa CM-v1.

## Qué separa y qué sigue acoplado

Las parciales pertenecen a matrices distintas, pero comparten el error de la mezcla. Para memorias lineales escalares con claves y pesos de ruta fijos, la Hessiana es 2k²aaᵀ. Sus términos fuera de la diagonal suelen ser distintos de cero. Cambiar un banco puede cambiar la señal de los demás. No son gradientes estadísticamente independientes ni una eliminación de interferencias.

Las máscaras siguen controlando lectura. En `per_bank`, un banco no leído tiene parcial nula del dato actual. Puede seguir cambiando por olvido y por momentum anterior. En `mean_partial`, la media también puede llegar a bancos no leídos. Ninguna de las dos reglas equivale a borrar, congelar o reiniciar un banco. La comparación principal mantiene los tres disponibles.

Existe un caso construido donde compartir cancela una señal útil de la pérdida interna. Con tres MLP escalares de dos capas, activación GELU, claves y valores iguales a uno, ruta uniforme, primeras matrices cero y segundas matrices (1, −1, 0), la lectura inicial es cero. Las parciales de la primera capa son (−1/3, 1/3, 0) y las de la segunda son cero. La traslación común y la proyección media cancelan la escritura. Con olvido 1/2, momentum inicial cero y θ = 0,05, `per_bank` produce una lectura θ/18. Se usa GELU(x) − GELU(−x) = x. Las pruebas lo comprueban en CPU y CUDA. Es un contraejemplo a la equivalencia de las reglas, no evidencia de ventaja en retornos residuales.

En otros estados, conservar diferencias puede aumentar ruido, coste o sobreajuste. La pérdida interna no es el MAE residual. Reducir aquella no garantiza mejorar este. La hipótesis se rechaza si la regla separada no aporta utilidad frente a la proyección media, los controles de ruta y el residual cero con el protocolo temporal, o si su coste no compensa el efecto observado.

## Estado, recuperación y alcance matemático

Las dos reglas nuevas almacenan tres pesos y tres momentum por capa. La regla anterior almacena tres pesos y un momentum. Se mantienen los campos del payload, con distinta longitud de `momentum` y una identidad de configuración nueva. Estados y checkpoints de reglas distintas se rechazan, aunque sus formas coincidan. La opción por defecto conserva la identidad v1. El importador de recetas sigue aceptando sus tres campos originales.

Con profundidad L y dimensión D, las nuevas reglas añaden 2LD² elementos por flujo respecto a `common_translation`. `mean_partial` y `per_bank` tienen el mismo almacenamiento de tensores. La primera conserva momentum iguales cuando parte de cero y recibe la misma media en cada paso. Igual almacenamiento no implica igual capacidad efectiva. Los contadores nuevos distinguen matrices de gradiente y actualizaciones de momentum. No son FLOPs ni lanzamientos CUDA.

El cálculo de presupuesto, transporte por bloques y diagnóstico del Jacobiano incluye los momentum adicionales. Se comprueban diferencias finitas de las parciales y del gradiente externo de las direcciones a través de la derivada interna. El caso lineal escalar permite escribir la transición completa de orden seis, con estado z = (w₁,w₂,w₃,S₁,S₂,S₃), D = diag(d) y c = 2θk²:

$$
A=\begin{bmatrix}D-cQaa^\mathsf T&\eta I\\-cQaa^\mathsf T&\eta I\end{bmatrix},\qquad
b=\begin{bmatrix}2\theta kvQa\\2\theta kvQa\end{bmatrix},
$$

donde Q = I para `per_bank` y Q = P para `mean_partial`. Las pruebas contrastan A y b con autograd y encuentran amplificación con una clave sin normalizar aunque cada olvido sea menor que uno. El diagnóstico denso de MAC incluye además atención, puertas y todos los estados declarados. Estos cálculos locales no demuestran estabilidad global ni retención semántica. No se ha ejecutado una formalización Lean. PT1, plasticidad y C local siguen rechazados en esta combinación.

La implementación respeta la separación entre parámetros compartidos congelados por tramo y pesos rápidos que evolucionan mediante la política de inferencia. La memoria de mercado PT4 conserva su configuración anterior. La primitiva `update_shared` admite las reglas nuevas y promedia la pérdida de la cohorte, pero este cambio no conecta automáticamente esos bancos a la memoria de mercado del predictor.

## Fuentes y configuraciones

[Roy y colaboradores (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8980018/), figura 6, comparan reactivar conjuntos de engramas de varias regiones en ratones. El cuarto conjunto no añadió una mejora significativa frente al tercero. Motiva estudiar cooperación y límites, sin convertir matrices en regiones ni prometer que añadir bancos ayude siempre.

[MoM, versión 4](https://arxiv.org/html/2502.13685v4), sección 3.2, ya enruta entradas a estados de memoria y mantiene un estado compartido. Aquí no se implementa su selección top-k ni sus proyecciones por memoria. Los tres bancos se evalúan y siguen las reglas de olvido declaradas.

[Nested Learning](https://proceedings.neurips.cc/paper_files/paper/2025/file/4309616aaed8e848009bc4a7ef73b493-Paper-Conference.pdf), apéndice C, ecuación (42), ya agrega bloques de memoria. Esta variante no reproduce su sistema de frecuencias ni HOPE.

[uTTT, versión 1 de octubre de 2026](https://arxiv.org/html/2610.05484v1), secciones 2–3 y apéndice C, distingue estados compartidos entre capas, contribuciones agregadas y buffers de optimización. Aquí no se comparte una reserva entre profundidades ni se implementan su selección dispersa, pérdida interna o Muon. No se reivindica novedad por varias memorias ni se trasladan sus resultados al mercado.

La [declaración de diez configuraciones](../../configs/experiments/bank-write-v1.json) compara las dos reglas con rutas uniforme, constante y por consulta, réplicas y el mismo grafo regional. Reutiliza la receta cronológica de BANK-READ-v1 y K = 1. Son configuraciones preparadas, no entrenamientos completados. El bloqueo de aprendizaje y el test final siguen vigentes.
