# Regímenes de Markov como contexto de memoria

Revisión del 21 de septiembre de 2026. Esta nota propone un contraste para O3.
No hay un detector implementado ni resultados propios con HMM sobre FinMultiTime.

## Qué aporta el PDF

[Jurafsky y Martin](https://web.stanford.edu/~jurafsky/slp3/A.pdf) explican cadenas
de Markov, modelos con estados ocultos, forward, Viterbi y Baum Welch. Se leyó el
apéndice completo de 17 páginas, en su versión del 19 de agosto de 2026. La
recurrencia de la página 7 se comprobó también sobre el PDF renderizado.

El algoritmo forward evita enumerar todas las rutas de estados. Viterbi busca una
ruta de estados de máxima probabilidad bajo el modelo, no una distribución de
rentabilidad. Baum Welch ajusta parámetros
mediante expectativas de estados y transiciones. El texto enseña inferencia
probabilística, no demuestra capacidad para predecir precios. Tampoco equivale
a una memoria episódica ilimitada ni a un modelo del aprendizaje humano.

La copia local conserva su SHA-256 y recibo de descarga. No se redistribuye.
El enlace social asociado no pudo leerse directamente, por lo que no se le
atribuye contenido adicional al PDF identificado.

## Antecedentes financieros y límites de lectura

| Fuente primaria | Aportación que motiva el contraste | Límite |
| --- | --- | --- |
| [Hamilton, 1989](https://doi.org/10.2307/1912559) | Parámetros que cambian según un estado Markoviano no observado | Aplicación macroeconómica. Resumen y metadatos, no lectura integral |
| [Turner, Startz y Nelson, 1989](https://www.nber.org/papers/w2818) | Riesgo y aprendizaje sobre estados latentes en retornos bursátiles | Otra cartera, frecuencia y tarea. No se trasladan resultados al proyecto |
| [Hamilton y Susmel, 1994](https://doi.org/10.1016/0304-4076(94)90067-1) | Volatilidad condicional con cambios de régimen | Referencia documental, no reproducción de SWARCH |
| [Kim, 1994](https://pure.korea.ac.kr/en/publications/dynamic-linear-models-with-markov-switching/) | Diferencia entre filtrado y suavizado en modelos con cambio de estado | Se revisaron resumen y metadatos |
| [Hansen, 1992](https://users.ssc.wisc.edu/~behansen/papers/jae_92.html) | Condiciones no estándar al contrastar cambios de régimen | No seleccionar estados con una prueba chi cuadrado rutinaria |

Estas referencias justifican estudiar el contexto de régimen. No justifican
sustituir el núcleo multimodal por una cadena de Markov ni anunciar una mejora.
La ficha, el acceso y la bibliografía están en
[markov-sources.json](../references/markov-sources.json) y
[markov.bib](../references/markov.bib). La síntesis general de Hamilton ya incluida
en el catálogo financiero se conserva sin duplicarla.

## Diseño mínimo propuesto

Un único HMM de dos estados resume el contexto de mercado una vez por sesión.
Sus observaciones serían un vector pequeño de retorno de mercado, volatilidad
y dispersión transversal. Todas deben estar disponibles antes de la predicción.
La dispersión se calcula sobre el universo elegible en ese corte, no sobre una
lista actual de supervivientes. Las ausencias tienen una regla explícita.

Sean $P_{ij}$ las probabilidades de transición, $b_j(x_t)$ la densidad de emisión
y $\alpha_{t-1}$ el estado filtrado anterior. La actualización normalizada es

$$
\pi^-_t(j)=\sum_i\alpha_{t-1}(i)P_{ij},\qquad
\alpha_t(j)=\frac{b_j(x_t)\pi^-_t(j)}{\sum_\ell b_\ell(x_t)\pi^-_t(\ell)}.
$$

Aquí $\alpha_t=P(S_t\mid x_{1:t})$. No debe confundirse con la variable forward
sin normalizar del apéndice. El suavizado $P(S_t\mid x_{1:T})$, con $T>t$, incorpora
futuro y no puede ser una entrada, etiqueta de memoria o criterio de escritura.
Tampoco sirve una ruta Viterbi reconstruida con la secuencia completa.

El prototipo usaría emisiones gaussianas diagonales, con escalado ajustado solo
en entrenamiento. El cálculo se haría con normalización por paso o en dominio
logarítmico. Se detectarían probabilidades no finitas, varianzas degeneradas,
filas de transición inválidas y estados sin ocupación suficiente. Las colas
financieras pueden incumplir la emisión gaussiana. Ese fallo se diagnostica antes
de proponer otra distribución.

El vector filtrado puede incorporarse a la consulta de memoria, compartido entre
activos de la misma sesión. Cada episodio conserva el contexto conocido cuando
se admitió. No se recalculan episodios antiguos con parámetros futuros. Las
cuatro modalidades y el contexto macro siguen presentes en el predictor.

La probabilidad de cambio y la sorpresa del detector serían extensiones separadas:

$$
\xi_t(i,j)=\frac{\alpha_{t-1}(i)P_{ij}b_j(x_t)}
{\sum_{a,b}\alpha_{t-1}(a)P_{ab}b_b(x_t)},\qquad
q_t=\sum_{i\ne j}\xi_t(i,j),
$$

$$
a_t=-\log\sum_j b_j(x_t)\pi^-_t(j).
$$

Son puntuaciones estadísticas, no identificación de causas económicas. La
sorpresa predictiva de una etiqueta solo puede intervenir una vez madurada.
Para escribir el episodio $j$ se conservan $q_j$ y $a_j$, calculados cuando llegó
$x_j$, y se combinan con su error al madurar la etiqueta. No se sustituyen por
el contexto de la sesión posterior de maduración, que tendría otro significado.
Todos los activos de una sesión se predicen antes de las escrituras compartidas
que deban afectar únicamente a decisiones posteriores.

## Comparación que puede descartar la idea

| Variante | Contexto y memoria | Qué permite comprobar |
| --- | --- | --- |
| M0 | Sin memoria, sin régimen | Referencia del predictor con las mismas modalidades |
| M1 | Memoria global, sin régimen | Efecto de la memoria |
| R0 | Sin memoria, con probabilidades filtradas | Señal directa del régimen |
| R1 | Memoria global, con probabilidades filtradas | Interacción entre régimen y memoria |
| C1 | Memoria global, con volatilidad y dispersión continuas | Control barato frente al HMM |
| R2, condicionado | R1 y puntuación de escritura con $q_t$ o $a_t$ | Selección de episodios, solo si R1 lo justifica |

La ausencia de régimen es una ablación, no una retirada del componente de la
arquitectura completa prevista. Para R2 se igualan capacidad, bytes, consultas
y número de escrituras. No se empieza dividiendo la memoria en bancos rígidos,
porque aumentaría el riesgo de bancos vacíos y confundiría contexto con capacidad.

MAE del retorno residual sigue siendo la métrica principal. Se añaden Rank IC por
sesión, calibración, cobertura, coste y fallos por ventana. Las diferencias se
comparan por sesión y con intervalos por bloques. El margen mínimo relevante se
fija en desarrollo, nunca después de ver el test final. La falta de significación
no demuestra equivalencia.

Antes de conservar el HMM se comprueban ocupación, entropía, transiciones,
verosimilitud predictiva frente a un único estado y estabilidad entre
inicializaciones. Si el control continuo explica el efecto, la solución simple
tiene prioridad. Si solo mejora una ventana o un estado queda casi vacío, se
registra ese resultado y no se amplía automáticamente el modelo.

## Ajuste y temporalidad

1. Ajustar normalizadores, emisiones, transiciones e inicialización con pasado.
   Generar contexto de entrenamiento mediante cortes internos cronológicos o un
   periodo de calentamiento anterior. Un ajuste sobre todo el entrenamiento no
   reproduce el conocimiento disponible en sus primeras fechas.
2. Fijar dos estados. Elegir semillas, reinicios y regla de selección durante
   desarrollo. Ordenar etiquetas técnicas por volatilidad de entrenamiento si
   hace falta compararlas, sin llamarlas automáticamente crisis o mercado alcista.
3. Congelar parámetros durante cada tramo de evaluación. Solo evoluciona el
   filtro con nuevas observaciones admitidas. Reajustes únicamente entre cortes
   programados y con historia autorizada.
4. Reiniciar o reconstruir filtro y memoria dentro de cada fold, sin heredar el
   final de otro experimento. Registrar el calentamiento exacto y las huellas.
5. Conservar el estado filtrado, parámetros, cursor y versión de las entradas en
   los puntos de control. Cambiar el sufijo futuro no puede cambiar el prefijo.

## Coste y alternativas

Con $K$ estados y $d$ observaciones, la propuesta de emisión diagonal requiere
$O(K^2+Kd)$ operaciones por sesión, $O(K)$ de estado y $O(K^2+Kd)$ parámetros.
Guardar toda la historia no es necesario para inferencia. En entrenamiento,
forward y backward sí necesitan una política de almacenamiento o recomputación.
Se medirán inicializaciones descartadas y ajuste, no solo el paso barato del filtro.

Python con NumPy basta como primera referencia del detector de dos estados.
No compensa presumir una ganancia CUDA para matrices tan pequeñas. C++20 solo se
considerará si el perfil completo demuestra un coste material de este módulo.
Se registrarán latencia p50/p95/p99, tiempo total, bytes de estado y pico de memoria.

El HMM de primer orden impone permanencias geométricas. Un
[modelo semi Markov](https://doi.org/10.1016/S0165-1684(02)00378-X) permite duración
explícita, pero añade parámetros y estado. Las
[transiciones variables de Filardo](https://doi.org/10.1080/07350015.1994.10524545)
son otra ampliación, no un requisito inicial.
[Adams y MacKay](https://www.cs.princeton.edu/~rpa/pubs/adams2007changepoint.pdf)
plantean detectar puntos de cambio mediante una distribución sobre la longitud
de la racha. Es útil cuando interesa una ruptura y no un estado recurrente.
Su soporte puede crecer si no se limita, por lo que tampoco se asume coste constante.

La aportación investigable es comprobar si contexto filtrado y memoria interactúan
de forma útil con un presupuesto fijo. Es una hipótesis de adaptación, no una
afirmación de novedad, recuerdo perfecto ni anticipación garantizada del mercado.
