# Fusión regional con dos ritmos de trabajo

`regional_fusion_v1` es una variante opcional anterior a MAC. Comunica las cinco representaciones existentes mediante un grafo pequeño y un resumen que se actualiza entre grupos de pasos locales. Reutiliza codificadores, cabeza, memoria de Titans, banco episódico y canalización. No sustituye las referencias ni está activada por defecto. No se ha entrenado ni hay una mejora predictiva medida.

## Entradas, regiones y estados

[`RegionalFusion`](../../src/mars_titan/models/titans/regional_fusion.py) recibe bloques `[B,5,H]` ya proyectados y presencia `[B,5]`. El orden es precios, noticias, gráficos, fundamentales y macro. Son etiquetas de procedencia de los datos, no regiones cerebrales ni regímenes económicos inferidos. El grafo no conecta activos. Cada fila tiene sus propios estados y se puede dividir el lote sin transportar estado regional entre filas.

La fusión histórica sigue anulando las proyecciones ausentes después del sesgo del codificador. El tratamiento anterior de la representación de precios se conserva. Dentro del grafo, un nodo ausente no envía mensajes, no participa en el resumen y no recibe contenido nuevo. Su bloque original se devuelve sin cambios. En la comparación estricta, la ausencia de máscara significa que las cinco fuentes están presentes. No autoriza rellenar una fuente histórica desconocida.

Los estados locales y el resumen empiezan de nuevo en cada predicción. «Lento» describe la frecuencia del resumen dentro de ese cálculo. No es un estado histórico entre sesiones. Los parámetros regionales pertenecen al ajuste externo compartido. No son los pesos rápidos de Titans, sus parámetros persistentes, el contexto PT4 ni episodios. Los estados por flujo conservan sus tensores y su tamaño anteriores.

## Recurrencia ejecutada

Para una fila, X contiene cinco vectores de dimensión H y P es su presencia. Las multiplicaciones por P se aplican por nodo. El estado inicial es R⁰ = P ⊙ X, con S⁰ = 0 y U⁰ = 0. Dentro del ciclo c se ejecutan L pasos locales:

$$
Z = P\odot\operatorname{LN}(R+U^{c-1}),\qquad
(Q,K,V)=ZW_{qkv}^{\mathsf T},
$$

$$
A=\operatorname{softmax}\left(QK^{\mathsf T}/\sqrt H+\mathcal M\right),\qquad
R\leftarrow P\odot\left(R+\eta A V W_o^{\mathsf T}\right).
$$

La máscara M admite solo las aristas declaradas y las claves presentes. Para una consulta ausente se habilita únicamente su propio cero y se anula su actualización. Esto evita una fila de softmax sin ninguna clave admisible. La ruta uniforme calcula QKV y sustituye Q y K por cero. Distribuye peso uniforme sobre las claves admisibles y no utiliza los gradientes de Q/K.

Después de esos pasos, si el resumen está activado y existe algún nodo presente:

$$
m=\frac{\sum_{j=1}^{5}R_j}{\max(1,\sum_{j=1}^{5}P_j)},\qquad
S^c=S^{c-1}+\gamma\operatorname{SiLU}\left((\operatorname{LN}_s(S^{c-1})+m)W_1^{\mathsf T}\right)W_2^{\mathsf T},
$$

$$
U^c=S^c W_{s\to r}^{\mathsf T}.
$$

U se difunde a los cinco nodos en el ciclo siguiente. Con todos los nodos ausentes, el resumen conserva cero. Al acabar C ciclos:

$$
Y=X+P\odot\left((R^{C,L}-R^0+U^C)W_r^{\mathsf T}\right).
$$

Sin resumen, S y U permanecen en cero. Se comprueba que desplazar una actualización del resumen entre pasos cambia la recurrencia y que, sin ella, particionar el mismo número de pasos locales conserva el resultado. Los gradientes se calculan sobre todas las iteraciones ejecutadas. No se aplica una aproximación de gradiente de punto fijo ni se afirma haber alcanzado un equilibrio.

Las ganancias por defecto, η = γ = 0,25, son hiperparámetros, no constantes obtenidas de neurociencia. Una ganancia menor que uno no demuestra contracción del Jacobiano completo. La atención depende del estado y las actualizaciones incluyen identidad, normalizaciones y proyecciones aprendidas. No se atribuye a Crouzeix una garantía de estabilidad de este grafo.

Con parámetros y máscaras fijos, cada operación conserva la separación entre filas. Por inducción sobre los pasos locales y ciclos, permutar activos permuta sus salidas y dividir el lote conserva el resultado dentro de la tolerancia declarada. Para los nodos presentes, cambiar únicamente el contenido de nodos ausentes no cambia los mensajes ni el resumen. La salida de un nodo ausente conserva su bloque original, así que esa última propiedad no se formula como invariancia de sus propios bytes de salida. Las pruebas contrastan ambas distinciones, incluidos valores ausentes de magnitud 10³⁰⁸ en FP64. Es un argumento sobre la recurrencia, no una prueba formal en Lean ni una garantía de disponibilidad temporal de las fuentes. Esa disponibilidad se comprueba antes de formar los bloques.

## Topologías y controles

La topología completa conecta los cinco nodos. El anillo usa el orden declarado, sus dos vecinos y el propio nodo. La topología `self` admite solo la arista propia. Ese orden del anillo es una decisión de cómputo que debe contrastarse, no una distancia económica. Cuando existe resumen, el hub crea conexiones indirectas entre todos los nodos presentes, incluso en `self`.

La [declaración de configuraciones](../../configs/experiments/regional-fusion-v1.json) conserva la fusión anterior y siete alternativas regionales. Compara nodos sin comunicación ni resumen, mensajes uniformes con resumen, atención completa con resumen, anillo con y sin resumen y dos frecuencias adicionales del resumen. Los tres ritmos del anillo ejecutan cuatro pasos locales, pero una, dos o cuatro actualizaciones del resumen. Su coste no es idéntico y debe medirse.

Todas las alternativas regionales almacenan 10H² + 4H parámetros nuevos y 25 valores booleanos de aristas. En el tensor `torch.bool` son 25 bytes de datos, además de los metadatos y la serialización. La referencia anterior tiene menos parámetros. Uniformidad y retirada del resumen dejan parámetros sin uso efectivo. Por tanto, igualdad de parámetros almacenados no significa igualdad de capacidad efectiva ni de trabajo. Esos controles sirven para separar comunicación dependiente de entrada, conexiones indirectas y capacidad añadida. No establecen especialización aprendida por tener etiquetas distintas.

El anillo y el nodo global tienen un antecedente directo en [Star-Transformer, NAACL 2019](https://aclanthology.org/N19-1133/), cuya sección 3 describe estados satélite y relay. Aquí los nodos representan fuentes, el resumen usa una media y una MLP, y su influencia entra mediante una proyección aditiva. No es una reproducción de ese modelo. La [atención enmascarada de GAT, versión 3 del 4 de febrero de 2018](https://arxiv.org/abs/1710.10903v3), es otro antecedente de comunicación sobre vecindades. El módulo actual usa atención de producto escalar de PyTorch. No se reivindica novedad por usar un grafo, un anillo o un hub.

La [revisión de contexto y memoria distribuida](../research/context-and-nested-control.md) conserva los antecedentes de neurociencia, SEAL, HRM, TRM y composición modular. Esta implementación no reproduce engramas biológicos, no aprende todavía a elegir acciones del catálogo y no proporciona memoria histórica repartida en varios almacenes regionales. Tampoco prueba recuperación después de cualquier lesión. Esas hipótesis requieren otras variantes y controles.

## Presupuesto, integración y recuperación

El módulo admite hasta cuatro ciclos y cuatro pasos locales por ciclo, con la condición conjunta C(5L + 1_resumen) ≤ 64 evaluaciones de nodo por fila. El lote y H también están acotados. La atención emplea una matriz densa de 5 × 5 con máscara. El anillo no se presenta como un backend de complejidad lineal ni como una aceleración sin medidas. La memoria de trabajo depende del lote, H y las iteraciones, con activaciones adicionales al calcular gradientes. El límite de visitas no es una cota del pico completo de RAM o VRAM del framework.

Con m nodos, la aritmética densa dominante es O(BCL(mH² + m²H) + BCH²). En esta implementación m = 5, también para el anillo. El término del resumen desaparece cuando se desactiva. La propagación de gradientes conserva activaciones de las iteraciones y añade memoria proporcional a su número. El coste y el workspace de las primitivas del backend necesitan medidas aparte. No se construye una matriz de relaciones entre todos los activos ni entre todos los episodios históricos.

[`FinancialConfig`](../../src/mars_titan/models/titans/financial.py) incorpora la opción solo cuando se declara. Con `None` conserva el formato anterior. Los generadores del módulo nuevo no alteran los parámetros del tronco ni el RNG global. Se mantiene una transición MAC por observación, después de la fusión, con independencia de las vueltas del grafo. Los refinamientos episódicos K son otro eje. La combinación con PT6 se rechaza hasta definir un contraste propio.

La [receta independiente](../../configs/experiments/regional-fusion-recipe-v1.json) usa el constructor cronológico existente. No se ha añadido una canalización de entrenamiento separada. Los parámetros regionales entran en el grupo compartido del ajuste externo. La ruta admite FP32 y FP64 explícitos y rechaza autocast. Las verificaciones de finitud anteriores a los enmascaramientos impiden que una máscara vacía oculte un intermedio inválido.

El consumidor congelado incorpora la fuente y la configuración regional a sus comprobaciones. Se rechazan cambios de planificación después de sellar y checkpoints incompatibles. Las nuevas huellas de código no autorizan reutilizar un runtime científico histórico como si conservara su identidad. La referencia anterior permanece disponible en su versión fijada.

La carga deriva las aristas esperadas de la configuración, no de los valores mutables de `edges`. También comprueba las formas físicas derivadas de H, incluidas QKV `[3H,H]` y las proyecciones del resumen `[2H,H]` y `[H,2H]`. El buffer instalado debe conservar tipo booleano, forma `[5,5]`, layout y dispositivo compatibles. Un checkpoint válido puede reparar sus valores alterados, pero no una estructura incompatible. El predictor prevalida el hijo regional antes de copiar cualquier parámetro del predictor y comprueba la geometría al volver a sellar tras una carga o conversión. La finitud del payload se resuelve inmediatamente, incluso dentro de `deferred_checks`, para rechazar NaN o infinito antes de copiar.

Los métodos públicos `load_state_dict` del predictor y del módulo regional rechazan un `assign` no booleano y metadatos de asignación contradictorios o no booleanos. Al cargar la fusión regional, con `assign=True` los tensores del payload deben compartir dispositivo. Con `assign=False`, se admiten orígenes CPU/CUDA mezclados y se conserva el dispositivo del destino. El rechazo previo cubre estas APIs directas, sin garantizar atomicidad universal ante hooks arbitrarios o contenedores externos que eludan su entrada pública.

La corrección no modifica el `forward` ni sus ecuaciones y no completa los programas regionales coordinados de #709 ni el ciclo exterior de propuestas de #711. El [informe de geometría regional](../../reports/engineering/regional-geometry-20261022/README.md) recoge las verificaciones y costes del arreglo, con sus fuentes fijadas. La compatibilidad del checkpoint no implica equivalencia de runtimes con distintas fuentes.

Los enlaces de sesión registran evaluaciones locales y del resumen. Son evaluaciones de nodos, incluidas las consultas enmascaradas que se calculan, no FLOPs ni actualizaciones de parámetros. El enlace de mecanismos conserva esos contadores en el mismo historial confirmado que la emisión. La recuperación tras un corte posterior a publicar reproduce el estado y la siguiente emisión sin duplicar feedback. No necesita publicar los estados intermedios del grafo, porque se reconstruyen dentro del evento desde la entrada y el snapshot confirmados.

Las pruebas de componente y de integración usan fixtures identificados. La selección ampliada, la cobertura, la mutación y los costes se registran en el [informe técnico](../../reports/engineering/regional-fusion-20261021/README.md). La campaña completa y el test final permanecen cerrados. Un funcionamiento correcto sobre fixtures no demuestra una reducción de error en FinMultiTime.
