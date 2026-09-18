# Arquitectura candidata de memoria y recurrencia

Esta especificación concreta una hipótesis de MARS-TITAN. Se apoya en mecanismos publicados y en la [revisión adversarial](adversarial-review.md). Su funcionamiento, coste y posible aportación propia están pendientes de implementación y experimentación.

## Separar memoria persistente y cálculo interno

Se distinguen tres objetos. Los parámetros compartidos aprenden representaciones durante entrenamiento. El estado persistente conserva información autorizada entre decisiones. Un estado de trabajo existe solo mientras se refina una predicción y se descarta al terminar. Repetir este último estado es lo que aquí se llama recurrencia interna. No produce una cadena verbal de pensamientos ni constituye una reproducción del cerebro humano.

El [aprendizaje complementario](../references/brain-review.md) motiva estudiar captación rápida de episodios e integración más lenta. La traducción algorítmica es una hipótesis funcional. Las evidencias de replay humano y animal, de separación de patrones y de sorpresa se mantienen diferenciadas de los resultados de redes artificiales.

```mermaid
flowchart LR
    D[Datos disponibles<br/>precios, noticias y macro] --> E[Representación compacta<br/>versión congelada]
    E --> H[Estado temporal<br/>por activo]
    H --> Q[Consulta de episodios<br/>instantánea inmutable]
    Q --> K[Refinamiento<br/>K = 1, 2 o 4]
    K --> P[Retorno e intervalos<br/>calibración y abstención]
    P --> L[Registro de la predicción<br/>versión y corte]
    L --> Y[Etiqueta madura<br/>error originalmente emitido]
    Y --> W[Selección y escritura<br/>capacidad limitada]
    W --> Q
    W -. entrenamiento permitido .-> C[Consolidación y replay<br/>fuera de la ruta rápida]
```

La flecha de escritura solo afecta a decisiones posteriores. El gráfico no autoriza un ciclo que conozca el retorno futuro antes de predecirlo.

## Núcleo que se comparará primero

Un codificador temporal compacto procesa precios y máscaras. Las noticias se representan con un codificador congelado y una proyección pequeña. Las variables macro usan su última versión disponible y registran antigüedad y ausencias. La fusión se mantiene sencilla para poder atribuir el efecto de la memoria.

La referencia inicial es una GRU. Una SSM o regla delta entra como contraste moderno cuando la medida justifique su coste. El candidato comparte pesos entre activos y guarda estados pequeños por activo. No replica una red entrenable completa para cada uno de los miles de símbolos.

La memoria episódica tiene capacidad explícita. El punto inicial de dimensionamiento es E = 8.192 referencias de episodios, rasgos base de dimensión 256, claves de dimensión 128 y recuperación de hasta ocho entradas únicas por consulta. Son valores para el piloto, no hiperparámetros ya validados. Cada episodio conserva identificador estable, mercado, corte, origen, revisión de representación y motivo de escritura. Los vectores no sustituyen su procedencia.

En el núcleo, los rasgos base dependen de transformaciones y codificadores congelados dentro del fold. Las claves se obtienen normalizando una proyección fija R, inicializada con una semilla registrada. No se almacenan estados GRU ni proyecciones entrenables antiguas como si siguieran vigentes después de cada paso del optimizador. La consulta aprendida se adapta a ese espacio de claves fijo. Los valores se proyectan al leer, sin conservar proyecciones entrenables entre pasos. Modificar R o los rasgos base exige reconstruir la memoria. Una proyección de claves aprendida es una variante distinta con ese coste incluido.

Se utilizará primero una memoria global con etiquetas de contexto. Separar bancos de mercado, sector o régimen será una ablación con la misma capacidad total. Las clasificaciones sectoriales o de régimen deben existir en el corte correspondiente. Un régimen retrospectivo puede servir para describir resultados, pero no entrar como característica de la predicción pasada.

## Escritura y retención

El error de un evento se calcula cuando su etiqueta ha madurado:

$$e_j = \left|y_j-\widehat y_j^{\mathrm{emitida}}\right|.$$

La predicción usada es la que se conservó en el registro original, junto con su versión. Recalcularla con pesos posteriores altera el significado de sorpresa. La escala del error se estima con historia permitida y se limita su influencia para que un dato erróneo no monopolice la memoria.

La puntuación candidata combina error normalizado, diversidad respecto a episodios existentes y relevancia de información publicada. Los coeficientes y umbrales se eligen en desarrollo. El desacuerdo entre modelos o un error grande no demuestran que un evento sea aprendible. Se contrastarán con escrituras uniformes y muestreo aleatorio de igual coste.

Para limitar sesgos del selector se propone una mezcla inicial de cupos: la mitad mediante reservoir sampling uniforme de la historia elegible, un cuarto por puntuación selectiva y otro cuarto para eventos recientes. Cada evento elegible se ofrece a tres índices. El uniforme usa el algoritmo de reservorio con semilla y contador persistidos. El reciente conserva los últimos eventos en el orden temporal canónico. El selectivo conserva las mayores puntuaciones de admisión, con desempate determinista por identificador.

La puntuación de error, novedad y relevancia gobierna únicamente el índice selectivo. Se calcula una vez al admitir el candidato frente al conjunto selectivo existente y permanece guardada. La novedad al ingresar no garantiza diversidad óptima del conjunto futuro. Así se evita recalcular todos los pares de recuerdos en cada escritura. El coste de consultar las claves para puntuar una admisión también debe medirse.

Los índices pueden apuntar al mismo evento, pero comparten un único registro de rasgos por `event_id`. Se eliminan duplicados antes de recuperar las ocho entradas. La suma de capacidades de índices no supera E y su unión puede contener menos episodios. Se informa número de registros únicos, referencias de índices y bytes totales. Las comparaciones igualan esos costes, no solo un parámetro llamado capacidad. La composición de cupos puede revisarse en el piloto, antes del test, y se contrastará con una política uniforme sin cupos.

El olvido se medirá como pérdida de capacidad predictiva o de recuperación sobre contextos anteriores. Se distinguirá del descarte de información obsoleta y de la pérdida de plasticidad para aprender contextos nuevos. Un archivo histórico puede conservar datos en disco, pero la memoria activa seguirá siendo finita.

El registro separará candidatos examinados, actualizaciones de índices, inserciones de valores únicos y bytes transferidos. Ofrecer un evento a tres índices no equivale a una escritura del control uniforme. Esos costes se incluyen al interpretar el efecto de la selección.

## Contraste con memoria neural de pesos rápidos

El banco de episodios es una referencia de memoria externa con lectura aprendida. No se identifica con la memoria neuronal paramétrica de Titans. Para mantener esa distinción, la comparación incluirá una memoria asociativa lineal compacta con regla delta como referencia neural adaptativa, cuya formulación concreta se fijará antes de implementar.

En esa variante, una matriz de estado A relaciona claves y valores permitidos. Una forma candidata de actualización es

$$A^+=(1-\lambda_j)A+\eta_j k_j(v_j-A^T k_j)^T.$$

Las dimensiones de A son dimensión de clave por dimensión de valor. Las tasas, la normalización y los límites de norma se fijan en desarrollo. Los valores supervisados solo se incorporan al madurar. Las actualizaciones usan el orden canónico y su coste se mide. La regla es un antecedente conocido de memoria asociativa, no una invención de MARS-TITAN ni una reproducción completa de Titans.

A cambia durante evaluación como estado o peso rápido de memoria bajo una política predefinida. Esto se distingue del reajuste de los parámetros compartidos del codificador y de la cabeza, que permanecen congelados en el núcleo. Las claves y valores de la referencia se construyen en un espacio estable. Si su proyección cambia, se reinicia o reconstruye el estado compatible. La combinación del banco episódico con A es una extensión, no un requisito para que ambas referencias puedan evaluarse por separado.

## Refinamiento de una predicción

Con representación h y memoria fija M, K cuenta el número total de lecturas y bloques de refinamiento. La inicialización no consulta memoria:

$$z^{(0)}=g_0(h),\qquad
z^{(k+1)}=z^{(k)}+\eta_k g_\theta\!\left(z^{(k)},h,\operatorname{read}(M,z^{(k)})\right).$$

El índice k recorre de cero a K menos uno. K = 1 realiza una lectura y un refinamiento. Solo se consideran salidas finales en K = 1, 2 o 4. La memoria M representa la variante evaluada, banco episódico o estado asociativo, sin combinarlas de forma implícita.

Las cabezas producen cuantiles a partir de z. El rango de la escala de actualización y las normas del estado se controlarán durante entrenamiento. Esto no garantiza convergencia de la dinámica. El máximo de pasos será explícito y se registrarán divergencia, oscilaciones y saturación.

La selección top-k de índices se trata como constante en la diferenciación. Dentro del conjunto recuperado se aplica una lectura ponderada con `softmax(q·clave/τ)`, temperatura positiva y suma de valores proyectados a dimensión 128. De ese modo la consulta puede recibir gradiente a través de los pesos, sin afirmar que top-k duro sea diferenciable. Con una sola entrada ese camino no aporta gradiente de selección, una condición que se registrará en el piloto. Con memoria vacía se devuelve un vector cero y una máscara de ausencia.

El valor del episodio contiene rasgos base estables y, cuando se incluya, su retorno residual ya maduro. Este último se almacena con precisión suficiente y disponibilidad explícita. Ninguna etiqueta pendiente entra en el conjunto recuperable. Las consultas de varias iteraciones leen la misma instantánea.

La primera comparación usa K fijo en {1, 2, 4}. Una puerta aprendida solo se incorpora si existe una utilidad medible. Sus entradas pueden incluir el estado observable del mercado, presencia y antigüedad de una publicación y señales de incertidumbre generadas con cortes válidos. No recibe el error futuro de la predicción actual. Se contrasta con una puerta basada solo en presupuesto y otra aleatoria de igual cómputo medio.

La política completa se calibra con un tramo separado, incluyendo su elección de K y la ruta de retorno por tiempo agotado. Una reducción del cambio entre z sucesivos es un criterio de estabilidad numérica, no una medida de probabilidad de acierto. Si no puede completarse el K elegido, se permite devolver la salida de K = 1 que se haya conservado y validado, o abstenerse. No se emite una salida intermedia de K = 3 ni se supone que K = 1 siempre termina dentro del plazo. El cálculo descartado y su consumo también se registran.

## Consolidación y destilación

El núcleo mantiene codificador, normalizadores y parámetros congelados dentro de cada tramo de evaluación. La adaptación permitida afecta a estados explícitos. La consolidación paramétrica mediante replay ocurre en entrenamiento y en reajustes programados que solo utilizan pasado. Una actualización de pesos durante test requeriría una variante online distinta, predefinida y comparada con esa misma información disponible.

Se estudiará un replay sencillo antes de EWC, SI, GEM o replay generativo. Mantener generadores, matrices de importancia y proyecciones puede costar más que la mejora que aporten. Los ejemplos retenidos se estratifican con información conocida, no según el régimen que el futuro termine revelando.

Una variante de destilación podrá aprender de un profesor K = 4 y responder con K = 1. El profesor se ajusta solo en el tramo autorizado y su coste se incluye. Sus salidas de entrenamiento no se presentan como predicciones históricas fuera de muestra. Las señales usadas para aprender la puerta o para reconstruir errores de escritura necesitan una generación histórica válida. Se recalibra el estudiante y se compara con K = 1 entrenado directamente.

## Instantánea coherente y concurrencia

Cada decisión fija un conjunto compatible con identificadores de datos y sus versiones, codificador, normalizadores, parámetros, memoria, calibrador, política de K, reglas de riesgo y corte de información. La cohorte utiliza esa misma instantánea aunque el trabajo se divida en microlotes.

Las etiquetas maduras se ordenan según la política canónica del protocolo. Una agregación conjunta de escrituras sería otra regla de actualización que requiere su propio contraste. Publicar una instantánea nueva necesita una barrera coherente, con sincronización de las operaciones pendientes y verificación de compatibilidad.

Un episodio codificado con otra revisión no puede mezclarse silenciosamente con consultas nuevas. Cambiar el codificador obliga a reconstruir o migrar la memoria y a comprobar el resultado. Una revisión macro es un evento nuevo conocido en su publicación. No reescribe la historia de decisiones ya emitidas.

El doble buffer se cuenta dentro de la VRAM. Si compartir GPU entre consolidación e inferencia deteriora p99, se separan temporalmente ambas cargas. La disponibilidad 24/7 permite programar ese trabajo y reanudarlo, no elimina la competencia física por memoria y cómputo.

## Qué constituiría una aportación

La contribución candidata es determinar si la retención con diversidad y la asignación de pocos pasos internos a eventos disponibles mejoran la predicción financiera a igualdad de recursos. La interacción entre esas decisiones puede estudiarse mediante un diseño factorial pequeño. La memoria episódica, el replay, la recurrencia y la destilación ya tienen antecedentes directos.

La [revisión reciente](../references/frontier-review.md) incluye TIEM y modelos de memoria financiera. Se deberá comparar qué estado cambia durante evaluación, qué objetivo predicen, qué datos usan y qué coste introducen. MARS-TITAN no se declara original por cambiar el nombre de esos mecanismos. El [registro de hipótesis](novelty-ledger.md) define las diferencias y las pruebas que pueden refutarlas.
