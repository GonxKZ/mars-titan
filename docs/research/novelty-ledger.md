# Antecedentes, hipótesis y criterios de aportación

Estado: propuestas contrastables. No hay una mejora experimental demostrada ni una certificación de originalidad. La búsqueda incluye fuentes primarias recientes y una revisión crítica independiente de las hipótesis.

## Hipótesis que merecen un experimento

| ID | Pregunta concreta | Antecedentes próximos | Diferencia propuesta | Qué la refutaría |
| --- | --- | --- | --- | --- |
| N1 | ¿La retención por diversidad y error maduro mejora la recuperación útil frente a un buffer uniforme del mismo tamaño? | CLS, replay, DER, Latent Replay, Titans, MIRAS y memoria financiera. | Selección sobre predicciones realmente emitidas, errores maduros, disponibilidad de eventos y revisitas de régimen, con igual número de bytes y escrituras. | El buffer aleatorio o uniforme iguala o mejora el error y la adaptación, o el efecto desaparece al igualar cómputo. |
| N2 | ¿Asignar hasta cuatro pasos de lectura según información observable mejora la frontera de error y latencia? | ACT, PonderNet, Universal Transformer, recurrencia latente, RD-VLA y TTC. | Política pequeña condicionada por eventos financieros disponibles, evaluada con calibración de la política completa. | K fijo o una puerta sencilla domina en error, cobertura y p99. La convergencia latente no cuenta como confirmación. |
| N3 | ¿La selección de recuerdos y la política de recurrencia se complementan? | Combinaciones de memoria y cómputo adaptativo ya publicadas, incluido TIEM como antecedente financiero cercano. | Contraste factorial de retención básica/propuesta y puerta básica/propuesta sobre retorno residual bajo presupuesto. | La interacción no es estable entre ventanas o se explica por más actualizaciones, datos o capacidad. |
| N4 | ¿Un alumno de un paso conserva una mejora del profesor recurrente con menor latencia? | Destilación, DER y refinamiento recurrente. | Transferencia de una política numérica financiera con cortes verificables, coste total del profesor y recalibración independiente. | K = 1 directo es igual o mejor, o la ventaja de tiempo desaparece al contabilizar preparación y recuperación de memoria. |

N1 constituye la opción principal. N2 puede entrar con una decisión temprana de continuidad. N3 solo se ejecuta si N1 y N2 justifican el coste. N4 es una extensión. Esta jerarquía protege la comparación y la entrega frente a una combinación de todos los mecanismos sin atribución posible.

## Diseño mínimo para N1 y N2

Se conservarán el mismo codificador, entradas, objetivo residual, particiones y presupuesto de búsqueda. Para N1 se comparan ausencia de memoria, memoria uniforme y memoria selectiva. Para N2 se comparan K = 1, 2 y 4, además de puertas solo si los pasos adicionales muestran valor durante desarrollo.

La especificación selectiva utiliza tres índices y un almacén único de episodios. Su puntuación de admisión se conserva, de modo que no se interpreta como una optimización continua de diversidad global. La capacidad se iguala en bytes, contabilizando claves, valores y referencias. Las claves fijas y los valores proyectados al leer evitan mezclar representaciones entrenables antiguas en el núcleo. Estas decisiones también deben declararse al compararlo con una memoria de claves aprendidas.

Los contrastes principales se fijan antes del test. El resto se identifica como exploratorio. El error se compara de forma pareada por sesión y la incertidumbre mantiene los activos de la misma fecha juntos. La variable macro o el régimen no se seleccionan retrospectivamente para hacer aparecer una mejora.

K cuenta lecturas y refinamientos, sin una lectura extra en la inicialización. La comparación entre banco episódico y memoria neural de pesos rápidos mantiene identificada cada variante. La revisión de antecedentes exige esa referencia neural antes de atribuir una aportación frente a Titans. Se registran candidatos de escritura, operaciones de índices, valores únicos y cómputo efectivo de cada celda.

En un diseño factorial, la interacción puede escribirse como diferencia entre dos efectos:

$$\Delta_{int}=(L_{11}-L_{10})-(L_{01}-L_{00}),$$

donde L es pérdida fuera de muestra, el primer índice indica retención y el segundo política de pasos. Su signo y tamaño se interpretan junto a la incertidumbre y los recursos. No representa un efecto causal económico. Si los intervalos admiten mejoras y empeoramientos relevantes, la conclusión es incierta.

## Deducciones de diseño

**Disponibilidad por composición.** Si parámetros, transformaciones, índices y estado inicial se han construido con información permitida, cada lectura solo accede a esa información y cada actualización utiliza eventos ya conocidos, la salida sigue dependiendo del pasado permitido. Se comprueba por inducción sobre actualizaciones y pasos internos. Es una propiedad condicional de la especificación, no una prueba de que los datos reales cumplan las premisas.

**Información frente a cómputo.** Repetir una transformación de las mismas entradas y memoria no añade una observación externa. Puede mejorar la aproximación o la recuperación que realiza la red. Por tanto, el bucle tiene sentido como presupuesto de cálculo, pero no como garantía de eliminar incertidumbre del mercado. Esta distinción es conocida y no se reivindica como teorema nuevo.

**Memoria activa frente a corpus.** Con capacidad de episodios, colas y lote fijadas, los tensores de trabajo no necesitan crecer con todos los registros históricos. El estado por activo sí crece con el número de activos, salvo que se pagine. La cota debe incluir copias y optimizadores. De esta forma se puede diseñar un recorrido mayor que la RAM sin afirmar tiempo constante ni memoria infinita.

Estas deducciones ayudan a construir pruebas de invariancia y de consumo. El conocimiento nuevo, si lo hay, dependerá del resultado de aplicar y contrastar el diseño en el problema financiero.

## Cómo se controla la afirmación de novedad

Para cada comparación final se conservarán el artículo más cercano, su versión, objetivo, datos, política de memoria, regla de adaptación, calibración y coste. Se comprobarán especialmente SFM, DoubleAdapt, FinMem, FinAgent, FinCon, MacroHFT y TIEM, además de los trabajos de memoria y recurrencia general.

Un componente basado en código ajeno se identifica y conserva su licencia. Una adaptación se describe mediante sus cambios verificables. La frase «no se ha probado antes» solo se sustituiría por una afirmación limitada al alcance y fecha de una búsqueda documentada, nunca por una garantía universal.

El registro de ensayos incluye intentos manuales, modelos descartados, búsquedas de variables y decisiones de alcance. Obtener un resultado favorable tras muchos intentos requiere un análisis que tenga en cuenta ese proceso. El test final no se reutiliza para inventar una explicación o diseñar otra variante.

## Resultado científicamente útil

Una mejora pequeña que resista controles, una reducción de coste sin pérdida relevante de calidad o un límite bien caracterizado pueden constituir una aportación defendible. Si una referencia sencilla domina, se conserva ese resultado y se explica. La selección final no está condicionada a que el modelo más complejo gane.
