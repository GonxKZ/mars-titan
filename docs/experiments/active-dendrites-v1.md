# Modulación contextual por segmentos dendríticos, N2

Están implementados el operador neuronal `ActiveDendrites`, su contexto de régimen observable y la conexión con `FinancialPredictor`. N2 se aplica después del contexto compartido de sesión y antes de MAC. La misma configuración funciona con el Transformer directo y los controles MAC desactivado, congelado y en línea. La sesión durable y el consumidor de mecanismos todavía no admiten N2. No se han realizado entrenamientos, comparaciones predictivas ni medidas de coste del recorrido con N2.

La referencia es [Iyer y colaboradores, 2022](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.846219/full), secciones 3 y 6.3. Sus segmentos proyectan un contexto y el de mayor respuesta absoluta modula la activación, conservando el signo. El artículo también usa kWTA y conexiones dispersas. Este componente conserva la selección contextual, pero cambia la modulación y no incorpora esas otras piezas. No reproduce sus experimentos ni acredita sus resultados en finanzas. Se inspeccionaron esas secciones y partes de los resultados y la discusión, no el artículo completo. No se incorporó código de la implementación externa.

## Ecuación e inicialización

Para `h` de forma `[B, H]`, contexto `c` de forma `[B, C]` y pesos `U` de forma `[H, S, C]`:

```text
r[b,i,j] = sum_k U[i,j,k] c[b,k]
j* = primer argmax_j |r[b,i,j]|
d[b,i] = r[b,i,j*]
h'[b,i] = h[b,i] (1 + tanh(a[i]) tanh(d[b,i] / 2))
```

La identidad `tanh(d/2) = 2 sigmoid(d) − 1` permite evaluar la expresión sin restar números cercanos a 0,5. El vector `a` empieza en cero. Por tanto, la salida inicial coincide exactamente con `h`, aunque los segmentos sean distintos. Con contexto nulo también hay paridad para cualquier ganancia finita.

Los segmentos se inicializan con un generador CPU propio, uniforme entre `−1/sqrt(C)` y `1/sqrt(C)`. No se avanza el RNG global. Inicializar también `U` en cero dejaría nula la derivada inicial respecto a `a`. Con la inicialización propuesta, esa derivada es `h tanh(d/2)`, que puede ser distinta de cero cuando hay entrada y contexto informativos. Los gradientes de `U` empiezan en cero por la ganancia nula y aparecen cuando esta cambia. Esto describe el mecanismo, no prueba que aprenda una relación útil.

Cada unidad selecciona un segmento. En los empates gana el primero, con un desempate estable. La derivada de esa selección es local a la región de ganadores. No se afirma diferenciabilidad en las fronteras donde cambia el ganador.

En aritmética real, la amplitud puntual satisface `|h'| ≤ 2 |h|`. Esto no limita el Jacobiano conjunto cuando el contexto depende de `h`, ni demuestra estabilidad de una recurrencia. Una prueba compone ambos argumentos con la misma entrada y obtiene una derivada mayor que dos pese a respetar la cota puntual.

## Contratos y memoria

El operador admite FP32 y FP64 explícitos y rechaza autocast. Representación, contexto y parámetros deben tener el mismo tipo y dispositivo. Las filas son independientes y no se modifican las entradas. Se rechazan NaN, infinitos y desbordamientos de la proyección antes de aplicar las saturaciones. La ganancia inicial nula no oculta una proyección inválida.

Hay `H S C + H` parámetros compartidos y ningún estado temporal ni banco. La proyección cuesta `O(B H S C)` y produce `B H S` respuestas. Se limita a `2^20` el número de elementos de parámetros y, por llamada, el de respuestas. Estos límites no son un presupuesto del RSS, de toda la memoria temporal ni del grafo de autograd. La implementación usa operaciones densas de PyTorch. No se atribuye aceleración a tener pocos gradientes distintos de cero.

El `state_dict` permite recuperar los tensores del operador con la carga restringida de PyTorch. La identidad expone dimensiones, semilla, ecuación y límites. La recuperación de una sesión financiera y la validación de procedencia del contexto siguen pendientes y no quedan resueltas por ese archivo de pesos.

El [recibo del 31 de octubre](../../reports/engineering/active-dendrites-20261031/verification.json) registra 41 casos CPU sin omisiones. Comprueban la referencia escalar, signos y desempates, paridad, gradientes por diferencias finitas, aislamiento del RNG y de las filas, presupuestos, errores y persistencia. Se cubrieron las 53 sentencias y 16 ramas del módulo. Sus cuatro funciones tienen un CRAP máximo de 16. Esto no equivale a comprobar todos los operandos de las condiciones ni garantiza ausencia de defectos.

La primera pasada detectó nueve de diez mutaciones. La eliminación de la cota de ganancia sobrevivió porque el caso no activaba una unidad con una ganancia suficiente. Al añadir ese caso, la repetición detectó el defecto con controles G1/M1/G2 verdes. El recibo conserva ambas pasadas. Una prueba CUDA está preparada, pero se excluyó de la selección CPU porque la GPU estaba ocupada. Esa comprobación sigue pendiente.

## Contexto y conexión financiera

`active_dendrites=None` conserva la identidad de configuración anterior, las claves de sus pesos y la ruta de cálculo sin N2. Al declarar `DendriticRegime`, el predictor utiliza cuatro canales de contexto y añade `H (4 S + 1)` parámetros. Los tres modos de contexto tienen los mismos parámetros iniciales. El [catálogo](../../configs/experiments/active-dendrites-v1.json) identifica la referencia y las tres configuraciones, desactivadas por defecto. Es un catálogo del predictor y no una campaña científica ejecutada.

El productor `dendritic_regime.py` reutiliza `RegimeRule` de `memory/regimes.py`. Esta regla resume la volatilidad y tendencia de cada mercado con las ventanas de 64 sesiones ya recibidas. No estima umbrales con etiquetas ni consulta objetivos financieros. Las cuatro rutas conocidas se codifican con un vector one-hot. El régimen desconocido produce un vector nulo y conserva la paridad del operador. La definición y los mínimos de activos y rendimientos forman parte de la identidad. No es un modelo de estados latentes.

El lote debe declarar la sesión completa. El régimen se calcula por mercado, con todas sus filas, y no depende del orden de activos. Se usa la entrada recibida antes de las transformaciones de rangos de sesión. El productor verifica el corte y la disponibilidad y devuelve ruta, posición fuente y disponibilidad máxima de los inputs del mercado por fila. Una posición de ventana no se presenta como una fecha histórica reconstruida. La procedencia sigue dependiendo del contrato de datos, sin acreditar disponibilidad histórica perfecta de fuentes externas.

La asignación discreta no transporta gradientes hacia los precios a través del contexto. Los codificadores conservan su recorrido de gradientes y los pesos dendríticos siguen siendo parámetros compartidos. La sorpresa asociativa y las políticas de error financiero maduro no se sustituyen por esta clasificación.

`observable` usa la ventana completa. `constant` fija el contexto en la primera ruta conocida y conserva el enmascarado de régimen desconocido. `shuffled_past_prefix` permuta las posiciones de los prefijos anteriores elegibles y utiliza la primera, con una semilla derivada del predictor, mercado y corte. Solo puede elegir un prefijo que ya reúna el mínimo de rendimientos y termine antes de la posición actual. Si no existe, devuelve contexto nulo.

El último modo es un control de muestreo del pasado. No es una permutación global del tramo ni conserva la distribución del régimen contemporáneo. El contraste también cambia recencia y número de rendimientos usados para clasificar. Estas diferencias forman parte de su interpretación. Permutar filas del mismo mercado, que comparten contexto, sería un control equivalente y no se usa.

El contexto se reconstruye desde el lote y no añade historial a los pesos rápidos ni otro estado mutable que recuperar. En CUDA, la implementación copia explícitamente los precios a CPU para reutilizar la referencia NumPy y devuelve el contexto al dispositivo. Ese coste y sus transferencias aún no se han medido. No se atribuye una aceleración al nuevo recorrido.

Las pruebas CPU comprueban predicción y gradientes reales en los cuatro modos del predictor, paridad inicial y del estado MAC, recuperación de la siguiente decisión, separación de identidades, orden de filas, fuentes anteriores a los rangos, fechas futuras y desbordamientos. La configuración se admite en las recetas comunes del predictor. La primera regresión dirigida contiene 235 casos aprobados y seis casos CUDA excluidos. El recibo de esta integración se publica por separado del correspondiente al operador aislado.

Quedan la publicación durable del contexto y sus metadatos, la integración con episodios y mecanismos, las composiciones aún no comprobadas, CUDA y el coste del recorrido completo. `FrozenFinancialConsumer` rechaza N2 mientras no lo registre. El control local de C también se rechaza porque necesita una comparación sobre el operador modulado real. No se cierra N2 ni la issue de N1, N2 y BDH. No hay mejora predictiva medida.
