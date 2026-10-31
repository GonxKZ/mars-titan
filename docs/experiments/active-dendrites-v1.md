# Modulación contextual por segmentos dendríticos, N2

Están implementados el operador neuronal `ActiveDendrites`, su contexto de régimen observable y la conexión con `FinancialPredictor` y `FinancialSession`. N2 se aplica después del contexto compartido de sesión y antes de MAC. El predictor admite el Transformer directo y los controles MAC desactivado, congelado y en línea. La sesión durable se ha comprobado con MAC en línea y M0–M3. El consumidor de mecanismos y E2 todavía rechazan N2. No se han realizado entrenamientos, comparaciones predictivas ni medidas de coste del recorrido con N2.

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

El `state_dict` permite recuperar los tensores del operador con la carga restringida de PyTorch. La identidad expone dimensiones, semilla, ecuación y límites. La recuperación de una sesión financiera utiliza además sus estados rápidos, inputs y control confirmado, descritos a continuación. Un archivo de pesos aislado no contiene esa cronología.

El [recibo del 31 de octubre](../../reports/engineering/active-dendrites-20261031/verification.json) registra 41 casos CPU sin omisiones. Comprueban la referencia escalar, signos y desempates, paridad, gradientes por diferencias finitas, aislamiento del RNG y de las filas, presupuestos, errores y persistencia. Se cubrieron las 53 sentencias y 16 ramas del módulo. Sus cuatro funciones tienen un CRAP máximo de 16. Esto no equivale a comprobar todos los operandos de las condiciones ni garantiza ausencia de defectos.

La primera pasada detectó nueve de diez mutaciones. La eliminación de la cota de ganancia sobrevivió porque el caso no activaba una unidad con una ganancia suficiente. Al añadir ese caso, la repetición detectó el defecto con controles G1/M1/G2 verdes. El recibo conserva ambas pasadas. Una prueba CUDA está preparada, pero se excluyó de la selección CPU porque la GPU estaba ocupada. Esa comprobación sigue pendiente.

## Contexto y conexión financiera

`active_dendrites=None` conserva la identidad de configuración anterior, las claves de sus pesos y la ruta de cálculo sin N2. Al declarar `DendriticRegime`, el predictor utiliza cuatro canales de contexto y añade `H (4 S + 1)` parámetros. Los tres modos de contexto tienen los mismos parámetros iniciales. El [catálogo](../../configs/experiments/active-dendrites-v1.json) identifica la referencia y las tres configuraciones, desactivadas por defecto. Es un catálogo del predictor y no una campaña científica ejecutada.

El productor `dendritic_regime.py` reutiliza `RegimeRule` de `memory/regimes.py`. Esta regla resume la volatilidad y tendencia de cada mercado con las ventanas de 64 sesiones ya recibidas. No estima umbrales con etiquetas ni consulta objetivos financieros. Las cuatro rutas conocidas se codifican con un vector one-hot. El régimen desconocido produce un vector nulo y conserva la paridad del operador. La definición y los mínimos de activos y rendimientos forman parte de la identidad. No es un modelo de estados latentes.

El lote debe declarar la sesión completa. El régimen se calcula por mercado, con todas sus filas, y no depende del orden de activos. Se usa la entrada recibida antes de las transformaciones de rangos de sesión. El productor verifica el corte y la disponibilidad y devuelve ruta, posición fuente y disponibilidad máxima de los inputs del mercado por fila. Una posición de ventana no se presenta como una fecha histórica reconstruida. La procedencia sigue dependiendo del contrato de datos, sin acreditar disponibilidad histórica perfecta de fuentes externas.

La asignación discreta no transporta gradientes hacia los precios a través del contexto. Los codificadores conservan su recorrido de gradientes y los pesos dendríticos siguen siendo parámetros compartidos. La sorpresa asociativa y las políticas de error financiero maduro no se sustituyen por esta clasificación.

`observable` usa la ventana completa. `constant` fija el contexto en la primera ruta conocida y conserva el enmascarado de régimen desconocido. `shuffled_past_prefix` permuta las posiciones de los prefijos anteriores elegibles y utiliza la primera, con una semilla derivada del predictor, mercado y corte. Solo puede elegir un prefijo que ya reúna el mínimo de rendimientos y termine antes de la posición actual. Si no existe, devuelve contexto nulo.

El último modo es un control de muestreo del pasado. No es una permutación global del tramo ni conserva la distribución del régimen contemporáneo. El contraste también cambia recencia y número de rendimientos usados para clasificar. Estas diferencias forman parte de su interpretación. Permutar filas del mismo mercado, que comparten contexto, sería un control equivalente y no se usa.

El contexto se reconstruye desde el lote y no añade historial a los pesos rápidos. El predictor invocado con una sesión completa puede copiar los precios de CUDA a CPU para reutilizar la referencia NumPy. El recorrido durable prepara el contexto sobre los inputs CPU antes de transferir cada bloque. Todavía no se han medido los costes ni las transferencias de estos recorridos y no se atribuye una aceleración a esa organización.

El [recibo de integración](../../reports/engineering/dendritic-regime-20261031/verification.json) registra 240 casos CPU aprobados y seis casos CUDA excluidos. Comprueban predicción y gradientes reales en los cuatro modos del predictor, FP32 y FP64, paridad inicial y del estado MAC, recuperación de la siguiente decisión, separación de identidades, orden de filas, fuentes anteriores a los rangos, fechas futuras y desbordamientos. La configuración se admite en las recetas comunes del predictor. Los casos incluyen los 41 del operador anterior y no se suman como pruebas nuevas independientes.

Se detectaron quince mutaciones de integración y se repitieron los diez casos del operador. No son veinticinco defectos nuevos. Las repeticiones conservan las fuentes y los controles verdes. Una de las mutaciones verifica el diagnóstico explícito del consumidor. La lista de módulos también rechaza N2, por lo que ese caso no demuestra que se llegase a aceptar una configuración insegura.

El productor cubre sus 90 sentencias y 22 ramas y el operador sus 53 sentencias y 16 ramas. La cobertura de los archivos existentes seleccionados sigue siendo parcial. Las diecinueve funciones cambiadas están medidas, sin funciones nuevas sin medir. El CRAP máximo de ese conjunto es 272,28 en la preparación financiera. No se presenta la selección como una comprobación de toda la clase ni como garantía de ausencia de errores.

## Contexto común en la sesión durable

`TitansBinding` prepara una sola asignación `DendriticPlan` sobre todas las filas del evento, después de verificar y conservar sus inputs. La clasificación reutiliza la misma referencia NumPy del predictor. No reúne toda la cohorte en una inferencia ni aumenta el tamaño de bloque configurado.

Cada `DendriticView` se vincula al bloque CPU validado mediante su huella completa de inputs. Antes de conceder esa vista se comprueban los precios, contrato, identificadores y fechas frente al evento. La vista guarda rutas y referencias inmutables y crea un tensor propio para el dispositivo. El consumidor comprueba también que pertenece al mismo `context_id` de la instantánea episódica. El predictor solo permite fragmentar una sesión con esa vista verificada cuando no hay otra operación que requiera el grupo completo.

La traza `dendritic_event_context_v1` incluye versión, nombres de columnas, configuración, contrato, contexto y una fila por activo. Conserva identificador de flujo y muestra, corte, disponibilidad, SHA-256 de precios, régimen, slot fuente y disponibilidad máxima de los inputs del mercado. Se publica dentro del control de la misma generación. La comprobación del bundle reconstruye la asignación a partir de sus inputs confirmados y rechaza una traza distinta.

`FinancialSession.dendritic_event()` devuelve una copia de esa traza para la generación actual. Devuelve `None` antes del primer evento o tras una generación de settlement sin observaciones. La traza sigue la retención del control existente y esta API no ofrece un archivo histórico completo. Las vistas son transitorias y no se guardan como objetos Python ni grafos de autograd.

El calentamiento puede actualizar la memoria neuronal con su contexto y no emite predicciones. Un settlement no calcula otro contexto. La recuperación comprueba la fuente confirmada y repite la siguiente decisión. Tras una interrupción debe repetirse el mismo evento antes de avanzar, conforme al contrato de transacciones inmutables del ejecutor.

Las pruebas nativas utilizan cuatro activos ficticios, supervisión manual y escalas manuales en M3. Comprueban bloques de uno y cuatro activos, lectura de un banco no vacío, etiquetas maduras, recuperación, calentamiento y publicación interrumpida. Con ganancia inicial nula, la secuencia nativa conserva exactamente las predicciones de la referencia con el mismo tamaño de bloque. No son resultados predictivos reales.

El [recibo de sesión](../../reports/engineering/dendritic-session-20261031/verification.json) conserva la regresión de 350 casos CPU sin omisiones, con 53 CUDA excluidos, y sus ampliaciones. Hay 352 identificadores de caso distintos entre las selecciones. Algunas pruebas se repiten o se amplían, por lo que sus recuentos no se suman como casos nuevos. Se detectaron catorce mutaciones adicionales con controles G1/M1/G2 verdes, sin intentos de optimizador.

Los módulos `dendritic_context`, `dendritic_regime` y `dendritic_session` cubren, en conjunto, sus 206 sentencias y 56 ramas. Las 32 funciones cambiadas están medidas. La cobertura de los archivos existentes sigue siendo parcial y el CRAP máximo es 212,82 en el constructor del consumidor de mecanismos. El enlace nativo utilizado es el existente, identificado por su SHA-256. No se atribuye una compilación nueva ni una comprobación CUDA a esos resultados.

Quedan el consumidor de mecanismos, E2 y otras composiciones aún no comprobadas, CUDA, el coste del recorrido completo y la integración en el estudio científico. El control local C sigue rechazándose porque necesita una comparación sobre el operador modulado real. No se cierra N2 ni la issue de N1, N2 y BDH. No hay mejora predictiva medida.

## Medidor técnico de coste

`benchmarks.dendritic_session_cost` prepara un contraste con referencia sin N2, parámetro dendrítico `a=0`, observable con `a=1`, prefijo pasado con `a=1` y constante con `a=1`. Se fija el parámetro previo a `tanh`, sin ajustarlo a datos. No es una comparación de modelos entrenados. El medidor reutiliza los mismos fixtures que las pruebas de sesión.

El protocolo de 64 y 256 activos usa un calentamiento excluido y tres repeticiones por modelo, con orden rotatorio. Son 40 casos, diez de calentamiento y treinta medidos. Se comparte fuente por tamaño, bloque de 64, CPU FP64, un hilo, H32, M1, capacidad episódica 64 y lector K1. La tercera decisión lee un banco no vacío. El intervalo de `step` incluye persistencia y comprobaciones. La recuperación mantiene el modelo congelado residente y se verifica la decisión siguiente. No incluye carga en frío de sus pesos ni codificación de noticias e imágenes originales.

Se exige paridad de predicciones, tensores rápidos, contadores y episodios con procedencia entre la referencia y la ganancia cero. Se excluyen los identificadores del runtime, que dependen de la identidad del modelo. Cada fila registra el consumidor realmente utilizado. Los pesos y el RNG deben permanecer iguales. Las otras variantes no tienen que coincidir con la referencia, pero sí reproducirse entre sus repeticiones.

El comando sin `--run` solo imprime el protocolo. Para ejecutarlo se necesita el enlace nativo verificado, un `TMPDIR` en disco y una salida nueva fuera del checkout. Los casos propios terminados se retiran tras cerrar sus sesiones. Se conservan fuentes, protocolo, mediciones y casos fallidos.

```bash
uv --no-cache run --no-sync python -m benchmarks.dendritic_session_cost \
  --repository "$PWD" \
  --native "${MARS_TITAN_EPISODIC_NATIVE:?}" \
  --output "$(mktemp -d "${TMPDIR:?}/dendritic-cost.XXXXXX")/run" \
  --sizes 64 256 --warmup 1 --repeats 3 --seconds 1800 --run
```

Las ejecuciones pequeñas y las instrumentadas con cobertura verifican el medidor y no se presentan como medidas representativas de latencia. El RSS del proceso es global, sin atribuir un pico a cada modelo. No se miden energía, VRAM ni transferencias CUDA. Tres repeticiones no permiten estimar colas fiables. La comparación 64/256 todavía no se presenta como completada.
