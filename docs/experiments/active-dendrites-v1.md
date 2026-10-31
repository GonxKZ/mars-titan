# Modulación contextual por segmentos dendríticos, N2

Están implementados el operador neuronal `ActiveDendrites`, su contexto de régimen observable y la conexión con `FinancialPredictor` y `FinancialSession`. N2 se aplica después del contexto compartido de sesión y antes de MAC. El predictor admite el Transformer directo y los controles MAC desactivado, congelado y en línea. La sesión ordinaria se ha comprobado con MAC en línea y M0–M3 y dispone de una medida técnica con fixtures de 64/256 activos. E2 comparte ya ese contexto en sus lecturas ordinarias y regionales. El consumidor de mecanismos comparte también el origen del evento y sus vistas por bloque. No se han realizado entrenamientos ni comparaciones predictivas.

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

La primera pasada detectó nueve de diez mutaciones. La eliminación de la cota de ganancia sobrevivió porque el caso no activaba una unidad con una ganancia suficiente. Al añadir ese caso, la repetición detectó el defecto con G1 y G2 verdes y M1 rojo. El recibo conserva ambas pasadas. Una prueba CUDA está preparada, pero se excluyó de la selección CPU porque la GPU estaba ocupada. Esa comprobación sigue pendiente.

## Contexto y conexión financiera

`active_dendrites=None` conserva la identidad de configuración anterior, las claves de sus pesos y la ruta de cálculo sin N2. Al declarar `DendriticRegime`, el predictor utiliza cuatro canales de contexto y añade `H (4 S + 1)` parámetros. Los tres modos de contexto tienen los mismos parámetros iniciales. El [catálogo](../../configs/experiments/active-dendrites-v1.json) identifica la referencia y las tres configuraciones, desactivadas por defecto. Es un catálogo del predictor y no una campaña científica ejecutada.

El productor `dendritic_regime.py` reutiliza `RegimeRule` de `memory/regimes.py`. Esta regla resume la volatilidad y tendencia de cada mercado con las ventanas de 64 sesiones ya recibidas. No estima umbrales con etiquetas ni consulta objetivos financieros. Las cuatro rutas conocidas se codifican con un vector one-hot. El régimen desconocido produce un vector nulo y conserva la paridad del operador. La definición y los mínimos de activos y rendimientos forman parte de la identidad. No es un modelo de estados latentes.

El lote debe declarar la sesión completa. El régimen se calcula por mercado, con todas sus filas, y no depende del orden de activos. Se usa la entrada recibida antes de las transformaciones de rangos de sesión. El productor verifica el corte y la disponibilidad y devuelve ruta, posición fuente y disponibilidad máxima de los inputs del mercado por fila. Una posición de ventana no se presenta como una fecha histórica reconstruida. La procedencia sigue dependiendo del contrato de datos, sin acreditar disponibilidad histórica perfecta de fuentes externas.

La asignación discreta no transporta gradientes hacia los precios a través del contexto. Los codificadores conservan su recorrido de gradientes y los pesos dendríticos siguen siendo parámetros compartidos. La sorpresa asociativa y las políticas de error financiero maduro no se sustituyen por esta clasificación.

`observable` usa la ventana completa. `constant` fija el contexto en la primera ruta conocida y conserva el enmascarado de régimen desconocido. `shuffled_past_prefix` permuta las posiciones de los prefijos anteriores elegibles y utiliza la primera, con una semilla derivada del predictor, mercado y corte. Solo puede elegir un prefijo que ya reúna el mínimo de rendimientos y termine antes de la posición actual. Si no existe, devuelve contexto nulo.

El último modo es un control de muestreo del pasado. No es una permutación global del tramo ni conserva la distribución del régimen contemporáneo. El contraste también cambia recencia y número de rendimientos usados para clasificar. Estas diferencias forman parte de su interpretación. Permutar filas del mismo mercado, que comparten contexto, sería un control equivalente y no se usa.

El contexto se reconstruye desde el lote y no añade historial a los pesos rápidos. El predictor invocado con una sesión completa puede copiar los precios de CUDA a CPU para reutilizar la referencia NumPy. El recorrido durable prepara el contexto sobre los inputs CPU antes de transferir cada bloque. Su coste CPU con fixtures se recoge al final. Las transferencias CUDA no están medidas y no se atribuye una aceleración a esa organización.

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

El [recibo de sesión](../../reports/engineering/dendritic-session-20261031/verification.json) conserva la regresión de 350 casos CPU sin omisiones, con 53 CUDA excluidos, y sus ampliaciones. Hay 352 identificadores de caso distintos entre las selecciones. Algunas pruebas se repiten o se amplían, por lo que sus recuentos no se suman como casos nuevos. Se detectaron catorce mutaciones adicionales con G1 y G2 verdes y M1 rojo, sin intentos de optimizador.

Los módulos `dendritic_context`, `dendritic_regime` y `dendritic_session` cubren, en conjunto, sus 206 sentencias y 56 ramas. Las 32 funciones cambiadas están medidas. La cobertura de los archivos existentes sigue siendo parcial y el CRAP máximo es 212,82 en el constructor del consumidor de mecanismos. El enlace nativo utilizado es el existente, identificado por su SHA-256. No se atribuye una compilación nueva ni una comprobación CUDA a esos resultados.

Quedan otras composiciones aún no comprobadas, CUDA, el coste representativo del corpus completo y la integración en el estudio científico. El control local C sigue rechazándose porque necesita una comparación sobre el operador modulado real. No se cierra N2 ni la issue de N1, N2 y BDH. No hay mejora predictiva medida.

## Composición con la retención E2

`SpacedTitansBinding` recibe la misma vista dendrítica que la lectura ordinaria, tanto en el calentamiento como al emitir. `neural_options` concentra la comprobación de esa vista, su configuración y el contexto de la instantánea. La identidad del enlace E2 incorpora también las dependencias heredadas de N2. La preparación asigna cuantiles y utilidad por nombre para conservar sus campos separados cuando se amplíe el resultado común.

La política de admisión sigue siendo M1 y el lector sigue siendo K1. No se cambia la definición de utilidad ni se permite acceder a etiquetas pendientes. Se comprueban `utility`, `permuted_utility`, `uniform_frontier`, `recent` y `reservoir`, además del calentamiento sin emisiones y el procesamiento de etiquetas maduras. La recuperación reproduce la predicción siguiente y sus dos registros. Con `reservoir` y capacidad suficiente para la secuencia, E2 conserva exactamente las predicciones y el contexto del mismo consumidor N2 sin esos diagnósticos.

La composición regional se comprueba con puentes en anillo y árbol. Ambos reutilizan el contexto N2 del evento. Se conserva una sola identidad episódica recuperable y la utilidad regional del lector. Estas pruebas usan cuatro activos ficticios y supervisión manual. No demuestran el comportamiento de todas las topologías ni una mejora predictiva. La medida de coste 64/256 que figura después corresponde al código anterior sin E2 y no mide esta composición.

El [recibo de la composición](../../reports/engineering/dendritic-spaced-20261031/verification.json) conserva 494 identificadores de caso distintos. La selección principal pasa 382 casos CPU sin omisiones y excluye 53 CUDA. La regresión de consumidores y adaptadores E2 pasa 125 y excluye cuatro CUDA. Las ampliaciones de 19 y cuatro casos se solapan con las anteriores. No se suman como casos nuevos ni representan una repetición de toda la batería del repositorio.

Se detectaron diez mutaciones distintas con G1 y G2 verdes y M1 rojo y cero intentos de optimizador. Incluyen pérdida de contexto en emisión y calentamiento, campos de cuantiles y utilidad, dependencias heredadas, lector regional, selección de C y aceptación de vistas incompatibles. La comprobación de C usa el modelo anterior sin N2 y no habilita esa combinación. Las dos mutaciones de opciones de una pasada intermedia se repitieron en el lote final y se cuentan una sola vez.

Los dos archivos de producción seleccionados cubren 368 de 406 sentencias y 147 de 178 ramas. Las cinco funciones modificadas están medidas, incluido el constructor cuya guarda se suprime sin añadir líneas. El resumen completa ese caso mediante comparación AST con la base. El CRAP máximo de esas funciones es 9,29. El helper de opciones cubre sus once sentencias y ocho ramas. La preparación congelada y el resto de los archivos conservan cobertura parcial. CUDA y las medidas de coste de esta composición siguen pendientes.

## Propuestas de mecanismos con un origen común

`FrozenMechanismConsumer.begin` recibe `dendritic_plan` cuando el predictor utiliza N2. El plan comprueba configuración, semilla, contrato de entradas, contexto, flujos y corte de toda la cohorte. La propuesta conserva solo su `dendritic_source_id`. Cada llamada a `prepare` recibe una vista del bloque y exige que su origen sea el de esa propuesta. También se comprueban la huella de inputs, las fechas y la configuración mediante `neural_options`. Una vista válida para un bloque de otro evento completo se rechaza aunque coincidan ese bloque y el corte.

`MechanismBinding` calcula una asignación en CPU antes de preparar los bloques. Cada vista se obtiene con el bloque CPU validado, antes de transferir sus tensores al dispositivo. La traza `dendritic_context` se confirma tanto en el control de la generación como en el archivo de mecanismos. Se conserva también durante el calentamiento sin emisiones. En un evento `settlement` sin observaciones vale `None`. Ese valor vacío solo se admite con N2 y sin observaciones. La recuperación comprueba la generación actual a partir de los inputs confirmados.

`MechanismCollector` reutiliza la misma asignación en todas las alternativas y en la emisión. Si se invoca directamente, construye el plan de las filas CPU completas que ya recibe. Rechaza sustituirlo por otro origen después de abrir la recogida. Las alternativas consultan o desactivan el contexto PT4 y cambian la lectura o la calibración, conservando los inputs observables de N2. Desactivar la consulta PT4 no desactiva N2. Esta separación permite contrastar ambos mecanismos sin cambiar silenciosamente su significado.

Los estados propuestos al recoger alternativas se descartan. La actualización que se confirma corresponde a la predicción emitida, con su propia maduración. Las lecturas K1, K2 y K4 siguen siendo otro eje y no cuentan las actualizaciones asociativas de Titans. Los fixtures comprueban también el selector con buffers fijados manualmente. No acreditan haber ajustado una política ni verifican por sí solos la procedencia de un selector entrenado.

El [recibo de mecanismos](../../reports/engineering/dendritic-mechanisms-20261031/verification.json), con código `8ce5470ee`, registra 426 identificadores de caso distintos. La selección principal pasa 422 casos CPU sin omisiones y excluye 102 CUDA. La ampliación pasa 24 casos y la repetición reforzada otros dos, con solapamiento. No se suman como 448 casos nuevos ni como una batería completa del repositorio.

Se comprueban los tres contextos N2, bloques particionados y reordenados, fuentes de otro contrato, semilla o cohorte, planes alterados después de sellarlos, lectura de banco, cinco cuantiles, maduración, recuperación y publicación interrumpida. La recogida se compara apagada, activa y tras recuperar. Conserva exactamente las predicciones, el estado rápido y el auxiliar en los fixtures. Incluye PT4, un selector con buffers manuales y puentes en anillo y árbol. Estos resultados de mecanismos utilizan M1. La comprobación M0–M3 anterior corresponde al consumidor ordinario.

Las 21 mutaciones dirigidas se detectaron con G1 y G2 verdes y M1 rojo, sin intentos de optimizador. El resumen conserva también el rechazo inicial de N2 y el fallo intermedio de un fixture que nombraba una ruta inexistente, corregida a `k4cr` sin modificar producción. Los seis archivos seleccionados cubren 970 de 1.105 sentencias y 370 de 484 ramas. Las catorce funciones modificadas están medidas. El CRAP máximo es 140,35 en `MechanismCollector.add`. El nuevo validador cubre cuatro sentencias y dos ramas. La cobertura de los archivos existentes sigue siendo parcial y el lector adaptativo no se ha ejercitado en esta selección.

El caso CUDA de mecanismos prepara la comparación con CPU FP32, banco poblado y recuperación, sin TF32. Pytest lo ha recogido y no se ha ejecutado. Usa la tolerancia técnica de los fixtures N2, `rtol=atol=2e-5`, sin modificar la del corpus. La GPU seguía ocupada en la comprobación de disponibilidad. Ese caso no cubre la recogida regional en CUDA. Tampoco se han medido el coste integral ni una mejora predictiva de esta composición. El control C y otras operaciones globales sin integración propia siguen pendientes. E2 y los mecanismos tienen integraciones separadas. No se han habilitado todos sus cruces.

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

Las ejecuciones pequeñas y las instrumentadas con cobertura verifican el medidor y no se presentan como medidas representativas de latencia. El [recibo del medidor](../../reports/engineering/dendritic-cost-driver-20261031/verification.json) conserva 30 casos CPU sin omisiones y nueve mutaciones detectadas. El análisis registra 220 de 230 sentencias y 45 de 48 ramas entre el medidor y los fixtures. Los callbacks de rechazo de optimizador y de plazo no se ejercitaron. El CRAP máximo es 33 en el resumen de resultados. Estas métricas no garantizan ausencia de errores.

La [medida 64/256](../../reports/engineering/dendritic-cost-driver-20261031/cost-cpu.json), con código `aa7c01f46`, completó los 40 casos previstos. Se conciliaron diez calentamientos excluidos, treinta casos medidos, fuente común por tamaño, identidad del consumidor real, repetición de estados y paridad de referencia y ganancia cero, también en la decisión posterior a la recuperación. Los pesos y el RNG permanecieron iguales y no hubo intentos de optimizador.

| Activos | Referencia, mediana en s | N2 con a=0 | Observable con a=1 | Prefijo pasado con a=1 | Constante con a=1 |
| --- | --- | --- | --- | --- | --- |
| 64 | 0,647 | 0,669 | 0,697 | 0,549 | 0,588 |
| 256 | 2,267 | 1,979 | 2,096 | 1,946 | 1,998 |

Los rangos se solapan. Con 256 activos, la referencia va de 1,849 a 2,275 s y el observable de 1,991 a 2,257 s. La ganancia cero va de 1,856 a 2,768 s, aunque su mediana sea inferior a la referencia. En 64 activos el control constante alcanza 1,134 s frente a su mediana de 0,588 s. No se establece una aceleración atribuible a N2 ni se elige un modelo por estas diferencias.

La recuperación con 256 activos tiene medianas de 0,764 s para la referencia, 0,746 para ganancia cero, 0,767 para observable, 0,876 para prefijo y 0,751 para constante. El tamaño JSON lógico de la traza, con la convención del medidor, es 11 479 bytes por evento con 64 activos y 44 503 con 256. La referencia no genera esa traza. Los tamaños totales de archivos tras la decisión siguiente constan en cada caso. Son sumas lógicas, no espacio físico asignado ni pico transitorio.

El bucle y sus comprobaciones duraron 285,32 s y el comando externo 290,18 s. Ambos contadores de RSS máximo registraron 922 516 KiB para el proceso completo. No hay pico de RAM por modelo ni ahorro de memoria demostrado. Se observó power-saver y había otras aplicaciones activas. No se aisló el almacenamiento ni se midieron energía, VRAM o transferencias CUDA. Tres muestras por configuración no estiman colas fiables.

El resultado describe el coste de estos fixtures compactos con modelos sin entrenar y un solo indicador macro. No mide el corpus completo ni la utilidad financiera de la intervención. No hay mejora predictiva medida y N2 sigue desactivado por defecto.
