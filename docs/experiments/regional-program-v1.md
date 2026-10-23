# Programas regionales antes de MAC

`pre_mac_regional_program_v1` añade dos calendarios finitos a la [fusión regional existente](regional-fusion-v1.md). Consulta el contexto confirmado de PT4 antes de ejecutar el grafo, conserva esa consulta durante sus pasos y prepara el token final para la siguiente escritura de contexto. Reutiliza los parámetros del operador regional, los codificadores, MAC y la cabeza financiera. La opción está desactivada por defecto.

La implementación permite contrastar el orden y la frecuencia de las operaciones. No se ha entrenado esta variante ni se ha medido una mejora predictiva. El [informe técnico del 22 de octubre de 2026](../../reports/engineering/regional-program-20261022/README.md) reúne las comprobaciones y las medidas del incremento.

## Consulta confirmada y recurrencia

[`FinancialPredictor._program_fusion`](../../src/mars_titan/models/titans/financial.py) recibe los cinco bloques ya proyectados, en el orden precios, noticias, gráficos, fundamentales y macro. Para cada fila, X contiene esos vectores y P indica su presencia. La consulta semilla se calcula antes del programa:

$$
\bar x=\frac{\sum_{j=1}^{5}P_jX_j}{\max(1,\sum_{j=1}^{5}P_j)},\qquad
c=\operatorname{read\_confirmed}(\bar x,f,t,\mathcal C_{<t}).
$$

f identifica el flujo, t es el corte de predicción y C contiene el contexto confirmado anterior. La lectura usa `SessionContext.read_confirmed`, incluida su proyección de salida. No incorpora los tokens pendientes de otros bloques de la misma cohorte. Con la acción de contexto `off`, c es cero y esa fila tampoco prepara una escritura PT4.

Los estados locales empiezan en R⁰ = P ⊙ X y el resumen en S⁰ = 0. Con vectores columna y T como la proyección `summary_to_local`, el contexto que reciben los nodos es:

$$
U^{(0)}=c,\qquad U^{(r)}=c+T S^{(r)}.
$$

Cada paso local normaliza R + U, calcula QKV y aplica atención de producto escalar sobre las aristas y claves presentes. El resultado entra por una actualización residual con la ganancia local del operador. Después del grupo de pasos, el resumen combina su propia normalización con la media de los nodos presentes y se actualiza mediante su MLP y la ganancia de resumen. La consulta c se vuelve a sumar después de cada resumen. Sus bytes no cambian dentro del programa.

Con todos los nodos ausentes, los divisores tienen mínimo uno, los estados locales se anulan y el resumen conserva cero. Las consultas ausentes disponen de una diagonal propia para evitar una atención sin claves admisibles. La salida de un bloque ausente conserva su valor de entrada. Se comprueba la finitud de las entradas y de los intermedios, también antes de aplicar máscaras que podrían ocultar un valor inválido.

LayerNorm, softmax y ganancias menores que uno no demuestran contracción ni estabilidad global. La atención depende del estado y el Jacobiano completo incluye los caminos residuales y las proyecciones aprendidas. Las derivadas se calculan sobre la composición finita ejecutada. No se supone un punto fijo.

La fusión habitual convierte los bloques regionales finales y la presencia en el token financiero. `SessionContext.stage_tokens` prepara ese token una sola vez por flujo seleccionado. MAC recibe el token después de esta fusión. Los estados locales y el resumen se reinician en cada preparación, mientras PT4 conserva su estado confirmado por sesión.

## Calendarios, selección e identidades

[`RegionalExecution`](../../src/mars_titan/models/titans/regional_execution.py) mantiene un registro cerrado de dos programas:

| Programa | Ciclos | Pasos locales por ciclo | Evaluaciones por fila con resumen |
| --- | --- | --- | --- |
| `summary_at_end_v1` | 1 | 4 | 20 locales y 1 resumen |
| `summary_interleaved_v1` | 2 | 2 | 20 locales y 2 resúmenes |

El programa sustituye el calendario por defecto de `RegionalFusionConfig.cycles` y `local_steps` durante esa ejecución. No modifica la configuración almacenada ni crea una segunda copia del operador. Ambos calendarios reutilizan sus pesos y la topología declarada. El ejecutor no añade parámetros ni consume RNG. Sin resumen, ambos realizan los mismos cuatro pasos locales, una equivalencia que se contrasta en las pruebas.

`RegionalExecutionConfig.selection="fixed"` elige el programa indicado por `program`. La regla alternativa `context_norm_v1` requiere PT4 y un `context_threshold` explícito. Elige `summary_interleaved_v1` cuando ‖c‖ supera el umbral y `summary_at_end_v1` en los demás casos, incluida la igualdad. Es una regla programada sobre información disponible antes de MAC. No recibe etiquetas futuras ni el error financiero de una predicción que todavía no ha madurado. La elección discreta no recibe un gradiente continuo ficticio. El contexto utilizado dentro del recorrido elegido conserva sus gradientes.

Los cuatro pasos locales no son K. K cuenta refinamientos del lector episódico después de MAC. J cuenta las iteraciones del puente regional episódico, con sus propios límites. Las actualizaciones de memoria neuronal de Titans también tienen su contrato separado. Estos programas no cambian el significado de C y M en CM-v1, que conserva su identidad y permanece desactivada por defecto.

`FinancialConfig.regional_execution` exige una `RegionalFusionConfig` explícita. El consumidor durable es `FrozenMechanismConsumer`, con un `MechanismCatalog` que declara el mismo programa. El catálogo conserva sus 36 acciones de contexto, ruta y calibración. El programa es un eje distinto y no reinterpreta columnas de una política anterior. Configuraciones, checkpoints, consumidores y trazas incompatibles se rechazan.

Con `regional_execution=None` se mantiene el recorrido anterior y se omiten los campos nuevos y el índice regional. Si hay PT4, ese recorrido fusiona primero y consulta y suma su contexto después de la fusión. El programa activo realiza la consulta anterior y la introduce en el operador regional. Por tanto, activarlo define otra variante, aunque reutilice los pesos. Las huellas de código identifican la implementación y no autorizan migrar silenciosamente runtimes científicos históricos.

La cohorte lógica admite hasta 8192 filas. Cada bloque físico respeta el máximo del predictor y del consumidor, que no supera 4096. La validación de las trazas agregadas conserva el límite lógico y la exportación de cada bloque conserva el físico. Partir una cohorte no crea otro presupuesto ni otro snapshot confirmado.

## Reserva, emisión y recuperación

La sesión exige `regional_work_limits` con exactamente nueve recursos y enteros no negativos. `regional_max_attempts` vale dos por defecto y admite un máximo de 32. Los límites y el número de intentos forman parte del contrato. El enlace rechaza C activo, cuyo trabajo adicional requeriría otra reserva.

Sea N el número de filas y R la cota de preparaciones del núcleo. Sin colección, R = N. Con colección, R = 3N si existe PT4 y R = 2N si no existe. La colección prepara las alternativas y la emisión vuelve a preparar su ruta desde el prefijo común. Cada preparación avanza MAC una vez por observación cuando ese módulo existe. Solo el estado de la emisión se confirma en la generación nativa. Los estados de las alternativas no se publican.

| Recurso | Cota conservadora de la cohorte |
| --- | --- |
| `core_observations` | R preparaciones del núcleo |
| `selector_decisions` | R selecciones regionales por fila preparada |
| `local_node_evaluations` | 20R evaluaciones locales |
| `summary_evaluations` | Hasta 2R resúmenes, cero con resumen retirado |
| `context_queries` | Hasta R consultas si existe PT4, cero sin él |
| `mac_observations` | R si existe MAC, incluido `mac_disabled`, cero en `transformer_direct` |
| `reader_refinements` | Hasta 38N con colección y el límite del catálogo sin ella, cero sin lector |
| `reader_searches` | Hasta 22N con colección y el límite del catálogo sin ella, cero sin lector o con `no_bank` |
| `context_commits` | Hasta una escritura por mercado con PT4 |

Los 38 refinamientos cubren hasta 34 de colección y cuatro de emisión. Las 22 búsquedas mantienen un margen conservador sobre las cotas del lector. Los límites de profundidad del puente episódico, bytes y episodios siguen vigentes y se comprueban antes del primer MAC. Estas unidades no se convierten en FLOPs ni sustituyen la medida de RAM, VRAM o tiempo total.

El calentamiento prepara una vez el núcleo y no ejecuta colección ni lectores. Un `settlement` reserva cero trabajo predictivo en estos nueve recursos, aunque pueda aplicar feedback maduro. Puede compartir el corte de una decisión anterior. El ejecutor nativo rechaza otro `settlement` consecutivo en ese mismo corte antes de reservar.

[`RegionalWorkLedger`](../../src/mars_titan/memory/regional_work.py) publica la reserva en `Boundary.before_predictions`, después de la admisión nativa y antes del hook y del cálculo. Usa los artefactos sellados y el bloqueo de salida existentes. El índice `regional-work.json` se escribe mediante temporal, `fsync`, reemplazo atómico y `fsync` del directorio. Un fallo o una reapertura no devuelve las unidades reservadas ni reinicia el ordinal del intento.

Control, bundle e historial enlazan el mismo cuerpo `regional_work`, con ID, referencia, cohorte y recibos completos. Antes de podar se concilian latest y la reserva. Si la cabeza nativa ya avanzó, la apertura recupera esa confirmación sin repetir MAC, contexto ni feedback. Una confirmación contable aislada no acredita publicación nativa ni permite renovar cuota.

El índice conserva dos cohortes confirmadas y cualquier cohorte activa. La compactación traslada el prefijo acumulado al resumen archivado y mantiene los recibos completos dentro del historial confirmado. Los blobs individuales antiguos pueden retirarse cuando dejan de ser referencias vivas. La apertura verifica que el bloque del ancla contiene el evento, la generación y la reserva declarados. La ausencia o contradicción de una raíz necesaria se rechaza antes de podar los artefactos de sesión.

Las cuotas son cargos conservadores. `observed_units` permanece separado y no se rellena con esas cifras. Las trazas registran fases locales y de resumen invocadas, programa, norma contextual y huellas del contrato y del operador. Permiten comprobar el calendario y sus contadores. No autentican el contenido neuronal original del contexto ni certifican kernels, FLOPs o trabajo desconocido tras un corte. Las huellas locales tampoco autentican una sustitución coordinada de todo el archivo por una copia histórica coherente.

## Comprobaciones y alcance pendiente

La revisión técnica incluye un pase de 70 pruebas del ledger y otro de 30 pruebas de sesión e integridad. El fixture de retención conserva 600 eventos del ejecutor, con cinco intentos abandonados y reaperturas cada cien eventos. Después de retirar blobs antiguos se leen y verifican los 600 eventos históricos y sus 605 recibos completos. Este caso comprueba almacenamiento, continuidad y cargos. Las pruebas del predictor financiero completo pertenecen al grupo separado de sesión, con casos CPU y CUDA.

Las pruebas del componente contrastan ambos calendarios con una referencia independiente, retirada del resumen, máscaras, orden y partición de filas, RNG, gradientes y Jacobianos pequeños. Las de integración comprueban lectura anterior, staging del token final, checkpoint y siguiente emisión, cambios después del sellado, cuotas insuficientes y cortes antes del cálculo o después de `committed`. El [informe técnico](../../reports/engineering/regional-program-20261022/README.md) recoge el alcance concreto de cada pase y sus límites.

Desde la raíz del repositorio, estas selecciones reproducen los grupos ejecutados. Las pruebas nativas necesitan `MARS_TITAN_EPISODIC_NATIVE` configurado con el artefacto compilado, según la [documentación nativa](../../native/README.md). Los casos CUDA necesitan `cuda:0`. `MARS_TITAN_REQUIRE_NATIVE=1` y `MARS_TITAN_REQUIRE_CUDA=1` permiten exigir esas rutas en la verificación.

```bash
uv run --no-sync pytest -q \
  tests/models/titans/test_financial_regional_program.py \
  tests/models/titans/test_regional_execution.py

uv run --no-sync pytest -q tests/memory/test_regional_work_reservation.py

uv run --no-sync pytest -q \
  tests/memory/test_regional_program_session.py \
  tests/memory/test_regional_session_integrity.py
```

[#709](https://github.com/GonxKZ/mars-titan/issues/709) sigue pendiente. Estos calendarios no incorporan todavía los grupos y la raíz jerárquica anteriores a MAC ni un selector aprendido de programas. El ciclo exterior de propuestas y revisión de mecanismos de [#711](https://github.com/GonxKZ/mars-titan/issues/711) también sigue separado. Los antecedentes recogidos en la [revisión de recurrencia regional](../research/regional-recurrence-extensions.md) motivan controles y preguntas, sin acreditar una reproducción biológica ni novedad por usar un grafo o un resumen.

## Matriz técnica de modos de memoria

[`regional_program_matrix_cuda.py`](../../benchmarks/regional_program_matrix_cuda.py) cruza las seis configuraciones de la declaración con los cuatro modos existentes de memoria y dos opciones de contexto. Son 48 casillas. El constructor real de `FinancialConfig` admite 26 y rechaza 22 antes de inicializar CUDA. PT4 exige `mac_online` y la selección por norma contextual exige PT4. Los rechazos se conservan con su motivo y no cuentan como pruebas omitidas.

El [informe de la matriz](../../reports/engineering/regional-program-matrix-20261022/README.md) recoge sus 208 condiciones ejecutadas, paridad, costes, mutaciones y límites de cobertura.

`transformer_direct` no construye MAC. `mac_disabled` conserva el módulo con lectura y actualización de memoria desactivadas. `mac_frozen` permite lectura sin actualización asociativa y `mac_online` permite actualizar sus pesos rápidos. Esta matriz utiliza el mismo codificador de fixture. No compara cuatro codificadores distintos ni sustituye la comparación entre GRU, Transformer y Titans.

El protocolo técnico usa la cabeza de cuantiles existente, H = 32, semilla 42, FP32 y FP64 y lotes de 1 y 256 filas. Los cuatro estratos se ejecutan por separado. Cada uno recorre las 26 configuraciones en orden original y en orden global inverso, con dos calentamientos y cinco muestras por registro. El conjunto previsto contiene 208 registros, 416 preparaciones de prefijo y 1.456 preparaciones entre calentamientos y medidas. Cada condición reconstruye sus dos observaciones previas y utiliza su propio estado rápido y contexto confirmado.

Las entradas y los tensores de parámetros con nombres comunes se contrastan exactamente. Los parámetros adicionales de MAC y PT4 se contabilizan aparte. Las 40 parejas sin resumen deben conservar todos los tensores del resultado, excluida la traza regional, dentro del mismo modo, contexto, precisión, tamaño y orden. No se exige igualdad entre arquitecturas diferentes. La matriz tampoco identifica por separado consulta y calendario cuando compara PT4 posterior a la fusión con un programa que consulta antes.

Sin `--run`, el guion imprime el protocolo sin importar Torch ni crear una salida. La ejecución exige un estrato explícito y un directorio nuevo fuera del repositorio. Desde su raíz:

```bash
set -e
matrix_artifacts=$(mktemp -d)
for dtype in float32 float64
do
  for batch in 1 256
  do
    timeout --signal=TERM --kill-after=10s 300s \
      uv run --no-sync python benchmarks/regional_program_matrix_cuda.py \
      --run --dtype "$dtype" --batch "$batch" \
      --output "$matrix_artifacts/$dtype-$batch"
  done
done
```

El límite interno es de 240 segundos por estrato. La ronda documentada dispone de 300 segundos de supervisión por proceso y 1.500 segundos técnicos en total. El asignador de Torch se limita a 512 MiB y se exige una reserva global libre de 256 MiB. La caché de tensores CPU tiene un máximo de 128 MiB, distinto del RSS del runtime. Se conserva una sola instancia de predictor GPU por condición. Las cinco muestras describen mínimo, mediana, máximo y caudal, sin estimar percentiles de cola ni intervalos de confianza. La construcción, los prefijos y las comprobaciones se registran fuera de la latencia de `prepare`.

La guarda local mantiene bloqueada la campaña de aprendizaje. Las verificaciones históricas registradas no la abren automáticamente. Estas comprobaciones técnicas no ejecutan pasos de optimizador ni constituyen un entrenamiento, piloto o evaluación científica. El test final de 2024 permanece cerrado.
