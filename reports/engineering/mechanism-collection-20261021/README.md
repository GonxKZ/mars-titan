# Colección de alternativas y pérdidas maduras

`MechanismCollector` calcula alternativas de contexto, ruta episódica y calibración desde un prefijo común del predictor. `CollectedMechanisms` conserva sus predicciones y procedencia sin etiquetas. `MechanismSupervision` recibe resultados parciales cuando están disponibles y calcula pérdidas por mercado y sesión. Son APIs explícitas de desarrollo. El predictor y el selector no se han entrenado en este incremento.

El productor y la recepción están implementados y comprobados por separado. Se ha probado su transporte con `SessionArtifacts`. Queda pendiente incorporar sus referencias a las generaciones, pendientes, publicación, poda y recuperación de `FinancialSession`. Un paquete sellado no demuestra por sí solo que se publicase antes de conocer una etiqueta. La campaña completa y el test final siguen cerrados. No hay una mejora predictiva medida ni evidencia de superar el SOTA.

Los commits de código son `ce8ccfbb5`, `883dca869` y `fd21b78d1`. El primero corrige el retorno del selector por agotamiento de bytes de PT2. El segundo incorpora el productor y las pérdidas maduras. El tercero normaliza los escalares del feedback antes de sellarlos.

## Alternativas condicionadas al mismo prefijo

El constructor recibe el consumidor congelado, su estado auxiliar anterior, las filas admitidas, corte, contexto, madurez explícita por fila y snapshot opcional. Extrae la observación antes de MAC. Los bloques se procesan secuencialmente y deben cubrir exactamente los flujos declarados. Cada bloque conserva la huella de su estado rápido de partida. Se comprueban las huellas del contexto, calibrador y snapshot durante la colección, además de los bytes de las entradas. Un hash declarado no sustituye la validación de esos bytes.

Para cada contexto admisible se ejecuta el predictor desde el mismo estado rápido anterior. Con lector se reutilizan las seis rutas y sus prefijos mediante `collect_routes`, que necesita nueve refinamientos. Si ese recorrido no cabe en el presupuesto de trabajo del lector, se ejecutan las rutas admisibles por separado. Las seis rutas individuales requieren diecisiete refinamientos. Las propuestas no publican pesos rápidos, contexto ni calibrador. El productor contabiliza el cálculo adicional y no lo presenta como una sola inferencia de despliegue.

Cada alternativa cambia una acción de un mercado y conserva las acciones ancla de los demás. La calibración se propone sobre la cohorte completa para respetar reservas compartidas. Un retorno provocado en otro mercado queda registrado. El conjunto contiene 36 alternativas condicionales por mercado. No enumera las `36^G` decisiones conjuntas de `G` mercados ni estima por sí mismo la utilidad de toda la trayectoria autónoma del selector.

Las acciones no admisibles conservan su motivo y `quantiles=None`. Una acción que acaba en `pending_calibration_bytes` guarda los cuantiles K1 realmente emitidos y el estado `fallback`. Esa salida no se atribuye como una observación correcta de la solicitud online o K4. La alternativa sin calibración del mismo contexto y ruta se conserva de forma independiente y puede aportar una pérdida puntual válida. Los fallos inesperados abortan la propuesta, sin producir pérdidas cero o un paquete parcial.

## Formato, recepción y objetivos

El paquete incluye filas y madurez, observación original FP64, catálogo, disponibilidad, anclas, estados anteriores, precisión y dispositivo, alternativas, retornos y trabajo. Guarda JSON canónico inmutable con SHA-256 y un límite de 48 MiB. El formato admite hasta 8.192 filas, mientras el productor respeta el límite del padre, que actualmente no supera 4.096. Los bloques también respetan el límite del consumidor. Estas cotas de formato no representan el pico transitorio de RAM. El resultado y el constructor terminado no retienen los arrays de entrada.

`record()` devuelve bytes y huella que pueden guardarse en un artefacto existente. `restore()` verifica integridad, esquema, identidades, corte, medianas compartidas entre calibraciones, retornos K1, bloques y trabajo. No autentica un host capaz de modificar código y archivos. La publicación previa al feedback deberá conciliarse con el registro nativo cuando se complete esa integración.

La supervisión es otro estado. Recibe `CohortOutcome` por grupo, predicción y muestra. Una etiqueta exige madurez conocida y disponibilidad posterior a ella y no posterior al reloj de recepción. Una exclusión o finalización necesita evidencia y no contiene etiqueta. Los resultados desconocidos, repetidos, futuros o con números no finitos se rechazan sin cambiar el estado anterior. Los escalares NumPy admitidos se convierten a `float` e `int` conservando su valor. Un `np.float32(0.1)` conserva `0.10000000149011612`, sin fingir precisión adicional.

Las pérdidas usan FP64 y dispositivo explícito. Primero promedian los activos maduros de cada mercado y sesión. Devuelven doce MAE puntuales, 36 interval scores, cobertura y anchura, con ganancias firmadas frente a `off/k1/none`. Las calibraciones comparten la mediana de su contexto y ruta. El [núcleo de utilidades](../mechanism-policy-20261021/README.md) documenta el interval score de los intervalos 80 % y 95 %. El promedio de magnitudes no negativas se escala para evitar desbordar una suma cuyo resultado medio sí sea representable. Los errores individuales no representables siguen produciendo un error.

Pendientes, alternativas no observadas y retornos no atribuibles permanecen como `None`. Una exclusión no elimina la pérdida válida de otro activo. Cada resultado declara las muestras maduras, los denominadores y los estados de los candidatos. La ponderación temporal y entre mercados corresponde al protocolo externo. Estos activos y alternativas no se cuentan como réplicas temporales independientes.

## Comprobaciones

La [verificación final](verification.json) pasó 423 casos, 93 CUDA, sin fallos, errores ni omisiones. Incluye 45 casos nuevos del productor y la supervisión, ocho CUDA, y diez casos adicionales del retorno de PT2. No es toda la suite del proyecto. Comprueba comparación de las 36 alternativas con controles fijos desde un prefijo con memoria, banco poblado, FP32/FP64, partición y orden invertido, presupuestos, retornos cruzados, corrupción, liberación de arrays, recepción parcial y restauración.

La [mutación dirigida final](mutations.json) detectó 22/22 mutantes. Las referencias inicial y restaurada pasaron 37 casos CPU cada una. Incluye sustituir datos o estados, modificar otra ancla, retener entradas, omitir huellas, ocultar trabajo, aceptar futuro o doble recepción, cambiar el signo o los coeficientes del interval score, fabricar pérdidas cero, atribuir retornos a la acción solicitada y borrar pérdidas válidas por una exclusión. Es evidencia sobre esos fallos elegidos, no sobre todas las mutaciones posibles.

La [cobertura y CRAP](quality.json) registra 1.124/1.194 sentencias y 350/420 ramas en ocho módulos y 75 funciones. Se usaron coverage.py 7.16.1 con ramas y Radon 6.0.1. CRAP sigue `C² × (1 − cobertura de sentencias)³ + C`. Su máximo es 55,92 en `FrozenMechanismConsumer.begin`. La validación del paquete tiene CRAP 48,09 y `MechanismCollector.add` 40,80. Quedan rechazos sin recorrer y complejidad visible. Las puntuaciones no prueban ausencia de errores.

Los recibos conservan los fallos reproducidos y sus correcciones. La versión anterior a la normalización de escalares pasó 422 casos y 21 mutantes dirigidos. Se conserva aparte y no se suma como evidencia independiente de la versión final. La primera medida también se conserva fuera de Git, identificada por sus huellas. No se cambiaron datos, cargas, tolerancias o reglas para favorecer el resultado posterior.

## Coste observado en CUDA

El [protocolo](cost-protocol.json) se guardó antes de medir. El [benchmark](benchmark.py) compara una referencia de 36 ejecuciones fijas, la colección con validación y sellado y esa colección seguida de recepción de etiquetas manuales y métricas. La referencia no extrae la observación ni sella el paquete. Es un control sencillo que repite el consumidor, no un ejecutor óptimo de todas las alternativas.

Se usaron fixtures US, contexto inicial y banco vacío, dimensión oculta 32, FP64 y `cuda:0` en una RTX 4070 Laptop. Los precios tienen forma `[64,5]`, noticias y gráficos anchura 2 y fundamentales y macro anchura 3. Son representaciones pequeñas, con un concepto fundamental y un indicador macro, que no representan todo el corpus. Hubo un calentamiento y cuatro medidas por modo y tamaño, con orden rotativo. No se ejecutaron otras cargas propias de pruebas, mutación o perfilado durante las medidas, tampoco en CPU. Las aplicaciones del escritorio permanecieron activas.

| Activos | Recorrido | Mediana (ms) | p95 (ms) |
| --- | --- | ---: | ---: |
| 64 | 36 controles fijos | 733,34 | 767,91 |
| 64 | Colección y sellado | 126,27 | 129,30 |
| 64 | Colección, recepción y métricas | 135,41 | 240,13 |
| 256 | 36 controles fijos | 1058,66 | 1186,15 |
| 256 | Colección y sellado | 301,88 | 305,80 |
| 256 | Colección, recepción y métricas | 444,27 | 508,62 |

Las matrices de los 36 candidatos tienen exactamente el mismo SHA-256 entre modos y repeticiones. La diferencia absoluta máxima observada fue cero. Con 256 activos, el trabajo del núcleo pasa de 9.216 a 512 observaciones y los refinamientos de 26.112 a 4.608. La [comparación](comparison.json) y los [datos brutos](cost.json) conservan tiempos, dispersión, p99 empírico, caudal y trabajo. Los tiempos son inclusivos y no se suman. Cuatro repeticiones no estiman con precisión las colas, como muestra la dispersión del tercer recorrido con 64 activos.

El paquete ocupa 261.496 bytes con 64 activos y 992.248 con 256. Con 256, el pico asignado por Torch fue 110.911.488 bytes en la referencia y 102.487.552 en la colección. Es una medida del asignador del proceso, no de toda la VRAM del dispositivo. El pico RSS global dentro de Python fue 1.526.996.992 bytes y no permite atribuir ahorro de RAM a un modo. El [recibo del proceso](process-cost.json) separa 15,40 segundos internos y 19,12 externos. No se midió energía.

La medida incluye transferencias de los bloques y huellas de estados. Excluye corpus real, codificadores externos, publicación de una generación nativa y escritura a disco. Mide obtener alternativas de desarrollo. No sustituye el [coste de la sesión financiera](../mechanism-selection-20261021/README.md#coste-integrado) ni demuestra menor latencia del predictor desplegado o menor error financiero.

Para repetir la selección técnica en un entorno compatible:

```bash
PYTHONPATH=src:. uv run --no-sync pytest -q tests/memory/test_mechanism_collection.py tests/memory/test_mechanism_supervision.py tests/memory/test_mechanism_selection.py
```

El enlace nativo debe estar configurado para la prueba de transporte. La medida exige CUDA y un archivo de salida nuevo:

```bash
PYTHONPATH=src:. uv run --no-sync python reports/engineering/mechanism-collection-20261021/benchmark.py --output /tmp/mars-mechanism-collection-cost.json --rows 64 256 --warmup 1 --repetitions 4
```

Las [huellas de artefactos externos](external-artifacts.json) identifican logs, JUnit, cobertura, recetas de mutación y medidas conservadas fuera de Git. Siguen pendientes la unión con la generación nativa, el ajuste protegido con padres y cortes temporales disjuntos y sus contrastes científicos en [#701](https://github.com/GonxKZ/mars-titan/issues/701).
