# Catálogo y propuestas del controlador de mecanismos

La integración de un controlador de mecanismos sigue en curso en [#701](https://github.com/GonxKZ/mars-titan/issues/701). Están implementados el acumulador de feedback de PT2, un catálogo finito, el consumidor que combina contexto PT4, rutas episódicas y calibración y su enlace durable con `FinancialSession` para M0/M1. Admite acciones fijas o una política congelada opcional que selecciona por mercado. Faltan el ajuste del selector con cortes disjuntos y sus contrastes predictivos. Las variantes por defecto no cambian.

El [núcleo de utilidades](../../reports/engineering/mechanism-policy-20261021/README.md) añade selección numérica entre las 36 combinaciones con coeficientes dados y objetivos puntuales y de intervalos separados. El [extractor de observaciones](../../reports/engineering/mechanism-observations-20261021/README.md) calcula y sella sus 21 rasgos desde las filas admitidas y el snapshot anterior, sin consultar supervisión. La [integración opcional](../../reports/engineering/mechanism-selection-20261021/README.md) los conecta antes de MAC y conserva la decisión en la misma generación financiera. Las pruebas, cobertura, CRAP, mutación y medidas de coste no acreditan un selector entrenado.

`FrozenMechanismConsumer(..., policy=policy)` exige una `MechanismPolicy` en modo `eval`, con buffers congelados y el mismo dispositivo que el predictor, además de una cabeza de cinco cuantiles. No admite a la vez una acción explícita. `policy=None` conserva la ruta fija y su formato. El predictor puede ser FP64 y el selector FP32 si se declara esa conversión. La identidad registra configuración, buffers, fuentes y precisión del selector. Su origen se declara como coeficientes suministrados sin ajuste acreditado.

El enlace extrae una observación por cohorte desde las filas ya admitidas. Antes de elegir construye las 36 resoluciones del catálogo por mercado y excluye las que requieren un retorno por falta de mecanismo, calendario o presupuesto. Las restricciones compartidas son conservadoras. Si el conjunto de mercados supera el límite de contexto, ninguno puede activarlo. Si la reserva de PT2 no cubre todas las cohortes online elegibles del evento, se desactiva online para todas. No se favorece al mercado que se procese primero. El retorno por soporte marginal no observado elige la referencia `off/k1/none`.

La traza `selection` se guarda una vez por evento, con una lista por mercado. Contiene observación original FP64 sellada, vector tras la conversión realmente usada, vector normalizado de la política, admisibilidad, ganancias puntuales y de intervalos, puntuaciones y acción. La normalización registrada es cero cuando el soporte obliga a sustituir la observación por la media. La verificación recalcula las utilidades congeladas y las concilia con las filas emitidas. El calentamiento y settlement no eligen y guardan `selection=None`. Con el selector desactivado ese campo no existe. El calentamiento automático usa contexto desactivado y la primera decisión empieza sin contexto confirmado. Una decisión que active PT4 puede aportar contexto a la siguiente.

Los hashes del contrato y del contexto de sesión pueden cambiar al variar el bloque físico. Eso no autoriza cambiar los rasgos consumidos ni la acción. La recuperación exige el mismo consumidor y sus coeficientes. Un selector incompatible se rechaza antes de reemplazar la cabeza confirmada. El archivo conserva las decisiones originales después de recibir feedback y de podar los checkpoints operativos.

El [productor de alternativas y pérdidas maduras](../../reports/engineering/mechanism-collection-20261021/README.md) está disponible como API explícita. Cambia una acción de un mercado manteniendo las anclas de los demás, calcula las alternativas desde el mismo prefijo y registra los retornos de reservas compartidas. Conserva ausencias como `None` y separa punto e intervalos. La recepción parcial valida identidad y madurez, conserva ganancias negativas y no borra pérdidas válidas por una exclusión. La [recogida nativa opcional](../../reports/engineering/native-mechanism-collection-20261021/README.md) enlaza sus referencias con la publicación, los pendientes y la recuperación de `FinancialSession`. Conserva la procedencia nativa de cada etiqueta y las censuras administrativas separadas. Las comprobaciones utilizan fixtures y no acreditan una colección científica ejecutada.

La [preparación del ajuste protegido](mechanism-policy-fitting.md) añade un contrato de fuentes cerradas y objetivos firmados. La máscara opcional de acciones se combina con el catálogo antes de elegir. Sin ella permanece el formato anterior. Una referencia a un recibo se declara sin acreditar por sí sola que el ajuste haya ocurrido. La ejecución positiva del lector y del solver, la selección temporal y los contrastes siguen pendientes. La [comprobación posterior](../../reports/engineering/mechanism-fit-20261021/README.md) incluye su uso en una sesión nativa FP32.

El selector puede elegir una acción online que posteriormente agote la reserva de bytes. La traza conserva esa elección y la ruta ejecutada, mientras concilia la salida K1 realmente emitida, su contexto, cuantiles y trabajo descartado. El retorno no se atribuye como acierto de la solicitud original.

## Catálogo y momentos de ejecución

[`MechanismCatalog`](../../src/mars_titan/models/titans/mechanism_catalog.py) registra 36 combinaciones. El contexto puede estar desactivado o utilizar el componente del padre. El lector conserva las seis rutas existentes, `k1`, `k2c`, `k2r`, `k4cc`, `k4cr` y `k4rc`. La calibración puede estar desactivada, usar la corrección inicial estática o recibir feedback de sus emisiones online. No se admiten comandos, código generado, acceso a red ni callbacks arbitrarios como acciones.

El contexto se elige antes de MAC. Solo los tokens fusionados de las filas seleccionadas llegan a PT4 y se acumulan sin realimentar el contexto añadido. El núcleo avanza una vez por observación. Después se ejecuta la ruta episódica elegida, conservando K1 y compartiendo su prefijo con los pasos posteriores. La calibración recibe la cohorte completa de cada mercado. Estos refinamientos no son otras actualizaciones de la memoria neuronal de Titans.

Los presupuestos distinguen refinamientos y búsquedas por fila, mercados con escritura de contexto y filas online. El lector se admite con el tamaño lógico del evento, limitado por el bloque declarado del padre. Dividirlo después en bloques físicos más pequeños no habilita una ruta que antes no cabía. Las unidades de trabajo no se convierten implícitamente en MAE ni en utilidad financiera.

El predictor puede incorporar la [fusión regional](../experiments/regional-fusion-v1.md) antes de PT4 y MAC. Es una opción de arquitectura fija, no otra acción aprendida del catálogo. Sus contadores distinguen evaluaciones de nodos locales y del resumen, con un límite conjunto por fila. El enlace los conserva en la misma traza confirmada. Los estados intermedios se reinician en cada predicción y no se añaden al estado histórico del banco ni a los pesos rápidos.

Cuando falta un mecanismo o presupuesto, se registra el motivo y se emite K1 sin calibración. Si el fallo se conoce después de elegir el contexto, se conserva ese contexto pre-MAC. No se recalcula la transición para aparentar que la decisión anterior no ocurrió. Una entrada inválida, una identidad incompatible o un valor no finito producen un error, no un retorno silencioso.

## Consumidor provisional y estado

[`FrozenMechanismConsumer`](../../src/mars_titan/models/titans/mechanism_consumer.py) verifica módulos, parámetros congelados, métodos, precisión y fuentes. `begin` declara identidades, corte y madurez por fila. `prepare` devuelve inmediatamente el estado rápido del bloque para su descarga por el enlace financiero. El evento conserva tokens de contexto pendientes y salidas CPU acotadas, no los estados rápidos de toda la historia. `finish` exige todas las filas, propone una escritura de contexto por mercado y aplica la calibración.

La emisión lleva `publication_status="proposed_not_durable"`. Su estado en memoria no acredita publicación en disco. Las trazas separan ruta solicitada, ejecutada y emitida. Los contadores distinguen observaciones MAC, refinamientos, búsquedas, escrituras de contexto propuestas y propuestas de calibración creadas, seleccionadas o descartadas. Contabilizar un intento no lo convierte en una confirmación.

Si una propuesta online supera los bytes disponibles o la reserva operativa para feedback, se descartan juntas las propuestas online de ese evento. También vuelve a K1 la primera cohorte de mercado que había cabido. Se conservan los contadores del trabajo ya realizado. La reserva usa el tamaño del JSON y un margen por fila pendiente. No es una cota de RAM, una certificación numérica ni una garantía de que todo resultado futuro quepa.

La identidad declara `predict_all_then_feedback_v1`: primero se emiten todas las predicciones del evento y después se reciben resultados anteriores disponibles. El replay original de PT2 aplicaba las etiquetas maduras antes de emitir. Son políticas diferentes. Una cola llena puede provocar un retorno aunque el feedback posterior del mismo evento vaya a liberar capacidad. No se adelantan esas etiquetas.

El estado auxiliar exporta contexto y calibración con versión e identidad. Un cierre administrativo conserva pendientes como archivo terminal censurado, sin inventar su madurez ni avanzar el reloj del calibrador al futuro. Ese archivo no puede continuar otra fase. La recuperación durable utiliza el enlace descrito más abajo.

[`CohortFeedback`](../../src/mars_titan/calibration/cohort_feedback.py) recibe cohortes completas de los cinco cuantiles financieros y conserva sus identidades y fechas de madurez por fila. Su calibrador es el [`OnlineConformal`](../../src/mars_titan/calibration/online_conformal.py) existente. La madurez la aporta el contrato temporal que use el consumidor. El acumulador no la sustituye por un día añadido a la fecha de decisión.

`emit` devuelve los cuantiles en el orden recibido, el registro de emisión y un snapshot nuevo. Internamente ordena las filas por identidad, junto con sus cuantiles y madurez. No lee etiquetas ni modifica el snapshot original. Las propuestas descartadas no se publican por el mero hecho de calcularlas.

`feedback` recibe resultados tipados con mercado, decisión, identidad y disponibilidad. Rechaza valores futuros, etiquetas anteriores a su madurez, duplicados e identidades desconocidas. Valida cada entrada antes de acceder a su valor numérico. Si falla una entrada posterior del lote, el snapshot de partida permanece intacto.

`unresolved_keys` devuelve las identidades que el calibrador aún espera, sin copiar cuantiles o etiquetas. Una etiqueta parcial ya recibida no vuelve a figurar en esa vista. La cola financiera puede seguir conteniendo otras predicciones después de cancelar una cohorte de calibración. El enlace debe conservar sus pérdidas válidas y enviar a PT2 únicamente las claves que todavía espera.

Las etiquetas parciales quedan pendientes hasta completar la cohorte más antigua de cada mercado. Las posteriores pueden recibir feedback, pero esperan su turno para actualizar la corrección. La actualización usa los cuantiles y la corrección realmente guardados al emitir, como ya hacía PT2. El reloj global y los relojes por mercado no retroceden.

[`PrefixTargetVerifier.expected_maturity`](../../src/mars_titan/training/prefix_eligibility.py) obtiene la disponibilidad prevista de la sesión siguiente mediante el calendario y los manifiestos admitidos. No consulta retornos ni garantiza que exista la etiqueta. Conserva festivos, cambios horarios y medias sesiones. En el contrato auditado, los precios están disponibles cinco minutos después del cierre. Sin siguiente sesión anterior al periodo reservado devuelve `None`. El consumidor conserva esas filas y desactiva su ruta online con el motivo `unavailable_label_calendar`. Las acciones estáticas conservan su ruta y los otros mercados no pierden capacidad por las filas sin calendario.

## Exclusiones y recuperación

Una exclusión o finalización necesita un estado identificado y una huella de evidencia. El acumulador verifica su formato, no su autenticidad. El enlace recibe los resultados admitidos por el ejecutor nativo. No convierte finalizaciones administrativas en etiquetas.

Si una fila queda excluida, se cancela la cohorte completa al alcanzar su madurez. No se inventan etiquetas para las demás filas ni se actualizan correcciones, errores o número de cohortes maduras con esa cancelación. El registro distingue etiquetas válidas recibidas, exclusiones y filas aún sin resultado. Las métricas del ejecutor deben conservar las pérdidas válidas de las otras predicciones, aunque esa cohorte ya no sirva para actualizar PT2. Los resultados posteriores de filas canceladas no se vuelven a enviar al acumulador de calibración.

`OnlineConformal.cancel` permite retirar únicamente la cohorte pendiente más antigua, tras su madurez y respetando el reloj. Las llamadas anteriores a emisión, maduración, exportación y restauración conservan su comportamiento y formato. La cobertura marginal de las cohortes completas no acredita cobertura de las canceladas ni de los intervalos que seleccione una política.

El estado del acumulador tiene versión y política propias, huella SHA-256 y una representación JSON con copias de cuantiles y metadatos. La restauración comprueba correspondencia de cohortes, identidades, orden, fechas y resultados parciales. La huella detecta cambios accidentales, no autentica un origen hostil.

Los límites por defecto son 32.768 filas pendientes y 8 MiB de JSON canónico. El máximo configurable de JSON es 16 MiB. Cada mercado conserva el límite de 64 cohortes de PT2 y el formato admite hasta ocho mercados. Los identificadores tienen como máximo 128 bytes. El presupuesto de JSON no representa un límite del pico transitorio de RAM. No se conserva un historial ilimitado de cohortes resueltas.

[`MechanismHistory`](../../src/mars_titan/memory/mechanism_history.py) prepara el archivo de emisiones y actualizaciones sobre `SessionArtifacts`, con bloques sellados, generaciones contiguas y lectura incremental. Su manifiesto declara las referencias que la sesión conserva al podar. No publica otra cabeza ni confirma por sí mismo una propuesta. El formato distingue el historial del estado operativo y permite leer generaciones antiguas sin retener sus payloads en la caché.

Cada evento lleva identidad de modelo, generación, corte, tipo y huella del registro nativo, además de un payload JSON acotado. El módulo valida ese transporte. El enlace verifica la correspondencia financiera con inputs, acciones, cuantiles y resultados. La compresión `zlib1` es opcional y su versión forma parte de la identidad. `none` sigue siendo el valor por defecto. El agotamiento de capacidad produce un error y no autoriza borrar evidencia o cambiar la predicción para continuar.

## Publicación en la sesión financiera

[`MechanismBinding`](../../src/mars_titan/memory/financial_mechanisms.py) reutiliza el banco, el codec, los snapshots y la descarga de pesos rápidos del enlace Titans. Admite el consumidor de mecanismos por su tipo exacto, con M0/M1. Rechaza M2/M3 y el control local C activo mientras esas interacciones no estén comprobadas. B6 mantiene su enlace independiente. Estas restricciones no eliminan esas variantes de sus canalizaciones anteriores.

El bundle añade `mechanisms` y `mechanism_history` únicamente para la nueva identidad. La preparación utiliza el mismo snapshot de banco para toda la sesión. Descarga el estado rápido después de cada bloque físico y reúne las salidas CPU. La propuesta auxiliar no se publica en `finish`. El resolver recibe el registro nativo ya sellado, comprueba los puntos y resultados reales, añade el evento histórico y prepara el bundle. `latest.json` sigue siendo la única publicación. La poda conserva las referencias históricas de las dos generaciones recuperables.

Las trazas conservan cuantiles brutos, K1 y emitidos, niveles, corrección aplicada al emitir, acciones, motivos de retorno e identidades de inputs, contexto, parámetros y snapshot. La validación reaplica esa corrección archivada y la concilia con los puntos nativos. No reconstruye una emisión antigua con el calibrador actual. `mechanism_state()` recupera el estado auxiliar confirmado y `mechanism_event(generation)` lee una generación confirmada, también después de la poda. Una generación futura o una referencia incompatible se rechazan.

El feedback se filtra una vez por evento mediante las claves que PT2 todavía espera. Las pérdidas financieras válidas permanecen aunque no exista una cohorte de calibración. El calentamiento no recibe resultados para PT2. En el cierre de la frontera reservada, el watermark de feedback es el mínimo entre el corte y el instante reservado menos un microsegundo. Es una frontera exclusiva del prefijo. No cambia la disponibilidad real de ninguna etiqueta ni amplía el acceso al periodo reservado. El cierre administrativo posterior conserva el estado terminal sin trasladarlo implícitamente a otra fase.

`FinancialSession(..., mechanism_collection=MechanismCollectionConfig())` activa la recogida. El valor por defecto es `None`. El bundle añade `mechanism_collections` y la traza incorpora `collection_packet`. Los paquetes se calculan y sellan antes del feedback. Sus estados contrafactuales se descartan. La ruta real se ejecuta también y sigue gobernando las emisiones, los pesos rápidos y el banco. La identidad declara fuentes, límite y tolerancias del ancla (FP64 1e-12, FP32 1e-5). El índice conserva referencias directas, con 64 decisiones por defecto y un máximo configurable de 256. Su agotamiento o el de los presupuestos generales detiene la propuesta sin borrar evidencia. No es un archivo ilimitado.

`mechanism_collection(generation)` devuelve el paquete, la recepción parcial y las censuras. `mechanism_collection_metrics(generation, device="cuda:0")` calcula pérdidas sobre etiquetas maduras. Las censuras no son etiquetas ni ceros y se descuentan de los pendientes. El reloj del receptor es el máximo de las disponibilidades recibidas. El registro nativo conserva cuándo se recibieron realmente. La verificación comprueba esa procedencia incluso cuando PT2 no esperaba el resultado. La recuperación valida todas las recogidas y las comprobaciones ordinarias cargan las pendientes y la actual. Las referencias históricas se enumeran desde el índice al podar.

Los formatos anteriores se mantienen cuando no se usa el nuevo enlace. Las identidades incluyen código y política, por lo que esta modificación no autoriza cargar un checkpoint de otra versión como si conservara su identidad histórica. Los sellos y conciliaciones detectan corrupción e incompatibilidades. No autentican el sistema frente a un host que pueda modificar arbitrariamente código y archivos.

## Comprobación y coste

Los estados neuronales adicionales se describen en [BANK-READ](../experiments/bank-readout-v1.md) y [BANK-WRITE](../experiments/bank-write-v1.md). Sus informes separan mecanismos, pruebas y coste de una utilidad predictiva todavía no medida. El [ensayo de préstamo de almacenamiento](../../reports/engineering/bank-write-storage-20261021/README.md) conserva un candidato descartado por su latencia en sesiones pequeñas. No cambia la implementación activa ni los diez criterios pendientes de #701.

Las [pruebas y medidas del acumulador](../../reports/engineering/cohort-feedback-20261021/README.md) usan fixtures identificados, sin corpus real ni pasos de optimizador. Cubren recepción dividida, exclusiones, permutaciones, recuperación, aislamiento y rechazos. El [informe del consumidor](../../reports/engineering/mechanism-consumer-20261021/README.md) añade las 36 combinaciones en CPU y CUDA, presupuesto lógico, recuperación auxiliar, detección de cambios y coste del recorrido provisional. Ambos registran mutación dirigida y CRAP con su alcance.

El [incremento de archivo y madurez](../../reports/engineering/mechanism-history-20261021/README.md) comprueba las claves pendientes, el calendario y el formato histórico. El [informe de la sesión integrada](../../reports/engineering/mechanism-session-20261021/README.md) añade cortes antes y después de publicar, recuperación, poda histórica, fronteras reservadas, corrupción y coste CUDA con 64 y 256 activos sintéticos.

La [comprobación regional](../../reports/engineering/regional-fusion-20261021/README.md) añade paridad de la ruta anterior, gradientes, aislamiento de activos, ocho controles y recuperación de la siguiente emisión con sus contadores. Registra coste CUDA residente y de sesión, cobertura, CRAP y mutación dirigida. No se ha medido su utilidad predictiva ni se ha entrenado el selector.

El acumulador y la serialización trabajan en CPU con NumPy. Su medida aislada utilizó propuestas completas con 64 y 256 filas y una o cuatro cohortes pendientes. La copia explícita de contenedores redujo el coste observado frente a las copias recursivas, sin cambiar los valores ni los estados exportados. Esa comparación del acumulador no incluía el predictor ni la escritura atómica. El informe de la sesión integrada mide ese recorrido adicional por separado.

Para repetir la selección técnica en el entorno del proyecto:

```bash
PYTHONPATH=src:. uv run --no-sync pytest -q tests/calibration/test_cohort_feedback.py tests/calibration/test_online_conformal_cancellation.py tests/calibration/test_online_conformal.py
```

Para repetir la medida se necesita una ruta de salida cuyo directorio exista y cuyo archivo no exista todavía:

```bash
PYTHONPATH=src:. uv run --no-sync python reports/engineering/cohort-feedback-20261021/benchmark.py --output /tmp/mars-titan-cohort-feedback-cost.json
```

Faltan el ajuste del selector con cortes disjuntos y sus controles de utilidad. La [revisión de contexto y bucles](../research/context-and-nested-control.md) distingue esta integración de una especialización aprendida o una arquitectura anidada. El acumulador, el catálogo y la persistencia no completan ese objetivo. La campaña completa y el test final permanecen cerrados. No hay una mejora predictiva medida por este componente.
