# Catálogo y propuestas del controlador de mecanismos

La integración de un controlador de mecanismos sigue en curso en [#701](https://github.com/GonxKZ/mars-titan/issues/701). Están implementados el acumulador de feedback de PT2, un catálogo finito, el consumidor que combina contexto PT4, rutas episódicas y calibración y su enlace durable con `FinancialSession` para M0/M1. Las acciones son controles fijos. Faltan el selector aprendido y sus contrastes predictivos. Las variantes por defecto no cambian.

El [núcleo de utilidades](../../reports/engineering/mechanism-policy-20261021/README.md) añade selección numérica entre las 36 combinaciones con coeficientes dados y objetivos puntuales y de intervalos separados. Todavía no se conecta al consumidor ni extrae las observaciones del prefijo. Sus comprobaciones CPU/CUDA, cobertura, CRAP y mutación no acreditan esa integración ni un selector entrenado. La ejecución financiera sigue usando las acciones fijas descritas a continuación.

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

Los formatos anteriores se mantienen cuando no se usa el nuevo enlace. Las identidades incluyen código y política, por lo que esta modificación no autoriza cargar un checkpoint de otra versión como si conservara su identidad histórica. Los sellos y conciliaciones detectan corrupción e incompatibilidades. No autentican el sistema frente a un host que pueda modificar arbitrariamente código y archivos.

## Comprobación y coste

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
