# Ejecución de cohortes y recuperación

El ejecutor C++20 calcula todas las predicciones de un corte con el mismo estado anterior, conserva las salidas y después aplica el feedback maduro de decisiones anteriores. El estado resultante comienza a utilizarse en la siguiente cohorte. Implementa la barrera descrita en [la integración del sistema](../research/system-integration.md), con un control numérico sintético y sin implementar ni entrenar el candidato MARS-TITAN.

La API está en [`cohort_execution.hpp`](../../native/include/mars_titan/cohort_execution.hpp). `Executor` recibe una `Definition`, dos callbacks y un directorio de salida. La definición contiene las identidades del origen, vista, representación y modelo, las tareas con sus horizontes, el estado inicial JSON y los presupuestos. La identidad confirmada incluye también las tareas, los límites y la huella del estado inicial.

## Contrato de una cohorte

`step` recibe observaciones, un corte creciente y el cursor esperado. Cada observación tiene activo, disponibilidad y un vector numérico. Los objetivos llegan por un argumento separado. No forman parte de la entrada del predictor. La biblioteca rechaza activos repetidos, observaciones futuras, dimensiones incorrectas, valores no finitos y cursores que salten o repitan una cohorte.

Los activos y tareas se ordenan antes de invocar `predict`. Cada lote físico recibe una vista constante del mismo estado. Cambiar el lote no crea otra decisión ni adelanta el feedback. El callback devuelve una salida finita por activo. Para conservar invariancia al lote, cada salida debe ser independiente de qué otros activos compartan ese lote físico. El determinismo por sí solo no basta para modelos que mezclen activos dentro del callback. Los callbacks deben mantener todo el estado mutable del consumidor en el JSON explícito. Los spans prestados solo son válidos durante la llamada. No se admiten llamadas simultáneas a una misma instancia ni conservar esos spans para trabajo posterior.

La identidad de una predicción combina ejecución, corte, activo, tarea y horizonte. El feedback añade su revisión a esa identidad. Esta versión admite únicamente la revisión fija cero. Una revisión posterior requiere otro contrato, no una corrección silenciosa. `Task.horizon` identifica el objetivo, pero no calcula calendarios de mercado ni acredita la publicación de una etiqueta. El proveedor debe aportar su disponibilidad.

`PredictionMode::prepared` admite consumidores cuyo estado también cambia al
leer entradas. En ese modo, `prepare` sustituye a `predict` y se invoca una sola
vez con todas las observaciones y tareas ordenadas, el corte, el estado anterior
y el tamaño de lote físico. Devuelve un `PreparedCohort` con una salida por
activo y tarea, en ese orden, y `proposed_state`. No recibe el feedback. El
consumidor divide internamente el cálculo en lotes y conserva la misma
instantánea del banco durante toda la preparación. Los presupuestos y la
selección de flujos del grupo lógico deben fijarse antes de esa división.

El ejecutor comprueba tamaño, finitud y presupuesto de la propuesta. Sella
todas las predicciones y después llama a `update(proposed_state, feedback)`.
El resultado se publica con el cursor y la cola bajo el mismo `latest.json`.
La propuesta de un intento interrumpido no es estado confirmado. `prepare` y
`update` pueden repetirse al recuperar un intento anterior a la publicación y
no deben producir efectos externos. Una generación ya confirmada no se repite.

El JSON de ese consumidor contiene metadatos y referencias verificables a
artefactos sellados para los tensores, no listas de pesos densos. Su adaptador
debe comprobar formato, tamaño e identidad de esos artefactos antes de
utilizarlos. Esta API no interpreta ni valida tensores externos por su cuenta.
No se debe publicar otra cabecera independiente para ellos. El modo preparado
usa una identidad de configuración de versión 2. El modo clásico conserva su
identidad y sigue pasando el estado anterior directamente a `update`.

La cola guarda la salida original de cada predicción pendiente. `update` recibe esa salida y la etiqueta ya disponible, después de registrar todas las predicciones del corte actual. El error utiliza la salida conservada, aunque el estado haya cambiado desde que se emitió. Cada horizonte madura de forma independiente. Una etiqueta futura, repetida, desconocida o ya aplicada se rechaza sin volver a producir su efecto. El proveedor puede consultar `pending()` al recuperar para solicitar solo los resultados que faltan.

`replay_id` crea identidades de exposición distintas de la identidad del feedback y de otras visitas. No vuelve a aplicar feedback ni introduce un calendario de entrenamiento. El ejecutor llama una vez a `update` por cohorte, incluso si el conjunto de etiquetas maduras está vacío.

## Publicación y recuperación

Se reutilizan `OutputLock`, `content_sha256`, `parse_bounded_json`, `read_bounded_file` y las escrituras atómicas de `simulation_files.hpp`. El backend de archivos necesita POSIX. No añade un lector Parquet ni un parser JSON.

Los archivos tienen funciones distintas:

- `identity.json` fija el contrato de la ejecución.
- `record-N.json` conserva las predicciones, feedback aplicado, corte y contadores acumulados. Cada registro enlaza la huella del anterior y no se sustituye por otro contenido.
- `checkpoint-N.json` guarda cursor, estado y predicciones pendientes. No incluye una copia de toda la historia.
- `latest.json` confirma la generación y las huellas de los checkpoints actual y anterior.

La secuencia de un paso es predicción completa, escritura del registro, cálculo del estado siguiente, escritura del checkpoint y sustitución atómica de `latest.json`. Los lectores deben seguir esa cabecera. La mera presencia de un archivo preparado no acredita su confirmación.

| Frontera de interrupción | Generación recuperada | Tratamiento del intento |
| --- | --- | --- |
| Antes de sustituir `latest.json` | La anterior | Se repite desde el estado confirmado. Los archivos preparados solo se reutilizan si coinciden exactamente. |
| Después de sustituir `latest.json` | La nueva | El cursor ya incluye las predicciones y el feedback del paso. |

Una excepción después de iniciar los callbacks invalida la instancia. Hay que destruirla y reabrir con `resume=true`. Esto evita continuar con un estado de memoria que no coincida con la publicación en disco. La recuperación comprueba las dos generaciones retenidas, el último registro y la transición entre ellos. Contrasta también los pendientes y los totales fijados en el registro, por lo que modificar contadores en dos checkpoints no basta para obtener una transición válida.

Se conservan dos checkpoints confirmados. Durante un intento puede existir un tercero. La generación antigua se retira después de confirmar la nueva, sin borrar el único estado confirmado. Al reabrir se revisa un inventario acotado y se eliminan temporales propios de escrituras interrumpidas. La recuperación rechaza corrupción e identidades incompatibles. No retrocede automáticamente a un checkpoint anterior cuando el actual está corrupto.

La garantía afecta al estado local confirmado. Un callback puede ejecutarse otra vez tras un intento no confirmado, por lo que no debe producir efectos externos. Este contrato no ofrece entrega única a servicios ni persiste por su cuenta optimizadores o tensores externos al estado JSON.

## Presupuestos

Los valores predeterminados son 4.096 activos, 32.768 predicciones pendientes, 1 MiB de estado del consumidor, 16 MiB por checkpoint y registro, 1 GiB de registros acumulados y 100.000 cohortes. Se comprueba la ocupación transitoria de la cola antes de predecir. Agotar un presupuesto detiene el avance y conserva la última generación confirmada.

La biblioteca mantiene en memoria la cohorte actual, el estado y la cola acotada. Cada paso escribe solo su registro y checkpoint y revisa el checkpoint que debe retirar. No relee el contenido de toda la historia. La recuperación enumera archivos para controlar retención y temporales, pero carga únicamente los checkpoints necesarios y el último registro. `record` comprueba la cadena desde su extremo confirmado, con un máximo predeterminado de 4.096 registros leídos. Los límites de archivos y tensores no equivalen al RSS total del proceso.

## Ejecutable y comprobaciones

```bash
PYTHONPATH=src cmake --preset native-release -S native -B /tmp/cohort-release \
  -DMARS_TITAN_BUILD_COHORT_EXECUTION=ON -DMARS_TITAN_BUILD_RUNNER=OFF
cmake --build /tmp/cohort-release --target mars-titan-cohorts cohort_execution_tests -j 1
ctest --test-dir /tmp/cohort-release -R '^cohort_' --output-on-failure -j 1

/tmp/cohort-release/mars-titan-cohorts --output /tmp/cohort-control \
  --cohorts 8 --assets 16 --features 32 --batch 8
/tmp/cohort-release/mars-titan-cohorts --output /tmp/cohort-control --resume \
  --cohorts 32 --assets 16 --features 32 --batch 16 --reverse
```

El control usa señales modulares deterministas, dos horizontes y una corrección escalar con resultados maduros. Los identificadores y objetivos son sintéticos. No representan activos reales ni resultados financieros. `report.json` conserva tiempos por paso, coste de callbacks, reaperturas, bytes y huellas de la ejecución. `--features` permite comprobar distintas dimensiones y `--help` describe los argumentos.

El 5 de octubre de 2026 pasaron los ocho CTest del perfil Debug y los dos del ejecutor en Release. Las tres entradas CTest del componente con ASan y UBSan, incluido el smoke de fuzzing, también pasaron. Las pruebas comprueban siete fronteras de interrupción, muertes reales con `SIGKILL` tras el registro y tras la confirmación, invariancia al orden de activos, tamaño de lote y sufijo futuro, salidas inválidas, corrupción, límites y retención. LibFuzzer completó otras 10.000 ejecuciones con semilla 42, entradas de hasta 4.096 bytes y un límite de RSS de 1 GiB, sin fallos detectados.

Clang 21.1.8 compiló con avisos estrictos tratados como errores, endurecimiento de libstdc++ y Lifetime Safety experimental. clang-tidy 21.1.6 y el analizador de rutas Clang 21.1.8 no emitieron diagnósticos propios en las dos unidades de producción. Se detectaron siete mutaciones dirigidas sobre aplicación anticipada de feedback, salida original, duplicados, revisiones, capacidad, finitud del estado e identidad de replay. No se ejecutaron TSan ni MSan en este componente. La comprobación de memoria no instrumenta las bibliotecas precompiladas de terceros.

LLVM 21.1.8 midió un 87,90 % de líneas y un 60,27 % de ramas en `cohort_execution.cpp`. En el CLI fueron un 92,74 % y un 77,91 %. La complejidad se calculó con Lizard 1.24.0. CRAP usa `CC² × (1 − cobertura_de_líneas_ejecutables)³ + CC`, con rangos de funciones de Lizard y líneas `DA` de LLVM LCOV. El mayor valor fue 29,28 en `check_transition`, con CC 24. Estos valores describen lo ejercitado por las pruebas y no certifican ausencia de errores.

La ampliación del modo preparado se comprobó el 8 de octubre de 2026 con los
dos CTest del componente en Debug, ASan/UBSan y cobertura. Incluye siete
fronteras de interrupción, una preparación para varias tareas, orden y lote
físico, propuestas inválidas e incompatibilidad de modos. Cinco mutaciones
dirigidas se detectaron. Los ocho archivos de identidad, registros, checkpoints
y cabecera de un control clásico de cuatro cohortes, cinco activos y siete
variables coincidieron byte a byte antes y después del cambio.

Clang 21.1.8 compiló con avisos estrictos y Lifetime Safety experimental.
clang-tidy y el analizador de rutas no emitieron diagnósticos propios en la
unidad de producción modificada. LLVM cubrió 616 de 693 líneas y 289 de 466
ramas de `cohort_execution.cpp`. La función nueva `prepare` tuvo todas sus
líneas ejecutables cubiertas, CCN 5 y CRAP 5 según la misma convención LCOV.
Estas comprobaciones usan estados escalares sintéticos. La validación y
recuperación de artefactos de tensores externos corresponde al consumidor.

## Coste observado

Se ejecutaron tres procesos Release por carga, sin descartar pasos de calentamiento y con cinco reaperturas completas por proceso. El lote físico fue de 32 activos y `seed=42`. El equipo tenía un AMD Ryzen 9 8945HS y 31.556.064 KiB de RAM visible, con Linux 7.0, Clang 21.1.8, OpenSSL 3.5.5 y el SDK C++ de Arrow 25.0.1. No se detuvieron otras aplicaciones.

| Carga | Pasos | Paso p50, ms | Paso p95, ms | Paso p99, ms | Reapertura p50, ms | Filas/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 activos × 32 variables | 32 | 6,059 | 7,259 | 8,812 | 1,386 | 1.299,77 |
| 128 activos × 1.681 variables | 128 | 40,651 | 44,706 | 46,543 | 10,280 | 3.120,29 |

La tabla recoge la mediana de las métricas de los tres procesos. El p50 por proceso osciló entre 5,940 y 6,060 ms en la carga pequeña y entre 40,133 y 40,768 ms en la mayor. La forma de 1.681 variables reproduce el tamaño numérico descrito en la integración del sistema, con valores sintéticos.

El tiempo completo de proceso, medido con `/usr/bin/time`, fue de 0,22 s y 5,46 s en la mediana. Los máximos de RSS medianos fueron 25,09 MiB y 41,55 MiB. Las entradas numéricas sumaron 65.536 y 220.332.032 bytes y los registros confirmados, 281.651 y 17.885.315 bytes. En la carga mayor, los callbacks de predicción acumularon 43,47 ms y los de actualización 0,32 ms. El tiempo del paso incluye además validación, serialización, huellas, escritura, sincronización y retención. No se atribuye todo ese coste a uno de esos componentes.

El [registro de validación y medidas](../../reports/engineering/cohort-execution-20261005.json) conserva las repeticiones, versiones, huellas de fuentes y resultados de calidad. No se utilizó CUDA ni se midieron energía o coste económico. Estas cifras describen el control y su persistencia, no el rendimiento de un modelo MARS-TITAN.
