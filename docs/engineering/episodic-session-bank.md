# Banco episódico y publicación de sesión

`memory/episodic_session.py` conecta el codec fijo, las políticas de retención y
el ejecutor C++20. Cada instancia pertenece a una tarea, horizonte, predictor,
partición y fold. La comparación principal conserva un único objetivo residual
al horizonte declarado. Si se abren instancias para otros objetivos, sus costes
y capacidad retenida se suman.

La integración con `FinancialPredictor` sigue pendiente. El puerto de preparación
se ha ejercitado con callbacks escalares sintéticos. Falta reunir estados por
flujo en bloques compatibles y consumir `working_state` después de MAC para el
refinamiento episódico. Los tokens de fusión no son claves estables ni sustituyen
a ese estado. La GRU y las cabezas del predictor no se implementan en este módulo.

## Entradas y ciclo temporal

`EpisodicSession.step` recibe vistas `CPUDecisionBatch` de un mismo corte y
feedback separado. Verifica todas las vistas y calcula una huella por observación
con contrato, IDs, fechas, máscaras y bytes de las cinco modalidades. El grupo
se ordena por flujo. Dividir sus entradas en lotes físicos no cambia esa huella.

Las modalidades originales se sellan en bloques canónicos de hasta 256 filas.
La preparación vuelve a leerlos con su hash esperado y reutiliza el validador
financiero antes de llamar al callback. El sobre nativo de 136 números conserva
las claves y valores del codec y la huella de la observación. Las entradas del
predictor proceden de los artefactos completos referenciados.

El callback recibe filas inmutables, el estado rápido anterior, una vista de
consulta del banco confirmado, la identidad del checkpoint y grupo, y el tamaño
de lote físico. Devuelve predicciones por flujo y una propuesta de estado rápido.
Todo el grupo consulta la misma instantánea. El estado del consumidor debe ser
explícito y la preparación debe ser determinista. Los controles de C deben
seleccionar sus flujos sobre el grupo completo antes de dividirlo.

El `Executor` nativo sella todas las predicciones antes de aplicar el feedback.
Cada etiqueta se enlaza con la predicción pendiente de esa misma tarea y con sus
rasgos de entrada. El error es `etiqueta - predicción_emitida`, aunque haya
cambiado el estado rápido. No se guarda ese error en el campo de recompensa PPO.
El envelope del episodio conserva predictor, tarea, horizonte, muestra, fechas,
valor emitido, etiqueta, error y huella de inputs.

Los IDs numéricos del banco siguen el orden canónico de admisión madura del
ejecutor. Los rasgos pendientes no contienen objetivos futuros y no son
consultables como episodios. Sus modalidades originales siguen referenciadas
mientras quede alguna decisión pendiente del bloque. Una revisión de etiqueta
distinta de cero necesita otro contrato.

## Retención y coste

El banco reutiliza `EpisodicMemory`, su normalización, consultas, validación y
archivo de recuperación. Cada banco admite claves y valores FP32 de 64
coordenadas y hasta 1.024 registros. El codec y el banco tienen identidad propia.
Este formato no equivale al candidato de claves 128 y rasgos 256.

`RetentionBank.propose` devuelve una copia y deja intacta la instantánea anterior.
La escritura por lote valida todos los registros maduros y solo permite elegir
IDs del conjunto E formado por el banco anterior y las nuevas admisiones. Los
registros seleccionados conservan sus campos. Las políticas disponibles son:

- `reservoir` conserva el reservorio nativo original, que puede descartar la nueva
  admisión después de llenar la capacidad.
- `uniform` admite cada registro y elige una posición para sustituir con un PCG64
  propio. Su estado se guarda para recuperar exactamente la continuación.
- `recent` conserva las últimas admisiones. La antigüedad corresponde al orden
  de maduración, que puede diferir del orden de predicción.
- `anchored` abre los registros retenidos más antiguos como frontera. Conserva
  el resto como centros fijos y completa esos centros con las primeras nuevas
  admisiones si aún falta capacidad. Los candidatos variables incluyen la
  frontera y un subconjunto de nuevos IDs ordenado por una huella con semilla.

La política anclada llama al [selector revisado](../experiments/mars_titan_cm_v1/anchored_retention.md).
El objetivo incluye todos los clientes de E, aunque no sean candidatos. Los
centros fijos son clientes reales y aportan cero. El fondo usa NumPy y distancias
euclídeas FP64 sobre las claves FP32 normalizadas efectivamente almacenadas.
Es una pseudométrica sobre episodios que comparten representación. El valor
`1 - cosine` de una consulta no se usa como métrica de retención.

Los valores predeterminados son frontera 8, hasta 8 candidatos nuevos, 8
intercambios y 50 millones de pares de distancia. La búsqueda está restringida
a esos candidatos y no tiene garantía de óptimo global ni ratio de aproximación.
Agotar el presupuesto de pares, buffers o intercambios rechaza la propuesta.
El recibo conserva E, centros fijos, candidatos, representantes, objetivo y
trabajo de distancias. E puede tener hasta 9.216 filas con 8.192 admisiones,
aunque solo se retengan 1.024. Entre políticas, E puede divergir tras varias
sesiones, por lo que sus objetivos locales no son una evaluación común externa.

Al recuperar se exige el esquema completo del recibo, sus tipos, los estados de
selección admitidos y los presupuestos. Los IDs de centros y representantes deben
concordar con E, la capacidad y el banco nativo. La huella de E debe tener formato
SHA-256, pero el snapshot no conserva las coordenadas de los clientes descartados.
La recuperación no puede recalcular esa huella ni su objetivo anterior.

## Artefactos y recuperación

El estado JSON contiene metadatos y referencias de contenido a estado rápido,
banco, rasgos pendientes e inputs completos. `SessionArtifacts` usa carga
`weights_only`, tensores CPU finitos sin grafo y almacenamiento acotado. Comprueba
tipo JSON, esquema, versiones, dimensiones, identidad y SHA-256 antes de consumir
un artefacto. Las escrituras reutilizan las funciones nativas con `fsync` y
publicación de archivos inmutables.

`latest.json` del ejecutor es el único punto de publicación. Una propuesta puede
dejar archivos preparados sin quedar confirmada. Al reabrir se comprueban ambas
generaciones retenidas, se reconcilian las predicciones pendientes y se regeneran
sus claves y valores desde las modalidades originales. Solo después se retiran
los artefactos propios que ninguna de esas generaciones referencia. El callback
puede repetirse tras un intento no confirmado. Una generación confirmada no
vuelve a emitir sus decisiones ni a admitir sus etiquetas.

El enlace rechaza `close()` durante un `step` activo, también desde los callbacks
de preparación, actualización o interrupción. La marca de actividad se restaura
al retornar o propagar una excepción. Después del fallo se puede cerrar la
instancia y recuperar la última generación confirmada. Un `step` anidado también
se rechaza sin desactivar la protección del paso exterior.

El modo preparado admite hasta 8.192 flujos y conserva un límite de 64 MiB para
las modalidades del grupo lógico. El modo clásico mantiene 4.096 activos. La
cola permite 32.768 decisiones pendientes y cada maduración hasta 8.192 registros.
La retención descuenta una estimación para copias y archivos transitorios de su
presupuesto propio de 64 MiB, además del espacio calculado por el selector.
Estos límites no describen el RSS total del proceso.

La sesión permite hasta 512 artefactos de 64 MiB como máximo, con un límite
conjunto de 2 GiB. El estado rápido debe separarse en bloques cuando no quepa en
un artefacto. Esta versión rechaza el exceso y no trocea pesos de forma implícita.
Los dos checkpoints confirmados y los intentos preparados cuentan para esos
presupuestos. El registro de decisiones conserva sus límites nativos de disco.

## Enlace nativo

El enlace usa las cabeceras pybind11 incluidas en el SDK PyTorch, bajo su
licencia BSD de tres cláusulas. No compila PPO para usar el banco. La carga exige
un módulo ya compilado, conserva su SHA-256 y rechaza otro binario o una versión
incompatible de PyTorch en el mismo proceso.

`MARS_TITAN_TORCH_ENVIRONMENT` permite seleccionar un SDK externo. El enlace
consulta Python en ese mismo entorno, aunque `UV_PROJECT_ENVIRONMENT` señale
otro. La prueba CTest `episodic_python_environment` contrasta ambos intérpretes.

```bash
cmake --preset native-release -S native \
  -DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON \
  -DMARS_TITAN_BUILD_RUNNER=OFF -DMARS_TITAN_LIBTORCH_ENABLE_CUDA=OFF
cmake --build --preset native-release --target _episodic_native episodic_memory_tests
```

`MARS_TITAN_EPISODIC_NATIVE` indica la ruta del módulo compilado para las pruebas
Python. El enlace y el banco son CPU. No se ha ejecutado el predictor financiero,
aprendizaje, evaluación científica ni CUDA en las pruebas de esta integración.

## Comprobaciones y medidas

El [recibo técnico](../../reports/engineering/episodic-session-20261008.json)
recoge 80 pruebas Python, tres CTest en Release y los mismos tres con ASan/UBSan.
Se detectaron seis mutaciones sobre retención, RNG, objetivo, consulta del banco,
predicción original y huellas de artefactos. Ocho archivos del control clásico
coinciden byte a byte con la referencia anterior. Clang 21.1.8 compiló con avisos
estrictos y Lifetime Safety experimental. clang-tidy y el analizador de rutas
no emitieron diagnósticos propios en las tres unidades de producción modificadas.
Las bibliotecas precompiladas no quedan instrumentadas por esos sanitizadores.

Con 1.024 registros previos y 512 admisiones sintéticas, la mediana de tres
propuestas fue de 0,274 s para el reservorio, 0,278 s para sustitución uniforme,
0,274 s para recientes y 0,362 s para retención anclada. Incluye copia y
recuperación del banco y el cálculo de J sobre E completo para las cuatro
políticas. Quedan fuera el codec y el disco. La prueba de 8.192 admisiones
conservó 1.024 de 9.216 clientes, calculó 9.971.200 pares y tardó 5,53 s.

Dos sesiones con 5.676 flujos, repartidos en 23 bloques de inputs, tardaron 2,64
y 6,43 s. Ese recorrido incluye codec, modalidades completas, maduración,
retención, escritura, verificación y poda. La reapertura recuperó exactamente
el segundo estado en 2,09 s. Quedaron 5.676 admisiones, 1.024 registros retenidos
y 5.676 decisiones pendientes. Los 61 archivos ocuparon 32.722.231 bytes.
El proceso completo tardó 13,18 s y alcanzó 824.528 KiB de RSS, incluido el
runtime y las entradas. Fue una ejecución sintética con callbacks escalares,
no una medida del predictor ni una prueba de integridad del censo real.
