# Contraste de trayectorias registradas

Los módulos `mars_titan.proposals.candidate_runtime`, `trajectory_reader` y `trajectory_contrast` añaden un contraste posterior a la [generación finita de propuestas](error-proposals.md). Materializan la referencia y las candidatas admitidas, ejecutan sesiones nativas independientes y comparan sus propias emisiones maduras en un tramo de selección. No reutilizan las doce pérdidas condicionales del consumidor padre para puntuar otra configuración.

La confirmación resultante identifica una configuración experimental destinada a una fase posterior. No sustituye el consumidor de una sesión iniciada ni cambia la referencia canónica del proyecto. Este recorrido no entrena un proponente, no ajusta parámetros compartidos y no acredita una mejora predictiva.

## API y tramos temporales

Las entradas parten de `ProposalRegistry`, un `ProposalLedger` ligado al plan de generación y su `ErrorProposalEvidence`. La generación debe haber terminado. Solo participan las propuestas terminales `proposed`, en el orden previo del plan. La configuración de referencia participa siempre, incluso cuando no se haya propuesto ninguna candidata.

El plan conserva tres intervalos sucesivos y disjuntos:

- G contiene las emisiones del padre y los errores maduros utilizados para generar las propuestas.
- S contiene nuevas emisiones propias de la referencia y de cada candidata. Solo este tramo interviene en su comparación y selección.
- E identifica el tramo posterior previsto. El contraste conserva sus límites, pero no abre ni ejecuta E.

Desde el comienzo de G hasta el final de E no pueden transcurrir más de 366 días. El contrato permite ventanas de meses y mantiene la reserva de 2024. La declaración del plan y la selección previa del padre preceden a G. Sus fechas son declaraciones locales comprobadas contra los datos disponibles, no una certificación externa de cuándo se decidió el protocolo.

S se recibe como `FinancialObservationSource`, con el mismo corpus y geometría de entradas que G, inicio de decisión igual a `selection_start` y cierre anterior a `selection_end`. La partición de almacenamiento debe ser `train` o `validation`. El rol temporal S se identifica por sus cortes explícitos. No se admite una fuente de `calibration` o `evaluation` para elegir la configuración.

## Materialización y reasignación de vista

`materialize_registered_candidate(registry, configuration_id, *, parent, parent_codec, input_spec, device="cuda:0")` devuelve un `CandidateRuntime`. Expone `consumer`, `codec`, `configuration_id`, `effective_configuration` y `record()`.

El padre es un `FrozenMechanismConsumer` ya existente y verificado. La receta basal del registro debe coincidir con su predictor, lector, precisión, acción y catálogo. Este contraste excluye selectores aprendidos, calibración y controles locales ajenos a la receta. No carga un checkpoint arbitrario ni reconstruye los pesos del padre a partir de su semilla.

El alcance inicial permite cambiar únicamente el programa de `regional_execution` y los pasos J de la recurrencia del puente, junto con su límite de actualizaciones por fila. Los componentes conservan su presencia y tipo. No se cambian semilla, arquitectura, codificador, topología, dimensiones, acción o número de parámetros. J sigue siendo distinto de K, que conserva la configuración del lector registrado.

El índice de observaciones de S tiene su propia huella de vista. La factoría exige que su `FinancialInputSpec` coincida con el original salvo `view_sha256`. Deben permanecer iguales el corpus, la representación completa, las dimensiones, el contexto y la política de entradas. Se registra la configuración original de G y una `RegisteredConfiguration` efectiva para S, sin modificar ni renombrar la propuesta original.

El nuevo codec conserva semilla, pesos de modalidades, peso de presencia, constante de clave y límite de memoria. Su proyección numérica debe ser idéntica. Cambian las identidades de entrada y de codec, que se trasladan explícitamente al lector y a su puente. `FrozenEpisodeCodec.from_identity` no sustituye esta operación, porque exige restaurar una identidad completa idéntica.

Se construyen módulos propios y se copian todos los parámetros y buffers admitidos con los mismos nombres, formas, tipos y bytes finitos, sin alias con el padre. El modo identificado es `paired_parent_weights_fresh_state_v1`. Se conservan los RNG globales de CPU y CUDA. El recibo registra las dos configuraciones, los consumidores, codecs y huellas por tensor. `record()` vuelve a contrastar los objetos efectivos y no vuelve a sellar cambios silenciosamente.

La copia de parámetros no transfiere memoria rápida, momentum, contexto, episodios ni una sesión del padre. Cada sesión de S comienza con estado propio y el mismo calentamiento declarado. Sus estados evolucionan a partir de sus propias emisiones y resultados recibidos.

## Ejecución independiente y cuotas

La entrada principal es:

```text
run_registered_contrast(
    output,
    *,
    ledger,
    plan,
    evidence,
    parent,
    parent_codec,
    source,
    prefixes,
    retention,
    native,
    admission="m1",
    block_rows=64,
    selection_min_gain=0.0,
    regional_work_limits=None,
    resume=False,
)
```

El coordinador materializa una configuración cada vez y utiliza `FinancialSession` y `run_observation_source`. Todas reciben la misma fuente S, prefijos, retención, admisión y bloque físico. Cada una mantiene sus predicciones pendientes, estado rápido, contexto y banco en un directorio propio. Las etiquetas se enlazan con los identificadores nativos de las predicciones realmente emitidas por esa sesión. Los IDs de muestra e inputs son comunes, mientras que los identificadores de predicción y de consumidor son propios.

La recogida `mechanism_collection` está desactivada. Se archivan emisiones propias, sin calcular otra colección de alternativas condicionales.

La retención debe tener el tipo correspondiente al modo de admisión:

| Modo | Tipo de configuración |
| --- | --- |
| M0 o M1 | `RetentionConfig` |
| M2 | `MatureErrorConfig` |
| M3 | `CompositeScoreConfig` |

`regional_work_limits` exige las nueve cuotas del contrato regional cuando hay un programa regional activo. Si no lo hay, no admite cuotas regionales. Las unidades siguen siendo observaciones del núcleo, consultas de contexto, decisiones del programa, evaluaciones locales y de resumen, observaciones MAC, refinamientos, búsquedas y confirmaciones de contexto. Se comprueba la reserva conservadora antes de predecir. Los contadores observados se informan por separado y no reembolsan trabajo reservado.

La generación y la ejecución tienen cuotas distintas. El contraste conserva como máximo dos intentos de ejecución por configuración, sin renovar los cargos al fallar o recuperar. Cada sesión tiene un límite de 128 MiB para su registro nativo. El estado del contraste admite hasta 4 MiB y cada archivo de trayectoria hasta 8 MiB. Estos límites son cotas de artefactos y trabajo declarado, no una medición del pico de RAM o VRAM.

## Lectura y comparación

`read_mature_trajectory(session, *, before, max_records=256, max_bytes=64*1024**2)` devuelve un `MatureTrajectory`. Recorre el prefijo completo de una sesión cerrada de entrenamiento o validación y reutiliza el enlace histórico verificado. Comprueba las huellas, la cadena nativa, las emisiones y la procedencia de cada resultado. No necesita una recogida de alternativas.

Cada fila conserva la muestra, el flujo, la decisión, la disponibilidad y huella de inputs, la madurez, la predicción nativa y sus cinco cuantiles. La pérdida usa su mediana exacta. La recepción real es el corte del registro nativo que recibió el resultado. Madurez, disponibilidad y recepción de una etiqueta utilizada deben preceder estrictamente a `before`. Si solo consta la huella de un snapshot, su número de episodios permanece desconocido.

El marco debe coincidir exactamente entre las configuraciones: muestras, flujos, decisiones, disponibilidad y huella de inputs y madurez. Las etiquetas puntuadas también deben coincidir. Un faltante no reduce el marco de comparación. Pendientes, exclusiones, censuras o soporte insuficiente dejan las métricas desconocidas y el contraste inconcluyente.

Se promedian activos dentro de cada sesión y mercado, después sesiones por mercado y finalmente los pesos de mercado del plan. Se conserva `delta_mae = mae_candidate - mae_baseline` y su ganancia con signo opuesto. `selection_min_gain` es el margen propio de selección, distinto del margen que disparó la propuesta en G.

La candidata seleccionada debe tener el menor MAE y superar estrictamente ese margen respecto a la referencia. Un empate con la referencia la conserva. Un empate entre candidatas respeta su orden previo. La comparación utiliza S para seleccionar y no constituye una prueba inferencial sobre E.

## Recuperación y confirmación

La salida conserva identidad, estado, sesiones nativas, trayectorias derivadas y, cuando procede, `confirmation.json`. Reabrir con `resume=True` exige las mismas identidades, código, datos, pesos, cuotas y margen. Los diarios nativos conservan los cursores confirmados y los estados parciales de cada configuración.

El contrato de recuperación exige reconstruir las trayectorias desde el archivo nativo y contrastar con ellas las hojas derivadas, sus estadísticas y la inicialización. Un JSON resellado no sustituye esa comprobación. Una sesión ya cerrada no debe repetir sus forwards para recuperar un recibo. Los intentos cargados permanecen cargados y una interrupción sin recibo de coste conserva ese coste como `None`, sin inventar cero ni devolver cuota.

Antes de publicar una confirmación se validan el estado y el resultado recuperados. La confirmación fija únicamente la configuración experimental destinada a E, ligada al padre esperado y a las trayectorias comparadas. Repetir la operación conserva su identidad. Otro padre o resultado incompatible produce rechazo. E permanece preparada por sus límites y sin ejecutar.

Las huellas comprueban coherencia local y enlaces con el archivo verificado. No autentican al host ni garantizan detectar una restauración coherente de todo el estado local. La preparación y las comprobaciones técnicas de este recorrido tampoco acreditan un proponente aprendido, una reproducción de SEAL o una mejora financiera observada.
