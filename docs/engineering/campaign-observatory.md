# Seguimiento de campañas y publicación periódica

El recolector lee los informes de las campañas en `data/interim`, incluidos los
recibos anteriores y los índices pequeños de checkpoints. No carga pesos,
predicciones Parquet ni modalidades. Se ejecuta fuera del entrenamiento y guarda
su caché de fuentes y su historial en SQLite. Las fuentes se vuelven a leer cuando
cambia su identidad de archivo, tamaño o fecha de modificación.

La configuración de fuentes está en
[`configs/observatory/campaigns.json`](../../configs/observatory/campaigns.json).
Las rutas de campañas que aún no existen son destinos previstos. Su ausencia se
muestra como trabajo pendiente o dependiente de otras campañas. Los recuentos se
calculan con las configuraciones de búsqueda y adaptación. El diseño actual tiene
80 ejecuciones neuronales, 17 tabulares y 216 ajustes predictivos. Los informes
anteriores tienen sus propios recuentos y procedencia.

El postentrenamiento emparejado añade 396 ajustes previstos. El recuento se deriva
de tres semillas, tres condiciones, seis modos para cada uno de los seis padres y
dos continuaciones adicionales para cada una de las cuatro familias neuronales.
La familia de cada padre se declara en la configuración de fuentes. La campaña
depende de las búsquedas neuronal y tabular originales. El plan financiero añade
seis entrenamientos previstos, dos algoritmos por tres semillas. Las evaluaciones
de sus referencias y costes no multiplican ese recuento de entrenamientos.

Las configuraciones canónicas siguen en `configs/baselines/paired-posttraining.json`
y `configs/simulation/comparators.json`. Si una raíz de datos anterior aún no las
contiene, `configuration_snapshot` permite leer sus copias de
`data/interim/experimental-configurations-20260924`. El manifiesto de esa carpeta
registra la revisión `0e5d85c66a14300f0f6d8e00c63a50771cc15678`, las rutas originales
y sus SHA-256. Las copias quedan separadas de las futuras salidas de las colas.
`configuration_sha256` exige los mismos bytes en la ruta canónica y en la copia,
también al leer desde caché. Una configuración presente con otra huella se rechaza.

## Campañas por ventanas de A y A v2

La campaña con máscaras desde 2000 tiene cuatro etapas por variante y cada una escribe
su propio `summary.json` con el mapa de trabajos confirmados: la campaña base, los
adaptadores por etapas, la ablación de modalidades y las políticas. La configuración
declara las ocho como fuentes de tipo `window_campaign`, con su etapa y su declaración
en `configs/`. Sus salidas previstas son `data/interim/historical-masked-a/<etapa>` y
`data/interim/historical-masked-a-v2/<etapa>`, con `base`, `adapters`, `ablation` y
`policies` como etapa. `run_masked_campaign.py rolling` debe recibir esas rutas en
`--output`, `--adapter-output`, `--ablation-output` y `--rl-output` para que el
observatorio las encuentre. Mientras no exista el resumen, la etapa aparece como
declarada y sin resumen, nunca como trabajos completados.

Estas fuentes no producen registros por ejecución. A suma 22.188 trabajos entre sus
etapas y A v2 21.331, muy por encima del presupuesto por defecto de 4.096 registros, y
la web descarga todas las páginas del historial al abrirse.
`observatory/window_campaigns.py` convierte cada resumen en una matriz de ámbito,
ventana, brazo y nombre con el estado de cada trabajo, la fecha de su recibo y las
curvas por época de hasta ocho intentos abiertos. El mismo módulo alimenta el servidor
en directo. La etapa de políticas usa identificadores de seis partes
(`ámbito/mercado/ventana/predictor/brazo/nombre`), así que su ámbito lleva el mercado
y su brazo el predictor. Las selecciones de la cadena de la etapa de adaptadores no
tienen recibo en `jobs/` y se confirman con su `selection.json`. Un trabajo sin
confirmar con una carpeta `attempt-*` o `run` cuenta como intento sin confirmar. El
resumen de la etapa de políticas incluye métricas financieras, que no se copian.

Cada matriz se publica en `windows/<sha256>.json` y el índice lleva solo su etapa, su
declaración, su estado y los recuentos de trabajos, confirmados e intentos. La web
descarga únicamente la matriz de la campaña elegida. Cada brazo lleva el modelo del
catálogo: los adaptadores, el padre congelado y la cadena heredan el de su brazo base,
`titans_*` es Titans-MAC, `mars_titan_*` es MARS-TITAN, `cm_v1_*` es CM-v1 y en las
políticas cuenta el algoritmo o la referencia. Una prueba recorre los planes reales de
las ocho etapas y exige que ningún brazo quede como modelo no identificado.

Recorrer los 13.029 trabajos de los adaptadores de A con la mitad confirmados cuesta
0,11 s, frente a 0,16 s con rutas de `pathlib` (mediana de cinco lecturas en esta
máquina). Con el mismo resumen solo pueden cambiar los intentos abiertos, así que el
recolector reutiliza la matriz durante 60 segundos y la recalcula en cuanto cambia la
identidad del resumen. Las matrices que el índice deja de enumerar se borran de la
salida una hora después de su última escritura, porque una publicación en curso o un
navegador con el índice anterior todavía pueden pedirlas.

## Recolección local

Desde un checkout dedicado al seguimiento, la siguiente orden observa las fuentes
del repositorio y deja estado y salida fuera de sus datos científicos:

```bash
uv run --locked python scripts/collect_observatory.py \
  --root . \
  --state-dir artifacts/observatory-state \
  --output artifacts/observatory-public \
  --watch
```

Sin `--watch` realiza una única recolección. Con él, el intervalo es de 15 segundos.
El bloqueo local impide dos recolectores sobre el mismo estado. SIGTERM y SIGINT
marcan la parada sin tomar bloqueos desde el manejador. Se completa la operación
en curso y, si ya comenzó, la espera local restante del intervalo de 15 segundos.
Una fuente corrupta conserva el índice público
anterior. Las lecturas admiten hasta 2 MiB por JSON, 4096 fuentes y 64 MiB de
contenido acumulado por defecto. `--max-files` permite declarar hasta 65.536
fuentes y registros y `--max-bytes` hasta 1 GiB de contenido acumulado. El límite
individual de 2 MiB por JSON se mantiene. Estas opciones permiten ampliar el
historial con un presupuesto explícito. Si se supera, se informa del error y se
conserva la última publicación válida.

El recorrido excluye los directorios `private` y `checkpoints`. Los estados
privados tampoco se admiten como rutas de lectura explícitas. Solo se consultan
los índices históricos pequeños `checkpoints/latest.json` cuando el recibo no
declara su propio resumen de recuperación. Las rutas de los checkpoints del
ejecutable C++ no se siguen.

El historial conserva las ejecuciones aunque una fuente deje de estar presente.
Los intentos explícitos del coordinador tabular tienen registros separados. Los
informes antiguos sin identidad de intento utilizan `legacy`. No se separan sus
curvas en intentos cuya frontera se desconoce.

La identidad pública se conserva desde la primera observación. Una asociación
privada entre campaña, ruta del recibo y trabajo evita crear otra ejecución cuando
el coordinador incorpora ese recibo a su resumen. Al recuperar una caché anterior,
se consultan los vínculos explícitos del resumen guardado antes de reemplazarlo.
Solo se concilian alias con esa procedencia, conservando los intentos distintos.
Las asociaciones contradictorias revierten la transacción. La tabla de asociaciones
comparte los límites de fuentes y bytes del recolector y no se publica.

La conciliación del 6 de octubre eliminó 578 alias de una copia del historial, de
los que 539 todavía indicaban actividad antigua. Quedaron 2.051 registros, con los
mismos campos e identidades de los recibos confirmados. La
[comprobación de la migración](../../reports/resources/observatory-identity-20261006.json)
documenta los recuentos y las pruebas.

El contrato público de versión 2 utiliza páginas de 64 registros, con máximo de
128 por página. El índice se confirma después de las páginas inmutables cuyos
nombres contienen sus huellas. La web descarga la primera página y solicita las
siguientes al navegar. Los filtros, el CSV y la comparación se aplican a la página
visible. La reserva del test se mantiene en todas las páginas.

Las métricas predictivas publicadas proceden de validación. La comparación usa la
identidad de la fuente, la ponderación, la actividad y el objetivo declarado. Sin
identidad comparable no se asigna grupo. Los registros distinguen corpus real,
escenarios sintéticos y comprobaciones técnicas. El candidato MARS-TITAN no se ejecuta.

## Actividades y validación financiera

Cada registro v2 incluye `activity`: `initial_training`, `supervised_continuation`,
`predictive_adaptation`, `rl`, `synthetic_generation`, `simulation` o `evaluation`.
Las tres primeras admiten curvas y comparaciones predictivas. La continuación
supervisada y la adaptación predictiva conservan grupos distintos del entrenamiento
inicial. Los informes anteriores se clasifican por el método y la etapa del
coordinador. La web sigue leyendo instantáneas v1 y v2 anteriores a este campo.

Los nuevos productores escriben `run.json` con `schema_version: 1`, `activity`,
`model`, `domain` y `status`. Se admiten `factor_world`, `ppo`, `double_dqn` y
`simulator`, las referencias `cash`, `hold_initial` y `rebalance_50`, y el resumen
`financial_comparison`. `domain` debe coincidir con la fuente configurada.
`identity.case`, `identity.model_config` o `identity.config` identifica la configuración.
`identity.objective` separa
objetivos dentro de una misma actividad sin copiar su texto al registro público.
`global_step`, `total_steps`, `samples` y `seed` conservan únicamente contadores
observados. El total no se deduce de la existencia de un proceso.

La fuente `synthetic-worlds-20260924-v2` incorpora los doce mundos generados por el
productor `factor_world`. Su origen es `synthetic` y su actividad es generación,
sin presentarlos como entrenamientos de un modelo predictivo.

La fuente `financial-check-20260924` incorpora seis ajustes, 81 evaluaciones y su
resumen con procedencia `technical`. Sus resultados son comprobaciones del motor.
La identidad de las observaciones, `tape_sha256`, puede sustituir al manifiesto
como base del grupo. La moneda, el coste en puntos básicos, la partición y las
condiciones financieras separan los grupos. La moneda declarada acompaña a los costes.
No se leen operaciones, posiciones ni órdenes para calcular nuevas medidas.

La fuente `native-financial-check-20260924` incorpora los nueve recibos de la
comprobación C++, con procedencia `technical` y `comparison.json` como resumen.
Los resúmenes pueden declarar ejecuciones mediante una lista o un diccionario,
y señalar directamente la ruta de cada `run.json`. El postentrenamiento conserva
su condición, modelo padre y modo. Los recibos que declaran `primary: "median"`
usan las métricas de esa mediana, tanto en el resumen como en las épocas.

RL, simulación y evaluación financiera pueden proporcionar `financial_validation`.
La web muestra estos agregados en una sección propia y los excluye del ranking y
de las curvas predictivas. No exporta pérdidas de PPO o del crítico como MAE.
Solo se admiten los campos siguientes:

| Campo | Contrato |
| --- | --- |
| `net_return` | Retorno neto finito, no inferior a -1. |
| `max_drawdown` | Caída relativa desde el máximo, entre 0 y 1. |
| `costs`, `turnover` | Costes y rotación agregados, finitos y no negativos. |
| `steps` | Pasos evaluados, entero no negativo representable en JavaScript. |
| `completed` | Booleano que indica si se completó el episodio. |
| `invalid_reason` | `missing_close`, `ruined`, `incomplete`, `none` o `null`. |

Estos límites corresponden al motor sin deuda, posiciones cortas ni apalancamiento.
Los valores desconocidos se mantienen como `null`. La falta de un cierre admite
retorno y caída desconocidos, `completed: false` y `missing_close`. La ruina
conocida admite `completed: true`, retorno -1, caída 1 y `ruined`.
No se publican mensajes libres, operaciones individuales, rutas ni pesos.
El bloque se oculta si `final_test_opened` no es `false`, si la partición explícita
no es `validation` o si la fase es `test` o `evaluation`. Una actividad llamada
`evaluation` no libera las fases reservadas ni el test.

Los intentos explícitos pueden declarar `attempt_id`. `checkpoint` admite `step`,
`saved_at` y `resumable`, sin publicar la ruta del archivo. El paso cero es válido.
Anunciar recuperación exige paso y fecha observados. Un índice antiguo sin fecha
permite mostrar su paso, pero no afirmar que el estado sea recuperable.
`parent_frozen` conserva la declaración del productor y no acredita por sí solo
que todos los parámetros permanezcan congelados.

La RAM distingue dos alcances. `executable_peak_rss_bytes` procede de `VmHWM`
después de `exec`, mientras que `process_lifetime_peak_rss_bytes` conserva el máximo
de `getrusage`, que puede incluir picos anteriores del proceso. El valor principal
`metrics.ram_peak_mib` usa el primero cuando está disponible y, si falta, el segundo.
`metadata.ram_peak_scope` identifica la elección. Ambos valores se conservan por
separado en MiB y la interfaz muestra sus etiquetas. El CSV incluye el alcance.
Estos máximos pueden incluir varios casos ejecutados en el mismo proceso.
Las fases reservadas ocultan también estas medidas.

La fecha del informe, la observación del proceso y la fecha de recolección son
campos distintos. Un bloqueo de campaña activo permite observar el proceso,
pero no aumenta épocas ni pasos. La fecha de modificación del recibo se identifica
como tal. Cuando no hay fechas originales por época, `recorded_at` es `null` y el
eje representa épocas. No se reconstruyen horas de entrenamiento.

Desde #451, cada punto del historial predictivo copia también las medidas de la
época que declare el recibo: `train_mae`, `train_samples_per_second` y
`train_seconds` del entrenamiento, y `session_mae` y `validation_seconds` de la
validación. `metrics.session_mae` acompaña al MAE del resumen y sale de la misma validación.
`metadata.best_epoch` y `metadata.stopped_early` copian la selección declarada por
el entrenador. `best_epoch` cuenta las épocas desde 1 y el cero corresponde al estado
inicial del padre. El recolector no calcula ninguna medida nueva. Un valor ausente,
negativo o no finito se publica como `null`, de modo que los recibos anteriores
siguen siendo válidos y la página muestra la ausencia. `metrics.samples_per_second`
sigue procediendo de la validación y la página lo rotula como caudal de la última
validación, separado del caudal de entrenamiento por época.

## Publicación en Pages

La opción `--publish-checkout` recibe un checkout independiente cuya rama debe ser
`observatory-data`. Esa rama contiene únicamente `observatory.json`, las páginas JSON
saneadas y las matrices de las campañas por ventanas. Cada publicación crea un commit
`chore(observatory)` y un push normal. Nunca cambia el checkout científico ni fuerza la
historia remota.

Cada publicación retira de la rama las páginas y matrices que el índice ya no enumera.
Antes se conservaban todas: el 10 de octubre la rama tenía 7.773 páginas y 1,8 GB,
mientras el índice solo enumeraba 51, y el workflow copiaba todas al paquete de Pages.
Las versiones anteriores siguen en la historia de Git, y la página desplegada lee el
índice y sus documentos del mismo commit.

Los cambios se envían cada cinco minutos. Un cambio de estado terminal tiene
prioridad en la siguiente recolección. Los errores de transporte aplazan el envío
con esperas de 15, 30, 60, 120, 240 y hasta 300 segundos. No se considera confirmado
un envío fallido. El intervalo de recolección no depende de que Pages haya acabado
su despliegue anterior.

El publicador solicita el workflow `pages.yml` en `main` con el SHA completo de
los datos. El workflow comprueba que ese commit pertenece a `observatory-data`,
recoge el frontend del SHA que inició el despliegue y prepara únicamente los
archivos estáticos. `deployment.json` identifica ambos commits y la fecha de
preparación del paquete. Actions no ejecuta pruebas, entrenamientos ni recolección.
El código del workflow debe promocionarse a `main` antes de activar los envíos
periódicos.

El grupo de concurrencia permite terminar el despliegue en curso y mantiene una
única ejecución pendiente. Una nueva petición sustituye a la pendiente, sin
cancelar la que ya publica. `cancel-in-progress: false` evita que una sucesión de
estados terminales interrumpa repetidamente el despliegue. Este comportamiento
corresponde al [contrato de concurrencia de GitHub](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

GitHub Pages sirve archivos estáticos. Actualizar el navegador consulta el último
paquete desplegado, sin ejecutar el recolector local. El periodo local no garantiza
la latencia de publicación de GitHub.
[Descripción oficial de Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages).

## Comprobaciones y medidas

Las pruebas cubren caché persistente, recuentos, páginas, fechas ausentes,
ocultación del test, errores de lectura, bloqueos y conservación de intentos.
La prueba de publicación usa un repositorio remoto local. La prueba de navegador
recorre nueve páginas, comprueba que no se precargan y conserva la anterior ante
un HTTP 503. También comprueba la vista móvil.

El [benchmark](../../reports/campaign-observatory-benchmark.json) leyó 526 registros,
con 22,6 MB de JSON en la primera pasada. Tardó 0,847 s en frío y una mediana de
0,573 s en cinco repeticiones posteriores, sin volver a leer el contenido de las
fuentes. El pico de RAM fue 34,4 MiB. Había otra carga científica activa. Son
medidas del recolector, no del entrenamiento, del disco físico ni de la red.

La [cobertura y complejidad](../../reports/campaign-observatory-quality.json)
registran la herramienta y la fórmula de CRAP.
Las [mutaciones dirigidas](../../reports/campaign-observatory-mutations.json)
comprueban que las pruebas detectan la exposición del test, la invención de un
heartbeat, la pérdida de páginas y la omisión de la caché.

La ampliación de actividades verificó 635 registros, incluidos los doce mundos
sintéticos y la comprobación financiera, y diez páginas compatibles con el navegador. Pasaron 110 pruebas
Python, 25 pruebas JavaScript y un recorrido de generación, PPO, simulación y
estados bloqueados en escritorio y móvil. El [informe de calidad](../../reports/observatory-activities-quality.json)
registra cobertura de líneas del 89,6 %, cobertura de ramas del 77,5 %, complejidad,
CRAP y tres mutaciones detectadas sobre protección del test, actividad y procedencia.
El [benchmark de esta ampliación](../../reports/observatory-activities-benchmark.json)
midió una mediana de 0,595 s en cinco pasadas posteriores a la inicial y un pico
de RAM de 35,9 MiB. No se ejecutaron entrenamiento ni pruebas de GPU.

La ampliación experimental del 24 de septiembre de 2026 comprobó 669 registros en
once páginas, incluidos dieciocho recibos nativos de dos comprobaciones conservadas
por separado. Pasaron 132 pruebas Python y 27 JavaScript. Chrome comprobó las
etiquetas y valores distintos de `VmHWM` y del máximo del proceso, la recuperación
declarada y una fixture técnica de continuación neuronal con su condición y padre.
La prueba espera el texto completo de cada página y revisa escritorio y móvil.

El [informe de esta ampliación](../../reports/observatory-experimental-tracking-quality.json)
registra cobertura de sentencias del 91,52 %, cobertura de ramas del 82,71 %,
complejidad y CRAP por función. Tres mutaciones dirigidas fueron detectadas por
pruebas sobre la huella de configuración, el recuento de controles neuronales y
la elección del alcance de RAM. La auditoría comprobó 1345 accesos JSON por pasada,
sin leer Parquet ni estados privados.

La recolección y escritura de páginas tardó 0,971 segundos con una caché nueva y
0,667 y 0,653 segundos en dos pasadas posteriores. La primera leyó 23,2 MB y las
siguientes no volvieron a leer contenido sin cambios. El máximo `VmHWM` del proceso
fue 51,9 MiB, incluidas las herramientas de comprobación. Había otras validaciones
activas, por lo que estas medidas no establecen una mejora frente a las anteriores.

```bash
uv run --locked pytest tests/tooling/test_campaign_observatory.py \
  tests/tooling/test_observatory.py tests/tooling/test_observatory_benchmark.py \
  tests/tooling/test_observatory_publication.py tests/tooling/test_observatory_activities.py \
  tests/tooling/test_observatory_experiments.py
node --test site/tests/state.test.mjs
uv run --locked python scripts/benchmark_campaign_observatory.py \
  --root . --output reports/campaign-observatory-benchmark.json
```

Las pruebas de navegador de la página rehecha en #451 están descritas en
[observatory.md](observatory.md#pruebas-de-la-página). Sustituyen a los recorridos
anteriores, cuyas huellas siguen registradas en los informes de calidad de cada
ampliación.
