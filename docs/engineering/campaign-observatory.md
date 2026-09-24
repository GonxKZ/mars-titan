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
terminan tras la operación en curso. Una fuente corrupta conserva el índice público
anterior. Las lecturas admiten hasta 2 MiB por JSON, 4096 fuentes y 64 MiB de
contenido acumulado. Si se supera el presupuesto se informa del error.

El historial conserva las ejecuciones aunque una fuente deje de estar presente.
Los intentos explícitos del coordinador tabular tienen registros separados. Los
informes antiguos sin identidad de intento utilizan `legacy`. No se separan sus
curvas en intentos cuya frontera se desconoce.

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

La fecha del informe, la observación del proceso y la fecha de recolección son
campos distintos. Un bloqueo de campaña activo permite observar el proceso,
pero no aumenta épocas ni pasos. La fecha de modificación del recibo se identifica
como tal. Cuando no hay fechas originales por época, `recorded_at` es `null` y el
eje representa épocas. No se reconstruyen horas de entrenamiento.

## Publicación en Pages

La opción `--publish-checkout` recibe un checkout independiente cuya rama debe ser
`observatory-data`. Esa rama contiene únicamente `observatory.json` y las páginas
JSON saneadas. Cada publicación crea un commit `chore(observatory)` y un push
normal. Nunca cambia el checkout científico ni fuerza la historia remota.

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

```bash
uv run --locked pytest tests/tooling/test_campaign_observatory.py \
  tests/tooling/test_observatory_publication.py
node --test site/tests/state.test.mjs
uv run --locked python scripts/benchmark_campaign_observatory.py \
  --root . --output reports/campaign-observatory-benchmark.json
```
