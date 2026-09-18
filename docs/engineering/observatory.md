# Observatorio local de MARS-TITAN

El observatorio permite consultar el estado de las ejecuciones sin abrir sus datos ni sus checkpoints. El exportador de `scripts/export_observatory.py` transforma estados agregados locales en una instantánea JSON pública. No entrena modelos, calcula resultados científicos, abre registros completos ni publica archivos en Internet.

La interfaz puede existir antes que el entrenador. Mientras no haya estados registrados, la salida contiene el catálogo de modelos, `source_status: "no_runs_registered"` y una lista vacía de ejecuciones. No se generan métricas, curvas o entrenamientos de demostración como si fueran observaciones reales.

La [página pública](https://gonxkz.github.io/mars-titan/) quedó desplegada mediante [PR #73](https://github.com/GonxKZ/mars-titan/pull/73). El [despliegue de Pages](https://github.com/GonxKZ/mars-titan/actions/runs/35396836916) terminó correctamente y se comprobaron sus cinco archivos por SHA-256 frente a la versión local. La vista pública se revisó en escritorio y móvil, sin errores de ejecución. El estado inicial contiene cero ejecuciones y nueve modelos planificados.

## Uso manual

Desde la raíz del repositorio:

```bash
uv run --locked python scripts/export_observatory.py \
  --input-dir artifacts/runs \
  --output site/data/observatory.json
```

`--output` es obligatorio. La entrada predeterminada es `artifacts/runs`, dentro de un directorio excluido de Git. Se leen únicamente archivos `artifacts/runs/<run_id>/status.json`, sin recorrer subdirectorios adicionales ni seguir enlaces simbólicos. Una fuente todavía inexistente produce el estado sin ejecuciones. Un archivo presente pero inválido provoca un error.

La orden usa la biblioteca estándar de Python y el entorno gestionado por uv. No requiere PyTorch, CUDA, dependencias adicionales, conexión de red ni un servicio instalado. La generación y la publicación son acciones separadas. Generar este JSON no ejecuta Git, no hace un push y no activa GitHub Actions. La primera integración se realiza manualmente.

El exportador utiliza descriptores de archivo POSIX y está comprobado en Linux. No se declara compatibilidad con Windows sin una implementación y una verificación específicas. El frontend es independiente de esa plataforma y se ejecuta en el navegador.

## Contrato privado de entrada

Cada `status.json` contiene un objeto con `schema_version: 1` y tres identificadores obligatorios: `run_id`, `attempt_id` y `model_id`. `run_id` debe coincidir con el nombre del directorio. Los identificadores públicos admiten letras ASCII, dígitos, puntos, guiones y guiones bajos, empiezan por letra o dígito y tienen hasta 96 caracteres. No deben contener información personal, secretos o rutas.

El catálogo admitido es B0, B1, B2, B3, B6, M0, M1, M2 y M3. B0-B3 y B6 son referencias, M0-M2 son ablaciones y M3 identifica la propuesta compacta. La pertenencia al catálogo no significa que un modelo esté implementado o evaluado.

Los demás campos son opcionales. Una observación desconocida se conserva como `null`. Un estado conocido debe pertenecer a `queued`, `running`, `paused`, `completed`, `failed` o `cancelled`. Las fases admitidas son `prepare`, `train`, `validation`, `calibration`, `test` y `evaluation`. Un texto diferente no se convierte en desconocido, se rechaza como dato inválido.

El siguiente fragmento ilustra el contrato, no una ejecución realizada:

```json
{
  "schema_version": 1,
  "run_id": "run-example",
  "attempt_id": "attempt-01",
  "model_id": "B1",
  "variant_id": "ridge-example",
  "status": null,
  "phase": null,
  "heartbeat_at": null,
  "started_at": null,
  "updated_at": null,
  "completed_steps": null,
  "total_steps": null,
  "epoch": null,
  "max_epochs": null,
  "seed": null,
  "fold": null,
  "comparison_contract": {
    "target": "target-version",
    "universe": "universe-version",
    "splits": "splits-version",
    "evaluation_regime": "policy-version"
  },
  "metrics": {},
  "history": [],
  "checkpoint": {"step": null, "saved_at": null, "resumable": null}
}
```

Las fechas necesitan zona horaria y se normalizan a UTC. No se aceptan fechas futuras respecto a la exportación. El inicio, la actualización y los puntos del historial deben conservar un orden coherente. Esto exige que el reloj del productor y el del exportador estén sincronizados.

Los contadores son enteros no negativos que puedan representarse exactamente en JavaScript. El progreso no puede superar el total declarado. La época no puede superar el máximo y el paso del checkpoint no puede ir por delante del progreso confirmado. `checkpoint.resumable: true` exige un paso y una fecha observados, incluido el paso cero si corresponde a un guardado real. La ausencia de cualquiera de esos datos impide anunciar capacidad de recuperación.

El productor entrega el historial agregado del intento y la fase actuales. Cada punto necesita `step` y `recorded_at`, con pasos estrictamente crecientes. Puede incluir `loss` y `mae`. Si incluye `attempt_id` o `phase`, deben coincidir con la raíz. Al reanudar, el productor cambia `attempt_id` y comienza una curva separada. El exportador no mezcla archivos anteriores ni reconstruye curvas a partir del registro privado.

`comparison_contract` contiene versiones o huellas estables del target, universo, particiones y régimen de evaluación. El exportador calcula `comparison_group` a partir de esos cuatro campos, el fold y la fase. Si falta el contrato, el fold o la fase, el grupo es `null`. Una etiqueta `comparison_group` recibida en la fuente se ignora. Esto permite impedir clasificaciones de MAE o Rank IC entre tareas distintas, aunque el productor sigue siendo responsable de que esas identidades describan los experimentos reales.

## Salida pública y reserva del test

La identidad del régimen de evaluación debe resolver también agregaciones, ventanas, unidades y protocolo de medida. Por ejemplo, MAE por sesión no se mezcla con MAE global por fila. Una medida RSS no se etiqueta como PSS y una latencia solo de la cabeza no se compara con una ruta que incluye codificación. Los parámetros propios de una variante no sustituyen la identidad del protocolo común.

La salida v1 contiene `schema_version`, `project`, `generated_at`, `source_status`, `poll_interval_seconds`, `stale_after_seconds`, `models`, `runs` y `notes`. La consulta sugerida es cada 60 segundos y la antigüedad de referencia es 180 segundos. El catálogo solo contiene identificador, nombre y clase de cada modelo.

Cada ejecución exporta su identidad, intento, modelo, variante, estado, fase, fechas, progreso, época, semilla, fold, grupo de comparación, métricas, historial, resumen del checkpoint y `test_released`. Las métricas permitidas son:

| Campo | Significado y validación |
| --- | --- |
| `mae`, `mse` | Valores finitos no negativos, sin calcularlos de nuevo. |
| `loss` | Valor finito con signo. Una NLL de densidad continua puede ser negativa. Su definición pertenece al experimento. |
| `rank_ic` | Valor finito entre -1 y 1. |
| `coverage_80`, `coverage_95` | Proporciones entre 0 y 1, no porcentajes de 0 a 100. |
| `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms` | Latencias no negativas en milisegundos. |
| `vram_peak_mib`, `ram_peak_mib` | Memoria observada no negativa en MiB. |
| `elapsed_seconds`, `samples_per_second` | Tiempo y caudal observados no negativos. |

Una métrica ausente se publica como `null`, nunca como cero. Los booleanos no se aceptan como números. El exportador descarta campos adicionales en la raíz, las métricas, el historial y el checkpoint. No copia mensajes de error, rutas, notas privadas, tokens ni contenido de entrenamiento. No interpreta código o variables incluidos en el JSON.

Las fases `test` y `evaluation` permanecen selladas por defecto. La segunda también se protege porque puede contener el resumen del test final. Una fase desconocida oculta igualmente sus métricas. En estos casos, todos los valores de `metrics` son `null`, `history` es una lista vacía y `test_released` es `false`. Una marca `test_released: true` en la fuente no concede permiso.

Solo una decisión explícita permite ejecutar:

```bash
uv run --locked python scripts/export_observatory.py \
  --input-dir artifacts/runs \
  --output site/data/observatory.json \
  --release-test
```

La opción libera únicamente las fases reservadas de ejecuciones en estado `completed`, `failed` o `cancelled`. Si alguna ejecución reservada sigue activa, en pausa, en cola o sin estado conocido, la exportación falla y conserva la instantánea anterior. Esta opción se usa después del cierre científico y la decisión de divulgar resultados. No sirve para seguir el rendimiento parcial del test durante la selección. La herramienta comprueba la opción y el estado, pero no acredita por sí misma esa decisión científica.

La interfaz debe separar estado declarado y salud observada. Un `running` con `heartbeat_at` ausente o antiguo no demuestra actividad actual. El exportador conserva el estado declarado y añade una advertencia de salud de texto fijo. La interfaz debe mostrar actividad no confirmada o estado antiguo. No debe inventar `paused`, sustituir datos desconocidos por ceros o afirmar que un proceso está vivo solo porque la exportación es reciente.

## Límites y publicación íntegra

La instantánea no es el archivo maestro. El exportador conserva el intento actual de cada ejecución seleccionada. MT-031 deberá guardar todos los intentos y eventos en el registro privado, aunque el resumen visible cambie. Si la campaña supera el límite de ejecuciones, se preparará un directorio de estados seleccionados para `--input-dir`, por ejemplo por campaña o periodo, sin borrar ni mover los originales para satisfacer el límite. La utilidad falla ante exceso en vez de ocultar filas silenciosamente. El frontend puede mostrar varios intentos de un resumen agregado válido, pero esta primera utilidad no reconstruye ese archivo desde los logs completos.

Los límites predeterminados son 128 ejecuciones, 256 KiB por estado, 500 puntos públicos por historial, 8 MiB de salida y cinco segundos de presupuesto de procesamiento. La profundidad máxima del JSON es 16 y se inspeccionan como máximo cuatro veces el límite de ejecuciones en entradas directas del directorio. Se rechazan claves JSON duplicadas, archivos especiales y enlaces simbólicos. No se leen los logs completos.

`--max-runs`, `--max-file-bytes` y `--max-output-bytes` permiten reducir los límites. `--timeout-seconds` admite un valor positivo de hasta 30 segundos. El presupuesto se comprueba durante el procesamiento y antes del reemplazo. No constituye una garantía de respuesta ante un sistema de archivos bloqueado por el sistema operativo.

La salida debe estar fuera del directorio de estados privados. Los componentes de las rutas se abren sin seguir enlaces. El exportador valida todos los estados antes de escribir, crea un temporal en el directorio de destino, sincroniza sus bytes y reemplaza el nombre final atómicamente. Después sincroniza el directorio. Ante un dato inválido o un límite excedido conserva la instantánea previa. Los temporales se limpian si una operación falla.

La salida se ordena por identidad de ejecución e intento y sus claves JSON se ordenan de forma estable. Con la misma entrada y fecha de exportación, el resultado es idéntico. El productor también debe publicar su `status.json` de forma atómica para evitar que una lectura coincida con una escritura parcial.

## Integración futura con MT-031

MT-031 mantiene el registro completo de eventos, la persistencia y las identidades. Su callback entrega un resumen pequeño a una cola acotada. Un consumidor fuera del camino de entrenamiento actualiza `status.json` con los últimos valores confirmados y el historial de la fase e intento actuales. Si hay saturación, puede sustituir un resumen pendiente por otro más reciente. Eso no autoriza a descartar eventos científicos del registro privado.

El callback no debe ejecutar `cuda.synchronize()` ni `.item()` en cada batch para alimentar la web. Debe reutilizar las métricas obtenidas en los puntos de registro ya previstos, limitar la frecuencia y separar la serialización y exportación de la GPU. El heartbeat puede actualizarse desde un componente ligero sin fabricar progreso. El historial público acotado no sustituye al historial científico completo.

No se introduce un segundo formato de checkpoint. `checkpoint` solo informa del paso, fecha y capacidad de reanudación declarados por MT-065. La página no puede validar ni restaurar esos pesos. Tampoco se requiere un servicio Go, una cola remota o una instalación permanente para este seguimiento local.

## Verificación realizada

La revisión final pasó 109 pruebas Python del repositorio, incluidas 55 del exportador, y 21 pruebas JavaScript. La prueba real de navegador ejercitó importación, filtros, CSV, teclado, móvil, impresión, recuperación de errores, caducidad y límites. Las correcciones de contrato se contrastaron mediante CLI real a Node y a navegador, no solo con fixtures independientes.

El [diagnóstico de complejidad y cobertura](../../reports/observatory-quality.json) registra la huella del exportador, Radon 6.0.1 y coverage.py 7.16.1. La cobertura de sentencias observada es del 87,38 % y la de ramas del 86,07 %. La estimación CRAP usa cobertura de sentencias por función, no cobertura de caminos base. El máximo observado de esa variante es 24,03. Es una señal para orientar revisión, no una garantía de corrección ni una medición de rendimiento. Las invocaciones de CLI se prueban en subprocesos, pero no están instrumentadas en esa captura de cobertura del proceso principal.

Las pruebas de `tests/tooling/test_observatory.py` ejercitan la API y la CLI sobre directorios temporales. Cubren ausencia de ejecuciones, campos permitidos, reserva del test, estados inválidos, fechas, progreso, valores finitos, grupos de comparación, cambios de intento, tamaño, profundidad, enlaces y conservación del archivo anterior. Una prueba ejecuta la CLI y pasa su JSON a `validateSnapshot` de `site/state.mjs` con Node.js, incluyendo datos desconocidos, pérdida con signo y test reservado. Node.js es necesario para esa verificación de integración, pero no para ejecutar el exportador. Las fixtures permanecen en directorios temporales y no se publican. Las pruebas se ejecutan con:

```bash
uv run --locked pytest tests/tooling/test_observatory.py
```

Se probaron tres mutaciones dirigidas sobre copias temporales del módulo. Desactivar el sellado provocó tres fallos, copiar el diccionario privado produjo un fallo de filtración y omitir la validación de estados y fases provocó dos fallos. La implementación original permaneció intacta. Esta comprobación fue dirigida y no equivale a ejecutar una campaña con mutmut.

El 18 de septiembre de 2026 se midió la utilidad en CPU con Python 3.12.14 y una fixture sintética de 32 estados, cada uno con 500 puntos. Cinco invocaciones reales de la CLI, incluyendo el arranque de Python, tardaron entre 0,1009 y 0,1033 segundos, con mediana de 0,1014 segundos. El máximo RSS de los procesos hijos fue de 26.828 KiB y la salida ocupó 1.430.395 bytes. La fixture se creó fuera del repositorio y se eliminó al terminar. Estos datos miden una exportación local acotada. No representan coste de entrenamiento, latencia de MARS-TITAN ni rendimiento de la GPU.

La [repetición sobre la versión final](../../reports/observatory-benchmark.json), con el mismo tamaño de muestra sintética y cinco invocaciones, observó una mediana de 0,1026 segundos, máximo RSS hijo de 26.956 KiB y salida de 1.435.027 bytes. La diferencia de tamaño responde a la versión del catálogo y del contrato, no a datos financieros. Tampoco mide el efecto de ejecutar la utilidad a la vez que un entrenamiento. Esa comparación pertenece a la integración de MT-031.
