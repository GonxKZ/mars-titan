# Supervisión periódica de la campaña

`src/mars_titan/training/campaign_health.py` consulta un servicio de usuario sin iniciarlo, pararlo ni reiniciarlo. Se ejecuta como archivo independiente con la biblioteca estándar. No importa PyTorch ni Arrow y no reserva memoria GPU.

El estado JSON separa la fase declarada y la salud observada. `service_failed` corresponde a un fallo de systemd. `waiting_gpu` requiere un estado reciente del supervisor que declare espera. `preparing_cpu` necesita una preparación activa de la invocación actual. `training` requiere un resumen de actividad con una ejecución declarada activa o un aumento del contador de ajustes completados. La existencia de un contexto CUDA, por sí sola, conserva `running_unknown`. La preparación explícita tiene prioridad sobre la inferencia de entrenamiento.

Los resúmenes de actividad se indican mediante hasta ocho argumentos `--activity`. Pueden ser los resúmenes de los estudios de cada padre o recibos de ejecución con `phase: train`. El monitor consulta exclusivamente esas rutas, el resumen de campaña y su preparación. No recorre los datasets ni busca nuevos recibos. La consulta opcional `--probe-gpu` registra los PID de cómputo que también pertenecen al servicio. No atribuye al servicio la utilización global de otras aplicaciones.

Cada muestra conserva CPU y memoria del cgroup y sus bytes de I/O de almacenamiento cuando están disponibles. Para detectar actividad compara los ticks de CPU y contadores de I/O de los procesos de trabajo que siguen teniendo el mismo PID y hora de inicio. Si se configura el supervisor, se excluyen el PID principal y el intérprete que ejecuta `gpu_supervisor.py`, al igual que `nvidia-smi`. Esto permite reconocer al supervisor cuando uv es el proceso principal. El consumo de sus consultas y la escritura de su heartbeat no prueban avance del trabajo. El contador de I/O de proceso suma bytes lógicos y físicos únicamente para detectar cambios, no para calcular caudal.

Los incrementos de pasos, ajustes completados y particiones confirmadas también cuentan como avance. Una nueva fecha, un cambio de tamaño o un heartbeat no lo hacen. El estado conserva las rutas de origen y los contadores utilizados. Al cambiar la invocación, el PID principal o la identidad de un proceso se establece una referencia nueva. Un reinicio del equipo elimina la referencia anterior.

Tras quince minutos sin cambios comparables se emite `possible_inactivity`. Es una advertencia que requiere inspección, no la afirmación de que el proceso está bloqueado. La espera de GPU y las transiciones de arranque o parada quedan fuera de ese umbral. Si faltan contadores, se declara salud desconocida. Una CPU ocupada tampoco demuestra por sí sola que se estén actualizando modelos.

Un fallo explícito avisa incluso en la primera consulta. Un servicio inactivo solo produce aviso de parada cuando el monitor ya lo ha observado en ejecución y no existe una finalización confirmada. Así se evita avisar al instalar el temporizador antes del primer arranque. La finalización del supervisor debe pertenecer a la invocación observada. La del resumen debe corresponder a la ruta de campaña configurada.

## Ejecución y temporizador

La línea de comandos admite `--service`, `--state`, `--guard`, `--summary`, `--preparation`, `--activity`, `--idle-seconds`, `--probe-gpu` y `--notify`. El servicio predeterminado es `mars-titan-scientific-campaign.service`. `--state` es obligatorio y debe apuntar a un archivo distinto de las entradas. Los resúmenes ausentes son admisibles antes de crear la campaña. Las entradas inválidas y los errores de lectura quedan registrados.

Las unidades de usuario propuestas están en [configs/systemd](../../configs/systemd). La instancia de `mars-titan-campaign-health@.timer` identifica el SHA-256 del módulo instalado. El servicio lee ese archivo desde `~/.local/lib/mars-titan/campaign-health/` bajo una carpeta con la misma huella. Cambiar de versión requiere instalar otra copia y elegir otra instancia del temporizador. No se utiliza el checkout de la campaña científica como fuente del monitor.

El archivo local `~/.config/mars-titan/campaign-health.env` debe definir `MARS_TITAN_GPU_STATE`, `MARS_TITAN_CAMPAIGN_SUMMARY` y `MARS_TITAN_PREPARATION_STATE` con rutas absolutas. Las expansiones de systemd conservan los espacios de cada ruta. `MARS_TITAN_ACTIVITY_ARGS` es opcional y contiene los argumentos `--activity` de los resúmenes concretos. Si sus rutas contienen espacios, cada ruta debe ir entre comillas dentro del valor. No se interpreta mediante un shell.

El temporizador ejecuta una consulta cada cinco minutos. La unidad usa uv sin sincronizar dependencias del proyecto, sin red y sin caché. Tiene un límite de 96 MiB, 16 tareas, una cuota del 10 % de CPU y treinta segundos para completar la comprobación. Solicita aislamiento del sistema de archivos y deja escribible el directorio de estado y su espacio temporal. Estas unidades son una propuesta de instalación, no una prueba de despliegue.

La muestra sustituye atómicamente el JSON anterior con permisos privados. Un bloqueo evita dos escritores concurrentes. El journal recibe las transiciones y un aviso por cambio de incidencia. `--notify` añade un aviso de escritorio mediante `notify-send`, si la sesión local permite mostrarlo. Si falla ese aviso, el journal conserva el error. No se envían mensajes a servicios externos.

## Límites de observación

Cada archivo JSON admite hasta 256 KiB. Se leen como máximo 128 procesos, 16 KiB por archivo de estadísticas, 4 KiB de I/O y 4 KiB de argumentos por proceso y 64 KiB por consulta externa. Cada comando externo tiene un tiempo máximo de cinco segundos. Los procesos que aparecen y desaparecen entre consultas pueden no quedar reflejados en la comparación. Los contadores disponibles y los errores se conservan para distinguir esa falta de evidencia. La ausencia de `io.stat` del cgroup no impide consultar `/proc`.

El estado del supervisor activo debe tener como máximo noventa segundos. La finalización confirmada sigue siendo válida después de ese margen porque el supervisor deja de escribir cuando termina. El monitor no garantiza detectar una interrupción y recuperación completas que ocurran entre consultas. Tampoco sustituye la comprobación científica de checkpoints, datos o resultados.

El observatorio conserva los estados explícitos de campañas sin ejecuciones registradas. Una dependencia solo cambia una campaña en cola a bloqueada. Su configuración apunta a la nueva edición, sin modificar los recibos de intentos anteriores.

La [comprobación local](../../reports/resources/campaign-health-quality.json) registra 249 pruebas CPU y siete mutaciones detectadas. En cinco consultas con la campaña activa, la mediana fue de 0,118 segundos y el máximo de RSS fue de 24,9 MiB. Estas medidas incluyen uv, la consulta de procesos GPU y los resúmenes configurados, fuera de la cuota del servicio propuesto. El recibo del padre RNN permitió comprobar la fase `training` y la salud `ok` con avance de CPU e I/O.
