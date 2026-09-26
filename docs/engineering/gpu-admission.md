# Admisión de memoria para la cola científica

El supervisor de `training/gpu_supervisor.py` ejecuta un comando externo sin importar PyTorch ni reservar VRAM. Consulta la GPU 0 mediante `nvidia-smi` y conserva una instantánea operativa pequeña. Puede ejecutarse como archivo independiente para supervisar una revisión científica congelada sin cambiar sus módulos ni las huellas de sus checkpoints.

Antes de arrancar exige, por defecto, 6656 MiB libres. Durante la ejecución solicita una pausa si quedan menos de 768 MiB o aparece un proceso de cómputo CUDA ajeno al grupo que ha creado. Los procesos que el controlador identifica como gráficos o mixtos siguen incluidos en la memoria total observada. No se excluyen aplicaciones por su nombre. Una lectura inválida del controlador impide admitir trabajo y provoca una pausa si ya estaba ejecutándose.

La consulta ordinaria se repite cada cinco segundos. El despliegue local de esta cola utiliza dos segundos. El supervisor envía SIGTERM únicamente a su propio grupo de procesos y espera hasta 600 segundos para que termine de confirmar el estado. También espera a los descendientes cuando el lanzador ha terminado antes. Si se necesita SIGKILL por exceder ese tiempo, declara la pausa bloqueada y no vuelve a lanzar el comando.

Tras una pausa confirmada espera al menos 30 segundos y vuelve a comprobar el margen completo de arranque. Se permiten como máximo tres pausas durante una ejecución del supervisor. Cualquier salida fallida del comando queda registrada y requiere revisión. El servicio debe utilizar `Restart=no`, evitando un ciclo de reinicios externos que anule esa decisión.

Para systemd se utiliza `KillMode=mixed` y un tiempo de parada superior a los 600 segundos internos. Así SIGTERM llega al supervisor, que coordina la parada del grupo. El servicio conserva su arranque al iniciar sesión. Los límites de memoria del servicio y la recuperación científica siguen teniendo sus contratos propios.

El estado operativo se sustituye atómicamente como máximo cada 15 segundos cuando no cambia de fase. El diario recibe las transiciones, no una línea por consulta. Un bloqueo de archivo evita dos supervisores sobre el mismo estado. El comando supervisado debe permanecer en primer plano y gestionar SIGTERM como una solicitud de checkpoint y salida coherente.

## Incidente comprobado y límites

El 26 de septiembre de 2026, el caso de XGBoost con profundidad 6 y 256 bins agotó la VRAM durante la actualización de árboles. Uno de los mensajes indicó 114,438 MB libres frente a una petición de 256 MB. La telemetría conservó una carga concurrente de inferencia y unos 6001 MiB ocupados incluso después de terminar el proceso científico. El servicio anterior reintentó seis veces. Al desaparecer esa carga, la campaña terminó sus 17 casos sin modificar el diseño.

Se comprobó por separado la vida de matrices, iteradores y modelos con seis ajustes pequeños, tres interrupciones y recorridos completos con recuperación. No quedaron objetos vivos ni bytes activos en los pools observados después del retorno. La reserva asíncrona se estabilizó en 96 MiB. Esa reserva es coherente con la [caché de memoria de CuPy](https://docs.cupy.dev/en/stable/user_guide/memory.html). La comprobación no justifica añadir vaciados del pool por lote ni modificar el algoritmo.

La [evidencia local](../../reports/resources/gpu-admission-quality.json) distingue pruebas de procesos, comprobaciones CUDA y cargas pequeñas de la campaña completa. Los resultados científicos anteriores no se reetiquetan como ejecuciones del nuevo supervisor. El test final permanece sellado.

La supervisión es preventiva, no una reserva exclusiva del dispositivo ni una garantía de ausencia de OOM. Otra aplicación puede asignar memoria entre dos consultas, y una operación propia puede necesitar más espacio del previsto. El margen, la pausa y la ausencia de reintentos ciegos reducen esos riesgos sin falsear los fallos registrados. La comprobación de capacidad no añade una cuota de CPU ni modifica el perfil energético.
