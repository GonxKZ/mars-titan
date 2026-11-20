# Dos trabajos de la comparación a la vez en la GPU de PC1

La cola de referencias de PC1 ejecutaba un trabajo cada vez. Con un ajuste de red en marcha, la RTX 4070 Laptop trabajaba en torno al 36 % y le quedaba libre más de la mitad de la memoria. Este informe recoge qué se gana al ejecutar dos trabajos a la vez, qué lo limita y qué se ha dejado activo. El cambio está en el lanzador [`run_comparison_resources.py`](../../../scripts/run_comparison_resources.py), que decide cuándo empieza cada trabajo. El runtime científico sigue siendo la copia fija `d5f27f50073d12706c08130a155ea958d822b28b` y no cambian datos, recetas ni operaciones.

## Cómo se midió

Cada ensayo repite un trabajo real de la campaña en un estado de prueba aparte, con su propio permiso, y lanza el mismo proceso `job` del runtime fijo. [`gemelos.py`](../../../benchmarks/comparison_gpu_sharing_20261120/gemelos.py) prepara esos estados, los ejecuta hasta un paso dado mientras muestrea la GPU, la temperatura, la frecuencia de la CPU y la memoria, y coteja el resultado con el trabajo original. El cotejo compara el patrón de bits de cada valor de las trazas de aprendizaje, que estas redes guardan cada 32 pasos y la de áreas cada 64, y de las métricas de cada época, además de la identidad del ajuste. No usa tolerancias. Un valor que difiera en la última cifra binaria cuenta como distinto.

Los tiempos son los que pasan entre dos partes de trazas consecutivas, que cada trabajo cierra en sus barreras. Así se mide igual una prueba y su original. Las medidas completas están en [`measurements.json`](measurements.json).

El equipo es un portátil con AMD Ryzen 9 8945HS, 30 GiB de RAM y RTX 4070 Laptop de 8 GiB, con el controlador 595.99.02, PyTorch 2.14.0 con CUDA 13.0 y el perfil de energía de ahorro. Todos los ensayos son del 20 de noviembre de 2026 y usan trabajos del origen 2019 con la semilla 43. La cola estuvo pausada en los ensayos de las cuatro primeras filas de la tabla siguiente y tuvo un trabajo en marcha en los de las dos últimas.

## Las salidas no cambian

| Ensayo | Trabajo | Hasta el paso | Filas de trazas comparadas | Filas distintas | Épocas comparadas |
| --- | --- | ---: | ---: | ---: | --- |
| Dos copias a la vez con MPS | `asset_groups_shuffled` | 559 | 1.053 en cada copia | 0 | 0 y 1 |
| Dos copias a la vez sin MPS | `asset_groups_shuffled` | 513 | 984 en cada copia | 0 | 0 |
| Junto a otro modelo, con MPS | `area_network` | 513 | 496 | 0 | 0 |
| Con un tope de 3.000 MiB, con MPS | `asset_groups_shuffled` | 513 | 984 | 0 | 0 |
| Con segmentos ampliables y tope de 3.100 MiB | `cross_asset_attention` | 1.118 | 2.090 | 0 | 0 y 1 |
| Reanudado con segmentos ampliables desde el paso 257 | `asset_groups_shuffled` | 1.118 | 2.090 | 0 | 0 y 1 |

En ninguna de las 9.734 filas comparadas cambió un bit. La identidad del ajuste y su `run_id` también coinciden con los del original. La comprobación cubre compartir la GPU con y sin MPS, limitar la VRAM del proceso, cambiar el asignador de memoria de PyTorch y reanudar con el asignador nuevo un ajuste empezado con el habitual.

Son tramos de una o dos épocas de trabajos de diez. Ninguna prueba llegó al final de su ajuste, así que la igualdad del checkpoint final con el de un trabajo en solitario queda sin comprobar.

Las dos copias del primer ensayo acabaron como fallidas por un error del propio ensayo. Se cambió la afinidad de CPU de los procesos en marcha y el entrenador lo trató como un cambio de configuración numérica, porque el número de hilos forma parte de su huella. Ocurrió después de los tramos medidos y solo afectó a los estados de prueba. La afinidad de un trabajo en marcha no se toca.

## Caudal

El factor de caudal suma, para los dos trabajos, el tiempo en solitario dividido entre el tiempo compartiendo la GPU. Un factor de 1 equivale a ejecutarlos uno detrás de otro.

| Trabajos a la vez | Segundos por tramo de 256 pasos | En solitario | Factor de caudal |
| --- | --- | --- | ---: |
| Dos redes de grupos, con MPS | 139,4 y 139,2 en el primero, 169,7 y 174,3 en el segundo | 115,5 y 117,6 | 1,66 y 1,37 |
| Dos redes de grupos, sin MPS | 159,6 y 161,2 en el primero, 194,9 y 189,5 en el segundo | 115,5 y 117,6 | 1,44 y 1,22 |
| Red de áreas y red de grupos con tope, con MPS | 128,0 y 162,3 en el primero, 153,8 y 164,4 en el segundo | 96,1 la de áreas, 115,5 y 117,6 la de grupos | 1,46 y 1,34 |

MPS rinde más que el reparto de tiempo del controlador, que alterna los procesos en la GPU. Por eso la cola compartida exige el demonio.

### La primera pareja de la cola

La primera pareja que formó la cola fue `cross_asset_attention` con `cross_asset_attention_self`, las dos de 2019 con la semilla 43, cada una con un tope de 3.100 MiB y segmentos ampliables. Permite comparar cada trabajo consigo mismo. El primero estuvo solo siete minutos antes de que entrara el segundo, y el segundo siguió solo cuando terminó el primero. Las cifras son medianas de segundos por paso y solo cuentan los tramos que no coincidieron con las comprobaciones de código de este cambio.

| Trabajo | Solo | Con el otro en marcha | Velocidad que conserva |
| --- | ---: | ---: | ---: |
| `cross_asset_attention` | 0,349 en 3 tramos | 0,548 en 7 tramos | 0,64 |
| `cross_asset_attention_self` | 0,364 en 10 tramos | 0,562 en 10 tramos | 0,65 |

El factor de caudal de la pareja es 1,28. Antes de medirla se había fijado conservar una pareja solo si superaba 1,10 en al menos tres tramos limpios de cada trabajo, y lo cumple. Las dos redes de grupos midieron más en los ensayos, pero su pareja aún no se ha formado en la cola.

Con los dos trabajos, el uso medio de la GPU pasó del 29 al 36 % y su potencia de 33 a 41 W. La tarjeta llegó a 6.324 MiB ocupados y cada proceso a 2.856. La GPU alcanzó 75 °C y la placa 70, frente a 69 y 64 con un trabajo. La RAM disponible bajó hasta 6,0 GiB.

## La CPU limita la ganancia

Con dos trabajos la GPU siguió lejos de saturarse. En la pareja de la cola, los cuatro núcleos más rápidos promediaron 2,35 GHz con un trabajo y 1,62 con dos, mientras que la media de los 16 hilos apenas cambió (1,30 y 1,25 GHz). Cada paso del ajuste depende sobre todo de un hilo de Python, así que esa bajada de frecuencia se traslada casi entera al tiempo por paso.

La explicación más probable es el límite de potencia del perfil de ahorro de energía, que repartiría entre dos procesos casi la misma energía que antes usaba uno. No está confirmada, porque leer el consumo del procesador exige permisos de administrador.

De ahí salen dos consecuencias. La ganancia de compartir la GPU es moderada y depende de cuánto use la GPU cada modelo. Y cualquier otra carga de CPU frena los entrenamientos. En el ensayo de reanudación, que coincidió con las mutaciones del lanzador lanzadas con prioridad baja, los tramos de la red de grupos duraron entre 207 y 306 segundos, frente a los 162 a 164 del ensayo con la red de áreas y el equipo en calma. El trabajo que la acompañaba no era el mismo, así que la diferencia solo es orientativa. Las medidas de coste de cada modelo deben hacerse con el equipo libre.

No se ha cambiado el perfil de energía ni ningún límite del equipo. El perfil de ahorro se mantiene por decisión previa, tras apagados por temperatura con otros perfiles.

## Un tope ajustado falló por fragmentación

La primera versión de los presupuestos ponía a las redes medianas un tope de VRAM menor que la reserva habitual de PyTorch, que ronda los 3.200 MiB con un pico asignado de 2.500. En el ensayo previo, un tope de 3.000 MiB había aguantado dos tramos sin cambiar el resultado. En la cola, `cross_asset_attention` de 2019 con la semilla 43 falló a los 391 segundos con un tope de 3.050 MiB. PyTorch tenía 2.376 MiB asignados y otros 446 reservados sin usar, repartidos en huecos que no admitían un bloque de 64 MiB. El controlador de la campaña lo anotó como intento fallido y lo reintentó desde su barrera del paso 512. Después se pausó en una barrera para lanzarlo sin ese tope. Es el único intento fallido que ha causado este trabajo.

Los topes ya no bajan de la reserva medida. Para que dos redes medianas quepan con margen se usa la opción `expandable_segments:True` del asignador de PyTorch, que amplía un mismo segmento en vez de pedir uno por tamaño y evita esos huecos. Solo decide dónde se guarda cada tensor.

| Trabajo | Reserva habitual de PyTorch | Proceso con el asignador habitual | Reserva con segmentos ampliables | Proceso con segmentos ampliables |
| --- | ---: | ---: | ---: | ---: |
| `cross_asset_attention` | hasta 3.252 MiB | unos 3.500 MiB | 2.642 MiB | 2.856 MiB |
| `asset_groups_shuffled` | hasta 3.214 MiB | 3.464 MiB | 2.574 MiB | 2.824 MiB |

La reserva habitual es el máximo de los trabajos completos de la campaña, y el proceso de `cross_asset_attention` con el asignador habitual es una estimación a partir de esa reserva. Las cifras con segmentos ampliables de la tabla son de los ensayos de dos épocas. En la cola, `cross_asset_attention` repitió esos mismos picos en sus seis últimas épocas y `cross_asset_attention_self` reservó 2.642 MiB en su ajuste completo. El tope declarado es de 3.100 MiB por proceso, unos 250 más que lo medido.

## Qué queda activo

La cola de PC1 se ejecuta desde las 16:08 del 20 de noviembre con los presupuestos de [`comparison-campaign-packing-rtx4070-laptop.json`](../../../configs/experiments/comparison-campaign-packing-rtx4070-laptop.json). Solo tienen tope las cuatro redes medianas de la tanda de comparación (`asset_groups`, `asset_groups_shuffled`, `cross_asset_attention` y `cross_asset_attention_self`), con 3.100 MiB por proceso y segmentos ampliables. Dos de ellas caben a la vez en el presupuesto de 7.075 MiB cuando son del mismo origen. La cola usa cuatro trabajadores, pero `max_jobs` limita a dos los trabajos simultáneos. Los demás modelos cuentan con el valor por defecto de su ejecutor, que no deja sitio a un segundo trabajo, y se ejecutan solos, sin tope y con el asignador habitual, igual que antes.

Con el factor medido, cada origen de ocho redes ahorraría en torno a 0,7 de las 7,2 horas que tarda en serie, unas 7 horas en los diez orígenes pendientes. Es una estimación a partir de los tiempos en solitario de los trabajos ya terminados, no un resultado.

Las redes de áreas quedan fuera por ahora. Su reserva habitual llega a 4.004 MiB y no cabe junto a una mediana, y con segmentos ampliables no se ha medido su pico ni su identidad. La pareja de una red de áreas con una de grupos midió entre 1,34 y 1,46 con la cola pausada, así que es la ampliación con más recorrido. Tampoco se han medido los modelos pequeños de la tanda principal. Las dos ampliaciones y la repetición de un ajuste completo quedan en la tarea #880.

El lanzador, además, no añade un trabajo sin lecturas de VRAM, RAM y temperatura, espera a que el anterior se asiente, solo junta trabajos del mismo origen porque comparten el archivo de entradas preparado, y no lanza nada si el demonio MPS no está vivo. Los presupuestos se vuelven a leer en cada decisión y `packing-log.jsonl` anota cada solape en el estado de la campaña.

## Comprobaciones

Las 101 pruebas del lanzador y las 30 de la herramienta de medida pasan. Ruff, el formato y la comprobación del repositorio pasan.

Se aplicaron 99 mutaciones dirigidas al lanzador y 58 a la herramienta, y las pruebas detectan todas, con los controles sin mutar en verde antes y después. En las primeras pasadas seis no se detectaron. En el lanzador, tomar el proceso del primer intento en vez del último no cambiaba la admisión con los datos de la prueba. Otras dos, no liberar la plaza de un trabajo terminado y esperar memoria con el límite del grupo en vez de con la RAM de admisión, dejaban la prueba sin terminar en lugar de hacerla fallar. En la herramienta no se distinguían conservar solo la última lectura de VRAM, preparar un estado sobre una carpeta vacía ya existente y no enviar la señal de parada. Las pruebas se ampliaron hasta que todas fallan con cada mutación. Los [recibos](mutations.json) guardan cada resultado.

La [cobertura](quality.json), medida con coverage.py 7.16.1 y la complejidad de Radon 6.0.1, es de 356 de 367 sentencias y 122 de 130 ramas en el lanzador, y de 230 de 231 sentencias y 75 de 76 ramas en la herramienta. El CRAP más alto es 28,00 en `run_worker` del lanzador, que en el [recibo del 17 de noviembre](../comparison-job-resources-20261117/quality.json) ya tenía 25,79 con complejidad 24 y ahora suma las tres opciones nuevas del trabajador. En la herramienta es 20,0 en `compare_run`, con todas sus sentencias recorridas. De las 460 líneas ejecutables que cambia este trabajo en los dos archivos, las pruebas recorren todas. Estas cifras localizan complejidad poco comprobada. No demuestran ausencia de errores. Las partes que lanzan procesos reales se comprueban con procesos simulados y con los ensayos de este informe.

## Límites

- Los ensayos cubren una o dos épocas de trabajos de un solo origen. No se ha repetido ningún ajuste completo ni se ha comparado un checkpoint final.
- La identidad se comprobó en `asset_groups_shuffled`, `area_network` y `cross_asset_attention`. `asset_groups` y `cross_asset_attention_self` se emparejan por ser variantes de esas redes con los mismos picos de memoria, sin ensayo propio.
- El factor de caudal depende de la pareja y del resto de la carga del equipo. No se extiende a modelos sin medir, que siguen ejecutándose solos.
- El tiempo de un trabajo que compartió la GPU no es su coste en solitario. `packing-log.jsonl` permite saber cuáles se solaparon.
- Los picos de VRAM con segmentos ampliables se han medido en dos modelos. Si un trabajo supera su tope fallará ese intento y la cola lo reintentará desde su última barrera.
- MPS añade un proceso del que dependen los trabajos. Un fallo grave de GPU en uno puede cerrar el servidor y con él los demás.
- No se ha medido energía. La potencia de GPU es la que informa `nvidia-smi` y el consumo del procesador no se ha podido leer.

## Reproducción

Con las variables de la campaña exportadas como en su arranque habitual, el demonio MPS en marcha y la cola pausada o con memoria libre suficiente:

```bash
read -r -p "Copia científica fija: " runtime
read -r -p "Estado de la campaña: " campaign
read -r -p "Carpeta nueva para la prueba: " trial
tool=benchmarks/comparison_gpu_sharing_20261120/gemelos.py
key=cross_asset_attention/2019/seed-43
uv run --no-sync python "$tool" prepare --key "$key" --runtime "$runtime" \
  --campaign "$campaign" --state "$trial" --note "Prueba de GPU compartida, fuera de la campaña"
uv run --no-sync python "$tool" run --key "$key" --runtime "$runtime" --state "$trial" \
  --until-step 1118 --mps --vram-limit-mib 3100 --alloc-conf expandable_segments:True \
  --thermal-zone /sys/class/thermal/thermal_zone0/temp --out "$trial/medida.json"
uv run --no-sync python "$tool" compare --key "$key" --campaign "$campaign" --state "$trial"
```

`compare` termina con código 1 si alguna prueba no es idéntica al original. Repetir `--state` en `run` lanza varias copias a la vez. La [verificación](verification.json) reúne los recuentos de esta revisión y el estado de cada criterio de la tarea. No aporta resultados predictivos ni abre el test final.
