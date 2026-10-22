# Programas regionales anteriores a MAC

Se han implementado dos calendarios finitos sobre el mismo operador regional. `summary_at_end_v1` ejecuta cuatro pasos locales y un resumen final. `summary_interleaved_v1` ejecuta dos grupos de dos pasos locales, cada uno seguido de un resumen. La selección puede ser fija o depender de la norma del contexto confirmado, con un umbral explícito. El programa no añade parámetros ni amplía las 36 acciones del catálogo de mecanismos. Sus ciclos son un eje distinto de K y J del lector episódico.

La consulta del contexto y la preparación de su escritura están separadas. La consulta utiliza el prefijo confirmado y la escritura pendiente recibe una sola vez el token fusionado final. Las trazas conservan programa, norma, fases y trabajo por fila, también en la colección de alternativas. El verificador reconstruye la selección en la precisión declarada y exige norma cero en las filas sin consulta al padre. Comprueba coherencia estructural y contable, sin autenticar un contexto no guardado ni demostrar por sí solo qué kernels se ejecutaron.

La reserva durable admite el trabajo de toda la cohorte antes de emitir predicciones. Un intento abandonado conserva su cargo. La recuperación distingue una emisión ya confirmada de un intento pendiente y la compactación conserva evidencia en el archivo de mecanismos. Las unidades reservadas son cotas y se mantienen separadas de las fases observadas.

La [especificación](../../../docs/experiments/regional-program-v1.md) y la [declaración de configuración](../../../configs/experiments/regional-program-v1.json) describen estos contratos. La implementación inicial corresponde a `04a13c6fa`, el libro de reservas a `a7785b4c1` y las pruebas adicionales de compresión a `6e2e91cb3`. La corrección de identidades está en `c598ea961565c95f8a554bd0fafdf972c51e4a52` y el ajuste de formato de las pruebas en `1a076f90d`. [source-manifest.json](source-manifest.json) conserva las huellas de las fuentes medidas.

## Verificación, mutación y cobertura

La selección final reúne 805 casos únicos aprobados, sin fallos, errores ni omisiones. Incluye rutas CPU, CUDA y persistencia nativa. No es la suite completa del repositorio.

| Selección | Casos únicos |
| --- | ---: |
| Núcleo, contexto e integración del predictor | 608 |
| Libro de reservas y recuperación | 70 |
| Sesión financiera y colección | 115 |
| Compresión y recuperación del archivo | 12 |
| Total | 805 |

[verification.json](verification.json) conserva los comandos ejecutados, los identificadores y los recibos de cada selección. Los intentos anteriores quedan separados del total final. El primer pase conjunto de cobertura de sesión agotó 180 segundos y quedó incompleto. Las comprobaciones posteriores detectaron un desbordamiento al convertir un umbral entero finito extremo a FP32 y ajustaron su tratamiento, además de fijar en CPU la conversión usada para contrastar la norma FP32 archivada. El primer benchmark del núcleo también falló antes de completar registros porque llamaba al método inexistente `config_id()`. Se corrigieron las dos llamadas a `_config_id()` y se ejecutó otra salida completa. Los intentos fallidos e incompletos quedan fuera de los resultados aprobados.

La revisión adicional reprodujo ocho fallos al contrastar umbrales `1` y `1.0`. La igualdad numérica de las configuraciones no conservaba su identidad serializada. La corrección compara identidades canónicas en el constructor e incluye el tipo del umbral en el sellado del predictor, para rechazar discordancias de catálogo y sustituciones de configuración antes de ejecutar. Los ocho casos están incluidos en los 608 del núcleo que pasaron después de esta corrección.

Se detectaron 15 mutantes, los 12 anteriores y tres añadidos para las guardas de identidad. Cubren lectura y escritura del contexto, gradientes, conservación del contexto entre ciclos, fases, trabajo, recuperación de cuotas, igualdad canónica y tipo del umbral. Las versiones mutadas provocaron 73 fallos en la fase `call` de las pruebas. Los controles de origen y restaurados suman 81 ejecuciones de casos cada uno al seleccionar la ejecución final de cada identificador. Dos mutantes del ejecutor se habían repetido sobre su fuente final, cuyo SHA-256 comienza por `64b4c2c9`. Las repeticiones sustituyen la evidencia anterior del mismo mutante y no aumentan el número de mutantes distintos. [mutations.json](mutations.json) conserva la procedencia y las huellas.

coverage.py 7.16.1 registra 3.048 de 3.611 sentencias y 1.020 de 1.484 ramas en 14 fuentes, un 84,41 % y un 68,73 % respectivamente. El diagnóstico de Radon 6.0.1 comprende 261 funciones. La convención de CRAP es `C² × (1 − c)³ + C`, donde C es la complejidad y c la cobertura de sentencias de la región de función. Tras incorporar los 12 casos de compresión, `RegionalWorkLedger._archive_event` pasa de CRAP 48,15 a 25 con la misma complejidad 25 y 24/24 sentencias cubiertas. Estas medidas localizan recorridos pendientes de comprobar y no demuestran ausencia de defectos. [coverage.json](coverage.json) y [quality.json](quality.json) contienen el detalle y sus convenciones.

El ensayo del libro de reservas recorrió 600 eventos con reaperturas e intentos abandonados. Exigió un índice de hasta 16 KiB, como máximo cinco referencias activas, hasta 24 archivos de artefactos tras la poda y no más de 96 lecturas de recibos por evento. Son cotas comprobadas mediante aserciones, no máximos de recursos obtenidos con un perfilador. El archivo completo conserva los 600 eventos y sus recibos aunque se retiren los blobs individuales ya archivados.

## Coste de `FinancialPredictor.prepare`

El perfil definitivo completó 32 registros sobre las fuentes corregidas de `c598ea961`, en una NVIDIA GeForce RTX 4070 Laptop GPU, con PyTorch 2.14.0+cu130, CUDA 13.0 y Python 3.12.14. Utilizó dos hilos CPU, un hilo de interoperabilidad, H = 32, topología `ring`, FP32 y FP64 y lotes de 1 y 256 filas. TF32, autocast y el fastpath de MHA estaban desactivados y los algoritmos deterministas activados.

Cada condición tiene tres calentamientos y diez muestras en orden normal e inverso. Los cuatro controles comparten exactamente los parámetros y las entradas. Cada identidad prepara dos emisiones previas y reutiliza después su propio estado rápido y contexto confirmado. Los valores de esos prefijos pueden diferir entre controles. El intervalo rodea `prepare` completo y su sincronización CUDA, con consulta de contexto, codificadores, fusión, MAC, cabeza escalar y propuesta de estado. La construcción del lote, las emisiones previas y la exportación de comprobaciones quedan fuera.

Medianas en milisegundos, en formato normal / inverso. El selector usa el umbral `1e-9`.

| Precisión | B | `None` | Resumen final | Resumen intercalado | Selección por norma |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1 | 16,925 / 17,288 | 17,591 / 21,937 | 19,326 / 21,456 | 13,778 / 22,290 |
| FP32 | 256 | 10,944 / 13,406 | 12,063 / 17,304 | 17,127 / 14,514 | 15,531 / 13,015 |
| FP64 | 1 | 15,566 / 8,913 | 17,858 / 9,054 | 14,104 / 8,630 | 9,946 / 10,207 |
| FP64 | 256 | 11,638 / 11,853 | 12,882 / 13,702 | 13,126 / 14,915 | 13,061 / 14,048 |

La dispersión entre muestras y órdenes impide elegir una variante por aceleración con esta medida. Los p95 y p99 del JSON interpolan diez muestras y son descriptivos, sin acreditar una distribución de latencia de cola. Se comprobaron resultados completos idénticos en las repeticiones de cada condición y la conservación de parámetros, entradas, estados y RNG. La medida no exige paridad numérica entre arquitecturas distintas.

El máximo asignado por Torch entre las muestras fue de 130.280.448 bytes y el máximo reservado de 165.675.008 bytes. El RSS muestreado alcanzó 1.638.416.384 bytes. Estas cifras incluyen las instancias residentes y el runtime, y las comprobaciones fuera del intervalo también pueden afectar al máximo acumulado de RAM. El proceso hijo empleó 22,673 s, el supervisor 24,202 s y el recibo externo completo 24,321 s. [cost-core.json](cost-core.json) conserva las muestras, ambos órdenes, caudal, memoria, identidades y comprobaciones.

## Coste de la sesión durable

El perfil durable definitivo completó cuatro casos FP64 sobre las fuentes corregidas de `c598ea961`, con cabeza de cuantiles, contexto de mercado y colección desactivada. Cada caso usa una sesión de calentamiento separada y tres sesiones nuevas para medir. Antes de cada medida se procesa el evento 125, se cierra la sesión y se reabre su prefijo confirmado. La creación de fuentes, modelos, directorios y prefijos y la reapertura quedan fuera del intervalo y se registran aparte.

El intervalo comprende `FinancialSession.step` del evento 126 y su sincronización CUDA. Incluye las validaciones de entrada y prefijo, la reserva cuando está activa, la emisión, la confirmación de contexto, el commit nativo, la verificación confirmada, la conciliación y compactación cuando correspondan y la poda. El lote físico es de 64 filas, por lo que B = 256 recorre cuatro bloques.

Tiempos de `step` en milisegundos. Se muestran las tres muestras y su mínimo, mediana y máximo, sin intervalos de confianza ni percentiles de cola.

| B | Programa | Muestra 1 | Muestra 2 | Muestra 3 | Mínimo | Mediana | Máximo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `None` | 389,930 | 375,000 | 431,695 | 375,000 | 389,930 | 431,695 |
| 1 | Intercalado fijo | 592,311 | 736,616 | 543,857 | 543,857 | 592,311 | 736,616 |
| 256 | `None` | 3.005,984 | 2.289,733 | 2.990,001 | 2.289,733 | 2.990,001 | 3.005,984 |
| 256 | Intercalado fijo | 2.411,995 | 3.193,103 | 3.303,031 | 2.411,995 | 3.193,103 | 3.303,031 |

Las medianas de creación de sesión y reapertura del prefijo, calculadas sobre esas tres réplicas y excluidas de `step`, fueron las siguientes.

| B | Programa | Creación (ms) | Reapertura (ms) |
| --- | --- | ---: | ---: |
| 1 | `None` | 39,863 | 156,041 |
| 1 | Intercalado fijo | 71,308 | 155,396 |
| 256 | `None` | 37,482 | 471,481 |
| 256 | Intercalado fijo | 43,439 | 555,496 |

El máximo asignado por Torch entre las muestras fue de 78.796.288 bytes, el reservado de 102.760.448 bytes y el RSS muestreado de 1.602.093.056 bytes. El proceso completo empleó 63,788 s y su ejecución externa supervisada 64,766 s. Las repeticiones conservaron el resultado durable completo y la reapertura conservó el prefijo. [cost-session.json](cost-session.json) contiene las medidas y los tiempos de preparación separados. Los cuatro casos se midieron en orden fijo y sus tres réplicas no establecen una aceleración. Este perfil usa otra fixture y otra cabeza que el del núcleo, por lo que la diferencia entre ambos no aísla el coste de E/S ni del coordinador.

Las instantáneas de ambas mediciones registran un proceso `llama-server` que ocupaba 5.438 MiB de VRAM. La GPU se compartía con ese proceso y con aplicaciones del escritorio. Las rutas locales se normalizan en los JSON públicos y los ejecutables se identifican por su nombre. Las huellas del contenido original permanecen disponibles para contrastarlo con los artefactos locales.

Los perfiles completos anteriores, medidos sobre el predecesor `6e2e91cb3`, se conservan en [cost-predecessor.json](cost-predecessor.json) como evidencia de aquella versión. Ambos protocolos se repitieron después de la corrección de identidades, sin cambiar guiones, formas, calentamientos, repeticiones ni criterios. Las tablas muestran únicamente las ejecuciones definitivas de la versión corregida, con todos sus órdenes y muestras, y no una selección de las ejecuciones más rápidas.

Los guiones medidos se incorporaron en `dc8ab69b5c4c31793dd6007b9e06bc6d5807e33e`. Los [guiones del núcleo](../../../benchmarks/regional_program_cuda.py) y de [sesión durable](../../../benchmarks/regional_program_session_cuda.py) se copiaron fuera del repositorio para medir. Los registros conservan el protocolo fijado antes de cada ejecución y las huellas de sus fuentes. El límite del asignador fue de 512 MiB y se exigieron al menos 256 MiB globales libres antes y después de cada llamada medida. No se midieron energía ni transferencias físicas y esas cotas no equivalen al consumo total de VRAM del proceso.

Las comprobaciones utilizaron fixtures técnicas con valores y etiquetas manuales, sin entrenamientos ni evaluación científica. No aportan una mejora predictiva, una formalización en Lean ni evidencia de novedad. La guarda de aprendizaje conservó el SHA-256 `8a658080c1ac4e63847eba58b177020339da6ec47021cf26d2701830073c87fd` y el test final de 2024 siguió cerrado. En [#709](https://github.com/GonxKZ/mars-titan/issues/709) siguen pendientes la jerarquía anterior a MAC y un selector aprendido de programas. En [#711](https://github.com/GonxKZ/mars-titan/issues/711) sigue pendiente la ampliación de propuestas.
