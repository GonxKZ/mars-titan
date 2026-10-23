# Préstamo de almacenamiento de BANK-WRITE, 21 de octubre de 2026

El candidato se conserva para reproducir el ensayo y no se adopta en la implementación activa. Ahorra memoria transitoria, pero el caso pequeño repetido presenta una mediana de latencia mayor. La fuente productiva de `NeuralMemory` vuelve a coincidir exactamente con `0f05b62e23302551a63b3bfa857e06661b0494d9`. BANK-WRITE y sus reglas siguen implementados. Este resultado solo descarta una optimización de almacenamiento, no la arquitectura.

El objetivo era evitar copias locales de los seis pesos de tres redes durante una actualización asociativa no diferenciable. El candidato usa `detach()` sobre el almacenamiento anterior, que solo lee. Publica pesos y momentum nuevos. Conserva la copia cuando se piden derivadas externas o cuando el propietario es un tensor de `inference_mode`. No cambia gradientes, puertas, olvido, estado persistente, identidades de configuración o parámetros. No introduce kernels ni dependencias.

La [referencia](copied_reference.py), el [candidato](borrowed_candidate.py) y el [parche exacto sobre la fuente de referencia](candidate.patch) permiten inspeccionar la diferencia. La referencia procede del commit indicado. El cuerpo del candidato coincide por AST con la función medida. Las fuentes completas y las versiones exactas de los bancos de medidas están identificadas en [external-artifacts.json](external-artifacts.json).

## Medidas

RTX 4070 Laptop de 8.188 MiB, controlador 595.91.07, PyTorch 2.14.0+cu130, CUDA 13.0, Python 3.12.14 y dos hilos CPU. Se utilizó `cuda:0`, determinismo activado, sin TF32 ni autocast. No hubo otras cargas propias en paralelo ni se detuvieron aplicaciones ajenas. La ocupación externa observada antes de medir se conserva en los registros. Los picos siguientes son del asignador de PyTorch, no toda la VRAM del proceso.

La memoria usa dimensión 64 y dos capas. Los componentes tienen tres calentamientos y veinte repeticiones, alternando las versiones en cada repetición. La preparación financiera usa bloques de 64, un prefijo de tres observaciones y cuatro configuraciones, con y sin el mismo grafo regional. Mantiene `max_batch=128`. La memoria aislada y la sesión nativa declaran 256. No se ha relajado ningún guardia de presupuesto.

| Recorrido con 256 flujos | Copias, p50 ms | Préstamo, p50 ms | Reducción del pico transitorio |
| --- | ---: | ---: | ---: |
| Actualización, media, FP32 | 5,014 | 4,972 | 24 MiB |
| Actualización, parciales, FP32 | 5,597 | 5,589 | 24 MiB |
| Actualización, media, FP64 | 7,959 | 8,296 | 48 MiB |
| Actualización, parciales, FP64 | 6,129 | 6,052 | 48 MiB |
| Preparación financiera, media | 45,852 | 46,714 | 6 MiB |
| Preparación financiera, parciales | 42,273 | 41,350 | 6 MiB |
| Preparación con grafo, media | 51,661 | 54,253 | 6 MiB |
| Preparación con grafo, parciales | 63,511 | 62,472 | 6 MiB |

No hay una aceleración general de los componentes. Se conservan las 24 comparaciones, incluidas las siete condiciones con mayor p50 del candidato, en [cost.json](cost.json). El ahorro de matrices no reduce el estado persistente. El perfil de tres actualizaciones FP32 de 256 flujos registra 21 llamadas a `aten::clone` con copias y tres con préstamo, y dieciocho lanzamientos menos. Conserva 45 `bmm` y 105 sincronizaciones. Son observaciones del profiler del framework, sin atribuirles una explicación única del tiempo total ni un ahorro energético.

Las sesiones completas utilizan procesos separados con árboles de fuentes identificados, un modelo residente por proceso, un calentamiento y cuatro repeticiones. El orden de versiones se invierte entre condiciones, pero las repeticiones de una versión quedan agrupadas. Incluyen calentamiento de memoria, tres decisiones, feedback manual posterior a las emisiones, cierre, publicación, poda y recuperación de la fase cerrada. No miden una simulación de rentabilidad ni entrenamiento.

| Sesión completa | Copias, p50 ms | Préstamo, p50 ms | Cambio descriptivo |
| --- | ---: | ---: | ---: |
| Cuatro flujos, media | 1.784,408 | 1.765,013 | −1,09 % |
| Cuatro flujos, parciales | 1.690,335 | 2.099,083 | +24,18 % |
| 256 flujos, media | 16.838,258 | 16.501,392 | −2,00 % |
| 256 flujos, parciales | 17.275,985 | 17.226,283 | −0,29 % |

Los dos casos de 256 flujos reducen el pico en 6 MiB. El ahorro con cuatro flujos es de 393.216 bytes. Las predicciones puntuales emitidas, los parámetros iniciales, el contrato de entrada, la configuración y los bytes retenidos coinciden entre versiones. La identidad del consumidor registra fuentes distintas y no se fuerza una recuperación cruzada entre ellas. La preparación financiera compara también los cuantiles y el siguiente estado completos. Los percentiles se conservan en [native-cost.json](native-cost.json), que identifica el artefacto externo con fases, configuraciones y muestras completas. Cuatro repeticiones no precisan bien las colas.

El resultado pequeño motivó cuatro pares adicionales con orden invertido por bloques, registrados antes de ejecutarlos. Sus cambios de p50 son −6,83 %, +4,30 %, +24,09 % y +7,74 %. Al incluir la tanda original hay veinte observaciones temporizadas por versión, con medianas de 1.788,131 ms y 1.913,721 ms, un aumento del 7,02 %. Son bloques del mismo fixture, no observaciones financieras independientes. La dispersión aparece en varias fases y no identifica una causa única. [small-confirm.json](small-confirm.json) resume todos los bloques e identifica el artefacto externo con sus muestras.

[decision.json](decision.json) registra la decisión de no adoptar. El margen inicial para 256 flujos se cumple y el ahorro mínimo de memoria también. La preocupación posterior por el caso pequeño se investigó sin cambiar ese margen ni eliminar la primera tanda. No se atribuye beneficio integral a partir del contador de copias o del resultado de 256 flujos.

## Comprobaciones y ejecuciones incompletas

La selección final pasa 440 casos, 84 marcados CUDA, sin fallos ni omisiones y con un aviso conocido del detector experimental de sincronizaciones de PyTorch. Comprueba el predictor activo y dieciséis casos del candidato aislado. No es toda la suite del proyecto. Una tanda anterior de 440 casos comprobó el candidato integrado antes de descartarlo. Las tandas no se suman como casos distintos.

Los casos nuevos comprueban propiedad del almacenamiento, flags del propietario, ausencia de gradientes acumulados, publicación sin alias, referencia de parciales, fallback para tensores de inferencia y curvatura diferenciable. Las pruebas previas añaden recuperación, ecuaciones, partición, madurez y fallos numéricos. La [verificación](verification.json) identifica las fuentes y las salidas JUnit.

La mutación dirigida del candidato aislado detecta cinco cambios, con 41 casos CPU antes y después de restaurar: copia redundante, pérdida de `detach`, préstamo de un tensor de inferencia, pérdida de curvatura y modificación del propietario. No hay errores de colección ni omisiones. [mutations.json](mutations.json) conserva el resultado. La tanda anterior de cinco mutaciones del candidato integrado permanece separada en [candidate-mutations.json](candidate-mutations.json). No es una medida exhaustiva de mutantes posibles.

La cobertura final utiliza coverage.py 7.16.1 y Radon 6.0.1. Mide 388 de 414 sentencias y 143 de 168 ramas en dos fuentes seleccionadas. La función activa del problema asociativo tiene 12/12 sentencias, 6/6 ramas y CRAP 13. El candidato aislado tiene 6/12 sentencias, 2/6 ramas y CRAP 38,5 porque conserva ramas ajenas al contraste que no recorren sus dieciséis casos directos. La tanda integrada anterior cubrió 12/12, 6/6 y CRAP 14 en esa función. Los dos alcances están separados en [quality.json](quality.json) y [candidate-quality.json](candidate-quality.json). CRAP usa cobertura de sentencias, con `C²(1−cobertura)³+C`, y no acredita ausencia de errores.

El primer banco de medidas completó los componentes y se detuvo al sustituir un método en un consumidor protegido. La protección se mantuvo y las sesiones se trasladaron a procesos con fuentes reales. El primer par de esos procesos terminó sus sesiones, pero se rechazó al detectar contratos de entrada distintos porque las rutas de los fixtures formaban parte de su identidad. Se comprobó la reproducción de esa identidad y se reconstruyó el fixture secuencialmente en una misma ruta temporal. No se relajó la igualdad exigida. Los intentos incompletos, fuentes y protocolos permanecen identificados, junto con una primera invocación de pruebas que no ejecutó casos por una ruta inexistente.

Los componentes completos emplearon 32,30 s de proceso externo, las sesiones completas 437,24 s y la confirmación pequeña 122,36 s. El pico RSS externo fue de 1.744.805.888, 1.706.508.288 y 1.653.493.760 bytes respectivamente. Los tiempos internos y procesos hijos se separan en [process-cost.json](process-cost.json). No se midió energía ni coste monetario. Los registros de [protocol.json](protocol.json) preceden a cada ejecución y conservan sus huellas.

## Reproducción y alcance

La prueba aislada se ejecuta con `uv run --no-sync pytest -q tests/models/titans/test_bank_write_storage_candidate.py`, en el entorno CUDA declarado del proyecto. No usa optimizador externo. Los scripts [benchmark.py](benchmark.py) y [native_benchmark.py](native_benchmark.py) exponen `--help` y exigen salidas nuevas en almacenamiento externo. El primero selecciona explícitamente las dos funciones conservadas y comprueba igualdad exacta antes de temporizar. El segundo requiere `--reference-source` y `--candidate-source` apuntando a los directorios `src` de dos árboles, uno de la referencia y otro con `candidate.patch` aplicado. Sus trabajadores comprueban la huella de la fuente importada y no sustituyen métodos del consumidor.

Las fuentes exactas utilizadas para medir se conservan por separado de esos scripts de reproducción. Los cambios posteriores explicitan la selección del candidato descartado y la ubicación de sus fuentes. Las huellas originales permiten distinguir ambas versiones. Los artefactos voluminosos y los snapshots técnicos quedan fuera de Git. El índice externo contiene evidencia seleccionada, no todos los archivos de los checkpoints.

Tras comprobar la recuperación se conserva la última sesión de cada uno de los dieciocho procesos completos. Se han retirado 72 sesiones de fixtures redundantes, con 914.765.888 bytes liberados. [artifact-retention.json](artifact-retention.json) registra esa limpieza de artefactos propios. No altera los bytes retenidos durante las mediciones ni los checkpoints históricos.

Todos los datos son fixtures identificados. No hay corpus real, aprendizaje del selector, ajuste de parámetros compartidos o mejora predictiva medida. Las actualizaciones de pesos rápidos forman parte de la inferencia de Titans. La campaña y el test final siguen cerrados. La [revisión de antecedentes](source_audit.md) distingue memoria distribuida, concentración de atención y edición retrospectiva, sin atribuirles superioridad financiera.
