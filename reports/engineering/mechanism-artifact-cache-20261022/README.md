# Caché acotada del control de mecanismos

El [cambio d3345cf47](https://github.com/GonxKZ/mars-titan/commit/d3345cf47e75c6a0d978d67100c408177c020ffe) evita deserializar repetidamente el mismo control de `FinancialSession`. El almacén conserva una copia privada y entrega copias independientes. Antes de cada acierto relee el archivo y comprueba referencia, tamaño y SHA-256. No reutiliza otros tipos de artefacto ni modifica el formato `torch_weights_only_ordered_maps_v1` o los bytes guardados.

`SessionArtifacts` conserva la caché desactivada por defecto. El enlace de mecanismos declara 8 MiB para un único control. Los otros enlaces mantienen cero. Una entrada demasiado grande no se guarda y la poda libera un control huérfano. El cargo suma el presupuesto consumido por `_payload`, el tamaño serializado y 512 bytes de margen. Esta convención no mide exactamente el heap de Python. Durante una sustitución pueden coexistir el control anterior y el nuevo, además de las copias transitorias de lectura y salida. La caché es estado operativo prescindible, no otro banco episódico ni parte del checkpoint científico.

## Medidas y alcance

La ronda inicial ejecuta cinco casos con dos calentamientos y tres parejas alternadas por caso. Cada fase tiene cinco grupos de observaciones, cuatro decisiones y cierre. Usa fixtures con identidades ficticias, etiquetas manuales, ausencias explícitas, 4, 64 o 256 activos, capacidad episódica 16, dimensión 32, tres regiones, bloque físico 64, FP64 y `cuda:0`. Los identificadores adicionales reutilizan las series del fixture con perturbaciones manuales. No son 256 empresas históricas de FinMultiTime. El lector declara 128 MiB. El selector del cuarto caso usa coeficientes dados, sin ajuste. Se mide el recorrido completo desde el constructor hasta el cierre, incluidas validación, publicaciones y lectura del estado entre pasos. No se cronometra una reapertura posterior al cierre ni la construcción inicial del fixture y del modelo.

| Caso | Sin caché, s por fase | Con caché, s por fase | Cambio de la mediana |
| --- | ---: | ---: | ---: |
| m3-4-fixed-k4cc | 3,412 | 4,390 | +28,69 % |
| m2-64-fixed-k4cc | 11,112 | 7,669 | -30,99 % |
| m3-64-fixed-k4cc | 19,686 | 15,075 | -23,43 % |
| m3-64-manual-k2c | 10,475 | 8,231 | -21,43 % |
| m3-256-fixed-k4cc | 52,878 | 39,947 | -24,46 % |
| M3, 4 activos, repetición adicional | 2,873 | 2,517 | -12,42 % |

El signo negativo indica menos tiempo. El primer caso pequeño empeora un 28,69 % al comparar medianas. Dos de sus tres diferencias emparejadas favorecen la caché y la tercera la perjudica. Se declaró después una [repetición técnica](small-followup-protocol.json) de nueve parejas y un calentamiento, manteniendo el mismo código de producción. Siete parejas favorecen la caché y dos la perjudican. Los [resultados iniciales](benchmark.json) y los [adicionales](small-followup.json) permanecen separados. La dispersión y la GPU compartida impiden asegurar una mejora estable para cuatro activos. En las cargas iniciales de 64 y 256 activos las medianas son favorables. No se deduce una aceleración universal ni un cambio predictivo.

Las predicciones FP64 y la huella del snapshot final coinciden exactamente entre métodos en las 24 parejas medidas. La diferencia máxima de predicciones es 0,0. La igualdad incluye las referencias del banco, pendientes, historial y estado rápido que forman el snapshot. No equivale a revalidar todo el corpus real ni un checkpoint de entrenamiento.

La RTX 4070 Laptop tiene 8.188 MiB y el controlador 595.91.07. Se usó PyTorch 2.14.0+cu130 con CUDA 13.0. Otro modelo local ocupaba unos 5,4 GiB de la GPU y permaneció en ejecución. Cada ronda registra sus consultas de hardware antes y después. No hubo otras pruebas, mutaciones o consultas propias a GitHub durante las medidas. Los tiempos no describen una GPU aislada y no permiten inferir consumo energético.

El pico incremental de memoria asignada por PyTorch es idéntico entre métodos: 685.056 bytes con 4 activos, 10.825.216 con 64 y 12.922.880 con 256. No son toda la VRAM ni su caché reservada. El máximo cargo observado del control es 8.286.385 bytes con 256 activos, bajo el límite declarado de 8.388.608. El pico global de RAM del proceso inicial es 1.602.514.944 bytes y el de la repetición 1.517.527.040 bytes. No hay una medida separada de RAM por método que permita atribuir ahorro de memoria. El cronómetro global se inicia después de importar los módulos y cargar el backend, por lo que no mide todo el arranque del proceso. Registra 874,185 s en la ronda inicial y 59,705 s en la adicional. Tres o nueve repeticiones de fases no justifican publicar p95 o p99 de predicciones individuales.

## Pruebas y revisión

La [verificación](verification.json) reúne 185 casos distintos, 37 CUDA, sin fallos, errores ni omisiones. La regresión amplia pasa 181 casos. La tanda posterior de 28 incluye cuatro integraciones nuevas y sustituye la prueba de contadores reforzada. Los conjuntos se deduplican por nombre e identificador de pytest. Se comprueban copias de listas y tensores, conservación de binary64 y tipos, referencias incompatibles, sustitución del archivo con tamaño y fecha restaurados, cuotas, desactivación, sustitución de la entrada, poda y activación solo para el consumidor previsto. Las sesiones originales de M2/M3, sus emisiones maduras, recuperación y colecciones nativas también pasan la regresión.

La primera [mutación dirigida](initial-mutations.json) detecta 11 de 13 cambios incorrectos. Dos se compensaban en un contador agregado de deserializaciones. Se añadieron expectativas en cada paso. La [ronda final](mutations.json) detecta 13 de 13, con 26 casos CPU válidos antes y después de restaurar las fuentes. Esto no es una puntuación de mutación de todo el proyecto. Las mutaciones trabajan sobre una copia, sin cambiar el árbol utilizado para las medidas.

La [calidad](quality.json) utiliza coverage.py 7.16.1 y Radon 6.0.1 con el cálculo canónico de `scripts/summarize_python_quality.py`. Se registran ramas. CRAP usa cobertura de sentencias y complejidad de la región de función. Esta medida corresponde a los 28 casos focalizados, no a toda la regresión.

| Función | Sentencias | Ramas | CRAP |
| --- | ---: | ---: | ---: |
| `SessionArtifacts.__init__` | 7/7 | 2/2 | 9,000 |
| `SessionArtifacts.read` | 20/24 | 10/12 | 13,782 |
| `SessionArtifacts.prune_unreferenced` | 18/18 | 14/14 | 10,000 |
| `FinancialSession.__init__` | 63/78 | 17/30 | 34,981 |

Ruff pasa en el repositorio y su comprobación de formato registra 1.820 archivos. El comprobador local verifica 3.555 archivos, 361 referencias y 64 tareas. Las 140 entradas del catálogo macro tienen dependencias válidas, lo que no acredita disponibilidad completa de sus series históricas. La puntuación no demuestra ausencia de defectos. Esta ronda no cambia código nativo, bibliotecas, kernels, precisión ni políticas de memoria financiera. No atribuye mejoras predictivas, originalidad científica o superioridad frente a otros modelos.

## Reproducción y archivos

El [programa medido](benchmark.py) conserva el generador de los cinco casos y rechaza reutilizar un directorio de salida. Requiere el entorno del proyecto con CUDA y el módulo nativo indicado por `MARS_TITAN_EPISODIC_NATIVE`. El comando ejecutado desde la raíz fue:

```bash
uv run --no-sync python reports/engineering/mechanism-artifact-cache-20261022/benchmark.py --output /home/gonzalo/.local/state/mars-titan/artifact-throughput-20261022/benchmark
```

Una nueva ejecución necesita otra ruta de salida. El registro posterior `execution-environment.json`, incluido en la relación de artefactos, conserva las variables declaradas y las versiones del entorno de comandos utilizado. No es un muestreo retrospectivo del proceso terminado. La [relación de artefactos](external-artifacts.json) incluye hashes de JUnit, cobertura, logs, las dos rondas de mutación, el programa y protocolo adicionales y los índices de fixtures y 48 ejecuciones nativas medidas. Cada ejecución conserva identidad, cabeza, checkpoints de las dos últimas generaciones y los seis registros. Los datos y estados voluminosos permanecen fuera de Git.

Las medidas se hicieron sobre el padre `72f80ccbb` con los cambios de producción aún sin commit. Sus fuentes y las de los generadores coinciden por hash con `d3345cf47`. El programa medido tiene su propia huella, comprobada contra el archivo publicado. Los pesos compartidos del fixture no se exportaron como checkpoint de entrenamiento. Se conservan su generador versionado, las identidades y los estados operativos. No se han ejecutado optimizadores, ajuste del selector, corpus real ni test final en esta ronda.
