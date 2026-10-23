# Composición de jerarquías antes y después de MAC

La composición permite usar la ejecución regional anterior a MAC junto con el
puente episódico jerárquico posterior mediante `FrozenMechanismConsumer`. Cuando
ambas etapas están activas, la identidad usa el esquema 6 y la receta
`frozen_pre_mac_hierarchical_bridge_cohort_v1`. El catálogo conserva las 36 acciones
y registra el eje regional anterior a MAC. Los sellos incluyen las fuentes de
ambas etapas. Sus estados temporales y presupuestos se contabilizan por separado.

## Comprobaciones técnicas

Las selecciones finales reúnen 243 IDs únicos, 69 parametrizados en CUDA. Los XML registran seis casos de identidad, dos de
catálogo, cuatro de recuperación nativa CPU/CUDA y cuatro de presupuesto de
colección y emisión. Pasaron también 215 regresiones y once casos de CLI, sin
fallos, errores u omisiones. Los casos CLI impiden importar Torch o NumPy y
comprueban el protocolo, las rutas obligatorias, las salidas inválidas, el rechazo
del archivo canónico y la igualdad de la copia privada con su fuente pública.
La raíz del repositorio en esas pruebas se deriva de la ubicación del archivo.

El caso existente
`test_factory_preserves_the_typed_hierarchy_and_omits_null` pasó en una ejecución
adicional de 31,15 segundos supervisados. Comprueba la reconstrucción desde JSON,
la configuración tipada, el aislamiento frente a cambios en el diccionario de
entrada y la omisión de la jerarquía cuando es None.

La mutación dirigida detectó cuatro de cuatro sustituciones: omitir la receta
compuesta, eliminar el eje regional del catálogo, retirar el sello del puente y
subestimar el presupuesto conjunto de colección y emisión. Cada mutante produjo
un fallo esperado en CALL, con su control inicial y restaurado aprobados, sin
errores de colección ni omisiones. Son cuatro ejecuciones iniciales y cuatro
restauradas, no ocho pruebas nuevas. Las fuentes originales y la guarda de
aprendizaje conservaron sus huellas.

[attempts.json](attempts.json) identifica los intentos previos sin sumarlos como
ejecuciones aprobadas. El fixture se corrigió para admitir cuatro flujos lógicos
con capacidad ocho, manteniendo lector y bloques físicos de dos. Las pruebas de
identidad usan su campo `configuration` y comparan las representaciones JSON
canónicas. La primera captura de cobertura no encontró su dependencia, que se
cargó mediante `uv --with` con las versiones fijadas. La exportación definitiva
usa una copia idéntica de la base para evitar confundir el sidecar JSON con otra
base. El primer validador de mutación buscaba el nombre de la excepción en el
texto abreviado. La versión final obtiene su tipo real mediante `call.excinfo`
y conserva las demás condiciones. También se rechazó antes de ejecutar un
mutante un recibo procedente de una tanda con fallos. Se sustituyó por una
ejecución limpia de los cuatro casos de presupuesto.

La cobertura focal combina 156 CALL de contratos y puente, una recuperación
nativa CPU y el caso de fábrica. Mide importación y CALL de cuatro archivos, sin
instrumentar preparación o limpieza de fixtures, bibliotecas ni el resto de la
sesión nativa. Las 215 regresiones funcionales no se convierten por ello en 215
casos de captura de cobertura.

| Archivo | Sentencias cubiertas | Ramas cubiertas |
| --- | ---: | ---: |
| `bridge_hierarchy.py` | 92/92 | 38/38 |
| `frozen_financial.py` | 153/190 | 65/98 |
| `mechanism_catalog.py` | 109/136 | 32/58 |
| `mechanism_consumer.py` | 263/338 | 96/168 |
| Total | 617/756 | 231/362 |

Coverage 7.16.1 y Radon 6.0.1 conservan las 46 funciones, sin excluir líneas.
`scripts/summarize_python_quality.py` calcula CRAP con cobertura de sentencias y
publica las ramas aparte. El máximo es 116,81877076969555 en
`MechanismCatalog.resolve`, con complejidad 28 y 16/31 sentencias cubiertas.
`FrozenMechanismConsumer.close_phase`, `ResolvedMechanisms.status` y
`FrozenFinancialConsumer.identity` siguen sin sentencias cubiertas. La rama que
instala el esquema 6 y su alternativa sí están cubiertas. Los archivos
`coverage-joint-final.json` y `quality-joint-final.json` conservan el resultado
completo. Los tres recibos de alcance coinciden entre sí y con las cuatro fuentes
medidas. La cobertura y CRAP son diagnósticos y no demuestran ausencia de errores.

## Protocolo de coste y límites

`benchmarks/composed_hierarchy_cuda.py` fija cuatro controles: puente plano sin
etapa anterior, solo jerarquía anterior, solo puente jerárquico y ambas etapas.
Usa FP64, H=32, cohortes de 4, 64 y 256 filas y bloques físicos comunes de 64.
El predictor admite 256 flujos lógicos. Los dos órdenes, un calentamiento y tres
muestras por control producen 24 registros, 96 invocaciones de cohorte
y 192 preparaciones físicas, todos completados. Se usa el estado inicial propio, sin prefijo, y un
fixture común de ocho episodios maduros, sin PT4. Se mide `begin`, todas las preparaciones y `finish`,
con sincronización CUDA. La construcción se registra fuera de la latencia por
petición. La auditoría adicional del benchmark sobre parámetros, estados,
resultados completos y RNG se realiza fuera del intervalo medido. Las
verificaciones fuertes del consumidor permanecen dentro de `begin` y `finish`.
Los 49 tensores de parámetros comunes coinciden exactamente. El control plano y
el puente posterior tienen 48.902 parámetros. Activar la etapa anterior añade
10.368, hasta 59.270, porque incorpora también el operador de fusión regional.
Este contraste no aísla el calendario jerárquico frente a otra fusión regional
ni iguala toda la capacidad o el cómputo. No es una comparación predictiva.

El guion público y su copia privada coinciden en SHA
`dad9eaee2579591a5e6a44a8f9e6419da85c3b4e795b915d67be485fddf14e2e`.
El protocolo V3 registra 480 segundos internos y 510 externos, con 512 MiB de
asignador y 256 MiB de margen libre. La ejecución V3 terminó en 269,49 segundos
internos y 275,31 supervisados, con cero intentos de optimizador y fuentes,
script, helper, parámetros compartidos, inputs, estados iniciales y RNG intactos.
Los pesos rápidos propuestos avanzan una observación por cohorte, mediante la
adaptación asociativa interna de Titans. Los intentos anteriores
se conservan como incompletos: V1 agotó
100 segundos con 14 registros completos y V2 agotó 240 con 20 completos y otro
parcial. No cuentan como ejecuciones aprobadas ni se seleccionan sus tiempos
como resultado final.

Tres muestras por orden no permiten estimar colas con precisión fiable. Los
percentiles publicados son exploratorios. La sonda no mide persistencia durable,
colección, feedback, codificación del corpus, energía ni transferencias físicas.
No hay pasos de optimizador ni evaluación predictiva del corpus real en estas
comprobaciones y no se atribuye una mejora financiera a la composición.


Para 256 filas, las medianas observadas fueron:

| Control | Orden inicial (ms) | Orden inverso (ms) |
| --- | ---: | ---: |
| Plano J4 | 323,402 | 552,167 |
| Solo etapa anterior | 506,764 | 871,132 |
| Solo puente jerárquico L2 | 521,706 | 744,778 |
| Ambas etapas | 856,738 | 807,809 |

El recorrido combinado añade tiempo respecto al plano en ambos órdenes. La
variación entre pasadas no permite clasificar diferencias pequeñas como mejoras
estables. La medición incluye comprobaciones fuertes al abrir y cerrar la
cohorte, las preparaciones por bloques y las trazas exportadas por el consumidor.
No es latencia de kernel ni un límite de rendimiento de la RTX. Durante la ronda
había otra aplicación residente en GPU y tareas de metadatos y Git en CPU. No
se detuvieron para presentar una ejecución aislada.

El máximo observado por Torch fue de 90.123.264 bytes asignados y 111.149.056
reservados. El RSS muestreado alcanzó 1.633.800.192 bytes y el máximo comunicado
por `getrusage` fue 1.632.071.680. Son contadores distintos y no una medida
exacta de la memoria exclusiva de cada variante. El estado inicial conserva
8.392.704 bytes lógicos de tensores en los cuatro controles de 256 filas. Los
auxiliares regionales siguen siendo temporales. No se acredita un ahorro global
de memoria.

## Reproducción y trazabilidad

El código medido corresponde a
`478b0892ae36f4cd4c02b9c132729a3c8a905dfd`. El manifiesto de 1.346 archivos de
fuentes, fixtures y configuración se contrasta con sus bytes actuales y con los
blobs de ese commit antes de generar estos resúmenes. El número definitivo y la
huella están en [verification.json](verification.json). Los comandos reales
conservan las rutas mediante `$REPOSITORY`, `$RUN_ROOT` y
`$UV_PROJECT_ENVIRONMENT`. Los recibos originales y las bases permanecen fuera de
Git, identificados por nombre, tamaño y SHA. [quality.json](quality.json)
conserva las 46 funciones. [mutations.json](mutations.json) conserva los cuatro
objetivos y las doce fases. [cost.json](cost.json) incluye las 24 condiciones,
las muestras, percentiles exploratorios, picos y huella del resultado completo.

Con el entorno CUDA del proyecto preparado, la ejecución comprobada se puede
repetir desde una copia externa del mismo guion:

```bash
mt_repository=$PWD
mt_run_dir=$(mktemp -d)
cp benchmarks/composed_hierarchy_cuda.py "$mt_run_dir/probe.py"
timeout --kill-after=5s 510s uv run --no-sync python "$mt_run_dir/probe.py" \
  --run --repository "$mt_repository" --output "$mt_run_dir/cost.json"
```

La carga exige CUDA0 en una RTX 4070. Sin `--run`, el guion imprime el protocolo
sin importar Torch. El acceso al test final y el bloqueo de aprendizaje siguen
intactos. La evidencia financiera permanece pendiente y no se promociona la
composición a configuración predeterminada.
