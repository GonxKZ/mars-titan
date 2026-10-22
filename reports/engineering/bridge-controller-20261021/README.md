# Controlador con lecturas regionales sobre evidencia compartida

`FrozenMechanismConsumer` acepta el puente con `MechanismCatalog(reader_contract="shared_evidence_bridge_v1")`. El catálogo anterior sigue siendo el valor por defecto y rechaza el lector regional. El catálogo nuevo y el consumidor tienen identidades propias. El selector debe declarar un `catalog_id` compatible. Ese vínculo no acredita que sus coeficientes se hayan entrenado.

Las seis rutas son `k1`, `k2c`, `k2r`, `k4cc`, `k4cr` y `k4rc`. La primera lectura se comparte con la ruta solicitada. c conserva posiciones y r renueva la selección común al empezar su bloque. Las proyecciones se recalculan en cada lectura y se contabilizan. c no designa el factor C de CM-v1. No se reinterpretan los pesos regionales como `EpisodeRead.weights[B,k]`.

El selector decide antes de MAC con las observaciones admitidas de la cohorte. Las vueltas trabajan sobre el estado de una predicción, con una sola actualización MAC por observación. Los recibos guardan IDs comunes, pesos y vistas regionales derivadas, lectura y procedencia del snapshot. No crean otro banco por región ni archivan valores fuente o etiquetas. La publicación, el feedback y la recuperación siguen usando el ejecutor y el archivo de mecanismos existentes. La [especificación](../../../docs/experiments/shared-evidence-bridge-v1.md#selección-mediante-el-controlador) describe los límites del contrato.

## Pruebas y calidad

La [verificación](verification.json) distingue cada tanda. La regresión final, posterior al rechazo estricto del contador, pasó 453 casos, 111 marcados CUDA, sin fallos, errores ni omisiones, en 816,45 segundos. Incluye catálogo, selector, consumidor, rutas anteriores, lector opcional, trazas y sesiones nativas. En las seis rutas CUDA FP32 frente a CPU FP64 se conservaron los IDs y se cumplieron `rtol=2e-5` y `atol=1e-6` para estados y pesos regionales. Las tolerancias estaban escritas antes de ejecutarlos. No se extrapolan a otras formas o datos.

Las tandas anteriores de 432, 28 y siete casos se conservan por separado. Se solapan con la selección final y no se suman como casos distintos. Cubren también el presupuesto sobre toda la cohorte, el vínculo con el catálogo y dos límites de planificación. Las sesiones nuevas comprueban selección fija y automática, CPU y CUDA, y recuperación tras un corte posterior a confirmar feedback. La siguiente ruta, predicción y estado rápido coinciden y la maduración no se aplica dos veces. El selector utiliza coeficientes suministrados en fixtures, sin ajuste externo.

La [mutación dirigida final](mutations.json) detectó los doce cambios seleccionados. La referencia y la restauración aprobaron 38 casos CPU cada una. Se alteraron identidad del catálogo, compatibilidad del lector y selector, selección fija y renovada, gradientes, proyecciones, snapshot, pesos y presupuesto lógico. No es una puntuación sobre todos los mutantes posibles.

Una revisión posterior detectó que el contador agregado aceptaba `False` o `0.0` al valer cero. Dos pruebas nativas reprodujeron el fallo. El validador ahora exige un entero. La [mutación específica del contador](counter-mutation.json) vuelve a provocar ambos fallos al retirar esa comprobación y la restauración aprueba los dos casos. Las medidas de coste preceden a este rechazo adicional. Se conservan sus fuentes exactas y no se presentan como una nueva medida de latencia de la revisión posterior.

La [cobertura y CRAP](quality.json) usan coverage.py 7.16.1, Radon 6.0.1 y el resumidor común. La selección final ejecutó 861/936 sentencias y 358/434 ramas en seis módulos. El verificador regional tiene CRAP 47,04 y `FrozenMechanismConsumer.begin` 58,88. Quedan rutas y combinaciones de condiciones sin recorrer. Los fallos iniciales se conservan por separado. Estas medidas no demuestran ausencia de errores.

La primera prueba de planificación usó un generador de claves de hasta 64 filas para pedir 256. Falló por formas incompatibles. Se conservó ese fallo, se creó un fixture válido de 256 claves y los siete casos finales pasaron. No se debilitó el validador. Un intento de mutación que partió de aquel fixture incorrecto también queda registrado y no se cuenta como una pasada válida.

## Entradas y protocolo de coste

Se midieron rutas fijas con H = 32, tres regiones, dos ciclos, FP64 en `cuda:0`, TF32 y autocast desactivados, dos hilos CPU, bloques físicos de 64 y M1 con capacidad 256. Cada condición tiene un calentamiento y tres repeticiones. Se mide la decisión nativa con consulta, núcleo, lector, cabeza, archivo, verificación y confirmación. La maduración posterior se cronometra aparte. La preparación de fuentes y el historial previo quedan dentro del proceso total, fuera de esas dos llamadas.

`flow_source` replica el historial de precios de un fixture por cada identificador. La máscara, en orden precios, noticias, gráficos, fundamentales y macro, es `[true, false, true, false, false]`. El canal gráfico es un valor constante `3.0`, de dimensión uno. Noticias, fundamentales y macro están ausentes. Las etiquetas son manuales. No son 256 historias bursátiles independientes ni una medida del codificador de gráficos reales o del corpus multimodal completo.

El [protocolo original](cost-protocol.json) describía de forma inexacta las otras modalidades como ausentes. Se conserva y se añade una [corrección con la fuente inspeccionada](data-description-correction.json). La corrección también se aplica a la descripción del informe anterior. No se modificaron las entradas usadas ni los números medidos.

## Parada detectada y pareja adicional

La primera pasada terminó con salida 1 tras completar once de doce condiciones. En la última, 256 consultas regionales con `k4cr`, el controlador aplicó K1 con motivo `reader_budget`. La aserción exigía K4 realmente ejecutado y detuvo la medida. El JSON original conserva `running_not_complete`, acompañado por el [registro de terminación](first-termination.json).

El límite de planificación del lector era 256, mientras la sesión ejecutaba bloques de 64. El estimador daba 52.633.600 bytes para K4 con 256 filas y 13.600.768 con 64, frente a 33.554.432 bytes permitidos. La traza de toda la cohorte estimaba 8.945.664 bytes. Son estimaciones de trabajo, sin demostrar que se hubiera agotado RAM o VRAM. La regla del catálogo devolvió el K1 declarado. La prueba añadida comprueba ambos límites sin cambiar la protección.

Una [pasada adicional registrada antes de ejecutarse](continuation-protocol.json) comparó ambos lectores con 256 consultas, `k4cr` y planificación explícita de 64. Conservó el límite de 32 MiB y la aserción de cuatro lecturas por fila. Terminó las dos condiciones con salida 0. No se repitió la campaña original ni se sustituyó su fallo por esta pasada.

Las parejas de cada pasada comparten banco, entradas y parámetros del padre. Entre pasadas cambian las huellas de entradas y banco, aunque coinciden los parámetros del padre. También cambia el límite de planificación. No se mezclan como una única comparación de K con idéntica configuración. El control anterior de 256/K4 de la primera pasada queda conservado, sin utilizarlo como pareja de la ejecución regional posterior.

## Coste observado

Estas son las medianas de emisión y confirmación nativas, en segundos:

| Consultas | Ruta | Lector anterior | Regional | Planificación y pasada |
| --- | --- | ---: | ---: | --- |
| 4 | k1 | 0,606 | 0,551 | 256, primera incompleta |
| 4 | k4cr | 0,323 | 0,388 | 256, primera incompleta |
| 64 | k1 | 0,783 | 1,037 | 256, primera incompleta |
| 64 | k4cr | 1,730 | 4,425 | 256, primera incompleta |
| 256 | k1 | 4,795 | 8,452 | 256, primera incompleta |
| 256 | k4cr | 5,640 | 14,230 | 64, pareja adicional |

El [resumen legible por máquina](cost.json) conserva dispersión, p50/p95/p99, caudal, costes de maduración, bytes del control y pico asignado por Torch. Con tres muestras los percentiles son descriptivos. El orden de las condiciones no se aleatorizó y había otras aplicaciones usando el equipo. La diferencia favorable en la primera condición no acredita una aceleración general. La variante añade coste en varias cargas.

En la pareja adicional de 256/K4, la mediana de maduración fue 2,030 s para el lector anterior y 3,716 s para el regional. El control archivado pasó de 194.669 a 1.807.149 bytes. El pico asignado por Torch fue 78.488.576 y 78.587.392 bytes. Esa cifra no es la VRAM total. Las trazas del lector anterior tienen menos contenido. Falta perfilar el recorrido y usar un control de instrumentación comparable para atribuir el sobrecoste a cálculo, validación o almacenamiento.

Los [costes de proceso](process-cost.json) registran 470,32 segundos externos y salida 1 en la primera pasada, y 278,93 segundos y salida 0 en la pareja adicional. Esta última registró 273,01 segundos internos. Los picos RSS externos fueron 1.661.412 y 1.657.944 KiB.

Se usaron una RTX 4070 Laptop de 8.188 MiB, driver 595.91.07, Torch 2.14.0+cu130 y CUDA 13.0. Antes de las pasadas había 5.947 y 5.966 MiB ocupados y una utilización del 0 % y 13 %. No se detuvieron otras aplicaciones. No hubo otras pruebas, mutaciones ni consultas propias durante los cronometrajes. No se midió energía ni contadores de caché, lanzamientos o transferencias físicas.

## Reproducción y trabajo pendiente

Con el entorno uv CUDA y la extensión nativa configurados, el comando que ejecutó la pareja adicional es:

```bash
evidence_dir="$(mktemp -d)"
mkdir "$evidence_dir/fixtures"
uv run --no-sync python reports/engineering/bridge-controller-20261021/benchmark.py --output "$evidence_dir/cost.json" --storage-root "$evidence_dir/fixtures" --rows 256 --routes k4cr --reader-max-batch 64
```

La salida debe ser nueva. El valor por defecto de planificación es 256 para reproducir el contrato original, cuyo conjunto completo detectó el retorno a K1. Los filtros y el límite explícito distinguen la pareja adicional. Cada ejecución conserva el HEAD padre y las huellas del código realmente utilizado antes del commit. El [índice externo](external-artifacts.json) identifica registros, fuentes, pruebas y archivos nativos sin versionar sus artefactos pesados.

El enlace de mecanismos mantiene su admisión M0/M1. Las sesiones regionales nuevas y estos costes usan M1. La integración con M2/M3, la declaración arquitectónica `MarsTitanVariant`, el colector y el ajuste de alternativas regionales necesitan comprobaciones propias. Las regiones siguen siendo vistas genéricas, sin sectores históricos o especialización aprendida acreditados.

No se ejecutaron entrenamientos, postentrenamientos, RL, evaluaciones del corpus ni acceso al test final. No hay mejora predictiva, novedad universal o superioridad frente al estado del arte medida. La variante continúa opcional. El siguiente trabajo de rendimiento debe localizar el coste integrado antes de sustituir componentes o añadir cachés y código nativo.
