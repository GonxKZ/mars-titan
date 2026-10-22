# Sesión durable de mecanismos y coste integrado

El enlace de `5b6f8d72b` incorpora el consumidor de mecanismos a `FinancialSession`, con M0/M1. Continúa [#701](https://github.com/GonxKZ/mars-titan/issues/701). Contexto, calibración, emisiones y resultados admitidos se confirman en la misma generación. El selector aprendido y sus contrastes siguen pendientes. Las variantes anteriores conservan su canalización y no cambia el modelo por defecto.

Las comprobaciones usan fixtures con precios replicados, modalidades pequeñas, máscaras y etiquetas manuales. No hay corpus real, pasos de optimizador, entrenamiento, estudio predictivo nuevo ni acceso al test final. No se calcula una supuesta mejora financiera sobre esos valores manuales.

## Qué se integra

[`MechanismBinding`](../../../src/mars_titan/memory/financial_mechanisms.py) reutiliza el enlace Titans, el ejecutor nativo y `FinancialStateArtifacts`. El banco tiene un snapshot común durante cada evento. Los pesos rápidos se reúnen y descargan por bloques físicos. Se propone contexto después de procesar la cohorte completa. El resolver añade feedback validado y archivo histórico antes de publicar el bundle mediante `latest.json`. La poda conserva las referencias históricas de las dos generaciones recuperables.

Las trazas conservan cuantiles brutos, K1 y emitidos, niveles, corrección usada al emitir, acciones solicitadas y ejecutadas, motivos de retorno, trabajo e identidades. Las comprobaciones concilian las salidas con el registro nativo y reaplican la corrección archivada. No utilizan el calibrador actual para reconstruir una emisión anterior. `mechanism_state()` y `mechanism_event(generation)` leen el estado y los eventos confirmados, también después de la poda.

La cola financiera y las cohortes de PT2 tienen miembros diferentes. El enlace conserva las pérdidas financieras válidas aunque una acción no haya reservado calibración o su cohorte se haya cancelado. El cierre administrativo no se convierte en etiqueta. En la frontera reservada, el watermark de feedback conserva el prefijo exclusivo sin modificar la disponibilidad real de los resultados. El estado terminal no puede continuar otra fase.

La integración inicial rechaza M2/M3 y C local activo. B6 mantiene su enlace independiente. La regresión de esas sesiones anteriores no acredita que puedan combinarse con este enlace nuevo. Los sellos y las conciliaciones detectan corrupción e incompatibilidades, sin autenticar un host que pueda modificar arbitrariamente código y archivos.

## Comprobaciones técnicas

La [selección de regresión](verification.json) pasó 710 casos en 317,14 s, sin fallos ni omisiones. Incluye 70 casos marcados CUDA, conciliados con el XML de ejecución. Los subprocesos internos no se cuentan como nuevos casos. El módulo del enlace aporta 37 casos, siete de ellos CUDA. Se mantiene una advertencia de PyTorch sobre su diagnóstico experimental de sincronización. Es una selección focalizada, no toda la suite del proyecto.

Las pruebas del enlace cubren paridad de puntos al desactivar mecanismos, siguiente actualización, feedback parcial, settlement, cierre con y sin resultados anteriores, snapshots comunes, permutación de activos, partición física, poda histórica y reanudación. Los cortes se introducen en `record_written`, `checkpoint_written`, `before_commit` y `committed`. La recuperación compara la siguiente emisión y actualización y comprueba que no se aplique dos veces el mismo resultado.

Los casos adversariales rechazan cuantiles alterados, correcciones distintas, campos no conciliados, feedback ajeno, referencias auxiliares incompatibles, corrupción y agotamiento del archivo sin sustituir la última cabeza confirmada. Una prueba comprueba cuantiles finitos extremos mediante comparación de orden, sin una resta que pueda desbordar.

Las [13 mutaciones dirigidas del enlace](mutations.json) fueron detectadas. Las referencias inicial y restaurada de esa copia CPU pasaron 30 casos sin omisiones. La selección CUDA se comprueba por separado en la regresión. La extracción posterior del generador de fixtures conservó el comportamiento de cuatro flujos y volvió a comprobarse con las sesiones anteriores. Las mutaciones seleccionadas no representan todos los errores posibles.

La [cobertura y CRAP](quality.json) usan coverage.py 7.16.1 y Radon 6.0.1 con la herramienta común del proyecto. Sobre cuatro módulos y 87 funciones se ejecutaron 996/1.094 sentencias y 351/448 ramas. CRAP utiliza cobertura de sentencias. El máximo es 72 en la función anterior `bank_retention`, que esta selección no ejecuta. En el enlace nuevo es 28,43 en `_check_record`. Hay fronteras y rechazos que siguen sin cubrir. Estas medidas describen recorridos observados y complejidad, sin demostrar ausencia de errores.

Los [artefactos externos](external-artifacts.json) identifican los logs, XML, cobertura, perfiles y recetas conservados mediante sus huellas. Incluyen los intentos fallidos de preparación. La factoría inicial modificaba parámetros después de fijar identidad y se corrigió mediante `state_dict` y carga FP64. Dos primeras comparaciones de estados usaban `==` sobre diccionarios con tensores y se corrigió el comparador de pruebas. La primera medida no había declarado `fastpath=False` y falló antes de medir. El primer resumen final de CRAP recibió varios archivos tras una sola opción repetible. Se corrigió el comando y se resumió la cobertura existente, sin repetir las pruebas ya aprobadas. Esos fallos no se cuentan como detecciones de mutantes.

## Coste del recorrido completo

El [benchmark](benchmark.py) mide apertura, un evento de calentamiento, tres emisiones de cohorte, dos recepciones de resultados, cierre censurado, publicación sellada, comprobaciones, poda y recuperación. El total incluye consultar una traza antigua y contar los archivos conservados. La construcción de datos y modelos queda fuera de las latencias y dentro del proceso completo. Se utilizan 64 o 256 activos US sintéticos, bloque físico de 64 y banco de 64 episodios.

Cada fila tiene una ventana de precios de 64 × 5, noticias de dimensión 2, gráficos de dimensión 1, fundamentales de dimensión 3 y macro de dimensión 3. El estado oculto tiene dimensión 32. No incluye los codificadores externos ni el corpus histórico. La precisión es FP64, con TF32 y fastpath MHA desactivados, algoritmos deterministas y `cuda:0`. Se registran dos calentamientos y seis repeticiones por condición. Cada repetición escribe un historial nuevo en ext4 del SSD local.

La [referencia medida](cost-reference.json) utiliza el código conservado. `previous` es el consumidor financiero anterior. `disabled` añade el nuevo enlace con contexto y calibración desactivados. `context_online` añade contexto y PT2 online. `context_reader_online` incorpora además la ruta `k4cr` con el mismo lector y cabeza existentes. Los puntos de `previous` y `disabled` coinciden exactamente por digest en ambos tamaños. Activar la nueva persistencia añade trabajo y bytes, aunque sus mecanismos predictivos estén apagados.

| Activos | Condición | p50 total (ms) | p95 total (ms) | Bytes conservados por trayectoria |
| ---: | --- | ---: | ---: | ---: |
| 64 | previous | 2.134,82 | 2.393,51 | 2.492.453 |
| 64 | disabled | 3.403,71 | 3.622,04 | 2.940.441 |
| 64 | context_online | 3.804,42 | 4.292,52 | 3.162.377 |
| 64 | context_reader_online | 4.082,02 | 4.347,02 | 3.167.305 |
| 256 | previous | 6.682,54 | 7.439,03 | 9.627.314 |
| 256 | disabled | 9.958,44 | 10.083,43 | 11.364.198 |
| 256 | context_online | 11.007,47 | 11.185,94 | 11.824.214 |
| 256 | context_reader_online | 11.512,94 | 11.856,91 | 11.842.646 |

Son latencias de una trayectoria con varias sesiones y operaciones de disco. No son latencias de un forward ni de una petición individual. El recibo conserva cada etapa, p99, media, dispersión y caudal por fila amortizado. Seis muestras no permiten caracterizar de forma precisa las colas de latencia.

Los picos asignados por PyTorch van de 78.161.920 a 80.499.200 bytes. No representan toda la VRAM del proceso ni el uso gráfico del equipo. El máximo reservado del allocator llega a 100.663.296 bytes. El proceso externo de referencia tarda 428,28 s y registra 1.751.224.320 bytes de RSS máximo, incluidos framework, construcción y todas las condiciones. Los bytes conservados de la tabla no son el máximo transitorio de disco o RAM.

El equipo es un Ryzen 9 8945HS con unos 32 GB de RAM y una RTX 4070 Laptop de 8.188 MiB, controlador 595.91.07, Python 3.12.14 y PyTorch 2.14.0+cu130. Se mantuvo la carga gráfica existente. No hubo otras pruebas o perfiles propios simultáneos durante las medidas de coste. No se controlaron temperatura, frecuencias o cachés. No se midió energía. El orden de condiciones fue fijo.

## Optimización descartada

El perfil instrumentado localizó introspección repetida en las verificaciones. Se probó una enumeración por MRO con retorno a `inspect` para metaclases y descriptores especiales, sin caché entre auditorías ni supresión de controles de instancia, parámetros, fuentes o precisión. Sus 18 pruebas focalizadas pasaron. La [mutación de esa candidata](rejected-candidate-mutations.json) detectó ocho fallos dirigidos, con referencias inicial y restaurada aprobadas. Las pruebas incluyen firmas completas de las cuatro referencias financieras, herencia, cambios posteriores, metaclases, descriptores, `staticmethod`, `classmethod` y el comportamiento de `inspect.isfunction` ante `__class__`.

La [comparación](comparison.json) conserva paridad exacta de los ocho digests de predicción y de los bytes retenidos entre referencia y candidata. El tiempo interno de proceso pasa de 424,18 a 422,96 s, un cambio del −0,29 %. La p50 empeora en dos condiciones y no hay una reducción consistente del recorrido completo. La candidata se retiró del código de ejecución. Su [parche reproducible](rejected-class-enumeration.patch) queda como artefacto experimental, sin importar ni activar sus funciones en producción.

La [ronda inicial](cost-initial-diagnostic.json) reconstruía un calendario por cada etiqueta manual. Se conserva como diagnóstico. El benchmark corregido reutiliza la misma disponibilidad ya contenida en el batch y el cierre declarado de la fase. El [protocolo corregido](cost-protocol.json) se guardó antes de medir referencia y candidata con ese mismo archivo. La reducción de coste entre aquella ronda inicial y las posteriores no se atribuye al predictor ni a la enumeración de métodos.

## Reproducción y alcance pendiente

En el entorno del proyecto, con el enlace nativo configurado, las pruebas nuevas se ejecutan mediante:

```bash
MARS_TITAN_REQUIRE_CUDA=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:. uv run --no-sync pytest -q tests/memory/test_financial_mechanisms.py
```

La selección completa se conserva en [verification.json](verification.json). El benchmark exige un directorio de almacenamiento existente en el dispositivo elegido y un archivo de salida nuevo:

```bash
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python reports/engineering/mechanism-session-20261021/benchmark.py --rows 64 256 --warmup 2 --repetitions 6 --storage-root /tmp --output /tmp/mars-session-cost.json
```

`/tmp` puede ser `tmpfs`. Para repetir la medida de SSD se debe indicar un directorio verificado de ese dispositivo. El artefacto descartado se reproduce aplicando su parche sin contexto con `git apply --unidiff-zero` en una copia del código de este informe y repitiendo exactamente el benchmark corregido. No debe aplicarse sobre los runtimes científicos históricos.

Faltan el selector aprendido, los rasgos y objetivos realmente utilizados por él, la composición anidada y sus contrastes temporales. La [revisión de contexto y especialización](../../../docs/research/context-and-nested-control.md) separa esas propuestas de la persistencia implementada. Las comprobaciones presentes no respaldan SOTA ni una mejora predictiva. La campaña histórica completa y el test final siguen cerrados.
