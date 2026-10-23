# Recogida de alternativas en la generación financiera

Fecha de la comprobación: 21 de octubre de 2026.

La recogida opcional guarda las alternativas condicionales antes de recibir etiquetas y conserva sus resultados bajo la cabeza de publicación de `FinancialSession`. No ajusta el selector ni los parámetros compartidos. No se han usado datos reales ni el test final. Este trabajo no mide una mejora predictiva.

`mechanism_collection=MechanismCollectionConfig()` activa el archivo y `None` conserva la ruta anterior. El índice `mechanism_collections` enlaza paquetes y recibos separados. El control y el historial guardan `collection_packet`. Se siguen usando el mismo banco, codec, ejecutor, snapshots y bloques de pesos rápidos. No hay una segunda cabeza de publicación.

Cada bloque calcula alternativas desde el estado anterior y ejecuta también la ruta real. Los estados contrafactuales se descartan. Los cuantiles del ancla se contrastan con la salida real con tolerancias declaradas de 1e-12 en FP64 y 1e-5 en FP32. El error y la admisión episódicos utilizan la predicción nativa emitida, que se conserva aparte. Cambiar una acción de un mercado mantiene las anclas de los demás. Esto no enumera todas las acciones conjuntas ni estima la utilidad de una trayectoria autónoma.

La recepción utiliza todos los resultados validados por el ejecutor, incluidos los que PT2 no espera por sus reservas o por la cancelación de una cohorte. Cada resultado enlaza fila, generación y hash del registro nativo. Una recepción parcial no elimina las pérdidas de otras filas. El reloj del receptor es el watermark de disponibilidades recibidas. El instante real de recepción sigue en el registro nativo. El cierre administrativo guarda sus censuras por separado, con su fecha original. No crea etiquetas, no desplaza su disponibilidad y no asigna pérdida cero a los casos pendientes o censurados.

Las APIs `mechanism_collection(generation)` y `mechanism_collection_metrics(generation, device="cuda:0")` recuperan el paquete, los resultados, las censuras y las pérdidas por mercado y sesión. La recuperación contrasta todo el archivo. La verificación ordinaria lee la recogida actual y las pendientes. La poda enumera sus referencias directamente desde el índice, sin buscar archivos dentro de los payloads históricos.

La capacidad por defecto es de 64 decisiones y se puede configurar entre 1 y 256. Continúan los límites generales de 1.024 artefactos y 2 GiB, que pueden agotarse antes. Una propuesta que no cabe falla sin sustituir la cabeza confirmada ni eliminar evidencia. El formato del paquete admite 8.192 filas, pero el productor sigue limitado por el máximo del consumidor, que no supera 4.096. No se declara preparada una recogida científica de duración ilimitada.

## Comprobaciones ejecutadas

La [verificación](verification.json) registra 220 pruebas aprobadas sin fallos, errores ni omisiones, en una pasada de 217 y otra de 3 para la guarda adicional. Hay 38 casos marcados CUDA. Las dos nuevas baterías aportan 40 casos, 10 de ellos CUDA. La selección incluye las regresiones de GRU, corrección asociativa, retención y mecanismos anteriores. No es toda la suite del proyecto.

Se comprueban cuatro cortes de publicación en CPU y CUDA, recuperación de la siguiente recogida y recepción parcial, permanencia de artefactos tras podar checkpoints, igualdad de las emisiones y estados rápidos, errores basados en la predicción realmente emitida, las 36 alternativas desde un prefijo poblado con selección automática, independencia de RNG, corrupción de archivos, capacidad agotada sin sustituir la cabeza y cierre administrativo sin etiquetas ficticias. Esta ronda de publicación y coste utiliza FP64. La tolerancia FP32 está declarada en la implementación y no sustituye una comprobación positiva de esta unión en FP32.

La [mutación dirigida](mutations.json) detecta 10 de 10 cambios seleccionados, con bases inicial y restaurada de 27 pruebas CPU cada una. Incluye pérdida de referencias vivas, omisión del paquete o del recibo, etiquetas cambiadas, procedencia omitida, watermark futuro, censuras mal contadas y ejecución de un resolver sustituido. Una primera detección dependía del mensaje de error y no se usó como evidencia de conducta. El intento siguiente conservó un superviviente por otras validaciones redundantes. La comprobación final exige rechazar el resolver sustituido antes de invocarlo y detecta la omisión por una publicación que indebidamente tendría éxito. No es un porcentaje global de mutación.

La [cobertura y el CRAP](quality.json) usan coverage.py 7.16.1 con ramas y Radon 6.0.1. Se midieron 69 funciones de tres módulos, con 911 de 1.026 sentencias y 313 de 410 ramas ejecutadas. El módulo nuevo alcanza 181 de 184 sentencias y 38 de 40 ramas. Su mayor CRAP es 30,24 en la verificación de procedencia. En la selección conjunta quedan funciones antiguas sin cubrir, como `_compact_inputs` y `_admission_features`, con CRAP 132 y 110. No se han dividido funciones ni eliminado validaciones para reducir esas puntuaciones.

Ruff, el formato y la comprobación de repositorio pasan. El [índice externo](external-artifacts.json) conserva hashes de los recibos completos. No se han añadido dependencias, kernels ni cambios C++, por lo que no se atribuye una compilación o un análisis nativo nuevo a esta ronda.

Para repetir la selección en el entorno fijado, con el enlace nativo y CUDA disponibles:

```bash
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync pytest -q tests/memory/test_native_mechanism_collection.py tests/memory/test_native_collection_guard.py tests/memory/test_financial_mechanisms.py tests/memory/test_mechanism_selection.py tests/memory/test_mechanism_collection.py tests/memory/test_mechanism_supervision.py tests/memory/test_financial_session_gru.py tests/memory/test_financial_session_associative.py tests/memory/test_financial_session_controls.py
```

## Coste del recorrido completo

La prueba utiliza 64 y 256 activos US sintéticos, FP64, tamaño oculto 32, bloques físicos de 64, dos hilos CPU y `cuda:0` en una RTX 4070 Laptop. Se declara un calentamiento y tres repeticiones medidas por modo y tamaño, alternando los modos. Cada recorrido comprende calentamiento, tres decisiones, recepción de etiquetas manuales y cierre. El banco tiene capacidad para 256 episodios y la última decisión dispone de un prefijo poblado.

| Activos | Ruta fija, mediana | Ruta con recogida, mediana | Diferencia máxima en puntos |
| --- | ---: | ---: | ---: |
| 64 | 3.287,93 ms | 4.894,21 ms | 0 |
| 256 | 10.342,55 ms | 16.158,05 ms | 0 |

Son trabajos distintos. La segunda ruta emite el control y recoge además las 36 alternativas disponibles. Incluye admisión, codificación del codec existente, transferencias, inferencia, calibración, registros, sellado, verificaciones, poda y `fsync`. Excluye preparación de fixtures, construcción del modelo y de la sesión, extracción externa de modalidades, ajuste del selector y consulta posterior de métricas. Los archivos temporales estaban en `/tmp`, montado como `tmpfs`. Esta medida no caracteriza la escritura en SSD ni la durabilidad ante un corte de alimentación.

Los estados rápidos y los recuentos financieros coinciden. Con 256 activos, el pico de memoria asignada por PyTorch fue de 80.499.200 bytes en ambos modos. Los archivos nativos y artefactos ocuparon 12.588.780 bytes en la ruta fija y 15.590.805 en la recogida, dentro de `tmpfs`. El último paquete ocupó 982.146 bytes. El [resumen comparativo](comparison.json) conserva también el caudal de observaciones del recorrido. El proceso completo registró 151,69 s internos y 155,27 s externos. Su máximo residente externo fue 1.739.902.976 bytes. No se atribuye ahorro de RAM a cada modo ni se confunde la memoria del asignador PyTorch con toda la VRAM. Los percentiles p95 y p99 del JSON son descriptivos de tres muestras, insuficientes para estimar colas con precisión. No se midió energía. No hubo otras cargas propias durante estas medidas ni se detuvieron aplicaciones ajenas.

El [protocolo](cost-protocol.json), los [resultados](cost.json) y el [recibo del proceso](process-cost.json) conservan sus parámetros y huellas. El ensayo pequeño de cuatro filas se utilizó para comprobar el ejecutable y queda separado de estas cifras.

Para repetir la medida en el entorno fijado del proyecto, con el enlace nativo configurado y una salida nueva:

```bash
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python reports/engineering/native-mechanism-collection-20261021/benchmark.py --output /tmp/mars-titan-native-collection-cost.json
```

El ajuste protegido del selector, su selección temporal y los contrastes predictivos siguen pendientes. La preparación de estas pérdidas no demuestra utilidad financiera ni superioridad frente al estado del arte.
