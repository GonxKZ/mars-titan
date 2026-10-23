# Catálogo y consumidor provisional de mecanismos

El código de este incremento está en `2a5d23191`, sobre la rama de [#701](https://github.com/GonxKZ/mars-titan/issues/701). Añade 36 combinaciones fijas de contexto, rutas episódicas y calibración. Ejecuta una transición MAC por observación y devuelve propuestas sin publicación durable. No implementa todavía el selector aprendido ni su enlace completo con `FinancialSession`. El [contrato](../../../docs/engineering/mechanism-controller.md) describe esos límites.

Las pruebas y medidas utilizan fixtures explícitos. No hay corpus real, pasos de optimizador, estudio predictivo nuevo ni acceso al test final. El origen biológico de las hipótesis anteriores no atribuye propiedades cerebrales ni una mejora financiera a este consumidor.

## Comprobación

La selección final pasó 530 casos en 109,17 s, sin fallos ni omisiones. Incluye los 62 casos identificados por la marca `cuda` y comprobaciones nativas de las sesiones anteriores. Los subprocesos internos no se suman como nuevos casos. El [recibo](verification.json) enumera los módulos y los casos CUDA. El consumidor nuevo aporta 117 casos, con las 36 combinaciones ejecutadas tanto en CPU como en CUDA.

Se comprobaron aislamiento de estados y RNG, desactivación, un avance MAC, snapshots comunes, permutaciones, partición del lote, madurez y orden del feedback, capacidad anterior al feedback, reversión conjunta entre mercados, cuantiles, recuperación auxiliar y archivo terminal censurado. Una rejilla lógica de trabajo limita la ruta antes de dividir el evento en bloques físicos. Las propuestas descartadas conservan sus costes lógicos y no alteran el snapshot original.

El desarrollo detectó un desbordamiento de la cabeza escalar y una repetición innecesaria de la cabeza por mercado. Los errores numéricos se rechazan y las filas con la misma ruta comparten su cálculo. La prueba inicial de partición exigía igualdad de bits entre GEMM de distintos tamaños. El contrato corregido utiliza `rtol=atol=1e-12` en FP64 y conserva igualdad exacta de las decisiones y contadores. No se atribuye paridad de bits a todos los repartos físicos.

Las [18 mutaciones del catálogo y consumidor](mutations.json) y las [tres de introspección](signature-mutations.json) quedaron detectadas. Sus referencias iniciales y restauradas pasaron. Se ejecutaron en CPU, en una copia separada. Las [recetas](mutation-recipes.json) identifican fuente, cambios y prueba. Son 21 fallos dirigidos, no el conjunto de mutaciones posibles. La repetición de introspección comprueba un contrato de trabajo, no un error numérico.

La [cobertura y CRAP](quality.json) corresponden a coverage.py 7.16.1 y Radon 6.0.1. Sobre nueve módulos se ejecutaron 1.353 de 1.473 sentencias y 446 de 562 ramas. CRAP usa cobertura de sentencias. El mayor valor de una función nueva es 41,52 en `begin`. La selección no cubre todas las validaciones históricas de `FinancialConfig.__post_init__`, cuyo CRAP es 454,70 en este recibo. Estos números no demuestran ausencia de errores ni equivalen a auditar todo el proyecto.

El primer intento de cobertura con nombres de módulos terminó con señal 139 antes de registrar pruebas. El diagnóstico mínimo terminó con `UnicodeDecodeError` al importar `torch._C`. La ejecución con `--source=src/mars_titan` pasó. Los logs se conservan con [huellas externas](external-artifacts.json). No se ha identificado una causa interna definitiva de ese fallo de importación. La selección final conserva una advertencia de PyTorch sobre su diagnóstico experimental de sincronización.

## Coste y optimización

El [programa](benchmark.py) mide tres sesiones consecutivas, una de calentamiento y dos de predicción, con 64 o 256 activos repartidos entre dos mercados. Incluye preparación CPU de entradas, transferencias, auditorías, estado inicial, MAC, creación y lectura de una instantánea de 64 episodios, contexto, calibración, feedback manual, exportación, restauración auxiliar y serialización en RAM. No incluye publicación atómica, disco ni restauración completa de los pesos rápidos. La dimensión oculta es 32 y el contexto de precios tiene 64 sesiones.

Se utilizó `cuda:0`, RTX 4070 Laptop de 8.188 MiB, controlador 595.91.07, PyTorch 2.14.0+cu130 y FP64 estricto, con dos hilos CPU. Los recibos conservan las demás versiones, calentamiento, muestras, p50/p95/p99, caudal y memoria. No había otra carga propia de GPU y se mantuvo la aplicación gráfica existente. No se controlaron las frecuencias ni la temperatura. No se midió energía.

El perfil identificó introspección repetida en las auditorías. Las instancias de una clase comparten métodos, por lo que se reutiliza su inspección solo dentro de cada llamada a `_execution_signature`. Se siguen comprobando hooks, atributos y parámetros por instancia. La siguiente auditoría vuelve a inspeccionar la clase y detecta cambios de métodos. En el perfil de la condición combinada, las llamadas a `inspect.getmembers` bajaron de 969 a 323. Los perfiles [anterior](profile-before.json) y [posterior](profile-after.json) son diagnósticos instrumentados, separados de las medidas de latencia.

La [primera ronda](cost-before.json) y la [primera candidata](cost-first-candidate.json), con veinte repeticiones por condición, se conservan completas. La candidata aumentó un 3,54 % el p50 de la combinación online de 256 activos. Ese resultado motivó un contraste alternado, cuyo [protocolo](cost-protocol.json) se guardó antes de ejecutarlo. No se retiró la condición desfavorable.

El contraste A1–B1–B2–A2 utilizó dos calentamientos y ocho repeticiones por ronda. Se agrupan las dieciséis muestras de cada versión y condición. A conserva la inspección anterior y B reutiliza métodos dentro de la llamada. Los [resultados comparados](comparison.json) y los recibos [A1](cost-A1.json), [B1](cost-B1.json), [B2](cost-B2.json) y [A2](cost-A2.json) conservan todos los intentos. Los tiempos de la tabla son p50 de una trayectoria completa de tres sesiones, en milisegundos.

| Activos | Condición | A | B |
| --- | --- | ---: | ---: |
| 64 | Consumidor anterior | 59,58 | 45,90 |
| 64 | Consumidor nuevo desactivado | 131,70 | 108,72 |
| 64 | Lector K4 con renovación | 143,71 | 112,77 |
| 64 | Contexto y calibración estática | 164,92 | 152,32 |
| 64 | Contexto, lector y calibración online | 198,43 | 183,99 |
| 64 | Retorno por falta de bytes | 180,69 | 161,03 |
| 256 | Consumidor anterior | 83,14 | 78,36 |
| 256 | Consumidor nuevo desactivado | 173,48 | 145,65 |
| 256 | Lector K4 con renovación | 200,51 | 157,18 |
| 256 | Contexto y calibración estática | 245,56 | 180,25 |
| 256 | Contexto, lector y calibración online | 328,16 | 283,71 |
| 256 | Retorno por falta de bytes | 218,58 | 200,31 |

El p50 baja entre un 5,75 % y un 26,60 %. Los percentiles superiores no mejoran de forma uniforme. El p95 de la condición online de 64 activos pasa de 208,36 a 219,86 ms y el del retorno de 256 activos pasa de 235,57 a 237,92 ms. Dieciséis muestras no caracterizan de forma precisa las colas de latencia. La optimización reduce trabajo de inspección y el tiempo central observado, sin acreditar una aceleración uniforme de todos los percentiles o de una campaña real.

La [comparación numérica separada](benchmark-parity.json) conserva paridad exacta de las predicciones FP64, los contadores y los tamaños serializados en las doce condiciones. Las fuentes A y B tienen huellas diferentes y explícitas. Los recibos de coste parten de `e8675908e` con las huellas reales de los cambios que después se publican en `2a5d23191`. El consumidor nuevo requiere más comprobaciones y estado auxiliar que la API anterior, por lo que esta tabla no demuestra que sea más rápido que ella.

El pico asignado por PyTorch ronda 98 MiB en las condiciones de 256 activos. Este contador no incluye toda la memoria del controlador CUDA. El RSS máximo registrado desde Python está entre 1,60 y 1,61 GB en las rondas alternadas y corresponde al proceso completo. Los recibos conservan además memoria reservada y bytes de propuestas. No se observa un ahorro de memoria que justifique atribuirlo a la caché de inspección.

## Reproducción y límites de integración

En el entorno del proyecto con CUDA y los artefactos nativos ya configurados, la selección está enumerada por módulos en `verification.json`. Para ejecutar los componentes nuevos:

```bash
MARS_TITAN_REQUIRE_CUDA=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:. uv run --no-sync pytest -q tests/models/titans/test_mechanism_consumer.py tests/models/titans/test_mechanism_catalog.py tests/models/titans/test_mechanism_routes.py tests/models/titans/test_context_selection.py tests/models/titans/test_frozen_signature.py tests/calibration/test_cohort_capacity.py
```

La salida del benchmark debe ser nueva y su directorio debe existir:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src:. uv run --no-sync python reports/engineering/mechanism-consumer-20261021/benchmark.py --output /tmp/mars-titan-mechanism-cost.json --repetitions 20 --warmup 3
```

Faltan la publicación conjunta del contexto, calibrador y trazas en la generación nativa, sus cortes adversariales, el filtrado de resultados ya validados, la recuperación durable y la poda. También faltan el ajuste y la evaluación temporal del selector con objetivos apropiados para puntos e intervalos. No se ha ejecutado una evaluación científica del controlador ni se ha demostrado una mejora sobre sus padres o sobre residual cero. La campaña completa y el test final siguen cerrados.
