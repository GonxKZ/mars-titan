# Simulación financiera nativa: decisión y medidas

La simulación se conserva como ejecutable autónomo C++20. La medición del 24 de septiembre de 2026 registra una reducción del tiempo total de 3,383 s en la referencia Python a 0,325 s con un trabajador C++ y 0,115 s con ocho. Esta decisión se limita al recorrido financiero con precios y predicciones ya preparados. No acredita mejoras de entrenamiento neuronal, error predictivo o rentabilidad real.

## Cuello de botella y cambio

El [perfil previo del recorrido](../episode-pipeline-benchmark.json) utilizó 128 activos, cuatro modalidades, macro y 192 decisiones. Con cuatro lectores, la mediana de simulación fue 0,3575 s dentro de 0,5444 s del recorrido, aproximadamente el 65,7 %. Ocho lectores aumentaron el tiempo a 0,5602 s y elevaron la memoria. La ejecución separada con `cProfile` localizó trabajo en actualización y validación de cartera, valoración, construcción de observaciones y objetos de cotización. Sus tiempos instrumentados no se mezclan con los tiempos sin perfilador.

`mars-titan-sim` lee una cinta mediante Arrow y Parquet C++, mantiene sus datos contiguos e inmutables y reutiliza buffers contables por sesión. Las nueve combinaciones de política y coste comparten esa cinta. Cada trabajador posee su cartera, sus órdenes y sus archivos. El contador de trabajos es atómico y el límite es de ocho trabajadores. Arrow no crea otros hilos de lectura en esta ruta.

El núcleo de ejecución de órdenes y valoración recorre los activos y las cuentas de forma lineal. La sesión también ordena activos y, al rebalancear, candidatos positivos por predicción e identificador, con coste O(N log N). Los eventos y derechos de cobro añaden trabajo según sus cantidades. Se conserva el orden temporal de las sesiones. El paralelismo se aplica a experimentos independientes, sin introducir reducciones concurrentes que cambien la contabilidad.

La contabilidad utiliza FP64 y sumas por parciales. Las observaciones conservan FP32, como la referencia. Las compilaciones mantienen `-fno-fast-math` y `-ffp-contract=off`. No se ha reducido la precisión para obtener los tiempos. Los comprobadores incluyen desbordamiento, importes no finitos, cancelación, efectivo próximo al máximo representable, lotes fraccionarios y pagos diferidos. Esta precisión numérica no elimina los límites del mecanismo generador ni demuestra precisión sobre mercados.

## Comparación del ejecutable completo

El [registro de medición](native-financial-benchmark.json) contiene cinco repeticiones por configuración, tras un calentamiento, en un Ryzen 9 8945HS de 16 CPU lógicas. Se usaron Clang 21.1.8, C++20, Python 3.12.14, NumPy 2.5.3 y Arrow 25.0.1. Los tiempos abarcan creación del proceso, carga de bibliotecas, lectura y salida. La generación previa de la cinta es común y queda fuera de ambos tiempos.

La cinta sintética tiene 128 activos, 193 sesiones y 192 decisiones. Se evalúan efectivo, conservación inicial y rebalanceo al 50 %, cada uno con costes de 0, 10 y 25 puntos básicos. El padre es una referencia analítica fija. El Parquet ocupa 1.015.016 bytes. C++ deja aproximadamente 218.534 bytes entre recibos y checkpoints, mientras Python escribe un resumen de unos 2.321 bytes.

| Implementación | Trabajadores | Mediana total (s) | Mínimo y máximo (s) | Desviación (s) | Mediana del pico (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python | 1 | 3,3833 | 3,3354 a 3,4023 | 0,0251 | 133,75 |
| C++20 | 1 | 0,3247 | 0,3176 a 0,3531 | 0,0157 | 37,18 |
| C++20 | 2 | 0,2488 | 0,2479 a 0,2529 | 0,0022 | 37,47 |
| C++20 | 4 | 0,1756 | 0,1726 a 0,1808 | 0,0038 | 37,94 |
| C++20 | 8 | 0,1150 | 0,1111 a 0,1164 | 0,0022 | 38,88 |

La mejora de las medianas es 10,42 veces con un trabajador y 29,41 veces con ocho. Con ocho se procesan unas 15.023 decisiones de cartera por segundo en este lote de nueve escenarios. Se conservan todas las repeticiones. Cinco repeticiones no permiten caracterizar de forma estable percentiles extremos ni acreditar un número óptimo de trabajadores para otras cargas.

La diferencia absoluta máxima observada en las métricas financieras fue cero. El comprobador admite tolerancias de 1e-12 relativas y 1e-10 absolutas, y contrasta también los campos no numéricos. Las pruebas de integración comparan estados, órdenes, observaciones y recuperación, además de las métricas agregadas.

La memoria usa `VmHWM` de Linux después de `exec`. Se publica por separado el máximo de `getrusage`, que puede incluir memoria heredada antes de ejecutar el binario. Todos los hilos comparten el pico del proceso. La [documentación de procfs](https://www.kernel.org/doc/html/latest/filesystems/proc.html#process-specific-subdirectories) define `VmHWM` como el máximo residente y advierte que la contabilidad RSS asíncrona puede ser imprecisa. No representa una contabilidad exacta de cada asignación.

La caché del sistema estaba caliente. No se detuvieron aplicaciones ni la campaña científica activa. El registro incluye carga media del sistema antes y después. C++ guarda checkpoints y bloqueos, mientras la referencia Python de esta medida solo evalúa y escribe el resumen. Por tanto, los servicios de persistencia difieren y se declaran expresamente. No se midieron energía, coste monetario, VRAM ni transferencias CPU/GPU. Los bytes persistidos describen archivos finales, no el tráfico físico del dispositivo.

## Reproducción y uso admitido

Desde `native`, construir `native-release` con CMake y ejecutar sus CTests. Desde la raíz del repositorio, con una carpeta de trabajo nueva:

```bash
uv run --no-sync python benchmarks/native_financial.py \
  --binary build/native/native-release/mars-titan-sim \
  --work artifacts/benchmarks/native-financial-local \
  --output artifacts/benchmarks/native-financial-local.json \
  --repetitions 5
```

El informe identifica la fuente, el ejecutable, su configuración de compilación y el código del benchmark mediante SHA-256. La [guía del ejecutable](../../docs/engineering/native-financial-simulation.md) recoge admisión, fallos y recuperación. La referencia Python y el adaptador C permanecen disponibles para contrastar y conectar otros consumidores. El ejecutable funciona sin intérprete Python.

La [verificación nativa](native-financial-verification.md) recoge pruebas, análisis, sanitizadores, cobertura y mutación sobre este corte.

Se admite esta ruta para las comparaciones sintéticas descritas. La simulación histórica de varios días sigue condicionada a acreditar ajustes OHLC y acciones corporativas. El test final continúa cerrado y MARS-TITAN permanece como diseño futuro.
