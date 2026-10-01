# Ejecutar la simulación en C++20

`mars-titan-sim` es un ejecutable C++20 autónomo. Lee precios y predicciones congeladas desde Parquet, mantiene posiciones y órdenes entre sesiones y guarda el estado necesario para continuar. La lectura utiliza Arrow C++ y Parquet, con OpenSSL para las huellas de los archivos. El proceso no carga Python.

La versión actual admite escenarios sintéticos de validación. El histórico real necesita una auditoría de ajustes OHLC y acciones corporativas que todavía no está acreditada. Los datos sintéticos permanecen identificados como tales y el test final se rechaza antes de abrir el Parquet.

## Construcción y uso

Desde la carpeta `native`:

```bash
cmake --preset native-release
cmake --build --preset native-release
ctest --preset native-release
```

El ejecutable queda en `build/native/native-release/mars-titan-sim`, respecto a la raíz del repositorio. Las [instrucciones de construcción](../../native/README.md) recogen las dependencias y los perfiles de diagnóstico. La alternativa de construcción basada en PyArrow toma sus bibliotecas C++, sin enlazar `arrow_python` ni `libpython`.

Una cinta contiene `manifest.json` y `market.parquet`. La [preparación de escenarios](persistent-simulation.md) puede producirlas con una referencia analítica o con un padre congelado y sus codificadores verificados. Una vez preparada la cinta, la simulación se ejecuta directamente desde la raíz:

```bash
build/native/native-release/mars-titan-sim \
  --input data/interim/financial-scenarios-v1/known_signal-validation-1042 \
  --output artifacts/native/financial-comparison-v1 \
  --compare --workers 4 --diagnostic
```

`--compare` ejecuta efectivo, conservación de posiciones iniciales y rebalanceo al 50 %, con costes de 0, 10 y 25 puntos básicos. `--workers` admite 1, 2, 4 y 8. Las sesiones comparten una cinta inmutable y cada trabajador posee su estado y su salida. El número de trabajadores no cambia las decisiones ni los resultados. No se presupone que ocho sea la opción más rápida.

`--policy` ejecuta una sola referencia. También admite rebalanceo al 25 %, 75 % y 100 %. `--capital`, `--cost-bps` y `--participation` fijan sus parámetros. `--diagnostic` identifica las comprobaciones técnicas y no modifica los cálculos.

## Cálculo y estado

Las órdenes se calculan al cierre y se ejecutan en la apertura posterior. Las ventas preceden a las compras y cada compra reserva sus costes. La capacidad usa el volumen observado antes de la apertura. Las posiciones no se liquidan al terminar el periodo. No hay cortos, deuda, conversión implícita de monedas ni conexión con un bróker.

El núcleo usa `double` y sumas por parciales para los agregados que pueden perder precisión por cancelación. El redondeo residual del efectivo se concilia con un límite de ocho ULP de los movimientos agregados. Release desactiva `fast-math` y la contracción de operaciones. Los controles de finitud rechazan los desbordamientos detectados antes de confirmar el estado.

La sesión C++ aplica splits, reconoce dividendos y separa el derecho de cobro del efectivo disponible. Un pago al cierre no financia compras de esa apertura. Las bajas requieren una acción acreditada y no se deducen de una cotización ausente. Cuando falta el cierre de una posición, la evaluación queda incompleta. La ruina conocida termina el episodio con la penalización declarada y conserva un retorno de −1.

Los buffers de posiciones, cuentas y trabajo se reutilizan. El cálculo de un paso prepara su estado y recibo antes de confirmarlos. Las sesiones independientes se ejecutan mediante `std::jthread` y un contador atómico de trabajos. No comparten carteras mutables ni crean hilos de lectura adicionales dentro de Arrow.

## Recuperación y verificación

`--stop-after` permite detener una comprobación en una barrera conocida. SIGINT y SIGTERM solicitan una pausa recuperable. `--resume` exige la misma fuente, configuración, política, fuentes nativas y configuración de compilación. Debug y Release tienen identidades distintas. Los checkpoints incluyen posiciones, efectivo, órdenes, dividendos pendientes, acciones aplicadas, cursor y estado de valoración. Las políticas de referencia son deterministas y no utilizan un generador aleatorio.

Los archivos se confirman mediante escritura temporal, sincronización y cambio de nombre. El índice conserva dos estados con huellas verificables. Una corrupción o una identidad incompatible impide recuperar. La salida queda separada de los datos de entrada y cada ejecución utiliza un bloqueo exclusivo.

El programa emite `run.json` por escenario y un resumen de comparación. Los estados contables completos permanecen en `private/checkpoints`. Los recibos distinguen tiempo de lectura compartida, tiempo de ejecución y origen de los datos. Linux proporciona el pico de memoria desde `exec` mediante `VmHWM`. El máximo de `getrusage`, que puede incluir memoria anterior a `exec`, se registra por separado. El observatorio publica únicamente su selección saneada de campos.

Los controles nativos incluyen consumidores C17, contabilidad, eventos, recuperación y concurrencia. Se ejecutan en perfiles separados de Clang, GCC, análisis estático, vida útil, ASan/UBSan, TSan, MSan y fuzzing. MSan cubre el núcleo y la sesión con una biblioteca estándar instrumentada. Las bibliotecas precompiladas Arrow y OpenSSL delimitan su aplicación al ejecutable completo.

La referencia Python y la interfaz C siguen disponibles para comprobaciones y para integrar el motor con los comparadores neuronales. Las pruebas contrastan decisiones, observaciones, costes y estados recuperados. Las cifras de rendimiento deben proceder del ejecutable Release y del recorrido completo, con las diferencias de persistencia entre implementaciones declaradas.

La [medición del recorrido financiero](../../reports/resources/native-go-no-go.md) compara 1, 2, 4 y 8 trabajadores con la referencia Python y recoge sus límites.

La [medición de serialización](../../reports/resources/native-checkpoints.md) contrasta el guardado de eventos al confirmar checkpoints frente a su construcción en cada transición, con 16, 128 y 512 activos.
