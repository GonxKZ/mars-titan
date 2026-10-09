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

## Reglas de mercado

Una cinta con manifiesto de versión 2 declara las reglas de cada activo. La sesión aplica entonces lote, mínimo por orden, venta del resto impar, bandas diarias y timbre por fecha mediante `mt_simulation_step_v2`, y los recibos añaden `market_rules = "mt_simulation_step_v2/mt_rules_v1"`. Sin reglas, la llamada sigue siendo v1. La semántica, las fuentes de las acciones A y la paridad con Python están en la [revisión de reglas chinas](china-market-rules.md#motor-nativo).

El límite exacto necesita la representación decimal más corta de la referencia y de la banda. En el primer perfil con callgrind, `std::to_chars` era la función nueva más costosa (10,8 % de las instrucciones con reglas). Para decidir si una apertura bloquea una orden, el núcleo compara ahora con el producto binario de referencia y banda. Ese producto difiere del límite decimal en medio céntimo más menos de 1e-15 veces el límite superior. Si la apertura está a más de un céntimo más 1e-12 veces ese límite, la decisión coincide con la del cálculo exacto, que solo se hace dentro de esa franja o con límites desde una décima parte de 2^53 céntimos. La banda se lee una vez mientras se repite entre activos. `simulation_tests` recorre 2.000 referencias, en céntimos y fuera de la rejilla, con ocho bandas hasta 0,9999999999. Coloca aperturas en el límite, en los `double` contiguos, a medio céntimo, a un céntimo y en el producto binario, y exige la misma decisión que `mt_simulation_price_limits_v1`. Las mutaciones del margen, de la reutilización de la banda, de cada dirección y del tope de 2^53 céntimos se detectan.

Las medidas usan Release con Clang 21.1.8 en un AMD Ryzen 9 8945HS de 16 hilos, compartido con otros procesos (carga media entre 10 y 11). `mars-titan-batch-benchmark --rules cn --capital 10000000` usa una cinta sintética con precios en céntimos, lote de 100, venta del resto impar, banda del 10 % y timbre de venta del 0,5 ‰. `--rules none` recorre la misma cinta con el mismo capital. Cada configuración tiene 128 sesiones y las variantes se intercalan en 15 repeticiones. La tabla da la mediana de transiciones por segundo y, entre paréntesis, el mínimo y el máximo.

| Entornos | Activos | Trabajadores | Sin reglas | Reglas con límite siempre exacto | Reglas con decisión acotada | Relación con sin reglas |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | 1 | 178.654 (108.352 a 211.823) | 133.803 (84.298 a 156.324) | 164.852 (102.380 a 192.382) | 0,92 |
| 256 | 64 | 1 | 152.280 (103.654 a 175.235) | 107.511 (73.151 a 121.798) | 128.610 (95.742 a 163.665) | 0,84 |
| 64 | 512 | 1 | 19.520 (14.488 a 25.607) | 14.428 (10.901 a 16.500) | 18.356 (12.983 a 21.238) | 0,94 |
| 256 | 64 | 2 | 210.496 (177.463 a 261.728) | 154.123 (130.844 a 209.612) | 185.981 (155.360 a 276.887) | 0,88 |

Las dos variantes con reglas dan el mismo checksum en cada configuración. La dispersión es amplia por la carga ajena, así que el recuento de instrucciones de callgrind 3.26 sirve de referencia más estable. Con 16 entornos, 64 activos y 64 sesiones, la ruta sin reglas ejecuta 70,24 millones, las reglas con límite siempre exacto 97,27 millones y la versión final 79,09 millones. El sobrecoste de las reglas baja así del 38,5 % al 12,6 %. Lo que queda se reparte sobre todo entre la preparación de reglas por activo en la sesión y su validación en el núcleo, con el 5,6 % y el 3,7 % de las instrucciones con reglas. Las cintas con y sin reglas no ejecutan exactamente las mismas órdenes, porque el lote de 100 cambia las cantidades, y parte de la diferencia no procede del cálculo de reglas.

La ruta sin reglas no empeora frente a `develop`. Con el capital por defecto, el binario y la biblioteca anteriores dan el mismo checksum que los nuevos. Sus medianas en las mismas cuatro configuraciones son 183.763, 160.537, 22.405 y 206.345 transiciones por segundo, frente a 183.077, 153.698, 22.977 y 207.539, dentro de la dispersión. Callgrind cuenta 70,19 millones de instrucciones antes y 69,97 millones después.

En Python, `FinancialEnv(backend="native")` tarda una mediana de 0,639 ms por paso sin reglas y 0,659 ms con reglas con 64 activos, y 1,299 ms y 1,289 ms con 256 activos (128 sesiones, 9 repeticiones, con desviaciones típicas entre 0,13 y 0,27 ms). La envoltura Python domina ese tiempo y la diferencia queda dentro de la dispersión.

## Recuperación y verificación

`--stop-after` permite detener una comprobación en una barrera conocida. SIGINT y SIGTERM solicitan una pausa recuperable. `--resume` exige la misma fuente, configuración, política, fuentes nativas y configuración de compilación. Debug y Release tienen identidades distintas. Los checkpoints incluyen posiciones, efectivo, órdenes, dividendos pendientes, acciones aplicadas, cursor y estado de valoración. Las políticas de referencia son deterministas y no utilizan un generador aleatorio.

Los archivos se confirman mediante escritura temporal, sincronización y cambio de nombre. El índice conserva dos estados con huellas verificables. Una corrupción o una identidad incompatible impide recuperar. La salida queda separada de los datos de entrada y cada ejecución utiliza un bloqueo exclusivo.

El programa emite `run.json` por escenario y un resumen de comparación. Los estados contables completos permanecen en `private/checkpoints`. Los recibos distinguen tiempo de lectura compartida, tiempo de ejecución y origen de los datos. Linux proporciona el pico de memoria desde `exec` mediante `VmHWM`. El máximo de `getrusage`, que puede incluir memoria anterior a `exec`, se registra por separado. El observatorio publica únicamente su selección saneada de campos.

Los controles nativos incluyen consumidores C17, contabilidad, eventos, recuperación y concurrencia. Se ejecutan en perfiles separados de Clang, GCC, análisis estático, vida útil, ASan/UBSan, TSan, MSan y fuzzing. MSan cubre el núcleo y la sesión con una biblioteca estándar instrumentada. Las bibliotecas precompiladas Arrow y OpenSSL delimitan su aplicación al ejecutable completo.

La referencia Python y la interfaz C siguen disponibles para comprobaciones y para integrar el motor con los comparadores neuronales. Las pruebas contrastan decisiones, observaciones, costes y estados recuperados. Las cifras de rendimiento deben proceder del ejecutable Release y del recorrido completo, con las diferencias de persistencia entre implementaciones declaradas.

La [medición del recorrido financiero](../../reports/resources/native-go-no-go.md) compara 1, 2, 4 y 8 trabajadores con la referencia Python y recoge sus límites.

La [medición de serialización](../../reports/resources/native-checkpoints.md) contrasta el guardado de eventos al confirmar checkpoints frente a su construcción en cada transición, con 16, 128 y 512 activos.
