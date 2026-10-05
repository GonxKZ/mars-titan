# Escenarios controlados de memoria y ruido

`mars-titan-memory-stress` ejecuta un control técnico de retención con secuencias que superan la capacidad del banco. Compara expulsión aleatoria, reciente y selectiva, con la misma entrada, consulta, capacidad y admisión. No implementa ni entrena el candidato MARS-TITAN. Los errores sintéticos no acreditan predicción financiera ni confirman N1 o N6 del [registro de hipótesis](../research/novelty-ledger.md).

## Contrato de la comparación

El ejecutable reutiliza `MemoryRecord`, los vectores de 64 componentes y la capacidad máxima de 1.024 entradas del núcleo episódico. Mantiene un banco independiente por política y utiliza GEMV de ATen en CPU para consultar las claves normalizadas. La predicción es la media de las etiquetas de hasta cuatro vecinos, con desempate por ID creciente. El banco vacío emite cero. No hay ajuste de pesos, HMM ni codificador entrenable.

Todos los registros válidos entran una vez cuando madura su etiqueta. Las tres políticas escriben el mismo registro y expulsan una fila si el banco está lleno:

| Política | Fila expulsada | Consecuencia del contrato |
| --- | --- | --- |
| `uniform` | Una posición con probabilidad uniforme | Es expulsión aleatoria, no un reservorio uniforme sobre todo el historial |
| `recent` | La entrada más antigua | Conserva los últimos candidatos admitidos |
| `selective` | La prioridad de admisión más baja | El nuevo candidato entra siempre y las prioridades antiguas permanecen fijas |

El uniforme histórico necesita rechazos y tendría otro número de escrituras. Este contraste fija la admisión y cambia solo la expulsión. No añade escrituras de relleno para igualar cifras. La consulta examina el mismo número de filas en las tres políticas, aunque cambia su contenido. La selectiva también recorre las prioridades al expulsar y ese trabajo aparece por separado.

La prioridad usa el error de un control común exponencial, cuyo pronóstico se guarda antes de observar el resultado. Para el error maduro e, la escala anterior s y el estado de persistencia p:

```text
z = clip(e / (1 + s), -4, 4)
p_nuevo = 0,95 × p + 0,05 × z
prioridad = abs(z × p_nuevo)
s_nueva = 0,95 × s + 0,05 × abs(e)
```

La predicción del control común se actualiza con `0,95 × predicción + 0,05 × etiqueta`. Estas constantes están fijadas en el ejecutable. La prioridad no utiliza verdad latente, etiquetas futuras, relevancia económica ni una clasificación retrospectiva del régimen. Es un control por error persistente, no la puntuación completa de N1. Su coste común se incluye en el tiempo del recorrido.

## Escenarios y cronología

Cada observación avanza dos unidades de tiempo. `delay` fija el retraso de maduración en esas unidades. Se emiten las tres predicciones antes de entregar las etiquetas que maduran en el corte actual. Al terminar se vacían los pendientes sin generar predicciones adicionales.

| Escenario | Secuencia | Información disponible para predecir |
| --- | --- | --- |
| `recurrence` | A, B, C, A, repetida por bloques de `regime_length` | Clave con ruido alrededor de tres prototipos observables |
| `persistent` | Medias 1, −1, 0,5, 1, por los mismos bloques | Clave alrededor de un único prototipo, sin indicador del cambio |
| `noise` | Media latente cero y etiqueta independiente de las claves | Claves con la misma estructura contextual, sin señal sobre la etiqueta |
| `outliers` | Recurrencia con perturbaciones de ±8 cada 31 observaciones | La misma clave ruidosa, sin aviso de la perturbación |

Las etiquetas añaden ruido uniforme centrado, con desviación típica `noise`. Los generadores de claves, etiquetas y expulsión tienen estados separados. Cambiar la semilla de retención no modifica los datos ni los controles deterministas. La configuración identifica las semillas de datos y retención. La reproducibilidad comprobada corresponde al mismo ejecutable, biblioteca estándar, LibTorch y equipo.

Cada `invalid_every` observaciones se alternan NaN, infinito y clave nula. La admisión comprueba la clave real. Una observación inválida se cuenta y se excluye de las predicciones evaluadas y de las escrituras. Los atípicos finitos del cuarto escenario siguen siendo observaciones válidas. `invalid_every=0` desactiva la corrupción.

La verdad latente solo alimenta el MAE diagnóstico y la agrupación por fase. El MAE frente a la etiqueta observada se informa por separado. Una prueba modifica la verdad latente pendiente y verifica que permanecen iguales las predicciones y los recuerdos retenidos.

## Ejecución y recuperación

El módulo CMake `native/cmake/MemoryStress.cmake` define `mars_titan_memory_stress`, `mars-titan-memory-stress` y `memory_stress_tests`. Se activa con `MARS_TITAN_BUILD_MEMORY_STRESS=ON`. Usa el SDK de LibTorch del entorno uv existente, sin sincronizarlo ni cambiar dependencias.

```bash
cmake -S native -B build/memory-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DMARS_TITAN_BUILD_SIMULATION=OFF \
  -DMARS_TITAN_BUILD_MEMORY_STRESS=ON \
  -DMARS_TITAN_LIBTORCH_ENABLE_CUDA=OFF \
  -DMARS_TITAN_WARNINGS_AS_ERRORS=ON
cmake --build build/memory-release --parallel 1
ctest --test-dir build/memory-release --output-on-failure
build/memory-release/mars-titan-memory-stress --device cpu \
  --config configs/memory/stress.json --output build/memory-result.json
```

La CPU se declara expresamente y ambos grupos de hilos de ATen se fijan a uno. No existe una ruta CUDA en este control. La configuración independiente contiene 8.192 pasos, capacidad 1.024, bloques de 2.048 pasos, retraso 8, semillas 42 y 73, ruido 0,25 e inválidos cada 97 observaciones. La CLI también admite estos valores por argumentos. `--config` y `--resume` rechazan cambios simultáneos de parámetros para evitar una reanudación con otra definición del experimento.

```bash
build/memory-release/mars-titan-memory-stress --device cpu \
  --config configs/memory/stress.json --stop-after 6001 \
  --checkpoint build/memory-checkpoint.json --output build/memory-paused.json
build/memory-release/mars-titan-memory-stress --device cpu \
  --resume build/memory-checkpoint.json --output build/memory-resumed.json
```

El checkpoint incluye cursor, filas, prioridades, RNG, predicciones pendientes, etiquetas aún no entregadas, control común, acumuladores y mediciones. El código que calcula consultas y prioridades no recibe la verdad del evaluador. La recuperación construye y valida otro estado antes de sustituir el vigente. Rechaza versiones incompatibles, configuración distinta, registros duplicados, maduraciones incoherentes, bancos posteriores al cursor, dimensiones erróneas y archivos truncados.

El archivo también identifica fuentes, compilador, biblioteca estándar y LibTorch. Las salidas se escriben en un archivo provisional exclusivo, se sincronizan y se publican mediante renombrado. Se sincroniza el directorio y se conserva el checkpoint anterior si falla antes del renombrado. La publicación no sustituye el archivo de configuración ni confunde el informe con el checkpoint. No se conserva un archivo por paso.

## Límites y costes medidos

El generador produce una observación cada vez. No materializa el historial. La capacidad está entre 1 y 1.024, el recorrido no supera un millón de pasos y el retraso no supera 4.096 unidades. El recorrido debe superar capacidad más retraso. El estado crece con la capacidad y el horizonte pendiente, no con todos los pasos. Las lecturas JSON están limitadas a 32 MiB y 16 niveles de anidamiento.

El informe distingue tiempo del motor, consultas y escrituras por política, entrada procesada, candidatos examinados, ocupación, expulsiones y bytes de escritura. Los bytes son contabilidad de las estructuras, no tráfico de RAM medido por contadores del hardware. Para capacidad 1.024, cada banco reserva 1.114.112 bytes de registros, prioridades y caché FP64 de claves. Esto no incluye el runtime, contenedores, RNG, temporales de consulta o serialización. El pico RSS del proceso proporciona una medida independiente.

Los percentiles son el límite superior de intervalos de un histograma fijo de 64 posiciones, en potencias de dos de nanosegundos. No son percentiles exactos. Incluyen generación, consultas, feedback y escrituras de un paso, pero no la descarga final de pendientes ni el guardado del archivo. Ambos costes sí forman parte del tiempo total correspondiente. Las asignaciones y transferencias CUDA son cero en esta configuración CPU. Energía y coste monetario no se han medido.

Las ejecuciones del 5 de octubre de 2026 usaron Clang 21.1.8, C++20, LibTorch 2.14.0+cu130 con enlace CUDA desactivado, Linux x86_64 y AMD Ryzen 9 8945HS de ocho núcleos y dieciséis hilos. Se descartó una ejecución completa por combinación y se repitió tres veces en procesos independientes, con las mismas semillas. El arranque del proceso se midió con `/usr/bin/time`, separado del motor. Había actividad de desarrollo concurrente y no se detuvieron otras aplicaciones. La dispersión registrada impide atribuir las diferencias entre escenarios únicamente a su contenido.

| Escenario | Pasos / capacidad | Motor, mediana y rango (s) | Proceso, mediana (s) | RSS máximo (MiB) |
| --- | --- | --- | --- | --- |
| Regreso de contexto | 256 / 16 | 0.0055 (0.0033 a 0.0065) | 0.31 | 113.66 |
| Regreso de contexto | 8192 / 1024 | 1.2249 (1.1954 a 1.4109) | 1.66 | 121.93 |
| Regreso de contexto | 32768 / 1024 | 4.9405 (4.9088 a 4.9686) | 5.41 | 122.10 |
| Cambio persistente | 256 / 16 | 0.0058 (0.0046 a 0.0060) | 0.31 | 113.77 |
| Cambio persistente | 8192 / 1024 | 1.1920 (1.1835 a 1.2697) | 1.65 | 122.07 |
| Sin señal | 256 / 16 | 0.0032 (0.0032 a 0.0048) | 0.20 | 113.69 |
| Sin señal | 8192 / 1024 | 0.5393 (0.5191 a 0.5480) | 0.83 | 123.26 |
| Atípicos | 256 / 16 | 0.0033 (0.0032 a 0.0033) | 0.19 | 113.93 |
| Atípicos | 8192 / 1024 | 0.5214 (0.5125 a 0.5591) | 0.82 | 123.26 |

En los recorridos de 8.192 pasos, cada política confirmó 8.108 escrituras, retuvo 1.024 entradas y contabilizó 8.821.504 bytes de escritura. Se procesaron 2.162.688 bytes de entrada y hubo como máximo cinco predicciones pendientes. El recorrido de 32.768 pasos produjo 32.431 escrituras por banco con la misma capacidad. Su RSS máximo fue 122,10 MiB, frente a 121,93 MiB en la recurrencia de 8.192 pasos. Estas dos medidas no sustituyen una cota de todo el runtime.

| Escenario de 8.192 pasos | MAE uniforme | MAE reciente | MAE selectivo | MAE control exponencial |
| --- | --- | --- | --- | --- |
| Regreso de contexto | 0.238225 | 0.238921 | 0.249271 | 0.226435 |
| Cambio persistente | 0.547938 | 0.426668 | 0.721829 | 0.226435 |
| Sin señal | 0.235652 | 0.235733 | 0.248906 | 0.217850 |
| Atípicos | 0.708865 | 0.714033 | 1.183080 | 0.537177 |

La selectiva no mejora estos controles. Con atípicos conserva información perjudicial y alcanza MAE 1,183080, frente a 0,708865 del uniforme. En el control sin señal tampoco mejora al control exponencial. Son resultados de estas formas y semillas, sin una búsqueda para favorecer una política. Las repeticiones de tiempo no son muestras estadísticas independientes del proceso generador.

La interrupción en el paso 6.001 y la recuperación de la configuración de 8.192 pasos reprodujeron exactamente predicciones, métricas, pendientes y contenido retenido. El checkpoint ocupó 5.628.337 bytes y su publicación tardó 0,149 s. Los procesos parcial y reanudado tardaron 0,79 y 0,71 s, con picos de RSS de 132,45 y 138,29 MiB. La fase de recuperación incluye lectura y validación JSON antes de continuar.

## Verificación y límites de la evidencia

El [registro reproducible](../../reports/resources/memory-stress-20261005.json) contiene configuración, versiones, huellas, errores por fase y las tres repeticiones de cada medida. Las comprobaciones ejecutadas incluyen:

- Pruebas del banco y de la CLI en Debug, Release y ASan/UBSan con detección de fugas. Cubren presupuesto, empate de vecinos, cosenos de referencia, semillas, corte anterior a la maduración, corrupción, errores de publicación y recuperación interrumpida.
- Clang Static Analyzer 21.1.8 y clang-tidy 21.1.6 sin diagnósticos propios pendientes. Se mantienen avisos como errores, análisis experimental de lifetime y `_GLIBCXX_ASSERTIONS`. Las anotaciones locales de las llamadas POSIX y `ru_maxrss` identifican las interfaces del sistema que no tienen una sustitución C++ equivalente aquí.
- 500 ejecuciones de libFuzzer con ASan/UBSan, semilla 212 y un checkpoint válido como corpus inicial. No se observó un fallo. Ese número de ejecuciones no acredita cobertura exhaustiva del parser.
- Tres mutaciones detectadas: retirar el filtro de maduración, expulsar la prioridad mayor y reiniciar el RNG de claves al recuperar.

La cobertura LLVM 21.1.8 de fuentes propias fue 960/1.015 líneas (94,58 %) y 355/480 ramas (73,96 %), excluyendo pruebas y dependencias. Para CRAP se utilizó Lizard 1.24.0, complejidad ciclomática C y fracción de líneas con contador de `llvm-cov show` cubiertas dentro de cada función: `C² × (1 − cobertura)³ + C`. El máximo fue 37 en `Experiment::restore`. La restauración y el parser concentran las condiciones. La cobertura de líneas no prueba todas sus combinaciones de error.

Los perfiles se preparan en directorios distintos, conservando los argumentos de configuración anteriores. ASan/UBSan añade `-DCMAKE_BUILD_TYPE=Debug -DMARS_TITAN_SANITIZER=address-undefined`. Cobertura añade `-DMARS_TITAN_ENABLE_COVERAGE=ON` y genera el resumen mediante `memory-stress-coverage-report` después de ejecutar CTest. El análisis estático habilita `-DMARS_TITAN_ENABLE_STATIC_ANALYZER=ON` y expone `memory-stress-static-analysis`. `-DMARS_TITAN_ENABLE_CLANG_TIDY=ON` integra sus comprobaciones en la compilación.

El fuzzing añade `-DMARS_TITAN_MEMORY_STRESS_FUZZ=ON` al perfil ASan/UBSan. El objetivo `memory_stress_fuzz` recibe un directorio de corpus con un checkpoint generado por la CLI. El recorrido ejecutado usó `-seed=212 -runs=500 -max_len=131072 -timeout=10 -rss_limit_mb=2048`.

LibTorch es una dependencia precompilada y no quedó instrumentada por estas compilaciones. MSan y TSan requieren runtimes compatibles instrumentados y no se ejecutaron. No se midieron energía, transferencias reales de RAM ni costes monetarios. La memoria de este control es local a un proceso y la recuperación se hace en un corte confirmado, sin transacción distribuida ni entrenamiento de un modelo.
