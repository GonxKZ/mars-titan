# Selección de mecanismos con publicación recuperable

`FrozenMechanismConsumer` admite una política congelada opcional que elige una de las 36 combinaciones de contexto, ruta episódica y calibración por mercado. `MechanismBinding` extrae la observación desde las filas admitidas, antes de MAC y del feedback del evento actual. La elección, la emisión y los estados auxiliares se confirman en la misma generación de `FinancialSession`. El código de esta integración está en el commit `7ca32e9ab`.

La opción sigue desactivada por defecto. `policy=None` conserva los controles fijos y el formato sin campo `selection`. No se han ajustado los coeficientes del selector ni ejecutado entrenamientos, postentrenamientos, RL o evaluaciones predictivas en este incremento. El bloqueo de la campaña completa y el test final no ha cambiado. No hay una mejora predictiva medida por este componente.

## Decisión y estados

La política recibe los 21 rasgos originales del [extractor](../mechanism-observations-20261021/README.md). Calcula ganancias puntuales para doce combinaciones de contexto y ruta, compartidas entre sus tres calibraciones, y ganancias de intervalos para las 36 alternativas. Combina ambas con el peso, coste y margen declarados. El soporte comprueba rangos marginales. Fuera de ellos elige `off/k1/none`. No es un detector OOD calibrado.

El consumidor resuelve las precondiciones y el presupuesto de todas las alternativas antes del argmax. Una alternativa con motivos de retorno queda excluida. Las reservas compartidas entre mercados se tratan de forma conservadora y no dependen de cuál se procese primero. El contexto elegido se aplica antes de MAC. La ruta se ejecuta después de K1 y la calibración al emitir. Las propuestas no avanzan MAC más de una vez por observación ni publican intentos descartados.

Se exige una cabeza de cinco cuantiles y una política `MechanismPolicy` en `eval`, con buffers congelados y el mismo dispositivo que el predictor. La precisión puede diferir si se declara. La acción explícita y la selección automática son modos excluyentes. La identidad comprueba configuración, buffers, métodos, fuentes, dispositivo y precisión. Declara `supplied_not_verified_as_fitted` y `new_learned_selector=False`. Suministrar coeficientes no acredita haber aprendido su utilidad.

La traza se guarda una vez por evento con una lista por mercado. Conserva la observación FP64 sellada, el vector tras la conversión realmente usada, el vector normalizado, la disponibilidad, las ganancias, las puntuaciones y la acción. Se concilia con las identidades de las filas, parámetros, contexto y snapshot. La verificación recalcula las utilidades congeladas y las compara con las acciones ejecutadas y emitidas. Las puntuaciones no se repiten por activo.

Calentamiento y settlement no seleccionan. El calentamiento automático usa contexto desactivado. La primera decisión empieza sin contexto confirmado y puede escribirlo para una decisión posterior. El feedback actual se procesa después de emitir. Los valores de etiquetas pendientes nunca son rasgos del selector. `latest.json` sigue siendo la única publicación y la recuperación requiere el consumidor identificado. Los hashes detectan alteraciones e incompatibilidades, sin autenticar un host que pueda cambiar arbitrariamente código y archivos.

Este catálogo no genera datos ni directivas de ajuste como [SEAL, versión 2](https://arxiv.org/abs/2506.10943v2), cuyo bucle exterior utiliza el rendimiento posterior a la actualización como recompensa. Tampoco convierte la fusión regional en memoria histórica distribuida. Faltan el colector de alternativas anteriores a la etiqueta, el ajuste temporal protegido, la selección del padre, sus controles científicos y esa ampliación de memoria. Las variantes GRU, Transformer, Titans-MAC y CM-v1 conservan sus identidades y contratos propios.

## Comprobaciones

La [selección final](verification.json) pasó 355 casos, 80 marcados CUDA, sin fallos, errores ni omisiones. Incluye 55 casos nuevos, once CUDA. No es toda la suite del proyecto. Comprueba selección previa a MAC, mercados separados, máscara antes del argmax, acción fija y automática K4 equivalentes, conversión FP32 y normalización efectivamente consumida, cambios de política, corrupción de observación y utilidades, partición, orden de activos y rechazo de recuperación con un selector incompatible.

Los cortes nativos cubren escritura del registro, escritura del checkpoint, instante anterior a publicar e instante posterior a publicar, tanto en CPU como en CUDA. Se recuperan la siguiente elección, predicción, contexto y actualización de calibración. La prueba comprueba que la observación de una emisión aún ve las filas pendientes anteriores, aunque sus resultados lleguen en ese evento. La recuperación no los aplica dos veces.

Las [doce mutaciones dirigidas](mutations.json) se detectaron. Incluyen retirar la máscara, omitir la selección, cambiar el contexto o la ruta ejecutada, archivar otro vector normalizado, omitir la conciliación de la traza y aceptar otro productor. Las referencias inicial y restaurada pasaron 44 casos CPU cada una. Es el resultado sobre esos fallos elegidos, no una cobertura universal de mutaciones posibles.

La [cobertura y CRAP](quality.json) usa coverage.py 7.16.1 con ramas y Radon 6.0.1 en cinco módulos, incluidos los componentes existentes modificados. Registra 700/759 sentencias y 248/308 ramas, en 48 funciones. CRAP usa `C² × (1 − cobertura de sentencias)³ + C`. El máximo es 55,92 en `FrozenMechanismConsumer.begin`, con complejidad 54. `check_trace` tiene complejidad 36 y CRAP 36,08. Quedan rechazos sin recorrer y complejidad que revisar. Estas puntuaciones no prueban ausencia de errores.

Los fallos técnicos anteriores se conservan en el recibo. La cobertura seleccionada por nombres de módulos terminó una vez con código 139 y después falló al importar `torch._C` con `UnicodeDecodeError`. La selección por rutas pasó un diagnóstico de tres casos y la pasada final. Otra pasada terminó con 353 casos aprobados y dos fallos en una expectativa de identidad. El contrato incluye el bloque físico, por lo que deben variar `context_id` y la huella que lo contiene. La prueba final exige que solo cambien esos campos, con los mismos rasgos, utilidades y acciones y salidas dentro de la tolerancia declarada. No se retiró ninguna comprobación de producción.

## Coste integrado

El [protocolo](cost-protocol.json) se guardó antes de medir. El [benchmark](benchmark.py) reutiliza el recorrido de sesión existente. Compara K1 fijo con la misma referencia elegida por una política de ganancias cero, sin ajuste. Ambos mantienen presentes PT4 y PT2, sin activar contexto ni calibración y sin lector episódico. La medida incluye apertura, calentamiento de memoria, tres emisiones, feedback manual, cierre, escritura, verificación, poda, lectura de trazas y recuperación.

Se ejecutaron veinte trayectorias completas, con una de calentamiento y cuatro medidas por control y tamaño. El orden de controles alterna en cada repetición. Usó `cuda:0`, RTX 4070 Laptop, FP64, dimensión oculta 32, bloques de 64, capacidad de banco 64 y dos hilos CPU. Las entradas son fixtures US con precios `[64,5]`, noticias de anchura 2, gráficos de anchura 1 y fundamentales y macro de anchura 3, con máscaras y precios replicados. No representan el corpus histórico completo ni incluyen codificadores externos. No hubo otras cargas propias de pruebas, mutación o perfilado durante el benchmark. Las aplicaciones del escritorio permanecieron activas.

| Activos | Control | Mediana total (ms) | p95 total (ms) | Predicciones/s sobre tiempo total |
| --- | --- | ---: | ---: | ---: |
| 64 | K1 fijo | 3729,16 | 3928,06 | 50,95 |
| 64 | Selector que elige K1 | 3465,07 | 3559,43 | 55,13 |
| 256 | K1 fijo | 9626,68 | 10311,96 | 78,48 |
| 256 | Selector que elige K1 | 9940,67 | 10528,07 | 76,83 |

Cada total corresponde a la trayectoria completa, no a una petición de un activo ni a un kernel. Los [datos brutos](cost.json) conservan tiempos por etapa, dispersión y p99 empírico. Con cuatro repeticiones esos percentiles no estiman con precisión las colas. La [comparación](comparison.json) registra una diferencia de mediana de −264,09 ms con 64 activos y +313,99 ms con 256. No se atribuye una aceleración al selector por la primera cifra ni se infiere un coste universal de la segunda. La funcionalidad añade trabajo y necesita perfilado del recorrido representativo antes de optimizarla o promoverla.

Las predicciones conservan exactamente el mismo SHA-256 entre controles y repeticiones. Los archivos retenidos añaden 24.960 bytes por trayectoria en ambos tamaños. Los picos del asignador Torch fueron iguales entre controles, 78.303.744 bytes con 64 activos y 80.699.392 con 256. Incluyen ambos consumidores residentes y no son la VRAM completa del dispositivo. No demuestran coste nulo de memoria del selector. El pico RSS medido dentro de Python fue 1.499.287.552 bytes. El máximo externo de `/usr/bin/time` fue 1.598.124.032 bytes y el tiempo externo fue 146,40 segundos, frente a 142,58 segundos del tramo interno. No se midió energía ni se atribuye ahorro de RAM.

Para reproducir la selección técnica en un entorno compatible:

```bash
PYTHONPATH=src:. uv run --no-sync pytest -q tests/memory/test_mechanism_selection.py tests/models/titans/test_mechanism_consumer.py tests/memory/test_financial_mechanisms.py tests/memory/test_mechanism_observations.py tests/models/titans/test_mechanism_policy.py tests/models/titans/test_mechanism_catalog.py tests/models/titans/test_mechanism_routes.py tests/models/titans/test_mechanism_maturity.py
```

La medida exige CUDA y el enlace nativo configurado, una carpeta de almacenamiento existente y un archivo de salida nuevo:

```bash
PYTHONPATH=src:. uv run --no-sync python reports/engineering/mechanism-selection-20261021/benchmark.py --output /tmp/mars-mechanism-selection-cost.json --storage-root /tmp --rows 64 256 --repetitions 4 --warmup 1
```

Las [huellas de los artefactos externos](external-artifacts.json) identifican logs, JUnit, cobertura, receta de mutación, medidas y diagnóstico conservados fuera de Git. El informe acredita integración técnica y coste observado. El ajuste del selector y su utilidad predictiva siguen pendientes en [#701](https://github.com/GonxKZ/mars-titan/issues/701).
