# Recoger alternativas del puente regional

El colector ya admite `shared_evidence_bridge_v1` y publica sus alternativas mediante la sesión financiera nativa. Cada episodio conserva una sola identidad y sus datos. Las regiones aplican sus propios pesos y transformaciones sobre esa evidencia común. Los prefijos de las rutas se comparten dentro de una llamada y no crean bancos regionales persistentes.

`regional_routes.collect_routes` calcula las seis rutas con nueve refinamientos. Con episodios necesita tres búsquedas y nueve proyecciones regionales. Ejecutarlas por separado necesita 17 refinamientos, nueve búsquedas y 17 proyecciones. La búsqueda conserva o renueva posiciones según la ruta. Cada nodo vuelve a proyectar los valores. Si nueve refinamientos no caben en la cuota estimada, el colector ejecuta cada ruta de forma individual y contabiliza su trabajo completo. La API del lector anterior conserva su recorrido.

`MechanismCatalog.restore` exige la identidad completa y rechaza cambios de semántica, campos ausentes y sustituciones de tipos. El registro regional comprueba que las proyecciones correspondan a cada refinamiento sobre un banco poblado. `MechanismFitConfig.catalog_id` vincula las fuentes con el catálogo declarado. El vínculo se comprueba antes de abrir ventanas y en los paquetes recuperados. Omitirlo mantiene la identidad histórica del protocolo y solo admite el lector anterior. El código del ajuste propaga el vínculo a las dos construcciones de política, pero esas instrucciones protegidas todavía no tienen una ejecución positiva.

## Comprobaciones

La [regresión seleccionada](verification.json) pasó 237 pruebas, 41 en `cuda:0`, sin fallos, errores ni omisiones. No es toda la suite del proyecto. Incluye las seis rutas, banco vacío y poblado, ausencia de banco, gradientes, RNG, presupuesto, permutaciones, bloques, las 36 alternativas frente a emisiones fijas, supervisión, cortes de recepción y recuperación nativa. La recogida opcional conserva exactamente las emisiones y los tensores rápidos restaurados de su control. Los resultados recibidos más tarde y el cierre posterior no cambian la auditoría del mismo prefijo. Los datos y las etiquetas son fixtures.

La [mutación dirigida](mutations.json) detectó los 12 cambios seleccionados. Las referencias inicial y restaurada de la pasada final aprobaron 28 casos CPU cada una. La primera selección dejó sobrevivir dos mutantes: reutilizar una lectura regional anterior y desconectar el estado de entrada del grafo. Solo comparaba recorridos que compartían el mismo motor interno. Se amplió la selección con las pruebas existentes de lectura secuencial y gradientes, cuya referencia es independiente. Ambos mutantes fallaron entonces. Se conservan las dos pasadas y no se presenta el resultado como mutación global.

La [cobertura y CRAP](quality.json) usa coverage.py 7.16.1, Radon 6.0.1 y `scripts/summarize_python_quality.py`. Los siete módulos modificados registran 682/811 sentencias y 183/218 ramas. `MechanismCatalog.restore`, `MechanismFitConfig.check_catalog`, `_Decision.extend` y `collect_routes` cubren todas sus sentencias medibles y sus ramas en esta selección. Sus CRAP son 4, 5, 9 y 4. El método completo `MechanismCollector.add` registra 59/64 sentencias, 25/30 ramas y CRAP 41,80. El solver `fit_policy` registra solo su guardia de entrada, 1/68 sentencias y 0/8 ramas. Estas cifras identifican comprobaciones pendientes y no acreditan ausencia de errores.

El primer ensayo rechazó las rutas regionales por falta de conexión. Dos fallos posteriores correspondían al fixture de cuota, que declaraba presupuestos distintos para lector y puente. Otros dos comparaban referencias de artefactos entre contratos diferentes. Se corrigieron los fixtures y se compararon los tensores restaurados, sin relajar producción. El intento inicial de comprobar el fallo de restauración del catálogo llegó después de su edición y pasó. No se cuenta como una prueba previa fallida. La mutación posterior comprueba el rechazo de catálogos alterados.

## Medición CUDA

La [medición](benchmark.json) usa la RTX 4070 Laptop de 8.188 MiB, PyTorch 2.14.0+cu130, FP64, dimensión 32, tres regiones y ocho vecinos. El banco tiene 256 episodios ficticios, con claves repetidas e IDs únicos. Cada alternativa recibe dos calentamientos y cinco repeticiones, alternando el orden. Los tiempos incluyen el trabajo del host y una sincronización CUDA antes y después. No se solaparon pruebas o mutaciones propias. Había otras aplicaciones en la GPU, con un 22 % de utilización observado antes y un 18 % después. Esas muestras no describen toda la carga concurrente.

| Recorrido | Rutas individuales, mediana | Prefijos compartidos, mediana | Pico incremental asignado por PyTorch, individual / compartido |
| --- | ---: | ---: | ---: |
| Seis rutas, 2 filas | 28,74 ms | 15,97 ms | 218.112 / 206.848 bytes |
| Seis rutas, 64 filas | 26,05 ms | 13,62 ms | 3.943.424 / 2.586.112 bytes |
| Seis rutas, 256 filas | 27,46 ms | 12,86 ms | 15.645.184 / 10.097.664 bytes |
| Colector completo, 64 filas, dos contextos y 36 alternativas | 510,56 ms | 417,13 ms | 9.050.624 / 8.816.128 bytes |

Las salidas comprobadas coinciden exactamente. El colector incluye preparación, cuantiles, calibración de fixture, transferencia a CPU, validación y codificación del paquete. Sus tiempos oscilaron entre 194,58 y 562,78 ms con prefijos y entre 266,59 y 671,13 ms por separado. Cinco repeticiones con esa dispersión no sostienen una aceleración general ni una conclusión sobre latencias extremas.

La ruta individual `k4cr` de 64 filas también se comparó con su implementación en `f1ac211936ef96fc57655f91d4a4d1f3a766b5cd`. Dio 6,48 ms frente a 7,19 ms de mediana y el mismo pico incremental de 1.812.480 bytes. Sus rangos se solapan, de 6,29 a 8,15 ms y de 6,33 a 8,48 ms. Esta pasada no identifica una diferencia estable entre ambas versiones.

El pico RAM acumulado del proceso fue 1.501.384.704 bytes. No permite comparar RAM por alternativa. Los picos GPU de la tabla cuentan asignaciones de PyTorch, no toda la VRAM del sistema ni su caché reservada. Para el caso de 256 filas se declaró una cuota estimada de 128 MiB, sin cambiar el valor por defecto. La cuota habitual de 32 MiB rechazó ese caso antes de medirlo. Los otros dos intentos fallidos del arnés detectaron el modo de atención no declarado y el límite de 64 claves del fixture original. Se corrigió el arnés, se conservan sus registros y esos intentos no cuentan como resultados válidos.

La comparación mide el coste de recoger alternativas para desarrollo. No es la latencia de emitir una única predicción ni el coste de una campaña real. No se midieron energía, p95 o p99. No hay nuevos kernels, dependencias ni cambios C++ o CUDA.

El lector protegido de ganancias, la normalización, el solver y la publicación del recibo aprendido siguen sin ejecución positiva. También faltan los contrastes predictivos del controlador regional y su incorporación a una campaña identificada. No se ejecutaron optimizadores externos, entrenamientos, postentrenamientos, RL ni evaluación científica. El test final permanece cerrado. No hay mejora predictiva medida.

Las [huellas y registros externos](external-artifacts.json) permiten contrastar las fuentes, XML, scripts privados de medida, intentos fallidos y salidas. Con el entorno uv CUDA y el enlace nativo del proyecto, las comprobaciones nuevas y su referencia independiente se ejecutan con:

```bash
uv run --no-sync pytest -q tests/memory/test_regional_mechanism_collection.py tests/memory/test_regional_mechanism_fit_window.py tests/models/titans/test_bridge_mechanism_routes.py
```
