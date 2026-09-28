# PPO nativo: comprobaciones del 28 de septiembre de 2026

La entrega conecta `FinancialBatch` con una política C++20 y LibTorch. El proceso nativo carga Parquet, recoge decisiones por lotes, calcula GAE, actualiza con Adam, evalúa y guarda checkpoints. El lanzador Python administra la admisión y las señales, sin importar PyTorch. La [guía](../../docs/engineering/native-ppo.md) describe configuración, fuentes y recuperación.

Se utilizó Clang 21.1.8, C++20 y LibTorch 2.14.0+cu130, con avisos estrictos, Lifetime Safety experimental y la semántica numérica del proyecto. Release conserva `-O3`, sin `fast-math`. CUDA quedó enlazado, pero todas las comprobaciones de aprendizaje usaron CPU explícita y hasta 32 transiciones por ejecución. La campaña científica existente siguió activa.

## Correctitud y recuperación

Pasan los diez ejecutables de CTest y las 262 pruebas Python de simulación. Estas incluyen 20 casos del ejecutable PPO y 19 del lanzador. Pasan también las 18 pruebas del supervisor GPU. Las 75 pruebas del ejecutable contable anterior se repitieron con la nueva biblioteca de archivos y cobertura para comprobar la extracción de ese código compartido.

Las pruebas numéricas contrastan el objetivo recortado con valores analíticos, GAE con terminaciones y truncaciones independientes y exclusión de recompensas inválidas. Las comparaciones analíticas FP64 usan tolerancias absoluta y relativa de `1e-10`. Una actualización controlada aumenta la probabilidad de la acción con ventaja positiva. La recuperación CPU reproduce exactamente los siguientes muestreos, pesos, pasos de Adam y estados contables en los casos ensayados. Esto no demuestra utilidad financiera.

La revisión adversarial produjo regresiones permanentes para ocho fallos: bootstrap no finito, crecimiento logarítmico de un patrimonio positivo muy pequeño, modificación de la máscara activa durante la confirmación, retroceso incoherente del presupuesto, actualización extra al recuperar una parada temprana, código de salida incorrecto de una pausa, selección de un mejor checkpoint corrupto y discrepancia entre presupuestos de VRAM del lanzador y el binario. Los rechazos conservan el estado confirmado. Si Adam ha aplicado parte de una actualización que después falla, el entrenador exige recuperar un checkpoint y bloquea el guardado del estado parcial.

ASan, UBSan y detección de fugas pasan las 20 pruebas del ejecutable y las pruebas nativas de política, entrenador y lote. El lector de contexto y el almacén de checkpoints también se comprobaron con esos instrumentos. TSan y MSan pasan el lote contable en compilaciones separadas. LibTorch y Arrow precompilados no quedan instrumentados por estas comprobaciones. TSan y MSan del ejecutable PPO completo requieren dependencias instrumentadas y no se presentan como ejecutados.

Clang-tidy y Clang Static Analyzer no emitieron diagnósticos propios en las fuentes modificadas. Ruff y la comprobación de enlaces, formatos y referencias del repositorio pasan. Una prueba que inspecciona el archivo C++ mediante `torch.jit.load` conserva visible su aviso de obsolescencia.

## Cobertura y pruebas de mutación

El [registro de calidad](native-ppo-quality.json) delimita las fuentes PPO, sus SHA-256 y los contadores de LLVM 21. Se ejecutaron 1989 de 2140 líneas y 888 de 1280 ramas. La cobertura no comprende kernels de LibTorch ni una ejecución CUDA. El resumen global de LLVM emitió 14 discrepancias `hash=0` de dos funciones inline de `CashMovements` en siete objetos. Se identificaron con el volcado del instrumento y no corresponden a las fuentes PPO resumidas.

CRAP se calcula mediante `CCN² × (1 − cobertura de líneas)³ + CCN`, con Lizard 1.17.31 y las líneas ejecutadas de LLVM. El máximo en PPO es 25 y corresponde a la validación de hiperparámetros. Las validaciones de recursos y la lectura de argumentos concentran otras ramas pendientes. Este diagnóstico localiza complejidad, sin sustituir las pruebas de comportamiento.

La revisión final detectó otra restauración incoherente: un contador de episodios distinto de cero permitía recuperar un entorno inicial junto con transiciones confirmadas. La regresión falló antes de añadir la comprobación y pasó después en Release y ASan/UBSan. Un reinicio y su paso siguiente se confirman juntos, por lo que una pausa posterior al comienzo no puede conservar cursor cero.

Seis mutaciones dirigidas del entrenador se compilaron y fueron detectadas. Eliminaban la comprobación de finitud del bootstrap, permitían adelantar el cursor respecto al presupuesto o restaurar un entorno inicial con progreso, omitían la restauración del RNG, sustituían la copia del snapshot por una vista o permitían continuar tras un fallo parcial de Adam. Los hashes de la fuente y los resultados están en el mismo registro. No se presenta esta selección como cobertura exhaustiva de defectos.

Para repetir las comprobaciones principales, desde la raíz y con las dependencias preparadas:

```bash
cmake --preset native-ppo-release -S native
cmake --build build/native/native-ppo-release -j 1
ctest --test-dir build/native/native-ppo-release --output-on-failure -j 1
CUDA_VISIBLE_DEVICES=-1 \
  MARS_TITAN_PPO_EXECUTABLE=build/native/native-ppo-release/mars-titan-ppo \
  MARS_TITAN_SIM_EXECUTABLE=build/native/native-ppo-release/mars-titan-sim \
  uv run pytest tests/simulation tests/training/test_gpu_supervisor.py -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_repository.py
```

Los presets `native-ppo-asan-ubsan`, `native-ppo-static-analysis` y `native-ppo-coverage` separan instrumentación y rendimiento. La cobertura requiere ejecutar también las pruebas del CLI con `MARS_TITAN_PPO_EXECUTABLE` apuntando al binario instrumentado y `LLVM_PROFILE_FILE` dentro de su directorio `coverage`.

## Medición del recorrido completo

El [registro de medición](native-ppo-diagnostic-20260928.json) contiene 60 ejecuciones medidas y doce calentamientos, con 1, 2, 4 y 8 entornos, semillas 42, 43 y 44 y cinco repeticiones por combinación. Cada proceso realiza 32 transiciones globales, dos actualizaciones de rollout, validación y persistencia. Utiliza ocho activos ficticios, dos campos de contexto, un trabajador contable y un hilo de LibTorch sobre el Ryzen 9 8945HS. Las fuentes y la validación son comunes. El modelo padre es una referencia congelada de la prueba, sin entrenamiento científico.

Las medianas por combinación están entre 0,449 y 0,470 segundos de proceso y entre 0,122 y 0,134 segundos internos. El tiempo interno excluye el arranque del proceso y la publicación del último recibo. El mayor pico de RSS observado es 414,7 MiB. Cada salida conserva dos o tres bundles, con 272336 a 407148 bytes de archivos en total. El JSON distingue tamaño de archivo y espacio asignado por el sistema de archivos.

Esta carga pequeña no muestra una mejora consistente al aumentar el lote. Cambiar el número de entornos también modifica el horizonte de cada recorrido y las trayectorias de acciones, aunque conserve las 32 muestras globales y los pasos de Adam. Por eso las cifras no se presentan como aceleración de un cálculo numéricamente equivalente ni se utilizan para cambiar los valores de producción.

La campaña científica y otras tareas locales siguieron activas. No se midieron energía, transferencias, asignaciones ni bytes físicos de E/S. `perf_event_paranoid=4` impidió acceder a los contadores solicitados y se conservó la configuración del sistema. Se descartó una primera tanda porque la espera del medidor añadía retardos por sondeo. El registro publicado usa finalización mediante pipes y conserva las huellas del ejecutable, la biblioteca contable y el instrumento. El rendimiento CUDA y las cargas largas quedan pendientes de una medición con admisión GPU disponible.

El [instrumento reutilizable](../../scripts/benchmark_native_ppo.py) reproduce los ocho SHA-256 de las fuentes del registro. Su huella se identifica por separado de la versión medida. La copia publicada añade comprobaciones de salida y disponibilidad de `perf`, sin modificar las cifras anteriores. Pasan tres pruebas permanentes que comprueban la protección de artefactos y el tratamiento de `perf` ausente. Para repetir el protocolo con directorios nuevos:

```bash
uv run python scripts/benchmark_native_ppo.py \
  --root . \
  --binary build/native/native-ppo-release/mars-titan-ppo \
  --private artifacts/native/ppo-benchmark-work-v1 \
  --output artifacts/native/ppo-benchmark-v1.json
```
