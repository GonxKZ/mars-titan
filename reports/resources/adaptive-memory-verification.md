# Memoria, contexto y replay: comprobaciones del 29 de septiembre de 2026

Se comprobaron los módulos C++20 `EpisodicMemory`, `PolicyContext` y `LearningReplay` en CPU. El [registro JSON](adaptive-memory-verification.json) conserva las huellas de las fuentes, los contadores de cobertura, las mutaciones y las medidas. Estas pruebas no evalúan rentabilidad ni el rendimiento del entrenamiento completo.

## Contratos comprobados

La memoria mantiene hasta 1024 registros por banco, con claves y valores FP32 de 64 componentes. El ámbito separa mundo, partición, fold, representación y lane. Las consultas recuperan cuatro vecinos, excluyen registros futuros y resuelven empates por ID. La referencia numérica utiliza `AccurateSum`. El reservorio tiene un RNG privado recuperable. Preparar una escritura conserva un candidato con registro, posición y RNG, sin copiar el banco completo. La consulta provisional incorpora ese candidato para calcular la observación de bootstrap antes de confirmar el paso financiero.

Las siete pruebas de memoria comprueban consultas, fechas de disponibilidad y maduración, normalización, NaN, ámbitos, tokens ajenos o caducados, recuperación del RNG y límites. Incluyen una comprobación de inclusión con 256 semillas. Los archivos se limitan a 2 MiB por banco. Los lectores rechazan el contenedor truncado antes de abrirlo con LibTorch. Esta comprobación evita la fuga de 152 bytes observada en `mz_zip_reader_init` con un archivo incompleto, sin desactivar LeakSanitizer.

El contexto añade ocho campos de feedback, 64 valores recuperados, presencia, similitud máxima y dos probabilidades HMM. Los controles conservan los mismos campos vacíos. `ppo_window` apila 16 observaciones. La proyección es fija, usa ATen en CPU y semilla 1729. Las siete pruebas de contexto cubren las variantes, el acceso temporal a `ContextTape`, las lanes inactivas, la escritura provisional, el reset y la restauración. El reset crea un nuevo identificador de episodio y elimina solo el historial de la lane afectada. El archivo de contexto se limita a 64 MiB.

La auditoría de sensibilidad clona las observaciones y pone a cero los 66 campos de memoria de cada frame. La prueba comprueba que conserva los campos financieros, el feedback, el HMM y el archivo completo del contexto, incluido el RNG. Esta copia se realiza únicamente al solicitar la auditoría en evaluación. No se añade al collector de entrenamiento.

El replay admite FIFO y reservorio uniforme. Double DQN permite hasta 4096 filas con observación siguiente y 128 MiB. Los auxiliares permiten hasta 8192 filas y 32 MiB. La precomprobación del presupuesto precede a la reserva. Las cinco pruebas contrastan FIFO con una tabla explícita, alias de entrada, muestreo con reemplazo, RNG independientes, recuperación y rechazo sin cambios. El muestreo de ATen conserva la ley uniforme del replay Python, sin exigir los mismos índices que NumPy. Los límites de tensores y archivos no son límites del RSS total del proceso.

Una regresión de integración modifica el bloque financiero del contexto y su último frame, conservando su coherencia interna y el cursor. Antes de la corrección, `PpoTrainer::restore` aceptaba esa observación distinta de la cartera confirmada. La prueba falla con ese comportamiento y pasa al comparar exactamente el bloque financiero recuperado con `FinancialBatch::observations()`. El rechazo conserva el estado anterior.

## Instrumentación y alcance

Se utilizó Clang 21.1.8, C++20 y LibTorch 2.14.0+cu130, con avisos como errores, Lifetime Safety experimental, `-fno-fast-math` y `-ffp-contract=off`. Pasan las tres suites con AddressSanitizer, UndefinedBehaviorSanitizer y detección de fugas. Los módulos de memoria y replay pasan también clang-tidy 21.1.6 y el analizador de rutas de Clang. En contexto, esos dos análisis y la cobertura corresponden al núcleo anterior al helper de sensibilidad. El helper y su prueba posterior sí se compilaron con los diagnósticos estrictos y pasaron los sanitizadores.

La cobertura LLVM 21 es de fuentes propias. El JSON identifica la revisión medida de cada archivo para no atribuir los resultados anteriores al helper añadido después.

| Fuente | Líneas ejecutadas | Ramas ejecutadas |
| --- | ---: | ---: |
| `episodic_memory.cpp` | 446/450 (99,11 %) | 179/240 (74,58 %) |
| `policy_context.cpp`, antes del helper | 667/680 (98,09 %) | 281/398 (70,60 %) |
| `learning_replay.cpp` | 302/309 (97,73 %) | 106/138 (76,81 %) |

CRAP utiliza la complejidad ciclomática de lizard 1.24.0 y las regiones LLVM agrupadas por función, con `CCN² × (1 − cobertura)³ + CCN`. El máximo es 39,00 en la restauración del contexto, con CCN 39. La preparación del contexto obtiene 31,08 y la restauración del replay 14,01. Esta convención usa regiones y no debe compararse directamente con informes que utilicen cobertura de líneas.

Las siete mutaciones dirigidas fueron detectadas: eliminar el corte por maduración, perder el RNG recuperado, consultar la memoria anterior durante el bootstrap, reutilizar el identificador de episodio, escribir siempre en la primera posición FIFO, sortear dentro de la capacidad en vez del número de registros vistos y muestrear siempre la primera fila. No constituyen una búsqueda exhaustiva de defectos.

LibTorch y las dependencias precompiladas no quedan instrumentados íntegramente. No se ejecutaron pruebas TSan o MSan específicas de estos tres módulos. La cobertura no incluye kernels de LibTorch ni ejecución CUDA.

Para repetir las suites integradas, desde la raíz del repositorio:

```bash
cmake --preset native-ppo-asan-ubsan -S native
cmake --build build/native/native-ppo-asan-ubsan -j 1 \
  --target episodic_memory_tests policy_context_tests learning_replay_tests
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  ctest --test-dir build/native/native-ppo-asan-ubsan \
    -R '^(episodic_memory|policy_context|learning_replay)$' \
    --output-on-failure -j 1
```

## Consultas de memoria y trabajadores contables

La sonda de consultas compara la GEMV FP64 de ATen con productos escalares `AccurateSum` y selección parcial de vecinos. Utiliza 32 consultas, dos tandas de calentamiento y siete medidas por capacidad, con un hilo CPU. El JSON describe la construcción determinista de las claves. Coinciden los cuatro IDs seleccionados.

| Registros | ATen, mediana por consulta | Referencia, mediana por consulta |
| ---: | ---: | ---: |
| 16 | 5,01 µs | 8,06 µs |
| 128 | 7,18 µs | 71,04 µs |
| 1024 | 28,12 µs | 525,76 µs |

El proceso de esa sonda duró 0,40 s y alcanzó 116516 KiB de RSS. Las cifras son medianas de promedios por tanda. No se conservaron sus muestras individuales ni se midió variación entre procesos. Había otras tareas locales activas.

El benchmark existente de `FinancialBatch` se ejecutó sobre un Ryzen 9 8945HS con 16 activos y 256 sesiones. Cada combinación tiene un proceso de calentamiento y cinco repeticiones para 16 y 128 entornos, o tres para 4096. El binario calienta también un paso antes de medir. Las 64 ejecuciones terminaron en 22,34 s. No había procesos de compilación al comenzar ni cargas GPU durante la medición. Los servicios de fondo siguieron activos.

| Entornos | 1 trabajador | 2 trabajadores | 4 trabajadores | 8 trabajadores |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 393437 | 427612 | 473628 | 299808 |
| 128 | 525161 | 700975 | 955469 | 1069314 |
| 4096 | 723867 | 717704 | 1012492 | 1205806 |

La tabla expresa la mediana de transiciones por segundo. Los checksums coinciden entre todos los trabajadores para cada tamaño. El JSON conserva todas las ejecuciones, la dispersión, p50/p95/p99 del paso, el tiempo del proceso y las reservas de memoria. El RSS observado va de 5,00 a 91,77 MiB. El tiempo de rollout incluye la generación de acciones y el checksum de recompensas y observaciones. Los percentiles del paso excluyen esas dos operaciones.

Se puede repetir cada combinación con el [benchmark](../../native/benchmarks/batch_rollout.cpp) de Release, cambiando `--environments` y `--workers`:

```bash
cmake --preset native-release -S native
cmake --build build/native/native-release -j 1 --target mars-titan-batch-benchmark
build/native/native-release/mars-titan-batch-benchmark \
  --environments 128 --assets 16 --sessions 256 --workers 4
```

Los resultados no justifican aumentar siempre los trabajadores. Con 16 entornos, ocho fueron más lentos que cuatro. Con 128 y 4096, ocho obtuvieron el mayor caudal medido. No se modificó la concurrencia de producción. Estas medidas excluyen inferencia, aprendizaje, transferencias, energía, coste económico y VRAM.
