# Controles experimentales y preparación de MARS-TITAN

Estas piezas permiten comprobar datos, memoria, repetición, adaptación y recuperación antes de implementar el candidato. Tienen ejecutables o APIs verificadas y configuraciones propias. Sus controles sintéticos no constituyen una comparación financiera ni un entrenador de MARS-TITAN.

| Componente | Entrada y salida | Evidencia |
| --- | --- | --- |
| [Memoria y ruido](memory-stress.md) | Secuencias con etiquetas diferidas y bancos de capacidad fija. Compara expulsión aleatoria, reciente y selectiva. | Saturación, regreso de contextos, ruido sin señal, perturbaciones y recuperación íntegra. |
| [Índice documental](document-index.md) | Manifiestos de noticias y embeddings compatibles. Conserva documentos y menciones antes de promediar. | Disponibilidad, procedencia, corrupción, cursor y comparación de media frente a selección. |
| [Vistas de información](information-views.md) | Corpus admitido y catálogo acreditado. Retira valores, máscaras, edades y dependencias de todas las rutas declaradas. | Misma población, macro efectiva, procedencia, padre y estados incompatibles rechazados. |
| [Replay](replay-schedules.md) | Referencias a episodios maduros y multiplicidades. Genera calendarios de exposiciones recuperables. | Igual multiconjunto y presupuesto, espaciado, token y cursor confirmado. |
| [Adaptadores](adapter-controls.md) | Problema matricial controlado. Compara ajuste completo, residual y bajo rango con LibTorch. | Congelación, padre seleccionable, recuperación, identidad de contenido y paridad CPU/CUDA. |
| [Cohortes](cohort-execution.md) | Observaciones, tareas y feedback maduro. Registra predicciones y confirma estado local por generación. | Orden temporal, invariancia bajo el contrato del predictor y siete fronteras de interrupción. |

La predicción futura debe usar una vista declarada y un estado compatible. El registro conserva la salida emitida, que sirve para calcular el error cuando llega la etiqueta. El replay intencional y el feedback confirmado una vez tienen identidades distintas. Los controles de cada componente delimitan esos contratos, sin suponer que todos los consumidores nuevos sean modelos ya entrenados.

## Compilación conjunta

Los perfiles conservan C++20, Ninja, el compilador y los diagnósticos del proyecto. Reutilizan LibTorch, Arrow, OpenSSL y las primitivas existentes. No instalan otro runtime de aprendizaje ni cambian el entorno científico.

La [verificación conjunta del 5 de octubre](../../reports/engineering/experimental-controls-verification-20261005.md) recoge la suite Python, los cuatro perfiles nativos, las comprobaciones CUDA y los límites de la instrumentación.

```bash
cmake -S native --preset native-controls-release
cmake --build build/native/native-controls-release --parallel 1
ctest --test-dir build/native/native-controls-release --output-on-failure -LE '^optimizer-steps$'
```

Mientras rija el bloqueo de aprendizaje, `-LE '^optimizer-steps$'` deja fuera las pruebas que aplican pasos de optimizador. `ctest --preset native-controls-release` ya las excluye.

`native-controls-asan-ubsan` prepara instrumentación separada. `native-controls-static-analysis` incluye clang-tidy durante la compilación y el objetivo `static-analysis` para análisis de rutas. `native-controls-coverage` permite obtener el objetivo `coverage-report` después de ejecutar las pruebas. La cobertura conjunta incluye los ejecutables y pruebas de las piezas nuevas, además del núcleo anterior.

Las pruebas CUDA de adaptadores requieren una ventana exclusiva y se activan expresamente con `MARS_TITAN_TEST_ADAPTER_CUDA=ON`. Las pruebas Python de GPU conservan sus controles de activación. Las pruebas CPU no sustituyen esa verificación. El control de memoria y el ejecutor de cohortes de esta entrega tienen rutas CPU explícitas.

El informe LLVM incluye funciones de ejecutables que CTest no recorre. Su diagnóstico `--dump` identifica avisos de perfiles ausentes para los `main` del simulador y PPO, y para instancias sin uso de `CashMovements::add` y `CashMovements::reconciled` con hash cero en otros binarios. No se utiliza el porcentaje agregado como umbral de aceptación. Las comprobaciones por componente conservan sus medidas de líneas, ramas, complejidad y mutaciones, con el alcance explicado en cada informe.

## Qué permiten concluir los controles

La política selectiva de memoria empeoró las cuatro secuencias medidas. En el control con atípicos, su MAE fue 1,183080 frente a 0,708865 de expulsión aleatoria. La prioridad utilizada se basa en error persistente y no incluye toda la propuesta de sorpresa de N1. El resultado desaconseja atribuir utilidad a una política por el mero hecho de priorizar errores grandes.

El adaptador de bajo rango utilizó menos parámetros y obtuvo mayor MSE en el problema y presupuesto medidos. La ruta CUDA también fue más lenta que CPU en el control intermedio. Las cifras de capacidad, calidad, memoria y tiempo se conservan por separado. No se adopta una alternativa únicamente por reducir parámetros o utilizar GPU.

Las vistas actuales mantienen las dimensiones para conservar los contratos de los lectores. Poner posiciones a cero no ahorra por sí mismo lectura o inferencia. La retirada física de columnas, el reajuste de modelos reducidos y su comparación de calidad siguen necesitando una edición experimental propia.

El ejecutor de cohortes confirma estado JSON local. Los callbacks deben conservar determinismo y no depender de la composición física del lote para sostener su invariancia. Los efectos externos, los optimizadores y los tensores no incluidos en ese estado necesitan un contrato adicional de su productor. El caso medido no demuestra atomicidad de un modelo todavía no implementado.

## Supervisión y alcance científico

La [supervisión CUDA](gpu-admission.md) y los [recibos adaptativos](campaign-receipts.md) se despliegan de forma independiente del runtime científico. El seguimiento no carga pesos ni convierte una publicación parcial en una etapa terminada. Los [recuentos del 5 de octubre](../../reports/baselines/campaign-status-20261005.md) separan las ejecuciones acabadas del análisis y de los trabajos posteriores.

El candidato, su entrenamiento y sus ablaciones siguen pendientes de una fase específica. El test final permanece reservado. Los avisos de compactación de GRU se siguen en [#145](https://github.com/GonxKZ/mars-titan/issues/145) y las asignaciones señaladas al cerrar LibTorch en [#202](https://github.com/GonxKZ/mars-titan/issues/202). Esta entrega no declara resueltas esas investigaciones.
