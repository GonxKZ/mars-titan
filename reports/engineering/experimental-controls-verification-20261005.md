# Verificación conjunta de los controles experimentales

El 5 de octubre se comprobaron juntos los controles de memoria, replay, adaptadores y cohortes, los lectores de documentos y vistas y las correcciones operativas de supervisión y recibos. El [registro verificable](experimental-controls-verification-20261005.json) conserva las huellas de las fuentes modificadas y de las salidas de pruebas. La [guía de uso](../../docs/engineering/experimental-controls.md) enlaza los contratos y las medidas de cada componente.

La suite Python completa pasó 2.746 pruebas en 315,57 segundos, con 19 omisiones y ocho avisos. Ocho omisiones necesitan `hmmlearn`, ausente en ese entorno. Las otras once requieren activar expresamente una ventana CUDA. Se comprobaron por separado las rutas afectadas: supervisor con procesos reales, codificadores congelados, recuperación de PPO episódico y adaptadores.

Tras resolver los diagnósticos del código C++ se repitieron las comprobaciones nativas. Los tiempos siguientes corresponden a las pruebas y no son una comparación de rendimiento de modelos.

| Perfil conjunto | Resultado CTest | Tiempo |
| --- | --- | ---: |
| Release | 24 aprobadas | 8,72 s |
| ASan y UBSan | 24 aprobadas | 38,56 s |
| Análisis estático y Debug | 24 aprobadas | 13,28 s |
| Cobertura LLVM | 24 aprobadas | 13,48 s |

La compilación completa con clang-tidy 21.1.6 y el análisis de rutas de Clang terminaron con código cero. Se conservaron C++20, avisos como errores, Lifetime Safety experimental y las opciones de precisión del proyecto. También se comprobó la configuración de cobertura con `BUILD_TESTING=OFF`.

Los ajustes de diagnóstico nombran límites y tolerancias, usan rangos acotados y conservan los bytes de los archivos de recuperación mediante `std::bit_cast`. No cambian semillas, presupuestos ni formatos. Las excepciones documentadas en las pruebas se limitan a sus datos y resultados numéricos explícitos y a una conversión intencional de un byte inválido para comprobar su rechazo. Los [diagnósticos de ensanchamiento](https://clang.llvm.org/extra/clang-tidy/checks/bugprone/implicit-widening-of-multiplication-result.html) y [acceso a arrays](https://clang.llvm.org/extra/clang-tidy/checks/cppcoreguidelines/pro-bounds-constant-array-index.html) siguen activos.

La recuperación CUDA de PPO pasó con las semillas 42, 43 y 44, conservando pesos, momentos de Adam, trazas y selección. Los tests y la CLI final de adaptadores pasaron en `cuda:0`. En el control pequeño de ocho filas y cuatro pasos, el error máximo frente a la referencia fue 2,22e-16. Esta comprobación complementa las medidas anteriores de tamaños y costes, no las sustituye.

Los perfiles de cobertura se regeneraron después de compilar. LLVM emitió 26 avisos que el diagnóstico `--dump` reduce a dos entradas `main` sin recorrido en CTest y a instancias sin uso de `CashMovements::add` y `CashMovements::reconciled` con hash cero. Por ello el porcentaje agregado no se utiliza como criterio de aceptación. Las medidas de cobertura, complejidad, CRAP y mutación por componente conservan sus convenciones y límites en los informes enlazados.

LibTorch y las otras dependencias precompiladas no quedan instrumentadas por los sanitizadores de los objetivos propios. Esta comprobación no incluye MSan, TSan ni Windows. Las investigaciones de compactación de GRU y de asignaciones al cerrar LibTorch siguen en #145 y #202.

La campaña comparativa se pausó para las comprobaciones CUDA y se reanudó desde su estado confirmado. Conserva la revisión científica fijada, mientras los módulos de supervisión y lectura de recibos se despliegan por separado. Los controles de esta entrega no implementan ni entrenan MARS-TITAN y no abren el test final.
