# Política adaptativa: comprobaciones del 29 de septiembre de 2026

Pasan las 46 pruebas nativas de política en CPU tras conservar los validadores optimizados y la preparación de rollouts MLP desde CPU. El [registro JSON](adaptive-policy-verification.json) identifica las fuentes, las pruebas y los contadores de esta ejecución. La fuente comprobada de `ppo_policy.cpp` tiene SHA-256 `b0d54ed761c0fc6cdabdfc3330e7164364e92555f35edad5484b663f4dab3217`.

La validación omite la reducción de finitud cuando un tensor está vacío, después de comprobar su definición, forma, precisión y dispositivo. El caso habitual es el estado recurrente vacío de la MLP. En Double DQN, las recompensas y los dos conjuntos de valores Q se concatenan para comprobar su finitud con una reducción y una lectura escalar. Se conserva la comprobación posterior de los objetivos calculados.

La actualización MLP termina antes de calcular GAE cuando no hay recompensas válidas. La validación completa del rollout precede a esa salida. La prueba compara seis estadísticas nulas con un control sin etiquetas y comprueba igualdad exacta del actor, el crítico y los dos RNG. También exige rechazar NaN y acciones con forma incompatible aunque la máscara esté vacía.

La entrada `update_from_cpu` admite únicamente PPO MLP. Reutiliza GAE de ATen en FP64 sobre CPU y selecciona las filas válidas antes de trasladar observaciones, acciones, logprobabilidades, ventajas y retornos. La conversión de precisión y la normalización se realizan después del traslado al dispositivo de la política. Comparte el bucle PPO existente y conserva Adam, sus RNG y el formato de checkpoints. `update` mantiene su contrato de dispositivo. La entrada CPU rechaza GRU y Double DQN.

Las pruebas comparan `update` y `update_from_cpu` en tres arquitecturas MLP, con entradas FP32 y FP64, dos actualizaciones sucesivas y rollouts de tres pasos. Coinciden exactamente seis métricas, los seis tensores de parámetros PPO, los pasos de Adam y ambos RNG. Los casos incluyen máscaras parciales, terminaciones y truncaciones. Se rechazan datos no finitos en filas sin etiqueta válida y esquemas incompatibles antes de modificar el estado.

Las pruebas de los validadores comprueban objetivos analíticos exactos en ocho combinaciones FP32/FP64 con vistas no contiguas, 18 entradas NaN o infinitas distribuidas entre los tres tensores y rechazos de definición, forma, tipo y dispositivo. La prueba de la MLP compara exactamente logits y valores con estado vacío explícito y rechaza esquemas vacíos incompatibles. Los casos de dispositivo utilizan tensores Meta y no ejecutan CUDA.

La suite incluye también objetivo recortado, GAE, estados recurrentes, límites de episodio, consolidación auxiliar y recuperación del siguiente paso de Adam. El recuento de 46 corresponde a las funciones de prueba. Algunas recorren varias configuraciones.

## Diagnósticos y cobertura

Se utilizó Clang 21.1.8, C++20 y LibTorch 2.14.0+cu130. La compilación conserva los avisos como errores, Lifetime Safety experimental, `-fno-fast-math` y `-ffp-contract=off`. Pasan AddressSanitizer, UndefinedBehaviorSanitizer y detección de fugas con las 46 pruebas. Clang-tidy 21.1.6 analiza fuente y pruebas con la configuración del repositorio. El analizador de rutas de Clang comprueba la fuente de producción. Ambos terminan sin diagnósticos propios.

LLVM 21 registra 1023 de 1041 líneas ejecutadas en `ppo_policy.cpp` y las dos líneas instrumentadas de su cabecera. El total es 1025/1043 líneas (98,27 %) y 433/564 ramas (76,77 %). Se ejecutaron completas las líneas de los validadores modificados, `double_dqn_targets`, `update_from_cpu` y `update_feedforward`. La cobertura corresponde a una compilación CPU y excluye el código CUDA.

CRAP utiliza lizard 1.17.31 y las líneas LCOV de cada función, con la fórmula `CCN² × (1 − cobertura de líneas)³ + CCN`. El máximo es 25 y corresponde a la validación de hiperparámetros y a la carga de la política. Los resultados por función están en el JSON. Esta convención difiere del informe de memoria y contexto, que utiliza regiones LLVM.

Dos mutaciones de la revisión final, compiladas por separado, fueron detectadas. Una aceptaba un rollout sin etiquetas antes de validarlo. La otra delegaba la entrada CPU a `update` y admitía GRU. En ambos casos la suite falló porque se aceptaba una entrada inválida. El JSON conserva por separado las dos mutaciones anteriores de los validadores, junto con su revisión de 44 pruebas. Las fuentes de producción no se modificaron para ejecutar estos casos.

LibTorch precompilado no queda instrumentado íntegramente. Estas pruebas no aportan cobertura CUDA ni una medida del rendimiento del entrenamiento completo. La cobertura y las dos mutaciones comprueban recorridos concretos, sin demostrar ausencia de defectos.

Para repetir la suite con sanitizadores y backend CPU:

```bash
cmake --preset native-ppo-asan-ubsan -S native \
  -DMARS_TITAN_LIBTORCH_ENABLE_CUDA=OFF
cmake --build build/native/native-ppo-asan-ubsan -j 1 --target ppo_policy_tests
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  ctest --test-dir build/native/native-ppo-asan-ubsan \
    -R '^ppo_policy$' --output-on-failure -j 1
```
