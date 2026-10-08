# KL categórica y controles de PPO

[`ppo_categorical_kl`](../../native/include/mars_titan/ppo_objectives.hpp) calcula
por fila la suma completa `D_KL(q_histórica || p_actual)` sobre las seis acciones
financieras existentes. Es una función pura de ATen. No crea una política, un
crítico, un optimizador o un entorno. No cambia `PpoPolicy`, su pérdida recortada
ni el formato vigente de `PpoRollout`.

El contrato `categorical_behavior_to_current_kl_v1` admite dos matrices `[B,6]`
FP32 o FP64 en el mismo dispositivo CPU o CUDA. Devuelve un vector FP64. El
gradiente solo pasa por la política actual. Las filas pueden tener strides
distintos y su orden no introduce interacción entre observaciones.

Los logaritmos deben estar normalizados con tolerancia de 10⁻⁶. Después se
renormalizan en FP64 para absorber ese redondeo. `-inf` representa masa cero.
Una acción con masa histórica positiva y masa actual cero se rechaza. También
se rechaza un logaritmo histórico finito cuya exponencial pierda soporte por
subdesbordamiento. No se añade un piso ni se inventa masa de exploración.

Se admiten entre una y 65.536 filas por llamada. La geometría se valida antes de
convertir o calcular los buffers. La cota permite procesar un rollout por
bloques y no limita el tamaño del censo. El cálculo es lineal en el número de
filas, con seis columnas fijas. No se materializa un Jacobiano ni una matriz
de relaciones entre muestras. La precisión de entrada y el redondeo siguen
limitando la interpretación de una divergencia cercana a cero.

## Compilación y comprobación local

El objetivo separado `mars_titan_rl_objectives` reutiliza el SDK LibTorch del
entorno uv existente. La siguiente comprobación solo compila y ejecuta las
pruebas del objetivo puro:

```bash
CUDA_VISIBLE_DEVICES=-1 UV_OFFLINE=1 cmake --preset native-release -S native \
  -B build/native/rl-objectives \
  -DMARS_TITAN_BUILD_SIMULATION=OFF -DMARS_TITAN_BUILD_RUNNER=OFF \
  -DMARS_TITAN_BUILD_PPO=OFF -DMARS_TITAN_BUILD_RL_OBJECTIVES=ON \
  -DMARS_TITAN_LIBTORCH_ENABLE_CUDA=OFF
cmake --build build/native/rl-objectives --target ppo_objective_tests -j 1
CUDA_VISIBLE_DEVICES=-1 ctest --test-dir build/native/rl-objectives \
  -R '^ppo_objectives$' --output-on-failure -j 1
```

Los perfiles `native-asan-ubsan`, `native-static-analysis` y `native-coverage`
aceptan las mismas opciones con directorios de salida distintos. No se ejecutan
las suites antiguas que contienen actualizaciones de parámetros. La
[prueba nueva](../../native/tests/ppo_objective_tests.cpp) usa distribuciones
literales, gradientes, diferencias finitas y casos degenerados sin aprendizaje.

Pasaron las comprobaciones CPU de Debug, Release, ASan/UBSan y análisis estático
con Clang 21.1.8, C++20 y LibTorch 2.14.0. clang-tidy 21.1.6 y el analizador de
rutas no dejaron diagnósticos propios abiertos. Se corrigió la guarda de
inclusión que señaló el análisis de portabilidad. Las bibliotecas precompiladas
de LibTorch no están instrumentadas por los sanitizadores del proyecto.

Cinco mutaciones sobre dirección KL, gradiente histórico, soporte, anchura y
límite de filas fueron detectadas. LLVM cubrió 44/44 líneas, 32/33 regiones y
25/34 ramas del archivo de producción. Lizard 1.24.1 dio CCN máximo 15. El CRAP
máximo fue 15,009 usando cobertura de regiones LLVM por función. Esta convención
no equivale a la cobertura de sentencias utilizada en Python.

Una medida con un hilo, un calentamiento y cinco repeticiones obtuvo medianas
de 0,044, 0,078 y 14,235 ms en FP32 para 1, 256 y 65.536 filas. En FP64 fueron
0,039, 0,073 y 9,778 ms. La mayor dispersión correspondió al caso grande FP32,
entre 9,124 y 18,084 ms. El proceso alcanzó 149.296 KiB de RSS, con bibliotecas e
inputs incluidos. Son costes del cálculo puro en una máquina no aislada, no
una comparación de algoritmos ni una medida del entrenador.

Una compilación separada con backend CUDA comprobó dos fixtures `[1,6]` en
FP32 y FP64 sobre la RTX 4070 Laptop. El mayor error absoluto frente a CPU
fue 6,94×10⁻¹⁸ en el valor y 7,45×10⁻⁹ en el gradiente. También pasaron el
soporte degenerado, seis rechazos de tipo o dispositivo y la conservación de
entradas y RNG. Se usó PyTorch 2.14.0+cu130, con una fracción máxima de allocator
de 0,008, menor que 64 MiB en ese dispositivo. El pico fue 8.192 bytes asignados
y 2 MiB reservados, sin incluir el contexto. El primer arnés omitía la
inicialización del dispositivo y falló antes del objetivo. Se conservó ese
fallo y se corrigió el arnés sin cambiar producción. No se midió rendimiento
ni se ejecutó ninguna actualización de parámetros.

## Integración y comprobaciones pendientes

La pérdida pura no completa por sí sola una variante de PPO. La
[integración explícita](ppo-objective-variants.md) añade captura, controlador y
recuperación, con las comprobaciones de optimización todavía pendientes.
Se mantienen el
[PPO nativo](native-ppo.md) y los
[controles KLPO predictivos](../references/klpo-quadratic.md) como problemas
distintos. El recorte de PPO no impone una cota dura de KL. La penalización
adaptativa y la parada por KL son alternativas conocidas, con reglas que deben
quedar identificadas. [PPO, secciones 3–5](https://arxiv.org/pdf/1707.06347v2),
[Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

| Variante | Cambio del objetivo | Estado adicional que debe recuperarse |
| --- | --- | --- |
| PPO-Clip con diagnóstico completo | Conservar el objetivo actual y medir la KL de las seis acciones. | Distribución histórica completa, versión del muestreador, máscara de filas válidas e identidad de observaciones. |
| PPO con penalización KL adaptativa | Sustituir el término del actor por `-mean(r*A - beta*KL)`, conservando crítico y entropía declarados. | Beta vigente y sus límites, KL objetivo, regla de adaptación, contador de actualizaciones y medición pendiente. |
| PPO-Clip con parada por KL | Medir el desplazamiento real de la política y detener las épocas restantes según una regla previa. | Última KL, épocas completadas y omitidas, umbral superado y resumen del rollout confirmado. |

Cada variante tiene un identificador distinto de objetivo y controlador,
además del contrato de esta función. Los campos se conservan en el checkpoint
existente y su validación, sin introducir un segundo escritor. El bloque
`policy_objective` activa las identidades y su ausencia conserva el camino anterior.

`PpoAction` expone probabilidades completas. El rollout legacy guarda solo el
logaritmo de la acción elegida. La extensión conserva las seis probabilidades
del muestreador real, su normalización y versión. No se reconstruyen
con los pesos nuevos. La integración conserva los seis pesos FP32
originales y los normaliza en FP64 al calcular la KL. Para 1.024 decisiones
son 24.576 bytes de contenido, aparte de metadatos y temporales. Las filas sin
recompensa válida se excluyen según la máscara existente.

`approximate_kl` utiliza las salidas calculadas antes de `optimizer.step()`.
No debe reinterpretarse como medición posterior. El controlador nuevo necesita
evaluar la política resultante sobre las mismas observaciones y, para GRU,
los mismos prefijos y reinicios. En la red compartida, una actualización del
crítico también puede cambiar el cuerpo de la política. Parar solo la cabeza
del actor no impide ese desplazamiento.

La recuperación conserva la frontera existente al terminar `advance()`. No se
publica un checkpoint entre el paso y su medición. Un fallo invalida el estado
en curso y exige cargar el último bundle confirmado, sin contar dos veces una
actualización confirmada. La parada por KL dentro del
rollout es distinta de la selección temporal de checkpoints. Una comparación
con igualdad de actualizaciones requiere una regla conjunta o presupuesto fijo.
Si se permiten paradas independientes, se registrará la diferencia efectiva.

Las variantes de [#137](https://github.com/GonxKZ/mars-titan/issues/137)
compartirán observaciones, acciones, costes, particiones y límites de recursos.
Se conservarán por separado transiciones, exposiciones y actualizaciones.
Siguen pendientes las pruebas de actualización y recuperación efectiva y los
experimentos. El aprendizaje permanece bloqueado hasta preparar y verificar
la edición histórica desde 2000, y la simulación financiera conserva sus
requisitos de OHLC y acciones corporativas. El test final sigue cerrado.

El [recibo de comprobación](../../reports/engineering/rl-objectives-verification-20261008.json)
reúne las versiones, huellas del código, pruebas, revisión independiente y
límites de estas primitivas. La revisión añadió seis casos Python y 24
contrastes con aritmética Decimal, además de sondas nativas en Release y
ASan/UBSan. No ejecutó optimizadores ni repitió las comprobaciones CUDA.
