# Controles nativos de adaptación de una matriz

`AdapterControl` compara tres formas de ajustar una proyección lineal FP64. El problema técnico es `Y = X Wᵀ`, con entrenamiento y validación generados por separado. No utiliza series financieras, políticas de negociación ni un modelo candidato.

El ajuste completo modifica una copia de W. La corrección residual calcula `X Wᵀ + X Dᵀ`, con D inicialmente cero y W congelada. El bajo rango calcula `X Wᵀ + (X Vᵀ) Uᵀ`, con U inicialmente cero y V obtenido de un generador CPU propio. La base no requiere gradientes y no forma parte del optimizador en los controles residual y de bajo rango. El código comprueba esa condición antes de actualizar.

Los tres controles usan autograd, productos matriciales y SGD de LibTorch. El ejecutable fija tasa 0,05 y momentum 0,5, sin weight decay, dampening ni Nesterov. Son opciones del control técnico, sin selección empírica de una configuración superior. Cada variante ejecuta los mismos lotes y pasos. No hay parada independiente ni ajuste del presupuesto según el resultado.

La selección calcula MSE sobre la misma matriz de validación antes y después de cada actualización. El padre continuo original es una alternativa explícita en el paso cero. Solo sustituye al mejor estado una reducción mayor que `minimum_improvement`, que vale cero en el ejecutable. `predict(inputs, true)` devuelve el estado seleccionado y reproduce exactamente la salida original del padre si este sigue siendo el mejor. `predict(inputs)` utiliza el último estado, que se conserva con su momentum para continuar.

## Dependencias semánticas

Las dimensiones iguales no acreditan que dos representaciones sean compatibles. `SemanticVersions` identifica vista, representación, claves, consulta y salida. Para las predicciones, el contador de salida se acompaña de `output_state`, una huella SHA-256 del contenido. Dos restauraciones pueden llegar al mismo paso con pesos distintos. El contador por sí solo no acredita compatibilidad. `invalidated` y `require_compatible` comparan también la huella y rechazan una predicción si falta en cualquiera de los dos estados. La memoria y las lecturas no requieren una huella de salida.

| Cambio | Representaciones | Memoria con claves | Lecturas | Predicciones |
| --- | --- | --- | --- | --- |
| Vista o representación | Invalidar | Invalidar | Invalidar | Invalidar |
| Claves | Conservar | Invalidar | Invalidar | Invalidar |
| Consulta | Conservar | Conservar | Invalidar | Invalidar |
| Contador o contenido de salida | Conservar | Conservar | Conservar | Invalidar |

Este ejecutable adapta solo la proyección de salida. Cada paso confirmado avanza su versión, sin cambiar vista, representación, claves ni consulta. La versión seleccionada corresponde al paso del mejor estado, que puede ser cero. Su huella se calcula con los parámetros seleccionados, no con los últimos pesos. El contrato no crea una memoria candidata ni reconstruye automáticamente sus datos. Un consumidor de memoria o cachés debe comprobar compatibilidad antes de reutilizar el artefacto y reconstruir lo que se haya invalidado.

OpenSSL Crypto calcula SHA-256 sobre el tipo de ajuste, el problema, las formas, las versiones iniciales, la base y los parámetros de la ruta solicitada. Los valores FP64 se codifican en orden little endian. La ruta que devuelve el padre original tiene una marca propia y usa solo su base. `versions()` calcula cada huella cuando se solicita y la guarda. Una actualización confirmada invalida la del estado actual y, si mejora la selección, la del mejor estado. Restaurar descarta ambas y las reconstruye desde los valores recuperados al volver a pedirlas. `predict()` no calcula huellas ni transfiere pesos para validarlas. El control tiene un solo propietario.

## Recuperación y límites

El snapshot guarda configuración, identidades del problema y la validación, versiones iniciales, base congelada, matrices de validación, parámetros actuales, momentum, parámetros seleccionados, pasos confirmados, paso seleccionado y su MSE. El RNG solo interviene al inicializar V. Sus valores iniciales, su semilla y los parámetros posteriores permiten recuperar sin consumir un stream global ni requerir un estado aleatorio durante los pasos deterministas del control.

`restore` verifica el problema, la vista, todos los valores del padre y la validación, las formas, la finitud, los contadores y la correspondencia de la métrica con el mejor estado. Prepara otro estado antes de sustituir el activo. Un gradiente, parámetro, momentum o error de validación no finito impide confirmar la actualización. Si el fallo aparece después de ejecutar SGD, se recuperan los parámetros y el momentum previos.

La serialización usa la versión 2 del formato binario, con enteros y valores IEEE 754 de 64 bits en orden little endian. Conserva también la huella inicial si se ha declarado como dependencia. Las huellas derivadas del estado actual y del seleccionado se reconstruyen desde los tensores, sin aceptar una huella cacheada del estado previo. La versión 1 del archivo se rechaza. El lector comprueba dimensiones y presupuesto antes de reservar cada tensor, tiene un máximo de 128 MiB y rechaza datos sobrantes. El archivo conserva matrices CPU y puede restaurarlas en el dispositivo explícito del consumidor. Publicar de forma atómica el archivo junto con un cursor externo sigue siendo responsabilidad del ejecutor que lo use.

Las dimensiones tienen un máximo de 2.048, los lotes de 65.536 filas y el presupuesto declarado de 512 MiB. El cálculo previo contempla copias para recuperación, parámetros, gradientes, momentum, selección y temporales. No incluye bibliotecas cargadas, memoria interna del backend ni el asignador del sistema. `state_tensor_bytes()` cuenta los valores persistentes que aparecen en el snapshot, sin metadatos, gradientes temporales ni buffers de biblioteca. El ejecutable limita además los datos preparados a 64 MiB y utiliza un hilo intraop y uno interop.

## Compilación y ejecución

El SDK procede del PyTorch ya instalado en el entorno uv del proyecto. El control reutiliza OpenSSL Crypto, que ya utiliza el núcleo nativo, para calcular SHA-256. No se descarga otro LibTorch ni se compila un kernel CUDA. El grupo `MARS_TITAN_BUILD_LEARNING_CONTROLS` permite preparar el replay sin LibTorch. El control matricial añade `MARS_TITAN_BUILD_ADAPTER_CONTROLS`.

```sh
cmake -S native -B build/learning-release -G Ninja \
  -DCMAKE_C_COMPILER=clang-21 -DCMAKE_CXX_COMPILER=clang++-21 \
  -DCMAKE_BUILD_TYPE=Release -DMARS_TITAN_BUILD_SIMULATION=OFF \
  -DMARS_TITAN_BUILD_LEARNING_CONTROLS=ON \
  -DMARS_TITAN_BUILD_ADAPTER_CONTROLS=ON \
  -DMARS_TITAN_LIBTORCH_ENABLE_CUDA=ON \
  -DMARS_TITAN_WARNINGS_AS_ERRORS=ON
cmake --build build/learning-release --parallel 1
ctest --test-dir build/learning-release --output-on-failure
build/learning-release/mars-titan-adapter-control cpu 128 64 32 4 16 7 71
```

Los argumentos son dispositivo, filas, entradas, salidas, rango, pasos, repeticiones y semilla. Cada ensayo solicita la identidad después de cada paso, reconstruye un checkpoint a mitad de la ejecución y verifica las identidades actual y seleccionada al recuperarlo. Después compara el resultado numérico con una ejecución CPU ininterrumpida. La salida JSON separa preparación, entrenamiento, recuperación y tiempo completo del ensayo. El entrenamiento medido incluye validación, comprobaciones de finitud y cálculo de la huella después de cada cambio. En CUDA también incluye las copias a CPU necesarias para esa huella. Las consultas posteriores del mismo estado utilizan la huella guardada. Hay un calentamiento por variante. Las ejecuciones de referencia también consumen recursos y quedan incluidas en el tiempo externo del proceso.

CTest registra por defecto solo las pruebas CPU. `MARS_TITAN_TEST_ADAPTER_CUDA=ON` añade las pruebas de actualización, recuperación y CLI CUDA, que necesitan una ventana exclusiva. También pueden ejecutarse directamente. No hay cambio automático a CPU si se solicita una GPU no disponible.

```sh
nvidia-smi
PYTHONPATH=src uv run --no-sync python -c 'import torch; print(torch.cuda.is_available())'
build/learning-release/adapter_control_tests cuda:0
build/learning-release/mars-titan-adapter-control cuda:0 128 64 32 4 16 7 71
```

La ruta CUDA sincroniza las mediciones de tiempo y consulta los picos de bytes asignados y reservados del asignador de PyTorch. No se han medido volumen real de transferencias, energía ni coste monetario. El registro mantiene esos límites explícitos. La inicialización del asignador se comprueba antes de consultar sus contadores, también al ejecutar el CLI desde un proceso nuevo.

Las pruebas verifican dos pasos analíticos de SGD con momentum para completo y residual, cuatro pasos de bajo rango con gradientes calculados por bucles escalares, identidad inicial, base congelada, selección del padre, recuperación, versiones incompatibles, NaN, infinitos, límites y reversión tras un desbordamiento. ASan y UBSan instrumentan el código propio. Los SDK de LibTorch y OpenSSL ya compilados no quedan instrumentados por esa configuración. El perfil de adaptadores no admite TSan ni MSan con este SDK. Clang Static Analyzer y Lifetime Safety complementan esas comprobaciones.

## Medición inicial del 5 de octubre de 2026

Las medidas de este apartado corresponden a la versión inicial, anterior a la comprobación de identidad por contenido. Se ejecutaron formas de 16 × 8 → 4, 128 × 64 → 32 y 512 × 256 → 128, con rangos 2, 4 y 8. Todas usaron 16 pasos, semilla 71, un calentamiento y siete repeticiones. El equipo fue un Ryzen 9 8945HS con Clang 21.1.8 y LibTorch 2.14.0+cu130. Los procesos tenían habilitado el enlace CUDA del SDK, aunque estas medidas utilizaron CPU. No se detuvieron otros procesos.

En la forma intermedia, el MSE inicial del padre fue 0,252674. El tiempo de ensayo incluye preparación, entrenamiento, validación, checkpoint y recuperación.

| Ajuste | Parámetros entrenables | Bytes del estado tensorial | MSE seleccionado | Mediana del ensayo |
| --- | ---: | ---: | ---: | ---: |
| Completo | 2048 | 163840 | 0.212137 | 4.610 ms |
| Residual | 2048 | 163840 | 0.212137 | 5.623 ms |
| Bajo rango | 384 | 123904 | 0.250933 | 5.075 ms |

El proceso de la forma intermedia, con todas las variantes, referencias y repeticiones, duró 0.63 s y alcanzó 507672 KiB de RSS según GNU time. El RSS incluye el runtime y no equivale a los bytes de tensores del cuadro. El error máximo entre cada ejecución recuperada y su referencia CPU fue cero en los tres tamaños. Las tolerancias de comprobación para comparar dispositivos son 1e-10 relativa y 1e-11 absoluta.

El bajo rango redujo parámetros entrenables, pero dejó un MSE mayor en este presupuesto. No se interpreta como sustituto numéricamente equivalente al ajuste completo ni como una mejora validada para datos financieros.

Pasaron las seis pruebas CTest en Release y con ASan y UBSan. Clang Static Analyzer no emitió diagnósticos. El lector completó 1.000 entradas de libFuzzer desde un checkpoint válido. Las cuatro mutaciones dirigidas que alteraban versión de salida, selección del padre, congelación y recuperación del momentum fueron detectadas. La cobertura LLVM de la biblioteca fue 98,74 % de líneas y 82,40 % de ramas. Lizard 1.24.0 registró CCN máxima 24 en la validación de configuración. CRAP tuvo máximo 24 usando `CCN² * (1 - cobertura)³ + CCN` y líneas ejecutables LLVM dentro de cada función. Son diagnósticos de cobertura y complejidad, no garantías de corrección. `clang-tidy` no estaba instalado. La distribución 21.1.8 solicitada mediante uvx no estaba disponible.

El [registro de medidas y comprobaciones](../../reports/engineering/adapter-controls-20261005.json) conserva los tres tamaños, dispersión, tiempo separado de recuperación, límites y huellas de los archivos comprobados.

El caso intermedio también pasó las pruebas de actualización y recuperación en `cuda:0`, una NVIDIA GeForce RTX 4070 Laptop GPU con 8.188 MiB y controlador 595.91.07. El CLI, ejecutado en una ventana GPU exclusiva, obtuvo medianas de 11,100 ms para completo, 12,010 ms para residual y 15,326 ms para bajo rango. Fueron mayores que las medianas CPU de esta carga. La diferencia máxima frente a la referencia CPU fue 2,22 × 10⁻¹⁵. Los picos asignados por el backend fueron 17.744.384 bytes para completo y residual y 17.614.848 para bajo rango. El pico reservado fue 23.068.672 bytes en los tres casos. Estos contadores pertenecen al asignador de PyTorch y no representan toda la memoria del contexto CUDA. El proceso duró 1,16 s y alcanzó 781.944 KiB de RSS.

## Comprobación de la identidad de salida

La comprobación por contenido reproduce dos continuaciones que parten del mismo snapshot y alcanzan el mismo contador con pesos distintos. También cubre clases de adaptación distintas, dos mejores estados en el paso 1, una selección anterior frente al estado actual y las consultas sin huella. La recuperación conserva la identidad del contenido guardado y no hereda la caché del estado que sustituye. Las huellas distinguen diferencias en los bits FP64, incluidas las que quedan dentro de la tolerancia numérica usada al comparar dispositivos.

En el mismo caso de 128 × 64 → 32, rango 4 y 16 pasos, el CLI solicitó identidad tras cada actualización y comprobó ambas huellas al recuperar. Las medianas CPU fueron 5,790 ms para completo, 6,286 ms para residual y 5,671 ms para bajo rango. Las medianas CUDA fueron 12,361, 12,799 y 15,806 ms. Este trabajo adicional está incluido en los tiempos. La diferencia numérica máxima respecto a la referencia CPU se mantuvo en 2,22 × 10⁻¹⁵. Las medidas anteriores se conservan identificadas y no se atribuye una aceleración a este cambio.

Pasaron las seis pruebas CPU con ASan y UBSan, las pruebas CUDA y el CLI CUDA. El formato 2 completó 1.000 entradas de fuzz desde dos archivos válidos. Las cuatro mutaciones que omitían la comparación de contenido, aceptaban huellas ausentes o conservaban las cachés actual y seleccionada tras cambiar sus pesos fueron detectadas. La biblioteca alcanzó 98,41 % de líneas y 83,10 % de ramas con LLVM. La complejidad máxima y CRAP se mantuvieron en 24 con la convención anterior. El [registro de identidades y coste](../../reports/engineering/adapter-output-identities-20261005.json) conserva el caso anterior, las nuevas medidas y las huellas del código comprobado.
