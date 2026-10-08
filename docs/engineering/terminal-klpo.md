# Núcleo terminal Full-KL de seis acciones

[`klpo_terminal_full_loss`](../../native/include/mars_titan/klpo_terminal.hpp)
implementa `klpo_terminal_token_full_v1` con primitivas ATen. Recibe logp y logq
`[B,T,6]`, acciones int64 y máscara bool `[B,T]`, retornos terminales `[B]` y beta
positiva. Devuelve un sustituto FP64 por trayectoria. Solo logp conserva gradiente.
La función suma decisiones, sin dividir por longitud ni promediar trayectorias.

En cada decisión calcula `ell=logp(a)-logq(a)` y
`z=logp(a)-sum(q*logp)`. El resultado es la suma de
`-stopgrad(R-beta*ell)*z`. Es el sustituto token Full-KL de
[arXiv v1, ecuaciones 4.9–4.10](https://arxiv.org/abs/2610.08963v1).
Su valor no es la varianza poblacional ni una métrica financiera. La
[selección de candidatos](../references/rl-algorithm-selection.md) explica por
qué se conserva aparte del ajuste predictivo y de PPO.

La implementación es propia a partir de esas ecuaciones y reutiliza ATen y las
convenciones de validación del proyecto. No incorpora código de la biblioteca
KLPO, que declara Apache 2.0. Se cita el
[pin 304e5ac7](https://github.com/yifanzhang-pro/KLPO/blob/304e5ac7ca573d45a0e42e6eaea95203260402bc/klpo/loss.py)
como referencia del sustituto. La identidad predictiva anterior permanece intacta.

## Tipos, máscara y presupuesto

Se admiten FP32 y FP64 en un mismo dispositivo CPU o CUDA, con B entre 1 y 128
y T entre 1 y 256. Las probabilidades activas necesitan soporte positivo
representable en la precisión de entrada. No basta con que logq sea finito.
Las filas se comprueban con tolerancia de normalización 10⁻⁶ y se renormalizan
en FP64. No se añade un piso ni masa exploratoria a posteriori.

Cada máscara contiene al menos una decisión seguida de padding, sin huecos.
El padding se neutraliza antes de exponenciales y gather. NaN/Inf y acciones
centinela en padding no intervienen en valores ni gradientes. Las entradas
activas inválidas rechazan la llamada completa.

La estimación previa es `65536 + B*T*(6*192+128) + B*128` bytes. Cuenta bytes lógicos
de entrada, promociones, temporales y buffers propios previstos de autograd, con margen por
elemento. El máximo permitido es 64 MiB y puede pedirse un presupuesto menor.
El máximo de forma estima 42.024.960 bytes. No incluye el grafo previo del actor,
el allocator, las bibliotecas ni el contexto del dispositivo y no es una cota
del RSS ni del almacenamiento ajeno a las vistas de entrada. Superar la forma
no autoriza cortar una trayectoria en varios episodios.

La media externa conserva el número total de trayectorias completas. Un lote
dividido en microbatches suma sus resultados con el mismo denominador global.
La pérdida no calcula descuento, GAE ni bootstrap. Si el retorno fijado es
`sum_t gamma^t r_t`, el retorno restante equivalente necesita el factor `gamma^u`.
El tiempo financiero original no se sustituye por el índice tras retirar pasos
forzados de la máscara.

## Registro y consumidor pendientes

El núcleo no recibe una bandera que pueda acreditar por sí sola completitud.
El registro externo deberá verificar las siguientes correspondencias antes de
llamar a la función:

| Registro requerido | Componente existente | Diferencia pendiente |
| --- | --- | --- |
| Observaciones, acciones, reinicios y prefijos | `PpoRollout` y `PpoPolicy` | Recalcular p desde la historia completa bajo el actor actual. No reutilizar el hidden histórico como sustituto. |
| Las seis probabilidades históricas y versión del sampler | `PpoAction.probabilities` | Sellarlas al recoger, junto con sus transformaciones, precisión e identidad. El logq de la acción elegida no permite reconstruirlas. |
| Retorno completo y causa terminal | `FinancialSession::StepOutcome` | Separar horizonte fijado, ruina, falta de valoración y corte por recursos. `truncated` por sí solo no basta. |
| Sampler fijo hasta terminar el episodio | Actor y RNG nativos | El colector PPO actual puede actualizar al llenar su buffer. No concatenar sus fragmentos como una trayectoria de q única. |
| Recuperación | Archivos y checkpoints nativos | Añadir registro parcial, cursor, acumulador del retorno, sampler/RNG y comprobación de continuidad sin repetir decisiones. |

Los pasos forzados no se presentan como acciones sorteadas. Si falta una
valoración según la cartera elegida, descartar solo ese resultado introduciría
selección. Se necesita un dominio de ejecución admisible y común, sin completar
el retorno con ceros. La ruina conserva su penalización contractual. Las fuentes,
folds, horizonte, escala de recompensa y estado inicial pertenecen a la identidad.

El consumidor mínimo puede recoger episodios completos bajo un sampler
congelado y publicar otros pesos al terminar el lote. No necesita un entrenador
nuevo ni concurrencia de versiones para empezar. Esa ruta, sus actualizaciones
y la recuperación del actor no están implementadas aquí. Las comprobaciones del
núcleo no habilitan aprendizaje ni una simulación histórica.

## Comprobación local sin aprendizaje

```bash
CUDA_VISIBLE_DEVICES=-1 UV_OFFLINE=1 cmake --preset native-release -S native \
  -B build/native/terminal-klpo -DMARS_TITAN_BUILD_SIMULATION=OFF \
  -DMARS_TITAN_BUILD_FINANCIAL=OFF -DMARS_TITAN_BUILD_RUNNER=OFF \
  -DMARS_TITAN_BUILD_PPO=OFF -DMARS_TITAN_BUILD_RL_OBJECTIVES=ON \
  -DMARS_TITAN_LIBTORCH_ENABLE_CUDA=OFF
cmake --build build/native/terminal-klpo --target klpo_terminal_tests -j 1
ctest --test-dir build/native/terminal-klpo -R '^klpo_terminal$' --output-on-failure
uv run --no-sync --offline python native/tests/klpo_terminal_oracle.py \
  --output /tmp/terminal-klpo-oracle.json
```

Los perfiles `native-asan-ubsan`, `native-static-analysis` y `native-coverage`
usan las mismas opciones con salidas separadas. Las pruebas solo evalúan
constantes y gradientes. El [oráculo portátil](../../reports/engineering/terminal-klpo-oracle-20261008.json)
conserva cuatro puntos de dos parámetros y contraejemplos de recompensa,
longitud, descuento y selección de resultados. No se ejecuta `optimizer.step`
ni ajuste de parámetros.

La [verificación del núcleo](../../reports/engineering/terminal-klpo-verification-20261008.json)
incluye Release, ASan/UBSan, análisis estático y cobertura, con ocho mutaciones
detectadas. LLVM cubrió 71/71 líneas, 73/74 regiones y 59/84 ramas. El CCN máximo
de Lizard fue 34 y el CRAP máximo 34,004, usando regiones de código LLVM. La
complejidad incluye las guardas de forma, tipo y presupuesto. No es una prueba
de ausencia de defectos.

Con dos hilos CPU y cinco repeticiones, el máximo `[128,256,6]` tuvo una mediana
de 6,55 ms en FP32 y 6,37 ms en FP64 para forward y gradiente. El profiler registró
picos de almacenamiento de tensores de 11.567.616 y 13.140.992 bytes, incluidos
los inputs del fixture y sin grafo previo del actor. La estimación previa es
42.024.960 bytes. El RSS máximo del proceso fue 694.400 KiB. Estas cifras no
describen el coste de un actor, de una trayectoria financiera ni de un entrenador.

Cuatro fixtures CUDA contrastan valores y gradientes FP32/FP64 en `[3,4,6]` y
`[128,256,6]`, con padding y q/R desacoplados. Pasan las tolerancias fijadas y
los rechazos de tres incompatibilidades. El pico fue de 13.140.992 bytes
asignados y 27.262.976 reservados, con límite de allocator de 128 MiB en `cuda:0`.
Esos contadores excluyen el contexto CUDA. No se midió aceleración ni se ejecutó
un actor. El registro de una diferencia emitió un aviso de conversión a escalar
con gradiente y el profiler CPU emitió una consulta fallida de dispositivos CUDA
al estar desactivados. Ambos diagnósticos se conservan en el recibo.

La revisión independiente de la implementación añadió nueve casos CPU. Una
referencia escalar y el gradiente cerrado del score centrado contrastaron dos
precisiones y tres valores de beta, con longitudes distintas. También rechazó
soporte perdido, retornos no finitos y máscaras con huecos. El recibo técnico
separa estas pruebas de los cuatro fixtures CUDA y de los perfiles nativos.
