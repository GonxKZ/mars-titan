# KLPO terminal con episodios completos

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

## Registro, recogida y pérdida diferenciable

[`KlpoTerminalCollector`](../../native/include/mars_titan/klpo_collection.hpp)
reutiliza `FinancialBatch`, `PolicyContext`, `PpoPolicy` y `PpoCheckpointStore`.
Recoge una oleada de hasta 128 episodios, con un máximo declarado de 256 pasos
por episodio. Rechaza horizontes mayores y cierres desde 2024. No corta episodios
para ajustarlos al límite ni sustituye carriles terminados por otros episodios.
La primera ruta admite MLP o GRU y su contexto PPO correspondiente. HMM,
memoria episódica, auxiliares y Double DQN quedan fuera de este contrato.

El [registro versionado](../../native/include/mars_titan/klpo_episodes.hpp)
conserva la fuente, el contexto, el calendario completo, el fold, el cursor,
las observaciones FP32, la acción, los seis pesos FP32 realmente muestreados,
la recompensa y las causas de cierre. El sampler, sus parámetros y precisión,
beta y gamma permanecen fijos durante la oleada. La huella de parámetros excluye
RNG, gradientes y direcciones de almacenamiento. La identidad incluye por
separado la semilla, el orden de los sorteos, las fuentes y la compilación.

Los pasos forzados actualizan la historia GRU sin sortear una acción. Conservan
la recompensa válida del motor y su ordinal original en `sum_t gamma^t r_t`.
La elegibilidad para sortear y la validez de la valoración son campos distintos.
Un episodio formado solo por pasos forzados conserva su identidad, retorno y
puesto en el denominador de la oleada, con contribución nula al gradiente.
Si no hay ninguna decisión, el resultado declara `no_policy_decisions` y no
llama al núcleo con un lote vacío.

El horizonte declarado y la ruina son cierres consumibles. Una valoración
inválida conserva la evidencia y bloquea la oleada completa. No se retira ese
episodio ni se usa su cero centinela como retorno. Un corte de recursos guarda
un prefijo recuperable que todavía no puede producir la pérdida terminal.

`PpoPolicy::terminal_forward` reconstruye la historia desde estado inicial
cero y conserva el grafo de los pasos anteriores. No usa el hidden guardado
como sustituto ni desacopla el prefijo cada 16 pasos. El consumidor reúne las
decisiones sorteadas para el núcleo y devuelve la media sobre todos los episodios
completos. Solo utiliza los logits del actor. La cabeza de valor no interviene
en esta pérdida, aunque comparte el tensor de salida con las seis acciones.
La separación de momentos de un futuro optimizador no se acredita aquí.

La codificación binaria comprueba la geometría y su presupuesto antes de
materializar observaciones. El registro admite hasta 128 MiB, mientras que el
colector limita su parte a 64 MiB para compartir el archivo de recuperación con
el contexto. El consumidor comprueba por separado el padding y las copias de
la historia antes de reservarlas. Estos límites describen almacenamiento lógico
propio, no el RSS, el grafo completo ni las bibliotecas cargadas.

La recuperación usa el escritor atómico existente y su identidad. Conserva
carteras, contexto, hidden, registros y RNG. Construye un candidato independiente
y reproduce el prefijo con el sampler fijo antes de sustituir el estado en uso.
Contrasta acciones, probabilidades, recompensas, observaciones y metadatos.
Un fallo previo al commit permite repetir el paso con el RNG anterior. Un fallo
posterior exige recuperar un checkpoint confirmado. La recuperación no publica
otro estado durante esa comprobación.

El modo técnico parte de una referencia inicial identificada por semilla y pesos.
La construcción optativa admite una copia de pesos y el RNG inicial de la oleada.
Su recuperación conserva ambos en vez de reconstruirlos solo desde la semilla.
El [controlador de actualizaciones](terminal-klpo-updates.md) añade un actor y
Adam separados, con cadencia y publicación propias. La recuperación después de
un paso real y los resultados financieros siguen sin comprobarse. La ruta PPO
anterior no usa estos métodos optativos. El bloqueo histórico sigue vigente.

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

## Verificación del colector

La [evidencia del registro y colector](../../reports/engineering/terminal-klpo-collection-verification-20261009.json)
separa la entrega del núcleo de la integración posterior. Pasan siete ejecutables
CPU en Release, ASan/UBSan y cobertura. Las doce mutaciones iniciales y las dos
de la guarda de precisión se detectan. El codec supera 5.000 ejecuciones de
libFuzzer con ASan/UBSan sobre cuerpos acotados. Esto no acredita todas las
entradas posibles ni una campaña de aprendizaje.

LLVM cubre 299/299 líneas del registro y 572/586 del colector. El CCN máximo
de las funciones modificadas es 23 y el CRAP máximo 23,0051, con regiones de
código LLVM como convención. El recibo conserva las coberturas de los archivos
PPO compartidos, cuyas rutas de actualización no se han ejecutado.

Los fixtures CUDA de MLP y GRU conservan la paridad dentro de `rtol=1e-5` y
`atol=1e-6`, el gradiente del prefijo GRU, la recuperación exacta y el RNG global.
El máximo error de logits fue 3,32×10⁻⁷ y el de gradientes 2,05×10⁻⁸. El pico
Torch fue de 76.160.000 bytes asignados y 90.177.536 reservados, con una cuota
de 128 MiB que no incluye el contexto CUDA. El primer intento detectó que la
consulta genérica de TF32 no admitía flags distintos de conv/RNN. La identidad
registra ahora ambos operadores y rechaza cambios durante la oleada.

La comprobación CPU reproducible evita las pruebas anteriores que actualizan
parámetros:

```bash
CUDA_VISIBLE_DEVICES=-1 UV_OFFLINE=1 cmake --preset native-ppo-release -S native \
  -B build/native/terminal-collector -DMARS_TITAN_BUILD_RUNNER=OFF \
  -DMARS_TITAN_LIBTORCH_ENABLE_CUDA=OFF
cmake --build build/native/terminal-collector --target klpo_episodes_tests \
  klpo_collection_tests klpo_policy_tests klpo_terminal_tests \
  ppo_variant_state_tests ppo_variant_objective_tests ppo_checkpoint_tests -j 1
CUDA_VISIBLE_DEVICES=-1 ctest --test-dir build/native/terminal-collector \
  -R '^(klpo_episodes|klpo_collection|klpo_policy|klpo_terminal|ppo_variant_state|ppo_variant_objective|ppo_checkpoint)$' \
  --output-on-failure
```

La revisión independiente del colector no encontró hallazgos materiales abiertos.
Ocho sondas CPU contrastan horizontes mixtos, q e historia alteradas, RNG
falsificado, gradiente GRU y fila del crítico, valoración inválida en un episodio
forzado, interrupción tardía, presupuesto y ruina del motor. Su recibo distingue
esas ejecuciones de las comprobaciones CUDA y sanitizadores anteriores.

La conciliación con `develop` `b79ab99c` recompila el perfil y pasa los mismos
siete ejecutables CPU en 1,98 segundos. El contrato episódico anterior sigue en
v1. Los archivos de cálculo de política, objetivo y motor financiero permanecen
idénticos, por lo que no se repite CUDA. El recibo separa esta compatibilidad de
la revisión y de las comprobaciones anteriores.
