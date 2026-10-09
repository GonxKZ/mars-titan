# GRU episódica en el coordinador financiero

La [referencia GRU](../../native/candidate.md), con su [entrada histórica con
máscaras](../../native/candidate_historical.md), se conecta ahora al mismo
`FinancialSession` que Titans-MAC. Conserva una identidad propia y no sustituye a B.
Esta guía describe el codec CPU fijo, el banco de claves 128 y valores 256, el enlace
con el coordinador y las comprobaciones ejecutadas. No se ha entrenado ningún modelo
ni se ha medido utilidad predictiva.

## Codec CPU fijo

`Candidate::cpu_episode_codec` copia en CPU las proyecciones fijas de rasgos (256) y
de claves (128). El objeto resultante no conserva el modelo ni sus parámetros
aprendidos, de modo que borrar o modificar el propietario no cambia sus salidas.
`encode_context` separa la GRU y la fusión del cálculo episódico. `encode` y `forward`
conservan su resultado, y el archivo estricto que genera `mars-titan-candidate cpu`
mantiene sus 8.031.885 bytes y su SHA-256 `00a99b03…`.

Cada fila se proyecta por separado con la misma operación nativa, para que el
resultado no dependa de cómo se agrupen las filas. La clave se normaliza con
épsilon 1e-12 y debe tener norma uno, con tolerancia 1e-5, o ser exactamente nula.
Una clave subunitaria no nula se rechaza en lugar de convertirse en cero. El
presupuesto de trabajo se comprueba antes de leer las entradas o reservar salidas.

`FrozenCandidateCodec` vincula el codec al contrato de entradas, a la precisión y a
los flags numéricos de CPU. Devuelve arrays de solo lectura y una huella de su
contenido. En una sonda con dos hilos, FP32 y FP64 y anchuras de 333 a 1.686 no
cambió ningún bit al desplazar la fila en memoria, por lo que no se añadió una copia
alineada por fila.

## Banco 128×256

`CandidateEpisodeBank` ([`candidate_bank.py`](../../src/mars_titan/memory/candidate_bank.py))
guarda solo episodios maduros. Las claves `[R,128]` y los valores `[R,256]` mantienen
la precisión del modelo. Los tiempos de decisión, disponibilidad y maduración son
enteros y las etiquetas originales se conservan en FP64. La capacidad admite de 1 a
8.192 plazas. Para contrastar la GRU con M0/M1/M2 de Titans hay que fijar la misma
capacidad, que en esos bancos no supera 1.024.

La admisión reutiliza el reservorio causal v2 del banco nativo. `causal_reservoir_state`
y `causal_reservoir_draws` exponen el mismo `mt19937_64` con `seed_seq` y la misma
elección de plaza que `EpisodicMemory`, que ahora comparte esa función. Con la misma
semilla, capacidad y orden de IDs, las plazas coinciden con las del banco de 64
coordenadas. El ámbito y los hashes de datos no intervienen en los sorteos.

Cada propuesta valida el lote completo antes de sortear y devuelve un banco nuevo.
Los IDs deben crecer y superar al último admitido, la maduración no puede ser
posterior al corte confirmado y el reloj del banco no retrocede. Un rechazo deja
tensores, contadores y RNG intactos. La estimación de trabajo cuenta dos generaciones
protegidas, las entradas, la propuesta y su copia de publicación, con un máximo
configurable de 256 MiB. No es una medida del RSS del proceso.

La vista de lectura ordena por ID sin reordenar las plazas y convierte las etiquetas a
la precisión de lectura después de comprobar que caben. `Candidate.snapshot` recibe
esa vista y selecciona hasta ocho vecinos. El estado se publica mediante
`SessionArtifacts` como tensores CPU propios, sin otro formato de archivo. La
recuperación exige la misma identidad, un tamaño igual a `min(seen, capacidad)`, IDs
únicos, tiempos coherentes, claves válidas y el texto canónico del reservorio, que
debe ser el inicial mientras no haya sustituciones.

## Enlace con el coordinador

`FinancialSession` mantiene una única cronología: fases, evidencias del prefijo, cola
de pendientes, cierre administrativo y publicación mediante el Executor. Lo que
depende del consumidor está en un enlace cerrado
([`financial_consumers.py`](../../src/mars_titan/memory/financial_consumers.py)).
`TitansBinding` contiene el código anterior de Titans y `CandidateBinding` el de la
GRU. La sesión rechaza otros tipos de enlace y métodos sustituidos en la instancia.

`FrozenCandidateConsumer` exige parámetros sin gradiente, modo eval y ausencia de
métodos sustituidos. Su identidad incluye el adaptador, K, los flags numéricos y el
código. La comprobación fuerte vuelve a calcular la huella de los parámetros. La GRU
reinicia su estado oculto en cada ventana y el estado de trabajo de los K
refinamientos se descarta. Como no hay pesos rápidos ni momentum,
[`FlowCursors`](../../src/mars_titan/memory/flow_cursors.py) guarda solo
observaciones, último corte y última muestra de cada flujo, con los campos que el
coordinador ya comprueba para Titans.

En cada evento se construye una instantánea del banco confirmado antes de procesar
las etiquetas maduras de ese evento, y todos los bloques la comparten. El orden es
emitir y registrar predicciones, resolver las etiquetas anteriores y publicar una
generación. La emisión es la mediana de los cinco cuantiles. La cola guarda las
claves y valores del codec en la precisión del modelo, fuera del banco recuperable.
Al madurar, M1 admite la clave y el valor de la fila emitida, su etiqueta y el error
respecto de la predicción realmente emitida. M0 conserva el refinador con lectura
vacía y no escribe. El calentamiento solo avanza cursores y un `settlement` no
ejecuta el modelo.

Cada observación del registro nativo lleva 16 palabras, ocho de la huella de los
inputs y ocho de la huella de la codificación, en lugar de los 384 valores de la fila.
M2, M3 y C/M no están definidos sobre esta representación y la sesión los rechaza.

## Comprobaciones

Las comprobaciones se ejecutaron en CPU con dos hilos, `CUDA_VISIBLE_DEVICES=-1` y
etiquetas manuales. Un guardia local de pytest impidió pasos de optimizador y
ajustes de objetivos residuales. Los resultados y registros están en el
[recibo técnico](../../reports/engineering/gru-financial-sessions-20261009.json).

- El codec pasa 19 pruebas Python y su CTest, que comprueba el motivo de cada
  rechazo. Doce mutaciones quedaron detectadas y una era equivalente, porque
  `DecisionBatch.from_validated` vuelve a verificar el lote.
- El banco pasa 14 pruebas, incluida la paridad de plazas con `EpisodicMemory` v2 para
  capacidades 1, 4 y 9 y un lote FP64 de 8.192 episodios en la capacidad máxima.
- La sesión GRU pasa 19 pruebas: instantánea común frente a la GRU nativa con K = 1,
  2 y 4, orden de activos, bloques físicos, calentamiento, cierre, cortes en tres
  fronteras, etiquetas futuras o no finitas, disponibilidad anterior a la decisión,
  duplicados y codecs ajenos. Los cursores tienen cuatro pruebas propias. Las 22
  mutaciones dirigidas al banco, los cursores, el consumidor y el enlace quedaron
  detectadas.
- Un arnés de paridad ejecutó ocho escenarios Titans (M0, M1 con reservorio,
  uniforme, reciente y anclado, M2, con y sin diagnóstico C y bloques de 1, 2 y 4
  filas) sobre `develop` y sobre esta rama. Predicciones, episodios retenidos, índices
  M2, cola, estado rápido y contadores coincidieron exactamente.
- Los CTest del candidato, el banco nativo, la sesión financiera y el ejecutor pasan en
  Release y con ASan/UBSan. clang-tidy y el analizador de Clang no emiten
  diagnósticos sobre los archivos tocados.

La suite completa de pytest se ejecutó antes y después del cambio con el mismo
guardia. No apareció ningún fallo nuevo en memoria, candidato ni Titans. Los fallos
añadidos proceden de pruebas nuevas de `develop` que necesitan el simulador nativo,
no compilado en este entorno, o un ajuste que el guardia impide.

Veinticinco pruebas existentes de `test_financial_session.py` y
`test_financial_observation_source.py` no se ejecutaron porque su fixture estima
objetivos residuales. Fallan igual en la línea base y no pertenecen a esta ruta.

## Coste medido

Con dos hilos CPU y un proceso no exclusivo, llenar un banco de 8.192 plazas con 8.192
episodios tardó 0,022 s en FP32 y 0,031 s en FP64 (mediana de tres repeticiones). Una
segunda propuesta de 8.192 episodios sobre el banco lleno tardó 0,045 y 0,047 s y
sustituyó 4.044 plazas. Restaurar el snapshot tardó entre 0,018 y 0,022 s. Los
tensores del snapshot ocupan 12.910.592 bytes en FP32 y 25.493.504 en FP64, frente a
una estimación de trabajo de 64.618.496 y 127.533.056 bytes. El pico trazado por
Python no incluye la memoria de los tensores y no se usa como cota. El codec CPU
codificó 256 filas con las dimensiones por defecto en 0,056 s en FP32 y 0,079 s en
FP64. El RSS máximo del proceso de medida fue de 818.085.888 bytes. Son medidas del
componente con datos manuales y no estiman el caudal del corpus.

## Pendiente

- CUDA del codec, el consumidor y la sesión GRU. La comprobación preparada en
  [`cuda_gru_session_check.py`](../../tests/memory/cuda_gru_session_check.py)
  compara CPU y `cuda:0` en FP32/FP64, K = 1 y 4, M0 y M1, y recupera tras un corte.
  Solo se ha ensayado en CPU, lo que no acredita el dispositivo. Cuando la GPU quede
  libre, desde `native/`:

  ```bash
  cmake --preset native-candidate-cuda -B ../build/native/gru-session-cuda \
    -DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON -DMARS_TITAN_BUILD_SIMULATION=ON
  cmake --build ../build/native/gru-session-cuda
  ctest --test-dir ../build/native/gru-session-cuda -R "^candidate_cuda$"
  cd .. && MARS_TITAN_EPISODIC_NATIVE=$PWD/build/native/gru-session-cuda/_episodic_native.cpython-312-x86_64-linux-gnu.so \
    MARS_TITAN_GRU_CHECK_REPORT=$PWD/gru-session-cuda.json \
    uv run --no-sync --offline pytest -q tests/memory/cuda_gru_session_check.py \
    tests/models/candidate/cuda_candidate_check.py
  ```
- La aceptación sobre fases históricas completas, con su cola y almacenamiento.
- El coste del recorrido completo con las dimensiones reales del corpus.
- El contraste científico de la GRU frente a Titans y sus ampliaciones.
