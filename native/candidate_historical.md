# Entradas históricas del candidato GRU

`historical_masked_2000_v1` añade una ruta explícita al candidato C++ descrito en
[candidate.md](candidate.md). Conserva la GRU de precios de 128 unidades, las
ventanas de 64 sesiones, la cabeza de cinco cuantiles y el refinador residual.
La mediana es la columna 2. K = 1, 2 o 4 cuenta refinamientos del mismo estado
inicial, sin volver a codificar la ventana.

La política predeterminada `strict_inputs_v1` exige las cinco modalidades. Mantiene
la representación `candidate-fixed-v1` y la lectura y escritura del archivo 2.
La ruta histórica usa `candidate-masked-fixed-v1` y archivo 3. Un lector estricto
rechaza este último. La carga histórica necesita indicar la política esperada.

## Máscaras y representación

La presencia contiene cinco booleanos, en el orden precios, noticias, gráficos,
fundamentales y macro. Precios y gráficos son obligatorios. Cada bloque ausente
contiene ceros y todos los valores deben ser finitos, también los ausentes.
Fundamentales y macro concatenan valores, observación y edad logarítmica. La
observación vale 0 o 1, la edad es no negativa y cada concepto ausente tiene valor
y edad cero. La presencia coincide con la existencia de algún concepto observado.

Cada modalidad lateral calcula `presence * SiLU(Wx + b)`. Aplicar la máscara
después de la proyección impide que el sesgo de un bloque ausente entre en la
fusión. Esta recibe los cinco vectores de 128 posiciones y los cinco bits de
presencia, por lo que su anchura de entrada pasa de 640 a 645.

El codec fijo recibe los inputs aplanados y esos mismos cinco bits. Mantiene
rasgos de 256 posiciones y claves de 128. Sus proyecciones utilizan generadores
CPU propios y quedan identificadas por sus bytes, dimensiones, normalización y
precisión. No dependen de los parámetros aprendidos ni conservan su grafo. La
representación distingue una ausencia de una noticia presente cuyo vector es
cero. Las claves pueden ser nulas, conforme al contrato original del candidato.

La cota de 16.384 posiciones aplanadas incluye los cinco bits nuevos. Se mantienen
los límites de 256 filas, 8.192 episodios y ocho vecinos del componente nativo.
El adaptador de esta fase usa exclusivamente memoria vacía.

## Enlace tensorial y traslado explícito

Al compilar `MARS_TITAN_BUILD_CANDIDATE=ON` junto con
`MARS_TITAN_BUILD_EPISODIC_PYTHON=ON`, `_episodic_native` incorpora el candidato.
Utiliza los tensores y autograd de la misma instalación de PyTorch. No serializa
tensores como JSON ni construye otra GRU en Python. El enlace episódico sin esta
opción mantiene su interfaz anterior.

`CandidateInputAdapter` en `mars_titan.models.candidate.input_adapter` recibe
`FinancialInputSpec` y consume `CPUDecisionBatch`. La validación compartida
comprueba el dtype float32 original, las máscaras, los rellenos, los IDs, las
fechas anteriores a 2024 y la disponibilidad. Su digest se verifica antes de
convertir al dispositivo y a FP32 o FP64. Los campos de supervisión no entran en
el cálculo. El adaptador devuelve los cinco cuantiles, su mediana y el resultado
nativo, incluidos sus rasgos fijos desacoplados del grafo.

`transfer_strict_parameters` requiere políticas distintas, las mismas dimensiones,
normalización semántica, precisión y dispositivo. Comprueba los parámetros antes
de copiar. Traslada las 640 columnas originales de fusión y pone a cero sus cinco
columnas de presencia nuevas. Conserva las proyecciones fijas del destino
histórico. Devuelve un recibo con las huellas de parámetros y representación, los
nombres copiados y las columnas inicializadas. No sustituye la carga ordinaria
ni permite equiparar catálogos diferentes porque tengan igual anchura. La
normalización semántica excluye únicamente los campos que declaran política y
máscara. La identidad completa conserva esos campos y la procedencia.

La exportación del adaptador añade al archivo nativo una identidad que vincula
`FinancialInputSpec`, configuración, parámetros reales, representación fija,
precisión, modo train/eval, versión de Torch y hashes de código y binario. La
restauración exige el mismo contrato y vuelve a calcular esas huellas. El límite
del archivo nativo es 128 MiB tanto codificado como expandido. Las huellas fuertes
pueden copiar parámetros a CPU en estas fronteras, no en cada `forward`.

La compilación CPU del enlace reutiliza el SDK instalado. Desde `native/`:

```bash
cmake --preset native-candidate-release \
  -DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON -DMARS_TITAN_BUILD_SIMULATION=ON
cmake --build --preset native-candidate-release
```

`MARS_TITAN_EPISODIC_NATIVE` debe apuntar al archivo `_episodic_native` resultante.
Con esa ruta explícita, las comprobaciones nuevas se ejecutan mediante
`uv run --no-sync --offline pytest tests/models/candidate`. La sonda
`tests/models/candidate/cuda_candidate_check.py` requiere recompilar con
`MARS_TITAN_LIBTORCH_ENABLE_CUDA=ON`, seleccionar `cuda:0` y disponer de su cuota
de 128 MiB para el allocator. Sus seis casos usan memoria vacía y contrastan
salidas, gradientes, recuperación y RNG con las tolerancias de la referencia GRU.

## Alcance de esta fase

La memoria vacía conserva el refinador. No equivale a omitirlo ni a usar la GRU
escalar de `MultimodalReference`. La política estricta y la histórica tienen
identidades diferentes aunque se hayan emparejado sus parámetros.

Falta conectar el candidato al recorrido cronológico, los cursores, warmup,
resolución de pendientes y publicación de una generación común. También falta
un banco identificado para claves de 128 y valores de 256, con sus reglas de
claves nulas, admisión, snapshot, retención y recuperación. No se reduce esta
geometría a la del banco compartido de 64 × 64. Esta entrega no implementa
M2/M3 ni C/M sobre la GRU y no acredita entrenamiento o comparación predictiva.
La integración cronológica deberá fijar también los flags de ejecución numérica
durante la sesión y comprobarlos al recuperar, además de la identidad de este
adaptador.

## Comprobaciones técnicas

El [recibo de verificación](../reports/engineering/historical-gru-inputs-20261009.json)
registra 49 pruebas nuevas y 14 de compatibilidad del enlace, seis CTest nativos,
ASan/UBSan en los ejecutables y diez mutaciones detectadas. El archivo estricto
de referencia conserva exactamente sus bytes. La revisión independiente CPU
ejecutó 49 pruebas públicas y ocho sondas propias, sin hallar defectos materiales
en ese alcance. No repitió los sanitizadores ni ejecutó GPU.

La comprobación CUDA posterior pasó los seis casos de memoria vacía, con 21
gradientes contrastados por caso, recuperación exacta y RNG intactos. El pico
reservado de Torch fue de 96 MiB, dentro de la cuota de 128 MiB. Se conservaron
las tolerancias de la referencia y el aviso de cuDNN sobre pesos GRU no contiguos.
Esta prueba no mide una mejora de rendimiento ni de predicción.
