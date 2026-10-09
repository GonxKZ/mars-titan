# Referencia GRU con lectura episódica

La [política histórica opt-in](candidate_historical.md) amplía esta referencia con
máscaras explícitas. Las ecuaciones y archivos descritos aquí corresponden a la
ruta estricta original.

`mars_titan::candidate::Candidate` implementa la referencia GRU con lectura episódica. Recibe entradas ya normalizadas y una instantánea de episodios maduros. No admite, madura ni expulsa episodios. Tampoco entrena, selecciona checkpoints o recorre un corpus. La integración cronológica y la comparación científica siguen pendientes en #23. Esta identidad se conserva separada de Titans-MAC y de sus ampliaciones. Elegirla como B exige fijar el contraste correspondiente.

Las entradas son precios `[N,64,Dp]`, noticias `[N,Dn]`, gráficos `[N,Dc]`, fundamentales `[N,Df]` y macro `[N,Dm]`. Las dimensiones y la identidad de normalización deben corresponder al manifiesto. `presence[N,5]` es booleana y debe ser verdadera en todas las posiciones. Una modalidad ausente se rechaza. La memoria vacía tiene otra máscara y sí está admitida.

Una GRU de una capa reinicia su estado en cada ventana. Sus 128 salidas se concatenan con cuatro proyecciones lineales con SiLU de 128 dimensiones. La fusión lineal con SiLU produce 256 valores. El estado de trabajo inicial es una proyección lineal a 128. No hay dropout ni estado recurrente arrastrado entre llamadas.

Los rasgos persistentes proceden de una proyección fija de las entradas aplanadas a 256 y otra proyección fija produce claves de 128. Ambas usan generadores CPU independientes y se guardan como tensores privados. Su identidad incluye versión matemática, normalización, dimensiones, precisión y SHA-256 de las matrices reales. Las semillas se conservan en la configuración. Las claves y consultas se normalizan mediante `v / max(||v||₂, 1e-12)`. Un vector cero permanece cero. El constructor y el cálculo no consumen el RNG global.

La norma se calcula tras escalar por la magnitud máxima para evitar desbordamientos intermedios con valores finitos grandes. La huella se calcula al construir, recuperar o cambiar de precisión, sin recalcularla ni copiar matrices a CPU durante la lectura. Las proyecciones no aparecen en `named_buffers` ni comparten almacenamiento con parámetros públicos. La recuperación las clona, incluso si el archivo contiene tensores con almacenamiento compartido. La conversión explícita de dispositivo conserva la identidad. La conversión de precisión la recalcula e invalida las instantáneas anteriores.

`snapshot` valida forma, precisión, dispositivo, finitud, revisión e IDs no negativos estrictamente crecientes en CPU. Copia claves, rasgos y retornos maduros y los separa del grafo. La responsabilidad de acreditar madurez temporal sigue siendo del ejecutor. Las claves tienen norma uno o son nulas. Los IDs se trasladan al dispositivo del modelo una sola vez.

La consulta selecciona hasta ocho vecinos. Una ordenación estable conserva el menor ID ante similitudes iguales. El top-k queda fuera de autograd. Los pesos `softmax(q·k/τ)` conservan el gradiente de la consulta normalizada. Los valores concatenan rasgos y retorno maduro y se proyectan a 128 al leer. Con un vecino, el peso es uno y el gradiente de selección es cero. La memoria vacía devuelve lectura cero y máscara falsa.

El refinamiento es `z' = z + sigmoid(a) tanh(W[z,h,lectura,presencia] + b)`, con paso inicial 0,1. `read` y `refine` exponen la transición para estudiar su Jacobiano completo. K admite 1, 2 o 4 y siempre reutiliza la misma instantánea. K=1 es la configuración principal. Las funciones de componentes reciben tensores finitos ya validados. `forward` comprueba la frontera y la salida.

La cabeza produce cuantiles 0,025, 0,1, 0,5, 0,9 y 0,975. La mediana es libre y los cuatro incrementos usan softplus. El orden es no decreciente en coma flotante. No se deduce una densidad o NLL de estos cuantiles.

Los límites son 256 filas por llamada, 8.192 episodios y 16.384 entradas aplanadas. La búsqueda materializa a lo sumo `[N,E]` similitudes, no distancias entre todos los episodios. El coste y las transferencias del recorrido real aún no están medidos. La validación finita se hace al codificar y crear la instantánea, no se repite en cada paso de lectura.

`save_state` y `load_state` guardan y validan versión, configuración, parámetros, proyecciones, precisión, modo train/eval y versión LibTorch. El formato 2 rechaza archivos del formato anterior. La recuperación compara también la huella almacenada con las matrices leídas. Admiten archivos locales de confianza de hasta 128 MiB y un máximo adicional de 128 MiB en la suma de registros descomprimidos. Se rechazan directorios ZIP inválidos, truncados, con nombres duplicados o más de 1.024 registros antes de que LibTorch cargue tensores. Los nombres se limitan a 511 bytes. Estos límites no son un máximo del RSS del proceso. No son un checkpoint de entrenamiento, pues no incluyen optimizador, cursor ni banco episódico. La escritura recibe un stream y su publicación atómica corresponde al consumidor. No se debe cambiar la precisión después de crear una instantánea.

La comprobación previa utiliza [miniz 3.1.0](https://github.com/richgel999/miniz/tree/174573d60290f447c13a2b1b3405de2b96e27d6c), con licencia MIT, descargado solo para el objetivo candidato y fijado por SHA-256 en CMake. La compilación conserva su licencia en `candidate-miniz-LICENSE`. No basta consultar tamaños con el lector de LibTorch, porque [su inicialización](https://github.com/pytorch/pytorch/blob/v2.14.0/caffe2/serialize/inline_container.cc) ya descomprime registros de versión e identidad. El control previo no descomprime registros y suma tamaños tras comprobar el presupuesto restante.

## Compilación y comprobación

El objetivo es opcional mediante `MARS_TITAN_BUILD_CANDIDATE`. Los presets `native-candidate-debug`, `native-candidate-release`, `native-candidate-asan-ubsan`, `native-candidate-static-analysis`, `native-candidate-coverage` y `native-candidate-fuzz` usan CPU y conservan C++20. `MARS_TITAN_TORCH_ENVIRONMENT` permite indicar otro entorno uv existente. La opción vacía mantiene `.venv` como origen del SDK. Los perfiles PPO no cambian.

El preset `native-candidate-cuda` activa el backend CUDA de LibTorch y añade `candidate_cuda` a CTest. Mantiene los casos CPU y exige `cuda:0` para el caso GPU. El ejecutable `mars-titan-candidate cpu` hace un cálculo sintético y comprueba su recuperación exacta. Un segundo argumento permite guardar el archivo en una ruta nueva. No hay fallback a CPU. La comprobación sintética no ejecuta aprendizaje ni acredita utilidad predictiva.

Las pruebas contrastan la GRU con sus ecuaciones explícitas, un caso analítico de atención, diferencias finitas de la consulta, orden y forma de cuantiles, entradas incompletas, empates, separación de memoria, independencia de semillas y recuperación de predicciones y gradientes.

La [comprobación del 8 de octubre de 2026](../reports/research/gru-cuda-verification-20261008.json) utiliza dos filas con las dimensiones por defecto, memoria vacía con K=1 y ocho episodios con K=1, 2 y 4. Los ocho casos, repartidos entre FP32 y FP64, comparan salidas y gradientes CPU/CUDA y recuperan el módulo. El error máximo de cuantiles fue 2,3842e-7 en FP32 y 6,6614e-16 en FP64. El pico del asignador Torch fue 99.061.760 bytes. El caso de recuperación reconstruye la instantánea del fixture, no recupera un banco persistente ni un cursor cronológico.

Pasaron seis casos CTest en Release y cinco CPU con ASan/UBSan y detección de fugas. Compute Sanitizer memcheck terminó sin errores en los ocho casos CUDA. Clang 21 no emitió diagnósticos propios en clang-tidy ni en el análisis de rutas ejecutado. La combinación ASan/UBSan con CUDA falló antes del modelo en `cudaGetDeviceCount`, con código 2 de memoria insuficiente. Un programa mínimo enlazado al mismo runtime reprodujo ese fallo y obtuvo un dispositivo al compilarse sin sanitizadores. Esa combinación queda sin verificar. No se han desactivado comprobaciones para hacerla pasar.

cuDNN avisa de que los pesos GRU se compactan en cada llamada. El recorrido completo no se ha perfilado y no se atribuye aceleración a esta ruta. Las comprobaciones técnicas no levantan el bloqueo de aprendizaje de la edición histórica.

## Referencias de implementación

La [GRU de PyTorch 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) aplica la puerta de reset al término recurrente de la candidata, incluido su bias. La referencia independiente reproduce ese orden. La [normalización](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.normalize.html) usa un suelo en el denominador y la [ordenación estable](https://docs.pytorch.org/docs/2.14/generated/torch.sort.html) conserva el orden previo de valores iguales. Las firmas ATen y el formato `torch::serialize::Archive` se contrastaron también con las cabeceras instaladas de LibTorch 2.14.
