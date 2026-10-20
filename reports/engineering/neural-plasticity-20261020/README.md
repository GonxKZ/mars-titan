# Coste de las variantes de plasticidad

Las mediciones usan la RTX 4070 Laptop, PyTorch 2.14.0+cu130 y dos hilos CPU. La precisión es explícita, con TF32 desactivado. Se recorren ocho decisiones históricas comunes a los activos seleccionados, con pesos compartidos iniciales, sin pasos de optimizador y sin leer etiquetas financieras. No son resultados predictivos.

`neural_plasticity_cost.py` mide el predictor financiero completo desde entradas precodificadas residentes. Incluye validación, codificadores del predictor, atención, lectura, actualización neuronal y cabeza. La preparación y transferencia del lote, la inicialización y el backward se registran aparte. El tiempo total del proceso incorpora las operaciones de auditoría. La lectura Parquet por decisión, los codificadores anteriores al corpus y la energía no se midieron.

## Comparación de configuraciones

La primera medida tiene dos calentamientos y diez repeticiones, alternadas entre diez modelos, para 1, 64 y 128 activos y para FP32 y FP64. Cada repetición contiene ocho decisiones. Los percentiles se calculan con interpolación lineal. Los [resultados completos resumidos](cost.json) incluyen p50, p95, p99, caudal, memoria, procedencia y huellas de los recibos originales.

Esta medida utiliza `linear_v1` y `batch_first_v1`, de la receta histórica tomada como referencia. El piloto preparado usa `titans_mac_paper_projections_v2` y `sequence_contiguous_v1`, con su propio registro de coste. Los tiempos de esta tabla no se trasladan a esa configuración.

La tabla recoge p50 por decisión para 128 activos, antes de la optimización de validación. El tamaño de estado es por activo y solo cuenta sus tensores persistentes, no el grafo temporal.

| Configuración | p50 FP32 (ms) | p50 FP64 (ms) | Estado FP32 (bytes) |
| --- | ---: | ---: | ---: |
| Titans sin modulación | 4,684 | 7,284 | 65552 |
| Escritura por coherencia | 7,080 | 10,222 | 99344 |
| PT5 fijo | 5,142 | 8,223 | 131088 |
| PT5 y escritura | 7,501 | 10,130 | 164880 |
| PT5 y captura | 6,708 | 9,612 | 164880 |
| PT5 y ambas modulaciones | 6,584 | 9,646 | 164880 |
| PT5 con ganancias unitarias | 7,497 | 10,133 | 164880 |

Las estadísticas nuevas añaden 33.792 bytes por activo en esta configuración FP32. No añaden parámetros compartidos. Los picos del asignador incluyen los diez modelos residentes usados para alternar las medidas. No deben interpretarse como memoria exclusiva de una petición. La dispersión entre repeticiones impide ordenar configuraciones próximas solo por su mediana.

Se intentó materializar 256 activos con el selector de ocho sesiones comunes del programa existente. Ese conjunto concreto no tenía suficientes activos presentes en todas las sesiones elegidas. No se duplicaron activos ni se completaron precios para alcanzar el tamaño. Esto no impide un estudio de 256 activos con cohortes variables en el lector cronológico.

El catálogo incorporó después un control adicional de escritura constante sin cascada. No aparece en esta primera medida de diez modelos. Su ejecución técnica está comprobada, pero no se le asignan los tiempos de otro modelo.

## Validación de frontera reutilizada

El [perfil instrumentado](profile.json) localizó trabajo repetido en las comprobaciones de entrada de la plasticidad. La validación pública y la de `NeuralMemory` comprobaban de nuevo configuración, precisión, estado y contador dentro de cada token. La ruta interna reutiliza la frontera ya validada. Conserva los controles de finitud de los intermedios y la función pública completa.

Se comparó la misma configuración combinada con ambas rutas, invirtiendo el orden de cada pareja. Hubo tres calentamientos y cien parejas por tamaño y modo. La medida final incluye inicialización, forward, backward cuando corresponde, comprobación de gradientes, copia de salidas a CPU y limpieza. Los parámetros compartidos y el RNG permanecieron iguales y las predicciones coincidieron bit a bit.

La mediana del cociente de tiempo completo entre la ruta interna y la referencia fue 0,9587 y 0,9673 en inferencia con 1 y 128 activos. Con grafo y backward fue 0,9776 y 0,9787. Equivale a reducciones medianas emparejadas del 4,13 %, 3,27 %, 2,24 % y 2,13 % en estos recorridos. La ruta interna fue más rápida en 72, 77, 73 y 74 de las cien parejas respectivas. Se conserva el cambio por esa reducción y por la paridad comprobada.

El [recibo emparejado](boundary-cost.json) conserva los tiempos individuales y la dispersión de sus cocientes. Son repeticiones del mismo recorrido, no nuevas observaciones temporales del mercado ni intervalos de generalización. Las rondas previas de 30 y 100 parejas que sumaban tiempos de etapas permanecen como antecedentes locales. La tabla anterior de configuraciones y esta comparación de rutas no se mezclan para fabricar una aceleración conjunta.

## Reproducción y límites

`uv run --no-sync python -m benchmarks.neural_plasticity_cost --help` describe los tamaños, calentamiento, repeticiones y precisión. `--data` recibe la materialización sin objetivos del programa `benchmarks.titans_paper_projections_cost`. `--output` exige un archivo nuevo. El lector rechaza metadatos que admitan objetivos o fechas desde 2024 antes de abrir los arrays.

`uv run --no-sync python -m benchmarks.neural_plasticity_boundary_cost --help` describe la comparación emparejada. Ambas órdenes requieren `cuda:0` y fallan si CUDA no está disponible. La medición instrumentada con coverage.py se usa únicamente para cobertura del recorrido y no se incorpora a las cifras de rendimiento.

No se introdujeron kernels nuevos ni cambios de driver, precisión o seguridad. Permanecieron las aplicaciones gráficas del equipo. Estas cifras no presupuestan por sí solas la campaña histórica completa ni demuestran una mejora financiera.
