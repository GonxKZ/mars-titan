# Referencia Transformer compacta

`MultimodalReference(kind="transformer", ...)` permite contrastar otro codificador de precios con las mismas proyecciones de modalidades, fusión y cabeza escalar de las referencias recurrentes. Es una referencia adicional con identidad propia. No representa Titans y no reemplaza la GRU. Esta incorporación no aporta resultados predictivos ni permite suponer una mejora.

La entrada de precios tiene forma `[B,T,F]`. Una proyección lineal lleva cada observación a `D` canales y añade posiciones sinusoidales explícitas. Entre una y dos capas de `TransformerEncoderLayer` aplican atención causal y una red FFN, con normalización previa. Una normalización final precede a la selección del último token. El codificador devuelve `[B,D]` y `encode_sequence` permite obtener `[B,T,D]` para comprobar prefijos temporales. Cada llamada procesa una ventana independiente.

La primitiva y sus argumentos corresponden a [TransformerEncoderLayer de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoderLayer.html). Se utiliza una máscara triangular explícita. Los bloques se construyen por separado y no comparten parámetros. La atención y la FFN no usan dropout. El parámetro `dropout` de la referencia sigue controlando la fusión común.

La anchura `D` admite 32, 64 o 128 canales. El contexto tiene entre 2 y 512 observaciones y cada modalidad mantiene el límite de 2.048 canales. `transformer` acepta exactamente `heads` en 1/2/4/8 y `feedforward_multiplier` en 2/4. Si no se proporciona, se resuelven cuatro cabezas y una FFN de anchura `2D`.

```python
from mars_titan.models.baselines.multimodal import MultimodalReference

# Dimensiones de un fixture técnico, sin datos de mercado ni aprendizaje.
dimensions = dict(prices=5, news=8, charts=7, fundamentals=6, macro=9)
model = MultimodalReference(
    "transformer",
    dimensions,
    context=64,
    hidden_size=64,
    layers=1,
    dropout=0.1,
    transformer=dict(heads=4, feedforward_multiplier=2),
)
configuration = model.configuration
```

Se admiten float32 y float64, con el mismo dtype y dispositivo en entradas y parámetros. Las posiciones se calculan desde una base float32 fija y se convierten al dtype de cálculo. Así no dependen del dtype global utilizado al reconstruir el modelo. No se admite `autocast`. La revisión CUDA y la precisión mixta son comprobaciones distintas y no se acreditan mediante pruebas CPU.

La [verificación del 8 de octubre](../../reports/research/compact-transformer-verification-20261008.json) incluye 115 pruebas CPU y una revisión independiente con 14 sondas adicionales. Las 144 configuraciones previas conservan exactamente pesos, salidas, gradientes y RNG. Se ejecutaron comprobaciones CUDA en FP32 y FP64 para `B=2`, `T=8`, `D=32` y dos capas. El prefijo y la recuperación son exactos y el gradiente desde tokens futuros es cero. El pico observado fue inferior a 18,1 MB asignados por Torch. El contexto CUDA ocupa memoria adicional. La aplicación externa permaneció activa y no se atribuye una aceleración a estas comprobaciones.

Cada llamada admite hasta 256 ventanas. También se exige `B × T² × heads × layers ≤ 2²⁴` antes de calcular atención. Esta cota limita posiciones de atención, no representa una medición de RAM ni garantiza un pico concreto. Una población mayor se procesa en lotes menores, sin eliminar activos o filas. No se aceptan ventanas vacías, formas incompatibles ni valores no finitos.

La configuración resuelta contiene `price_encoder_contract`, con versión, posiciones, normalización, causalidad, agregación, opciones y límites. El codificador guarda además ese contrato en `_extra_state` dentro de `state_dict`. Su hook de carga valida el contrato antes de copiar sus pesos, tanto en carga directa como al formar parte de otro módulo. El contrato sigue siendo obligatorio con `strict=False`. `initialize_weights` comprueba también la configuración externa, el estado adicional y las huellas antes de cargar. Así se rechazan cuatro cabezas frente a dos aunque las matrices tengan las mismas dimensiones. Las posiciones y la máscara se reconstruyen desde el contrato, sin tratarlas como pesos aprendidos. Cualquier integración posterior debe incluir también `transformer.py` entre las huellas de código.

Las entradas finitas no garantizan que la aritmética permanezca finita. Esta ruta comprueba la representación temporal, la fusión y la salida. Un desbordamiento o unos pesos alterados que produzcan NaN/Inf causan un error explícito. No se recortan ni sustituyen esos resultados.

Los constructores y el consumo de RNG de RNN, LSTM, GRU y DLinear se conservan. El contraste entre familias comparte la arquitectura de fusión y cabeza. No presupone que las distintas familias produzcan la misma representación ni que una semilla asigne idénticos pesos a todas ellas.

Las listas de familias de campañas, postentrenamiento y observatorio permanecen cerradas. No se ha añadido una configuración de experimento ni se ha ejecutado aprendizaje. Las pruebas técnicas usan fixtures para comprobar causalidad, gradientes, independencia de ventanas, serialización, límites y rechazo de contratos incompatibles. La comparación con la versión anterior comprueba además pesos, salidas, gradientes y RNG de las cuatro referencias existentes.
