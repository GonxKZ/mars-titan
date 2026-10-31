# Modulación contextual por segmentos dendríticos, N2

Está implementado el operador neuronal `ActiveDendrites` de `models/titans/active_dendrites.py`. Todavía no está conectado a `FinancialPredictor` ni al productor de contextos de régimen. No es una variante financiera ejecutable completa. No se han realizado entrenamientos, comparaciones predictivas ni medidas de coste de la sesión con N2.

La referencia es [Iyer y colaboradores, 2022](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.846219/full), secciones 3 y 6.3. Sus segmentos proyectan un contexto y el de mayor respuesta absoluta modula la activación, conservando el signo. El artículo también usa kWTA y conexiones dispersas. Este componente conserva la selección contextual, pero cambia la modulación y no incorpora esas otras piezas. No reproduce sus experimentos ni acredita sus resultados en finanzas. Se inspeccionaron esas secciones y partes de los resultados y la discusión, no el artículo completo. No se incorporó código de la implementación externa.

## Ecuación e inicialización

Para `h` de forma `[B, H]`, contexto `c` de forma `[B, C]` y pesos `U` de forma `[H, S, C]`:

```text
r[b,i,j] = sum_k U[i,j,k] c[b,k]
j* = primer argmax_j |r[b,i,j]|
d[b,i] = r[b,i,j*]
h'[b,i] = h[b,i] (1 + tanh(a[i]) tanh(d[b,i] / 2))
```

La identidad `tanh(d/2) = 2 sigmoid(d) − 1` permite evaluar la expresión sin restar números cercanos a 0,5. El vector `a` empieza en cero. Por tanto, la salida inicial coincide exactamente con `h`, aunque los segmentos sean distintos. Con contexto nulo también hay paridad para cualquier ganancia finita.

Los segmentos se inicializan con un generador CPU propio, uniforme entre `−1/sqrt(C)` y `1/sqrt(C)`. No se avanza el RNG global. Inicializar también `U` en cero dejaría nula la derivada inicial respecto a `a`. Con la inicialización propuesta, esa derivada es `h tanh(d/2)`, que puede ser distinta de cero cuando hay entrada y contexto informativos. Los gradientes de `U` empiezan en cero por la ganancia nula y aparecen cuando esta cambia. Esto describe el mecanismo, no prueba que aprenda una relación útil.

Cada unidad selecciona un segmento. En los empates gana el primero, con un desempate estable. La derivada de esa selección es local a la región de ganadores. No se afirma diferenciabilidad en las fronteras donde cambia el ganador.

En aritmética real, la amplitud puntual satisface `|h'| ≤ 2 |h|`. Esto no limita el Jacobiano conjunto cuando el contexto depende de `h`, ni demuestra estabilidad de una recurrencia. Una prueba compone ambos argumentos con la misma entrada y obtiene una derivada mayor que dos pese a respetar la cota puntual.

## Contratos y memoria

El operador admite FP32 y FP64 explícitos y rechaza autocast. Representación, contexto y parámetros deben tener el mismo tipo y dispositivo. Las filas son independientes y no se modifican las entradas. Se rechazan NaN, infinitos y desbordamientos de la proyección antes de aplicar las saturaciones. La ganancia inicial nula no oculta una proyección inválida.

Hay `H S C + H` parámetros compartidos y ningún estado temporal ni banco. La proyección cuesta `O(B H S C)` y produce `B H S` respuestas. Se limita a `2^20` el número de elementos de parámetros y, por llamada, el de respuestas. Estos límites no son un presupuesto del RSS, de toda la memoria temporal ni del grafo de autograd. La implementación usa operaciones densas de PyTorch. No se atribuye aceleración a tener pocos gradientes distintos de cero.

El `state_dict` permite recuperar los tensores del operador con la carga restringida de PyTorch. La identidad expone dimensiones, semilla, ecuación y límites. La recuperación de una sesión financiera y la validación de procedencia del contexto siguen pendientes y no quedan resueltas por ese archivo de pesos.

## Integración pendiente

N2 necesita una fuente de régimen que identifique corte, disponibilidad, representación y filas, y que publique el mismo contexto para todas las decisiones del grupo correspondiente. La fecha no puede comprobarse observando solo un tensor. Se ha localizado `memory/regimes.py`, que calcula regímenes observables a partir de ventanas pasadas y ofrece un control de calendario, pero aún no se ha conectado con N2.

También queda pendiente el control con contexto barajado. Si todas las filas de un mercado comparten régimen, permutarlas dentro de una sesión es un control equivalente. La alternativa temporal debe fijar qué contextos anteriores son elegibles, cómo se sortean y cómo se recupera su estado. No puede acceder a contextos futuros de una ventana para simular una permutación causal.

Faltan la configuración financiera desactivable, su punto de inserción, las pruebas de composición con Titans y las áreas, la recuperación de la siguiente decisión y el coste del recorrido completo. La comprobación CUDA se mantiene separada. El operador aislado no cierra N2 ni la issue de N1, N2 y BDH. No hay mejora predictiva medida.
