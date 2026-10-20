# Modulación de escritura y captura neuronal

`MemoryPlasticity` configura dos intervenciones independientes sobre `NeuralMemory`. `FinancialConfig.memory_plasticity=None` conserva la identidad, los campos serializados y el recorrido anteriores. La configuración activa tiene identidad propia. No se habilita en ninguna receta histórica.

## Estado y actualización

Para cada flujo, capa y fila de una matriz de pesos rápidos de dimensión D, el estado añade una media vectorial m de D elementos, una energía escalar v y una traza escalar z. Se inicializan a cero. Con L capas y N flujos, se añaden exactamente N·L·(D²+2D) elementos del tipo de los pesos. El contador de observaciones existente permite corregir el arranque de los momentos.

Sea g el gradiente de la pérdida asociativa, después del recorte PT1 si está habilitado. Con t observaciones anteriores:

```text
m' = beta·m + (1−beta)·g
v' = beta·v + (1−beta)·||g||²
a  = 1−beta^(t+1)
c  = clip(||m'||² / (a·v' + epsilon), 0, 1)
u  = clip(||g||² / (v'/a + epsilon), 0, 1)
z' = trace_decay·z + (1−trace_decay)·u
```

El cociente c mide persistencia de dirección en la historia observada. Si beta=0, a=1. En los demás casos se calcula a mediante `expm1` para evitar la resta de números próximos a uno. `epsilon` tiene unidades de gradiente al cuadrado. Se exige que sea normal y representable en FP32 o FP64.

Las escalas cuentan actualizaciones observadas. Con los valores iniciales 0,9 y 0,95, las semividas de los factores exponenciales son unas 6,58 y 13,51 actualizaciones. Un hueco sin observaciones no hace avanzar estos contadores. Esas semividas no describen retención semántica ni tiempos biológicos.

La escritura usa `w = write_floor + (1−write_floor)·c`. Con las puertas α, η y θ del núcleo, la actualización completa del momentum es:

```text
S' = η·S − (θ·w)·g
W' = (1−α)·W + S'
```

El modo `write="none"` usa w=1. El modo `constant` usa `write_constant`. Los controles activos conservan las estadísticas auxiliares para distinguir el efecto de la ganancia de su coste.

La captura requiere PT5 en modo `cascade`. La señal q combina la traza local con la media de coherencia de las filas de esa capa y flujo:

```text
q = z'·mean_rows(c)
h = capture_floor + (1−capture_floor)·q
e = g12·h
W'  = (1−α)·W + S' + e·(Z2−W)
Z2' = Z2 + (e·(W−Z2) + g23·(Z3−Z2))/2
Z3' = Z3 + (g23·(Z2−Z3) − g34·Z3)/4
```

Todos los valores del lado derecho corresponden al estado anterior, salvo las señales recién calculadas y S'. El intercambio e se aplica en ambos sentidos. Las capacidades 1, 2 y 4, g23 y g34 proceden de PT5 y no cambian. `capture="none"` conserva g12. `constant` utiliza `capture_constant` como multiplicador. Una configuración de captura sin cascada se rechaza.

No se añaden parámetros compartidos. Las estadísticas se actualizan exclusivamente con observaciones admitidas por Titans. En el modo compartido se emplea el gradiente de cohorte ya definido por PT4 y el contador avanza una vez por cohorte. No hay RNG nuevo.

## Alcance matemático

Con inicialización nula, los pesos de la media exponencial suman a. Cauchy–Schwarz ponderada da `||m'||² ≤ a·v'` en aritmética exacta. Por tanto el cociente previo a la saturación pertenece a [0,1] cuando epsilon>0. Esto no demuestra una varianza posterior, una cota del error financiero ni estabilidad de la red.

La saturación entre cero y uno forma parte del algoritmo numérico. Los gradientes, momentos, energías corregidas, denominadores y cocientes se comprueban antes de saturar. Un desbordamiento produce un error, no una ganancia aparentemente válida. Los resultados FP32/FP64 no son cotas numéricas certificadas.

La estadística por fila conserva su valor ante una rotación ortogonal conjunta de los gradientes y su media, porque solo utiliza sus normas. Esta propiedad no implica invariancia de toda la red. Una escala multiplicativa no conserva exactamente el cociente cuando epsilon es relevante. Para una secuencia constante no nula, `c = a²·||g||²/(a²·||g||²+epsilon)`. Por eso un gradiente persistente muy pequeño puede recibir una ganancia próxima al suelo.

La actualización de z es una combinación convexa, por lo que conserva [0,1] si z y u están en ese intervalo y `0≤trace_decay<1`. Las ganancias permanecen entre su suelo y uno. Un suelo positivo evita anular directamente la escritura, pero no garantiza conservar información útil ni aprender una asociación.

Para el intercambio aislado entre capacidades 1 y 2, fijado e, sean `x'=x+e(y−x)` e `y'=y+e(x−y)/2`. Entonces:

```text
(x')² + 2(y')² − (x² + 2y²) = −(2e−3e²/2)·(x−y)²
```

La energía ponderada no crece para `0≤e≤4/3`. Este resultado solo describe ese intercambio pasivo con e fijado. La actualización real incluye pérdida asociativa, puertas, olvido, momentum, otras interfaces y una ganancia dependiente del estado. La desigualdad no es un certificado de contracción de Titans ni del predictor.

El Jacobiano continuo incluye pesos rápidos, momentum, niveles PT5, media, energía y elegibilidad. El contador es una condición discreta del paso. `transition_jacobian.py` incluye esos elementos en su punto, dimensión y reconstrucción. Una estimación local sigue siendo un diagnóstico local.

La saturación introduce puntos no diferenciables y el estado admisible exige v≥0 y z∈[0,1]. Una matriz obtenida por autodiferenciación en esas fronteras no demuestra un Jacobiano clásico en un entorno abierto ni una cota uniforme sobre el dominio.

## Fronteras de software y comprobaciones

| Responsabilidad | Implementación | Comprobación principal |
| --- | --- | --- |
| Reglas y configuración | `plasticity_config.py`, `plasticity.py` | Momentos ponderados, límites, gradcheck, valores extremos y controles constantes |
| Escritura real | `neural_memory.py` | Ecuación del momentum y acoplamiento en ambos sentidos |
| Estado por flujo | `state.py`, `mac.py` | Copia, selección de filas, unión, alias y aislamiento |
| Persistencia financiera | `financial.py`, `financial_blocks.py` | Exportación CPU, restauración y siguiente predicción |
| Diagnóstico C | `transition_jacobian.py` | Dimensión completa y derivadas de los estados añadidos |
| Recetas | `FinancialConfig`, `checked_predictor_changes` | Identidad propia y rechazo de combinaciones incompatibles |

Las rutas anteriores están dentro de `src/mars_titan/models/titans`, salvo el registro de cambios de predictor, situado en `src/mars_titan/training/search_cases.py`. Las pruebas nuevas están en `tests/models/titans/test_plasticity.py`, `test_plasticity_integration.py` y `test_financial_plasticity.py`.

Con `differentiable=True`, los gradientes externos atraviesan los momentos, ganancias y trazas. En inferencia se separan del grafo. No se retiene el historial completo. Los estados exportados incluyen `modulation` únicamente en la identidad activa. Se rechazan campos ausentes, tipos o dispositivos incompatibles, energía negativa, trazas fuera de dominio y almacenamiento compartido entre estados que deban ser independientes.

La preasignación cuenta el estado auxiliar antes de crear el lote. Esta cuenta no incluye el grafo temporal ni los buffers de las bibliotecas. Los picos de RAM y VRAM necesitan una medida aparte. Una configuración con memoria auxiliar y ganancias unitarias es un control de coste, no una afirmación de memoria gratuita.

El recorrido interno reutiliza la validación de configuración, precisión, estado y contador que realiza `NeuralMemory` antes de recorrer los tokens. La función pública `advance` conserva esa validación para llamadas independientes. Ambas comparten `_advance_validated`, que mantiene la comprobación de los valores intermedios. La optimización evita comprobaciones redundantes, sin quitar el rechazo de estados externos inválidos ni de desbordamientos. La [comparación emparejada](../../reports/engineering/neural-plasticity-20261020/boundary-cost.json) registra el recorrido completo y la paridad comprobada.

La comprobación técnica cubre paridad exacta de la ruta desactivada y de los pesos con ganancias unitarias, partición de secuencias, recuperación, causalidad de prefijos, orden de flujos, FP32/FP64 y CUDA. Las mutaciones dirigidas alteran ganancias, captura, serialización, contador, presupuesto y dimensión del Jacobiano. Sus resultados y denominadores se registran en el informe de verificación.
