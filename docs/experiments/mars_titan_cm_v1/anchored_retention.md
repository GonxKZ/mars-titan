# Retención con centros fijos

[`select_anchored_medoids`](../../../src/mars_titan/cm/anchored_medoids.py) calcula la selección variable cuando parte del banco se conserva. La función recibe el conjunto actual E como una matriz de coordenadas y sus IDs. `fixed_ids` y `candidate_ids` deben pertenecer a E y ser disjuntos. Los centros son episodios reales. La API no decide si una etiqueta ha madurado ni publica el banco.

Si A contiene los centros fijos, sus clientes tienen contribución cero porque se mantienen como centros. Para los demás se calcula

\[
b(e)=\min_{a\in A}d(e,a),\qquad
J_E(A\cup S)=\sum_{e\in E\setminus A}\min\bigl(b(e),\min_{s\in S}d(e,s)\bigr).
\]

Todos los clientes de E contribuyen, aunque no estén autorizados como candidatos. La restricción afecta a los centros variables y al máximo recambio. El resultado no es un óptimo global sin esas restricciones. Los duplicados geométricos entre IDs mantienen su multiplicidad y definen una pseudométrica sobre episodios.

## Contrato

```python
select_anchored_medoids(
    points,
    ids,
    fixed_ids,
    candidate_ids,
    capacity,
    algorithm="greedy_swap",
    metric="euclidean",
    background_backend="numpy",
    reuse_distances=False,
    bounded_background=False,
)
```

`capacity` es la capacidad total e incluye los centros fijos. Debe ser un entero positivo y no puede ser menor que el número fijo. La capacidad variable puede ser cero. En ese caso se calcula el objetivo del conjunto fijo. E vacío produce selección vacía. Con clientes y sin ningún centro autorizado se produce un error. Si sobran plazas se conservan todos los candidatos.

La salida conserva los IDs fijos, variables y retenidos, así como sus índices en la entrada original. Registra objetivo, algoritmo, métrica, estado de búsqueda, tipos, versiones, bloques, bytes estimados y pares de distancia. `enumerated_restricted` identifica el oráculo limitado a A y F. `one_swap_local_restricted` identifica ausencia de mejora unitaria en la búsqueda completada. `swap_limit` identifica un resultado parcial. La enumeración flotante compara costes calculados en coma flotante.

El fondo `numpy` usa la referencia general estable. La selección variable reutiliza el greedy con intercambios y el oráculo existentes. Sin fondo, el selector anterior conserva su recorrido. La capacidad, la elección de A y F, frecuencia y representación deben formar parte de la identidad del consumidor.

## Precisión y recursos

`scipy_cdist_fp32_exploratory` utiliza [`scipy.spatial.distance.cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) con distancia euclídea, coordenadas FP32 finitas y cálculo FP64. Es un backend exploratorio y no un reemplazo equivalente de NumPy. Admite hasta 4.096 dimensiones. La conversión a FP64 ocurre antes de la resta y el cuadrado. No acepta FP64 arbitrario ni lo redondea a FP32. La referencia NumPy conserva ese rango general y la opción L1 entera. La selección variable sigue usando NumPy.

El intervalo de FP32 y la dimensión acotada evitan el desbordamiento y subdesbordamiento del cuadrado en FP64. No eliminan el redondeo. El contraste utiliza rtol=1e-14 y atol=0, y comprueba representantes y ceros por separado. El tipo de coordenadas, backend y versiones aparecen en el resultado y deben entrar en la identidad de ejecución. Si se solicita SciPy sin tenerlo instalado, la función falla sin cambiar de backend. SciPy está declarado en el extra `research` del proyecto.

Existe un contraejemplo de IDs distintos con el mismo objetivo observado. Para E formado por `a=c=(-100,-100,-100)`, `b=d=(0,0,0)` y `e=(0.3654440641403198,0.4127326011657715,0.4308210015296936)` en FP32, con centros fijos a/b, candidatos c/d y capacidad 3, NumPy conserva a/b/c y SciPy exploratorio a/b/d. Ambos devuelven `0.6996458385779952`. El fondo y las distancias variables redondean de forma distinta. No se introduce una tolerancia de desempate para ocultarlo. NumPy sigue siendo la referencia predeterminada para el banco y el nombre SciPy sin calificación exploratoria se rechaza.

El fondo se recorre por bloques en ambos ejes. Los tamaños solicitados son 256 clientes y 256 centros fijos, con 16 candidatos variables. El presupuesto predeterminado de 64 MiB estima copias normalizadas, IDs, índices, partición de E, vectores de fondo/vecinos y temporales simultáneos. No certifica el RSS o los espacios internos de biblioteca.

El límite predeterminado de 50 millones de pares es compartido por fondo y selección. La enumeración admite como máximo 10.000 combinaciones. Agotar un presupuesto produce `MedoidBudgetExceeded`, sin devolver un supuesto óptimo ni activar otra retención. El máximo de intercambios es independiente y su agotamiento se declara mediante `swap_limit`.

## Opciones que conservan los bits

Las dos opciones, desactivadas por defecto, reducen el trabajo de la ruta `numpy` sin cambiar ninguna distancia. La salida registra cuáles se aplicaron (`reused_distances` y `bounded_background`). Su memoria se suma a `estimated_peak_bytes` solo si cabe en `max_working_bytes` sin cambiar los bloques, primero la de la cota y después la de la tabla. La que no cabe se descarta y la selección sigue la referencia. Los pares se cuentan como antes, así que los presupuestos se agotan en el mismo punto.

`reuse_distances` guarda la distancia de cada bloque de clientes a un candidato variable la primera vez que se calcula, con la misma resta y la misma cadena de `hypot` o valor absoluto por dimensión, y la copia en las evaluaciones siguientes del greedy y de los intercambios. Ocupa 8 bytes por par cliente-candidato.

`bounded_background` solo se aplica con distancia euclídea, coordenadas FP32 y fondo `numpy`. Calcula el fondo b(e) como el mínimo exacto de las distancias de referencia, pero descarta los pares que una cota inferior excluye. La cota sale de un producto de matrices sobre coordenadas centradas en la media de los fijos. Con u = 2⁻⁵³ y D dimensiones, centrar en FP64 mueve cada coordenada como mucho u|x'|, el cuadrado s = q + p − 2x'·y' tiene un error de como mucho (D + 2)u(‖x'‖ + ‖y'‖)² con cualquier orden de suma, y la cadena de referencia, una resta por coordenada y D llamadas a `hypot` (una ulp como mucho en glibc, cuatro en la cuenta), no baja de la distancia exacta por más de un factor (1 − (8D + 1)u). La cota resta 4(D + 4)u(√q + √p)² al cuadrado, 4u(√q + √p) a la raíz y la multiplica por (1 − (8D + 2048)u), que deja 2047u para su propio redondeo con cualquier D. Un factor fijo como el (1 − 2048u) de la primera versión solo cubría hasta 255 dimensiones. Un par se descarta si su cota supera una distancia de referencia ya calculada del mismo cliente, así que el mínimo no cambia. Si un bloque conserva más de un octavo de sus pares, se calcula entero como en la referencia. La deducción supone el error de `hypot` documentado por glibc. Las pruebas la contrastan en casos adversos, pero no la sustituyen por una demostración formal.

La memoria de la cota cuenta las copias centradas y traspuestas, las normas, tres bloques FP64 y una máscara reservados una vez, los índices y buffers de los pares supervivientes (como mucho un octavo de un bloque), los vectores por fila y los temporales de NumPy (el iterador de una ufunc con operandos difundidos reserva hasta `np.getbufsize()` elementos por operando, y `argmin` copia un bloque parcial). No cuenta las reservas internas de OpenBLAS, que no dependen de la selección: un área de trabajo por proceso desde el primer producto de matrices y, cuando reparte un producto entre hilos, una tabla de tareas durante la llamada. En la forma de los lectores, con OpenBLAS 0.3.34 y dos hilos, el área ocupa 32,3 MiB de memoria virtual y unos 0,9 MB residentes, y la tabla 512 KiB ([medida](../../../reports/engineering/campaign-kernels-wiring-20261010/anchored-medoid-memory.json)). El producto usa los hilos de la BLAS que fije el proceso. La campaña fija `OMP_NUM_THREADS=2`, que OpenBLAS también respeta.

`tests/cm/test_anchored_medoids.py` compara todos los campos de la salida con la referencia en problemas con intercambios, empates, coordenadas enteras L1 y siete geometrías adversas (normal, duplicados, nube lejana y estrecha, escalas mezcladas, rejilla, constante y valores cercanos al máximo de FP32), con cada opción y con las dos. También comprueba que la tabla calcula cada par variable una sola vez, que la cota calcula menos de un 2 % de los pares del fondo en las geometrías favorables y el bloque entero cuando no poda, los presupuestos, la memoria escasa, la paridad con 300 dimensiones y que el pico rastreado no supera la estimación en un problema grande, uno mínimo, uno con un solo centro fijo, uno sin poda y uno de 300 dimensiones. `tests/memory/test_retention_bank.py` repite la comparación con el banco nativo en cuatro propuestas. Las medidas del recorrido real están en el [informe del 10 de octubre](../../../reports/engineering/campaign-kernels-wiring-20261010/README.md#medoids).

## Comprobaciones

La [revisión independiente](../../../reports/research/anchored-retention-verification-20261008.json) pasa 128 pruebas, incluidas siete sondas privadas y 60 instancias contrastadas con enumeración del objetivo completo. La discrepancia discreta entre backends queda identificada como un límite del modo exploratorio. La conexión con el banco sigue pendiente.

La suite de mecanismos suma 121 pruebas CPU. Incluye objetivo completo frente a enumeración independiente, clientes fuera de F, fondo cero, centros fijos, capacidades, permutaciones, duplicados, extremos FP32, rango general FP64, tipos incompatibles, presupuestos, memoria de IDs Unicode y el contraejemplo de divergencia entre backends. Ocho mutaciones dirigidas fueron detectadas. En 48 casos, todos los campos de salida del selector anterior coincidieron exactamente con la revisión integrada previa.

Coverage.py 7.16.2 y Radon 6.0.1 registraron 99,26 % de sentencias y 98,15 % de ramas del módulo nuevo. Su mayor complejidad ciclomática es 28, con CRAP 28 al quedar cubiertas sus sentencias. La convención es `CCN² (1 - cobertura_sentencias)³ + CCN`. Estas medidas son diagnósticos y no demuestran ausencia de defectos.

El [benchmark de la API pública](../../../scripts/benchmark_anchored_medoids.py) incluye validación, IDs, conversión, fondo y selección. No mide codec, elegibilidad temporal, snapshots, persistencia ni un modelo. El banco nativo sigue teniendo su contrato propio 64×64 y capacidad máxima de 1.024. Las formas de 8.192 del selector no acreditan esa capacidad en el banco.

La medida CPU del 8 de octubre de 2026 usa 8.184 centros fijos, 16 candidatos variables, capacidad 8.192 y 64 coordenadas FP32. Cada forma tuvo un calentamiento y tres repeticiones por backend, alternando su orden. En esas geometrías concretas coincidieron los representantes y el objetivo respetó la tolerancia. El contraejemplo anterior impide extender esa coincidencia a otras entradas. Las medidas preceden al etiquetado explícito como exploratorio, que no altera el cálculo.

| Nuevos clientes | Mediana NumPy | Mediana SciPy | Cociente observado |
| ---: | ---: | ---: | ---: |
| 128 | 0,5842 s | 0,05407 s | 10,81× |
| 512 | 2,1915 s | 0,17050 s | 12,85× |
| 2.048 | 8,6395 s | 0,63486 s | 13,61× |

Para 512 clientes nuevos, el máximo rastreado fue 11.032.664 bytes con NumPy y 10.222.821 con SciPy, frente a 24.236.920 bytes estimados. El comando completo tardó 52,19 s y alcanzó 84.996 KiB RSS. Se utilizó AMD Ryzen 9 8945HS, Python 3.12.14, NumPy 2.5.3 y SciPy 1.18.1, con dos hilos, cuota CPU 200 % y límite de proceso de 1 GiB. Otras aplicaciones siguieron activas. Estos cocientes pertenecen al selector medido y no al banco completo ni al modelo.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync --offline pytest -q tests/cm
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync --offline python scripts/benchmark_anchored_medoids.py
```
