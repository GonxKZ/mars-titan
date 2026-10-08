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
