# Puente de evidencia con recurrencia no lineal

La opción `shared_evidence_bridge_nonlinear_v1` está implementada y desactivada por defecto. Añade J = 1, 2 o 4 pasos síncronos con tanh y reutiliza los pesos de la región receptora. Las vistas proceden de un único snapshot y de una selección común de episodios. No añade parámetros de recurrencia ni bancos regionales persistentes. J es distinto de las lecturas K y de las actualizaciones de pesos rápidos de Titans.

La [especificación](../../../docs/experiments/nonlinear-evidence-bridge-v1.md) describe la actualización, la identidad y la reserva de trabajo antes del primer MAC. El [script histórico de coste](https://github.com/GonxKZ/mars-titan/blob/d94aae96504997f84cc3c626557f492d562f79c0/benchmarks/nonlinear_bridge_cuda.py) es una copia idéntica del ejecutado. El archivo actual incorpora después otra batería de intervenciones y tiene otra huella. Las pruebas y el perfil principal preceden al commit `3c2d0b639caf48552d86504178ffad1aa119220d` y quedan ligados a los mismos bytes mediante manifiestos y SHA-256. El diagnóstico de operadores se ejecutó después y se documenta por separado.

## Verificación técnica

La selección final reúne 692 IDs únicos aprobados. Las cuatro selecciones son disjuntas y 158 IDs tienen el parámetro `cuda:0`. No hubo fallos, errores ni omisiones en esas selecciones. Los tiempos siguientes son externos del proceso supervisado y no son latencias del modelo.

| Selección | Casos | Tiempo externo (s) | Fallos / errores / omitidos |
| --- | ---: | ---: | --- |
| `final-core` | 553 | 107,898 | 0 / 0 / 0 |
| `final-native-focus` | 7 | 117,317 | 0 / 0 / 0 |
| `final-factory-and-cpu-m3` | 106 | 50,442 | 0 / 0 / 0 |
| `final-collection-regression` | 26 | 61,481 | 0 / 0 / 0 |

Una comprobación posterior a aplicar los mismos bytes repitió cinco de esos IDs, con 10,419 s externos incluyendo la comprobación de CUDA y las pruebas. No añade cinco casos al total. La selección no es la suite completa. Los ocho casos nativos de variantes excluidos expresamente tampoco se presentan como ejecutados.

Se conservan los intentos anteriores. El primer contraste del componente registró tres fallos: dos llamadas de fixture sin el argumento `device` y una comparación excesivamente exacta. La integración tuvo sus propios contrastes iniciales fallidos. El intento `final-integrated` agotó 180 s, terminó con código −9 y quedó incompleto. Sus casos y su cobertura no se suman como una pasada completa. [verification.json](verification.json) conserva los IDs, las selecciones, los errores anteriores y las huellas de los recibos.

La configuración desactivada conserva el fixture anterior, sus tensores y el RNG. El modo activo tiene identidad propia y rechaza catálogos o checkpoints incompatibles antes de copiar parámetros. La reserva conjunta cuenta ambos contextos y la emisión final. El límite de 1.216 actualizaciones corresponde a R = 8, J = 4 y 38 lecturas por fila del evento. Es un contador de trabajo regional, no una cuenta de FLOPs.

## Mutación y cobertura

Las diez mutaciones del componente son evidencia de la fase anterior, sobre sus fuentes y pruebas selladas. Incluyen nueve de la primera tanda y una adicional que corta el gradiente del grafo. Las nueve primeras no se repitieron para añadir esa décima y tampoco se simula una repetición actual.

La integración detectó nueve mutantes distintos. La primera tanda clasificó siete como detectados y uno como error. Ese resultado original se conserva. La revisión identificó una `AssertionError` después de copiar parámetros antes de rechazar un checkpoint. Una repetición focal confirmó ese mutante y añadió el que incorpora `recurrence:null` a la identidad desactivada. Pasaron los controles inicial y restaurado de 48 casos en la primera tanda y de dos en la focal. No se repitieron los otros siete mutantes. [mutations.json](mutations.json) separa ambas fases, sus fuentes y sus hashes.

coverage.py 7.16.1 registra 1.904/2.045 sentencias y 727/862 ramas en once módulos. Las cinco funciones nuevas de admisión y conciliación enumeradas en [quality.json](quality.json) cubren 29/29 sentencias y 14/14 ramas. Radon 6.0.1 y la convención `C² × (1 − cobertura de sentencias)³ + C` dan un máximo CRAP de 62,416 para la función previa `MechanismBinding._check_quantiles`, con 12/22 sentencias cubiertas. Estas cifras no son cobertura global ni una garantía de corrección.

## Coste del forward del puente

El perfil completó 40 combinaciones en orden normal e inverso, 80 registros de condición en total, en 28,889975521 s externos. Usa FP32 en `cuda:0`, referencia CPU FP64 con los mismos parámetros e inputs FP32 elevados exactamente, H = 32, ocho vecinos y ganancia 0,25. Las tolerancias son `atol=2e-6` y `rtol=1e-5`. Coincidieron todos los IDs de candidatos y las salidas dentro de tolerancia en las 80 condiciones. La referencia numérica usa el mismo recorrido en otra precisión, no otro algoritmo independiente.

Cada condición tiene tres calentamientos y diez medidas. El orden de variantes se intercala con semilla 20261023 y se invierte en la segunda pasada. La semilla de datos es 20261022 y la de configuración 42. `cycles0` omite los pasos de comunicación previos y `cycles2` conserva dos ciclos lineales. J1, J2 y J4 activan el nuevo recorrido con `cycles=0`. El reloj de host rodea el forward completo con sincronización CUDA antes y después, incluida la selección de candidatos. No mide solo kernels ni el predictor completo.

La medida se hizo en una NVIDIA GeForce RTX 4070 Laptop GPU, controlador 595.91.07, PyTorch 2.14.0+cu130, CUDA 13.0, Python 3.12.14 y un hilo CPU. TF32 y autocast estaban desactivados y los algoritmos deterministas activados. La GPU tenía otras aplicaciones residentes, que no se pausaron. Las dos pasadas se muestran por separado y no se escoge la más rápida.

Medianas en milisegundos, en formato normal / inversa. B es lote, R regiones y N episodios.

| B × R × N | cycles0 | cycles2 | J1 | J2 | J4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 × 3 × 8 | 4,001 / 4,054 | 4,026 / 4,006 | 4,397 / 4,150 | 6,104 / 6,238 | 6,490 / 6,641 |
| 1 × 3 × 1024 | 4,027 / 4,004 | 4,043 / 4,005 | 4,122 / 4,275 | 6,302 / 5,943 | 6,428 / 6,605 |
| 1 × 8 × 8 | 6,010 / 3,964 | 4,605 / 3,950 | 4,465 / 4,073 | 5,802 / 5,899 | 6,461 / 6,313 |
| 1 × 8 × 1024 | 6,059 / 4,009 | 6,041 / 3,929 | 7,125 / 4,397 | 8,486 / 5,676 | 10,196 / 6,383 |
| 256 × 3 × 8 | 4,052 / 4,049 | 4,024 / 4,074 | 4,048 / 4,119 | 6,365 / 4,985 | 6,423 / 6,493 |
| 256 × 3 × 1024 | 4,176 / 4,205 | 4,253 / 4,250 | 4,283 / 4,295 | 4,775 / 5,181 | 6,402 / 6,653 |
| 256 × 8 × 8 | 4,079 / 4,068 | 4,079 / 4,257 | 4,464 / 4,385 | 6,352 / 6,391 | 6,507 / 6,850 |
| 256 × 8 × 1024 | 4,219 / 4,315 | 4,252 / 4,560 | 4,317 / 4,391 | 4,392 / 5,329 | 6,722 / 6,739 |

Para B = 256, R = 8 y N = 1.024, `cycles0` → J4 pasa de 4,219272 a 6,721836 ms en orden normal y de 4,314631 a 6,739499 ms en orden inverso. Es coste añadido del componente, no una mejora de rendimiento. La dispersión siguiente se limita a esa forma. σ es la desviación estándar muestral. p95 y p99 interpolan solo diez muestras por celda y son descriptivos, no estimaciones de latencia de cola.

| Modo | Normal: mediana / σ (ms) | Normal: p95 / p99 (ms) | Inversa: mediana / σ (ms) | Inversa: p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| cycles0 | 4,219 / 0,430 | 5,120 / 5,258 | 4,315 / 1,302 | 7,220 / 7,962 |
| cycles2 | 4,252 / 0,208 | 4,303 / 4,312 | 4,560 / 1,004 | 6,599 / 6,703 |
| j1 | 4,317 / 0,473 | 5,347 / 5,355 | 4,391 / 0,954 | 6,574 / 6,613 |
| j2 | 4,392 / 0,789 | 6,168 / 6,617 | 5,329 / 0,988 | 6,741 / 6,809 |
| j4 | 6,722 / 0,788 | 7,512 / 7,709 | 6,739 / 0,568 | 7,802 / 7,833 |

En esa misma forma, cada modo tiene 147.968 bytes de parámetros. El montaje mantiene cinco instancias instrumentales residentes. Los picos siguientes son del allocator de PyTorch y coinciden entre ambas pasadas. El extra se calcula frente a la asignación anterior a cada forward.

| Modo | Pico allocated (bytes) | Extra allocated (bytes) | Pico reserved (bytes) |
| --- | ---: | ---: | ---: |
| cycles0 | 41657344 | 6517760 | 62914560 |
| cycles2 | 41657344 | 6517760 | 62914560 |
| j1 | 41891328 | 6751744 | 62914560 |
| j2 | 42416640 | 7277056 | 62914560 |
| j4 | 43467264 | 8327680 | 62914560 |

El extra allocated crece 1.809.920 bytes entre `cycles0` y J4. No es el pico total de VRAM física por proceso, que no se muestreó. El RSS máximo fue 1.440.034.816 bytes en la pasada normal y 1.440.276.480 en la inversa. Incluye runtime, referencias CPU e instrumentación y no aísla la RAM de un único puente de producción.

La transferencia H2D inicial contable es de 1.577.792 bytes para esa forma. Incluye las cinco copias de parámetros y ocurre fuera del tiempo del forward. Las fechas del snapshot permanecen en CPU. La salida suma 458.752 bytes lógicos y conserva 2.539.520 bytes en sus almacenamientos únicos. Estos recuentos no son trazas de tráfico físico. La estimación de memoria de trabajo del contrato se guarda separada de los picos observados. No se midió energía ni se obtuvo un contador independiente de lanzamientos.

[benchmark.json](benchmark.json) conserva las 80 condiciones, los calentamientos, las diez muestras, los dos órdenes, las identidades, semillas, errores, caudal, bytes y memoria. No se selecciona J científicamente ni se atribuyen métricas financieras. La condición suficiente del Jacobiano expuesta en la especificación tiene hipótesis que el módulo no impone y no está formalizada en Lean. La campaña completa y el test final permanecen cerrados.

El diagnóstico posterior con CUPTI registró cinco forwards de `cycles0` y cinco de J4. La categoría `kernel` suma 2.940,622 y 4.573,534 µs, con 500 y 820 eventos. Las cinco anotaciones GPU y las quince copias por modo se cuentan aparte. Aparecen GEMM, reducciones AbsMax y ordenación radix. Estas sumas instrumentadas no son latencia representativa, un contador independiente de lanzamientos ni una prueba de cuello de botella. La primera clasificación mezclaba actividades `DeviceType.CUDA` con anotaciones y se conserva por su hash. Se corrigió sin repetir la carga, reclasificando las mismas trazas. [operator-diagnostic.json](operator-diagnostic.json) guarda los datos corregidos. Sus 10,924124623 s externos, sumados una vez a los 28,889975521 s del perfil principal, dan 39,814100144 s del presupuesto técnico de 360 s. No se ha implementado un kernel ni adoptado una optimización nueva.

Para reproducir el perfil principal, el guion debe copiarse primero a una carpeta externa al repositorio. Su salida debe quedar bajo la carpeta del guion y fuera de `source_root`. Con la guarda local existente configurada y desde el repositorio:

```bash
profile_dir=$(mktemp -d)
git show d94aae96504997f84cc3c626557f492d562f79c0:benchmarks/nonlinear_bridge_cuda.py > "$profile_dir/profile_bridge.py"
uv run --no-sync python "$profile_dir/profile_bridge.py" --run \
  --source-root "$PWD" --output "$profile_dir/run-1"
```

La copia histórica conserva el SHA-256 `00d7dacf799fd65e75120495e030e5613f5bb501ac1ce23f09174bf59b06d62a`. Para repetir aquella revisión también se necesita su árbol fuente y entorno, no solo el script. Las huellas y tamaños de los originales externos están en [external-artifacts.json](external-artifacts.json).
