# Memoria asociativa escrita con resultados maduros

`memory/associative_memory.py` implementa el comparador B6 de la [matriz de experimentos](../research/experiment-matrix.md): una matriz A de clave por valor que solo cambia al aplicar resultados ya maduros. Ofrece las dos reglas derivadas en las [condiciones matemáticas de memoria](../research/memory-mathematics.md), la regla delta con olvido escalar y la escritura proximal de una cohorte. La [revisión de integración](../research/system-integration.md) las sitúa en la actualización de estado maduro como alternativas identificadas.

No tiene parámetros entrenables y no se ha usado con datos del corpus. `MatureCorrection` la conecta con `FinancialSession` como corrección escalar de Titans-MAC sin banco episódico: la emisión es la predicción del núcleo más la lectura de A de la generación anterior y la escritura usa la etiqueta madura menos la predicción del núcleo. Ese contrato, sus pruebas y el control de clave constante están en [MARS-TITAN con ampliaciones](mars-titan-extensions.md#corrección-asociativa-b6-en-la-sesión). Su punto de inserción y sus controles se especifican en la [correspondencia de modificaciones](../research/titans-mac-architecture.md#correspondencia-de-las-modificaciones-de-integración).

## Qué no es

La memoria neuronal de Titans-MAC escribe con su sorpresa asociativa, calculada desde entradas disponibles, y usa momentum y puertas dependientes de la entrada sobre una red de dos capas. El banco episódico conserva episodios recuperables con su identidad. A no sustituye a ninguno de los dos. Es una aplicación lineal de clave a valor escrita con etiquetas maduras, que comprime en d·m números lo que el banco guarda como episodios.

## Contrato

| Elemento | Definición |
| --- | --- |
| Estado | A ∈ ℝ^{d×m} en FP64 y CPU, contador de resultados aplicados y cursor del último aplicado. Empieza en cero. |
| Claves | Matriz FP64 con norma euclídea por fila no mayor que uno, con un margen de 1e-6 para el redondeo de claves normalizadas en FP32. Deben proceder del mismo espacio estable con el que se leerá. |
| Valores | Matriz FP64 finita. No se recortan por magnitud, porque un evento raro auténtico puede contener señal. |
| Tiempos | `decision_at < available_at ≤ cutoff` para cada resultado. Una etiqueta inmadura hace fallar la escritura completa. |
| Orden | Canónico por disponibilidad de la etiqueta, decisión e identificador. No depende del orden de carga. |
| Cursor | La clave canónica del primer resultado de una escritura debe superar la del último aplicado. Una entrega repetida o fuera de orden se rechaza. |
| Cohorte vacía | Devuelve la misma memoria, sin aplicar olvido. Olvidar por el paso del reloj sería otra variante. |
| Escritura | Devuelve una memoria nueva. La anterior queda intacta también cuando la escritura falla. |
| Lectura | `read(keys)` devuelve `keys @ A`, es decir Aᵀk por fila, sin modificar estado. La memoria vacía lee cero. |
| Recuperación | `export` conserva identidad, matriz, huella SHA-256 de sus bytes, contador y cursor. `restore` rechaza otra configuración, una matriz alterada o un cursor incoherente. |

La identidad incluye la regla, las dimensiones, η, λ, el orden, la política del cursor, el tratamiento de la cohorte vacía y el método de resolución.

## Regla delta

Para cada resultado en orden canónico,

$$
A^+=(1-\lambda)A+\eta k(v-A^\top k)^\top.
$$

La diferencia entre dos estados con las mismas entradas evoluciona con $(1-\lambda)I-\eta kk^\top$. Con $\|k\|\le1$, la condición $\eta\le2-\lambda$ basta para que no se expanda en ninguna escritura. La configuración la exige y cada escritura comprueba además $\eta\|k\|^2\le2-\lambda$ con la norma real de sus claves. `contraction_bound()` devuelve $\max(1-\lambda,|1-\lambda-\eta|)$, que acota la norma espectral de ese operador para cualquier clave admitida. El contraejemplo de la derivación, λ = 0,1 y η = 1,95, se rechaza al construir la configuración.

Estas cotas comparan dos estados sometidos a las mismas entradas. No acotan por sí solas el estado absoluto, que necesita además una cota de los valores, ni dicen nada del error predictivo.

## Escritura proximal

Con pesos declarados $w_j\ge0$ que suman uno sobre la cohorte completa, $G=\sum_jw_jk_jk_j^\top$, $H=\sum_jw_jk_jv_j^\top$ y retención $\rho=1-\lambda$,

$$
(I+\eta G)A^+=\rho A+\eta H.
$$

El sistema es definido positivo para todo η finito no negativo. Se factoriza con `torch.linalg.cholesky_ex` y se resuelve con `torch.cholesky_solve`, ambos sobre LAPACK, sin formar la inversa. Si la factorización fallara en FP64, la escritura se rechaza. La diferencia entre dos estados se contrae al menos por $\rho/(1+\eta\lambda_{min}(G))$ en norma de Frobenius. Con claves de rango deficiente y ρ = 1, las direcciones ortogonales a las claves no cambian.

Una llamada recibe la cohorte completa de un evento, hasta 8.192 resultados. Dividirla en microlotes exigiría acumular G y H con los pesos globales y resolver una sola vez. No se implementa esa acumulación porque una llamada ya cubre el límite de la cohorte. Resolver por microlotes o renormalizar sus pesos cambiaría el algoritmo.

La regla proximal no es una versión numéricamente idéntica de la explícita. Su desarrollo para η pequeño es $\rho A+\eta(H-\rho GA)+O(\eta^2)$, mientras que la regla explícita análoga resta $\eta GA$. Una prueba comprueba que el resto respecto a ese primer orden decrece como η².

## Comprobaciones

`tests/memory/test_associative_memory.py` contiene 28 casos sin pasos de optimizador ni recorridos largos que ajusten una señal sintética. Comparan cada escritura con su ecuación o con su cota: igualdad con la fórmula de rango uno, operador de diferencias y su norma espectral en cuatro combinaciones de λ y η, rechazo del contraejemplo, no conmutación de dos escrituras (2,5 frente a 2) decidida por la disponibilidad y no por el orden de carga, desempates, rechazo de entregas repetidas o inmaduras, cohorte vacía, ecuaciones normales de la regla proximal, invariancia al orden de carga, contracción con claves ortonormales, claves de rango deficiente, diferencia de segundo orden con la regla explícita, pesos, lectura pura y recuperación.

La mutación dirigida alteró en memoria 15 partes del módulo, entre ellas el olvido, el signo del residual, la condición de estabilidad, la retención y los pesos proximales, el orden canónico, el cursor, la madurez, la norma de las claves, la cohorte vacía, la huella, la pureza de la lectura y la cota proximal. Las 15 provocaron fallos. No es una puntuación exhaustiva de todas las mutaciones posibles.

## Coste medido

Medida del 9 de octubre de 2026 en un AMD Ryzen 9 8945HS, Python 3.12.14, PyTorch 2.14.0+cu130 en CPU con dos hilos, d = 64 y m = 1, con tres calentamientos y veinte repeticiones. La carga media del sistema rondaba 9 por otros trabajos, que no se detuvieron.

| Resultados por evento | Delta, escritura p50 (ms) | Delta, p95 (ms) | Proximal, escritura p50 (ms) | Proximal, p95 (ms) | Lectura p50 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0,319 | 0,462 | 0,218 | 0,240 | 0,018 a 0,026 |
| 256 | 3,651 | 5,373 | 0,456 | 0,503 | 0,061 a 0,064 |
| 1.024 | 16,416 | 17,306 | 1,068 | 1,455 | 0,200 a 0,248 |
| 8.192 | 125,107 | 177,369 | 12,784 | 13,520 | 1,400 a 1,451 |

La regla delta es secuencial por definición y su coste está dominado por el despacho de cada escritura de rango uno, unos 16 µs por resultado. No se optimiza porque no hay evidencia de que limite un recorrido: el resto de un evento de la sesión no se ha medido junto a ella. Si llegara a pesar en el recorrido completo, la primera alternativa sería agrupar las operaciones de cada evento antes de considerar código nativo. No se midieron energía ni memoria del proceso y no se usó GPU.

## Pendiente

La emisión de B6 en el recorrido cronológico por ventanas, la elección de η y λ en desarrollo, el coste medido dentro de la sesión y cualquier comparación predictiva siguen pendientes. El bloqueo de aprendizaje impide elegir esos valores con datos y ejecutar la comparación. RLS con olvido, mencionado como control posible en la revisión de ampliaciones, no se implementa.
