# Implementaciones públicas de Titans y paridad del núcleo

El núcleo Titans-MAC de [`src/mars_titan/models/titans/`](../../src/mars_titan/models/titans/) se escribió a partir de las ecuaciones del artículo. Este documento revisa qué código público existe, si alguno es de los autores y hasta qué punto el núcleo del proyecto coincide con él. La revisión se hizo el 9 de octubre de 2026 y se sigue en [#434](https://github.com/GonxKZ/mars-titan/issues/434).

Las conclusiones, con su evidencia en las secciones siguientes, son cuatro:

- No hay código oficial de Titans ni de los trabajos posteriores del mismo grupo. El repositorio `ABehrouz/Titans` del primer autor existe, pero está vacío.
- La referencia pública más usada es `lucidrains/titans-pytorch`, no oficial y con licencia MIT. Configurada con la recurrencia del artículo, su memoria neuronal coincide con la del proyecto en lecturas, pesos rápidos, momentum y gradientes exteriores, con diferencias de 10⁻¹² o menores en FP64. En FP32 coinciden sin residual. Con residual y LayerNorm la tolerancia FP32 declarada no se cumple, y la diferencia es del mismo orden que el redondeo de cada implementación frente a su evaluación FP64.
- El MAC de esa referencia no sigue las ecuaciones 21 a 25 del artículo. No hay una implementación pública del MAC que coincida con el texto y permita una paridad numérica directa.
- La revisión encontró una desviación propia sin declarar. El artículo usa SiLU al calcular consultas, claves y valores, y el núcleo no la aplica.

## Código oficial

No se ha encontrado código de los autores ni de Google. Estas son las comprobaciones, todas del 9 de octubre de 2026:

- [arXiv 2501.00663](https://arxiv.org/abs/2501.00663) solo tiene la versión v1, del 31 de diciembre de 2024, sin enlace a código. La conclusión del preprint dice que Titans está implementado en PyTorch y JAX y que el código se publicará «soon».
- Las [actas de NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html) no tienen material suplementario ni enlace a código. Su lista de comprobación responde «Yes» a la pregunta de acceso abierto al código y lo justifica con un plan de publicarlo en GitHub tras la publicación.
- La búsqueda pública de OpenReview devuelve el envío `8GjSf9Rh7Z` de NeurIPS 2025 sin campos de material suplementario ni de código.
- La entrada [Titans + MIRAS](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) del blog de Google Research, del 4 de diciembre de 2025, enlaza los artículos, BABILong y la organización `google-research`, pero ningún repositorio de Titans.
- La [página del primer autor](https://alibehrouz.com/) lista Titans, MIRAS, ATLAS y Nested Learning sin enlaces a código. La cuenta de GitHub `ABehrouz`, con nombre Ali Behrouz y el repositorio de la web `abehrouz.github.io` que redirige a esa página, tiene un repositorio `Titans` creado el 9 de abril de 2025 que la API de GitHub describe como vacío.
- Las organizaciones `google` (2.919 repositorios), `google-research` (364) y `google-deepmind` (409) no tienen repositorios con esos nombres. La raíz de `google-research/google-research` tampoco tiene una carpeta de Titans.
- [PMLR v306](https://proceedings.mlr.press/v306/behrouz26b.html), donde se publicó ATLAS, no enlaza código.
- [Titans Revisited](https://arxiv.org/abs/2510.09551), una reimplementación de otro grupo, atribuye parte de su dificultad a la falta de código público y no enlaza código propio.

El [inventario](../../reports/engineering/titans-reference-parity-20261009/inventory.json) conserva estas comprobaciones junto con cada repositorio revisado.

## Inventario de implementaciones no oficiales

Todas las implementaciones encontradas son no oficiales. La tabla recoge las que se revisaron leyendo el código. El commit fijado es el `HEAD` de la rama por defecto el 9 de octubre de 2026.

| Repositorio | Licencia | Commit y fecha | Variantes | Revisión | Fidelidad y estado |
| --- | --- | --- | --- | --- | --- |
| [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) | MIT | `1d40c445`, 2026-07-13 | Memoria neuronal y MAC como modelo de lenguaje | Código completo leído y ejecutado en un entorno aislado | La memoria admite la recurrencia exacta del artículo, aunque no por defecto. El MAC difiere de las ecuaciones. 1.990 estrellas. Las pruebas comprueban formas, encadenamiento de estados e inferencia, sin compararlas con una transcripción de las ecuaciones |
| [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), `fla/ops/titans` | MIT | `855e5026`, 2026-10-09 | Memoria lineal con LayerNorm y residual al estilo TTT | Operación leída y ejecutada | La derivada de LayerNorm activa no coincide con autograd. Su única prueba está desactivada con `skipif(True, reason='FIXME')` |
| [omkarghugarkar007/titans-pytorch-unofficial-implementation](https://github.com/omkarghugarkar007/titans-pytorch-unofficial-implementation) | MIT | `6e4728b1`, 2026-09-06 | Memoria, MAC, MAG y MAL | Memoria, MAC y desviaciones leídas | Forma cerrada por bloques con prueba frente a la recurrencia literal. Declara once desviaciones, entre ellas SiLU y LayerNorm entre capas de la memoria, salida acotada y lectura de `M_{t−1}` en la puerta final. Una estrella |
| [Aedelon/titans-pytorch-mlx](https://github.com/Aedelon/titans-pytorch-mlx) | Apache-2.0 | `0dc7a4d2`, 2026-01-16 | Memoria, MAC, MAG y MAL | Memoria leída | Una sola actualización por llamada, tasas promediadas a un escalar sobre secuencia y lote, memoria compartida entre flujos y estado devuelto sin grafo |
| [mindemory/jax_titans](https://github.com/mindemory/jax_titans) | Apache-2.0 | `283e416a`, 2026-10-05 | MAC en JAX | README y núcleo leídos | Escritura secuencial por token. Normaliza los valores, enmascara `h` de forma causal y usa `y · sigmoid(RMSNorm(M(q)))` como puerta. Creado el 27 de septiembre de 2026 |
| [mahdi-shafiei/titans-flax](https://github.com/mahdi-shafiei/titans-flax) | MIT | `b0d1170d`, 2026-01-23 | Memoria, MAC, MAG y MAL en Flax | Memoria inspeccionada | Tasas como escalares aprendidos sin dependencia de la entrada y una actualización con claves promediadas |

Otros once repositorios se revisaron leyendo su README y su núcleo por la API de GitHub, sin descargarlos. Ninguno sirve como referencia numérica. Los motivos van desde no contener el modelo (`ai-in-pm/Titans---Learning-to-Memorize-at-Test-Time`, el segundo más popular, es una demostración que llama a modelos de lenguaje externos) hasta errores de signo (`aheschl1/titans`), gradientes sin calcular (`hoshuaclawdbot/titans-cuda`) o derivaciones tempranas de lucidrains sin atribución (`Yuan-ManX/Titans-PyTorch`). El inventario JSON conserva su commit, licencia y motivo.

Para los trabajos posteriores tampoco hay código oficial. Existen implementaciones no oficiales de ATLAS (`danielquintas8/atlas-torch`, derivada de lucidrains), MIRAS (`jonlukewatts/titans-miras`) y Nested Learning (`kmccleary3301/nested_learning`, la más seguida). Ninguna se usa aquí, porque el alcance es Titans-MAC.

## Descarga y aislamiento

Las referencias se descargaron fuera del repositorio, cada una en un directorio vacío con nombre `<repositorio>@<commit>` y con los hooks de Git desactivados. Nada se ejecutó desde esos directorios. El arnés vive en el repositorio y se lanza desde otro directorio con un entorno `uv` propio, separado del entorno del proyecto, con `python -I` y las rutas como argumentos. El entorno usa el mismo `torch 2.14.0+cu130` del proyecto y las dependencias de lucidrains resueltas con `--exclude-newer 2026-10-01`. Las versiones exactas quedan en cada recibo.

El arnés comprueba que cada directorio está en el commit fijado antes de importar nada. De `fla` solo carga los dos archivos de `fla/ops/titans`, sin ejecutar el `__init__` de la biblioteca. Ningún archivo de terceros se ha copiado al repositorio.

## Correspondencia ecuación a ecuación

La tabla compara el artículo (actas de NeurIPS 2025 y arXiv v1), las dos referencias ejecutadas, la implementación con MAC más cercana al texto y el núcleo del proyecto. La última columna clasifica cada diferencia del proyecto con estas categorías: equivalente, desviación declarada, error nuestro, decisión de la referencia ausente del artículo y ambigüedad del artículo.

| Elemento | Artículo | lucidrains | fla | omkarghugarkar007 | MARS-TITAN | Clasificación |
| --- | --- | --- | --- | --- | --- | --- |
| Red de memoria | MLP con `L_M ≥ 1` capas. Las actas describen `x + LN(W₁σ(W₂x))` y expansión 4 | `x @ W` por capa, GELU entre capas. Expansión 4 por defecto. `ResidualNorm` con ganancia de LayerNorm aprendida que también se actualiza como peso rápido | Una matriz con `k + LN(kM)·w + b` | SiLU y LayerNorm entre capas y salida acotada `2x/(1+‖x‖)` | `W x` con GELU exacta, matrices `D × D`. Residual y LayerNorm sin afinidad como componente desactivable | Expansión 1, desviación declarada. La ganancia como peso rápido es una decisión de la referencia. El artículo no fija si LayerNorm tiene afinidad, que es una ambigüedad |
| Pérdida asociativa | `‖M(k) − v‖²₂`, ecuación 2 | Media sobre las `D` componentes por defecto | Suma, con la derivada analítica de LayerNorm | Suma | Suma | Equivalente. La media de lucidrains equivale a dividir `θ` entre `D`, comprobado numéricamente |
| Sorpresa con momentum | `S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1}; x_t)`, ecuación 1 | Igual, con `θ` como peso de la pérdida y un scan asociativo para `η` | Igual | Igual, con `η` y `θ` acotados | Igual | Equivalente, comprobado numéricamente |
| Olvido | `M_t = (1 − α_t) M_{t−1} + S_t`. Escalar en arXiv v1 y vector en `[0, 1]^{d_in}` en las actas | Escalar por cabeza | Escalar | Escalar | Vector por filas de salida de cada matriz | Ambigüedad entre versiones y decisión declarada. El arnés iguala las filas para aislar el caso escalar |
| Puertas | Dependen de la entrada. Forma, bias e inicialización sin fijar | `Linear` con bias y sigmoide. `init_*_bias` anula los pesos | Entradas externas | `Linear` con pesos a cero y bias | Sigmoide sin bias en v1 y `gate_bias` declarado | Desviación declarada en la [inicialización de las puertas](../engineering/titans-gate-initialization.md) |
| Consultas, claves y valores | SiLU al calcularlas, convolución 1D separable tras cada proyección y normalización L2 de consultas y claves (sección 4.4 del preprint) | Sin SiLU ni convolución. RMSNorm opcional con ganancia y RMSNorm de la entrada por defecto | Entradas externas | L2 en claves, valores y consultas, sin convolución | L2 configurable, sin SiLU ni convolución | La convolución es una desviación declarada. La ausencia de SiLU no estaba declarada y se clasifica como error nuestro de documentación |
| Paralelización por bloques | Sección 3.2: gradientes en los pesos del inicio de cada bloque de tamaño `b` y tasas por token. Los experimentos usan tasas por token | `batch_size` fija el bloque de gradientes y `chunk_size` comparte tasas. Por defecto todo el tramo usa los pesos del inicio de la llamada | Forma por bloques coherente con su forma secuencial | Forma cerrada por bloques | Recurrencia exacta por token, sin forma paralela | Desviación declarada en el [núcleo](../engineering/titans-memory-core.md). El arnés verifica que el modo por bloques de lucidrains coincide con una transcripción de la sección 3.2 |
| Derivada interna | Derivada parcial respecto a la memoria | `torch.func.grad` respecto a los parámetros, con segundo orden en el gradiente exterior | Analítica | Pesos desacoplados, sin segundo orden por `M` | Copia diferenciable de `W_{t−1}` | Equivalente a lucidrains, comprobado con gradientes exteriores |
| Memoria persistente | Tokens aprendidos `p_1 … p_{N_p}` independientes de la entrada, ecuación 6 | Claves y valores aprendidos por capa y cabeza, más tokens aprendidos intercalados en cada segmento | No aplica | Tokens con ganancia aprendida por cabeza | Tokens `P` concatenados | Equivalente al artículo. Lo de lucidrains es una decisión de la referencia |
| Lectura sin actualizar | `M*(q)` sin escritura | `retrieve_memories` es puro, pero `forward` escribe y después lee con los pesos que ya incluyen el token | Salida con los pesos actualizados | Lectura pura antes de escribir | `read` puro con el estado recibido | Equivalente para la lectura tras escribir cada token, comprobado numéricamente |
| Segmentación MAC | `[P ‖ h ‖ S]` en la ecuación 22 del preprint y la figura 4a, `[P ‖ S ‖ h]` en la ecuación 7 de las actas | Sin tokens `h`. La lectura se suma al flujo residual antes de la atención | No aplica | `[P ‖ h ‖ S]` con posiciones y máscara propias | `[P ‖ h ‖ S]`, máscara triangular sobre la secuencia empaquetada y salida al cierre del segmento | Ambigüedad del artículo con convención declarada |
| Escritura en MAC | `M_t = M_{t−1}(y_t)` con la salida de atención | Escribe con la entrada de la capa, antes de la atención | No aplica | Con `y` | Con las posiciones de `S` en `y` | Equivalente al artículo. Lo de lucidrains es una decisión de la referencia |
| Combinación final | `o_t = y_t ⊗ M*_t(y_t)`, ecuación 10 de las actas | No existe. Opcionalmente puerta sigmoide de la lectura sobre la atención | No aplica | `y ⊙ puerta(M_{t−1}(y))` | `y_last ⊙ M_t(y_last)` | Desviación declarada en la [escala de la salida](../engineering/titans-mac-output-scale.md). El artículo no precisa `⊗` en MAC |
| Normalización y puerta final | Residual en todos los bloques, normalización y puerta lineal antes de la proyección final | RMSNorm previa, hyper-connections y residual de valores | No aplica | Propias | No están en el núcleo | Desviación declarada |

Dos diferencias de lucidrains merecen atención aparte porque cambian la regla de aprendizaje sin que lo indique el artículo. Con `batch_size=None`, que es su valor por defecto, todos los gradientes de una llamada se evalúan en los pesos con los que empieza esa llamada. La ganancia de su LayerNorm forma parte de los pesos rápidos y se actualiza en el bucle interno. Su ejemplo de entrenamiento activa además residual de pesos entre capas, vistas distintas para consultas, claves y valores, normalización espectral de la sorpresa y moduladores de tasa por capa. Son ampliaciones del autor de la referencia.

## Paridad numérica

[`benchmarks/titans_reference_parity.py`](../../benchmarks/titans_reference_parity.py) construye la memoria del proyecto, copia sus pesos en la de lucidrains y compara ambas con las mismas entradas. Las pruebas de [`test_reference_parity_harness.py`](../../tests/models/titans/test_reference_parity_harness.py) comprueban las piezas propias del arnés con el núcleo y sin código de terceros. Seis mutaciones dirigidas sobre esas piezas hacen fallar las pruebas.

### Configuración comparable

lucidrains se configura con la recurrencia del artículo, que es también la del proyecto: `chunk_size=1`, `batch_size=1`, una cabeza, pérdida sumada, sin RMSNorm de entrada ni de consultas y claves, MLP de expansión 1 y `max_lr = theta_max = 0,1`. El proyecto desactiva su normalización L2 (`normalize_qk=False`) para no comparar normalizaciones distintas. Las filas de la proyección de `α` se igualan, porque la referencia usa un olvido escalar. La memoria con residual se pasa a lucidrains como un modelo propio del arnés, `x + LN(MLP(x))` sin afinidad, para no introducir la ganancia aprendida de su `ResidualNorm`.

Solo se ejecutan pasos hacia delante, las escrituras internas de la memoria y `autograd.grad` de un funcional lineal fijo de las lecturas y de los pesos finales. No hay optimizador y ningún parámetro cambia salvo por la copia inicial de pesos.

Se comparan, para cada token, la lectura tras escribirlo y los pesos rápidos de cada capa, además del momentum final. En el gradiente exterior se comparan la entrada, las proyecciones de consulta, clave y valor, las tres puertas con sus bias y los pesos iniciales de cada capa. Los cinco casos cubren v1 con una y dos capas, `gate_bias` y `gate_bias` con residual y LayerNorm en `D = 16` y `D = 64`, con dos o tres flujos y 24 o 64 tokens.

Las tolerancias se declararon antes de ejecutar y se aplican elemento a elemento como `|a − b| ≤ atol + rtol·|b|`:

| Precisión | Lecturas y estados | Gradientes |
| --- | --- | --- |
| FP32 | `rtol = 10⁻⁵`, `atol = 10⁻⁶` | `rtol = 10⁻⁴`, `atol = 10⁻⁶` |
| FP64 | `rtol = 10⁻¹⁰`, `atol = 10⁻¹²` | `rtol = 10⁻⁹`, `atol = 10⁻¹²` |

### Resultados en CPU

El [recibo CPU](../../reports/engineering/titans-reference-parity-20261009/parity-cpu.json) se generó con dos hilos, TF32 desactivado y algoritmos deterministas. Una segunda ejecución produjo exactamente las mismas cifras.

| Caso | FP64 | Mayor cociente FP64 | FP32 | Mayor diferencia absoluta FP32 (lecturas y estados / gradientes) |
| --- | --- | --- | --- | --- |
| v1, dos capas, `D = 16` | Superado | 2,1·10⁻⁵ | Superado | 1,5·10⁻⁸ / 3,0·10⁻⁸ |
| v1, una capa, `D = 16` | Superado | 4,8·10⁻⁵ | Superado | 1,5·10⁻⁸ / 1,2·10⁻⁷ |
| `gate_bias`, dos capas, `D = 16` | Superado | 2,9·10⁻⁴ | Superado | 2,1·10⁻⁷ / 1,7·10⁻⁶ |
| `gate_bias` con residual y LN, `D = 16` | Superado | 7,9·10⁻³ | 6 de 19 tensores fuera | 3,3·10⁻⁶ / 1,3·10⁻⁴ |
| `gate_bias` con residual y LN, `D = 64` | Superado | 4,0·10⁻² | 10 de 19 tensores fuera | 1,1·10⁻⁵ / 1,2·10⁻³ |

El cociente es la mayor razón entre la diferencia y la tolerancia, de modo que 1 es el límite. En FP64 el mayor cociente de los cinco casos es 0,04. La mayor diferencia absoluta en FP64 es 1,9·10⁻¹² en gradientes de módulo cercano a 100.

Los dos casos con residual y LayerNorm no cumplen la tolerancia FP32 declarada. Para saber si la causa es el redondeo o una diferencia de cálculo, el arnés evalúa también en FP64 los mismos pesos y entradas FP32 en ambas implementaciones. Este análisis se añadió después de ver el resultado y es descriptivo. En todos los tensores de los cinco casos, la diferencia FP32 entre implementaciones es como mucho 1,75 veces la mayor de las diferencias de cada implementación frente a su propia evaluación FP64, y las dos evaluaciones FP64 coinciden hasta 1,9·10⁻¹². Es decir, cada implementación se separa del cálculo exacto tanto como se separan entre sí. La memoria con residual y LayerNorm amplifica el redondeo FP32, como ya se había observado entre CPU y CUDA en la [escala de la salida](../engineering/titans-mac-output-scale.md). Esto no cambia la tolerancia declarada, que sigue sin cumplirse en esos casos.

### Decisiones de la referencia

Con los mismos pesos, el caso con `gate_bias`, residual y LayerNorm, `D = 32`, dos flujos y 64 tokens cuantifica cuánto cambia el resultado al usar las opciones de lucidrains que no siguen la recurrencia del artículo. La distancia es relativa en norma de Frobenius, en FP64 (las cifras FP32 son muy parecidas).

| Opción de lucidrains | Lecturas | Pesos finales por capa |
| --- | --- | --- |
| Pérdida media con `θ` multiplicado por `D` | 1,8·10⁻¹⁵ | 1,6·10⁻¹⁵ y 1,3·10⁻¹⁵ |
| Ganancia de LayerNorm como peso rápido (`ResidualNorm`) | 0,92 | 0,66 y 0,69 |
| Gradientes por bloques de 4 tokens (sección 3.2) | 0,27 | 0,25 y 0,21 |
| Gradientes por bloques de 16 tokens (sección 3.2) | 0,60 | 0,63 y 0,63 |
| Tasas compartidas por chunk de 4 y bloques de 4 | 0,41 | 0,34 y 0,33 |
| Valor por defecto, gradientes en los pesos del inicio de la llamada | 0,85 | 2,9 y 2,6 |

La primera fila confirma la equivalencia de la pérdida media. En las filas por bloques, el arnés compara además lucidrains con una transcripción propia de la ecuación 16 y la recurrencia 18 del preprint, con tasas por token. Coinciden en FP64 con un cociente máximo de 6·10⁻³. En FP32 no cumplen la tolerancia, igual que el resto de casos con residual. El modo por bloques de lucidrains implementa por tanto la paralelización del artículo, y en este fixture se aleja mucho de la recurrencia exacta que usa el proyecto. Son medidas sobre entradas aleatorias. No dicen qué variante predice mejor.

### Comprobaciones de fla

Un paso con `θ = 1`, `η = 0` y `α = 0` aísla la derivada que usa la operación de fla. La actualización resultante tiene 5,4 veces la norma del gradiente que da autograd sobre su propia pérdida, con una distancia relativa de 4,8. La fórmula que aparece comentada en el mismo archivo, con todos los términos divididos por `rstd · D`, coincide con autograd hasta 3,6·10⁻¹⁵. En la fórmula activa solo los dos últimos términos están divididos. La forma por bloques y la forma secuencial de fla coinciden entre sí con una distancia relativa de 3·10⁻⁷ en FP32, porque comparten la misma fórmula. Su prueba está desactivada, así que nada en la biblioteca detecta la discrepancia. Por eso fla no sirve como referencia numérica del núcleo. Tampoco su salida sería comparable, porque lee `qM + LN(qM)·w + b` mientras su pérdida usa `k + LN(kM)`.

### CUDA

El [recibo CUDA](../../reports/engineering/titans-reference-parity-20261009/parity-cuda.json) repite la comparación con lucidrains en `cuda:0`, en una RTX 4070 Laptop GPU con TF32 desactivado y algoritmos deterministas. Reservó como máximo 433 MiB. En FP64 pasan los 5 casos, con un cociente máximo de 0,083. En FP32 pasan los 3 casos sin residual. Los 2 casos con residual y LayerNorm quedan fuera de la tolerancia declarada en 7 y 9 de 19 tensores, con una diferencia entre implementaciones que es como mucho 1,61 veces el redondeo de cada una frente a su evaluación FP64. Las comprobaciones de `fla` solo se ejecutaron en CPU. La orden y el resto de detalles están en el [resumen de medidas](../../reports/engineering/titans-reference-parity-20261009/README.md).

## Recomendación

La evidencia apoya la opción (c), que en la práctica es (b) para la memoria y una referencia externa fijada como oráculo.

1. Conservar el núcleo del proyecto. En la configuración que sigue la recurrencia del artículo, su memoria coincide con la de lucidrains en FP64 en lecturas, estados y gradientes exteriores, y en FP32 dentro de tolerancia sin residual. La memoria no es una interpretación sin contraste. Las diferencias que quedan están clasificadas en la tabla anterior.
2. No incorporar código de lucidrains. Sus valores por defecto no siguen la recurrencia del artículo, su MAC no sigue las ecuaciones 21 a 25 y adoptarla obligaría a fijar las opciones que ya reproduce el proyecto. Tampoco aportaría una ventaja numérica, porque los resultados coinciden.
3. Mantener lucidrains `1d40c445` como oráculo externo con este arnés, sin vendorizar. Cualquier cambio futuro de la memoria debe volver a pasar el arnés antes de declararse equivalente.
4. Corregir la desviación sin declarar de SiLU. Hay dos opciones. Declararla como omisión, que es lo que ya refleja el [núcleo](../engineering/titans-memory-core.md) desde esta revisión, o implementarla como componente desactivable con identidad nueva. La segunda cambiaría las proyecciones de MAC y de la memoria y necesita decidirse antes de la campaña, no durante ella.
5. Si el coste del entrenamiento lo exige, la paralelización de la sección 3.2 puede añadirse como identidad nueva. El arnés ya contiene la transcripción y la comparación con el modo por bloques de lucidrains. Cambia la regla de aprendizaje, con distancias de 0,27 a 0,60 en el fixture, así que no sería una optimización equivalente.

### Coste de migrar a la referencia

Adoptar lucidrains como base, la opción (a), tendría este coste. No se ha migrado nada.

| Área | Efecto |
| --- | --- |
| Estado y API | Los pesos de lucidrains están transpuestos, en `TensorDict` y con una dimensión de orden de momentum. `forward` escribe y lee a la vez y no ofrece lectura sin escritura en su API pública. Habría que reescribir `NeuralMemoryState`, `read`, `update`, la exportación y la recuperación |
| Identidades | Cambiarían las huellas de `MemoryConfig`, `MACConfig` y `FinancialConfig`, las recetas cronológicas y los recibos que las citan. Los controles emparejados y sus comprobaciones de igualdad de parámetros tendrían que regenerarse |
| Entrenador cronológico | `financial_run.py` asigna roles por nombre de parámetro (`mac.memory.initial_weights.`) y tendría que adaptarse |
| MARS-TITAN M0–M3 | Consumen la salida y el estado de MAC a través de la sesión financiera. La interfaz se podría conservar, pero sus pruebas de paridad y recuperación se basan en las identidades actuales |
| CM-v1 | El control C local y el Jacobiano de transición operan sobre pesos rápidos y momentum `[B, D, D]` en la orientación del proyecto. Habría que rehacer la correspondencia del operador y su alcance matemático |
| Código nativo | No hay una implementación C++/CUDA de Titans. El código nativo es de la GRU candidata y de las políticas, así que no cambiaría |
| Pruebas | Las 225 funciones de prueba de `tests/models/titans/` y los 76 archivos de `tests/` que mencionan Titans dependen de la API y de las identidades actuales |
| Dependencias y licencia | Habría que añadir `tensordict`, `einops`, `einx` y `assoc-scan`, conservar el aviso MIT de lucidrains y separar sus derechos de los del proyecto |

## Límites

El arnés compara la memoria neuronal, que es la parte que tiene una referencia comparable. El MAC del proyecto queda contrastado con el texto del artículo y no con un código público equivalente, porque no existe. Las comparaciones usan fixtures aleatorios con escala de entrada 0,3, `θ_max = 0,1` y como mucho 64 tokens. No miden rendimiento predictivo y no sustituyen la campaña. La revisión de los repositorios secundarios se limitó a leer su código y puede haber pasado por alto detalles que solo se verían ejecutándolo.
