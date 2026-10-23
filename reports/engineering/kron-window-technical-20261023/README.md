# KronLSTM y ejecución de ventanas cortas

La variante no canónica añade `kronlstm_v1` y el control `vector_delta_v1` al recorrido financiero de ventanas independientes. Ambos usan el mismo bloque y número de parámetros. `FinancialConfig.temporal_encoder=None` conserva la ruta anterior. GRU episódica, Transformer y Titans-MAC siguen como referencias separadas.

El código está recogido en los commits `056b2c5401fdce7533e6d85bbe4ad4d2c9bbe24b`, `df32bfd357b52f85c8edf933ea4c34f4587eebf0`, `b19e01ffca35d39d689f4cc63295caf59a8448ce` y `6385f3890406d5a6a0e77a0d2be1e49939277614`. La [auditoría de Kron](https://github.com/GonxKZ/mars-titan/blob/b19e01ffca35d39d689f4cc63295caf59a8448ce/docs/experiments/kronlstm/source_audit.md) distingue la matriz temporal completa, la caché de convolución y los parámetros compartidos. La acumulación QKV en FP64 de CPU con entrada y pesos FP32 tiene identidad propia y devuelve FP32. CUDA conserva la proyección nativa. No se presenta esa corrección como precisión íntegramente FP32 ni como reducción universal del error.

## Verificación técnica

| Grupo | Casos aprobados | Alcance |
| --- | ---: | --- |
| Kron, regresiones y finitud diferida | 275 | Contiene los 248 casos previos y los 27 de finitud diferida |
| Window | 56 | Seis rutas completas con fixtures, receta, admisión, recuperación, precisión y artefactos |
| CLI | 19 | Admisión y protocolo del lanzador |
| Readout | 3 | Huellas heredadas con y sin codificador opcional |

Los grupos terminaron sin fallos, errores u omisiones. El total de 275 contiene 248 casos anteriores y 27 de finitud diferida. No se vuelven a sumar esos componentes, las repeticiones instrumentadas ni las fases de mutación. La regresión conjunta duró 13,599 s supervisados. Los 50 casos históricos de Window configuraban las seis recetas, pero solo ejecutaban de principio a fin Transformer con AdamW, mediante BP o Dust. La ampliación pasó posteriormente los 56 casos y las seis rutas publicadas con fixtures, tras corregir el acceso a `_version` de los tensores de inferencia. Cada ruta comprueba tres actualizaciones mediante espía y checkpoint.

Los 87 registros del contador privado de la corrida de 56 casos son eventos post-hook de diagnóstico. Los espías que llaman a un `optimizer.step` envuelto pueden duplicarlos. No se interpretan como pasos físicos ni actualizaciones únicas. El registro histórico de 57 eventos tiene el mismo límite. La comprobación de tres actualizaciones por cada nueva ruta procede del espía y del checkpoint, no de sumar estos eventos. Son pruebas técnicas, no un ensayo financiero de desarrollo.

Se detectaron cuatro mutantes de Kron, uno de Window, uno de Readout y dos del chequeo de finitud. Cada objetivo tiene original G1 aprobado, fallo semántico CALL en M1 y restauración G2 aprobada, sin errores de colección ni omisiones. W1 relaja indebidamente el intervalo conjunto de 366 a 730 días. R1 llama al método de huellas del padre sin `self`. Q1 reintroduce el acceso al contador inexistente y Q2 permite reutilizar indebidamente un chequeo de inferencia tras una mutación. Estas comprobaciones no acreditan resistencia a cualquier mutación. Los cinco objetivos de Dust se documentan por separado, junto con sus 70 casos únicos, diez CUDA.

Coverage 7.16.1 y Radon 6.0.1 registran 336/360 sentencias, 110/130 ramas y CRAP máximo 44,0058 en los dos archivos del núcleo Kron. Window registra 454/496 sentencias, 134/178 ramas y CRAP máximo 53,5355. La convención usa `v²(1−c)³+v`, con complejidad Radon y cobertura de sentencias en el rango de función. Las ramas se informan aparte. Son diagnósticos de estos alcances.

El archivo `state.py` registra 127/137 sentencias y 38/44 ramas. La función modificada `require_finite` cubre 14/14 sentencias y 10/10 ramas, con complejidad 7 y CRAP 7. Ese CRAP corresponde solo a la función. Otras rutas del archivo conservan carencias de cobertura, con máximo 92,0605 en `join_mac_rows`.

La repetición instrumentada de los mismos tres casos Readout cubre las tres sentencias de `ReadoutAdapterTrainer._code`, con complejidad 2, CRAP 2 y ninguna rama en esa función. Duró 4,125 s supervisados. No añade casos únicos ni valida las demás rutas del entrenador, que no ejercitan estas pruebas.

## Perfil sintético compartido

El perfil compara Transformer, Kron y control vectorial con dimensión 32, una capa, contexto 64, lotes 32/256 y FP32 estricto en una RTX 4070 Laptop GPU. Completó doce condiciones, un calentamiento y cinco medidas por condición, 78 preparaciones y 42 backward de una pérdida cuadrática sintética. No ejecutó optimizadores ni utilizó objetivos financieros.

Con 256 filas, las medianas de forward fueron 12,863 ms para Transformer, 119,678 ms para Kron y 70,800 ms para el control vectorial. Al incluir pérdida y backward fueron 16,099, 408,023 y 318,786 ms. El proceso interno duró 37,584 s y la supervisión 41,222 s. Cada perfil conserva sus fuentes fijadas. Las huellas del árbol posterior se registran aparte.

Las medidas se hicieron bajo concurrencia técnica declarada. El intervalo CUDA incluye huecos entre lanzamientos y no equivale al tiempo exclusivo de kernels. Los picos de memoria Torch y el RSS máximo de vida se conservan separados en el resumen. Esta fixture no acredita aceleración global, coste del ajuste real, ahorro energético ni utilidad predictiva.

El [informe de Dust](https://github.com/GonxKZ/mars-titan/blob/3e4b6979fee0c008a78aaa8df30f07d47bcf1af3/reports/engineering/dust-technical-20261023/README.md) conserva el rechazo de chunk 4 por tolerancia de gradientes y la sonda posterior aprobada de chunk 8, sin relajar tolerancias. Su informe independiente explica sus tres muestras por configuración y sus límites.

## Intento sobre la ventana real

La preparación de `pilot-month-v3` produjo la vista de ajuste `[2015-01-01, 2015-02-01)` y validación `[2015-02-01, 2015-03-01)`, con SHA-256 `b4e970cc67c629e2579c6ff050fd514ba0311ae46dc96ad8f8141c55757caa8f`. Terminó con código 0 en 30,293 s. El intento completo duró 32,085 s y se detuvo antes de iniciar el ajuste: `GpuLease` rechazó la admisión con `another_compute_process` y código 75. No se inició ninguna configuración ni proceso de ajuste y se confirmaron cero actualizaciones. El permiso acotado del proceso quedó cerrado y la protección global permaneció intacta.

Una auditoría posterior recorrió todas las 3.249 filas de ajuste y 3.078 de validación, con huellas de IDs, entradas y objetivos para ambas particiones. El padre declara 256 activos y 171 aportan filas en cada partición. La auditoría no construyó predictores, no emitió predicciones ni ejecutó actualizaciones. Duró 5,041 s internamente y 7,082 s supervisados, con un pico RSS de vida del proceso de 758.120.448 bytes. Ese pico pertenece a la auditoría de entradas, no al entrenamiento.

Con lote 32 y dos épocas completas, la planificación requiere 204 actualizaciones y 6.498 exposiciones lógicas a filas de ajuste por configuración. Son cantidades proyectadas, no ejecutadas. La estimación parcial que usa el perfil sintético Dust tampoco representa el tiempo total del ajuste. No hay MAE financiero ni mejora predictiva observados. El test final permaneció cerrado.

## Fallos previos y trabajo pendiente

La inconsistencia inicial de CPU FP32 entre ventana completa y chunks apareció en las semillas 1 y 2. La corrección QKV declarada se contrastó con las tolerancias originales y sus desviaciones quedaron documentadas. No se oculta el aumento de error máximo de una coordenada en la comparación con la referencia de mayor precisión.

La primera corrida de los 50 casos Window terminó con 43 aprobados, un fallo y seis errores por cuota de almacenamiento temporal. Se conserva ese intento. La repetición con temporales en disco pasó los 50 casos y duró 67,826 s supervisados. Un intento anterior de regresión amplia acabó en timeout y tampoco cuenta como suite aprobada.

La primera ampliación a 56 casos terminó con 54 aprobados y dos fallos, en Kron y en su control vectorial durante la evaluación inicial. `state.require_finite` intentaba leer `_version` de un tensor creado bajo `inference_mode`. La corrección comprueba la finitud sin exigir ese contador y conserva el rechazo de NaN/Inf. La repetición pasó los 56 casos en 83,773 s supervisados. Se conserva el intento fallido anterior.

Dos intentos instrumentados de la suite de finitud terminaron con código 139 antes de ejecutar casos al usar un nombre de módulo como fuente de coverage. Uno registró una aserción interna de PyTorch en la conversión NumPy y el otro no produjo salida. Usar un directorio como fuente evitó esa importación anticipada y permitió pasar los 27 casos sin cambiar producto ni tolerancias. Esto no demuestra que los fallos de entorno estén corregidos de forma general.

Quedan la revisión e integración del conjunto y los ensayos financieros de desarrollo con fechas, filas, máscaras, pérdidas, selección, actualizaciones y presupuestos emparejados. No se ha medido aquí MAE financiero ni una ventaja predictiva. Las variantes siguen opcionales. La campaña completa, la reanudación automática y el test final permanecen cerrados.

El [resumen curado](summary.json) conserva huellas, grupos de pruebas, mutaciones y estadísticas de los perfiles sin incluir rutas personales ni datos de procesos concurrentes.
