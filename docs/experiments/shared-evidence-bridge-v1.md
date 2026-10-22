# Una evidencia y varias vistas regionales

`SharedEvidenceBridge` implementa un componente experimental de lectura sobre un único `EpisodeSnapshot`. El registro episódico conserva su identidad, valores y etiqueta madura. Cada región calcula pesos de atención y una transformación propios sobre la misma selección de episodios. Los puentes comunican las representaciones derivadas. No crean otro registro del episodio ni otra memoria persistente.

El componente está en `mars_titan.models.titans.evidence_bridge`. El lector episódico admite la opción explícita `EpisodicReadoutConfig.evidence_bridge`, que lo conecta a `FrozenFinancialConsumer` y a las sesiones nativas. `None` conserva la configuración, los parámetros y los resultados del lector anterior. La opción activa tiene identidad propia. No se activa por defecto. Las [pruebas y medidas iniciales del componente](../../reports/engineering/shared-evidence-bridge-20261021/README.md) describen la versión independiente anterior a esa integración. No hay una mejora predictiva medida.

## Integración con el predictor

El puente y el lector declaran el mismo codec, dimensión, vecinos, temperatura, lote máximo y presupuesto de trabajo. La construcción empareja sus proyecciones comunes y refinador con la inicialización anterior. Las proyecciones pasan al submódulo del puente y se retiran del padre, por lo que no quedan dos copias de sus parámetros. Las matrices regionales tienen una semilla propia identificada.

`RegionalEpisodeRead` contiene lectura, presencia y `BridgeRead`. `BridgedReadoutResult` conserva el estado refinado y esas lecturas. No tienen un campo que convierta los pesos regionales en `EpisodeRead.weights[B,k]`. El resultado anterior mantiene su tipo y significado. Las rutas anteriores siguen rechazando la ampliación. El controlador dispone de rutas regionales y de un contrato explícito distinto. `MarsTitanVariant` construye también esta opción desde el componente `evidence_bridge` de su declaración, manteniéndolo apagado por defecto. El [colector regional](../../reports/engineering/regional-mechanism-collection-20261022/README.md) y el [ajuste protegido](../../reports/engineering/mechanism-fit-execution-20261022/README.md) tienen verificación técnica. Su incorporación a la campaña completa y los contrastes predictivos siguen pendientes.

La [revisión de recurrencia](../research/regional-recurrence-extensions.md) distingue los ciclos lineales actuales de una transición no lineal propuesta. Con máscara fija, la mezcla repetida admite un control mediante su proyección efectiva. Ese análisis conserva las atenciones iniciales y no acredita una mejora predictiva o una aceleración por sí solo.

La misma cabeza escalar o de cuantiles consume el estado refinado. K = 1, 2 y 4 cuenta lecturas episódicas, con una sola actualización de MAC por observación. `per_step` vuelve a seleccionar candidatos comunes y `first_read` conserva la selección inicial. Las posiciones se validan antes del acceso a memoria, incluida su unicidad y rango. Esa validación inmediata puede sincronizar CUDA y forma parte del coste. Los ciclos del puente siguen siendo otro eje.

El componente del catálogo declara exactamente `regions`, `topology`, `cycles`, `bridge_gain` y `seed`. La validación de estas opciones es común al puente y al catálogo. El codec, dimensión, vecinos, temperatura y presupuestos proceden del lector. La identidad normaliza las regiones como tupla y la ganancia como número real. Cambiar cualquiera de sus cinco opciones cambia la identidad de la variante. El consumidor comprueba además que los metadatos correspondan al módulo físico del puente. Rechaza módulos ausentes, otra configuración o formas incompatibles antes de crear la sesión. Omitir el componente conserva las identidades de configuración y variante anteriores. Las huellas de ejecución también incluyen el código y corresponden a cada revisión.

En `no_bank` no se consulta un snapshot ni se inventa su huella. Un snapshot vacío sí conserva su procedencia, con lecturas cero y presencia falsa. Las nuevas matrices se clasifican como parámetros de lectura episódica. Esto prepara su papel en la canalización existente, sin ejecutar un ajuste ni confirmar la utilidad de entrenarlas.

La sesión guarda una traza regional por decisión con configuración, corte, contexto, huella y tamaño del snapshot, IDs comunes, pesos, vistas derivadas y lectura. El grafo compartido se guarda una vez y cada bloque conserva el orden de sus flujos. No se copian los valores fuente ni sus etiquetas a esa traza. Las copias archivadas son CPU, propias y sin autograd. Se valida su presupuesto antes de emitir y su coherencia antes de confirmar o recuperar. Los contadores distinguen consultas de candidatos, filas proyectadas y lecturas regionales. Son unidades de trabajo, no FLOPs.

La retención sigue las reglas y límites del almacén de artefactos de sesión. La traza recuperable de una generación no equivale a un archivo ilimitado de todas las vistas históricas. Los experimentos deberán registrar qué evidencia conservan. La admisión M0–M3, la maduración y las escrituras del banco mantienen sus contratos y no se multiplican por el número de regiones. El [informe de integración](../../reports/engineering/bridge-readout-integration-20261021/README.md) conserva las comprobaciones y el coste nativo, con sus ejecuciones incompletas y límites.

## Selección mediante el controlador

`MechanismCatalog(reader_contract="shared_evidence_bridge_v1")` declara la ejecución regional. Su identidad es distinta de la del catálogo anterior, que sigue siendo el valor por defecto. `FrozenMechanismConsumer` comprueba que el lector y el catálogo correspondan. Un `MechanismPolicy` regional debe declarar el `catalog_id` compatible. El identificador vincula el contrato, pero no demuestra que los coeficientes suministrados se hayan entrenado ni que su utilidad sea correcta.

`regional_routes.run_route_pair` ejecuta `k1`, `k2c`, `k2r`, `k4cc`, `k4cr` y `k4rc`. Comparte el primer refinamiento con la ruta elegida. En este catálogo, c conserva las posiciones comunes y r vuelve a seleccionarlas al empezar el bloque correspondiente. Se recalculan las proyecciones en cada lectura y se contabiliza ese trabajo. No se atribuye la caché de proyecciones del lector anterior a estas rutas. Estos símbolos no designan el factor C de CM-v1.

El selector utiliza observaciones de la cohorte anteriores a MAC, sin incorporar retroactivamente pesos o rasgos calculados después de K1. La ruta actúa sobre el estado de trabajo, con un solo avance del núcleo por observación. Sus recibos JSON conservan por fila los IDs comunes, los pesos y las vistas regionales derivadas, la lectura y el snapshot. La topología y los nombres regionales pertenecen a la configuración identificada del lector. La sesión comprueba los recibos al confirmar o recuperar y los guarda en el archivo de mecanismos existente.

El presupuesto estima las trazas de toda la cohorte antes de avanzar MAC, además del trabajo por bloque. Si K4 no cabe, el catálogo puede resolver la acción hacia K1 según su regla declarada. Si tampoco cabe K1, se rechaza el evento. Es una estimación de almacenamiento y metadatos, sin acreditar el pico total de RAM. Las búsquedas, las lecturas y las proyecciones siguen siendo unidades de trabajo distintas. La integración no cambia el baseline científico ni autoriza entrenamiento.

El enlace del controlador conserva su admisión M0/M1. La integración con M2/M3, la declaración arquitectónica y la recogida o el ajuste de alternativas regionales necesitan sus comprobaciones propias. El [informe del controlador regional](../../reports/engineering/bridge-controller-20261021/README.md) distingue las pruebas técnicas, las dos pasadas de coste y sus límites.

## Cálculo y propiedad de los datos

Para una consulta h de dimensión H, el lector proyecta una consulta común y selecciona hasta ocho episodios del snapshot. La selección usa similitud en FP64 y desempata por el orden estable de sus identificadores. Las claves tienen dimensión 64. El valor de cada episodio reúne sus 64 rasgos y la etiqueta financiera ya madura. Esa etiqueta no procede de una reconstrucción generativa.

Los valores seleccionados se proyectan una sola vez a H. No tienen un eje de regiones. Si uₑ es ese valor proyectado y kₑ su clave, cada región r calcula:

$$
q_r=\nu(Qh+b+Q_rh),\qquad
a_{r,e}=\operatorname{softmax}_{e}\left(q_r^\mathsf{T}k_e/T\right),\qquad
z_r^0=\tanh\left(V_r\sum_e a_{r,e}u_e\right).
$$

ν reutiliza la normalización escalada del lector episódico. Los parámetros Qᵣ y Vᵣ pertenecen a cada región. Las identidades e y sus valores fuente siguen siendo comunes. La salida conserva una lista de IDs por consulta y una matriz de pesos región × episodio, que permite comprobar esa correspondencia. Los pesos no acreditan que un episodio sea una causa económica o una explicación fiel de la predicción.

Las ecuaciones escriben vectores columna. El código utiliza lotes de vectores fila y almacena las transformaciones regionales con la orientación correspondiente a sus productos matriciales.

La topología puede ser diagonal, un anillo dirigido con conexiones propias o completa. P normaliza por fila las aristas que unen regiones disponibles. Las regiones ausentes no envían ni reciben mensajes. Para cada ciclo:

$$
z_r^{c+1}=(1-\gamma)z_r^c+\gamma\sum_s P_{r,s}z_s^c,\qquad
v=O\operatorname{concat}(z_1^C,\ldots,z_R^C).
$$

Hay de una a ocho regiones y de cero a cuatro ciclos, con γ entre cero y uno. La proyección O depende de la posición regional. Una media final con puentes simétricos podría cancelar matemáticamente su efecto. Las pruebas comprueban un caso donde la comunicación cambia la salida y otro donde un solo episodio produce vistas distintas.

En aritmética exacta, tanh deja cada coordenada inicial en [−1,1]. La mezcla convexa conserva ese intervalo para las regiones presentes y deja las ausentes a cero. Por tanto, la norma euclídea de la concatenación no supera √(RH). La salida está acotada por ‖O‖₂√(RH). Esta observación local no certifica los cálculos en coma flotante, los gradientes, la retención histórica ni la estabilidad del predictor completo. Los desbordamientos anteriores a tanh se detectan, aunque la activación pudiera convertirlos en valores finitos.

## Alcance temporal, memoria y controles

Se reutiliza el contrato de snapshot maduro, con codec, contexto, corte y precisión identificados. Las comprobaciones por llamada detectan cambios de versión de sus tensores. La huella completa se verifica en las fronteras de recuperación o auditoría correspondientes. Un hash declarado no autentica una fuente capaz de reescribir datos y código. Este componente no adquiere ni valida por sí solo la procedencia histórica de las fechas.

No se conserva estado entre llamadas ni un grafo de autograd de la historia. Sí hay índices, copias transitorias por consulta, valores proyectados, pesos de atención y vistas regionales. Compartir un registro no significa eliminar todas las copias de trabajo o las transferencias entre dispositivos. El presupuesto configurable es una estimación conservadora de trabajo, no una garantía del asignador de PyTorch ni un límite de RAM total.

Con B consultas, N episodios y k vecinos, la selección común cuesta O(BN log N) por la ordenación estable, además del producto de claves de dimensión 64. No se repite esa selección por región. Las proyecciones de valores cuestan O(Bk65H), las consultas regionales O(BRH64), la atención O(BRk(64 + H)), las transformaciones y fusión O(BRH²) y los C ciclos de mensajes O(CBR²H). Se almacenan 130H + 64 + 64RH + 2RH² parámetros y R² booleanos de topología. El banco fuente queda fuera de esa cuenta de parámetros. Estas expresiones describen el algoritmo y no sustituyen la medida de tiempos o memoria.

La elección común reduce trabajo de selección frente a buscar en todo el banco por región, pero puede excluir un episodio útil para una región concreta. Ese compromiso debe contrastarse. Los controles previstos son lectura única, varias vistas sin puentes, topología diagonal, anillo y comunicación completa. El contraste con copias independientes contabilizará cada copia, sus escrituras y su capacidad. Los tamaños y parámetros distintos no se describirán como presupuestos idénticos.

Los nombres de región son etiquetas de configuración. No demuestran especialización aprendida. El grafo de modalidades existente, los bancos de pesos rápidos y estas vistas episódicas tienen funciones distintas. Una clasificación por sector necesita pertenencia y disponibilidad históricas verificables. `SectorMap` conserva su declaración `point_in_time=False` y no se transforma en una clasificación válida desde 2000. La escala temporal puede definirse con edades o resúmenes calculados solo con el pasado permitido. No hay en este componente un experto por año, una clasificación sectorial histórica nueva ni un aprendizaje de esas etiquetas.

## Antecedentes y límites de la analogía

[Roy y colaboradores, 2022](https://www.nature.com/articles/s41467-022-29384-4), estudiaron conjuntos conectados de engramas en varias regiones de ratones y su reactivación. Esa evidencia motiva estudiar representaciones distribuidas y recuperación parcial. No demuestra copias completas de un recuerdo en todas las regiones ni determina un número de módulos para este modelo. Se revisaron el resumen y los resultados indexados de la fuente primaria. La nueva apertura del texto completo quedó bloqueada por el proveedor.

[Rigotti y colaboradores, 2013](https://www.nature.com/articles/nature12160), analizaron selectividad mixta y representaciones distribuidas en una tarea de memoria secuencial en monos. Su resumen motiva probar que una observación participe en varias representaciones, sin asignar una única función semántica a cada neurona. No demuestra una mejora financiera. En esta revisión se comprobaron resumen y metadatos, no una nueva lectura íntegra.

[Murray y colaboradores, 2014](https://pubmed.ncbi.nlm.nih.gov/25383900/), midieron escalas de fluctuación neuronal en áreas corticales de macacos. El resumen y los pies de figura consultados distinguen tiempos de autocorrelación, no años financieros ni periodos óptimos de retención de episodios. Las escalas económicas necesitan sus propios controles.

La lectura de varios estados desde una entrada común tiene antecedentes como [Perceiver, ICML 2021](https://proceedings.mlr.press/v139/jaegle21a.html), y [Slot Attention, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html). Se contrastaron sus resúmenes y la descripción indexada de sus mecanismos. Este componente no reproduce esos modelos ni reivindica novedad por compartir un pool de valores.

[SEAL, versión 2 del 18 de septiembre de 2025](https://arxiv.org/html/2506.10943v2), usa un bucle externo para aprender directivas de edición y un ajuste interno de pesos. Las secciones 3.1 y 5 consultadas también describen límites de retención y coste. Este puente no ejecuta ese algoritmo, no genera etiquetas y no aprende todavía a decidir qué modificación conviene. La futura integración con el controlador mantiene la separación entre datos disponibles al decidir y error financiero conocido después.

Las fuentes se comprobaron el 21 de octubre de 2026. Las ideas sobre conciencia o estructura del universo no constituyen hipótesis verificadas de esta implementación. El contraste del proyecto sigue siendo error predictivo, recuperación, retención y coste con controles temporales.
