# Una evidencia y varias vistas regionales

`SharedEvidenceBridge` implementa un componente experimental de lectura sobre un único `EpisodeSnapshot`. El registro episódico conserva su identidad, valores y etiqueta madura. Cada región calcula pesos de atención y una transformación propios sobre la misma selección de episodios. Los puentes comunican las representaciones derivadas. No crean otro registro del episodio ni otra memoria persistente.

El componente está en `mars_titan.models.titans.evidence_bridge`. Todavía no está integrado en `FrozenFinancialConsumer`, en sus trazas ni en una configuración entrenable del predictor completo. No sustituye al lector anterior y no se activa por defecto. Las [pruebas y medidas del componente](../../reports/engineering/shared-evidence-bridge-20261021/README.md) son técnicas. No hay una mejora predictiva medida.

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
