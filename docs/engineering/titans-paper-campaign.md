# Declaraciones de campaña con proyecciones del artículo

MARS-TITAN conserva la receta histórica de A v2 y añade dos declaraciones completas e independientes. La referencia `sequence` mantiene `linear_v1` y fija `attention_layout=sequence_contiguous_v1`. La variante `paper` cambia únicamente `memory_projections` a `titans_mac_paper_projections_v2` respecto a esa referencia. Así se puede contrastar la incorporación de las proyecciones sin mezclarla con el cambio de disposición de atención.

| Declaración | Campaña | Receta del núcleo | Factorial propio |
| --- | --- | --- | --- |
| Histórica A v2 | [A v2](../../configs/baselines/historical-masked-campaign-a-v2.json) | [Histórica](../../configs/titans/chronological-training-historical-masked.json) | [CM-v1 histórico](../../configs/titans/cm-v1-factorial.json) |
| Referencia con layout común | [A sequence v1](../../configs/baselines/historical-masked-campaign-a-sequence-v1.json) | [Sequence](../../configs/titans/chronological-training-historical-sequence-v1.json) | [CM-v1 sequence](../../configs/titans/cm-v1-sequence-factorial.json) |
| Proyecciones del artículo | [A paper v1](../../configs/baselines/historical-masked-campaign-a-paper-v1.json) | [Paper](../../configs/titans/chronological-training-historical-paper-v1.json) | [CM-v1 paper](../../configs/titans/cm-v1-paper-factorial.json) |

Las opciones del núcleo, sus [ecuaciones y desviaciones](titans-mac-equations.md), los pesos rápidos y las ventanas de convolución ya existen. Estas declaraciones los conectan con los cuatro controles, ampliaciones y factoriales de la campaña. No añaden otro predictor ni duplican su canalización. La GRU episódica, el Transformer compacto, las referencias tabulares y las modalidades conservan las mismas declaraciones. La adaptación financiera sigue siendo propia del proyecto, sin reproducir por sí sola los resultados de Titans.

Las tres campañas comparten datos, ventanas, casos de búsqueda, semillas, pérdidas, selección, treinta épocas máximas, parada conjunta y presupuestos pertinentes. Las dos nuevas recetas mantienen truncamiento ocho, acumulación y bloques de 1.024 filas y FP32 estricto. La diferencia entre `sequence` y `paper` está limitada a las proyecciones. La diferencia frente a A v2 incluye también el layout y se informa como tal. El nombre `linear_v1` describe las proyecciones y no implica que su memoria neuronal sea lineal.

Cada campaña tiene nombre y huella propios. Las recetas y declaraciones CM-v1 también tienen huellas distintas que entran en los casos y checkpoints. Las claves internas del esquema, como `titans_mac_online` o `cm_v1_b`, se interpretan dentro de su campaña. Por ejemplo, `historical-masked-2000-a-paper-v1/cm_v1_b` no es la B histórica. No se reutilizan sus checkpoints como si fueran compatibles. Cambiar la referencia requiere fijar de nuevo todos los componentes B, B+C, B+M y B+C+M y sus manifiestos de ejecución.

Los dos factoriales usan el mismo núcleo que su campaña. C conserva la base común, SDPA Math y la penalización local del Jacobiano proyectado. La opción de convolución tiene [comprobaciones del Jacobiano con sus ventanas](../../tests/models/titans/test_paper_projections.py). Eso no convierte C en un certificado de estabilidad global. M sigue consolidando episodios maduros del banco, separado de los pesos rápidos. Las hipótesis y limitaciones originales de C y M siguen vigentes.

Estas declaraciones permanecen sin ejecutar. El presupuesto tensorial y los límites se conservan, pero el coste de la campaña nueva no se obtiene de copiar las mediciones históricas ni de un piloto de 64 activos. Antes de ejecutarla necesita su informe de disjunción con la huella exacta de campaña y vistas. La aceptación de fases, colas, almacenamiento y selecciones completas mantiene su ámbito propio. Un informe de vistas sin selecciones no acredita una cadena de entrenamiento ejecutada.

La campaña completa y el test final de 2024 siguen cerrados. Ningún piloto anterior demuestra que estas declaraciones mejoren la predicción. Las nuevas rutas no pasan a ser el valor por defecto de los comandos existentes ni reemplazan las identidades y resultados históricos.
