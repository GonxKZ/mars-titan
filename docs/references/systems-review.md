# Modelos eficientes y costes que deben compararse

La eficiencia se mide sobre MARS-TITAN y su equipo. Un resultado publicado sobre generación de lenguaje, robótica o una GPU de servidor justifica estudiar un mecanismo, pero no permite anunciar esa aceleración en predicción financiera.

## Referencias de arquitectura

| Fuente primaria | Qué aporta a la comparación | Límite de transferencia |
| --- | --- | --- |
| [Mamba](https://arxiv.org/abs/2312.00752v2) | Estado selectivo y tratamiento eficiente de secuencias largas. | Estado de tamaño acotado no equivale a memoria perfecta. La selección y el olvido forman parte del diseño. |
| [Mamba-2](https://proceedings.mlr.press/v235/dao24a.html) | Referencia de estado estructurado y relación con atención. | Sus kernels CUDA, anchuras y cargas de lenguaje difieren de los lotes financieros pequeños. |
| [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | Separa retención de actualizaciones dirigidas mediante regla delta. | El entrenamiento paralelo depende de la forma concreta de la actualización. No se extiende a cualquier memoria mutable. |
| [xLSTM](https://papers.neurips.cc/paper_files/paper/2024/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html) | Referencias escalares y matriciales para memoria recurrente. | Una matriz por activo escala con el cuadrado de su dimensión y necesita medir estabilidad numérica. |
| [Kimi Linear](https://arxiv.org/abs/2510.26692v2) | Antecedente de memoria finita, actualización delta y diseño por bloques. | El modelo publicado tiene una escala muy superior al presupuesto local. Se estudia el mecanismo, no se propone entrenar sus pesos completos aquí. |
| [FlashAttention](https://arxiv.org/abs/2205.14135v2) y [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Referencias para comparar atención exacta con buena gestión de transferencias y partición de trabajo. | Reducir operaciones o memoria asintótica no garantiza menor latencia. La generación de kernels CUDA y su compatibilidad deben comprobarse por dispositivo. |

Se propone elegir una referencia recurrente sencilla y una referencia moderna de estado o regla delta para la comparación principal. Añadir todas las arquitecturas a la campaña completa multiplicaría el presupuesto sin aislar necesariamente una pregunta distinta. Las demás pueden entrar en un cribado de desarrollo con criterios de exclusión explícitos.

## Aprendizaje lento y respuesta rápida

[Hinton, Vinyals y Dean](https://arxiv.org/abs/1503.02531) establecen un antecedente directo de destilación. En MARS-TITAN se puede contrastar si una cabeza rápida aprende parte del comportamiento de una variante recurrente más costosa. El profesor respetará el corte de entrenamiento de cada partición temporal y se contabilizará su coste. La diferencia profesor-alumno no prueba que la recurrencia sea nueva ni que pueda recuperarse toda su capacidad.

[Latent Replay](https://arxiv.org/abs/1912.01100v2) advierte un problema especialmente relevante: los vectores almacenados dejan de representar lo mismo si cambia el codificador. La referencia propuesta congela el codificador durante la evaluación y liga cada episodio a su revisión. Una adaptación del codificador exige reconstrucción o migración verificada de memoria y una evaluación independiente de su coste.

[Recurrent-Depth VLA](https://arxiv.org/abs/2602.07845v1) es un antecedente de cabeza recurrente y parada adaptativa en robótica. Impide presentar esas dos ideas como inéditas. La utilidad financiera debe compararse a igualdad de llamadas, contexto, presupuesto y calibración. La convergencia entre estados puede ser una regla de parada, pero no demuestra que la predicción sea correcta.

## Documentación de implementación que se deberá respetar

La [documentación de datos de PyTorch](https://docs.pytorch.org/docs/2.14/data.html) sirve para concretar iteración, procesos y prelectura. El plan debe impedir duplicados entre procesos de trabajo y mantener el orden que requiera cada estado. La [semántica CUDA](https://docs.pytorch.org/docs/2.14/notes/cuda.html) exige distinguir una operación enviada al dispositivo de una operación terminada. El [tutorial de medición de rendimiento](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) orienta el calentamiento y la medida. La concurrencia se dimensionará con la [guía de NVIDIA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), midiendo transferencias, ocupación y competencia entre cargas.

Para referencias clásicas, la [guía de aprendizaje con datos grandes de scikit-learn](https://scikit-learn.org/stable/computing/scaling_strategies.html) distingue estimadores incrementales de los que requieren el conjunto materializado. Ridge por estadísticas suficientes es una decisión algebraica del proyecto. SGD ofrece un contraste incremental que no es matemáticamente idéntico a resolver Ridge.

La [memoria externa de XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html) permite recorrer características por bloques, pero conserva otras estructuras y puede quedar limitada por las transferencias. Una matriz cuantizada construida desde un iterador no implica automáticamente que se use memoria externa. El plan completo tendrá que medir RAM, VRAM, caché en disco, tiempo de construcción y tiempo por árbol antes de fijar la campaña.

## Criterio de decisión

La arquitectura elegida deberá mejorar la frontera de error, retención y coste con evidencia fuera de muestra. Las decisiones de diseño y las cifras estimadas están en [entrenamiento completo](../engineering/full-dataset-training.md) y [presupuesto de latencia](../engineering/latency-budget.md). Ninguna de las referencias demuestra todavía que MARS-TITAN alcance el estado del arte financiero.
