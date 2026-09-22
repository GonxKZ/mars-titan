# Actualización reciente y antecedentes financieros de memoria

Autor del proyecto: Gonzalo García Lama. Verificación: 18 de septiembre de 2026.

Esta ampliación incorpora diez fuentes primarias que precisan la posición de MARS-TITAN. Los trabajos financieros ya estudian memoria por capas, reflexión, contexto de mercado, enrutamiento por errores históricos y selección temporal de experiencias. Los trabajos de sistemas muestran que las mejoras de precisión y caudal dependen del tamaño del modelo, la concurrencia y el hardware. Esta selección complementa las revisiones [neuronal](neural-review.md), [biológica](brain-review.md) y [de sistemas](systems-review.md). No acredita una búsqueda exhaustiva ni prioridad científica de la propuesta.

Las referencias y sus versiones están en [frontier-sources.json](frontier-sources.json) y [frontier.bib](frontier.bib). Se han contrastado fichas primarias y pasajes relevantes de métodos, evaluación y limitaciones. La fecha de consulta no implica que todos los trabajos se publicaran en 2026. Las propuestas de MARS-TITAN siguen sin resultados experimentales propios.

## Qué ideas ya tienen antecedentes

| Antecedente | Coincidencia con ideas consideradas en MARS-TITAN | Diferencia que debe conservarse al comparar |
| --- | --- | --- |
| [FinMem, Yu et al. (2024)](https://ojs.aaai.org/index.php/AAAI-SS/article/view/31290) | Memoria financiera por capas, relevancia, importancia, antigüedad y reflexión. | Recupera recuerdos textuales para un agente LLM. No es una memoria neuronal pequeña actualizada mediante gradientes. |
| [FinAgent, Zhang et al. (2024)](https://doi.org/10.1145/3637528.3671801) | Información numérica, textual y visual, recuperación de memoria y reflexión a dos niveles. | Produce decisiones de negociación con herramientas y modelos fundacionales. Su tarea no es la regresión residual diaria del protocolo. |
| [FinCon, Yu et al. (2024)](https://proceedings.neurips.cc/paper/2024/hash/f7ae4fe91d96f50abc2211f09b6a7e49-Abstract-Conference.html) | Jerarquía, crítica de experiencias y actualización de creencias financieras. | La comunicación y el refuerzo son verbales entre agentes. No equivalen a iterar un bloque numérico con pesos compartidos. |
| [TIEM, Liu et al. (2026)](https://arxiv.org/abs/2608.13024v5) | Eventos, memoria de experiencias, procedencia completa y acceso restringido por tiempo. | Clasifica dirección a tres o cinco sesiones con LLM. La memoria de habilidades queda congelada durante evaluación. |
| [MacroHFT, Zong et al. (2024)](https://doi.org/10.1145/3637528.3672064) | Memoria numérica, contexto financiero y combinación jerárquica de políticas. | Usa aprendizaje por refuerzo y negociación de criptomonedas por minutos. Acredita antecedentes fuera de los agentes de lenguaje, pero no valida pronósticos residuales diarios. |
| [TRA, Lin et al. (2021)](https://arxiv.org/html/2106.12950v2) | Enrutamiento de predictores con memoria de errores pasados. | Predice ordenación mensual en CSI800. No implementa memoria neural de pesos rápidos ni el objetivo residual diario. |

TIEM es especialmente cercano en la combinación conceptual. Su versión v5 exige que todos los antecesores de una habilidad tengan resultados disponibles antes de usarla. En los experimentos principales desactiva el diagnóstico de estabilidad, la reponderación y el razonamiento multipath opcional. No debe describirse como evidencia de que esas extensiones funcionan ni como adaptación continua durante test. Su prueba de sensibilidad a nombres y fechas tampoco excluye toda contaminación del preentrenamiento. [Método y apéndices C, G y H](https://arxiv.org/html/2608.13024v5).

El espacio de contribución debe expresarse como una comparación concreta: memoria compacta y escritura selectiva para retornos residuales, bajo cortes temporales verificables y un presupuesto local medido. Una recurrencia corta puede estudiarse como componente adicional. La originalidad dependerá de las decisiones precisas y del conocimiento obtenido, no de reunir nombres de mecanismos existentes. Estos antecedentes no obligan a implementar todos sus modelos ni a convertir el proyecto en un sistema de negociación por refuerzo.

## Qué demuestran las cifras de eficiencia

| Fuente | Condiciones de la evidencia | Lectura válida para MARS-TITAN |
| --- | --- | --- |
| [Mamba-3](https://proceedings.iclr.cc/paper_files/paper/2026/file/8abd2043b71a074278d5f687947bff9c-Paper-Conference.pdf) | La comparación de decodificación de la tabla 4 usa una H100, lote 128 y modelos de 1.500 millones de parámetros. | La formulación MIMO puede mejorar utilización en ese régimen. Hace falta medir una variante pequeña y los lotes reales en la RTX 4070. |
| [The Hyperscale Lottery](https://arxiv.org/html/2604.07935v2) | Estima una penalización de latencia del 48 % a 15 millones de parámetros mediante un modelo roofline de un ASIC hipotético. | Es evidencia de sensibilidad al régimen de ejecución, no una medida del portátil. No permite afirmar que Mamba-3 sea siempre más lento. |
| [RW-TTT](https://arxiv.org/html/2605.28053v2) | Ocho flujos y un modelo de 4.000 millones de parámetros. El resultado destacado usa 33,53 GiB y mide la ventana de decodificación, excluyendo prefill y carga. | Mejorar caudal agregado no demuestra menor latencia individual ni viabilidad en 8 GB. Sí justifica aislar estados y sus versiones. |
| [TTC](https://arxiv.org/html/2603.09221v2) | En H100, la tabla 9 pasa de 47,16 tokens/s del Transformer a 45,77 con horizonte 8 y 36,18 con horizonte 128. | La planificación interna añade coste incluso con un solver fusionado. Más horizonte debe compensarse con una mejora observada. |

La versión v2 de The Hyperscale Lottery presenta una discrepancia que conviene conservar: el resumen atribuye un 28 % a latencia para 880 millones de parámetros, mientras el cuerpo informa un 22 % y la tabla asocia el 28 % a energía del modelo analítico. Esa cifra no debe reutilizarse como estimación de tiempo para MARS-TITAN. [Resumen y sección 4](https://arxiv.org/html/2604.07935v2).

La comparación local debe medir tiempo total de sesión, latencias mediana, p95 y p99, memoria máxima y coste de actualización. El caudal, las operaciones teóricas y el tiempo de una consulta son cantidades diferentes. Para cada familia se registrarán lote, secuencia, precisión, versiones y kernels CUDA empleados. Compartir pesos limita parámetros, pero puede mantener elevado el número de evaluaciones. No existe aquí evidencia que permita fijar de antemano una familia ganadora en precisión y latencia.

## Estado adaptativo y concurrencia

RW-TTT trata cada actualización como un efecto sobre un propietario y una versión. Sus resultados apoyan separar lectura, escritura y publicación del estado. En MARS-TITAN, la unidad propietaria debe definirse científicamente, por ejemplo activo o memoria global, y no depender de la posición accidental en un lote. La memoria global requiere una instantánea común para las predicciones simultáneas y una regla explícita de consolidación posterior. Esta adaptación es una propuesta del proyecto. [Contrato de estado de RW-TTT](https://arxiv.org/html/2605.28053v2).

Los controles relevantes son reproducir una ejecución con distinto tamaño de lote, permutar el orden de activos, restaurar un estado guardado y verificar que una etiqueta pendiente no pueda modificarlos. Si una optimización cambia las predicciones porque mezcla propietarios o publica escrituras antes de tiempo, no representa el mismo experimento más rápido. El [protocolo](../research/protocol.md) debe seguir siendo la referencia para el orden de información.

## Notas de las diez fuentes

Cada nota contiene menos de 80 palabras y utiliza el identificador del registro.

### 1. `lahoti2026mamba3`

**Estado:** ICLR 2026, revisado por pares. Mamba-3 combina discretización más expresiva, estados complejos y actualización MIMO. Sus experimentos cubren lenguaje, recuperación y seguimiento de estados. Actualiza las referencias de secuencia, pero el ahorro observado depende del régimen de GPU. La cabecera del PDF fija la autoría usada en la bibliografía. [Actas](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html).

### 2. `geens2026hyperscale`

**Estado:** taller ITEM junto a ECML-PKDD 2026, arXiv v2. Combina medidas y modelos de hardware para examinar el coste de orientar SSM a grandes lotes. La estimación para modelos pequeños corresponde a un ASIC supuesto. Es un contrapunto a extrapolar eficiencia, con la discrepancia numérica documentada arriba. Su aceptación figura en el programa del taller. [Versión consultada](https://arxiv.org/abs/2604.07935v2).

### 3. `liu2026tiem`

**Estado:** preprint v5 de septiembre de 2026. Organiza evidencia mediante hipergrafos temporales y conserva habilidades derivadas de casos resueltos. La evaluación principal usa recuperación de una memoria congelada. Constituye un antecedente directo de eventos, memoria y trazabilidad temporal, pero no de todas sus extensiones opcionales. El registro incluye los nueve autores de v5. [Preprint](https://arxiv.org/abs/2608.13024v5).

### 4. `yang2026rwttt`

**Estado:** preprint v2 de septiembre de 2026. Define un contrato de propietario, versión y efectos de lectura o escritura para servir modelos con adaptación. Agrupa fases compatibles conservando el orden local. Fundamenta comprobaciones de aislamiento y restauración, mientras sus resultados de concurrencia quedan ligados a la configuración publicada. No demuestra adaptación predictiva financiera. [Preprint](https://arxiv.org/abs/2605.28053v2).

### 5. `wang2026ttc`

**Estado:** ICML 2026, corroborado por programa oficial. La copia arXiv v2 cambia el título a Beyond Test-Time Memory: State-Space Optimal Control for LLM Reasoning. Integra planificación LQR de horizonte finito sobre estados latentes. Sus resultados son de razonamiento matemático y sus medidas de GPU no prueban utilidad financiera. Una trayectoria latente no representa por sí misma futuros estados del mercado. [Versión consultada](https://arxiv.org/abs/2603.09221v2).

### 6. `yu2024finmem`

**Estado:** AAAI Spring Symposium Series, 2024. La publicación breve describe memoria de trabajo, capas persistentes, decaimiento y priorización para un agente financiero LLM. Es un antecedente claro de selección de recuerdos con diferentes escalas. Se distingue el PDF editorial de tres páginas del informe extenso arXiv de 2023. Sus afirmaciones cognitivas no sustituyen validación biológica. [Publicación](https://ojs.aaai.org/index.php/AAAI-SS/article/view/31290).

### 7. `zhang2024finagent`

**Estado:** KDD 2024, revisado por pares. Integra información financiera multimodal, herramientas, recuerdos y reflexión. Acredita que combinar esas capacidades ya tiene antecedentes. Su evaluación de negociación depende de instrumentos, periodo y modelo fundacional, por lo que no se trasladan rentabilidades ni exactitudes a MARS-TITAN. La copia institucional corresponde al artículo publicado. [PDF del autor](https://www3.ntu.edu.sg/home/boan/papers/KDD24_FinAgent.pdf).

### 8. `yu2024fincon`

**Estado:** NeurIPS 2024, revisado por pares. Coordina analistas y gestor mediante lenguaje y utiliza crítica de experiencias para revisar creencias. Permite distinguir memoria narrativa y refuerzo verbal de actualización numérica de parámetros. Se utiliza la autoría publicada de 18 personas, que añade a Yuechen Jiang y cambia el orden respecto al preprint de 17. [Actas](https://proceedings.neurips.cc/paper/2024/hash/f7ae4fe91d96f50abc2211f09b6a7e49-Abstract-Conference.html).

### 9. `zong2024macrohft`

**Estado:** KDD 2024, revisado por pares. Combina políticas condicionadas por mercado y una política superior con memoria en criptomonedas. Es un antecedente numérico que evita limitar la búsqueda a agentes LLM. Su objetivo, frecuencia y aprendizaje por refuerzo difieren del protocolo residual. El título menciona alta frecuencia, pero los experimentos descritos operan por minutos. [Preprint de consulta](https://arxiv.org/abs/2406.14537v1).

### 10. `lin2021tra`

**Estado:** KDD 2021, copia arXiv v2. TRA enruta predictores mediante errores históricos con separación respecto al horizonte de etiqueta. Su caché se refresca durante entrenamiento y la evaluación respeta el orden temporal. Delimita la novedad del enrutamiento financiero, pero sus errores recalculados no sustituyen la predicción originalmente emitida exigida por MARS-TITAN. [Método, §§5.1 y 5.3](https://arxiv.org/html/2106.12950v2).

## Consecuencia para la comparación

Esta revisión aconseja conservar controles sencillos, justificar cada módulo con una ablación y medir recursos sobre el problema real. La pertinencia bibliográfica de una arquitectura no obliga a implementarla. MARS-TITAN todavía debe demostrar qué conserva su memoria, cuándo se adapta, si las iteraciones internas aportan valor y qué coste tienen. Estas preguntas pueden dar resultados negativos y seguir produciendo una comparación útil. La revisión no cierra los objetivos experimentales.
